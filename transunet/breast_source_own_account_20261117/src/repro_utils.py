"""Strict numerical policy, isolated data streams, and complete restart state."""
import hashlib
import json
import os
import pickle
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    h = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((str(x.dtype), list(x.shape))).encode())
            h.update(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                h.update(str(key).encode())
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x:
                visit(item)
        else:
            h.update(pickle.dumps(x, protocol=4))
    visit(value)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def configure_strict(seed):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG before starting Python")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def stream_seed(seed, stream, epoch):
    raw = hashlib.sha256(f"{seed}:{stream}:{epoch}".encode()).digest()
    return int.from_bytes(raw[:8], "little") % (2**63 - 1)


def permutation(n, seed, stream, cycle):
    gen = torch.Generator().manual_seed(stream_seed(seed, stream, cycle))
    return torch.randperm(n, generator=gen).tolist()


def source_indices(n, batch_size, seed, epoch):
    return permutation(n, seed, "source", epoch)[:n // batch_size * batch_size]


def target_indices(n, batch_size, batches, seed, epoch):
    effective = n // batch_size * batch_size
    if not effective:
        raise ValueError("Target domain smaller than one batch")
    start, stop = (epoch - 1) * batches * batch_size, epoch * batches * batch_size
    result = []
    for cycle in range(start // effective, (stop - 1) // effective + 1):
        order = permutation(n, seed, "target", cycle)[:effective]
        left, right = max(0, start - cycle * effective), min(effective, stop - cycle * effective)
        result.extend(order[left:right])
    return result


def seed_worker(_worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@lru_cache(maxsize=64)
def axis_weights(input_size, output_size, align_corners, double):
    dtype = torch.float64 if double else torch.float32
    positions = torch.arange(output_size, dtype=dtype)
    if align_corners:
        coordinates = positions * ((input_size - 1) / (output_size - 1) if output_size > 1 else 0)
    else:
        coordinates = ((positions + .5) * input_size / output_size - .5).clamp_min(0)
    low = coordinates.floor().long().clamp_max(input_size - 1)
    high = (low + 1).clamp_max(input_size - 1)
    fraction = (coordinates - low).clamp(0, 1)
    return (F.one_hot(low, input_size).to(dtype) * (1 - fraction[:, None]) +
            F.one_hot(high, input_size).to(dtype) * fraction[:, None])


class _Bilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, size, align_corners):
        ctx.shape = tuple(x.shape)
        ctx.size, ctx.align_corners = tuple(size), align_corners
        # Keep the original bilinear forward; only atomic-scatter backward changes.
        return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)

    @staticmethod
    def backward(ctx, grad):
        h, w = ctx.shape[-2:]
        oh, ow = ctx.size
        double = grad.dtype == torch.float64
        dtype = torch.float64 if double else torch.float32
        wh = axis_weights(h, oh, ctx.align_corners, double).to(device=grad.device)
        ww = axis_weights(w, ow, ctx.align_corners, double).to(device=grad.device)
        with torch.autocast(device_type=grad.device.type, enabled=False):
            g = grad.to(dtype)
            width = g.reshape(-1, ow).matmul(ww).reshape(*grad.shape[:-2], oh, w)
            result = width.transpose(-2, -1).matmul(wh).transpose(-2, -1)
        return result.to(grad.dtype), None, None


class DeterministicBilinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.size, self.scale_factor = original.size, original.scale_factor
        self.align_corners = original.align_corners

    def forward(self, x):
        if self.size is None:
            scale = self.scale_factor
            factors = (scale, scale) if isinstance(scale, (float, int)) else scale
            if any(float(v) != int(v) for v in factors):
                raise ValueError("Only integer-scale legacy upsampling is audited")
            size = tuple(int(n * s) for n, s in zip(x.shape[-2:], factors))
        else:
            size = (self.size, self.size) if isinstance(self.size, int) else self.size
        return _Bilinear.apply(x, size, self.align_corners)


def deterministic_resize_modules(model):
    replaced = []
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Upsample) and child.mode == "bilinear":
            setattr(model, name, DeterministicBilinear(child))
            replaced.append(name)
        else:
            replaced.extend(name + "." + key for key in deterministic_resize_modules(child))
    return replaced


def pixel_cross_entropy(logits, target):
    # Avoid the nondeterministic CUDA NLLLoss spatial mean reduction in torch 2.1.
    return -F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)


def runtime_environment():
    props = torch.cuda.get_device_properties(0)
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), gpu=props.name,
                total_memory=props.total_memory, capability=[props.major, props.minor],
                node=os.environ.get("SLURMD_NODENAME"), job=os.environ.get("SLURM_JOB_ID"),
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                convolution_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
