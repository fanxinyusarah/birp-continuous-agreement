"""Shared data, augmentation, model and metric utilities for the BiRP re-study."""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.environ.get("BIRP_DATA", "/scratch/xl5874/Eve/birp/data")


def load_domain(name):
    z = np.load(os.path.join(DATA, name + ".npz"))
    return {k: z[k] for k in z.files}


def load_splits():
    with open(os.path.join(DATA, "splits.json")) as f:
        return json.load(f)


def load_prior(fm, domain):
    """Precomputed frozen-FM foreground probability, uint8 [N,256,256] -> float in [0,1]."""
    path = os.path.join(DATA, f"fm_{fm}_{domain}.npz")
    z = np.load(path)
    return z["prob"]


class GPUData:
    """Holds a domain subset on GPU: images [N,1,H,W] float, masks, priors."""

    def __init__(self, dom, idx, device, priors=None):
        idx = np.asarray(idx)
        self.idx = idx
        im = torch.from_numpy(dom["img"][idx]).float().div_(255)
        self.img = (im.permute(0, 3, 1, 2) if im.dim() == 4 else im.unsqueeze(1)).contiguous().to(device)
        self.mask = torch.from_numpy(dom["mask"][idx]).float().unsqueeze(1).to(device)
        self.priors = {}
        for k, p in (priors or {}).items():
            self.priors[k] = torch.from_numpy(p[idx]).float().div_(255).unsqueeze(1).to(device)
        self.ids = dom["id"][idx]
        self.n = len(idx)

    def batch(self, bs, gen):
        j = torch.randint(0, self.n, (bs,), generator=gen).to(self.img.device)
        return j


# ---------------------------------------------------------------- augmentation
def rand_affine(bs, device, gen, scale=(0.8, 1.2), rot=15, trans=0.1, flip=True):
    s = torch.empty(bs).uniform_(*scale, generator=gen)
    a = torch.empty(bs).uniform_(-rot, rot, generator=gen) * math.pi / 180
    tx = torch.empty(bs).uniform_(-trans, trans, generator=gen)
    ty = torch.empty(bs).uniform_(-trans, trans, generator=gen)
    fl = (torch.rand(bs, generator=gen) < 0.5).float() * 2 - 1 if flip else torch.ones(bs)
    cos, sin = torch.cos(a) / s, torch.sin(a) / s
    theta = torch.stack([torch.stack([cos * fl, -sin, tx], 1), torch.stack([sin * fl, cos, ty], 1)], 1)
    return theta.to(device)


def warp(x, theta, mode="bilinear"):
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)


def photometric(x, gen, strength=1.0):
    """Strong ultrasound-plausible photometric perturbation (per-sample)."""
    b = x.shape[0]
    dev = x.device
    r = lambda lo, hi: torch.empty(b, 1, 1, 1).uniform_(lo, hi, generator=gen).to(dev)
    s = strength
    x = x.clamp(0, 1) ** r(1 - 0.4 * s, 1 + 0.4 * s)
    x = (x - 0.5) * r(1 - 0.3 * s, 1 + 0.3 * s) + 0.5 + r(-0.1 * s, 0.1 * s)
    noise = torch.randn(x.shape, generator=gen).to(dev)
    x = x * (1 + noise * r(0, 0.25 * s))  # multiplicative speckle
    # random blur
    k = torch.tensor([1., 4., 6., 4., 1.], device=dev)
    k = (k[:, None] * k[None, :]); k = (k / k.sum())[None, None]
    blur = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k.expand(x.shape[1], 1, 5, 5), groups=x.shape[1])
    w = (torch.rand(b, 1, 1, 1, generator=gen).to(dev) < 0.5 * s).float()
    x = w * blur + (1 - w) * x
    return x.clamp(0, 1)


def cutout(x, gen, n=1, size=0.25):
    b, _, h, w = x.shape
    m = torch.ones_like(x)
    for _ in range(n):
        ch, cw = int(h * size), int(w * size)
        ys = torch.randint(0, h - ch, (b,), generator=gen)
        xs = torch.randint(0, w - cw, (b,), generator=gen)
        for i in range(b):
            if torch.rand(1, generator=gen).item() < 0.5:
                m[i, :, ys[i]:ys[i] + ch, xs[i]:xs[i] + cw] = 0
    return x * m + 0.5 * (1 - m)


# ---------------------------------------------------------------- models
class FMSeg(nn.Module):
    """Foundation-model backbone + light multi-level (DPT-lite) decoder; returns 1-channel logits at input size."""

    def __init__(self, kind):
        super().__init__()
        self.hr = kind.startswith("dinov2hr")  # high-resolution CNN skip branch for boundaries / small objects
        size = 448 if kind.endswith("448") else 224
        kind = "dinov2" if kind.startswith("dinov2") else kind
        self.kind = kind
        if kind == "dinov2":
            from transformers import Dinov2Model
            self.enc = Dinov2Model.from_pretrained("facebook/dinov2-base")
            self.layers = [3, 6, 9, 12]; C = 768; self.in_size = size
            self.register_buffer("mu", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("sd", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:  # SAMUS image encoder (ultrasound-pretrained, US30K), 256 input, 32x32 tokens
            import sys
            from types import SimpleNamespace
            root = os.environ.get("SAMUS_ROOT", os.path.join(os.path.dirname(__file__), "samus"))
            if root not in sys.path:
                sys.path.insert(0, root)
            from models.segment_anything_samus.build_sam_us import samus_model_registry
            ck = os.environ.get("SAMUS_CKPT", "/scratch/xl5874/SAMUS_For_3D/checkpoints/samus_pretrained.pth")
            self.enc = samus_model_registry["vit_b"](SimpleNamespace(encoder_input_size=256), checkpoint=ck).image_encoder
            self.layers = [3, 6, 9, 12]; C = 768; self.in_size = 256
        self.proj = nn.ModuleList([nn.Conv2d(C, 128, 1) for _ in self.layers])
        self.fuse = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))
        self.up = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Conv2d(32, 1, 1))
        if self.hr:
            cbr = lambda i, o, st: nn.Sequential(nn.Conv2d(i, o, 3, st, 1), nn.BatchNorm2d(o), nn.ReLU(True))
            self.stem1 = nn.Sequential(cbr(3, 32, 2), cbr(32, 32, 1))      # 1/2 res
            self.stem2 = nn.Sequential(cbr(32, 64, 2), cbr(64, 64, 1))     # 1/4 res
            self.dec_lo = nn.Sequential(cbr(256, 128, 1))
            self.dec2 = nn.Sequential(cbr(128 + 64, 96, 1), cbr(96, 96, 1))
            self.dec1 = nn.Sequential(cbr(96 + 32, 48, 1), cbr(48, 48, 1))
            self.head = nn.Conv2d(48, 1, 1)

    def feats(self, x):
        if self.kind == "dinov2":
            x = F.interpolate(x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x, size=self.in_size, mode="bilinear", align_corners=False)
            o = self.enc(pixel_values=(x - self.mu) / self.sd, output_hidden_states=True)
            hs = o.hidden_states
            g = self.in_size // 14
            return [hs[l][:, 1:].transpose(1, 2).reshape(x.shape[0], -1, g, g) for l in self.layers]
        e = self.enc
        x3 = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        cnnx = e.cnn_embed(x3)
        h = e.input_Adapter(e.patch_embed(x3))
        if e.pos_embed is not None:
            h = h + e.post_pos_embed(e.pos_embed).repeat(h.shape[0], 1, 1, 1)
        out = []
        for i, blk in enumerate(e.blocks, 1):
            h, cnnx = blk(h, cnnx)
            if i in self.layers:
                out.append((h + 0.5 * cnnx).permute(0, 3, 1, 2))
        return out

    def forward(self, x):
        H = x.shape[-1]
        fs = self.feats(x)
        g = fs[0].shape[-1]
        y = self.fuse(torch.cat([p(f) for p, f in zip(self.proj, fs)], 1))
        if not self.hr:
            y = self.up(y)
            return F.interpolate(y, size=H, mode="bilinear", align_corners=False)
        x3 = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
        s1 = self.stem1(x3); s2 = self.stem2(s1)
        y = self.dec_lo(y)
        y = self.dec2(torch.cat([F.interpolate(y, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        y = self.dec1(torch.cat([F.interpolate(y, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
        return F.interpolate(self.head(y), size=H, mode="bilinear", align_corners=False)

    def param_groups(self, lr, enc_mult=0.1):
        enc = list(self.enc.parameters()); ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if id(p) not in ids]
        return [{"params": enc, "lr": lr * enc_mult}, {"params": rest, "lr": lr}]


class Student(nn.Module):
    def __init__(self, arch="unet", encoder="resnet50", pretrained=True):
        super().__init__()
        self.arch = arch
        if arch == "transunet":  # original BiRP student: R50-ViT-B/16 TransUNet, random init, 2-class softmax
            from transunet.vit_seg_modeling import VisionTransformer
            from transunet.vit_seg_configs import get_r50_b16_config
            cfg = get_r50_b16_config(); cfg.n_classes = 2; cfg.n_skip = 3; cfg.patches.grid = (16, 16)
            self.net = VisionTransformer(cfg, img_size=256, num_classes=2)
            return
        if arch.startswith(("dinov2", "samus")):
            self.net = FMSeg(arch)
            return
        import segmentation_models_pytorch as smp
        w = "imagenet" if pretrained else None
        if arch == "unet":
            self.net = smp.Unet(encoder, encoder_weights=w, in_channels=3, classes=1)
        elif arch == "segformer":
            self.net = smp.Segformer(encoder, encoder_weights=w, in_channels=3, classes=1)
        elif arch == "deeplab":
            self.net = smp.DeepLabV3Plus(encoder, encoder_weights=w, in_channels=3, classes=1)
        else:
            raise ValueError(arch)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.arch.startswith(("dinov2", "samus")):
            return self.net(x)
        if self.arch == "transunet":  # binary logit = l1 - l0 of the 2-class head; input in [0,1] as original
            lo = self.net(x)
            return lo[:, 1:2] - lo[:, 0:1]
        x = ((x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x) - self.mean) / self.std
        return self.net(x)


# ---------------------------------------------------------------- losses / metrics
def soft_dice_loss(logit, target, weight=None, eps=1.0):
    p = torch.sigmoid(logit)
    if weight is None:
        weight = torch.ones_like(p)
    inter = (p * target * weight).sum((1, 2, 3))
    den = ((p + target) * weight).sum((1, 2, 3))
    valid = (weight.sum((1, 2, 3)) > 0).float()
    d = 1 - (2 * inter + eps) / (den + eps)
    return (d * valid).sum() / valid.sum().clamp_min(1)


def weighted_bce(logit, target, weight=None):
    l = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    if weight is None:
        return l.mean()
    return (l * weight).sum() / weight.sum().clamp_min(1)


@torch.no_grad()
def predict(model, x, bs=32, tta=False):
    model.eval()
    out = []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs]
        p = torch.sigmoid(model(xb))
        if tta:
            p = 0.5 * (p + torch.sigmoid(model(xb.flip(-1))).flip(-1))
        out.append(p)
    model.train()
    return torch.cat(out)


def dice_per_image(pred_bin, gt, eps=1e-6):
    pred_bin = pred_bin.float(); gt = gt.float()
    inter = (pred_bin * gt).sum((1, 2, 3))
    den = pred_bin.sum((1, 2, 3)) + gt.sum((1, 2, 3))
    d = (2 * inter + eps) / (den + eps)
    # both empty -> 1, handled by eps
    return d


def iou_per_image(pred_bin, gt, eps=1e-6):
    pred_bin = pred_bin.float(); gt = gt.float()
    inter = (pred_bin * gt).sum((1, 2, 3))
    union = ((pred_bin + gt) > 0).float().sum((1, 2, 3))
    return (inter + eps) / (union + eps)


@torch.no_grad()
def ema_update(ema, model, m):
    for e, p in zip(ema.parameters(), model.parameters()):
        e.mul_(m).add_(p.detach(), alpha=1 - m)
    for e, b in zip(ema.buffers(), model.buffers()):
        e.copy_(b)


class FrozenFeat(nn.Module):
    """Frozen foundation-model last-layer patch features (L2-normalised), used only as a structure prior."""

    def __init__(self, kind="dinov2"):
        super().__init__()
        self.m = FMSeg(kind)
        self.m.eval()
        for q in self.m.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        f = self.m.feats(x)[-1].float()
        return F.normalize(f, dim=1)

    def train(self, mode=True):  # always eval
        return super().train(False)
