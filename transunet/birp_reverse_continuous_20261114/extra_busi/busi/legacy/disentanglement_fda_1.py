#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import json
import time
import copy
import random
import argparse
from typing import Dict, Optional, Tuple, List

import numpy as np
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None


# =============================================================================
# Utils (keep same style as your "GPT runnable" script)
# =============================================================================
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def now_str():
    t = time.localtime()
    return "%04d-%02d-%02d_%02d-%02d-%02d" % (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)


def sigmoid_rampup(current, rampup_length):
    if rampup_length <= 0:
        return 1.0
    current = float(np.clip(current, 0.0, rampup_length))
    phase = 1.0 - current / float(rampup_length)
    return float(math.exp(-5.0 * phase * phase))


def set_torch_backends(deterministic: bool = False):
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.backends.cudnn.deterministic = bool(deterministic)


def safe_mean(xs: List[float]) -> float:
    if xs is None or len(xs) == 0:
        return 0.0
    return float(sum(xs)) / float(len(xs))


def count_parameters(model: nn.Module) -> int:
    n = 0
    for p in model.parameters():
        n += int(p.numel())
    return n


# =============================================================================
# SDF
# =============================================================================
def compute_sdf_numpy(mask: np.ndarray, normalize: bool = True) -> np.ndarray:
    pos_mask = (mask > 0).astype(np.uint8)
    neg_mask = 1 - pos_mask
    pos_dist = distance_transform_edt(pos_mask)
    neg_dist = distance_transform_edt(neg_mask)
    sdf = neg_dist.astype(np.float32) - pos_dist.astype(np.float32)
    if normalize:
        sdf = np.tanh(sdf / 10.0)
    return sdf


@torch.no_grad()
def compute_sdf_batch(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    # masks: (B,H,W) int64 with {0..K-1}
    B, H, W = masks.shape
    K = int(num_classes)
    device = masks.device
    masks_np = masks.detach().cpu().numpy()

    sdf_batch = np.zeros((B, K - 1, H, W), dtype=np.float32)
    for b in range(B):
        for c in range(1, K):
            binary_mask = (masks_np[b] == c)
            if binary_mask.sum() > 0:
                sdf_batch[b, c - 1] = compute_sdf_numpy(binary_mask, normalize=True)

    return torch.from_numpy(sdf_batch).to(device)


# =============================================================================
# Loss
# =============================================================================
def dice_loss_soft(logits, target, eps=1e-5):
    # logits: (B,K,H,W)
    # target: (B,H,W) OR one-hot (B,K-1,H,W)
    prob = torch.softmax(logits, dim=1)[:, 1:]  # foreground channels

    if target.dim() == 3:
        B, H, W = target.shape
        K = logits.shape[1]
        target_oh = torch.zeros(B, K - 1, H, W, device=target.device, dtype=torch.float32)
        for c in range(1, K):
            target_oh[:, c - 1] = (target == c).float()
        target = target_oh

    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def dice_loss_soft_masked(logits, target, mask, eps=1e-5):
    # Applies Dice only to samples that have at least one accepted pseudo pixel.
    if mask is None:
        return dice_loss_soft(logits, target, eps=eps)
    prob = torch.softmax(logits, dim=1)[:, 1:]

    if target.dim() == 3:
        B, H, W = target.shape
        K = logits.shape[1]
        target_oh = torch.zeros(B, K - 1, H, W, device=target.device, dtype=torch.float32)
        for c in range(1, K):
            target_oh[:, c - 1] = (target == c).float()
        target = target_oh

    if mask.dim() == 3:
        mask_b = (mask.sum(dim=(1, 2)) > 0).float().view(-1, 1)
    else:
        mask_b = (mask.view(mask.shape[0], -1).sum(dim=1) > 0).float().view(-1, 1)

    inter = (prob * target).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    loss = (1.0 - dice) * mask_b
    return loss.sum() / mask_b.sum().clamp_min(1.0)


@torch.no_grad()
def dice_hard_binary(prob_fg: torch.Tensor, target_fg: torch.Tensor, thr: float = 0.5, eps: float = 1e-5) -> float:
    pred = (prob_fg > float(thr)).float()
    t = target_fg.float()
    inter = (pred * t).sum()
    union = pred.sum() + t.sum()
    dice = (2.0 * inter + eps) / (union + eps)
    return float(dice.item())


def bce_logits_loss(pred, is_real: bool):
    tgt = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
    return F.binary_cross_entropy_with_logits(pred, tgt)


def l1_loss(a, b):
    return (a - b).abs().mean()


def soft_dice_between_probs(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # p, q: (B,1,H,W)
    inter = (p * q).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + q.sum(dim=(2, 3))
    d = (2.0 * inter + eps) / (union + eps)
    return d.mean()


# =============================================================================
# FFT / Frequency helpers
# =============================================================================
def fft_amplitude_phase(x: torch.Tensor):
    orig_dtype = x.dtype
    with torch.cuda.amp.autocast(enabled=False):
        x_fp32 = x.float()
        fft = torch.fft.fft2(x_fp32, norm="ortho")
        amp = torch.abs(fft)
        phase = torch.angle(fft)
    return amp.to(orig_dtype), phase.to(orig_dtype)


def build_freq_view_from_image(x: torch.Tensor, eps: float = 1e-6):
    amp, _ = fft_amplitude_phase(x)
    amp = torch.log(amp + eps)
    mean = amp.mean(dim=(2, 3), keepdim=True)
    std = amp.std(dim=(2, 3), keepdim=True)
    return (amp - mean) / (std + eps)


def cosine_ortho(a: torch.Tensor, b: torch.Tensor):
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return (a * b).sum(dim=1).abs().mean()


# =============================================================================
# FDA v3 (low-frequency amplitude replacement)  (same as your GPT script)
# =============================================================================
def _fftshift2d(x: torch.Tensor) -> torch.Tensor:
    H = x.shape[-2]
    W = x.shape[-1]
    return torch.roll(torch.roll(x, shifts=(H // 2), dims=-2), shifts=(W // 2), dims=-1)


def _ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    H = x.shape[-2]
    W = x.shape[-1]
    return torch.roll(torch.roll(x, shifts=(-(H // 2)), dims=-2), shifts=(-(W // 2)), dims=-1)


def _make_low_freq_mask(H: int, W: int, beta: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    cy = H // 2
    cx = W // 2
    yy = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)
    dist2 = (yy - float(cy)) ** 2 + (xx - float(cx)) ** 2
    r = float(beta) * float(min(H, W)) * 0.5
    mask = (dist2 <= (r * r)).to(dtype)
    return mask.view(1, 1, H, W)


@torch.no_grad()
def fda_source_to_target(x_s: torch.Tensor, x_t: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    if (x_s.dim() != 4) or (x_t.dim() != 4):
        raise ValueError("fda_source_to_target expects 4D tensors (B,C,H,W)")

    B, C, H, W = x_s.shape
    if x_t.shape != x_s.shape:
        x_t = F.interpolate(x_t, size=(H, W), mode="bilinear", align_corners=False)

    orig_dtype = x_s.dtype
    device = x_s.device

    with torch.cuda.amp.autocast(enabled=False):
        xs = x_s.float().clamp(0.0, 1.0)
        xt = x_t.float().clamp(0.0, 1.0)

        fft_s = torch.fft.fft2(xs, norm="ortho")
        fft_t = torch.fft.fft2(xt, norm="ortho")

        amp_s = torch.abs(fft_s)
        pha_s = torch.angle(fft_s)
        amp_t = torch.abs(fft_t)

        amp_s_shift = _fftshift2d(amp_s)
        amp_t_shift = _fftshift2d(amp_t)

        mask = _make_low_freq_mask(H, W, beta=float(beta), device=device, dtype=amp_s_shift.dtype)
        if mask.shape[1] != C:
            mask = mask.repeat(1, C, 1, 1)

        amp_s_shift = amp_s_shift * (1.0 - mask) + amp_t_shift * mask
        amp_s_new = _ifftshift2d(amp_s_shift)

        real = amp_s_new * torch.cos(pha_s)
        imag = amp_s_new * torch.sin(pha_s)
        fft_new = torch.complex(real, imag)

        x_new = torch.fft.ifft2(fft_new, norm="ortho").real
        x_new = x_new.clamp(0.0, 1.0)

    return x_new.to(orig_dtype)


# =============================================================================
# GRL
# =============================================================================
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = float(lambda_val)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_val * grad_output, None


def gradient_reversal(x, lambda_val: float):
    return GradientReversalFunction.apply(x, lambda_val)


# =============================================================================
# MobileSAM Prior (teacher)
# =============================================================================
class MobileSAMPrior(nn.Module):
    def __init__(self, ckpt_path: str, device: torch.device, sam_type: str = "vit_t", freeze: bool = True):
        super().__init__()
        self.device = device
        self.sam_type = str(sam_type)

        from mobile_sam import sam_model_registry
        from mobile_sam.utils.transforms import ResizeLongestSide

        self.sam = sam_model_registry[self.sam_type](checkpoint=ckpt_path)
        self.sam.to(device)
        self.sam.eval()

        if bool(freeze):
            for p in self.sam.parameters():
                p.requires_grad = False

        self.resize = ResizeLongestSide(1024)

        self.pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def forward(self, x_gray_224: torch.Tensor) -> torch.Tensor:
        """
        x_gray_224: (B,1,224,224) in [0,1]
        return prob_sam: (B,1,224,224) in [0,1]
        """
        B = int(x_gray_224.size(0))
        x = x_gray_224.repeat(1, 3, 1, 1)  # (B,3,224,224)

        x_np = (x * 255.0).detach().cpu().numpy().astype(np.uint8)

        imgs_1024 = []
        for i in range(B):
            img = x_np[i].transpose(1, 2, 0)  # HWC
            img = self.resize.apply_image(img)  # H'W'3
            img = torch.from_numpy(img).permute(2, 0, 1).float()  # 3,H',W'
            h, w = img.shape[1], img.shape[2]

            if (h > 1024) or (w > 1024):
                img = F.interpolate(img.unsqueeze(0), size=(1024, 1024), mode="bilinear", align_corners=False).squeeze(0)
                h, w = 1024, 1024

            img = F.pad(img, (0, 1024 - w, 0, 1024 - h))
            imgs_1024.append(img)

        imgs_1024 = torch.stack(imgs_1024).to(self.device)  # (B,3,1024,1024)
        imgs_1024 = (imgs_1024 - self.pixel_mean) / self.pixel_std

        image_embeddings = self.sam.image_encoder(imgs_1024)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=None, masks=None)

        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        prob = torch.sigmoid(
            F.interpolate(low_res_masks, size=(224, 224), mode="bilinear", align_corners=False)
        )
        return prob


# =============================================================================
# Dataset (GPT-style robust: same as your first script)
# =============================================================================
class SegListDataset(Dataset):
    """
    Robust dataset for segmentation.
    - List file supports: {root}/{split}_list.txt | {root}/lists/{split}.txt | {root}/{split}.txt
    - Image/label supports: PNG/JPG and NPY (2D/3D/4D pseudo-3D)
    - Resize uses PIL only (avoid cv2 resize issues).
    Return:
      x: (1,H,W) float32 in [0,1]
      y: (H,W) int64 in {0,1} (if use_labels else zeros)
      name: str
    """
    IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def __init__(self, root: str, split: str, img_size: int = 224, return_name: bool = True, use_labels: bool = True):
        super().__init__()
        self.root = root
        self.split = split
        self.img_size = int(img_size)
        self.return_name = bool(return_name)
        self.use_labels = bool(use_labels)

        cand = [
            os.path.join(root, f"{split}_list.txt"),
            os.path.join(root, "lists", f"{split}.txt"),
            os.path.join(root, f"{split}.txt"),
        ]
        list_file = None
        for p in cand:
            if os.path.isfile(p):
                list_file = p
                break
        if list_file is None:
            raise FileNotFoundError("list file not found. tried: " + " | ".join(cand))

        with open(list_file, "r") as f:
            names = [x.strip() for x in f.readlines()]
        self.names = [n for n in names if len(n) > 0]

        self.img_dir = os.path.join(root, "images")
        self.lab_dir = os.path.join(root, "labels")

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError("images dir not found: " + self.img_dir)
        if self.use_labels and (not os.path.isdir(self.lab_dir)):
            raise FileNotFoundError("labels dir not found: " + self.lab_dir)

        from PIL import Image
        self.Image = Image

    def __len__(self):
        return len(self.names)

    def _resolve_with_exts(self, base_noext: str) -> Optional[str]:
        for ext in self.IMG_EXTS:
            p = base_noext + ext
            if os.path.isfile(p):
                return p
        return None

    def _resolve_img_path(self, name: str) -> str:
        p0 = os.path.join(self.img_dir, name)
        if os.path.isfile(p0):
            return p0

        stem = os.path.splitext(name)[0]
        base = os.path.join(self.img_dir, stem)

        p_img = self._resolve_with_exts(base)
        if p_img is not None:
            return p_img

        p_npy = base + ".npy"
        if os.path.isfile(p_npy):
            return p_npy

        p_npy2 = base + "_img.npy"
        if os.path.isfile(p_npy2):
            return p_npy2

        raise FileNotFoundError(f"cannot resolve image for '{name}'. tried: {p0}, {base}+[ext], {p_npy}, {p_npy2}")

    def _resolve_lab_path(self, name: str) -> str:
        if not self.use_labels:
            return ""

        p0 = os.path.join(self.lab_dir, name)
        if os.path.isfile(p0):
            return p0

        stem = os.path.splitext(name)[0]
        base = os.path.join(self.lab_dir, stem)

        p_lab = self._resolve_with_exts(base)
        if p_lab is not None:
            return p_lab

        p_npy = base + ".npy"
        if os.path.isfile(p_npy):
            return p_npy

        p_npy2 = base + "_label.npy"
        if os.path.isfile(p_npy2):
            return p_npy2

        raise FileNotFoundError(f"cannot resolve label for '{name}'. tried: {p0}, {base}+[ext], {p_npy}, {p_npy2}")

    def _read_png_gray_pil(self, path: str) -> np.ndarray:
        img = self.Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)

    def _to_2d_from_npy(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a)

        if a.ndim == 2:
            pass
        elif a.ndim == 3:
            if a.shape[0] == 1:
                a = a[0]
            elif a.shape[-1] == 1:
                a = a[..., 0]
            else:
                d = a.shape[0]
                a = a[d // 2]
        elif a.ndim == 4:
            if a.shape[0] == 1:
                a = a[0]
            if a.ndim == 3:
                d = a.shape[0]
                a = a[d // 2]
            else:
                a = np.squeeze(a)
                if a.ndim != 2:
                    raise ValueError(f"npy cannot convert to 2D, shape={a.shape}")
        else:
            a = np.squeeze(a)
            if a.ndim != 2:
                raise ValueError(f"npy cannot convert to 2D, shape={a.shape}")

        a = a.astype(np.float32)
        mx = float(a.max()) if a.size > 0 else 0.0
        if mx <= 1.5:
            a = a * 255.0
        a = np.clip(a, 0.0, 255.0).astype(np.uint8)
        return a

    def _read_npy_gray(self, path: str) -> np.ndarray:
        a = np.load(path, allow_pickle=False)
        return self._to_2d_from_npy(a)

    def _resize_gray_uint8(self, arr_u8: np.ndarray, interp: str) -> np.ndarray:
        arr_u8 = np.asarray(arr_u8, dtype=np.uint8)
        img = self.Image.fromarray(arr_u8, mode="L")
        if interp == "bilinear":
            img = img.resize((self.img_size, self.img_size), resample=self.Image.BILINEAR)
        else:
            img = img.resize((self.img_size, self.img_size), resample=self.Image.NEAREST)
        return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = self._resolve_img_path(name)

        if os.path.splitext(img_path)[1].lower() in self.IMG_EXTS:
            im = self._read_png_gray_pil(img_path)
        elif img_path.endswith(".npy"):
            im = self._read_npy_gray(img_path)
        else:
            raise FileNotFoundError("unknown image format: " + img_path)

        if self.use_labels:
            lab_path = self._resolve_lab_path(name)
            if os.path.splitext(lab_path)[1].lower() in self.IMG_EXTS:
                lb = self._read_png_gray_pil(lab_path)
            elif lab_path.endswith(".npy"):
                lb = self._read_npy_gray(lab_path)
            else:
                raise FileNotFoundError("unknown label format: " + lab_path)

            lb = (lb > 0).astype(np.uint8) * 255
        else:
            lb = np.zeros_like(im, dtype=np.uint8)

        im = self._resize_gray_uint8(im, interp="bilinear")
        lb = self._resize_gray_uint8(lb, interp="nearest")

        im = im.astype(np.float32) / 255.0
        lb = (lb > 0).astype(np.int64)

        x = torch.from_numpy(im).unsqueeze(0)
        y = torch.from_numpy(lb)

        if self.return_name:
            return x, y, name
        return x, y


class TargetSplitDataset(Dataset):
    """
    Reads target unlabeled images from a split file whose lines look like:
      BUSBRA::Images/xxx.png
    Image path: {busbra_root}/Images/xxx.png
    Returns:
      x: (1, img_size, img_size) float32 in [0,1]
      key: original split line (string)
    """
    IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def __init__(self, splits_file: str, busbra_root: str, img_size: int = 224):
        super().__init__()
        self.img_size = int(img_size)
        self.busbra_root = busbra_root

        with open(splits_file, "r") as f:
            keys = [x.strip() for x in f.readlines()]
        self.keys = [k for k in keys if len(k) > 0]

        from PIL import Image
        self.Image = Image

    def __len__(self):
        return len(self.keys)

    def _read_gray_pil(self, path: str) -> np.ndarray:
        img = self.Image.open(path).convert("L")
        return np.array(img, dtype=np.uint8)

    def _resize_u8(self, arr_u8: np.ndarray) -> np.ndarray:
        arr_u8 = np.asarray(arr_u8, dtype=np.uint8)
        img = self.Image.fromarray(arr_u8, mode="L")
        img = img.resize((self.img_size, self.img_size), resample=self.Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    def __getitem__(self, idx):
        key = self.keys[idx]
        if "::" in key:
            _, rel = key.split("::", 1)
        else:
            rel = key

        img_path = os.path.join(self.busbra_root, rel)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"target image not found: {img_path} (from key={key})")

        im = self._read_gray_pil(img_path)
        im = self._resize_u8(im)
        x = torch.from_numpy(im.astype(np.float32) / 255.0).unsqueeze(0)
        return x, key


# =============================================================================
# TransUNet builder + pretrained loader (GPT-style supports .npz)
# =============================================================================
def build_transunet(vit_name: str, img_size: int, num_classes: int):
    from networks.vit_seg_modeling import VisionTransformer as ViT_seg
    from networks.vit_seg_configs import get_r50_b16_config, get_b16_config

    if vit_name in ["R50-ViT-B_16", "R50ViTB16", "r50_b16"]:
        config = get_r50_b16_config()
    elif vit_name in ["ViT-B_16", "B_16", "vit_b16"]:
        config = get_b16_config()
    else:
        raise ValueError("Unknown vit_name: " + str(vit_name))

    config.n_classes = int(num_classes)
    config.n_skip = 3
    config.patches.grid = (img_size // 16, img_size // 16)
    model = ViT_seg(config, img_size=img_size, num_classes=int(num_classes))
    return model


def load_pretrained_weights(model, ckpt_path: str):
    if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
        print("pretrained not found:", ckpt_path)
        return

    ext = os.path.splitext(ckpt_path)[1].lower()

    if ext == ".npz":
        weights = np.load(ckpt_path)
        if hasattr(model, "load_from"):
            model.load_from(weights)
            print("loaded ViT npz pretrained:", ckpt_path)
        else:
            raise RuntimeError("Model has no load_from(); build_transunet mismatch?")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("loaded torch pretrained:", ckpt_path)
    print("missing:", len(missing), "unexpected:", len(unexpected))


# =============================================================================
# Blocks (same as before)
# =============================================================================
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class StyleEncoder(nn.Module):
    def __init__(self, in_ch=1, base=32, style_dim=8):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBnRelu(in_ch, base, 3, 2, 1),
            ConvBnRelu(base, base * 2, 3, 2, 1),
            ConvBnRelu(base * 2, base * 4, 3, 2, 1),
            ConvBnRelu(base * 4, base * 4, 3, 2, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base * 4, style_dim)

    def forward(self, x):
        feat = self.convs(x)
        feat = self.pool(feat).flatten(1)
        return self.fc(feat)


class Generator(nn.Module):
    def __init__(self, content_dim=256, style_dim=8, base=64, out_ch=1, img_size=224):
        super().__init__()
        self.img_size = int(img_size)
        self.fc = nn.Sequential(
            nn.Linear(content_dim + style_dim, base * 8 * 7 * 7),
            nn.ReLU(inplace=True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, base // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base // 2, out_ch, 4, 2, 1),
        )

    def forward(self, z_content, z_style):
        z = torch.cat([z_content, z_style], dim=1)
        feat = self.fc(z).view(z.size(0), -1, 7, 7)
        img = self.deconv(feat)
        if img.shape[-1] != self.img_size:
            img = F.interpolate(img, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return img


class DualTaskModule(nn.Module):
    def __init__(self, in_channels, num_classes, img_size=224):
        super().__init__()
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)

        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.num_classes, 1)
        )

        self.sdf_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, self.num_classes - 1, 1),
            nn.Tanh()
        )

    def forward(self, z_a):
        seg_logits = self.seg_head(z_a)
        seg_logits = F.interpolate(seg_logits, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        sdf_pred = self.sdf_head(z_a)
        sdf_pred = F.interpolate(sdf_pred, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return seg_logits, sdf_pred


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, base=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 4, base * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 8, mlp_ratio: float = 4.0, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)

        self.norm_q = nn.LayerNorm(self.dim)
        self.norm_kv = nn.LayerNorm(self.dim)

        self.attn = nn.MultiheadAttention(self.dim, self.num_heads, dropout=attn_drop, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)

        hidden = int(self.dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(proj_drop),
            nn.Linear(hidden, self.dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor):
        q = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = q_tokens + self.proj_drop(attn_out)
        x2 = self.mlp(self.norm2(x))
        x = x + x2
        return x


# =============================================================================
# DLaST (same as your code)
# =============================================================================
class DLaST(nn.Module):
    def __init__(
        self,
        seg_backbone: nn.Module,
        num_classes: int,
        img_size: int = 224,
        content_dim: int = 256,
        style_dim: int = 8,
        in_channels: int = 1,
        xattn_heads: int = 8
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.img_size = int(img_size)
        self.content_dim = int(content_dim)
        self.style_dim = int(style_dim)
        self.in_channels = int(in_channels)

        self.anatomy_encoder = seg_backbone
        self.vit_hidden_dim = 768
        self.grid_size = self.img_size // 16

        self.feat_proj = nn.Conv2d(self.vit_hidden_dim, self.content_dim, 1, bias=True)

        self.style_proj = nn.Linear(self.style_dim, self.content_dim, bias=False)
        self.style_proj_f = nn.Linear(self.style_dim, self.content_dim, bias=False)

        self.dual_task = DualTaskModule(self.content_dim, self.num_classes, self.img_size)

        self.style_enc_source = StyleEncoder(in_ch=self.in_channels, base=32, style_dim=self.style_dim)
        self.style_enc_target = StyleEncoder(in_ch=self.in_channels, base=32, style_dim=self.style_dim)

        self.freq_style_enc_source = StyleEncoder(in_ch=self.in_channels, base=32, style_dim=self.style_dim)
        self.freq_style_enc_target = StyleEncoder(in_ch=self.in_channels, base=32, style_dim=self.style_dim)

        self.cross_attn = CrossAttentionBlock(dim=self.vit_hidden_dim, num_heads=xattn_heads, mlp_ratio=4.0)

        self.generator = Generator(self.content_dim, self.style_dim, base=64, out_ch=self.in_channels, img_size=self.img_size)

        self.disc_source = PatchDiscriminator(self.in_channels, base=64)
        self.disc_target = PatchDiscriminator(self.in_channels, base=64)

        self.disc_seg = PatchDiscriminator(self.num_classes, base=32)
        self.disc_sdf = PatchDiscriminator(self.num_classes - 1, base=32)

    def _to_3ch_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def _infer_grid(self, n_tokens: int) -> int:
        gs = self.grid_size
        if n_tokens != gs * gs:
            gs2 = int(math.sqrt(n_tokens))
            if gs2 * gs2 == n_tokens:
                return gs2
        return gs

    def _forward_transformer_tokens(self, x: torch.Tensor):
        x3 = self._to_3ch_if_needed(x)

        if hasattr(self.anatomy_encoder, "transformer") and hasattr(self.anatomy_encoder, "decoder") and hasattr(self.anatomy_encoder, "segmentation_head"):
            x_tokens, _, features = self.anatomy_encoder.transformer(x3)
            B, N, H = x_tokens.shape
            gs = self._infer_grid(N)
            return x_tokens, features, gs

        return None, None, None

    def _tokens_to_featmap(self, x_tokens: torch.Tensor, gs: int):
        B, N, H = x_tokens.shape
        feat = x_tokens.permute(0, 2, 1).contiguous().view(B, H, gs, gs)
        if feat.shape[1] != self.vit_hidden_dim:
            if feat.shape[1] > self.vit_hidden_dim:
                feat = feat[:, :self.vit_hidden_dim]
            else:
                rep = max(1, self.vit_hidden_dim // feat.shape[1])
                feat = feat.repeat(1, rep, 1, 1)[:, :self.vit_hidden_dim]
        return feat

    def extract_anatomy_feature_with_freq_xattn(self, x_spatial: torch.Tensor, x_freq: torch.Tensor):
        x_tokens_s, features_s, gs_s = self._forward_transformer_tokens(x_spatial)
        x_tokens_f, features_f, gs_f = self._forward_transformer_tokens(x_freq)

        if x_tokens_s is None:
            x3 = self._to_3ch_if_needed(x_spatial)
            seg_logits = self.anatomy_encoder(x3)
            feat = F.interpolate(seg_logits, size=(self.grid_size, self.grid_size), mode="bilinear", align_corners=False)
            rep = max(1, self.vit_hidden_dim // max(1, feat.shape[1]))
            feat = feat.repeat(1, rep, 1, 1)[:, :self.vit_hidden_dim]
            z_a = self.feat_proj(feat)
            return z_a, seg_logits, None, None, None

        if x_tokens_f is None:
            x_tokens_f = x_tokens_s

        if x_tokens_f.shape[1] != x_tokens_s.shape[1]:
            Ns = x_tokens_s.shape[1]
            Nf = x_tokens_f.shape[1]
            if Nf > Ns:
                x_tokens_f = x_tokens_f[:, :Ns, :]
            else:
                rep = (Ns + Nf - 1) // Nf
                x_tokens_f = x_tokens_f.repeat(1, rep, 1)[:, :Ns, :]

        gs = gs_s
        x_tokens_fused = self.cross_attn(x_tokens_s, x_tokens_f)

        x_dec = self.anatomy_encoder.decoder(x_tokens_fused, features_s)
        seg_logits = self.anatomy_encoder.segmentation_head(x_dec)

        featmap = self._tokens_to_featmap(x_tokens_fused, gs)
        z_a_fused = self.feat_proj(featmap)

        return z_a_fused, seg_logits, x_tokens_fused, x_tokens_s, x_tokens_f

    def forward(self, x_s, x_t, mode="train") -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}

        x_s_f = build_freq_view_from_image(x_s)
        x_t_f = build_freq_view_from_image(x_t)
        outputs["x_s_f"] = x_s_f
        outputs["x_t_f"] = x_t_f

        z_a_s, seg_logits_s, tok_s_fused, tok_s, tok_s_freq = self.extract_anatomy_feature_with_freq_xattn(x_s, x_s_f)
        z_a_t, seg_logits_t, tok_t_fused, tok_t, tok_t_freq = self.extract_anatomy_feature_with_freq_xattn(x_t, x_t_f)

        outputs["z_a_s"] = z_a_s
        outputs["z_a_t"] = z_a_t
        outputs["seg_logits_s_backbone"] = seg_logits_s
        outputs["seg_logits_t_backbone"] = seg_logits_t

        if tok_s_fused is not None:
            outputs["tok_s_fused"] = tok_s_fused
            outputs["tok_s"] = tok_s
            outputs["tok_s_freq"] = tok_s_freq
        if tok_t_fused is not None:
            outputs["tok_t_fused"] = tok_t_fused
            outputs["tok_t"] = tok_t
            outputs["tok_t_freq"] = tok_t_freq

        z_m_s = self.style_enc_source(x_s)
        z_m_t = self.style_enc_target(x_t)
        outputs["z_m_s"] = z_m_s
        outputs["z_m_t"] = z_m_t

        z_m_s_f = self.freq_style_enc_source(x_s_f)
        z_m_t_f = self.freq_style_enc_target(x_t_f)
        outputs["z_m_s_f"] = z_m_s_f
        outputs["z_m_t_f"] = z_m_t_f

        seg_pred_s, sdf_pred_s = self.dual_task(z_a_s)
        seg_pred_t, sdf_pred_t = self.dual_task(z_a_t)
        outputs["seg_pred_s"] = seg_pred_s
        outputs["sdf_pred_s"] = sdf_pred_s
        outputs["seg_pred_t"] = seg_pred_t
        outputs["sdf_pred_t"] = sdf_pred_t

        if mode == "eval":
            return outputs

        x_rec_s = self.generator(z_a_s.mean(dim=(2, 3)), z_m_s)
        x_rec_t = self.generator(z_a_t.mean(dim=(2, 3)), z_m_t)
        outputs["x_rec_s"] = x_rec_s
        outputs["x_rec_t"] = x_rec_t

        x_s2t = self.generator(z_a_s.mean(dim=(2, 3)), z_m_t)
        x_t2s = self.generator(z_a_t.mean(dim=(2, 3)), z_m_s)
        outputs["x_s2t"] = x_s2t
        outputs["x_t2s"] = x_t2s

        x_s2t_f = build_freq_view_from_image(x_s2t)
        z_a_s2t, _seg_tmp, _tfused, _ts, _tf = self.extract_anatomy_feature_with_freq_xattn(x_s2t, x_s2t_f)
        seg_pred_s2t, sdf_pred_s2t = self.dual_task(z_a_s2t)
        outputs["seg_pred_s2t"] = seg_pred_s2t
        outputs["sdf_pred_s2t"] = sdf_pred_s2t

        return outputs


# =============================================================================
# EMA
# =============================================================================
@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, m: float):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k in esd.keys():
        if not torch.is_floating_point(esd[k]):
            esd[k].copy_(msd[k])
        else:
            esd[k].mul_(m).add_(msd[k], alpha=(1.0 - m))


def copy_model(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad = False
    m.eval()
    return m


# =============================================================================
# Eval
# =============================================================================
@torch.no_grad()
def evaluate_binary_dice(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    dices = []
    for x, y, _name in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, x, mode="eval")
        logits = out["seg_pred_s"]
        prob = torch.softmax(logits, dim=1)[:, 1]
        d = dice_hard_binary(prob, y, thr=0.5)
        dices.append(float(d))
    model.train()
    return safe_mean(dices)


@torch.no_grad()
def build_pseudo_labels(ema_model: nn.Module, x_t: torch.Tensor, conf_thr: float):
    out = ema_model(x_t, x_t, mode="eval")
    logits = out["seg_pred_s"]
    prob_all = torch.softmax(logits, dim=1)  # (B,2,H,W)
    conf = torch.max(prob_all, dim=1)[0]     # (B,H,W)
    prob_fg = prob_all[:, 1]                # (B,H,W)
    pseudo = (prob_fg > 0.5).long()
    mask = (conf > float(conf_thr)).float()
    return pseudo, mask, prob_fg.unsqueeze(1), conf.unsqueeze(1)


# =============================================================================
# Dynamic Prior: EMA-gated MobileSAM prior
# =============================================================================
@torch.no_grad()
def build_dynamic_prior(
    ema_model: nn.Module,
    sam_prior: MobileSAMPrior,
    x_t: torch.Tensor,
    ema_conf_thr: float,
    sam_thr: float,
    fuse_mode: str = "mul",
    sam_neg_thr: float = 0.15,
    gate_mode: str = "fg_only",
    reliability_thr: float = 0.5,
    min_cov: float = 0.0,
    agree_power: float = 1.0,
    weight_mode: str = "binary",
):
    out = ema_model(x_t, x_t, mode="eval")
    logits = out["seg_pred_s"]
    prob_all = torch.softmax(logits, dim=1)
    prob_ema = prob_all[:, 1:2]
    conf_ema = torch.max(prob_all, dim=1, keepdim=True)[0]

    prob_sam = sam_prior(x_t)

    sam_cert = (2.0 * (prob_sam - 0.5).abs()).clamp(0.0, 1.0)
    agree = (1.0 - (prob_sam - prob_ema).abs()).clamp(0.0, 1.0)
    reliability = (sam_cert * conf_ema * torch.pow(agree, float(agree_power))).clamp(0.0, 1.0)

    ema_mask = (conf_ema > float(ema_conf_thr)).float()
    sam_fg_mask = (prob_sam > float(sam_thr)).float()
    sam_bg_mask = (prob_sam < float(sam_neg_thr)).float()
    sam_bidir_mask = ((sam_fg_mask + sam_bg_mask) > 0).float()

    if gate_mode == "fg_only":
        mask_dyn = ema_mask * sam_fg_mask
    elif gate_mode == "bidir":
        mask_dyn = ema_mask * sam_bidir_mask
    elif gate_mode == "reliability":
        mask_dyn = (reliability > float(reliability_thr)).float() * sam_bidir_mask
    elif gate_mode == "balanced":
        base_mask = (reliability > float(reliability_thr)).float() * ema_mask * sam_bidir_mask
        mask_dyn = base_mask
        if float(min_cov) > 0.0:
            B = reliability.shape[0]
            flat_rel = (reliability * sam_bidir_mask).view(B, -1)
            n_pix = flat_rel.shape[1]
            k = max(1, min(n_pix, int(float(min_cov) * float(n_pix))))
            kth = torch.topk(flat_rel, k=k, dim=1).values[:, -1].view(B, 1, 1, 1)
            cov_mask = ((reliability >= kth) & (reliability > 0) & (sam_bidir_mask > 0)).float()
            mask_dyn = torch.maximum(mask_dyn, cov_mask)
    else:
        mask_dyn = ema_mask * sam_fg_mask

    if weight_mode == "reliability":
        mask_dyn = mask_dyn * reliability.detach()

    if fuse_mode == "mul":
        prior_dyn = prob_sam * prob_ema
    elif fuse_mode == "avg":
        prior_dyn = 0.5 * (prob_sam + prob_ema)
    elif fuse_mode == "ema_only":
        prior_dyn = prob_ema
    else:
        prior_dyn = prob_sam * prob_ema

    prior_dyn = prior_dyn.clamp(0.0, 1.0)
    return prior_dyn, mask_dyn, prob_sam, prob_ema, conf_ema, reliability


@torch.no_grad()
def filter_pseudo_mask_by_reliability(
    pseudo: torch.Tensor,
    conf_mask: torch.Tensor,
    prob_sam: torch.Tensor,
    reliability: torch.Tensor,
    gate_mode: str,
    sam_thr: float,
    sam_neg_thr: float,
    reliability_thr: float,
    keep_ratio: float,
    min_cov: float,
    weight_mode: str,
) -> torch.Tensor:
    """Keep a class-balanced subset of EMA pseudo labels trusted by MobileSAM."""
    if gate_mode == "none":
        return conf_mask

    sam_fg = prob_sam[:, 0] > float(sam_thr)
    sam_bg = prob_sam[:, 0] < float(sam_neg_thr)
    if gate_mode == "bidir":
        directional = ((pseudo == 1) & sam_fg) | ((pseudo == 0) & sam_bg)
    else:
        directional = torch.ones_like(pseudo, dtype=torch.bool)

    base = (conf_mask > 0) & directional
    candidate = base & (reliability[:, 0] >= float(reliability_thr))
    binary_mask = torch.zeros_like(conf_mask)
    B, H, W = conf_mask.shape
    flat_pixels = H * W

    for b in range(B):
        for label in (0, 1):
            class_candidate = candidate[b] & (pseudo[b] == label)
            count = int(class_candidate.sum().item())
            if count == 0:
                continue
            k = max(1, min(count, int(math.ceil(float(keep_ratio) * count))))
            scores = reliability[b, 0].masked_fill(~class_candidate, -1.0).reshape(-1)
            chosen = torch.topk(scores, k=k).indices
            binary_mask[b].reshape(-1)[chosen] = 1.0

        min_pixels = int(math.ceil(float(min_cov) * flat_pixels))
        current = int(binary_mask[b].sum().item())
        if min_pixels > current:
            fallback = base[b]
            count = int(fallback.sum().item())
            if count > 0:
                k = min(count, min_pixels)
                scores = reliability[b, 0].masked_fill(~fallback, -1.0).reshape(-1)
                chosen = torch.topk(scores, k=k).indices
                binary_mask[b].reshape(-1)[chosen] = 1.0

    if weight_mode == "reliability":
        return binary_mask * reliability[:, 0].detach()
    return binary_mask


def masked_mse(p: torch.Tensor, q: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = (p - q) * m
    num = (diff * diff).sum()
    den = m.sum() + eps
    return num / den


def masked_kl(p: torch.Tensor, q: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    term = q * (torch.log(q) - torch.log(p))
    term = term * m
    return term.sum() / (m.sum() + eps)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--source_root", type=str, required=True)

    # GPT-style: target_root optional (only needed if you have GT val in same structure)
    parser.add_argument("--target_root", type=str, default="",
                        help="Optional. If provided and has val GT, will eval. For unlabeled target train, prefer --tgt_splits + --busbra_root.")

    parser.add_argument("--pretrained_path", type=str, default="",
                        help="can be .npz (imagenet21k) or .pt/.pth")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--vit_name", type=str, default="R50-ViT-B_16")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2.5e-4)

    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--align_epochs", type=int, default=20)

    # pseudo
    parser.add_argument("--pseudo_conf_thr", type=float, default=0.95)
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--lambda_pseudo", type=float, default=1.0)
    parser.add_argument("--pseudo_gate_mode", type=str, default="none", choices=["none", "reliability", "bidir"])
    parser.add_argument("--pseudo_reliability_thr", type=float, default=0.5)
    parser.add_argument("--pseudo_keep_ratio", type=float, default=1.0)
    parser.add_argument("--pseudo_min_cov", type=float, default=0.0)
    parser.add_argument("--pseudo_weight_mode", type=str, default="binary", choices=["binary", "reliability"])
    parser.add_argument("--pseudo_strong_epochs", type=int, default=0,
                        help="0 keeps pseudo supervision for all strong epochs; positive values stop it early.")

    # losses
    parser.add_argument("--lambda_adv_img", type=float, default=0.05)
    parser.add_argument("--lambda_adv_out", type=float, default=0.005)
    parser.add_argument("--lambda_rec", type=float, default=0.05)
    parser.add_argument("--lambda_sdf", type=float, default=0.1)

    parser.add_argument("--lambda_freq_align", type=float, default=0.1)
    parser.add_argument("--lambda_freq_dis", type=float, default=0.05)
    parser.add_argument("--lambda_tok_cons", type=float, default=0.0)

    # FDA v3
    parser.add_argument("--fda_enable", action="store_true")
    parser.add_argument("--fda_prob", type=float, default=0.5)
    parser.add_argument("--fda_beta", type=float, default=0.1)

    # Target unlabeled source (GPT-style): can use splits + busbra_root (like your first script)
    parser.add_argument("--tgt_splits", type=str, default="",
                        help="e.g. splits/target_unlabeled_train.txt (lines like BUSBRA::Images/xxx.png)")
    parser.add_argument("--busbra_root", type=str, default="",
                        help="e.g. /scratch/.../BUSBRA (contains Images/...)")

    # Dynamic prior with finetuned MobileSAM
    parser.add_argument("--sam_prior_ckpt", type=str, required=True)
    parser.add_argument("--sam_prior_type", type=str, default="vit_t")
    parser.add_argument("--sam_freeze", action="store_true")
    parser.add_argument("--lambda_prior", type=float, default=0.1)

    # dynamic prior gating
    parser.add_argument("--prior_ema_conf_thr", type=float, default=0.95)
    parser.add_argument("--prior_sam_thr", type=float, default=0.85)
    parser.add_argument("--prior_sam_neg_thr", type=float, default=0.15)
    parser.add_argument("--prior_fuse_mode", type=str, default="mul", choices=["mul", "avg", "ema_only"])
    parser.add_argument("--prior_gate_mode", type=str, default="fg_only",
                        choices=["fg_only", "bidir", "reliability", "balanced"])
    parser.add_argument("--prior_reliability_thr", type=float, default=0.5)
    parser.add_argument("--prior_min_cov", type=float, default=0.0)
    parser.add_argument("--prior_agree_power", type=float, default=1.0)
    parser.add_argument("--prior_weight_mode", type=str, default="binary", choices=["binary", "reliability"])

    # prior schedule: only early strong
    parser.add_argument("--prior_rampup", type=int, default=1)
    parser.add_argument("--prior_strong_epochs", type=int, default=8)
    parser.add_argument("--prior_loss_type", type=str, default="mse", choices=["mse", "kl", "mse_plus_dice"])
    parser.add_argument("--prior_use_soft_dice", action="store_true")

    # robust strong: optionally disable adversarial in strong
    parser.add_argument("--strong_disable_adv", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--source_only_train", action="store_true",
                        help="Use source images in both model branches so target samples never affect source-only training.")
    parser.add_argument("--selection_metric", type=str, default="source", choices=["source", "target"])

    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()

    seed_all(args.seed)
    set_torch_backends(deterministic=bool(args.deterministic))
    ensure_dir(args.output_dir)
    save_json(vars(args), os.path.join(args.output_dir, "args.json"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("=" * 90)
    print("RUN", now_str())
    print("Script", "distanglement_fda3_full_dynamic_prior_gptstyle.py")
    print("output_dir", args.output_dir)
    print("device", device)
    print("use_amp", bool(args.use_amp), "deterministic", bool(args.deterministic))
    print("source_root", args.source_root)
    print("target_root", args.target_root)
    print("tgt_splits", args.tgt_splits)
    print("busbra_root", args.busbra_root)
    print("pretrained_path", args.pretrained_path)
    print("sam_prior_ckpt", args.sam_prior_ckpt, "sam_type", args.sam_prior_type, "freeze", bool(args.sam_freeze))
    print("lambda_pseudo", float(args.lambda_pseudo), "pseudo_conf_thr", float(args.pseudo_conf_thr))
    print("lambda_prior", float(args.lambda_prior), "prior_fuse_mode", args.prior_fuse_mode)
    print("prior_ema_conf_thr", float(args.prior_ema_conf_thr), "prior_sam_thr", float(args.prior_sam_thr),
          "prior_sam_neg_thr", float(args.prior_sam_neg_thr))
    print("prior_gate_mode", args.prior_gate_mode, "prior_reliability_thr", float(args.prior_reliability_thr),
          "prior_min_cov", float(args.prior_min_cov), "prior_weight_mode", args.prior_weight_mode)
    print("prior_strong_epochs", int(args.prior_strong_epochs), "prior_loss_type", args.prior_loss_type)
    print("strong_disable_adv", bool(args.strong_disable_adv))
    print("fda_enable", bool(args.fda_enable), "fda_prob", float(args.fda_prob), "fda_beta", float(args.fda_beta))
    print("=" * 90)

    # ---- Source loaders (must have GT) ----
    src_tr_ds = SegListDataset(args.source_root, "train", args.img_size, True, use_labels=True)
    src_va_ds = SegListDataset(args.source_root, "val", args.img_size, True, use_labels=True)

    src_tr = DataLoader(src_tr_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    src_va = DataLoader(src_va_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)

    print("[SRC] train", len(src_tr_ds), "val", len(src_va_ds))

    # ---- Target train (unlabeled): prefer splits+busbra_root, else fallback target_root(train) without labels ----
    if args.tgt_splits and args.busbra_root:
        tgt_tr_ds = TargetSplitDataset(args.tgt_splits, args.busbra_root, args.img_size)
        tgt_tr = DataLoader(tgt_tr_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print("[TGT] unlabeled train samples =", len(tgt_tr_ds), "(TargetSplitDataset)")
        tgt_mode = "split"
    else:
        if not args.target_root:
            raise ValueError("Provide (--tgt_splits and --busbra_root) for unlabeled target, or set --target_root for fallback.")
        tgt_tr_ds = SegListDataset(args.target_root, "train", args.img_size, True, use_labels=False)
        tgt_tr = DataLoader(tgt_tr_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True)
        print("[TGT] unlabeled train samples =", len(tgt_tr_ds), "(fallback SegListDataset use_labels=False)")
        tgt_mode = "fallback"

    # ---- Target val (optional, only if GT exists) ----
    tgt_va = None
    if args.target_root:
        try:
            tgt_va_ds = SegListDataset(args.target_root, "val", args.img_size, True, use_labels=True)
            tgt_va = DataLoader(tgt_va_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)
            print("[TGT] val samples =", len(tgt_va_ds), "(with GT)")
        except Exception as e:
            print("[TGT] val disabled (no GT val). Reason:", repr(e))
            tgt_va = None

    # ---- Model ----
    backbone = build_transunet(args.vit_name, args.img_size, args.num_classes)
    load_pretrained_weights(backbone, args.pretrained_path)
    model = DLaST(backbone, args.num_classes, args.img_size, 256, 8, 1, xattn_heads=8).to(device)
    print("model params", count_parameters(model))

    ema_model = copy_model(model)

    if bool(args.eval_only):
        src_dice = evaluate_binary_dice(model, src_va, device)
        tgt_dice = evaluate_binary_dice(model, tgt_va, device) if tgt_va is not None else -1.0
        metrics = {"mode": "eval_only", "src_dice": float(src_dice), "tgt_dice": float(tgt_dice)}
        save_json(metrics, os.path.join(args.output_dir, "eval_metrics.json"))
        print("[EVAL] SRC Dice = %.4f | TGT Dice = %.4f" % (src_dice, tgt_dice))
        return

    # ---- MobileSAM prior ----
    if not os.path.isfile(args.sam_prior_ckpt):
        raise FileNotFoundError("sam_prior_ckpt not found: " + str(args.sam_prior_ckpt))

    sam_prior = MobileSAMPrior(
        ckpt_path=args.sam_prior_ckpt,
        device=device,
        sam_type=args.sam_prior_type,
        freeze=bool(args.sam_freeze)
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(args.use_amp and (autocast is not None)))

    best_selection = -1.0
    tgt_iter = iter(tgt_tr)
    metrics_jsonl = os.path.join(args.output_dir, "metrics.jsonl")

    # ---- Train ----
    for epoch in range(1, int(args.num_epochs) + 1):
        model.train()

        if epoch <= int(args.warmup_epochs):
            stage = "warmup"
        elif epoch <= int(args.warmup_epochs) + int(args.align_epochs):
            stage = "align"
        else:
            stage = "strong"

        if stage == "warmup":
            grl = 0.0
        elif stage == "align":
            cur = epoch - int(args.warmup_epochs)
            grl = sigmoid_rampup(cur, max(1, int(args.align_epochs)))
            grl = float(min(1.0, grl))
        else:
            grl = 0.0

        if stage == "strong" and bool(args.strong_disable_adv):
            lambda_adv_img_eff = 0.0
            lambda_adv_out_eff = 0.0
            lambda_rec_eff = 0.0
        else:
            lambda_adv_img_eff = float(args.lambda_adv_img) if (stage in ["align", "strong"]) else 0.0
            lambda_adv_out_eff = float(args.lambda_adv_out) if (stage in ["align", "strong"]) else 0.0
            lambda_rec_eff = float(args.lambda_rec) if (stage in ["align", "strong"]) else 0.0

        strong_ep = 0
        if stage == "strong":
            strong_ep = epoch - (int(args.warmup_epochs) + int(args.align_epochs))
            if strong_ep <= int(args.prior_strong_epochs):
                w_prior = float(args.lambda_prior) * sigmoid_rampup(strong_ep, max(1, int(args.prior_rampup)))
            else:
                w_prior = 0.0
        else:
            w_prior = 0.0

        if stage == "strong" and (int(args.pseudo_strong_epochs) <= 0 or strong_ep <= int(args.pseudo_strong_epochs)):
            w_pseudo = float(args.lambda_pseudo)
        else:
            w_pseudo = 0.0

        t0 = time.time()
        running = {
            "loss": 0.0, "seg": 0.0, "sdf": 0.0, "rec": 0.0,
            "adv_img": 0.0, "adv_out": 0.0,
            "pseudo": 0.0, "prior": 0.0,
            "freq_align": 0.0, "freq_dis": 0.0, "tok_cons": 0.0,
            "prior_cov": 0.0,
            "pseudo_cov": 0.0, "pseudo_fg": 0.0,
            "sam_fg": 0.0, "ema_cov": 0.0
        }
        n_it = 0

        for it, (x_s, y_s, _name_s) in enumerate(src_tr, start=1):
            # ---- get a target batch (supports both TargetSplitDataset and fallback SegListDataset) ----
            try:
                batch_t = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_tr)
                batch_t = next(tgt_iter)

            if isinstance(batch_t, (tuple, list)) and len(batch_t) == 2:
                x_t, _key_t = batch_t
            else:
                x_t, _y_dummy, _name_t = batch_t

            x_s = x_s.to(device, non_blocking=True)
            y_s = y_s.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            # FDA v3
            if bool(args.fda_enable):
                p = float(args.fda_prob)
                if (p >= 1.0) or (random.random() < p):
                    x_s_in = fda_source_to_target(x_s, x_t, beta=float(args.fda_beta))
                else:
                    x_s_in = x_s
            else:
                x_s_in = x_s

            opt.zero_grad(set_to_none=True)

            with autocast(enabled=(args.use_amp and (autocast is not None))):
                x_t_model = x_s_in if bool(args.source_only_train) else x_t
                out = model(x_s_in, x_t_model, mode="train")

                seg_s = out["seg_pred_s"]
                sdf_s = out["sdf_pred_s"]
                seg_t = out["seg_pred_t"]
                sdf_t = out["sdf_pred_t"]

                # source supervised
                loss_seg = F.cross_entropy(seg_s, y_s) + dice_loss_soft(seg_s, y_s)

                sdf_gt = compute_sdf_batch(y_s, args.num_classes)
                loss_sdf = F.l1_loss(sdf_s, sdf_gt)

                # freq losses
                z_a_s_vec = out["z_a_s"].mean(dim=(2, 3))
                z_a_t_vec = out["z_a_t"].mean(dim=(2, 3))

                z_m_s = out["z_m_s"]
                z_m_t = out["z_m_t"]
                z_m_s_f = out["z_m_s_f"]
                z_m_t_f = out["z_m_t_f"]

                loss_freq_align = torch.tensor(0.0, device=device)
                loss_freq_dis = torch.tensor(0.0, device=device)

                if ("tok_s_freq" in out) and (out["tok_s_freq"] is not None):
                    tok_s_freq = out["tok_s_freq"]
                    tok_t_freq = out["tok_t_freq"]

                    B, N, D = tok_s_freq.shape
                    gs = int(math.sqrt(N))
                    if gs * gs != N:
                        gs = model._infer_grid(N)

                    if gs * gs == N:
                        tok_s_freq_map = tok_s_freq.permute(0, 2, 1).contiguous().view(B, D, gs, gs)
                        tok_t_freq_map = tok_t_freq.permute(0, 2, 1).contiguous().view(B, D, gs, gs)

                        z_a_s_f = model.feat_proj(tok_s_freq_map)
                        z_a_t_f = model.feat_proj(tok_t_freq_map)

                        z_a_s_f_vec = z_a_s_f.mean(dim=(2, 3))
                        z_a_t_f_vec = z_a_t_f.mean(dim=(2, 3))
                    else:
                        z_a_s_f_vec = tok_s_freq.mean(dim=1)[:, :model.content_dim]
                        z_a_t_f_vec = tok_t_freq.mean(dim=1)[:, :model.content_dim]

                    loss_freq_align = F.l1_loss(z_a_s_vec, z_a_s_f_vec) + F.l1_loss(z_a_t_vec, z_a_t_f_vec)

                    z_m_s_c = model.style_proj(z_m_s)
                    z_m_t_c = model.style_proj(z_m_t)
                    z_m_s_f_c = model.style_proj_f(z_m_s_f)
                    z_m_t_f_c = model.style_proj_f(z_m_t_f)

                    loss_freq_dis = (
                        cosine_ortho(z_m_s_c, z_a_s_f_vec) +
                        cosine_ortho(z_m_s_f_c, z_a_s_vec) +
                        cosine_ortho(z_m_t_c, z_a_t_f_vec) +
                        cosine_ortho(z_m_t_f_c, z_a_t_vec)
                    )

                loss_tok_cons = torch.tensor(0.0, device=device)
                if float(args.lambda_tok_cons) > 0.0 and ("tok_s_fused" in out) and (out["tok_s_fused"] is not None):
                    loss_tok_cons = F.mse_loss(out["tok_s_fused"], out["tok_s"]) + F.mse_loss(out["tok_t_fused"], out["tok_t"])

                # recon + adversarial
                loss_rec = torch.tensor(0.0, device=device)
                loss_adv_img = torch.tensor(0.0, device=device)
                loss_adv_out = torch.tensor(0.0, device=device)

                if stage in ["align", "strong"]:
                    if (lambda_rec_eff > 0.0) or (lambda_adv_img_eff > 0.0) or (lambda_adv_out_eff > 0.0):
                        x_rec_s = out["x_rec_s"]
                        x_rec_t = out["x_rec_t"]
                        loss_rec = l1_loss(x_rec_s, x_s_in) + l1_loss(x_rec_t, x_t)

                        d_s2t = model.disc_target(out["x_s2t"])
                        d_t2s = model.disc_source(out["x_t2s"])
                        loss_adv_img = bce_logits_loss(d_s2t, True) + bce_logits_loss(d_t2s, True)

                        seg_s_grl = gradient_reversal(seg_s, grl)
                        seg_t_grl = gradient_reversal(seg_t, grl)
                        ds_s = model.disc_seg(torch.softmax(seg_s_grl, dim=1))
                        ds_t = model.disc_seg(torch.softmax(seg_t_grl, dim=1))

                        sdf_s_grl = gradient_reversal(sdf_s, grl)
                        sdf_t_grl = gradient_reversal(sdf_t, grl)
                        dd_s = model.disc_sdf(sdf_s_grl)
                        dd_t = model.disc_sdf(sdf_t_grl)

                        loss_adv_out = (
                            bce_logits_loss(ds_s, True) + bce_logits_loss(ds_t, False) +
                            bce_logits_loss(dd_s, True) + bce_logits_loss(dd_t, False)
                        )

                # strong only: EMA pseudo + dynamic prior
                loss_pseudo = torch.tensor(0.0, device=device)
                loss_prior = torch.tensor(0.0, device=device)
                prior_cov = torch.tensor(0.0, device=device)
                pseudo_cov = torch.tensor(0.0, device=device)
                pseudo_fg = torch.tensor(0.0, device=device)
                sam_fg = torch.tensor(0.0, device=device)
                ema_cov = torch.tensor(0.0, device=device)
                prior_rel = torch.tensor(0.0, device=device)

                if stage == "strong" and (w_pseudo > 0.0 or w_prior > 0.0):
                    pseudo, mask_pseudo, _prob_ema_fg, _conf_ema = build_pseudo_labels(ema_model, x_t, float(args.pseudo_conf_thr))

                    prior_outputs = None
                    need_reliable_pseudo = str(args.pseudo_gate_mode) != "none"
                    if w_prior > 0.0 or need_reliable_pseudo:
                        prior_outputs = build_dynamic_prior(
                            ema_model=ema_model,
                            sam_prior=sam_prior,
                            x_t=x_t,
                            ema_conf_thr=float(args.prior_ema_conf_thr),
                            sam_thr=float(args.prior_sam_thr),
                            fuse_mode=str(args.prior_fuse_mode),
                            sam_neg_thr=float(args.prior_sam_neg_thr),
                            gate_mode=str(args.prior_gate_mode),
                            reliability_thr=float(args.prior_reliability_thr),
                            min_cov=float(args.prior_min_cov),
                            agree_power=float(args.prior_agree_power),
                            weight_mode=str(args.prior_weight_mode),
                        )
                        prior_dyn, mask_dyn, _prob_sam, _prob_ema2, _conf_ema2, _rel_prior = prior_outputs

                    if need_reliable_pseudo:
                        mask_pseudo = filter_pseudo_mask_by_reliability(
                            pseudo=pseudo,
                            conf_mask=mask_pseudo,
                            prob_sam=_prob_sam,
                            reliability=_rel_prior,
                            gate_mode=str(args.pseudo_gate_mode),
                            sam_thr=float(args.prior_sam_thr),
                            sam_neg_thr=float(args.prior_sam_neg_thr),
                            reliability_thr=float(args.pseudo_reliability_thr),
                            keep_ratio=float(args.pseudo_keep_ratio),
                            min_cov=float(args.pseudo_min_cov),
                            weight_mode=str(args.pseudo_weight_mode),
                        )

                    ce_map = F.cross_entropy(seg_t, pseudo, reduction="none")
                    denom = mask_pseudo.sum().clamp_min(1.0)
                    ce_t = (ce_map * mask_pseudo).sum() / denom

                    dice_t = dice_loss_soft_masked(seg_t, pseudo, mask_pseudo)
                    loss_pseudo = ce_t + dice_t
                    pseudo_binary = (mask_pseudo > 0).float()
                    pseudo_cov = pseudo_binary.mean()
                    pseudo_fg = (pseudo.float() * pseudo_binary).sum() / pseudo_binary.sum().clamp_min(1.0)

                    if w_prior > 0.0:
                        prob_model = torch.softmax(seg_t, dim=1)[:, 1:2]
                        prior_cov = (mask_dyn > 0).float().mean()
                        sam_fg = (_prob_sam > float(args.prior_sam_thr)).float().mean()
                        ema_cov = (_conf_ema2 > float(args.prior_ema_conf_thr)).float().mean()
                        prior_rel = _rel_prior.mean()

                        if args.prior_loss_type == "mse":
                            loss_prior = masked_mse(prob_model, prior_dyn.detach(), mask_dyn)
                        elif args.prior_loss_type == "kl":
                            loss_prior = masked_kl(prob_model, prior_dyn.detach(), mask_dyn)
                        else:
                            loss_prior = masked_mse(prob_model, prior_dyn.detach(), mask_dyn)
                            dsoft = soft_dice_between_probs(prob_model * mask_dyn, prior_dyn.detach() * mask_dyn)
                            loss_prior = loss_prior + (1.0 - dsoft)

                        if bool(args.prior_use_soft_dice):
                            dsoft2 = soft_dice_between_probs(prob_model * mask_dyn, prior_dyn.detach() * mask_dyn)
                            loss_prior = loss_prior + (1.0 - dsoft2)

                # total
                loss = loss_seg + float(args.lambda_sdf) * loss_sdf
                loss = loss + float(args.lambda_freq_align) * loss_freq_align + float(args.lambda_freq_dis) * loss_freq_dis
                loss = loss + float(args.lambda_tok_cons) * loss_tok_cons

                if stage in ["align", "strong"]:
                    loss = loss + lambda_rec_eff * loss_rec + lambda_adv_img_eff * loss_adv_img + lambda_adv_out_eff * loss_adv_out

                if stage == "strong":
                    loss = loss + w_pseudo * loss_pseudo
                    if w_prior > 0.0:
                        loss = loss + float(w_prior) * loss_prior

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ema_update(ema_model, model, float(args.ema_momentum))

            running["loss"] += float(loss.item())
            running["seg"] += float(loss_seg.item())
            running["sdf"] += float(loss_sdf.item())
            running["rec"] += float(loss_rec.item())
            running["adv_img"] += float(loss_adv_img.item())
            running["adv_out"] += float(loss_adv_out.item())
            running["pseudo"] += float(loss_pseudo.item())
            running["prior"] += float(loss_prior.item())
            running["freq_align"] += float(loss_freq_align.item())
            running["freq_dis"] += float(loss_freq_dis.item())
            running["tok_cons"] += float(loss_tok_cons.item())
            running["prior_cov"] += float(prior_cov.item())
            running["pseudo_cov"] += float(pseudo_cov.item())
            running["pseudo_fg"] += float(pseudo_fg.item())
            running["sam_fg"] += float(sam_fg.item())
            running["ema_cov"] += float(ema_cov.item())
            if "prior_rel" not in running:
                running["prior_rel"] = 0.0
            running["prior_rel"] += float(prior_rel.item())
            n_it += 1

            if (it % 20) == 0:
                denom2 = max(1, n_it)
                print("[E %d] [it %d/%d] stage %s grl %.3f w_prior %.3f "
                      "loss %.4f seg %.4f sdf %.4f rec %.4f adv_img %.4f adv_out %.4f "
                      "pseudo %.4f prior %.4f freqA %.4f freqD %.4f tokC %.4f "
                      "pseudo_cov %.4f pseudo_fg %.4f prior_cov %.4f prior_rel %.4f sam_fg %.4f ema_cov %.4f elapsed_s %.1f" %
                      (epoch, it, len(src_tr), stage, grl, float(w_prior),
                       running["loss"]/denom2, running["seg"]/denom2, running["sdf"]/denom2,
                       running["rec"]/denom2, running["adv_img"]/denom2, running["adv_out"]/denom2,
                       running["pseudo"]/denom2, running["prior"]/denom2,
                       running["freq_align"]/denom2, running["freq_dis"]/denom2, running["tok_cons"]/denom2,
                       running["pseudo_cov"]/denom2, running["pseudo_fg"]/denom2,
                       running["prior_cov"]/denom2, running["prior_rel"]/denom2,
                       running["sam_fg"]/denom2, running["ema_cov"]/denom2,
                       time.time() - t0))

        # eval
        src_dice = evaluate_binary_dice(model, src_va, device)
        if tgt_va is not None:
            tgt_dice = evaluate_binary_dice(model, tgt_va, device)
            print("[VAL] SRC Dice = %.4f | TGT Dice = %.4f" % (src_dice, tgt_dice))
        else:
            tgt_dice = -1.0
            print("[VAL] SRC Dice = %.4f | TGT Dice = N/A (no GT val)" % (src_dice))

        denom_epoch = max(1, n_it)
        epoch_metrics = {
            "epoch": int(epoch),
            "stage": stage,
            "src_dice": float(src_dice),
            "tgt_dice": float(tgt_dice),
            "loss": running["loss"] / denom_epoch,
            "pseudo_loss": running["pseudo"] / denom_epoch,
            "prior_loss": running["prior"] / denom_epoch,
            "pseudo_cov": running["pseudo_cov"] / denom_epoch,
            "pseudo_reject": 1.0 - (running["pseudo_cov"] / denom_epoch),
            "pseudo_fg": running["pseudo_fg"] / denom_epoch,
            "prior_cov": running["prior_cov"] / denom_epoch,
            "prior_reject": 1.0 - (running["prior_cov"] / denom_epoch),
            "prior_rel": running.get("prior_rel", 0.0) / denom_epoch,
            "sam_fg": running["sam_fg"] / denom_epoch,
            "ema_cov": running["ema_cov"] / denom_epoch,
        }
        with open(metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_metrics, ensure_ascii=False) + "\n")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "opt": opt.state_dict(),
            "best_selection": best_selection,
            "args": vars(args)
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))

        selection_score = src_dice if args.selection_metric == "source" else tgt_dice
        if selection_score >= 0.0 and selection_score > best_selection:
            best_selection = selection_score
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            print("saved best %s dice %.4f | target %.4f" %
                  (args.selection_metric, best_selection, tgt_dice))


if __name__ == "__main__":
    main()
