"""Frozen foundation priors: prompt-free AutoSAMUS and box/point-promptable SAMUS / SAM."""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

SAMUS_ROOT = os.environ.get("SAMUS_ROOT", os.path.join(os.path.dirname(__file__), "samus"))
if SAMUS_ROOT not in sys.path:
    sys.path.insert(0, SAMUS_ROOT)


class AutoSAMUS(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        from models.segment_anything_samus_autoprompt.build_samus import autosamus_model_registry
        self.m = autosamus_model_registry["vit_b"](SimpleNamespace(encoder_input_size=256), checkpoint=ckpt)
        self.m.eval()
        for q in self.m.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):  # x [B,1,256,256] in [0,1]
        return torch.sigmoid(self.m(x, None)["masks"])


class PromptSAMUS(nn.Module):
    """SAMUS (US30K-pretrained, ViT-B, 256 input) driven by box prompts."""

    def __init__(self, ckpt, kind="samus"):
        super().__init__()
        self.kind = kind
        if kind == "samus":
            from models.segment_anything_samus.build_sam_us import samus_model_registry
            self.m = samus_model_registry["vit_b"](SimpleNamespace(encoder_input_size=256), checkpoint=ckpt)
            self.size = 256
        else:  # original SAM ViT-B at 1024
            from models.segment_anything.build_sam import sam_model_registry
            self.m = sam_model_registry["vit_b"](checkpoint=ckpt)
            self.size = 1024
            self.register_buffer("mu", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1) / 255)
            self.register_buffer("sd", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1) / 255)
        self.m.eval()
        for q in self.m.parameters():
            q.requires_grad_(False)

    @torch.no_grad()
    def embed(self, x):
        if self.kind == "samus":
            return self.m.image_encoder(x)
        x = F.interpolate(x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x, size=(1024, 1024), mode="bilinear", align_corners=False)
        return self.m.image_encoder((x - self.mu) / self.sd)

    @torch.no_grad()
    def decode(self, emb, boxes, multimask=False):
        """boxes [B,4] (x0,y0,x1,y1) in 256-pixel coordinates. Returns prob [B,1,256,256] and iou."""
        b = boxes * (self.size / 256.0)
        se, de = self.m.prompt_encoder(points=None, boxes=b, masks=None)
        low, iou = self.m.mask_decoder(image_embeddings=emb, image_pe=self.m.prompt_encoder.get_dense_pe(),
                                       sparse_prompt_embeddings=se, dense_prompt_embeddings=de,
                                       multimask_output=multimask)
        if multimask:
            k = iou.argmax(1)
            low = low[torch.arange(low.shape[0]), k][:, None]
            iou = iou.max(1, keepdim=True)[0]
        prob = torch.sigmoid(F.interpolate(low, (256, 256), mode="bilinear", align_corners=False))
        return prob, iou

    @torch.no_grad()
    def forward(self, x, boxes):
        return self.decode(self.embed(x), boxes)


def mask_to_box(m, pad=0.0, min_size=4.0):
    """m [B,1,H,W] binary -> boxes [B,4] and validity; empty masks give a centered default box."""
    B, _, H, W = m.shape
    ys = m.squeeze(1).any(2)  # [B,H]
    xs = m.squeeze(1).any(1)  # [B,W]
    boxes = torch.zeros(B, 4, device=m.device)
    valid = ys.any(1)
    ar_h = torch.arange(H, device=m.device).float(); ar_w = torch.arange(W, device=m.device).float()
    big = 1e9
    y0 = torch.where(ys, ar_h, torch.full_like(ar_h, big)).min(1)[0]
    y1 = torch.where(ys, ar_h, torch.full_like(ar_h, -big)).max(1)[0]
    x0 = torch.where(xs, ar_w, torch.full_like(ar_w, big)).min(1)[0]
    x1 = torch.where(xs, ar_w, torch.full_like(ar_w, -big)).max(1)[0]
    bw = (x1 - x0).clamp_min(min_size); bh = (y1 - y0).clamp_min(min_size)
    boxes = torch.stack([x0 - pad * bw, y0 - pad * bh, x1 + pad * bw, y1 + pad * bh], 1).clamp(0, 255)
    default = torch.tensor([64., 64., 192., 192.], device=m.device)
    boxes = torch.where(valid[:, None], boxes, default)
    return boxes, valid


def largest_cc(m):
    """Keep the largest connected component per image (CPU, scipy)."""
    import numpy as np
    from scipy import ndimage
    out = m.clone()
    a = m.squeeze(1).cpu().numpy() > 0
    for i in range(a.shape[0]):
        lab, n = ndimage.label(a[i])
        if n > 1:
            sizes = ndimage.sum(a[i], lab, range(1, n + 1))
            keep = lab == (int(np.argmax(sizes)) + 1)
            out[i, 0] = torch.from_numpy(keep).to(m.device).float()
    return out
