#!/usr/bin/env python3
"""Bidirectional reliable-prior UDA with a frozen AutoSAMUS teacher.

This training entry intentionally excludes FDA, frequency disentanglement, DLaST,
SDF, and adversarial alignment. It isolates the proposed contribution: agreement-
based, foreground/background-aware pseudo supervision from AutoSAMUS and EMA.
"""

import argparse
import copy
import json
import math
import os
import sys
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Reuse only the data readers and TransUNet builder from the existing project.
from disentanglement_fda_1 import (
    SegListDataset,
    TargetSplitDataset,
    build_transunet,
    dice_loss_soft,
    load_pretrained_weights,
    seed_all,
    set_torch_backends,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(data, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


@torch.no_grad()
def evaluate_dice(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    values = []
    for x, y, _name in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prob = torch.softmax(model(x), dim=1)[:, 1]
        pred = prob >= 0.5
        target = y > 0
        inter = (pred & target).float().sum(dim=(1, 2))
        denom = pred.float().sum(dim=(1, 2)) + target.float().sum(dim=(1, 2))
        values.extend(((2.0 * inter + 1e-5) / (denom + 1e-5)).tolist())
    model.train()
    return float(sum(values) / max(1, len(values)))


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Dice over selected pixels, retaining class balance in a binary task."""
    prob = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    mask = mask.float()
    inter = (prob * target * mask).sum(dim=(1, 2))
    denom = ((prob + target) * mask).sum(dim=(1, 2))
    valid = (mask.sum(dim=(1, 2)) > 0).float()
    dice = (2.0 * inter + 1e-5) / (denom + 1e-5)
    return ((1.0 - dice) * valid).sum() / valid.sum().clamp_min(1.0)


def ultrasound_style_augment(x: torch.Tensor, strength: float) -> torch.Tensor:
    """Simulate moderate ultrasound gain, speckle, and resolution variation on source images."""
    if strength <= 0:
        return x
    batch = x.shape[0]
    shape = (batch, 1, 1, 1)
    gain = 1.0 + (2.0 * torch.rand(shape, device=x.device) - 1.0) * strength
    gamma = 1.0 + (2.0 * torch.rand(shape, device=x.device) - 1.0) * strength
    base = x.clamp(0.0, 1.0).pow(gamma) * gain
    speckle = torch.randn_like(base) * (strength * 0.35)
    styled = (base + base * speckle).clamp(0.0, 1.0)
    blurred = F.avg_pool2d(styled, kernel_size=3, stride=1, padding=1)
    blend = torch.rand(shape, device=x.device) * (strength * 0.5)
    return (1.0 - blend) * styled + blend * blurred


class FrozenAutoSAMUS(nn.Module):
    """Prompt-free AutoSAMUS prediction at the UDA student's image resolution."""

    def __init__(self, repo_root: str, checkpoint: str, device: torch.device):
        super().__init__()
        if not os.path.isdir(repo_root):
            raise FileNotFoundError("AutoSAMUS repository not found: " + repo_root)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("AutoSAMUS checkpoint not found: " + checkpoint)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from models.segment_anything_samus_autoprompt.build_samus import autosamus_model_registry

        build_args = SimpleNamespace(encoder_input_size=256)
        self.model = autosamus_model_registry["vit_b"](build_args, checkpoint=checkpoint).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = x.shape[-2:]
        x_256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        outputs = self.model(x_256, None)
        logits = outputs["masks"]
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


@torch.no_grad()
def build_reliable_labels(
    ema_model: nn.Module,
    autosamus: Optional[FrozenAutoSAMUS],
    x_target: torch.Tensor,
    mode: str,
    ema_conf_thr: float,
    autosamus_fg_thr: float,
    autosamus_bg_thr: float,
    autosamus_cert_thr: float,
    reliability_thr: float,
    keep_ratio: float,
    min_cov: float,
    agree_power: float,
    route_margin: float,
):
    """Build labels for each ablation while keeping the student protocol fixed.

    ``bidir`` is the proposed version: foreground and background must agree
    semantically across the dynamic EMA and frozen AutoSAMUS teachers, then both
    classes are retained independently. Other modes remove one assumption at a
    time for a direct ablation.
    """
    # Direct frozen-prior pseudo-labeling baseline. It deliberately bypasses
    # EMA predictions, agreement, and reliability selection: m(u)=1 everywhere.
    if mode == "hard_prior":
        if autosamus is None:
            raise RuntimeError("hard_prior mode requires the frozen AutoSAMUS prior")
        fm_prob = autosamus(x_target)[:, 0]
        fm_cert = 2.0 * torch.abs(fm_prob - 0.5)
        pseudo = (fm_prob >= 0.5).long()
        mask = torch.ones_like(fm_prob)
        diagnostics = {
            "pseudo_cov": 1.0,
            "pseudo_fg": float(pseudo.float().mean().item()),
            "reliability": float(fm_cert.mean().item()),
            "agreement": 0.0,
            "fm_fg": float(pseudo.float().mean().item()),
            "routed_to_fm": 1.0,
        }
        return pseudo, mask, diagnostics

    ema_prob_all = torch.softmax(ema_model(x_target), dim=1)
    ema_prob = ema_prob_all[:, 1]
    ema_conf = ema_prob_all.max(dim=1)[0]
    ema_pseudo = (ema_prob >= 0.5).long()
    uses_foundation = mode != "ema"
    if uses_foundation:
        if autosamus is None:
            raise RuntimeError("AutoSAMUS is required for pseudo mode: " + mode)
        fm_prob = autosamus(x_target)[:, 0]
        fm_cert = (2.0 * (fm_prob - 0.5).abs()).clamp(0.0, 1.0)
        agreement = (1.0 - (ema_prob - fm_prob).abs()).clamp(0.0, 1.0)
        reliability = (fm_cert * ema_conf * agreement.pow(float(agree_power))).clamp(0.0, 1.0)
    else:
        fm_prob = torch.zeros_like(ema_prob)
        fm_cert = torch.zeros_like(ema_prob)
        agreement = torch.ones_like(ema_prob)
        reliability = ema_conf

    pseudo = (fm_prob >= 0.5).long() if mode == "autosamus" else ema_pseudo
    routed_to_fm = torch.zeros_like(ema_prob)
    if mode == "ema":
        base = ema_conf >= float(ema_conf_thr)
        candidate = base
    elif mode == "autosamus":
        base = fm_cert >= float(autosamus_cert_thr)
        candidate = base
    elif mode == "agreement":
        base = ema_conf >= float(ema_conf_thr)
        candidate = base & (reliability >= float(reliability_thr))
    elif mode == "fg_only":
        base = (ema_conf >= float(ema_conf_thr)) & (ema_pseudo == 1) & (
            fm_prob >= float(autosamus_fg_thr)
        )
        candidate = base & (reliability >= float(reliability_thr))
    elif mode in {"bidir", "bidir_binary"}:
        direction = ((ema_pseudo == 1) & (fm_prob >= float(autosamus_fg_thr))) | (
            (ema_pseudo == 0) & (fm_prob <= float(autosamus_bg_thr))
        )
        base = (ema_conf >= float(ema_conf_thr)) & direction
        candidate = base & (reliability >= float(reliability_thr))
    elif mode == "route":
        fm_pseudo = (fm_prob >= 0.5).long()
        teacher_ema = ema_conf * agreement.pow(float(agree_power))
        teacher_fm = fm_cert * agreement.pow(float(agree_power))
        routed_to_fm = teacher_fm > (teacher_ema + float(route_margin))
        pseudo = torch.where(routed_to_fm, fm_pseudo, ema_pseudo)
        reliability = torch.maximum(teacher_ema, teacher_fm)
        same_label = ema_pseudo == fm_pseudo
        decisive_conflict = (teacher_ema - teacher_fm).abs() >= float(route_margin)
        base = (same_label | decisive_conflict) & (reliability >= float(reliability_thr))
        candidate = base
    else:
        raise ValueError("Unknown pseudo mode: " + mode)
    mask = torch.zeros_like(ema_prob)
    B, H, W = mask.shape
    pixels = H * W

    for b in range(B):
        for label in (0, 1):
            class_candidate = candidate[b] & (pseudo[b] == label)
            count = int(class_candidate.sum().item())
            if count == 0:
                continue
            k = max(1, min(count, int(math.ceil(float(keep_ratio) * count))))
            scores = reliability[b].masked_fill(~class_candidate, -1.0).reshape(-1)
            chosen = torch.topk(scores, k=k).indices
            weights = (
                torch.ones_like(reliability)
                if mode in {"ema", "autosamus", "bidir_binary"}
                else reliability
            )
            mask[b].reshape(-1)[chosen] = weights[b].reshape(-1)[chosen]

        minimum = int(math.ceil(float(min_cov) * pixels))
        selected = int((mask[b] > 0).sum().item())
        if minimum > selected and int(base[b].sum().item()) > 0:
            k = min(int(base[b].sum().item()), minimum)
            scores = reliability[b].masked_fill(~base[b], -1.0).reshape(-1)
            chosen = torch.topk(scores, k=k).indices
            weights = (
                torch.ones_like(reliability)
                if mode in {"ema", "autosamus", "bidir_binary"}
                else reliability
            )
            mask[b].reshape(-1)[chosen] = weights[b].reshape(-1)[chosen]

    binary_mask = (mask > 0).float()
    diagnostics = {
        "pseudo_cov": float(binary_mask.mean().item()),
        "pseudo_fg": float((pseudo.float() * binary_mask).sum().item() / binary_mask.sum().clamp_min(1.0).item()),
        "reliability": float(reliability.mean().item()),
        "agreement": float(agreement.mean().item()),
        "fm_fg": float((fm_prob >= float(autosamus_fg_thr)).float().mean().item()) if uses_foundation else 0.0,
        "routed_to_fm": float(routed_to_fm.float().mean().item()),
    }
    return pseudo, mask, diagnostics


def copy_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad = False
    return ema


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, momentum: float) -> None:
    for ema_value, value in zip(ema.state_dict().values(), model.state_dict().values()):
        if torch.is_floating_point(ema_value):
            ema_value.mul_(momentum).add_(value, alpha=1.0 - momentum)
        else:
            ema_value.copy_(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True)
    parser.add_argument("--target_root", required=True)
    parser.add_argument("--tgt_splits", required=True)
    parser.add_argument("--busbra_root", required=True)
    parser.add_argument("--pretrained_path", default="")
    parser.add_argument("--autosamus_root", required=True)
    parser.add_argument("--autosamus_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=55)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--pseudo_epochs", type=int, default=30)
    parser.add_argument(
        "--foundation_first_epochs", type=int, default=0,
        help="Within pseudo training, use the frozen foundation teacher alone before the requested mode.",
    )
    parser.add_argument(
        "--pseudo_mode", choices=["ema", "autosamus", "agreement", "fg_only", "bidir", "bidir_binary", "hard_prior", "route"], default="bidir",
        help="Ablation of pseudo-label source and reliability selection.",
    )
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--source_style_strength", type=float, default=0.0)
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--lambda_pseudo", type=float, default=0.15)
    parser.add_argument(
        "--lambda_entropy", type=float, default=0.0,
        help="Target prediction-entropy penalty for a MinEnt UDA baseline; zero disables it.",
    )
    parser.add_argument("--ema_conf_thr", type=float, default=0.90)
    parser.add_argument("--autosamus_fg_thr", type=float, default=0.85)
    parser.add_argument("--autosamus_bg_thr", type=float, default=0.12)
    parser.add_argument("--autosamus_cert_thr", type=float, default=0.60)
    parser.add_argument("--reliability_thr", type=float, default=0.55)
    parser.add_argument("--keep_ratio", type=float, default=0.25)
    parser.add_argument("--min_cov", type=float, default=0.02)
    parser.add_argument("--agree_power", type=float, default=1.5)
    parser.add_argument("--route_margin", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=0, help="Optional cap for a smoke run.")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_target_eval", action="store_true", help="Do not read target labels during source-only training.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--use_amp", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    save_json(vars(args), os.path.join(args.output_dir, "args.json"))
    seed_all(args.seed)
    set_torch_backends(deterministic=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    src_train = DataLoader(
        SegListDataset(args.source_root, "train", args.img_size, True, use_labels=True),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    src_val = DataLoader(
        SegListDataset(args.source_root, "val", args.img_size, True, use_labels=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    tgt_train = DataLoader(
        TargetSplitDataset(args.tgt_splits, args.busbra_root, args.img_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    tgt_val = DataLoader(
        SegListDataset(args.target_root, "val", args.img_size, True, use_labels=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    student = build_transunet("R50-ViT-B_16", args.img_size, 2).to(device)
    load_pretrained_weights(student, args.pretrained_path)
    ema = copy_ema(student)
    autosamus = None
    if args.pseudo_epochs > 0 and args.pseudo_mode != "ema":
        autosamus = FrozenAutoSAMUS(args.autosamus_root, args.autosamus_ckpt, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=bool(args.use_amp))
    target_iter = iter(tgt_train)
    best_source = -1.0
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    print("BiRP UDA", "mode", args.pseudo_mode, "device", device)
    print("source samples", len(src_train.dataset), "target samples", len(tgt_train.dataset))
    for epoch in range(1, args.num_epochs + 1):
        student.train()
        use_pseudo = args.warmup_epochs < epoch <= args.warmup_epochs + args.pseudo_epochs
        active_pseudo_mode = "autosamus" if (
            use_pseudo and args.foundation_first_epochs > 0 and epoch <= args.warmup_epochs + args.foundation_first_epochs
        ) else args.pseudo_mode
        running = {"loss": 0.0, "seg": 0.0, "pseudo": 0.0, "entropy": 0.0, "pseudo_cov": 0.0, "pseudo_fg": 0.0,
                   "reliability": 0.0, "agreement": 0.0, "fm_fg": 0.0, "routed_to_fm": 0.0}
        steps = 0

        for x_source, y_source, _ in src_train:
            try:
                x_target, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(tgt_train)
                x_target, _ = next(target_iter)
            x_source = x_source.to(device, non_blocking=True)
            y_source = y_source.to(device, non_blocking=True)
            x_target = x_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=bool(args.use_amp)):
                source_logits = student(ultrasound_style_augment(x_source, args.source_style_strength))
                loss_seg = F.cross_entropy(source_logits, y_source) + dice_loss_soft(source_logits, y_source)
                loss_pseudo = torch.zeros((), device=device)
                loss_entropy = torch.zeros((), device=device)
                diagnostics = {key: 0.0 for key in running if key not in {"loss", "seg", "pseudo", "entropy"}}
                if use_pseudo:
                    pseudo, mask, diagnostics = build_reliable_labels(
                        ema, autosamus, x_target, active_pseudo_mode, args.ema_conf_thr, args.autosamus_fg_thr,
                        args.autosamus_bg_thr, args.autosamus_cert_thr, args.reliability_thr, args.keep_ratio,
                        args.min_cov, args.agree_power, args.route_margin,
                    )
                    target_logits = student(x_target)
                    ce_map = F.cross_entropy(target_logits, pseudo, reduction="none")
                    loss_pseudo = (ce_map * mask).sum() / mask.sum().clamp_min(1.0)
                    loss_pseudo = loss_pseudo + masked_dice_loss(target_logits, pseudo, mask)
                if args.lambda_entropy > 0.0 and epoch > args.warmup_epochs:
                    target_prob = torch.softmax(student(x_target), dim=1).clamp_min(1e-6)
                    loss_entropy = -(target_prob * target_prob.log()).sum(dim=1).mean()
                loss = (
                    loss_seg
                    + (args.lambda_pseudo * loss_pseudo if use_pseudo else 0.0)
                    + args.lambda_entropy * loss_entropy
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema_update(ema, student, args.ema_momentum)
            steps += 1
            running["loss"] += float(loss.item())
            running["seg"] += float(loss_seg.item())
            running["pseudo"] += float(loss_pseudo.item())
            running["entropy"] += float(loss_entropy.item())
            for key in diagnostics:
                running[key] += float(diagnostics[key])
            if args.max_steps > 0 and steps >= args.max_steps:
                break

        if args.skip_eval:
            source_dice = -1.0
            target_dice = -1.0
        else:
            source_dice = evaluate_dice(student, src_val, device)
            target_dice = -1.0 if args.skip_target_eval else evaluate_dice(student, tgt_val, device)
        record = {"epoch": epoch, "stage": active_pseudo_mode if use_pseudo else "source_warmup",
                  "src_dice": source_dice, "tgt_dice": target_dice}
        record.update({key: value / max(1, steps) for key, value in running.items()})
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print("[E %d] %s src=%.4f tgt=%.4f loss=%.4f pseudo_cov=%.4f rel=%.4f agree=%.4f" % (
            epoch, record["stage"], source_dice, target_dice, record["loss"], record["pseudo_cov"],
            record["reliability"], record["agreement"],
        ))
        checkpoint = {"epoch": epoch, "model": student.state_dict(), "ema_model": ema.state_dict(),
                      "optimizer": optimizer.state_dict(), "args": vars(args), "metrics": record}
        torch.save(checkpoint, os.path.join(args.output_dir, "last.pt"))
        if source_dice > best_source:
            best_source = source_dice
            torch.save(checkpoint, os.path.join(args.output_dir, "best_source.pt"))
            print("saved source-selected checkpoint", "%.4f" % best_source)


if __name__ == "__main__":
    main()
