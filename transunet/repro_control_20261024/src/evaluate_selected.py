"""Evaluate the source-selected EMA only after the seed's fixed fits finish."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from boundary_metrics import segmentation_metrics
from repro_utils import atomic_json, configure_strict, file_hash, runtime_environment


def main(root, folder):
    root, folder = Path(root), Path(folder)
    done = json.loads((folder / "completed.json").read_text())
    output = folder / "locked_evaluation"
    if (output / "summary.json").exists():
        saved = json.loads((output / "summary.json").read_text())
        if saved["provenance_sha256"] != file_hash(root / "provenance.json"):
            raise ValueError("Incompatible evaluation provenance")
        return
    if done["epoch"] != 55:
        raise ValueError("Do not evaluate an incomplete fit")
    sys.path.insert(0, str(root / "legacy"))
    from disentanglement_fda_1 import SegListDataset, build_transunet
    configure_strict(done["args"]["seed"])
    checkpoint = torch.load(done["best_checkpoint"], map_location="cpu")
    model = build_transunet("R50-ViT-B_16", 224, 2).cuda().eval()
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    epoch = checkpoint["epoch"]
    del checkpoint
    ds = SegListDataset("/scratch/xf810/SegMamba/data/patient_protocol_v1/busbra_test", "val", 224, True, True)
    if len(ds) != 376:
        raise ValueError("Historical locked test changed")
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2,
                        generator=torch.Generator().manual_seed(99173))
    records = []
    with torch.inference_mode():
        for images, truth, names in loader:
            probability = torch.softmax(model(images.cuda()), 1)[:, 1]
            pred, target = probability >= .5, truth.cuda() > 0
            inter = (pred & target).float().sum((1, 2))
            den = pred.float().sum((1, 2)) + target.float().sum((1, 2))
            dice = ((2 * inter + 1e-5) / (den + 1e-5)).cpu().numpy()
            for name, p, t, score in zip(names, pred.cpu().numpy(), target.cpu().numpy(), dice):
                records.append(dict(case=name, dice=float(score), **segmentation_metrics(p, t)))
    if len({r["case"] for r in records}) != 376:
        raise ValueError("Duplicate/missing test IDs")
    test_hashes = {name: dict(image=file_hash(ds._resolve_img_path(name)),
                            label=file_hash(ds._resolve_lab_path(name))) for name in ds.names}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "per_case.json", records)
    atomic_json(output / "summary.json", dict(selected_epoch=epoch, n_cases=len(records),
        checkpoint=done["best_checkpoint"], checkpoint_sha256=file_hash(done["best_checkpoint"]),
        provenance_sha256=file_hash(root / "provenance.json"), environment=runtime_environment(),
        test_files_sha256=test_hashes,
        mean={k: float(np.mean([r[k] for r in records])) for k in ("dice", "iou", "hd95_px", "assd_px")},
        selection="student source-validation Dice; evaluate its EMA weights, no target-based selection",
        cohort="repaired strict shared-warm-up, separate from historical/gamma cohorts"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--folder", required=True)
    a = p.parse_args()
    main(a.root, a.folder)
