"""Immutable AMP prior outputs and code/data provenance, without target labels."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from repro_utils import atomic_checkpoint, atomic_json, digest, file_hash, runtime_environment


def main(root, template):
    root = Path(root)
    sys.path.insert(0, str(root / "legacy"))
    from disentanglement_fda_1 import SegListDataset, TargetSplitDataset, seed_all
    from train_birp_hard_prior import FrozenAutoSAMUS
    cfg = json.loads(Path(template).read_text())
    cache_path = root / "prior_cache.pt"
    if (root / "provenance.json").exists():
        raise FileExistsError("Refusing to replace frozen inputs")
    seed_all(1337)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    dataset = TargetSplitDataset(cfg["tgt_splits"], cfg["busbra_root"], cfg["img_size"])
    # Keep the original batch size/AMP policy. Do not cache an augmented view.
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"], pin_memory=True)
    prior = FrozenAutoSAMUS(cfg["autosamus_root"], cfg["autosamus_ckpt"], torch.device("cuda:0"))
    raw_weights = torch.load(cfg["autosamus_ckpt"], map_location="cpu")
    raw_weights = {k.removeprefix("module."): v for k, v in raw_weights.items()}
    loaded = prior.model.state_dict()
    missing, mismatched = [], []
    for key, value in loaded.items():
        if key not in raw_weights:
            missing.append(key)
        elif not torch.is_tensor(raw_weights[key]) or not torch.equal(value.cpu(), raw_weights[key]):
            mismatched.append(key)
    loading_audit = dict(state_keys=len(loaded), missing_keys=missing, mismatched_values=mismatched,
        passed=not missing and not mismatched,
        note="The original loader has a fallback that can retain random initialization; require complete actual weight correspondence.")
    atomic_json(root / "prior_loading_audit.json", loading_audit)
    if not loading_audit["passed"]:
        raise RuntimeError("Prior checkpoint not completely loaded; inspect prior_loading_audit.json before training")
    del raw_weights, loaded
    predictions, keys, image_hashes = [], [], []
    with torch.inference_mode():
        for batch, names in loader:
            images = batch.cuda()
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["use_amp"]):
                a, b = prior(images)[:, 0], prior(images)[:, 0]
            if not torch.equal(a, b) or not torch.isfinite(a).all():
                raise RuntimeError("Frozen prior forward is not bitwise repeatable/finite on this device")
            predictions.append(a.cpu())
            image_hashes.extend(digest(x) for x in batch)
            keys.extend(names)
            if len(keys) % 100 == 0:
                print("PRIOR_CACHE", len(keys), "/", len(dataset), flush=True)
    if keys != dataset.keys:
        raise ValueError("Incomplete/reordered cache")
    cache = dict(keys=keys, prob=torch.cat(predictions),
        input_tensor_sha256=image_hashes, batch_size=cfg["batch_size"], use_amp=cfg["use_amp"],
        repeated_forward_bitwise_equal=True, environment=runtime_environment(),
        ground_truth_used=False, target_images_augmented=False)
    if cache_path.exists():
        previous = torch.load(cache_path, map_location="cpu")
        if previous["keys"] != keys or not torch.equal(previous["prob"], cache["prob"]):
            raise RuntimeError("Interrupted cache does not match the fresh repeated forward")
    else:
        atomic_checkpoint(cache_path, cache)
    del prior, predictions
    source_train = SegListDataset(cfg["source_root"], "train", cfg["img_size"], True, True)
    source_val = SegListDataset(cfg["source_root"], "val", cfg["img_size"], True, True)
    paths = {Path(cfg["tgt_splits"]), Path(cfg["autosamus_ckpt"]), Path(template),
        Path("/scratch/xf810/SegMamba/data/patient_protocol_v1/busbra_test/val_list.txt"),
        Path(cfg["busbra_root"]) / "bus_data.csv"}
    source_ids = {}
    for label, ds in (("source_train", source_train), ("source_validation", source_val)):
        source_ids[label] = ds.names
        for name in ds.names:
            paths.add(Path(ds._resolve_img_path(name)))
            paths.add(Path(ds._resolve_lab_path(name)))
    if set(source_train.names) & set(source_val.names):
        raise ValueError("Source train/validation ID overlap")
    for key in dataset.keys:
        relative = key.split("::", 1)[-1]
        paths.add(Path(cfg["busbra_root"]) / relative)
    code = {str(p.relative_to(root)): file_hash(p)
            for folder in (root / "src", root / "legacy") for p in sorted(folder.rglob("*.py"))}
    prior_root = Path(cfg["autosamus_root"])
    prior_code = {str(p.relative_to(prior_root)): file_hash(p)
                  for p in sorted((prior_root / "models").rglob("*.py"))}
    provenance = dict(implementation="repro_control_v1", code_sha256=code,
        files_sha256={str(p): file_hash(p) for p in sorted(paths)},
        prior_code_sha256=prior_code, prior_repo=str(prior_root),
        source_ids=source_ids, target_adaptation_ids=keys,
        prior_cache_sha256=file_hash(cache_path), prior_environment=runtime_environment(),
        target_labels_used=False, cache_repeat_exact=True,
        complete_prior_loading=loading_audit,
        note="Separate repaired execution cohort, not a replacement/pooled version of historical scores.")
    atomic_json(root / "provenance.json", provenance)
    print("FROZEN_PROVENANCE", file_hash(root / "provenance.json"), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--template", required=True)
    a = p.parse_args()
    main(a.root, a.template)
