"""All seeds and same warm-up required; report every gamma and matched control."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from repro_utils import atomic_json, file_hash
from run_control import VARIANTS


def cluster_stats(delta, groups):
    unique = sorted(set(groups))
    sums = np.array([delta[groups == g].sum() for g in unique])
    counts = np.array([(groups == g).sum() for g in unique])
    rng = np.random.default_rng(20261024)
    draws = rng.integers(0, len(unique), size=(20000, len(unique)))
    bootstrap = sums[draws].sum(1) / counts[draws].sum(1)
    signs = rng.choice([-1., 1.], size=(20000, len(unique)))
    null = (signs * sums).sum(1) / counts.sum()
    return dict(mean_difference=float(delta.mean()),
        cluster_bootstrap_95ci=np.quantile(bootstrap, [.025, .975]).tolist(),
        two_sided_sign_p=float((1 + (np.abs(null) >= abs(delta.mean())).sum()) / 20001),
        images=len(delta), patient_cases=len(unique), draws=20000)


def main(root):
    root = Path(root)
    provenance = file_hash(root / "provenance.json")
    repeat = json.loads((root / "runs/seed1337/full_repeat.json").read_text())
    if not repeat["bitwise_logical_state_match"]:
        raise ValueError("Full-length repeat failed")
    ids, test_hash, arrays, entries = None, None, {}, {}
    for seed in (1337, 2027, 2028):
        folder = root / "runs" / f"seed{seed}"
        done = json.loads((folder / "seed_completed.json").read_text())
        shared = json.loads((folder / "shared_start_verified.json").read_text())
        if not done["passed"] or not shared["passed"] or done["provenance_sha256"] != provenance:
            raise ValueError("Incomplete/mismatched seed")
        initials = list(shared["variants"].values())
        if len({json.dumps(v, sort_keys=True) for v in initials}) != 1:
            raise ValueError("Not a shared warm-up")
        for name, mode, gamma in VARIANTS:
            location = folder / name
            complete = json.loads((location / "completed.json").read_text())
            args = complete["args"]
            if (complete["epoch"], args["seed"], args["pseudo_mode"], args["agree_power"]) != (55, seed, mode, gamma):
                raise ValueError("Configuration mismatch")
            result = json.loads((location / "locked_evaluation/summary.json").read_text())
            rows = json.loads((location / "locked_evaluation/per_case.json").read_text())
            case_ids = [r["case"] for r in rows]
            if ids is None:
                ids, test_hash = case_ids, result["test_files_sha256"]
            if (case_ids != ids or result["test_files_sha256"] != test_hash or
                    result["provenance_sha256"] != provenance or len(case_ids) != 376):
                raise ValueError("Test ID/data/provenance mismatch")
            for metric in ("dice", "iou", "hd95_px", "assd_px"):
                values = np.array([r[metric] for r in rows])
                if not np.isfinite(values).all() or abs(values.mean() - result["mean"][metric]) > 1e-10:
                    raise ValueError("Nonfinite or inconsistent metric")
                arrays.setdefault(name, {}).setdefault(metric, []).append(values)
    with open("/scratch/xf810/BUS-BRA/BUSBRA/bus_data.csv", newline="") as handle:
        mapping = {r["ID"]: r["Case"] for r in csv.DictReader(handle)}
    groups = np.array([mapping[Path(name).stem] for name in ids])
    if len(set(groups)) != 213:
        raise ValueError("Patient grouping changed")
    for name, metrics in arrays.items():
        entries[name] = {}
        for metric, values in metrics.items():
            means = np.array(values).mean(1)
            entries[name][metric] = dict(seed_means=means.tolist(), mean=float(means.mean()),
                                         sample_sd=float(means.std(ddof=1)))
    contrasts = {name: cluster_stats(np.mean(arrays["gamma1.5"]["dice"], 0) -
                                    np.mean(arrays[name]["dice"], 0), groups)
                 for name in ("hard_prior", "binary_reliability")}
    ordered = sorted(contrasts, key=lambda k: contrasts[k]["two_sided_sign_p"])
    maximum = 0.
    for rank, name in enumerate(ordered):
        maximum = max(maximum, (len(ordered) - rank) * contrasts[name]["two_sided_sign_p"])
        contrasts[name]["holm_p_two_prespecified_dice_contrasts"] = min(1., maximum)
    report = dict(methods=entries, paired=contrasts, full_repeat=repeat,
        provenance_sha256=provenance, all_15_fits_present=True,
        note="Repaired strict execution cohort; never pool with or overwrite historical/gamma results. No test-based gamma choice.")
    atomic_json(root / "CONTROLLED_RESULTS.json", report)
    lines = ["# Original BiRP: controlled reproduction cohort", "",
        "All variants share the same per-seed warm-up, optimizer, scaler, RNG and data stream.",
        "Separate from historical results; no test-selected gamma. Boundary units: 224-grid pixels.", "",
        "| Variant | Dice | IoU | HD95 | ASSD |", "|---|---:|---:|---:|---:|"]
    for name, _mode, _gamma in VARIANTS:
        values = [f"{entries[name][m]['mean']:.4f} +/- {entries[name][m]['sample_sd']:.4f}"
                  for m in ("dice", "iou", "hd95_px", "assd_px")]
        lines.append("| " + " | ".join([name, *values]) + " |")
    (root / "CONTROLLED_RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
