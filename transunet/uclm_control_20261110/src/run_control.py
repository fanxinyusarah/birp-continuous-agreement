"""One allocated GPU per seed: shared warm-up, fixed variants and exact repeat."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from repro_utils import atomic_json, configure_strict, digest, file_hash, runtime_environment

VARIANTS = (("bidir", "bidir", 1.5), ("ema_filtered", "ema", 1.5),
            ("ema_all", "ema_all", 1.5), ("hard_prior", "hard_prior", 1.5),
            ("soft_floor", "soft_floor", 1.5), ("soft_agreement", "soft_agreement", 1.5))


def command(root, cfg, folder, start="", stop=0, resume=""):
    options = dict(cfg, output_dir=str(folder), start_from=start, resume=resume,
                   stop_after_epoch=stop, provenance=str(root / "provenance.json"),
                   prior_cache=str(root / "prior_cache.pt"), skip_target_eval=True)
    argv = [sys.executable, "-u", str(root / "src/train_repro.py")]
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                argv.append("--" + key)
        else:
            argv.extend(["--" + key, str(value)])
    return argv


def train(root, cfg, folder, start="", stop=0):
    complete = folder / "completed.json"
    expected = stop or cfg["num_epochs"]
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved["epoch"] != expected or saved["provenance_sha256"] != file_hash(root / "provenance.json"):
            raise ValueError("Existing output has wrong completion/provenance")
        for key in ("seed", "pseudo_mode", "agree_power", "num_epochs", "warmup_epochs", "max_steps"):
            if saved["args"][key] != cfg[key]:
                raise ValueError("Existing output has wrong settings: " + key)
        return
    last = folder / "last.pt"
    resume = str(last) if last.exists() else ""
    subprocess.run(command(root, cfg, folder, start="" if resume else start,
                           stop=stop, resume=resume), check=True)


def compare_complete(a, b):
    sa = torch.load(a / "last.pt", map_location="cpu")
    sb = torch.load(b / "last.pt", map_location="cpu")
    fields = ("model", "ema_model", "optimizer", "scaler", "rng", "history", "global_step")
    checks = {key: digest(sa[key]) == digest(sb[key]) for key in fields}
    del sa, sb
    ca = json.loads((a / "completed.json").read_text())
    cb = json.loads((b / "completed.json").read_text())
    wa = torch.load(ca["best_checkpoint"], map_location="cpu")
    wb = torch.load(cb["best_checkpoint"], map_location="cpu")
    checks["selected_epoch"] = wa["epoch"] == wb["epoch"]
    checks["selected_ema"] = digest(wa["ema_model"]) == digest(wb["ema_model"])
    del wa, wb
    return dict(bitwise_logical_state_match=all(checks.values()), checks=checks,
                note="Compares logical tensors/RNG/history, not nondeterministic archive byte layout.")


def verify_inputs(root):
    provenance = json.loads((root / "provenance.json").read_text())
    for path, expected in provenance["files_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError("Raw input changed: " + path)
    for path, expected in provenance["code_sha256"].items():
        if file_hash(root / path) != expected:
            raise ValueError("Code changed: " + path)
    if file_hash(root / "prior_cache.pt") != provenance["prior_cache_sha256"]:
        raise ValueError("Prior cache changed")
    return provenance


def main(root, seed):
    root = Path(root)
    passed = json.loads((root / "preflight/passed.json").read_text())
    if not passed["passed"] or passed["provenance_sha256"] != file_hash(root / "provenance.json"):
        raise ValueError("Preflight not valid for current code/data")
    verify_inputs(root)
    configure_strict(seed)
    environment = runtime_environment()
    for key in ("gpu", "total_memory", "torch", "cuda", "cudnn"):
        if environment[key] != passed["environment"][key]:
            raise ValueError("Different controlled hardware/software: " + key)
    cfg = json.loads((root / "training_template.json").read_text())
    cfg.update(seed=seed, pseudo_mode="bidir", agree_power=1.5)
    folder = root / "runs" / f"seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    warmup = folder / "warmup"
    train(root, cfg, warmup, stop=cfg["warmup_epochs"])
    start = str(warmup / "last.pt")
    variants = list(VARIANTS)
    if seed == 1337:
        variants.append(("bidir_repeat", "bidir", 1.5))
    initials = {}
    for name, mode, gamma in variants:
        variant = dict(cfg, pseudo_mode=mode, agree_power=gamma)
        location = folder / name
        train(root, variant, location, start=start)
        initial = json.loads((location / "initial_state.json").read_text())
        initials[name] = {k: initial[k] for k in ("model", "ema", "optimizer", "scaler", "rng", "global_step", "epoch")}
    if len({json.dumps(v, sort_keys=True) for v in initials.values()}) != 1:
        raise ValueError("Variants did not start from identical warm-up logical states")
    atomic_json(folder / "shared_start_verified.json", dict(passed=True, variants=initials,
        provenance_sha256=file_hash(root / "provenance.json")))
    if seed == 1337:
        repeat = compare_complete(folder / "bidir", folder / "bidir_repeat")
        atomic_json(folder / "full_repeat.json", repeat)
        if not repeat["bitwise_logical_state_match"]:
            raise ValueError("Full gamma1.5 repeat failed; do not interpret mechanism comparisons")
    for name, _mode, _gamma in variants:
        subprocess.run([sys.executable, "-u", str(root / "src/evaluate_selected.py"),
                        "--root", str(root), "--folder", str(folder / name)], check=True)
    if seed == 1337:
        a = json.loads((folder / "bidir/locked_evaluation/per_case.json").read_text())
        b = json.loads((folder / "bidir_repeat/locked_evaluation/per_case.json").read_text())
        if a != b:
            raise ValueError("Same logical model produced different locked-test results")
        repeat["locked_per_case_exact_match"] = True
        atomic_json(folder / "full_repeat.json", repeat)
    atomic_json(folder / "seed_completed.json", dict(passed=True, seed=seed, variants=[v[0] for v in variants],
        provenance_sha256=file_hash(root / "provenance.json"), environment=environment))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, choices=(1337, 2027, 2028), required=True)
    a = p.parse_args()
    main(a.root, a.seed)
