"""Legacy first-divergence trace and strict fresh/repeat/restart contracts."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from repro_utils import atomic_json, configure_strict, digest, file_hash, runtime_environment
from run_control import command, compare_complete, train, verify_inputs


def main(root):
    root = Path(root)
    out = root / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "passed.json").exists():
        verify_inputs(root)
        print("Previously passed frozen preflight", flush=True)
        return
    subprocess.run([sys.executable, "-m", "unittest", "-v", "test_repro"], check=True, cwd=root / "src")
    cfg = json.loads((root / "training_template.json").read_text())
    subprocess.run([sys.executable, str(root / "src/audit_history.py"),
        "--root", "/scratch/xf810/birp_arxiv_review_20261023", "--out", str(out / "historical_trajectory_audit.json")], check=True)
    traces = []
    for name, gamma in (("legacy_a", .5), ("legacy_a_repeat", .5), ("legacy_b", 1.5)):
        folder = out / name
        if not (folder / "trace.json").exists():
            if folder.exists():
                folder.rename(out / f"{name}_interrupted_{time.time_ns()}")
            subprocess.run([sys.executable, "-u", str(root / "src/trace_legacy.py"),
                "--root", str(root), "--template", str(root / "training_template.json"),
                "--out", str(folder), "--gamma", str(gamma)], check=True)
        traces.append(json.loads((folder / "trace.json").read_text()))
    legacy = {}
    for index, name in ((1, "same_configuration_repeat"), (2, "gamma_only_change_during_warmup")):
        a, b = traces[0], traces[index]
        checks = {key: a[key] == b[key] for key in ("init", "batches", "forwards", "steps", "metrics")}
        first_step = next((i for i, (x, y) in enumerate(zip(a["steps"], b["steps"])) if x != y), None)
        legacy[name] = dict(checks=checks, first_different_optimizer_step=first_step,
                            same_initialization=a["init"] == b["init"],
                            caution="Same newly allocated GPU; not proof of a unique cause for historical runs.")
    atomic_json(out / "legacy_repeat_diagnostic.json", legacy)
    configure_strict(1337)
    native_checks = {}
    for name in ("bilinear_backward", "native_ce"):
        try:
            if name == "bilinear_backward":
                x = torch.randn(1, 2, 4, 4, device="cuda", requires_grad=True)
                F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True).sum().backward()
            else:
                x = torch.randn(2, 2, 8, 8, device="cuda", requires_grad=True)
                y = torch.randint(0, 2, (2, 8, 8), device="cuda")
                F.cross_entropy(x, y).backward()
            native_checks[name] = dict(strict_supported=True)
        except RuntimeError as error:
            native_checks[name] = dict(strict_supported=False, error=str(error))
    atomic_json(out / "native_operator_checks.json", native_checks)
    if not (root / "provenance.json").exists():
        subprocess.run([sys.executable, "-u", str(root / "src/make_cache.py"),
                        "--root", str(root), "--template", str(root / "training_template.json")], check=True)
    cfg.update(num_epochs=2, warmup_epochs=1, pseudo_epochs=1, max_steps=2, seed=1337,
               pseudo_mode="bidir", agree_power=1.5)
    for name in ("strict_a", "strict_b"):
        folder = out / name
        train(root, cfg, folder)
    fresh = compare_complete(out / "strict_a", out / "strict_b")
    atomic_json(out / "strict_fresh_repeat.json", fresh)
    if not fresh["bitwise_logical_state_match"]:
        raise RuntimeError("Strict fresh repeat did not match exactly")
    warm = out / "strict_warmup"
    train(root, cfg, warm, stop=1)
    resumed = out / "strict_resumed"
    train(root, cfg, resumed, start=str(warm / "last.pt"))
    restart = compare_complete(out / "strict_a", resumed)
    atomic_json(out / "strict_restart_repeat.json", restart)
    if not restart["bitwise_logical_state_match"]:
        raise RuntimeError("Shared-warm-up restart did not match uninterrupted training")
    a = json.loads((out / "strict_a/trace_epoch01.json").read_text())
    # Gamma must not affect the warm-up trace under the repaired policy.
    gamma_cfg = dict(cfg, agree_power=3.)
    gamma_folder = out / "strict_other_gamma_warmup"
    train(root, gamma_cfg, gamma_folder, stop=1)
    b = json.loads((gamma_folder / "trace_epoch01.json").read_text())
    if a != b:
        raise RuntimeError("Gamma changed the controlled source-only warm-up")
    atomic_json(out / "passed.json", dict(passed=True, fresh_repeat=fresh,
        restart_repeat=restart, gamma_independent_warmup=True, native_checks=native_checks,
        provenance_sha256=file_hash(root / "provenance.json"), environment=runtime_environment(),
        scope="Short real-data strict repeat/restart. Full-length repeat is still required in seed1337 production."))
    print("PREFLIGHT_PASSED", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    main(p.parse_args().root)
