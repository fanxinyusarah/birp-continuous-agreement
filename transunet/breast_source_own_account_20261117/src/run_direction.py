"""Orchestrate source-only / bidir(original BiRP) / ema_all(all-pixel EMA) / cont_full
(continuous-consistency) for one seed, reusing run_control.train()/command() as library
calls (bypassing run_control.main()'s historical preflight-gate requirement, which does
not apply to this brand-new BrEaST-source direction -- there is no prior historical run
to audit against). Mirrors method/pilot/runner.py's wrapper-monkeypatch pattern for the
continuous rule.

Usage: python run_direction.py --root <busi_or_uclm_root> --seed 1337 [--smoke]
"""
import argparse
import os
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "")  # root/src is already on PYTHONPATH by the caller
import run_control  # noqa: E402
from repro_utils import file_hash  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true", help="tiny 2-epoch/max_steps=2 run for pipeline validation")
    args = p.parse_args()
    root = Path(args.root)

    if not (root / "provenance.json").exists():
        raise FileExistsError("Run make_cache.py first to produce provenance.json/prior_cache.pt")

    cfg = json.loads((root / "training_template.json").read_text())
    cfg = dict(cfg, seed=args.seed, provenance=str(root / "provenance.json"),
               prior_cache=str(root / "prior_cache.pt"))
    tag = "smoke" if args.smoke else "runs"
    if args.smoke:
        cfg.update(num_epochs=2, warmup_epochs=1, pseudo_epochs=1, max_steps=2)
    folder_root = root / tag / f"seed{args.seed}"
    folder_root.mkdir(parents=True, exist_ok=True)

    warmup = folder_root / "warmup"
    run_control.train(root, cfg, warmup, stop=cfg["warmup_epochs"])
    start = str(warmup / "last.pt")

    # source_only does NOT resume the shared warmup: it needs pseudo_epochs=0 for the
    # FULL num_epochs budget (fair compute vs. the adapted branches), but train_repro.py's
    # resume guard refuses to change pseudo_epochs on a resumed run (by design -- see
    # PROTOCOL.md "Resume only compatible numerical environments/settings"). So source_only
    # is its own from-scratch run, not a branch of the shared warmup; it is intentionally
    # excluded from the shared-warmup-state check below.
    source_only_cfg = dict(cfg, pseudo_mode="ema", pseudo_epochs=0)
    run_control.train(root, source_only_cfg, folder_root / "source_only")

    variants = {
        "bidir": dict(cfg, pseudo_mode="bidir", agree_power=1.5),
        "ema_all": dict(cfg, pseudo_mode="ema_all", agree_power=1.5),
    }
    initials = {}
    for name, variant_cfg in variants.items():
        location = folder_root / name
        run_control.train(root, variant_cfg, location, start=start)
        initial = json.loads((location / "initial_state.json").read_text())
        initials[name] = {k: initial[k] for k in ("model", "ema", "optimizer", "scaler", "rng", "global_step", "epoch")}

    # cont_full: external wrapper monkeypatches train_repro.build_reliable_labels with
    # w = 0.5 + 0.5*c*a (c=2|q-0.5| clamp01, a=1-|p-q| clamp01) -- same formula as the
    # Nov-14 preprint's "Continuous agreement" rule and the repo's
    # ablations/external_soft_agreement_20261109/wrapper.py.
    cont_folder = folder_root / "cont_full"
    if not (cont_folder / "completed.json").exists():
        last = cont_folder / "last.pt"
        wrapper_cfg = dict(cfg, pseudo_mode="bidir", agree_power=1.5)  # pseudo_mode ignored by the monkeypatch
        argv = run_control.command(root, wrapper_cfg, cont_folder,
                                    start="" if last.exists() else start,
                                    resume=str(last) if last.exists() else "")
        argv[2] = str(root / "src" / "wrapper.py")
        env = dict(os.environ, BIRP_EXTERNAL_ROOT=str(root))
        subprocess.run(argv, check=True, env=env)
    initial = json.loads((cont_folder / "initial_state.json").read_text())
    initials["cont_full"] = {k: initial[k] for k in ("model", "ema", "optimizer", "scaler", "rng", "global_step", "epoch")}

    shared = len({json.dumps(v, sort_keys=True) for v in initials.values()}) == 1
    print("SHARED_WARMUP_STATE_MATCH", shared, flush=True)
    if not shared and not args.smoke:
        raise ValueError("Variants did not start from identical warm-up logical states")

    if not args.smoke:
        for name in variants.keys() | {"cont_full", "source_only"}:
            subprocess.run([sys.executable, "-u", str(root / "src" / "evaluate_selected.py"),
                            "--root", str(root), "--folder", str(folder_root / name)], check=True)
    print("DIRECTION_SEED_DONE", root.name, args.seed, tag, flush=True)


if __name__ == "__main__":
    main()
