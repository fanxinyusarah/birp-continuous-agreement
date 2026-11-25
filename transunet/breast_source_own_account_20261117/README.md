# BrEaST-as-source orchestration, run under xl5874's own Jubail account

Same controlled TransUNet trainer (`src/train_repro.py`, `legacy/`) and the
same continuous-agreement wrapper convention (`src/wrapper.py`, reading
`BIRP_EXTERNAL_ROOT`) as the other `transunet/` directories, adapted to run
under a different account against ACL-shared target-split data rather than
the primary `patient_protocol_v1` protocol. This is the code behind the
BrEaST-source results in `results/available_run_metrics.json` for the BUSI,
BUS-UCLM and OASBUD targets reported alongside the U-Net BrEaST-source table.

## What changed versus the other `transunet/` directories

- `src/make_cache.py` and `src/evaluate_selected.py`: only the provenance
  file paths and the held-out test directory per target domain were changed
  (`nonanchor_protocol_v1/{target}_target_test` for BUSI/UCLM;
  `external_grouped_v1/oasbud_test` for OASBUD, which lives under a
  differently-named protocol directory). The prior-loading, caching and
  evaluation logic is unchanged.
- `src/run_direction.py`: orchestrates `source_only` / `bidir` (original
  BiRP) / `ema_all` (all-pixel EMA) / `cont_full` (continuous agreement, via
  the `wrapper.py` monkeypatch) for one (target, seed) pair. `source_only`
  trains from scratch for the full epoch budget rather than resuming the
  shared warm-up with a changed `pseudo_epochs`, because `train_repro.py`'s
  own resume-consistency guard correctly refuses a scientific-setting change
  on resume -- this is intentional, not a workaround of that guard.
- `configs/{busi,uclm,oasbud}_training_template.json`: per-target
  `source_root`/`target_root`/`tgt_splits`/`autosamus_ckpt` paths. OASBUD's
  differ in protocol directory (`external_grouped_v1` vs
  `nonanchor_protocol_v1`) since its target split was shared separately.

## Running

`smoke.sbatch` runs a 2-epoch/4-variant smoke test for all three targets
before any real training; `train.sbatch <target> <seed>` runs the full
55-epoch protocol for one (target, seed) pair, internally covering all four
variants; `oasbud_submit.sh` submits the three OASBUD seeds. Replace
`/scratch/xl5874/...` paths with your own account's equivalents; the
AutoSAMUS checkpoint and target-split paths under `/scratch/xf810/...` are
read-only ACL-shared locations on the original account and are unchanged
from the other `transunet/` directories.
