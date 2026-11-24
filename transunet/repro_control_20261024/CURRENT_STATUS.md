# Original BiRP reproduction repair: current handoff

## Latest inspection, 2026-10-25 (Asia/Shanghai)

- 18932432_0 was CANCELLED by UID 0 after 04:12:09. The log records a
  system cancellation, not a Python exception; no more specific cause has
  been established. Seed1337 warm-up and all three gamma branches completed;
  hard_prior started but did not complete. Full repetition remains untested.
- 18932432_1 completed successfully: seed2027 has all five variants,
  shared-start verification, locked evaluations and seed_completed.json.
- 18932432_2 is RUNNING: seed2028 warm-up and gamma0.5 completed;
  gamma1.5 most recently reached epoch 14/55.
- Rescue 18938406, array task 0 only, was submitted with the same controlled
  node/GPU/code/settings and dependency afterany:18932432. It will skip
  completed branches and restore incomplete full epoch state. Last PENDING
  Dependency; it does not duplicate the currently running seed2028 task.
- Summary 18931380 now waits afterok:18938406. The summarizer still requires
  all three seeds and the exact full-repeat contract; no aggregate result or
  manuscript replacement has been produced.

The entries below document the earlier queue recovery history, not live state.

Remote root: /scratch/xf810/birp_repro_control_20261024
Local root: BiRP_arXiv_revision_20261023/experiments/repro_control_20261024

## Results already verified

Read FINDINGS.md and evidence/legacy_repeat_diagnostic.json first.
Same old configuration/device/seed/input/first-forward diverges at its first
optimizer update. All ten real-GPU unit checks and strict fresh/restart/gamma
warm-up contracts now pass. Prior loading is complete (520 state entries),
and every cached target-adaptation prior batch repeats exactly.

## Active jobs

- 18931295: preflight completed, successful exit. Evidence and provenance
  downloaded locally. The CPU-only test pass earlier had one GPU skip;
  the subsequent actual GPU pass has all ten tests, no skips.
- 18931324: controlled array 0-2 was PREEMPTED by the cluster, not a
  training exception. Seed1337 warm-up and gamma0.5 completed; gamma1.5
  saved metrics through epoch 39. Other seeds did not finish warm-up.
- 18932398: first replacement was also PREEMPTED after 26 seconds; its
  remaining pending tasks were canceled before submitting the stable replacement.
- 18932432: replacement controlled array 0-2, concurrency one, V100 on dn001,
  account students / partition nvidia / qos nvidias (PreemptMode=OFF).
  Last checked PENDING (AssocGrpCpuLimit). Association membership and partition
  policy were checked before submission; scientific settings are unchanged.
  Existing completed branches are skipped and partial branches restore full
  saved epoch state; code, data, numerical policy and science settings unchanged.
  Per seed: shared ten-epoch warm-up then five fixed 45-epoch branches.
  Seed1337 additionally repeats main gamma1.5 completely from the same warm-up.
- 18931380: strict CPU summary, dependency updated to afterok:18932432
  after its original dependency became unsatisfiable. No numeric summary until
  all fifteen scientific fits and the full repeat validate.

The preempt queue is allowed for xf810; jobs may be reclaimed. The original
array did not remain queued automatically despite --requeue; inspect actual
Slurm state after preemption rather than assuming automatic continuation.
They restore atomically saved complete epoch state. Do not restart from
model weights alone or submit duplicates over existing checkpoints.
Check Slurm state/logs and passed.json before modifying any source code:
its hash is frozen and enforced by every restored checkpoint.

## Interpretation boundary

The repaired cohort changes numerical algorithms/sampling control and cannot
be pooled with the earlier nine gamma runs or substituted for historical
scores. It tests controlled reproducibility and independent mechanism value,
not a target-test-selected gamma. The original arXiv and current open twin
paper are unchanged. No paper upload/submission has occurred.

Original old-review results are still in ../original_review/ and
/scratch/xf810/birp_arxiv_review_20261023. New twin/evidence jobs are separate
and were not canceled or scientifically altered for this repair.
