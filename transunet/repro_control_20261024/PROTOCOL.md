# Original BiRP controlled reproduction repair, 2026-10-24

This is a separate repaired execution cohort. It does not overwrite or pool
the historical WACV, later audit, or nine recent gamma results. The user
prioritized resolving the warm-up/reproduction discrepancy; the currently open
twin-DINOv2 LaTeX document is unrelated and is not edited by this experiment.

## Established facts versus hypotheses

- The nine recent gamma runs diverge before gamma is active, so their results
  are not a clean gamma-only causal experiment.
- Recorded hardware differs across some runs. That is not proof of the sole
  historical cause. The old checkpoints omit scaler and RNG states and do not
  contain a full original-source/data fingerprint.
- The legacy helper sets only cuDNN benchmark/deterministic options. PyTorch
  2.1 documents CUDA bilinear interpolation backward and some NLL reductions
  as nondeterministic. Actual native-operator checks and same-GPU repeat traces
  are saved by preflight; do not infer an observed root cause from docs alone.
- Unshuffled DataLoader iteration can still consume the model's global CPU
  random stream unless a separate generator is supplied. This is a code
  coupling, not yet a proven explanation for the historical trajectories.

Primary implementation references:
https://docs.pytorch.org/docs/2.1/generated/torch.use_deterministic_algorithms.html
https://raw.githubusercontent.com/pytorch/pytorch/v2.1.0/docs/source/notes/randomness.rst

## Diagnostic gate before production

1. Unit/gradient checks, including exact original bilinear forward, analytical
   backward against native CPU gradients/gradcheck, equal CE values/gradients,
   isolated sampling, and repeated GPU backward under strict algorithms.
2. Same newly allocated GPU, fresh subprocesses: unchanged legacy numerical
   path with gamma 0.5 twice and gamma 1.5 once, four source-only steps each.
   Hash initialization, RNG, actual image/label tensors, IDs, logits and
   post-optimizer weights. Preserve the first divergence, including a negative
   result. Logical checkpoint fingerprints replace diagnostic weight dumps.
3. Freeze source train/validation, target adaptation images, ordered IDs,
   helper/TransUNet code, prior source/weights, and test-ID/group metadata.
4. Freeze AutoSAMUS AMP probabilities at the original batch size 2. Every
   batch is evaluated twice and must match bitwise and be finite. No target
   label is loaded. Target images have no augmentation in this legacy recipe,
   so cached eval-mode predictions are the same fixed input/model operation,
   not a new learned prior. Preserve dtype, input hashes and device metadata.
   Verify every loaded prior state entry against the supplied checkpoint:
   the original builder's fallback can otherwise retain random missing weights.
   Missing/mismatched state blocks training rather than silently freezing it.
5. Strict real-data smoke: two independent two-epoch runs, warm-up-only restart
   versus uninterrupted training, and a gamma-only change during warm-up.
   Models, EMA, optimizer, scaler, RNG and source metrics must match exactly.
   An error blocks dependent production; no warn-only or tolerance relaxation.

## Repaired numerical/data execution policy

- Same original TransUNet, 224 input, batch 2, AdamW/loss/schedule, source-only
  checkpoint selection and final reporting of the selected EMA weights.
- Seed Python, NumPy, torch CPU/CUDA. Set PYTHONHASHSEED and CUBLAS workspace
  before process start. Disable TF32 and cuDNN benchmarking; enforce
  torch.use_deterministic_algorithms(True, warn_only=False).
- Keep original bilinear forward, shapes and align_corners. Replace only its
  atomic-scatter gradient with the mathematically equivalent separable linear
  adjoint via deterministic matrix multiplication. CPU gradient/gradcheck and
  GPU repeat tests are mandatory. Checkpoint parameter keys are unchanged.
- Compute pixel CE by log_softmax/gather and an explicit mean, avoiding the
  legacy spatial NLL mean reduction. It is the same mathematical objective,
  not a promise of bitwise equivalence to old floating-point reductions.
- Source permutations and a continuous target permutation/cursor are explicit
  functions of seed, stream and epoch. Validation/worker generators do not
  consume the model RNG. No stochastic reader augmentation is introduced.
- Save epoch, models, EMA, optimizer, GradScaler, Python/NumPy/CPU/CUDA states,
  history, best source score/checkpoint and input/code/environment fingerprints
  atomically. Resume only compatible numerical environments/settings. Requeue
  resumes the last complete epoch, not from weights alone.

These numerical/sampling changes deliberately define a NEW reproducible
execution cohort; they cannot retroactively recreate missing historical RNG
or source versions. Do not replace the old score by a nicer new result.

## Fixed production matrix, after successful gate

Seeds: 1337, 2027, 2028. On one allocated V100 per seed:

- Train exactly one 10-epoch warm-up with the unchanged 55-epoch configuration.
- Each of five branches restores that exact student, EMA, optimizer, scaler,
  RNG and source-selection history, then completes epochs 11-55:
  gamma 0.5, gamma 1.5 (main), gamma 3; direct hard-prior labels; binary mask.
- For seed 1337, repeat the entire gamma-1.5 adaptation from the shared warm-up.
  Require exact final logical states, source history, selected EMA and all
  locked-test per-case metrics, not just similar average Dice.

This is three shared warm-ups plus sixteen adaptation branches (including
the full repeat), scheduled as three seed jobs with concurrency one. They
run on a fixed V100 node to reduce machine variation. The account's allowed
preempt/preempt-gpu resources are used; jobs are requeueable and restartable.
Other existing jobs are not canceled or their scientific settings changed.

Source training/validation and locked BUS-BRA remain the historical protocol:
376 test images and 213 Case clusters. Target labels do not enter training or
checkpoint selection. Evaluation runs only after all fixed variants of a
seed finish. This test set has been inspected historically and is not fresh
confirmatory data. All values, including negative results, are reported.

## Reporting and interpretation

Strict summary requires all fifteen scientific fits, all three verified
shared starts, identical test data/IDs, the complete repeat, valid provenance
and finite metrics. Three-seed sample SD, patient-cluster 20,000-draw CIs and
sign randomization; Holm across the two prespecified Dice comparisons of main
gamma 1.5 versus hard-prior and binary mask. Gamma values are sensitivity
results, never test-selected as optimal. Historical scores stay separate.

Successful reproduction repairs experimental control. It does not guarantee
that BiRP beats same-prior baselines, proves novelty, resolves prior-pretraining
exposure, or makes the arXiv manuscript submission-ready automatically.
