# Observed reproduction findings, 2026-10-24

## Confirmed old-path failure

On the same allocated Tesla V100-SXM2-32GB, seed 1337, gamma 0.5 repeated in
fresh subprocesses, the recorded student initialization, post-initialization
RNG, source/target image and label tensors, sample IDs and first source forward
are EXACTLY identical. The model hash first differs after optimizer step 1;
the second and later source forwards subsequently differ. A gamma-only change
in source-only warm-up has the same pattern. This directly demonstrates
same-hardware nondeterminism in the old training-update path. It is not just
an inference discrepancy or a changed gamma/data order.

This does not prove that every historical metric difference has one cause.
Old missing RNG/scaler/source provenance prevents a retroactive exact replay.

## Operator checks on the actual environment

- Native spatial mean cross-entropy throws under strict deterministic mode:
  nll_loss2d_forward_out_cuda_template has no deterministic implementation.
- The tested native FP32 bilinear-backward example is ACCEPTED under strict
  mode on this installed build. Do not claim native bilinear is universally
  unsupported, or that this alone proves the unique historical culprit.
- The repaired trainer uses the validated original-forward/separable-adjoint
  resize uniformly and log-softmax/gather/mean CE. It defines an explicitly
  repaired numerical cohort, not bitwise old-trajectory equivalence.

## Passed checks

- All 10 unit checks passed on the real V100, including analytical gradient
  checks, exact resize forward, loss/gradient equivalence, sampler isolation
  and bitwise repeated custom backward under strict mode.
- All 520 prior state entries match the supplied checkpoint; no missing or
  mismatched prior tensors were retained by the builder fallback.
- All 1,312 adaptation images have frozen AMP prior outputs; two evaluations
  of every batch matched bitwise. No target labels were used for caching.
- Two independent two-epoch real-data strict runs match model, EMA, optimizer,
  GradScaler, RNG, history, global step, selected source epoch and selected EMA.
- Shared-warm-up checkpoint continuation matches uninterrupted training in
  those same fields exactly.
- Changing gamma leaves the repaired warm-up trace unchanged.

These are smoke/short-trajectory correctness results, not evidence of target
Dice improvement. Full-length seed1337 gamma1.5 repetition remains required.

## Current production

Preflight 18931295 completed successfully. Controlled array 18931324 is now
running seed1337 (task 0), with seeds2027/2028 pending and concurrency one.
Summary 18931380 waits for all seeds, strong controls and the full repeat.
Both preflight and production were moved, before any GPU output, from the
then-busy dn005 request to dn001; actual device/environment are recorded.

The production compares gamma0.5, main gamma1.5, gamma3, hard prior and binary
reliability from identical per-seed warm-up states. It cannot guarantee a
BiRP advantage. No old scores, source files or manuscripts were overwritten.

Evidence is in evidence/ and the remote preflight/ directory.
Frozen provenance SHA256:
f970582254b25fe8d3e10b976c1b909187dd987a8afaf8cb474ceb2248200ac4
