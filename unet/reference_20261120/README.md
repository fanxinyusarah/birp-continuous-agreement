# U-Net/ResNet-50 pipeline, final revision (supersedes `reference_20261021`)

This replaces the Oct 21 snapshot for anything using the final continuous
weighting modes. `reference_20261021` predates `cont_full`/`cont_cert`/
`cont_agree` and the BrEaST source domain; kept for history, not for
reproducing the manuscript's later numbers.

## What's new versus `reference_20261021`

- `train.py`: added `--mode cont_full` (w=0.5+0.5·c·a, the paper's
  "continuous agreement"), `cont_cert` (w=0.5+0.5·c, certainty-only
  ablation) and `cont_agree` (w=0.5+0.5·a, agreement-only ablation) --
  the same c/a formula as `birp_rule`/`birp_agree`, just without the
  hard top-r gate. Added `--no_pretrain` (random-init ResNet-50 encoder
  instead of ImageNet-pretrained, used for the backbone-pretraining
  ablation). `--mode mt --conf 0.0` (already-existing mean-teacher mode
  with the confidence gate disabled) is the "all-pixel EMA" baseline.
- `prep_data.py`: added `do_breast()` (BrEaST / TCIA Breast-Lesions-USG,
  DOI 10.7937/9WKK-0141, CC-BY 4.0 -- 252 usable cases, 4 normal cases with
  no tumor mask excluded, same convention as `do_busi()`).
- `make_splits.py`: added `"breast"` and `"oasbud"` to the per-domain seed
  table (case/patient-level 70/10/20 split, consistent with the other
  domains' grouping).
- `common.py`, `fm.py`, `fm_precompute.py`: unchanged in interface; included
  for a complete, matching snapshot.

## Scope note

This pipeline is independent of the `transunet/` directories: different
backbone (U-Net+ResNet50 vs. TransUNet), different data format (local
`.npz` packs vs. the original account's image/label directories), different
AutoSAMUS checkpoint (a separately-trained copy on this account, not the one
under `transunet/`). Its numbers must not be pooled with the `transunet/`
results. The final manuscript reports these separately, in Table 4 and Appendix A.
Evaluation averages the model predictions from the original image and its
horizontal flip. It does not load or run AutoSAMUS at inference time.
The archived `predict()` helper calls `model.train()` after evaluation,
including when invoked for the EMA teacher during training. This preserves
the behavior of this snapshot for comparison with the archived results.
