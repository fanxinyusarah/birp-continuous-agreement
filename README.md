# Continuous Agreement for Breast Ultrasound Domain Adaptation

Research code and experiment records for *Continuous Foundation-Guided
Pseudo-Label Weighting for Breast Ultrasound Domain Adaptation* by Xinyu Fan
and Xinyi Li ([arXiv:2611.33843](https://arxiv.org/abs/2611.33843)). The student learns from the EMA teacher's binary pseudo-labels.
A frozen AutoSAMUS prior contributes the pixel weight
`w = 0.5 + 0.5 * (2 * abs(q - 0.5)) * (1 - abs(p - q))`, where `p` and `q`
are foreground probabilities from the teacher and prior, respectively.

## Workflows

| Manuscript experiment | Entry points and provenance |
| --- | --- |
| BUSI as source, TransUNet | [Core controlled training and BUS-BRA evaluation](transunet/repro_control_20261024/). The `exploratory_variants_20261108` subdirectory contains continuous agreement and foreground-floor variants. [External evaluation](transunet/external_control_20261106/), [external continuous agreement](transunet/external_soft_agreement_20261109/), [external foreground floor](transunet/external_soft_floor_20261109/), and the [BUS-UCLM control](transunet/uclm_control_20261110/) cover the other target configurations. The `active_targets` and `active_target` subdirectories contain their deployed code and templates. |
| BUS-BRA as source, TransUNet | [Four direction snapshots](transunet/birp_reverse_continuous_20261114/), including final seed-specific pilot wrappers, full-length source-only runners, and training templates retrieved from Jubail. |
| BrEaST as source, TransUNet | The same controlled trainer and continuous wrapper under the separately accessed target partitions ([orchestration code, run under the second account](transunet/breast_source_own_account_20261117/)). [Per-seed summaries](results/available_run_metrics.json) give all three reported directions (BUSI, BUS-UCLM, OASBUD). |
| BrEaST as source and component ablation, U-Net | [Final U-Net trainer and data helpers](unet/reference_20261120/), including the `cont_full`/`cont_cert`/`cont_agree` modes and the BrEaST source domain ([`reference_20261021`](unet/reference_20261021/) is the earlier snapshot, kept for history). [Per-seed summaries](results/available_run_metrics.json) give the reported configuration and the final EMA Dice, including the certainty/agreement-only ablation and the prior-checkpoint and no-pretraining swaps used to narrow the BrEaST→BUS-UCLM cross-pipeline discrepancy. |
| Qualitative figures | [Panel generation script](tools/generate_qualitative_figures.py) ([Slurm wrapper](tools/generate_qualitative_figures.sbatch)): loads the seed-1337 TransUNet checkpoints for BrEaST→BUSI/BUS-UCLM, recomputes the frozen prior on the held-out test images because the cache covers only the adaptation split, and picks the largest-gain, median-gain and largest-loss cases by `cont_full` Dice minus the best baseline's Dice, per image. |

Submit the qualitative Slurm wrapper from the repository root. Set
`BIRP_FIGURE_SCRIPT` to the script's absolute path when submitting elsewhere.

The historical Python and Slurm files retain the original run layout. Build a
separate working copy for each target. Replace cluster-specific paths and
Slurm resource settings with local equivalents before running. Keep the
scientific settings, ordered split membership and seed fixed. In particular,
the BUSI-, BUS-BRA-, and BrEaST-source protocols use different test partitions.

## Training and evaluation

1. Obtain the datasets and frozen AutoSAMUS checkpoints from their providers
   under the applicable terms. The repository contains code and numeric
   metrics. Supply your own image and label directories and split lists.
2. Start with the version and `training_template.json` for the direction in
   the table. The controlled TransUNet runs used Python 3.9, NumPy 1.26.4 and
   CUDA 11.8 with PyTorch 2.1.0. The original code also uses torchvision and
   SciPy. The archived AutoSAMUS model code is under
   [third_party](third_party/AutoSAMUS/SAMUS_For_3D/). Set `autosamus_root`
   and `autosamus_ckpt` in the template to the correct model source and
   weights.
3. Run `src/make_cache.py` and the protocol preflight for the corresponding
   working copy, then use its runner and the continuous-agreement wrapper
   indicated above. Preserve seed and starting state within each matched
   comparison. The `src/run_control.py` command builder constructs the
   trainer arguments. `PROTOCOL.md` in the core folder describes the
   cache, software and input verification.
4. Run the protocol's held-out evaluator after training. TransUNet evaluates
   the EMA weights at the student's source-validation-selected checkpoint.
   The U-Net summaries report the final-iteration EMA test mean, with
   horizontal-flip test-time augmentation.
   The TransUNet boundary evaluator uses the image diagonal for exactly one
   empty mask and zero for two empty masks.

## Numeric results

Run `python3 tools/aggregate_results.py` to reproduce the means and sample
standard deviations for the runs in the bundled numeric record. The record
contains 207 runs spanning the BUSI-, BUS-BRA-, and BrEaST-source TransUNet
experiments, the U-Net BrEaST-source runs, and the U-Net component ablation.
All 16 BUSI-source Dice entries were checked against the original per-seed
Jubail evaluation summaries.

## Source coverage

The TransUNet files come from the manuscript handoff and the corresponding
Jubail execution directories. The [method diagram](figures/continuous_agreement_graph.tex)
is the source used to build the manuscript figure. The final U-Net training
revision, the BrEaST-source orchestration scripts run under the second
account, and the qualitative panel generation script are now included (see
the table above); `reference_20261021` remains for history only. Their
reported outputs are identified by pipeline, source, target, setting and
seed in the numeric record. Exact reproduction of all runs also requires the
corresponding split manifests and pretrained checkpoints. Training on GPU has
not been rerun for this archive.

## Third-party code

- The archived `legacy/networks/vit_seg_*` files include [TransUNet](https://github.com/Beckschen/TransUNet)
  code under [Apache 2.0](transunet/LICENSE.TransUNet).
- The archived AutoSAMUS tree includes [SAMUS and AutoSAMUS](https://github.com/xianlin7/SAMUS)
  code under the [upstream MIT license](third_party/AutoSAMUS/SAMUS_For_3D/LICENSE.SAMUS)
  and [Segment Anything](https://github.com/facebookresearch/segment-anything)-derived
  code under [Apache 2.0](third_party/AutoSAMUS/SAMUS_For_3D/LICENSE).
  The archived `utils/config.py` and `utils/evaluation.py` contain downstream
  changes, and `utils/data_abus.py` is an additional ABUS loader in this snapshot.
