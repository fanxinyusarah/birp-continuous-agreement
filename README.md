# Continuous Agreement for Breast Ultrasound Domain Adaptation

Research code and experiment records for *Continuous Foundation-Guided
Pseudo-Label Weighting for Breast Ultrasound Domain Adaptation* by Xinyu Fan
and Xinyi Li. The student learns from the EMA teacher's binary pseudo-labels.
A frozen AutoSAMUS prior contributes the pixel weight
`w = 0.5 + 0.5 * (2 * abs(q - 0.5)) * (1 - abs(p - q))`, where `p` and `q`
are foreground probabilities from the teacher and prior, respectively.

## Workflows

| Manuscript experiment | Entry points and provenance |
| --- | --- |
| BUSI as source, TransUNet | [Core controlled training and BUS-BRA evaluation](transunet/repro_control_20261024/). The `exploratory_variants_20261108` subdirectory contains continuous agreement and foreground-floor variants. [External evaluation](transunet/external_control_20261106/), [external continuous agreement](transunet/external_soft_agreement_20261109/), [external foreground floor](transunet/external_soft_floor_20261109/), and the [BUS-UCLM control](transunet/uclm_control_20261110/) cover the other target configurations. |
| BUS-BRA as source, TransUNet | [Four direction snapshots](transunet/birp_reverse_continuous_20261114/), including seed pilot wrappers and training templates from the Nov 14 handoff. |
| BrEaST as source, TransUNet | The same controlled trainer and continuous wrapper under the separately accessed target partitions. [Per-seed summaries](results/available_run_metrics.json) give all three reported directions. |
| BrEaST as source and component ablation, U-Net | [Early U-Net trainer and data helpers](unet/reference_20261021/). [Per-seed summaries](results/available_run_metrics.json) save the reported later configuration and the final EMA Dice. See the inventory below for source coverage. |

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
   The U-Net summaries report the final-iteration EMA test mean.
   The TransUNet boundary evaluator uses the image diagonal for exactly one
   empty mask and zero for two empty masks.

## Numeric results

Run `python3 tools/aggregate_results.py` to reproduce the means and sample
standard deviations for the runs in the bundled numeric record. The record
includes the BUS-BRA-source and BrEaST-source TransUNet runs, the U-Net
BrEaST-source runs and the U-Net component ablation. The reported BUSI-source
TransUNet table comes from the controlled manuscript record.

## Source coverage

The TransUNet files are historical snapshots from the prior manuscript and
collaboration handoff. The Oct 21 U-Net code predates the final continuous
modes. The exact later U-Net training revision, some BrEaST-source
orchestration scripts, and the qualitative panel generation script have yet
to be recovered into this package. Their reported outputs are identified by
pipeline, source, target, setting and seed in the numeric record. Exact
reproduction of all runs also requires the corresponding split manifests and
pretrained checkpoints. Training on GPU has not been rerun for this archive.

The legacy TransUNet and AutoSAMUS source trees retain the upstream notices
where supplied. Third-party model code and weights are subject to their
respective terms. See [AutoSAMUS](https://github.com/xianlin7/AutoSAMUS) and
[TransUNet](https://github.com/Beckschen/TransUNet).
