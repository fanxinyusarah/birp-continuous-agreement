# EMA-all exploratory audit

The three-seed EMA-all versus BiRP comparison uses the same 376 previously
inspected BUS-BRA test images and matched per-seed warm-up states. Cases are
sorted into equal image quartiles by ground-truth foreground fraction, solely
for post-hoc diagnosis. These are not independent-patient tests, confirmation
data, or a basis for selecting hyperparameters.

| Lesion fraction quartile | Cases | EMA-all minus BiRP Dice |
| --- | ---: | ---: |
| 0.00855--0.03344 | 94 | +0.09336 |
| 0.03356--0.05576 | 94 | +0.05119 |
| 0.05662--0.10603 | 94 | +0.02331 |
| 0.10605--0.36557 | 94 | -0.00472 |

Training logs through each source-selected checkpoint record BiRP pseudo-label
coverage around 0.219--0.222, versus 1.0 by construction for EMA-all. The
foreground share among selected pseudo-label pixels averages about 0.052--0.055
for BiRP and 0.093--0.099 for EMA-all. These are batch-level training-time
summaries, not pseudo-label accuracy, not case-level test coverage, and not a
causal explanation; the selected checkpoints and training trajectories differ.

On Jubail, historical BUSI-to-BrEaST and BUSI-to-OASBUD EMA and BiRP runs
exist for seed 1337. Their args explicitly specify BUSI as source; they must
not be confused with the manuscript's BUS-BRA-to-BrEaST protocol. Historical
EMA uses confidence threshold 0.9 and keep ratio 0.25, not the all-pixel EMA
control. In the stored summaries, BUSI-to-BrEaST Dice is 0.70548 (BiRP) versus
0.70011 (filtered EMA), n=51, and BUSI-to-OASBUD is 0.46562 versus 0.42353,
n=40. These are single-seed observations and the test sets have been inspected;
they do not establish reproducibility or prove superiority over EMA-all.

A fair rerun needs matched source and target splits, source-validation
checkpoint selection, a shared warm-up, consistent numerical environment,
and all comparison variants before test inspection. The strict reproducibility
pipeline's cached-prior preparation assumes the BUS-BRA layout and needs
adaptation and validation for these datasets before it can safely be reused.
No external training was launched by this audit.
