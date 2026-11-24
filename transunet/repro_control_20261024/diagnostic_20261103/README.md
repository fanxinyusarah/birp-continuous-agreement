# Original BiRP: exploratory diagnostic, 2026-11-03

Run `python3 audit.py` in this directory to reproduce the descriptive audit.
The input records are copied from the repaired shared-warm-up cohort; the
script verifies that all 12 runs use the same 376 test image and label hashes.
These are previously inspected locked-test data, not a fresh confirmation set.

Three-seed mean per-case Dice: gamma 1.5 minus hard prior is -0.01687.
BiRP wins on 142 images, loses on 233 and ties on one, averaging predictions'
Dice across seeds per image. Images are not independent patients.
Sorted by reference lesion area, four image quartiles of 94 have mean Dice
differences of -0.05153, -0.03480, -0.01985, and +0.03869, respectively.
This post-hoc descriptive pattern is not a causal or confirmatory finding.

The mean logged adaptation-time selected pixel fraction up to the independently
source-selected checkpoint is 0.2201, 0.2194 and 0.2217 across seeds for
gamma 1.5. Hard-prior uses all pixels by definition. These are training-batch
statistics, not per-test-case coverage or pseudo-label accuracy. BiRP also
uses EMA labels and continuous weights, whereas hard-prior uses prior labels
and uniform weights. Their difference cannot isolate the cause of the deficit.

To test mechanisms without using locked-test performance for selection, define
the variants and select checkpoints on source validation in advance. At minimum
compare EMA vs prior labels under the same eligibility/weight mask, and compare
the same pseudo-label source with all-pixel versus matched reliability-weighted
supervision. Check class-specific selected counts and pseudo-label quality only
on a separately designated diagnostic split with authorized labels; labels must
never enter adaptation, hyperparameter tuning, or model selection. Preserve the
existing cohort, shared warm-up, and all unfavorable findings. Reserve a fresh
patient-disjoint target test cohort for any new performance claim.
