# Original BiRP reproducibility repair

Remote directory: /scratch/xf810/birp_repro_control_20261024

1. Read PROTOCOL.md. Keep original_review/ and both existing manuscripts intact.
2. Submit jobs/preflight.sbatch. Actual GPU legacy trace, fixed prior cache,
   input provenance and strict fresh/restart contracts must all pass.
3. Submit jobs/controlled.sbatch with afterok dependency on preflight.
4. Submit jobs/summary.sbatch with afterok dependency on the entire array.

Use the existing /home/xf810/.conda/envs/segmamba/bin/python environment.
No package versions or shared original code are modified. Frozen original
readers/trainer and TransUNet are in legacy/. Repaired code is isolated in src/.

Success files: preflight/passed.json, runs/seed*/shared_start_verified.json,
runs/seed1337/full_repeat.json, runs/seed*/seed_completed.json and finally
CONTROLLED_RESULTS.json / CONTROLLED_RESULTS.md. Missing files are incomplete,
not zero results. Never relabel the nine older gamma results as controlled.

No paper edits, arXiv upload or conference submission are automatic.
