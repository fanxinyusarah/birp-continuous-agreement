#!/bin/bash
#SBATCH --job-name=birp_ablation_eval
#SBATCH --account=preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodelist=dn001
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/xf810/birp_repro_control_20261024/diagnostic_20261104/eval_%j.out
#SBATCH --error=/scratch/xf810/birp_repro_control_20261024/diagnostic_20261104/eval_%j.err
set -euo pipefail
R=/scratch/xf810/birp_repro_control_20261024
export PYTHONPATH="$R/legacy:$R/src" PYTHONHASHSEED=1337 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for seed in 1337 2027 2028; do
  for variant in prior_same_gate ema_all; do
    /home/xf810/.conda/envs/segmamba/bin/python -u "$R/src/evaluate_selected.py" --root "$R" --folder "$R/diagnostic_20261104/runs/seed$seed/$variant"
  done
done
