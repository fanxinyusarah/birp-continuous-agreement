#!/bin/bash
#SBATCH --job-name=birp_ext_eval_fix
#SBATCH --account=preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodelist=dn001
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/scratch/xf810/birp_external_control_20261106/eval_fixed_%A_%a.out
#SBATCH --error=/scratch/xf810/birp_external_control_20261106/eval_fixed_%A_%a.err
set -euo pipefail
targets=(breast oasbud)
seeds=(1337 2027 2028)
R=/scratch/xf810/birp_external_control_20261106/${targets[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
export PYTHONPATH="$R/legacy:$R/src" PYTHONHASHSEED=1337 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for variant in bidir ema_filtered ema_all; do
  /home/xf810/.conda/envs/segmamba/bin/python -u /scratch/xf810/birp_external_code_20261106/evaluate_external_fixed.py --root "$R" --folder "$R/runs/seed$seed/$variant"
done
