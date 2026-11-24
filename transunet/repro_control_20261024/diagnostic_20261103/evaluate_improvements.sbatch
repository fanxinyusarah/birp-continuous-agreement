#!/bin/bash
#SBATCH --job-name=birp_improve_eval
#SBATCH --account=preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodelist=dn001
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-8%1
#SBATCH --output=/scratch/xf810/birp_repro_control_20261024/improvement_eval_%A_%a.out
#SBATCH --error=/scratch/xf810/birp_repro_control_20261024/improvement_eval_%A_%a.err
set -euo pipefail
R=/scratch/xf810/birp_repro_control_20261024
seeds=(1337 2027 2028)
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}
case $((SLURM_ARRAY_TASK_ID / 3)) in
  0) folder="$R/soft_floor_20261108/seed$seed" ;;
  1) folder="$R/exploratory_variants_20261108/foreground_floor/seed$seed" ;;
  2) folder="$R/exploratory_variants_20261108/soft_agreement/seed$seed" ;;
esac
export PYTHONPATH="$R/legacy:$R/src" PYTHONHASHSEED=1337 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec /home/xf810/.conda/envs/segmamba/bin/python -u "$R/src/evaluate_selected.py" --root "$R" --folder "$folder"
