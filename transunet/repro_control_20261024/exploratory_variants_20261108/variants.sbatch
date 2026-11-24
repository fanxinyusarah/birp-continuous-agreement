#!/bin/bash
#SBATCH --job-name=birp_exploratory
#SBATCH --account=preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodelist=dn001
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/scratch/xf810/birp_repro_control_20261024/exploratory_variants_20261108/job_%A_%a.out
#SBATCH --error=/scratch/xf810/birp_repro_control_20261024/exploratory_variants_20261108/job_%A_%a.err
set -euo pipefail
R=/scratch/xf810/birp_repro_control_20261024
export PYTHONPATH="$R/legacy:$R/src" PYTHONHASHSEED=1337 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
seeds=(1337 2027 2028)
modes=(foreground_floor soft_agreement)
exec /home/xf810/.conda/envs/segmamba/bin/python -u "$R/exploratory_variants_20261108/runner.py" "${modes[$((SLURM_ARRAY_TASK_ID / 3))]}" "${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}"
