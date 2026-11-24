#!/bin/bash
#SBATCH --job-name=birp_uclm_preflight
#SBATCH --account=preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --nodelist=dn001
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/xf810/birp_uclm_control_20261111/preflight_%j.out
#SBATCH --error=/scratch/xf810/birp_uclm_control_20261111/preflight_%j.err
set -euo pipefail
R=/scratch/xf810/birp_uclm_control_20261111/uclm
export PYTHONPATH="$R/legacy:$R/src" PYTHONHASHSEED=1337 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
exec /home/xf810/.conda/envs/segmamba/bin/python -u "$R/src/preflight.py" --root "$R"
