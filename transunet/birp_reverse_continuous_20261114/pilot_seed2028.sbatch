#!/bin/bash
#SBATCH --job-name=birp_reverse_pilot
#SBATCH --account=preempt
#SBATCH --partition=preempt
#SBATCH --qos=preempt-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/xf810/birp_reverse_continuous_20261114/pilot_seed2028_%j.out
#SBATCH --error=/scratch/xf810/birp_reverse_continuous_20261114/pilot_seed2028_%j.err
set -euo pipefail
R=/scratch/xf810/birp_reverse_continuous_20261114/breast
export PYTHONPATH="$R/legacy:$R/src" PYTHONHASHSEED=1337 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
exec /home/xf810/.conda/envs/segmamba/bin/python -u /scratch/xf810/birp_reverse_continuous_20261114/pilot_seed2028/runner.py
