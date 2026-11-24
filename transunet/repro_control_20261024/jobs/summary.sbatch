#!/bin/bash
#SBATCH --job-name=birp_repro_summary
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/xf810/birp_repro_control_20261024/logs/summary_%j.out
#SBATCH --error=/scratch/xf810/birp_repro_control_20261024/logs/summary_%j.err
set -euo pipefail
R=/scratch/xf810/birp_repro_control_20261024
export PYTHONPATH="$R/legacy:$R/src"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$R/src"
/home/xf810/.conda/envs/segmamba/bin/python -u summarize.py --root "$R"
