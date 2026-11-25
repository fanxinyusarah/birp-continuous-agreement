#!/bin/bash
#SBATCH -A nyuny
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/xl5874/Eve/birp_breast_source/logs/figs_%j.log
set -e
P=/scratch/xl5874/Eve/birp_breast_source
mkdir -p $P/logs
W=/dev/shm/figs_$SLURM_JOB_ID; mkdir -p $W; trap "rm -rf $W" EXIT
export UV_CACHE_DIR=$W/uvc UV_PYTHON_INSTALL_DIR=$W/py UV_LINK_MODE=copy
UV=$HOME/.local/bin/uv
$UV venv -q -p 3.9 $W/env && source $W/env/bin/activate
$UV pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
$UV pip install -q numpy==1.26.4 scipy pillow ml_collections einops packaging setuptools
python -u $P/figs_gen.py
