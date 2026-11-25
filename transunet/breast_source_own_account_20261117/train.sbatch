#!/bin/bash
#SBATCH -A nyuny
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -o /scratch/xl5874/Eve/birp_breast_source/logs/train_%x_%j.log
set -e
P=/scratch/xl5874/Eve/birp_breast_source
W=/dev/shm/bsc_$SLURM_JOB_ID; mkdir -p $W; trap "rm -rf $W" EXIT
export UV_CACHE_DIR=$W/uvc UV_PYTHON_INSTALL_DIR=$W/py UV_LINK_MODE=copy
UV=$HOME/.local/bin/uv
$UV venv -q -p 3.9 $W/env && source $W/env/bin/activate
$UV pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
$UV pip install -q numpy==1.26.4 scipy pillow ml_collections einops packaging setuptools
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1337
export TORCH_HOME=/scratch/xl5874/.cache/torch HF_HOME=/scratch/xl5874/.cache/huggingface HF_HUB_OFFLINE=1
R=$P/$1
cd $R/src
PYTHONPATH="$R/legacy:$R/src" python -u run_direction.py --root "$R" --seed $2
