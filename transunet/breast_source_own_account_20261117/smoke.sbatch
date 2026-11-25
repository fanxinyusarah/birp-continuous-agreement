#!/bin/bash
#SBATCH -A nyuny
#SBATCH -p nvidia
#SBATCH --gres=gpu:v100:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 02:00:00
#SBATCH -o /scratch/xl5874/Eve/birp_breast_source/logs/smoke_%j.log
set -e
P=/scratch/xl5874/Eve/birp_breast_source
mkdir -p $P/logs
W=/dev/shm/bsc_$SLURM_JOB_ID; mkdir -p $W; trap "rm -rf $W" EXIT
export UV_CACHE_DIR=$W/uvc UV_PYTHON_INSTALL_DIR=$W/py UV_LINK_MODE=copy
UV=$HOME/.local/bin/uv
t0=$(date +%s)
$UV venv -q -p 3.9 $W/env && source $W/env/bin/activate
python --version
$UV pip install -q torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
$UV pip install -q numpy==1.26.4 scipy pillow ml_collections einops packaging setuptools
echo "setup $(( $(date +%s) - t0 ))s"
nvidia-smi -L
python -c "import tkinter; print('tkinter OK')" || echo "TKINTER IMPORT FAILED"
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=1337
export TORCH_HOME=/scratch/xl5874/.cache/torch HF_HOME=/scratch/xl5874/.cache/huggingface HF_HUB_OFFLINE=1

for dom in busi uclm oasbud; do
  R=$P/$dom
  echo "=== $dom: import check ==="
  PYTHONPATH="$R/legacy:$R/src" python -c "
import sys
sys.path.insert(0, '$R/legacy'); sys.path.insert(0, '$R/src')
from disentanglement_fda_1 import SegListDataset, TargetSplitDataset, build_transunet, load_pretrained_weights, seed_all
from train_birp_hard_prior import FrozenAutoSAMUS
import train_repro, run_control, repro_utils
print('$dom imports OK')
"
  echo "=== $dom: make_cache (prior caching) ==="
  if [ ! -f "$R/provenance.json" ]; then
    PYTHONPATH="$R/legacy:$R/src" python -u "$R/src/make_cache.py" --root "$R" --template "$R/training_template.json"
  else
    echo "already cached"
  fi
  echo "=== $dom: smoke training (2 epochs, max_steps=2, all 4 variants) ==="
  cd "$R/src"
  PYTHONPATH="$R/legacy:$R/src" python -u run_direction.py --root "$R" --seed 1337 --smoke
done
echo "SMOKE_ALL_DONE $(( $(date +%s) - t0 ))s"
