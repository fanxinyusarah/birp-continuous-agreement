import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/scratch/xf810/birp_repro_control_20261024')
EXPERIMENT = ROOT / 'exploratory_variants_20261108'
sys.path.insert(0, str(ROOT / 'src'))
import run_control

mode, seed = sys.argv[1], int(sys.argv[2])
assert mode in ('foreground_floor', 'soft_agreement') and seed in (1337, 2027, 2028)
run_control.verify_inputs(ROOT)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {n: sha(EXPERIMENT / n) for n in ('variants.py', 'test_variants.py', 'runner.py')}
manifest['provenance.json'] = sha(ROOT / 'provenance.json')
if manifest != json.loads((EXPERIMENT / 'manifest.json').read_text()):
    raise ValueError('Experiment code or provenance changed')
subprocess.run([sys.executable, str(EXPERIMENT / 'test_variants.py')], check=True)
cfg = json.loads((ROOT / 'training_template.json').read_text())
cfg.update(seed=seed, pseudo_mode='bidir', agree_power=1.5)
folder = EXPERIMENT / mode / f'seed{seed}'
if (folder / 'completed.json').exists():
    raise FileExistsError('Completed output exists; inspect before rerunning')
last = folder / 'last.pt'
warm = ROOT / 'runs' / f'seed{seed}' / 'warmup/last.pt'
args = run_control.command(ROOT, cfg, folder, start='' if last.exists() else str(warm),
                           resume=str(last) if last.exists() else '')
args[2] = str(EXPERIMENT / 'variants.py')
subprocess.run(args, check=True, env=dict(os.environ, BIRP_VARIANT=mode))
