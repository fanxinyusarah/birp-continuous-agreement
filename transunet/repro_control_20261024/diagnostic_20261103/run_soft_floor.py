import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/scratch/xf810/birp_repro_control_20261024')
EXPERIMENT = ROOT / 'soft_floor_20261108'
sys.path.insert(0, str(ROOT / 'src'))
import run_control

seed = int(sys.argv[1])
assert seed in (1337, 2027, 2028)
run_control.verify_inputs(ROOT)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {name: sha(EXPERIMENT / name) for name in ('soft_floor_wrapper.py', 'test_soft_floor.py', 'run_soft_floor.py')}
manifest['provenance.json'] = sha(ROOT / 'provenance.json')
if manifest != json.loads((EXPERIMENT / 'manifest.json').read_text()):
    raise ValueError('Frozen experiment manifest changed')
subprocess.run([sys.executable, str(EXPERIMENT / 'test_soft_floor.py')], check=True)
cfg = json.loads((ROOT / 'training_template.json').read_text())
cfg.update(seed=seed, pseudo_mode='bidir', agree_power=1.5)
folder = EXPERIMENT / f'seed{seed}'
if (folder / 'completed.json').exists():
    raise FileExistsError('Completed output exists; inspect before rerunning')
warm = ROOT / 'runs' / f'seed{seed}' / 'warmup/last.pt'
last = folder / 'last.pt'
argv = run_control.command(ROOT, cfg, folder, start='' if last.exists() else str(warm),
                           resume=str(last) if last.exists() else '')
argv[2] = str(EXPERIMENT / 'soft_floor_wrapper.py')
subprocess.run(argv, check=True, env=os.environ.copy())
