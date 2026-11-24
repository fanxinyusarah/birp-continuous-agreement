import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

target, seed = sys.argv[1], int(sys.argv[2])
assert target in ('breast', 'oasbud') and seed in (1337, 2027, 2028)
root = Path('/scratch/xf810/birp_external_control_20261106') / target
experiment = Path('/scratch/xf810/birp_external_soft_agreement_20261109')
sys.path.insert(0, str(root / 'src'))
import run_control

sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {name: sha(experiment / name) for name in ('wrapper.py', 'test_wrapper.py', 'runner.py')}
manifest.update({f'{name}_provenance': sha(root.parent / name / 'provenance.json') for name in ('breast', 'oasbud')})
if manifest != json.loads((experiment / 'manifest.json').read_text()):
    raise ValueError('Experiment code or data provenance changed')
run_control.verify_inputs(root)
passed = json.loads((root / 'preflight/passed.json').read_text())
if not passed['passed'] or passed['provenance_sha256'] != sha(root / 'provenance.json'):
    raise ValueError('External preflight failed or changed')
env = dict(os.environ, BIRP_EXTERNAL_ROOT=str(root))
subprocess.run([sys.executable, str(experiment / 'test_wrapper.py')], check=True, env=env)
cfg = json.loads((root / 'training_template.json').read_text())
cfg.update(seed=seed, pseudo_mode='bidir', agree_power=1.5)
folder = experiment / target / f'seed{seed}'
if (folder / 'completed.json').exists():
    raise FileExistsError('Result already completed; inspect before retry')
last, warm = folder / 'last.pt', root / 'runs' / f'seed{seed}' / 'warmup/last.pt'
args = run_control.command(root, cfg, folder, start='' if last.exists() else str(warm),
                           resume=str(last) if last.exists() else '')
args[2] = str(experiment / 'wrapper.py')
subprocess.run(args, check=True, env=env)
