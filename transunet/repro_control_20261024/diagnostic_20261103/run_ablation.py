import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path('/scratch/xf810/birp_repro_control_20261024')
experiment = root / 'diagnostic_20261104'
sys.path.insert(0, str(root / 'src'))
import run_control

seed = int(sys.argv[1])
assert seed in (1337, 2027, 2028)
run_control.verify_inputs(root)
wrapper = experiment / 'ablation_wrapper.py'
tests = experiment / 'test_ablation.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'wrapper_sha256': sha(wrapper), 'test_sha256': sha(tests),
            'runner_sha256': sha(Path(__file__)), 'provenance_sha256': sha(root / 'provenance.json')}
manifest_path = experiment / 'manifest.json'
if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
    raise RuntimeError('Frozen experiment manifest missing or changed')
subprocess.run([sys.executable, str(tests)], check=True, env=dict(os.environ, BIRP_ABLATION='prior_same_gate'))
cfg = json.loads((root / 'training_template.json').read_text())
cfg.update(seed=seed, pseudo_mode='bidir', agree_power=1.5)
warmup = root / 'runs' / f'seed{seed}' / 'warmup' / 'last.pt'
if not warmup.is_file():
    raise FileNotFoundError(warmup)
for mode in ('prior_same_gate', 'ema_all'):
    folder = experiment / 'runs' / f'seed{seed}' / mode
    complete = folder / 'completed.json'
    if complete.exists():
        saved = json.loads(complete.read_text())
        if saved['epoch'] != cfg['num_epochs'] or saved['provenance_sha256'] != manifest['provenance_sha256'] or saved['args']['seed'] != seed:
            raise ValueError(f'Existing result has incompatible settings: {complete}')
        continue
    last = folder / 'last.pt'
    argv = run_control.command(root, cfg, folder, start='' if last.exists() else str(warmup),
                               resume=str(last) if last.exists() else '')
    argv[2] = str(wrapper)
    subprocess.run(argv, check=True, env=dict(os.environ, BIRP_ABLATION=mode))
