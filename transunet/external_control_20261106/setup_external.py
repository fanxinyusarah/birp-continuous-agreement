"""Create fresh, isolated external cohorts; never overwrite an existing cohort."""
import json
import shutil
from pathlib import Path

source = Path(__file__).resolve().parent
destination = Path('/scratch/xf810/birp_external_control_20261106')
historical = Path('/scratch/xf810/SegMamba/logs/external_grouped_v1')
if destination.exists():
    raise FileExistsError(destination)
destination.mkdir()
for target in ('breast', 'oasbud'):
    root = destination / target
    root.mkdir()
    for folder in ('src', 'legacy'):
        shutil.copytree(source / folder, root / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    cfg = json.loads((historical / f'{target}_bidir_seed1337/args.json').read_text())
    cfg['output_dir'] = str(root / 'unused_default')
    (root / 'training_template.json').write_text(json.dumps(cfg, indent=2) + '\n')
    (root / 'logs').mkdir()
print('Prepared two isolated external cohorts:', destination)
