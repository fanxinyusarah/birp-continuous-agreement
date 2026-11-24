"""Create a fresh UCLM cohort without overwriting existing experiments."""
import json
import shutil
from pathlib import Path

source = Path(__file__).resolve().parent
destination = Path('/scratch/xf810/birp_uclm_control_20261111')
if destination.exists():
    raise FileExistsError(destination)
destination.mkdir()
root = destination / 'uclm'
root.mkdir()
for name in ('src', 'legacy'):
    shutil.copytree(source / name, root / name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
template = Path('/scratch/xf810/SegMamba/logs/external_grouped_v1/breast_bidir_seed1337/args.json')
cfg = json.loads(template.read_text())
data = Path('/scratch/xf810/SegMamba/data/nonanchor_protocol_v1')
cfg.update(source_root=str(data / 'busi_source'), target_root=str(data / 'uclm_target_dev'),
           tgt_splits=str(data / 'uclm_adapt_train.txt'), busbra_root=str(data / 'uclm_target_images'),
           output_dir=str(root / 'unused_default'))
(root / 'training_template.json').write_text(json.dumps(cfg, indent=2) + '\n')
(root / 'logs').mkdir()
print('Created isolated UCLM cohort', root)
