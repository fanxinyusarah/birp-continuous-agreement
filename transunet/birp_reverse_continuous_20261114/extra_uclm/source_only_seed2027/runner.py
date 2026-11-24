import json, subprocess, sys
from pathlib import Path
R=Path('/scratch/xf810/birp_reverse_continuous_20261114/extra_uclm/uclm')
O=Path('/scratch/xf810/birp_reverse_continuous_20261114/extra_uclm/source_only_seed2027')
E=Path('/scratch/xf810/birp_reverse_continuous_20261114/extra_uclm/pilot/evaluate.py')
sys.path.insert(0,str(R/'src'))
import run_control
from repro_utils import file_hash, runtime_environment
pre=json.loads((R/'preflight/passed.json').read_text())
if not pre['passed'] or pre['provenance_sha256']!=file_hash(R/'provenance.json'):
    raise ValueError('Preflight missing or changed')
run_control.verify_inputs(R)
env=runtime_environment()
for k in ('gpu','total_memory','torch','cuda','cudnn'):
    if env[k]!=pre['environment'][k]: raise ValueError('Environment differs: '+k)
cfg=json.loads((R/'training_template.json').read_text())
cfg.update(seed=2027, pseudo_mode='bidir', agree_power=1.5, warmup_epochs=55,
           pseudo_epochs=0, lambda_pseudo=0.0, lambda_entropy=0.0)
run_control.train(R,cfg,O/'fit')
subprocess.run([sys.executable,str(E),'--root',str(R),'--folder',str(O/'fit')],check=True)
(O/'source_only_completed.json').write_text(json.dumps(dict(seed=2027,passed=True)))
