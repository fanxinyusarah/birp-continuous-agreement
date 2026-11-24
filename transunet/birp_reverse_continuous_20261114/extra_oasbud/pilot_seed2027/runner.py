import json, os, subprocess, sys
from pathlib import Path
R=Path("/scratch/xf810/birp_reverse_continuous_20261114/extra_oasbud/oasbud")
P=R.parent/"pilot_seed2027"
sys.path.insert(0,str(R/"src"))
import run_control
from repro_utils import file_hash, runtime_environment
pre=json.loads((R/"preflight/passed.json").read_text())
if not pre["passed"] or pre["provenance_sha256"] != file_hash(R/"provenance.json"):
    raise ValueError("Preflight missing or provenance changed")
run_control.verify_inputs(R)
env=runtime_environment()
for k in ("gpu","total_memory","torch","cuda","cudnn"):
    if env[k] != pre["environment"][k]:
        raise ValueError("Environment differs: "+k)
cfg=json.loads((R/"training_template.json").read_text())
cfg.update(seed=2027,pseudo_mode="bidir",agree_power=1.5)
os.environ["BIRP_EXTERNAL_ROOT"]=str(R)
subprocess.run([sys.executable,str(P/"test_wrapper.py")],check=True)
warm=P/"warmup"
run_control.train(R,cfg,warm,stop=cfg["warmup_epochs"])
start=str(warm/"last.pt")
for name,mode in (("bidir","bidir"),("ema_all","ema_all"),("hard_prior","hard_prior")):
    run_control.train(R,dict(cfg,pseudo_mode=mode),P/name,start=start)
folder=P/"continuous"
if not (folder/"completed.json").exists():
    last=folder/"last.pt"
    args=run_control.command(R,cfg,folder,start="" if last.exists() else start,
                             resume=str(last) if last.exists() else "")
    args[2]=str(P/"wrapper.py")
    subprocess.run(args,check=True)
initials=[]
for name in ("bidir","ema_all","hard_prior","continuous"):
    initial=json.loads((P/name/"initial_state.json").read_text())
    initials.append({k:initial[k] for k in ("model","ema","optimizer","scaler","rng","global_step","epoch")})
if not all(v==initials[0] for v in initials):
    raise ValueError("Variants do not share the same warm-up state")
(P/"shared_start_verified.json").write_text(json.dumps({"passed":True,"seed":2027},indent=2))
for name in ("bidir","ema_all","hard_prior","continuous"):
    subprocess.run([sys.executable,str(P/"evaluate.py"),"--root",str(R),"--folder",str(P/name)],check=True)
(P/"pilot_completed.json").write_text(json.dumps({"passed":True,"seed":2027},indent=2))
