"""Precompute frozen FM priors for every image of every domain + zero-shot diagnostics."""
import argparse, json, os
import numpy as np, torch
from common import DATA, load_domain, load_splits, dice_per_image
from fm import AutoSAMUS, PromptSAMUS, mask_to_box

ap = argparse.ArgumentParser()
ap.add_argument("--auto", default="", help="name=ckpt,... prompt-free AutoSAMUS checkpoints")
ap.add_argument("--samus", default="", help="SAMUS ckpt for GT-box oracle diagnostic")
ap.add_argument("--sam", default="", help="SAM vit_b ckpt for GT-box oracle diagnostic")
ap.add_argument("--domains", default="busbra,busi,uclm,udiat")
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = torch.device("cuda")
SP = load_splits()
report = {}

def run(fn, x, bs=32):
    out = []
    for i in range(0, len(x), bs):
        out.append(fn(i, x[i:i + bs]))
    return torch.cat(out)

for dom in a.domains.split(","):
    D = load_domain(dom)
    x = torch.from_numpy(D["img"]).float().div(255).unsqueeze(1)
    gt = torch.from_numpy(D["mask"]).unsqueeze(1) > 0
    def dsum(prob, name):
        d = dice_per_image(prob > 0.5, gt).numpy()
        r = {k: float(d[SP[dom][k]].mean()) for k in "ADT"}
        report[f"{dom}/{name}"] = r
        print(dom, name, json.dumps(r), flush=True)
    for spec in [s for s in a.auto.split(",") if s]:
        name, ck = spec.split("=")
        m = AutoSAMUS(ck).to(dev)
        prob = run(lambda i, xb: 0.5 * (m(xb.to(dev)) + m(xb.to(dev).flip(-1)).flip(-1)).cpu(), x)
        np.savez_compressed(os.path.join(DATA, f"fm_{name}_{dom}.npz"), prob=(prob[:, 0] * 255).round().byte().numpy())
        dsum(prob, name)
        del m; torch.cuda.empty_cache()
    for kind, ck in [("samus", a.samus), ("sam", a.sam)]:
        if not ck:
            continue
        m = PromptSAMUS(ck, kind).to(dev)
        boxes, _ = mask_to_box(gt.float())
        prob = run(lambda i, xb: m(xb.to(dev), boxes[i:i + len(xb)].to(dev))[0].cpu(), x, bs=16 if kind == "samus" else 4)
        dsum(prob, kind + "_gtbox")
        del m; torch.cuda.empty_cache()
if a.out:
    json.dump(report, open(a.out, "w"), indent=1)
