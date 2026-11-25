"""Generate Fig.1 (method comparison panels) and Fig.2 (teacher/prior/weight) for
BrEaST-source -> BUSI/UCLM, using the real TransUNet checkpoints (seed 1337).
Prior q is computed LIVE on the test images (the cached prior_cache.pt only covers the
adapt split, not the held-out test split used here -- verified by reading make_cache.py).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

OUT = "/scratch/xl5874/Eve/birp/paper/qual_20261119"
os.makedirs(OUT, exist_ok=True)
MODES = ["source_only", "bidir", "ema_all", "cont_full"]
TARGETS = {
    "busi": ("/scratch/xf810/SegMamba/data/nonanchor_protocol_v1/busi_target_test", 126),
    "uclm": ("/scratch/xf810/SegMamba/data/nonanchor_protocol_v1/uclm_target_test", 128),
}
SEED = 1337
dev = torch.device("cuda")

results = {}
for target, (test_root, expected_n) in TARGETS.items():
    root = f"/scratch/xl5874/Eve/birp_breast_source/{target}"
    sys.path.insert(0, f"{root}/legacy")
    from disentanglement_fda_1 import SegListDataset, build_transunet
    from train_birp_hard_prior import FrozenAutoSAMUS

    ds = SegListDataset(test_root, "val", 224, True, True)
    assert len(ds) == expected_n, f"{target} test set size mismatch: {len(ds)} vs {expected_n}"
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    imgs, labs, names = [], [], []
    for x, y, n in loader:
        imgs.append(x); labs.append(y); names.extend(n)
    imgs = torch.cat(imgs); labs = torch.cat(labs)
    print(target, "test images", imgs.shape, "labels", labs.shape, "names", len(names), flush=True)

    cfg = json.loads(open(f"{root}/training_template.json").read()) if os.path.exists(f"{root}/training_template.json") else {}
    autosamus_ckpt = "/scratch/xf810/AutoSAMUS/SAMUS_For_3D/checkpoints/ABUS/AutoSAMUS_03071546_35_0.6162310679616952.pth"
    autosamus_root = f"{root}/autosamus_repo" if os.path.isdir(f"{root}/autosamus_repo") else "/scratch/xl5874/Eve/birp_breast_source/autosamus_repo"
    prior = FrozenAutoSAMUS(autosamus_root, autosamus_ckpt, dev)

    preds = {}
    probs_full = {}
    for mode in MODES:
        ck_path = f"{root}/runs/seed{SEED}/{mode}/best_source.pt"
        ck = torch.load(ck_path, map_location="cpu")
        model = build_transunet("R50-ViT-B_16", 224, 2).to(dev).eval()
        model.load_state_dict(ck["ema_model"], strict=True)
        probs_batches = []
        with torch.inference_mode():
            for i in range(0, len(imgs), 8):
                batch = imgs[i:i + 8].to(dev)
                p = torch.softmax(model(batch), 1)[:, 1]
                probs_batches.append(p.cpu())
        probs = torch.cat(probs_batches)
        preds[mode] = (probs >= 0.5)
        probs_full[mode] = probs
        del model
        torch.cuda.empty_cache()

    q_batches = []
    with torch.inference_mode():
        for i in range(0, len(imgs), 8):
            batch = imgs[i:i + 8].to(dev)
            q = prior(batch)[:, 0]
            q_batches.append(q.cpu())
    q_full = torch.cat(q_batches)
    del prior
    torch.cuda.empty_cache()

    gt = labs > 0
    dice = {}
    for mode in MODES:
        pred = preds[mode]
        inter = (pred & gt).float().sum((1, 2))
        den = pred.float().sum((1, 2)) + gt.float().sum((1, 2))
        dice[mode] = ((2 * inter + 1e-5) / (den + 1e-5)).numpy()

    best_baseline = np.maximum.reduce([dice[m] for m in ("source_only", "bidir", "ema_all")])
    gain = dice["cont_full"] - best_baseline
    order = np.argsort(gain)
    idx_best = int(order[-1])
    idx_worst = int(order[0])
    idx_tie = int(order[len(order) // 2])
    picks = {"improvement": idx_best, "near_tie": idx_tie, "regression": idx_worst}

    print(target, "picks:", {k: dict(idx=v, name=names[v], cont_full=float(dice["cont_full"][v]),
          best_baseline=float(best_baseline[v]), gain=float(gain[v])) for k, v in picks.items()}, flush=True)

    results[target] = {k: dict(name=names[v], cont_full_dice=float(dice["cont_full"][v]),
                                best_baseline_dice=float(best_baseline[v]), gain=float(gain[v]),
                                dice_by_method={m: float(dice[m][v]) for m in MODES})
                        for k, v in picks.items()}

    def to_u8(t):
        a = (t.clamp(0, 1) * 255).byte().cpu().numpy()
        return a

    def save_panel(idx, tag):
        img = to_u8(imgs[idx, 0])
        gt_m = (gt[idx].numpy() * 255).astype(np.uint8)
        cells = [Image.fromarray(img).convert("L"), Image.fromarray(gt_m).convert("L")]
        for mode in MODES:
            pm = (preds[mode][idx].numpy() * 255).astype(np.uint8)
            cells.append(Image.fromarray(pm).convert("L"))
        w, h = cells[0].size
        canvas = Image.new("L", (w * len(cells), h), 255)
        for i, c in enumerate(cells):
            canvas.paste(c.resize((w, h)), (i * w, 0))
        path = f"{OUT}/{target}_fig1_{tag}.png"
        canvas.convert("RGB").save(path)
        return path

    def heat_u8(t):
        a = (t.clamp(0, 1) * 255).byte().cpu().numpy()
        return a

    def save_method_fig(idx, tag):
        img = to_u8(imgs[idx, 0])
        p = probs_full["cont_full"][idx]
        q = q_full[idx]
        c = (2 * (q - 0.5).abs()).clamp(0, 1)
        a = (1 - (p - q).abs()).clamp(0, 1)
        w_map = 0.5 + 0.5 * c * a
        panels = [Image.fromarray(img).convert("L"),
                  Image.fromarray(heat_u8(p)).convert("L"),
                  Image.fromarray(heat_u8(q)).convert("L"),
                  Image.fromarray(heat_u8(w_map)).convert("L")]
        wdt, hgt = panels[0].size
        canvas = Image.new("L", (wdt * len(panels), hgt), 255)
        for i, pnl in enumerate(panels):
            canvas.paste(pnl.resize((wdt, hgt)), (i * wdt, 0))
        path = f"{OUT}/{target}_fig2_{tag}.png"
        canvas.convert("RGB").save(path)
        return path

    for tag, idx in picks.items():
        p1 = save_panel(idx, tag)
        p2 = save_method_fig(idx, tag)
        print("saved", p1, p2, flush=True)

    del imgs, labs, preds, probs_full, q_full
    torch.cuda.empty_cache()

with open(f"{OUT}/selection_summary.json", "w") as f:
    json.dump(dict(seed=SEED, modes=MODES, results=results,
                    note="cont_full vs max(source_only,bidir,ema_all) per image; "
                         "improvement=argmax(gain), near_tie=median(gain), regression=argmin(gain)"),
               f, indent=2)
print("ALL_DONE")
