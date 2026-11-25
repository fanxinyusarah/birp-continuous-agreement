"""Unified UDA trainer: one student, one target-pseudo-label rule per `--mode`.

Frozen foundation priors are precomputed (fm_<name>_<domain>.npz) and warped with
the same geometric augmentation as the image, so FM inference is not repeated.
Target dev labels are only used for diagnostics; test labels only with --eval_test.
"""
import argparse
import json
import math
import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

from common import (GPUData, Student, cutout, dice_per_image, ema_update, load_domain, load_prior,
                    load_splits, photometric, predict, rand_affine, soft_dice_loss, warp, weighted_bce)

p = argparse.ArgumentParser()
p.add_argument("--src", default="busi")
p.add_argument("--tgt", default="busbra")
p.add_argument("--mode", default="source")
p.add_argument("--fm", default="", help="comma list of precomputed priors; first is the main prior")
p.add_argument("--arch", default="unet")
p.add_argument("--encoder", default="resnet50")
p.add_argument("--no_pretrain", action="store_true")
p.add_argument("--iters", type=int, default=8000)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--bs", type=int, default=8)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--lam", type=float, default=1.0)
p.add_argument("--ema", type=float, default=0.999)
p.add_argument("--conf", type=float, default=0.9, help="teacher confidence threshold (mt/dacs)")
p.add_argument("--strong", type=float, default=1.0)
p.add_argument("--cutout", type=int, default=1)
# original BiRP hyper-parameters (defaults copied from train_birp_autosamus.py)
p.add_argument("--fg_thr", type=float, default=0.85)
p.add_argument("--bg_thr", type=float, default=0.12)
p.add_argument("--rel_thr", type=float, default=0.55)
p.add_argument("--keep", type=float, default=0.25)
p.add_argument("--min_cov", type=float, default=0.02)
p.add_argument("--gamma", type=float, default=1.5)
p.add_argument("--fm_cert", type=float, default=0.6)
p.add_argument("--tau", type=float, default=1.0, help="temperature/extra knob for new modes")
p.add_argument("--pfm_ckpt", default="/scratch/xl5874/SAMUS_For_3D/checkpoints/samus_pretrained.pth")
p.add_argument("--pfm_kind", default="samus")
p.add_argument("--box_pad", type=float, default=0.05)
p.add_argument("--njit", type=int, default=1)
p.add_argument("--mix", action="store_true")
p.add_argument("--enc_mult", type=float, default=0.1)
p.add_argument("--aff_fm", default="dinov2")
p.add_argument("--aff_tau", type=float, default=0.05)
p.add_argument("--aff_iter", type=int, default=2)
p.add_argument("--aff_alpha", type=float, default=0.5)
p.add_argument("--dump", default="", help="iters at which to dump 128px target probs (student+EMA) for label-free selection, e.g. 2000,4000,6000,8000")
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--eval_every", type=int, default=1000)
p.add_argument("--eval_test", action="store_true")
p.add_argument("--dump_test", default="")
p.add_argument("--pl_diag", action="store_true")
p.add_argument("--save", default="")
p.add_argument("--out", required=True)
p.add_argument("--tag", default="")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")
gen = torch.Generator().manual_seed(args.seed)
SP = load_splits()

fms = [f for f in args.fm.split(",") if f]
S = load_domain(args.src); T = load_domain(args.tgt)
tpri = {f: load_prior(f, args.tgt) for f in fms}
src_tr = GPUData(S, SP[args.src]["A"] + ([] if args.src == args.tgt else SP[args.src]["T"]), dev)  # all non-val source images (oracle: A only)
src_va = GPUData(S, SP[args.src]["D"], dev)
tgt_ad = GPUData(T, SP[args.tgt]["A"], dev, tpri)
tgt_dv = GPUData(T, SP[args.tgt]["D"], dev, tpri)
tgt_te = GPUData(T, SP[args.tgt]["T"], dev, tpri) if args.eval_test else None
print(f"src {args.src} {src_tr.n}/{src_va.n}  tgt {args.tgt} adapt {tgt_ad.n} dev {tgt_dv.n}", flush=True)

pfm = None
if args.mode.startswith("pl_") or args.mode.startswith("tri"):
    from fm import PromptSAMUS, mask_to_box, largest_cc
    pfm = PromptSAMUS(args.pfm_ckpt, args.pfm_kind).to(dev)

affnet = None
if args.mode.startswith("aff"):
    from common import FrozenFeat
    affnet = FrozenFeat(args.aff_fm).to(dev)

model = Student(args.arch, args.encoder, not args.no_pretrain).to(dev)
teacher = copy.deepcopy(model).eval()
for q in teacher.parameters():
    q.requires_grad_(False)
opt = torch.optim.AdamW(model.net.param_groups(args.lr, args.enc_mult) if args.arch.startswith(("dinov2", "samus")) else model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (1 - i / args.iters) ** 0.9)  # applies to all groups
scaler = torch.cuda.amp.GradScaler()


def evaluate(net, data, tta=True):
    pr = predict(net, data.img, tta=tta)
    d = dice_per_image(pr > 0.5, data.mask > 0.5)
    return float(d.mean()), d.cpu().numpy()


# ------------------------------------------------------------------ pseudo labels
def birp_rule(t, q, mode):
    """Faithful binary re-implementation of build_reliable_labels (bidir/agreement/fg_only)."""
    e = torch.maximum(t, 1 - t)
    c = (2 * (q - 0.5).abs()).clamp(0, 1)
    a = (1 - (t - q).abs()).clamp(0, 1)
    r = (c * e * a.pow(args.gamma)).clamp(0, 1)
    y = (t >= 0.5).float()
    if mode == "birp":
        direction = ((y == 1) & (q >= args.fg_thr)) | ((y == 0) & (q <= args.bg_thr))
        base = (e >= args.conf) & direction
    elif mode == "birp_agree":
        base = e >= args.conf
    elif mode == "birp_fg":
        base = (e >= args.conf) & (y == 1) & (q >= args.fg_thr)
    cand = base & (r >= args.rel_thr)
    w = torch.zeros_like(t)
    B = t.shape[0]
    npx = t[0].numel()
    for b in range(B):
        for lab in (0, 1):
            cc = cand[b] & (y[b] == lab)
            n = int(cc.sum())
            if n == 0:
                continue
            k = max(1, min(n, int(math.ceil(args.keep * n))))
            sc = r[b].masked_fill(~cc, -1).flatten()
            idx = torch.topk(sc, k).indices
            w[b].view(-1)[idx] = r[b].view(-1)[idx]
        need = int(math.ceil(args.min_cov * npx))
        if need > int((w[b] > 0).sum()) and int(base[b].sum()) > 0:
            k = min(int(base[b].sum()), need)
            sc = r[b].masked_fill(~base[b], -1).flatten()
            idx = torch.topk(sc, k).indices
            w[b].view(-1)[idx] = r[b].view(-1)[idx]
    return y, w


STATS = {}


def triangulate(P):
    """P [K,B,1,H,W] probs -> [K,B] quality from E[Dice(i,j)] ~ g_i g_j (least squares in log space)."""
    K = P.shape[0]
    Bm = (P >= 0.5).float()
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    Y = []
    for i, j in pairs:
        inter = (Bm[i] * Bm[j]).sum((1, 2, 3)); den = Bm[i].sum((1, 2, 3)) + Bm[j].sum((1, 2, 3))
        Y.append(((2 * inter + 1) / (den + 1)).clamp_min(1e-3).log())
    Y = torch.stack(Y)  # [P,B]
    M = torch.zeros(len(pairs), K, device=P.device)
    for r, (i, j) in enumerate(pairs):
        M[r, i] = 1; M[r, j] = 1
    X = torch.linalg.pinv(M) @ Y
    return X.clamp_max(0).exp()


def pseudo(t, qs, mode):
    """t: teacher prob [B,1,H,W]; qs: list of prior probs. Returns (target, pixel weight)."""
    q = qs[0] if qs else None
    if mode == "mt":
        y = (t >= 0.5).float(); w = (torch.maximum(t, 1 - t) >= args.conf).float()
    elif mode == "fm_soft":
        y = q; w = torch.ones_like(q)
    elif mode == "fm_hard":
        y = (q >= 0.5).float(); w = ((2 * (q - 0.5).abs()) >= args.fm_cert).float()
    elif mode in ("birp", "birp_agree", "birp_fg"):
        y, w = birp_rule(t, q, mode)
    elif mode in ("cont_full", "cont_cert", "cont_agree"):
        # continuous-consistency weighting (not hard-gated): keep ALL teacher pseudo-labels,
        # modulate CE/Dice contribution by a continuous reliability in [0,1] instead of
        # selecting a top-k subset. c = prior certainty, a = teacher-prior agreement.
        y = (t >= 0.5).float()
        c = (2 * (q - 0.5).abs()).clamp(0, 1)
        a = (1 - (t - q).abs()).clamp(0, 1)
        r = {"cont_full": c * a, "cont_cert": c, "cont_agree": a}[mode]
        w = 0.5 + 0.5 * r
    elif mode == "avg_soft":  # naive equal-weight fusion of teacher and prior
        y = 0.5 * (t + q); w = torch.ones_like(t)
    elif mode.startswith("aff"):
        # frozen-FM feature affinity propagation of teacher probabilities (structure prior, not a mask prior)
        with torch.no_grad():
            f = affnet(xw_cur)  # [B,C,g,g], L2-normalised
            B_, C_, g_, _ = f.shape
            ff = f.flatten(2)  # [B,C,N]
            Aff = torch.softmax(torch.einsum("bcn,bcm->bnm", ff, ff) / args.aff_tau, -1)  # [B,N,N]
            tl = F.adaptive_avg_pool2d(t, g_).flatten(1).unsqueeze(-1)  # [B,N,1]
            for _ in range(args.aff_iter):
                tl = torch.bmm(Aff, tl)
            tp = F.interpolate(tl.view(B_, 1, g_, g_), size=t.shape[-1], mode="bilinear", align_corners=False)
        tr = (1 - args.aff_alpha) * t + args.aff_alpha * tp
        y = (tr >= 0.5).float()
        w = (torch.maximum(tr, 1 - tr) >= args.conf).float()
        if mode == "aff_soft":
            y = tr; w = torch.ones_like(tr)
        STATS["aff_flip"] = float(((tr >= 0.5) != (t >= 0.5)).float().mean())
    elif mode.startswith("tri"):
        # label-free per-image triangulation over candidates {t, q, SAMUS(box t), SAMUS(box q)}
        emb = pfm.embed(xw_cur)
        bt, vt = mask_to_box(largest_cc((t >= 0.5).float()) > 0, pad=args.box_pad)
        bq, vq = mask_to_box(largest_cc((q >= 0.5).float()) > 0, pad=args.box_pad)
        ft = pfm.decode(emb, bt)[0]; fq = pfm.decode(emb, bq)[0]
        P = torch.stack([t, q, ft, fq])  # [K,B,1,H,W]
        g = triangulate(P)  # [K,B] estimated quality in (0,1]
        if mode == "tri_fuse":
            wk = torch.softmax(args.tau * g.clamp_min(1e-4).log(), 0).view(4, -1, 1, 1, 1)
            y = (wk * P).sum(0)
            w = torch.ones_like(t)
        elif mode == "tri_rescue":  # keep the teacher unless it is estimated to have failed
            wk = torch.softmax(args.tau * g.clamp_min(1e-4).log(), 0).view(4, -1, 1, 1, 1)
            fused = (wk * P).sum(0)
            bad = (g[0] < args.rel_thr).float().view(-1, 1, 1, 1)
            y = bad * fused + (1 - bad) * (t >= 0.5).float()
            w = bad + (1 - bad) * (torch.maximum(t, 1 - t) >= args.conf).float()
        elif mode == "tri_weight":  # teacher labels, image weight = estimated teacher quality (no label replacement)
            y = (t >= 0.5).float()
            w = (torch.maximum(t, 1 - t) >= args.conf).float() * g[0].view(-1, 1, 1, 1).pow(args.tau)
        STATS["tri_g_t"] = float(g[0].mean()); STATS["tri_g_q"] = float(g[1].mean())
    elif mode.startswith("pl_"):
        # prompt-in-the-loop: the teacher proposes a box, the frozen promptable FM segments inside it
        tb = largest_cc((t >= 0.5).float())
        boxes, valid = mask_to_box(tb > 0, pad=args.box_pad)
        if args.njit > 1:  # FM marginalised over jittered prompts (reduces box sensitivity)
            emb = pfm.embed(xw_cur)
            wh = torch.stack([boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]] * 2, 1)
            fbs = []
            for k in range(args.njit):
                jit = boxes if k == 0 else (boxes + 0.1 * wh * torch.randn(boxes.shape, generator=gen).to(dev)).clamp(0, 255)
                fbs.append(pfm.decode(emb, jit)[0])
            fb = torch.stack(fbs).mean(0)
        else:
            fb, iou = pfm(xw_cur, boxes)
        img_w = valid.float().view(-1, 1, 1, 1)
        if mode == "pl_box":
            y = fb
        elif mode == "pl_fuse":
            y = 0.5 * (t + fb)
        elif mode == "pl_boxw":  # image weight = consistency between teacher and FM-in-teacher-box
            fbin = (fb >= 0.5).float()
            inter = (fbin * tb).sum((1, 2, 3)); den = fbin.sum((1, 2, 3)) + tb.sum((1, 2, 3))
            cons = ((2 * inter + 1) / (den + 1)).view(-1, 1, 1, 1)
            img_w = img_w * cons.pow(args.tau)
            y = fb
        elif mode == "pl_hyb":  # consistency-weighted interpolation: trust FM where it agrees with the box owner
            fbin = (fb >= 0.5).float()
            inter = (fbin * tb).sum((1, 2, 3)); den = fbin.sum((1, 2, 3)) + tb.sum((1, 2, 3))
            cons = ((2 * inter + 1) / (den + 1)).view(-1, 1, 1, 1)
            y = cons * fb + (1 - cons) * t
        else:
            raise ValueError(mode)
        w = img_w.expand_as(t).contiguous()
    else:
        raise ValueError(mode)
    return y, w


# ------------------------------------------------------------------ training
log = []
DG = {"acc": 0.0, "cov": 0.0, "fg_prec": 0.0, "fg_rec": 0.0, "n": 0}
dump_its = [int(v) for v in args.dump.split(",") if v]
DUMP = {}
if dump_its:
    tgt_all = GPUData(T, np.arange(len(T["img"])), dev)


def dump_snapshot(it):
    for nm, net in [("s", model), ("e", teacher)]:
        pr = predict(net, tgt_all.img, tta=True)
        DUMP[f"{nm}{it}"] = (F.avg_pool2d(pr, 2) * 255).round().byte().cpu().numpy()[:, 0]
        DUMP[f"{nm}{it}_srcval"] = np.float32(evaluate(net, src_va)[0])


t0 = time.time()
for it in range(1, args.iters + 1):
    js = src_tr.batch(args.bs, gen)
    th = rand_affine(args.bs, dev, gen)
    xs = warp(src_tr.img[js], th); ys = (warp(src_tr.mask[js], th, "nearest") > 0.5).float()
    xs = photometric(xs, gen, 0.5 * args.strong)
    use_t = args.mode != "source" and it > args.warmup
    with torch.autocast("cuda", dtype=torch.float16):
        ls = model(xs)
    ls = ls.float()
    loss_s = weighted_bce(ls, ys) + soft_dice_loss(ls, ys)
    loss_t = torch.zeros((), device=dev)
    stats = {}
    if use_t:
        jt = tgt_ad.batch(args.bs, gen)
        th = rand_affine(args.bs, dev, gen)
        xw = warp(tgt_ad.img[jt], th)
        qs = [warp(tgt_ad.priors[f][jt], th) for f in fms]
        with torch.no_grad():
            tprob = torch.sigmoid(teacher(xw).float())
        if args.mode == "dacs":
            # ClassMix (binary): paste source lesion pixels onto target; labels mixed with teacher pseudo-labels.
            ytp = (tprob >= 0.5).float(); wt = (torch.maximum(tprob, 1 - tprob) >= args.conf).float()
            m = ys
            xin = m * xs.detach() + (1 - m) * photometric(xw, gen, args.strong)
            y = m * ys + (1 - m) * ytp; w = m + (1 - m) * wt
            if args.pl_diag:
                g = (warp(tgt_ad.mask[jt], th, "nearest") > 0.5).float()
                sel = wt; cor = (ytp == g).float(); fgsel = sel * ytp
                DG["acc"] += float((sel * cor).sum() / sel.sum().clamp_min(1)); DG["cov"] += float(sel.mean())
                DG["fg_prec"] += float((fgsel * g).sum() / fgsel.sum().clamp_min(1))
                DG["fg_rec"] += float((sel * ytp * g).sum() / g.sum().clamp_min(1)); DG["n"] += 1
        else:
            xw_cur = xw
            y, w = pseudo(tprob, qs, args.mode)
            xin = photometric(xw, gen, args.strong)
            if args.mix:  # ClassMix on top of any pseudo-label rule (mode=mt + --mix == DACS)
                m = ys
                xin = m * xs.detach() + (1 - m) * xin
                y = m * ys + (1 - m) * y; w = m + (1 - m) * w
            elif args.cutout:
                xin = cutout(xin, gen)
        with torch.autocast("cuda", dtype=torch.float16):
            lt = model(xin)
        lt = lt.float()
        loss_t = weighted_bce(lt, y, w) + soft_dice_loss(lt, y, w)
        stats = {"cov": float((w > 0).float().mean()), "wmean": float(w.mean()), **STATS}
    loss = loss_s + args.lam * loss_t
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(opt); scaler.update(); sched.step()
    ema_update(teacher, model, args.ema if it > args.warmup else 0.99)
    if it % args.eval_every == 0 or it == args.iters:
        rec = {"it": it, "loss_s": float(loss_s), "loss_t": float(loss_t), **stats,
               "src_val": evaluate(model, src_va)[0], "dev": evaluate(model, tgt_dv)[0],
               "dev_ema": evaluate(teacher, tgt_dv)[0], "min": (time.time() - t0) / 60}
        if args.pl_diag and DG["n"]:
            rec.update({"pl_" + k: v / DG["n"] for k, v in DG.items() if k != "n"}); DG.update(acc=0.0, cov=0.0, fg_prec=0.0, fg_rec=0.0, n=0)
        log.append(rec)
        print(json.dumps(rec), flush=True)
    if it in dump_its:
        dump_snapshot(it)

res = {"args": vars(args), "log": log}
d_dev, pc_dev = evaluate(model, tgt_dv); res["final_dev"] = d_dev; res["dev_pc"] = pc_dev.round(4).tolist()
d_ema, _ = evaluate(teacher, tgt_dv); res["final_dev_ema"] = d_ema
res["final_src_val"] = evaluate(model, src_va)[0]
res["final_adapt_ema"] = evaluate(teacher, tgt_ad)[0]  # transductive (labels of unlabeled adapt split, analysis only)
if args.eval_test:
    d, pc = evaluate(model, tgt_te); res["final_test"] = d; res["test_pc"] = pc.round(4).tolist()
    res["final_test_ema"], pce = evaluate(teacher, tgt_te); res["test_pc_ema"] = pce.round(4).tolist()
    if args.dump_test:
        pr = predict(teacher, tgt_te.img, tta=True)
        np.savez_compressed(args.dump_test, prob=(pr[:, 0] * 255).round().byte().cpu().numpy(), idx=tgt_te.idx)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump(res, f)
if DUMP:
    np.savez_compressed(args.out.replace(".json", ".preds.npz"), **DUMP)
if args.save:
    torch.save({"model": model.state_dict(), "teacher": teacher.state_dict(), "args": vars(args)}, args.save)
print("FINAL dev %.4f dev_ema %.4f" % (d_dev, d_ema), flush=True)
