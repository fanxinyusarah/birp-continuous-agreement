"""Patient(case)-level splits A(70%)/D(10%)/T(20%) per domain; BUS-BRA stratified by device.
A = source-train or target-unlabeled-adaptation, D = source-val or target-dev, T = target test."""
import json, os, sys
import numpy as np
from collections import defaultdict
DATA = sys.argv[1]
SEED = {"busbra": 1, "busi": 2, "uclm": 3, "udiat": 4, "kvasir": 5, "clinic": 6, "colon": 7, "etis": 8, "cvc300": 9,
        "breast": 10, "oasbud": 11}
out = json.load(open(os.path.join(DATA, "splits.json"))) if os.path.exists(os.path.join(DATA, "splits.json")) else {}
for dom in list(SEED):
    if dom in out or not os.path.exists(os.path.join(DATA, dom + ".npz")):
        continue
    rng = np.random.RandomState(20261018 + SEED[dom])
    z = np.load(os.path.join(DATA, dom + ".npz"))
    case, devc = z["case"], z["device"]
    if dom in ("clinic", "colon", "etis", "cvc300"):  # video frames numbered consecutively: block of 20 frames = pseudo-case
        num = np.array([int(os.path.splitext(os.path.basename(i))[0]) for i in z["id"]])
        blk = 20 if dom in ("clinic", "colon") else 8
        case = np.array(["%s_b%03d" % (dom, n // blk) for n in num])
    sp = {"A": [], "D": [], "T": []}
    for d in sorted(set(devc)):
        groups = defaultdict(list)
        for i in np.where(devc == d)[0]:
            groups[case[i]].append(int(i))
        keys = sorted(groups); rng.shuffle(keys)
        n = sum(len(groups[k]) for k in keys); acc = 0
        for k in keys:
            f = acc / n
            sp["A" if f < 0.7 else ("D" if f < 0.8 else "T")].extend(groups[k]); acc += len(groups[k])
    for k in sp:
        sp[k] = sorted(sp[k])
    ca = {k: set(case[sp[k]]) for k in sp}
    assert not (ca["A"] & ca["D"]) and not (ca["A"] & ca["T"]) and not (ca["D"] & ca["T"])
    out[dom] = sp
    print(dom, {k: (len(v), len(ca[k])) for k, v in sp.items()})
json.dump(out, open(os.path.join(DATA, "splits.json"), "w"))
