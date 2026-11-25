"""Pack breast-ultrasound datasets into single npz files (256x256 grayscale + binary mask + metadata)."""
import csv, json, os, re, sys, glob
import numpy as np
from PIL import Image

S = 256
OUT = sys.argv[1]
WANT = sys.argv[2].split(",") if len(sys.argv) > 2 else ["busbra", "busi", "uclm", "udiat"]
os.makedirs(OUT, exist_ok=True)

def load_img(p):
    im = Image.open(p).convert("L")
    return np.asarray(im.resize((S, S), Image.BILINEAR), dtype=np.uint8), im.size

def load_mask(paths):
    m = None
    for p in paths:
        a = np.asarray(Image.open(p).convert("L").resize((S, S), Image.NEAREST)) > 0
        m = a if m is None else (m | a)
    return m.astype(np.uint8)

def save(name, imgs, masks, meta):
    np.savez_compressed(os.path.join(OUT, name + ".npz"), img=np.stack(imgs), mask=np.stack(masks),
                        **{k: np.array(v) for k, v in meta.items()})
    print(name, len(imgs), "fg frac %.3f" % np.stack(masks).mean())


def do_busbra():
    # BUS-BRA: patient = Case, device known
    R = "/tmp/xl_prep/raw/BUSBRA"
    imgs, masks, meta = [], [], {"id": [], "case": [], "device": [], "path": [], "size": []}
    for row in csv.DictReader(open(R + "/bus_data.csv")):
        i, sz = load_img(f"{R}/Images/{row['ID']}.png")
        imgs.append(i); masks.append(load_mask([f"{R}/Masks/{row['ID'].replace('bus', 'mask')}.png"]))
        meta["id"].append(row["ID"]); meta["case"].append("bra" + row["Case"]); meta["device"].append(row["Device"])
        meta["path"].append(row["Pathology"]); meta["size"].append(sz)
    save("busbra", imgs, masks, meta)

def do_busi():
    # BUSI: benign + malignant (normal excluded, as in prior protocol); no patient ids -> image-level
    R = "/tmp/xl_prep/raw/Dataset_BUSI_with_GT"
    imgs, masks, meta = [], [], {"id": [], "case": [], "device": [], "path": [], "size": []}
    for cls in ["benign", "malignant"]:
        for p in sorted(glob.glob(f"{R}/{cls}/*).png"), key=lambda s: int(re.findall(r"\((\d+)\)", s)[0])):
            stem = p[:-4]
            mps = sorted(glob.glob(glob.escape(stem) + "_mask*.png"))
            i, sz = load_img(p)
            imgs.append(i); masks.append(load_mask(mps))
            n = os.path.basename(stem)
            meta["id"].append(n); meta["case"].append("busi_" + n); meta["device"].append("busi"); meta["path"].append(cls); meta["size"].append(sz)
    save("busi", imgs, masks, meta)

def do_uclm():
    # BUS-UCLM (lesion images available here); patient = 4-letter prefix
    R = "/tmp/xl_prep/raw/UCLM"
    imgs, masks, meta = [], [], {"id": [], "case": [], "device": [], "path": [], "size": []}
    for p in sorted(glob.glob(R + "/images/*.png")):
        n = os.path.basename(p)[:-4]
        if os.path.getsize(p) == 0 or os.path.getsize(f"{R}/masks/{n}.png") == 0:
            print("skip empty file", n); continue
        i, sz = load_img(p)
        imgs.append(i); masks.append(load_mask([f"{R}/masks/{n}.png"]))
        meta["id"].append(n); meta["case"].append("uclm_" + n.split("_")[0]); meta["device"].append("uclm"); meta["path"].append("na"); meta["size"].append(sz)
    save("uclm", imgs, masks, meta)

def do_breast():
    # BrEaST (TCIA Breast-Lesions-USG, CC-BY 4.0): caseNNN.png + caseNNN_tumor.png (1 scan = 1 patient).
    # 4 of 256 cases are normal (no _tumor.png) -> excluded, same convention as do_busi() excluding "normal".
    R = "/scratch/xl5874/Eve/birp/raw/BrEaST/BrEaST-Lesions_USG-images_and_masks"
    imgs, masks, meta = [], [], {"id": [], "case": [], "device": [], "path": [], "size": []}
    for p in sorted(glob.glob(R + "/case*.png")):
        n = os.path.basename(p)[:-4]
        if "_" in n:  # skip *_tumor.png / *_otherN.png, handled via the base case below
            continue
        mp = f"{R}/{n}_tumor.png"
        if not os.path.exists(mp):
            print("skip normal (no tumor mask)", n); continue
        i, sz = load_img(p)
        imgs.append(i); masks.append(load_mask([mp]))
        meta["id"].append(n); meta["case"].append("bre_" + n); meta["device"].append("breast"); meta["path"].append("na"); meta["size"].append(sz)
    save("breast", imgs, masks, meta)

def do_udiat():
    # UDIAT Dataset B
    R = "/tmp/xl_prep/raw/UDIAT_Dataset_B"
    imgs, masks, meta = [], [], {"id": [], "case": [], "device": [], "path": [], "size": []}
    for p in sorted(glob.glob(R + "/original/*.png")):
        n = os.path.basename(p)[:-4]
        i, sz = load_img(p)
        imgs.append(i); masks.append(load_mask([f"{R}/GT/{n}.png"]))
        meta["id"].append(n); meta["case"].append("udiat_" + n); meta["device"].append("udiat"); meta["path"].append("na"); meta["size"].append(sz)
    save("udiat", imgs, masks, meta)

for n in WANT:
    globals()['do_' + n]()
