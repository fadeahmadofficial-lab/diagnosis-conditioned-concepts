# Paper 2 - external replication job (see PAPER2_PREREGISTRATION.md).
# 1) Train the concept encoder on ALL 1011 Derm7pt images
#    (same recipe as the confirmation runs).
# 2) Save the weights.
# 3) Predict concept probabilities + black-box melanoma
#    probability on the 248 ISIC "EASY" images (no fine-tuning).
# Settings come from environment variables set by cell 2:
#   P2_JOBS      e.g. "effb4:0,effb4:1"
#   P2_EASY_DIR  folder with the EASY ISIC_*.jpg images
#   P2_INPUT     Kaggle input root (default /kaggle/input)
#   P2_OUT       output folder (default /kaggle/working)
#   P2_SMOKE=1   tiny local test only
import os
import glob
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.models as tvm
from sklearn.metrics import roc_auc_score

SMOKE = os.environ.get("P2_SMOKE", "0") == "1"
INPUT_ROOT = os.environ.get("P2_INPUT", "/kaggle/input")
OUT_DIR = os.environ.get("P2_OUT", "/kaggle/working")
EASY_DIR = os.environ.get("P2_EASY_DIR", "/tmp/easy_images")
JOB_TEXT = os.environ.get("P2_JOBS", "effb4:0")
JOBS = []
for j in JOB_TEXT.split(","):
    if j:
        name, sd = j.split(":")
        JOBS.append((name, int(sd)))
EPOCHS = 2 if SMOKE else 40
# identical to the confirmation runs (kaggle_dump_concepts.py)
RECIPE = {
    "effb4": dict(IMG=380, BS=16, LR=3e-4, WD=1e-4),
    "convnext_tiny": dict(IMG=384, BS=16, LR=1e-4, WD=5e-2),
}
CONCEPTS = {
    "pigment_network": ["absent", "typical", "atypical"],
    "blue_whitish_veil": ["absent", "present"],
    "vascular_structures": [
        "absent", "arborizing", "comma", "hairpin",
        "within regression", "wreath", "dotted", "linear irregular"],
    "streaks": ["absent", "regular", "irregular"],
    "pigmentation": [
        "absent", "diffuse regular", "localized regular",
        "diffuse irregular", "localized irregular"],
    "dots_and_globules": ["absent", "regular", "irregular"],
    "regression_structures": [
        "absent", "blue areas", "white areas", "combinations"],
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
if not SMOKE and DEVICE != "cuda":
    raise SystemExit("No GPU visible to this job.")
os.makedirs(os.path.join(OUT_DIR, "weights"), exist_ok=True)


def key_of(path):
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]).lower()


# ---------------- Derm7pt (training data) ----------------
pattern = os.path.join(INPUT_ROOT, "**", "meta.csv")
meta = None
for mp in sorted(glob.glob(pattern, recursive=True)):
    d = pd.read_csv(mp)
    if len(d) == 1011 and "derm" in d.columns:
        meta = d
        print("Derm7pt meta.csv:", mp, flush=True)
        break
if meta is None:
    raise SystemExit("Derm7pt meta.csv (1011 rows) not found under "
                     + INPUT_ROOT + " -> add the derm7pt dataset")
index = {}
all_files = glob.glob(os.path.join(INPUT_ROOT, "**", "*"), recursive=True)
for p in all_files:
    if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        index[key_of(p)] = p
paths = [index.get(key_of(d)) for d in meta["derm"]]
missing = [d for d, p in zip(meta["derm"], paths) if p is None]
if missing:
    raise SystemExit(str(len(missing)) + " Derm7pt images missing")
meta["img_path"] = paths
meta["mel"] = meta["diagnosis"].str.contains("melanoma").astype(int)
for c, cats in CONCEPTS.items():
    bad = set(meta[c]) - set(cats)
    assert not bad, "unexpected categories in " + c
    lookup = {k: i for i, k in enumerate(cats)}
    meta[c + "__y"] = meta[c].map(lookup)
n_mel = int(meta.mel.sum())
print("Derm7pt:", len(meta), "images,", n_mel, "melanomas", flush=True)

# ---------------- EASY (external test images) ----------------
pattern = os.path.join(EASY_DIR, "**", "ISIC_*.jp*g")
easy_paths = sorted(glob.glob(pattern, recursive=True))
easy_ids = [os.path.splitext(os.path.basename(p))[0] for p in easy_paths]
if not SMOKE and len(set(easy_ids)) != 248:
    raise SystemExit("expected 248 EASY images, found "
                     + str(len(set(easy_ids))))
easy = pd.DataFrame({"isic_id": easy_ids, "img_path": easy_paths})
easy = easy.drop_duplicates("isic_id")
# Quick gate check only (Benign/Malignant from ISIC). The official
# gate is computed later on CPU from the repository metadata.
easy_labels = None
LABEL_COLS = ["diagnosis_1", "benign_malignant"]
pattern = os.path.join(EASY_DIR, "**", "*.csv")
for mp in glob.glob(pattern, recursive=True):
    md = pd.read_csv(mp)
    md.columns = [c.lower() for c in md.columns]
    cols = [c for c in LABEL_COLS if c in md.columns]
    if "isic_id" in md.columns and cols:
        md = md.drop_duplicates("isic_id").set_index("isic_id")
        lab = md[cols[0]].astype(str).str.lower()
        easy_labels = lab.map({"malignant": 1, "benign": 0})
        break
label_note = "found" if easy_labels is not None else "not found"
print("EASY:", len(easy), "images; labels for gate:", label_note,
      flush=True)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def make_tf(img, train):
    steps = [T.Resize((img, img))]
    if train:
        steps += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                  T.RandomRotation(20),
                  T.ColorJitter(0.1, 0.1, 0.1, 0.02)]
    steps += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(steps)


class TrainDS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        y = torch.tensor([r[c + "__y"] for c in CONCEPTS],
                         dtype=torch.long)
        x = self.tf(Image.open(r.img_path).convert("RGB"))
        return x, y, torch.tensor(float(r.mel))


class PredDS(Dataset):
    def __init__(self, df, tf):
        self.df = df.reset_index(drop=True)
        self.tf = tf

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(Image.open(r.img_path).convert("RGB")), i


class Net(nn.Module):
    def __init__(self, bb, nf):
        super().__init__()
        self.bb = bb
        self.drop = nn.Dropout(0.3)
        self.heads = nn.ModuleList(
            [nn.Linear(nf, len(v)) for v in CONCEPTS.values()])
        self.mel = nn.Linear(nf, 1)

    def forward(self, x):
        h = self.drop(self.bb(x))
        return [hd(h) for hd in self.heads], self.mel(h).squeeze(1)


def build(backbone):
    try:
        if backbone == "convnext_tiny":
            w = None if SMOKE else tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            bb = tvm.convnext_tiny(weights=w)
            nf = bb.classifier[2].in_features
            bb.classifier[2] = nn.Identity()
        elif SMOKE:
            bb = tvm.efficientnet_b0(weights=None)
            nf = bb.classifier[1].in_features
            bb.classifier = nn.Identity()
        else:
            w = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1
            bb = tvm.efficientnet_b4(weights=w)
            nf = bb.classifier[1].in_features
            bb.classifier = nn.Identity()
    except Exception as e:
        raise SystemExit("Could not get ImageNet weights: " + str(e)
                         + " -> turn Internet ON")
    return Net(bb, nf).to(DEVICE)


@torch.no_grad()
def predict(model, df, img, bs):
    model.eval()
    rows = []
    loader = DataLoader(PredDS(df, make_tf(img, False)), bs,
                        shuffle=False, num_workers=2)
    for x, idx in loader:
        with torch.autocast(DEVICE, enabled=USE_AMP):
            outs, m = model(x.to(DEVICE))
        probs = [F.softmax(o.float(), 1).cpu().numpy() for o in outs]
        pm = torch.sigmoid(m.float()).cpu().numpy()
        for b in range(len(idx)):
            r = {"isic_id": df.iloc[int(idx[b])].isic_id,
                 "p_mel_bb": float(pm[b])}
            for (c, cats), P in zip(CONCEPTS.items(), probs):
                for k, cat in enumerate(cats):
                    r[c + "__" + cat] = float(P[b, k])
            rows.append(r)
    return pd.DataFrame(rows)


for backbone, seed in JOBS:
    cfg = RECIPE[backbone]
    IMG, BS = (64, 8) if SMOKE else (cfg["IMG"], cfg["BS"])
    tag = backbone + "_seed" + str(seed)
    pred_csv = os.path.join(OUT_DIR, "easy_preds_" + tag + ".csv")
    if os.path.exists(pred_csv):
        print("skip (exists):", pred_csv, flush=True)
        continue
    # full-data runs have no fold, so the seed is used directly
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build(backbone)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["LR"],
                            weight_decay=cfg["WD"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    dl = DataLoader(TrainDS(meta, make_tf(IMG, True)), BS,
                    shuffle=True, num_workers=2, drop_last=True)
    print("==", tag, ": training on all", len(meta),
          "Derm7pt images,", EPOCHS, "epochs, img", IMG, flush=True)
    log = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for x, y, m in dl:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            with torch.autocast(DEVICE, enabled=USE_AMP):
                outs, pm = model(x)
                ce = [F.cross_entropy(o, y[:, j])
                      for j, o in enumerate(outs)]
                bce = F.binary_cross_entropy_with_logits(pm, m)
                loss = sum(ce) + bce
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
        sched.step()
        msg = {"epoch": ep, "loss": round(tot / max(len(dl), 1), 4),
               "sec": round(time.time() - t0, 1)}
        log.append(msg)
        if ep in (1, EPOCHS) or ep % 10 == 0:
            print("[" + tag + "]", msg, flush=True)
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.half() if v.is_floating_point() else v
    torch.save(state, os.path.join(OUT_DIR, "weights", tag + ".pt"))
    P = predict(model, easy, IMG, BS * 2)
    P.insert(0, "seed", seed)
    P.insert(0, "backbone", backbone)
    P.to_csv(pred_csv, index=False)
    log_csv = os.path.join(OUT_DIR, "log_full_" + tag + ".csv")
    pd.DataFrame(log).to_csv(log_csv, index=False)
    if easy_labels is not None:
        y = P.isic_id.map(easy_labels)
        ok = y.notna()
        if ok.sum() > 10 and y[ok].nunique() == 2:
            auc = roc_auc_score(y[ok], P.p_mel_bb[ok])
            print("[" + tag + "] EASY gate: black-box melanoma AUC =",
                  round(auc, 3), "(n =", int(ok.sum()),
                  "; pre-registered gate >= 0.75)", flush=True)
    print("[" + tag + "] wrote", pred_csv, "and weights", flush=True)
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
print("DONE", flush=True)
