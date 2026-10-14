# Paper 2 - FIX experiment, Derm7pt part (pre-registered in the D1 protocol,
# "Pre-registration of the fix"). Built from the V2 script; adds LOSS = "aware"
# (censoring-aware partial-label loss) next to LOSS = "std" (unchanged V2 loss).
# Earlier description of V2 follows. Same recipe as the EASY replication job, except
# that the Derm7pt CONCEPT labels used for training are censored with
# probability PI: on naevi, melanoma-side categories are set to "absent";
# on melanomas, naevus-side categories are set to "absent". Diagnosis labels
# and all other lesions are untouched.
# 1) Train the concept encoder on ALL 1011 Derm7pt images (censored labels).
# 2) Save the weights and the censored label table.
# 3) Predict concept probabilities + black-box melanoma probability on EASY.
# Settings come from environment variables set by cell 2:
#   P2_JOBS      e.g. "effb4:0:1:aware,convnext_tiny:0:0:std"  (backbone:seed:pi:loss)
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
JOB_TEXT = os.environ.get("P2_JOBS", "effb4:0:0")
JOBS = []
for j in JOB_TEXT.split(","):
    if j:
        name, sd, pi, lossname = j.split(":")
        assert lossname in ("std", "aware")
        JOBS.append((name, int(sd), float(pi), lossname))
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

MEL_SIDE = {"pigment_network": ["atypical"], "streaks": ["irregular"],
            "pigmentation": ["diffuse irregular", "localized irregular"],
            "dots_and_globules": ["irregular"],
            "regression_structures": ["blue areas", "white areas",
                                      "combinations"],
            "blue_whitish_veil": ["present"],
            "vascular_structures": ["dotted", "linear irregular"]}
NEV_SIDE = {"pigment_network": ["typical"], "streaks": ["regular"],
            "pigmentation": ["diffuse regular", "localized regular"],
            "dots_and_globules": ["regular"]}


def censor(meta, pi):
    """Censored copy of the Derm7pt labels. One uniform draw per
    (image, criterion), seed fixed by pi, identical for every seed."""
    m = meta.copy()
    rng = np.random.default_rng(2026 + int(round(pi * 100)))
    dx = m["diagnosis"].str.lower()
    is_mel = dx.str.contains("melanoma").values
    is_nev = (dx.str.contains("nevus") & ~dx.str.contains("melanoma")).values
    changed = 0
    for c, cats in CONCEPTS.items():
        u = rng.random(len(m))
        drop = np.zeros(len(m), bool)
        if c in MEL_SIDE:
            drop |= is_nev & m[c].isin(MEL_SIDE[c]).values
        if c in NEV_SIDE:
            drop |= is_mel & m[c].isin(NEV_SIDE[c]).values
        drop &= u < pi
        changed += int(drop.sum())
        m.loc[drop, c] = "absent"
        lookup = {k: i for i, k in enumerate(cats)}
        m[c + "__y"] = m[c].map(lookup)
    return m, changed


def candidate_masks(m, aware):
    """Per image, per criterion: 0/1 mask over its categories of the labels the loss
    accepts. std: only the recorded category. aware: when a naevus records "absent", the
    melanoma-side categories of that criterion are also accepted (censoring could have
    hidden them); likewise naevus-side categories for a melanoma recording "absent"."""
    dx = m["diagnosis"].str.lower()
    is_mel = dx.str.contains("melanoma").values
    is_nev = (dx.str.contains("nevus") & ~dx.str.contains("melanoma")).values
    out = []
    for c, cats in CONCEPTS.items():
        M = np.zeros((len(m), len(cats)), np.float32)
        M[np.arange(len(m)), m[c + "__y"].values] = 1
        if aware:
            absent = (m[c].values == "absent")
            for side, who in ((MEL_SIDE, is_nev), (NEV_SIDE, is_mel)):
                if c in side:
                    for cat in side[c]:
                        M[absent & who, cats.index(cat)] = 1
        out.append(M)
    return np.concatenate(out, 1)

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
    def __init__(self, df, tf, masks):
        self.df = df.reset_index(drop=True)
        self.tf = tf
        self.masks = masks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        y = torch.tensor([r[c + "__y"] for c in CONCEPTS],
                         dtype=torch.long)
        x = self.tf(Image.open(r.img_path).convert("RGB"))
        return x, y, torch.tensor(float(r.mel)), torch.from_numpy(self.masks[i])


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


SPLITS = np.cumsum([len(v) for v in CONCEPTS.values()])[:-1]
for backbone, seed, PI, LOSS in JOBS:
    cfg = RECIPE[backbone]
    IMG, BS = (64, 8) if SMOKE else (cfg["IMG"], cfg["BS"])
    tag = "fix_pi%03d_%s_%s_seed%d" % (int(round(PI * 100)), LOSS, backbone, seed)
    pred_csv = os.path.join(OUT_DIR, "easy_preds_" + tag + ".csv")
    tmeta, n_changed = censor(meta, PI)
    print("==", tag, ": censoring pi =", PI, "changed", n_changed,
          "concept labels", flush=True)
    tmeta.drop(columns=["img_path"]).to_csv(
        os.path.join(OUT_DIR, "train_labels_" + tag + ".csv"), index=False)
    masks = candidate_masks(tmeta, LOSS == "aware")
    print("==", tag, ": loss", LOSS, "- labels accepting more than one category:",
          int((masks.sum(1) > len(CONCEPTS)).sum()), "images", flush=True)
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
    dl = DataLoader(TrainDS(tmeta, make_tf(IMG, True), masks), BS,
                    shuffle=True, num_workers=2, drop_last=True)
    print("==", tag, ": training on all", len(tmeta),
          "Derm7pt images,", EPOCHS, "epochs, img", IMG, flush=True)
    log = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        tot = 0.0
        for x, y, m, mk in dl:
            x, y, m, mk = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE), mk.to(DEVICE)
            with torch.autocast(DEVICE, enabled=USE_AMP):
                outs, pm = model(x)
                # partial-label loss: -log(total probability of the accepted categories).
                # With one accepted category this equals cross-entropy (std = V2 loss).
                ce = []
                for j, (o, mj) in enumerate(zip(outs, torch.tensor_split(mk, SPLITS.tolist(), 1))):
                    lp = F.log_softmax(o.float(), 1)
                    ce.append(-torch.logsumexp(lp + torch.log(mj.clamp_min(1e-12)), 1).mean())
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
