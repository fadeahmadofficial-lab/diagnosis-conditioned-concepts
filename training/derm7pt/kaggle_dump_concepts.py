# =============================================================================
# Paper 2 — Kaggle GPU job: train the 7-point concept encoder under 5-fold CV
# and DUMP out-of-fold concept probabilities at several checkpoints.
#
# This script does NOT analyse anything. It only produces CSVs of per-image
# concept probabilities; all Paper 2 analysis (nulls, DCR, fragility, regions,
# bootstrap) runs later on CPU from these CSVs (analysis_p2.py).
#
# Kaggle setup (Notebook -> Settings):
#   * Accelerator: GPU (T4 or P100)
#   * Internet: ON   (needed once to download ImageNet weights for EfficientNet-B4)
#   * Add Data: your dataset  fadehafezahmad/derm7pt
# Then paste this whole file into one cell (or upload it) and "Save & Run All".
# Outputs appear in /kaggle/working/ — download them or "Save Version" and add
# the output as a dataset.
#
# Time budget (ESTIMATE, not measured on your account): ~1.5–2 h per fold at
# 40 epochs on a T4. Kaggle sessions stop at ~12 h and the weekly GPU quota is
# ~30 h, so run 1–2 folds per session by editing FOLDS_TO_RUN.
# =============================================================================
import os, glob, time, json, random, math
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ----------------------------- CONFIG ---------------------------------------
SMOKE        = os.environ.get("P2_SMOKE", "0") == "1"   # local test only
INPUT_ROOT   = os.environ.get("P2_INPUT", "/kaggle/input")
OUT_DIR      = os.environ.get("P2_OUT", "/kaggle/working")
FOLDS_TO_RUN = [0, 1, 2, 3, 4]      # e.g. [0] in session 1, [1] in session 2 ...
SEEDS        = [0]                  # add more seeds later if budget allows
DATA_FRACS   = [1.0]                # optional degradation: [1.0, 0.5, 0.25]
EPOCHS       = 40
CKPT_EPOCHS  = [2, 4, 8, 16, 40]    # OOF dump points = free concept-quality sweep
BACKBONE     = "effb4"              # "effb4" (Paper-2 main runs) or "convnext_tiny" (backbone robustness run)
IMG, BS, LR, WD = 380, 16, 3e-4, 1e-4
if BACKBONE == "convnext_tiny":      # fixed in advance (pre-registered), standard ConvNeXt fine-tuning settings
    IMG, BS, LR, WD = 384, 16, 1e-4, 5e-2
N_FOLDS, FOLD_SEED = 5, 2026        # fold assignment is fixed and saved
if SMOKE:
    EPOCHS, CKPT_EPOCHS, IMG, BS = 2, [1, 2], 64, 8
# ----------------------------------------------------------------------------

CONCEPTS = {  # meta.csv column -> category order used by the head
    "pigment_network":       ["absent", "typical", "atypical"],
    "blue_whitish_veil":     ["absent", "present"],
    "vascular_structures":   ["absent", "arborizing", "comma", "hairpin", "within regression",
                              "wreath", "dotted", "linear irregular"],
    "streaks":               ["absent", "regular", "irregular"],
    "pigmentation":          ["absent", "diffuse regular", "localized regular",
                              "diffuse irregular", "localized irregular"],
    "dots_and_globules":     ["absent", "regular", "irregular"],
    "regression_structures": ["absent", "blue areas", "white areas", "combinations"],
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "| smoke:", SMOKE, "| backbone:", BACKBONE, "| img/bs/lr/wd:", IMG, BS, LR, WD)
if not SMOKE and DEVICE != "cuda":
    raise SystemExit("No GPU. Turn on the GPU accelerator in Kaggle settings.")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ---------------- locate meta.csv and build a case-insensitive image index ---
metas = sorted(glob.glob(os.path.join(INPUT_ROOT, "**", "meta.csv"), recursive=True))
if not metas:
    raise SystemExit(f"meta.csv not found under {INPUT_ROOT}")
META_PATH = metas[0]; print("meta.csv:", META_PATH)
meta = pd.read_csv(META_PATH)
assert len(meta) == 1011, f"expected 1011 rows, got {len(meta)}"

index = {}
for p in glob.glob(os.path.join(INPUT_ROOT, "**", "*"), recursive=True):
    if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        parts = p.replace("\\", "/").split("/")
        index["/".join(parts[-2:]).lower()] = p          # e.g. "nel/nel026.jpg"
paths, missing = [], []
for d in meta["derm"]:
    key = "/".join(d.replace("\\", "/").split("/")[-2:]).lower()
    (paths.append(index[key]) if key in index else missing.append(d))
if missing:
    print("MISSING images (first 10):", missing[:10])
    raise SystemExit(f"{len(missing)} dermoscopy images not found. Check the dataset is attached.")
meta["img_path"] = paths
print("all 1011 dermoscopy images located. example:", paths[0])

# ---------------- labels -----------------------------------------------------
meta["mel"] = meta["diagnosis"].str.contains("melanoma").astype(int)
for c, cats in CONCEPTS.items():
    bad = set(meta[c]) - set(cats)
    assert not bad, f"unexpected categories in {c}: {bad}"
    meta[c + "__y"] = meta[c].map({k: i for i, k in enumerate(cats)})
raw_cols = list(CONCEPTS)
g = meta.groupby(raw_cols)["mel"].agg(["min", "max"])
bad_profiles = set(g[g["min"] != g["max"]].index)
meta["bnd_raw"] = meta[raw_cols].apply(tuple, axis=1).isin(bad_profiles).astype(int)
assert meta["bnd_raw"].sum() == 306, "boundary does not reproduce Napoles (306)"

# fixed, saved fold assignment, stratified by melanoma x boundary
strat = meta["mel"] * 2 + meta["bnd_raw"]
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED)
meta["fold"] = -1
for f, (_, te) in enumerate(skf.split(meta, strat)):
    meta.loc[te, "fold"] = f
os.makedirs(OUT_DIR, exist_ok=True)
meta[["case_num", "fold", "mel", "bnd_raw"]].rename_axis("row").to_csv(
    os.path.join(OUT_DIR, "fold_assignment.csv"))

# ---------------- dataset / model -------------------------------------------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf_train = T.Compose([T.Resize((IMG, IMG)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                      T.RandomRotation(20), T.ColorJitter(0.1, 0.1, 0.1, 0.02),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
tf_eval = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(), T.Normalize(MEAN, STD)])

class DS(Dataset):
    def __init__(s, df, tf): s.df, s.tf = df.reset_index(), tf
    def __len__(s): return len(s.df)
    def __getitem__(s, i):
        r = s.df.iloc[i]
        x = s.tf(Image.open(r.img_path).convert("RGB"))
        y = torch.tensor([r[c + "__y"] for c in CONCEPTS], dtype=torch.long)
        return x, y, torch.tensor(float(r.mel)), int(r["index"])

import torchvision.models as tvm
def build_model():
    try:
        if BACKBONE == "convnext_tiny":
            bb = tvm.convnext_tiny(weights=None if SMOKE else tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            nf = bb.classifier[2].in_features          # classifier = LayerNorm2d, Flatten, Linear
            bb.classifier[2] = nn.Identity()
        else:
            bb = tvm.efficientnet_b0(weights=None) if SMOKE else \
                 tvm.efficientnet_b4(weights=tvm.EfficientNet_B4_Weights.IMAGENET1K_V1)
            nf = bb.classifier[1].in_features
            bb.classifier = nn.Identity()
    except Exception as e:
        raise SystemExit(f"Could not download ImageNet weights ({e}). Turn Internet ON. "
                         "Training from scratch would reproduce the old weak-encoder confound.")
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.bb = bb; s.drop = nn.Dropout(0.3)
            s.heads = nn.ModuleList([nn.Linear(nf, len(v)) for v in CONCEPTS.values()])
            s.mel = nn.Linear(nf, 1)   # black-box melanoma head on the SAME features
        def forward(s, x):
            h = s.drop(s.bb(x))
            return [hd(h) for hd in s.heads], s.mel(h).squeeze(1)
    return Net().to(DEVICE)

@torch.no_grad()
def predict(model, loader):
    model.eval(); rows = []
    for x, _, _, idx in loader:
        with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
            outs, m = model(x.to(DEVICE))
        probs = [F.softmax(o.float(), 1).cpu().numpy() for o in outs]
        pm = torch.sigmoid(m.float()).cpu().numpy()
        for b in range(len(idx)):
            r = {"row": int(idx[b]), "p_mel_bb": float(pm[b])}
            for (c, cats), P in zip(CONCEPTS.items(), probs):
                for k, cat in enumerate(cats):
                    r[f"{c}__{cat}"] = float(P[b, k])
            rows.append(r)
    return pd.DataFrame(rows)

def sanity(df_pred, df_true):
    d = df_pred.merge(df_true, left_on="row", right_index=True)
    out = {"mel_auc_bb": roc_auc_score(d.mel, d.p_mel_bb)}
    aucs = []
    for c, cats in CONCEPTS.items():
        P = d[[f"{c}__{k}" for k in cats]].values; y = d[c + "__y"].values
        present = [k for k in range(len(cats)) if 0 < (y == k).sum() < len(y)]
        try:
            aucs.append(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(len(cats))))
                        if len(present) == len(cats) else
                        np.mean([roc_auc_score(y == k, P[:, k]) for k in present]))
        except ValueError:
            pass
    out["mean_concept_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    return out

# ---------------- main loop --------------------------------------------------
for seed in SEEDS:
    for frac in DATA_FRACS:
        for fold in FOLDS_TO_RUN:
            tag = f"fold{fold}_frac{frac}_seed{seed}" if BACKBONE == "effb4" else f"{BACKBONE}_fold{fold}_frac{frac}_seed{seed}"
            out_csv = os.path.join(OUT_DIR, f"oof_{tag}.csv")
            if os.path.exists(out_csv):
                print("skip (exists):", out_csv); continue
            seed_all(seed * 100 + fold)
            tr = meta[meta.fold != fold]
            if frac < 1.0:   # stratified subsample of the TRAINING fold only
                tr = tr.groupby(strat[tr.index], group_keys=False).apply(
                    lambda g: g.sample(frac=frac, random_state=seed))
            te = meta[meta.fold == fold]
            dl_tr = DataLoader(DS(tr, tf_train), BS, shuffle=True, num_workers=2, drop_last=True)
            dl_te = DataLoader(DS(te, tf_eval), BS * 2, shuffle=False, num_workers=2)
            model = build_model()
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            scaler = torch.amp.GradScaler(enabled=(DEVICE == "cuda"))
            dumps, log = [], []
            print(f"== {tag}: train {len(tr)} / held-out {len(te)}")
            for ep in range(1, EPOCHS + 1):
                model.train(); t0 = time.time(); tot = 0
                for x, y, m, _ in dl_tr:
                    x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
                    with torch.autocast(DEVICE, enabled=(DEVICE == "cuda")):
                        outs, pm = model(x)
                        loss = sum(F.cross_entropy(o, y[:, j]) for j, o in enumerate(outs)) \
                               + F.binary_cross_entropy_with_logits(pm, m)
                    opt.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                    tot += loss.item()
                sched.step()
                msg = {"epoch": ep, "loss": tot / max(len(dl_tr), 1), "sec": round(time.time() - t0, 1)}
                if ep in CKPT_EPOCHS:
                    P = predict(model, dl_te)
                    P.insert(0, "epoch", ep); P.insert(0, "frac", frac)
                    P.insert(0, "seed", seed); P.insert(0, "fold", fold); P.insert(0, "backbone", BACKBONE)
                    dumps.append(P); msg.update(sanity(P, meta))
                log.append(msg); print(msg, flush=True)
            pd.concat(dumps).to_csv(out_csv, index=False)
            pd.DataFrame(log).to_csv(os.path.join(OUT_DIR, f"log_{tag}.csv"), index=False)
            print("wrote", out_csv)
            del model; torch.cuda.empty_cache() if DEVICE == "cuda" else None

print("DONE. Paper-1 reference at full training: melanoma AUC ~0.887-0.889, mean concept AUC ~0.817.")
