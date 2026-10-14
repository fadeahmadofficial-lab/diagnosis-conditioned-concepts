# Paper 2 - FIX experiment, Derma-XAI part (pre-registered in the D1 protocol,
# "Pre-registration of the fix"). Same data, split, recipe and seeds as D1 Step 3 /
# the EASY run, but the CONCEPT model is trained with a censoring-aware masked loss:
# on naevi the melanoma-side characteristics are "unknown" (masked out of the loss), on
# melanomas the naevus-side ones are. Only concept models are trained; the diagnosis
# (black-box) models are reused from the earlier, exactly reproducible run.
# Predictions are written for the 196 test images and the 248 EASY images.
import json
import os
import random
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LAB = os.environ.get("D1_LABELS", "/kaggle/working/d1_labels")
IMG_ROOT = os.environ.get("D1_IMAGES", "/kaggle/input")
OUT = os.environ.get("D1_OUT", "/kaggle/working/d1_out")
EASY_DIR = os.environ.get("D1_EASY", "/kaggle/working/easy_images")
WDIR = os.environ.get("D1_WEIGHTS_DIR", "/kaggle/working/weights")
os.makedirs(WDIR, exist_ok=True)
SEEDS = [int(s) for s in os.environ.get("D1_SEEDS", "0,1,2,3,4").split(",")]
EPOCHS = int(os.environ.get("D1_EPOCHS", "30"))
SMOKE = os.environ.get("D1_SMOKE", "0") == "1"
WEIGHTS = None if os.environ.get("D1_NOWEIGHTS") == "1" else "IMAGENET1K_V1"
os.makedirs(OUT, exist_ok=True)

MEL = ["TRBL", "ESA", "GP", "PV", "PRL", "BDG", "WLSA", "PLR", "PES", "PIF"]
NEV = ["OPC", "SPC", "MVP", "PRLC", "PLF", "PDES", "APC", "MS"]
ALL = MEL + NEV
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

train = pd.read_csv(os.path.join(LAB, "train_labels.csv"))
test = pd.read_csv(os.path.join(LAB, "test_images.csv"))
paths = {}
for root, _, files in os.walk(IMG_ROOT, followlinks=True):
    for f in files:
        if f.lower().endswith(".jpg") and f.startswith("ISIC_"):
            paths.setdefault(f[:-4], os.path.join(root, f))
need = set(train.image_id) | set(test.image_id)
missing = sorted(need - set(paths))
print("images found: %d of %d needed" % (len(need) - len(missing), len(need)))
if missing:
    raise SystemExit("missing images, e.g. %s: attach the HAM10000 dataset"
                     % missing[:3])
easy_paths = {}
for root, _, files in os.walk(EASY_DIR):
    for f in files:
        if f.startswith("ISIC_") and f.lower().endswith((".jpg", ".jpeg")):
            easy_paths.setdefault(f.split(".")[0], os.path.join(root, f))
print("EASY images found:", len(easy_paths))
if len(easy_paths) != 248:
    raise SystemExit("expected 248 EASY images: run cell 2 first")
for k, v in easy_paths.items():
    paths.setdefault(k, v)
easy = pd.DataFrame({"image_id": sorted(easy_paths)})
if SMOKE:
    train = pd.concat([train[train.split == "train"].head(40),
                       train[train.split == "val"].head(16)])
    test = test.head(16)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
T_TRAIN = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
    transforms.ToTensor(), norm])
T_EVAL = transforms.Compose([transforms.Resize((224, 224)),
                             transforms.ToTensor(), norm])


class DS(Dataset):
    def __init__(self, df, tf, target):
        self.df, self.tf, self.target = df.reset_index(drop=True), tf, target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = self.tf(Image.open(paths[r.image_id]).convert("RGB"))
        if self.target == "none":
            return x, torch.zeros(1)
        elif self.target == "concepts":
            y = torch.tensor(r[ALL].values.astype(np.float32))
            mel = float(r.benign_malignant) == 1.0
            mk = torch.tensor([1.0 if (c in MEL) == mel else 0.0 for c in ALL])
            return x, y, mk
        else:
            y = torch.tensor([float(r.benign_malignant)])
        return x, y


def backbone(name, n_out):
    if name == "effb0":
        m = models.efficientnet_b0(weights=WEIGHTS)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_out)
    elif name == "convnext_tiny":
        m = models.convnext_tiny(weights=WEIGHTS)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_out)
    else:
        raise ValueError(name)
    return m


def predict(m, df, target):
    dl = DataLoader(DS(df, T_EVAL, "none"), batch_size=64, num_workers=2)
    m.eval()
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.sigmoid(m(x.to(dev))).cpu().numpy())
    return np.concatenate(out)


def fit(name, target, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr, va = train[train.split == "train"], train[train.split == "val"]
    n_out = len(ALL) if target == "concepts" else 1
    m = backbone(name, n_out).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def lossf(z, y, mk):
        return (bce(z, y) * mk).sum() / mk.sum().clamp_min(1.0)
    dl = DataLoader(DS(tr, T_TRAIN, target), batch_size=32, shuffle=True,
                    num_workers=2, drop_last=True)
    dv = DataLoader(DS(va, T_EVAL, target), batch_size=64, num_workers=2)
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        m.train()
        for x, y, mk in dl:
            opt.zero_grad()
            loss = lossf(m(x.to(dev)), y.to(dev), mk.to(dev))
            loss.backward(); opt.step()
        m.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for x, y, mk in dv:
                vl += lossf(m(x.to(dev)), y.to(dev), mk.to(dev)).item() * len(x); n += len(x)
        vl /= n
        print("  %s %s seed %d epoch %d val loss %.4f" % (name, target, seed, ep, vl),
              flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= 5:
                break
    m.load_state_dict(best_state)
    return m, best


RUNS = [("effb0", s) for s in SEEDS] + [("convnext_tiny", SEEDS[0])]
if os.environ.get("D1_RUNS"):
    RUNS = [(a, int(b)) for a, b in (r.split(":") for r in os.environ["D1_RUNS"].split(","))]
summary = []
for name, seed in RUNS:
    t0 = time.time()
    tag = "masked_%s_seed%d" % (name, seed)
    if os.path.exists(os.path.join(OUT, "easy_%s.csv" % tag)):
        print("skip (exists):", tag, flush=True)
        continue
    print("==", tag, ": masked concept training", flush=True)
    mc, lc = fit(name, "concepts", seed)
    pd.DataFrame(predict(mc, test, "concepts"), columns=ALL).assign(
        image_id=test.image_id.values, y=test.benign_malignant.values).to_csv(
        os.path.join(OUT, "test_%s.csv" % tag), index=False)
    pd.DataFrame(predict(mc, easy, "concepts"), columns=ALL).assign(
        image_id=easy.image_id.values).to_csv(os.path.join(OUT, "easy_%s.csv" % tag), index=False)
    torch.save(mc.state_dict(), os.path.join(WDIR, "concepts_%s.pt" % tag))
    summary.append({"run": tag, "val_masked_concept_loss": lc, "minutes": (time.time() - t0) / 60})
    print("[" + tag + "] wrote predictions", summary[-1], flush=True)
pd.DataFrame(summary).to_csv(os.path.join(OUT, "run_summary_masked.csv"), index=False)
print("DONE", flush=True)
