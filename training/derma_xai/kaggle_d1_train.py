# D1 protocol, Step 3 (Section 5): train M1/M2 concept models and M3 black box
# on Derma-XAI diagnosis-aware labels (lesion-disjoint split), then save
# predictions on the 196 test images. Run on Kaggle: GPU T4, Internet ON,
# input dataset "kmader/skin-cancer-mnist-ham10000".
# Expects /kaggle/working/d1_labels/{train_labels.csv,test_images.csv}
# (written by cell 1 of the notebook).
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
            y = torch.zeros(1)
        elif self.target == "concepts":
            y = torch.tensor(r[ALL].values.astype(np.float32))
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
    lossf = nn.BCEWithLogitsLoss()
    dl = DataLoader(DS(tr, T_TRAIN, target), batch_size=32, shuffle=True,
                    num_workers=2, drop_last=True)
    dv = DataLoader(DS(va, T_EVAL, target), batch_size=64, num_workers=2)
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        m.train()
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(m(x.to(dev)), y.to(dev))
            loss.backward(); opt.step()
        m.eval(); vl, n = 0.0, 0
        with torch.no_grad():
            for x, y in dv:
                vl += lossf(m(x.to(dev)), y.to(dev)).item() * len(x); n += len(x)
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


runs = [("effb0", s) for s in SEEDS] + [("convnext_tiny", SEEDS[0])]
summary = []
for name, seed in runs:
    t0 = time.time()
    mc, lc = fit(name, "concepts", seed)
    pc_tr = predict(mc, train, "concepts")
    pc_te = predict(mc, test, "concepts")
    mb, lb = fit(name, "diagnosis", seed)
    pb_te = predict(mb, test, "diagnosis")[:, 0]
    tag = "%s_seed%d" % (name, seed)
    pd.DataFrame(pc_tr, columns=ALL).assign(
        image_id=train.image_id.values, split=train.split.values,
        y=train.benign_malignant.values).to_csv(
        os.path.join(OUT, "concepts_train_%s.csv" % tag), index=False)
    pd.DataFrame(pc_te, columns=ALL).assign(
        image_id=test.image_id.values, y=test.benign_malignant.values,
        p_mel_blackbox=pb_te).to_csv(
        os.path.join(OUT, "test_%s.csv" % tag), index=False)
    summary.append({"run": tag, "val_concept_loss": lc, "val_dx_loss": lb,
                    "minutes": (time.time() - t0) / 60})
    print(summary[-1], flush=True)

pd.DataFrame(summary).to_csv(os.path.join(OUT, "run_summary.csv"), index=False)
zp = "/kaggle/working/d1_models.zip" if not SMOKE else os.path.join(OUT, "smoke.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)):
        z.write(os.path.join(OUT, f), f)
print("DONE ->", zp)
