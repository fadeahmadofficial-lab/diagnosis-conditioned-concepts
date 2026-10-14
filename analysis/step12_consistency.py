# Paper 2: concept-only diagnosis accuracy ("consistency") per label set. DESCRIPTIVE.
# Written 2026-10-12 before computing. Quantifies the Parisini (0.999) / Napoles (Derm7pt+)
# point: how well do the concept LABELS alone recover the diagnosis?
# Per label set (melanoma vs naevus; SkinCon: malignant vs not), at read level, reads weighted
# 1/(reads per image):
#   (a) out-of-fold AUROC and accuracy (threshold 0.5) of an L2 logistic regression on the
#       binary concept labels, 5 folds split by image, seed 11;
#   (b) for sided vocabularies, accuracy of the fixed rule "melanoma if any melanoma-side
#       concept is marked".
# Usage: python -I step12_consistency.py <step7_coupling.py> <dermaxai_step1_out> <easy_dir>
#        <release_v0_dir> <result_dir>
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

S7, DX, EASY, REL, RES = sys.argv[1:6]
src = open(S7).read().split("# ------------------------------------------------------------------ run")[0]
sys.argv = ["step7", DX, EASY, REL, RES]
exec(src)  # loaders: load_dermaxai, load_derm7pt, load_ph2, load_easy, load_skincon
os.makedirs(RES, exist_ok=True)

sets = {}
sets.update(load_dermaxai())
sets["Derm7pt (atlas)"] = load_derm7pt()
sets["PH2 (one expert)"] = load_ph2()
sets["EASY (5 blinded experts)"] = load_easy()
for k, v in load_skincon().items():
    if "malignant" in k:
        sets[k] = v
# blind reads scored against the reader's own diagnosis (0.5 'excise' counted as melanoma)
r = pd.read_csv(os.path.join(DX, "reads_test.csv"))
b = r[r.arm == "blind"].copy()
D = json.load(open(os.path.join(DX, "definitions.json")))
b = b.rename(columns={"image_id": "image"})
b["y"] = (b.own_dx >= 0.5).astype(int)
sets["Derma-XAI test, blind, vs OWN diagnosis"] = (b[["image", "y", "reader"] + D["MEL"] + D["NEV"]],
                                                  dict(mel=D["MEL"], nev=D["NEV"]))

rows = []
for name, (df, sides) in sets.items():
    df = df.reset_index(drop=True)
    C = [c for c in df.columns if c not in ("image", "y", "reader")]
    C = [c for c in C if df[c].nunique() > 1]
    w = 1.0 / df.groupby("image")["image"].transform("size").values
    imgs = df.image.unique()
    fold = dict(zip(imgs, np.random.default_rng(11).integers(0, 5, len(imgs))))
    f = df.image.map(fold).values
    X, y = df[C].values.astype(float), df.y.values.astype(int)
    s = np.zeros(len(df))
    for k in range(5):
        tr, te = f != k, f == k
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y[tr], sample_weight=w[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
    row = dict(label_set=name, images=len(imgs), reads=len(df), concepts=len(C),
               oof_auroc=roc_auc_score(y, s, sample_weight=w),
               oof_accuracy=np.average((s >= 0.5) == y, weights=w))
    if sides:
        mel = [c for c in sides["mel"] if c in df]
        rule = df[mel].max(axis=1).values
        row["rule_any_mel_side_accuracy"] = np.average(rule == y, weights=w)
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(RES, "step12_consistency.csv"), index=False)
print(T.round(3).to_string(index=False))
