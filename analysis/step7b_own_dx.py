# Step 7b (added after seeing step 7; EXPLORATORY): Derma-XAI blind reads, coupling to the
# READER'S OWN diagnosis vs to the TRUE diagnosis, same concepts and same weights as step 7.
# own_dx: 0 naevus, 0.5 "naevus but excise", 1 melanoma (3 classes).
# Usage: python -I step7b_own_dx.py <dermaxai_step1_out> <result_dir>
import json
import os
import sys

import numpy as np
import pandas as pd

DX, RES = sys.argv[1:3]
D = json.load(open(os.path.join(DX, "definitions.json")))
r = pd.read_csv(os.path.join(DX, "reads_test.csv"))
b = r[r.arm == "blind"].copy()
b["w"] = 1.0 / b.groupby("image_id")["image_id"].transform("size")
C = D["MEL"] + D["NEV"]
an = [c for c in C if (b[c] * b.w).sum() >= 20 and ((1 - b[c]) * b.w).sum() >= 20
      and (b[c] * b.w).sum() / b.w.sum() >= 0.01]


def eps2(c, g, w):
    n, K = w.sum(), pd.Series(g).nunique()
    mu = np.average(c, weights=w)
    sst = (w * (c - mu) ** 2).sum()
    gm = pd.Series(w * c).groupby(g).sum() / pd.Series(w).groupby(g).sum()
    ssw = (w * (c - pd.Series(g).map(gm).values) ** 2).sum()
    return 1 - (ssw / (n - K)) / (sst / (n - 1))


def run(d):
    w = d.w.values
    t = np.mean([eps2(d[c].values.astype(float), d.y.values, w) for c in an])
    o = np.mean([eps2(d[c].values.astype(float), d.own_dx.values, w) for c in an])
    side = np.where(d.own_dx.values >= 0.5, 1, 0)
    mel, nev = [c for c in D["MEL"] if c in d], [c for c in D["NEV"] if c in d]
    x1 = (d[mel].max(axis=1).values[side == 0]).mean()
    x2 = (d[nev].max(axis=1).values[d.own_dx.values == 1]).mean()
    return t, o, x1, x2


t, o, x1, x2 = run(b)
imgs = b.image_id.unique()
idx = {i: np.where(b.image_id.values == i)[0] for i in imgs}
rng = np.random.default_rng(5)
bs = []
for _ in range(2000):
    pick = rng.choice(imgs, len(imgs))
    bs.append(run(b.iloc[np.concatenate([idx[i] for i in pick])]))
bs = np.array(bs)
lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
L = [f"Blind reads {len(b)}, images {len(imgs)}, concepts {len(an)}: {an}",
     f"  coupling eps2 to TRUE diagnosis   {t:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]",
     f"  coupling eps2 to OWN diagnosis    {o:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]",
     f"  difference own - true             {o - t:+.3f} [{np.percentile(bs[:,1]-bs[:,0],2.5):+.3f}, "
     f"{np.percentile(bs[:,1]-bs[:,0],97.5):+.3f}]",
     f"  reads with own dx naevus (0) marking >=1 melanoma-side concept: {x1:.3f}",
     f"  reads with own dx melanoma (1) marking >=1 naevus-side concept: {x2:.3f}",
     "  own_dx counts: " + str(b.own_dx.value_counts().sort_index().to_dict())]
print("\n".join(L))
open(os.path.join(RES, "step7b_log.txt"), "w").write("\n".join(L) + "\n")
