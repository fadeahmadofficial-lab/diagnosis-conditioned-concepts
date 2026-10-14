# Paper 2, B1 (pre-registered in the D1 doc 2026-10-13): Derma-XAI masked vs standard concept
# models scored against the BLIND-arm reads on the 196 Derma-XAI test images.
# Usage: python -I step15_b1_blind.py <step8_recipe.py> <dermaxai_step1_out> <d1_easy_out> <p2_dir>
#        <easy_dir> <fix_dxai_dir> <result_dir>
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

S8, DX, KO_NEW, P2, EASY, FX, RES = sys.argv[1:8]
src = open(S8).read().split("ALL, PER = [], []")[0]
sys.argv = ["step8", DX, KO_NEW, P2, EASY, RES]
exec(src)  # wauc, recipe, say, LOG
LOG.clear()
os.makedirs(RES, exist_ok=True)

D = json.load(open(os.path.join(DX, "definitions.json")))
C18 = D["MEL"] + D["NEV"]
tr = pd.read_csv(os.path.join(DX, "train_labels.csv"))
tr = tr[tr.split == "train"]
MEANS = {None: {c: (tr[tr.benign_malignant == 0][c].mean(), tr[tr.benign_malignant == 1][c].mean()) for c in C18}}

r = pd.read_csv(os.path.join(DX, "reads_test.csv"))
bl = r[r.arm == "blind"][["image_id", "y"] + C18].rename(columns={"image_id": "image"}).reset_index(drop=True)


def mean_pred(pattern, cols):
    fs = sorted(glob.glob(pattern))
    return sum(pd.read_csv(f).set_index("image_id")[cols] for f in fs) / len(fs), len(fs)


out = {}
for bb in ("effb0", "convnext_tiny"):
    # p = the standard D1 black box for both arms (masked files carry no black box), as in T2
    bbx, _ = mean_pred(os.path.join(KO_NEW, f"test_{bb}_seed*.csv"), ["p_mel_blackbox"])
    for kind, pat in (("standard", os.path.join(KO_NEW, f"test_{bb}_seed*.csv")),
                      ("masked", os.path.join(FX, f"test_masked_{bb}_seed*.csv"))):
        P, n = mean_pred(pat, C18)
        P = P.join(bbx).rename(columns={"p_mel_blackbox": "p"}).loc[bl.image.values].reset_index(drop=True)
        label = f"{bb} {kind} -> Derma-XAI BLIND test reads"
        say(f"[{label}: mean of {n} files]")
        recipe(bl, P[C18 + ["p"]], MEANS, C18, label)
        out[(bb, kind)] = P

w = 1.0 / bl.groupby("image")["image"].transform("size").values
codes, idx = np.unique(bl.image.values, return_inverse=True)
an = [c for c in C18 if (bl[c].values * w).sum() >= 20 and ((1 - bl[c].values) * w).sum() >= 20]


def excess(P, ww):
    p = P.p.values
    return np.mean([wauc(P[c].values, bl[c].values, ww)
                    - wauc(p * MEANS[None][c][1] + (1 - p) * MEANS[None][c][0], bl[c].values, ww) for c in an])


for bb in ("effb0", "convnext_tiny"):
    A, B = out[(bb, "masked")], out[(bb, "standard")]
    pt = excess(A, w) - excess(B, w)
    rng = np.random.default_rng(2026)
    bs = []
    for _ in range(1000):
        cnt = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx]
        bs.append(excess(A, w * cnt) - excess(B, w * cnt))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    tag = "B1 PRIMARY" if bb == "effb0" else "B1 secondary"
    say(f"=== {tag} {bb} masked - standard vs BLIND reads ({len(an)} concepts: {', '.join(an)}): "
        f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (f" -> {'PASSED' if lo > 0 else 'not passed'} (rule: CI above 0)"
                                              if bb == "effb0" else ""))
open(os.path.join(RES, "step15_log.txt"), "w").write("\n".join(LOG) + "\n")
