# Paper 2, V2 analysis: censored-training test, as pre-registered in the D1 doc
# ("Pre-registration of V2", written before the Kaggle run).
# Usage: python -I step11_v2_analysis.py <step8_recipe.py> <dermaxai_step1_out> <kaggle_out>
#        <p2_dir> <easy_dir> <v2_out_dir> <result_dir>
import glob
import os
import sys

import numpy as np
import pandas as pd

S8, DX, KO, P2, EASY, V2, RES = sys.argv[1:8]
src = open(S8).read().split("ALL, PER = [], []")[0]
sys.argv = ["step8", DX, KO, P2, EASY, RES]
exec(src)  # wauc, d7_probs, D7, setting_easy, recipe, say, LOG
LOG.clear()
os.makedirs(RES, exist_ok=True)

lab, _, _, C, _ = setting_easy("effb4")          # blinded EASY labels (read level), 10 concepts
lab = lab.reset_index(drop=True)
MEL = ["PN_atyp", "STR_irr", "PIG_irr", "DaG_irr", "RS", "BWV", "VS_atyp"]
NEV = ["PN_typ", "PIG_reg", "DaG_reg"]


def load(pi):
    fs = sorted(glob.glob(os.path.join(V2, f"easy_preds_v2_pi{pi:03d}_effb4_seed*.csv")))
    d = pd.concat(pd.read_csv(f) for f in fs).groupby("isic_id").mean(numeric_only=True)
    pr = d7_probs(d.reset_index())
    pr.index = d.index
    P = pr.loc[lab.image.values].reset_index(drop=True)[C + ["p"]]
    t = pd.read_csv(sorted(glob.glob(os.path.join(V2, f"train_labels_v2_pi{pi:03d}_*.csv")))[0])
    dx = t.diagnosis.str.lower()
    mel = dx.str.startswith("melanoma") & ~dx.str.contains("metastasis")
    keep = mel | dx.str.contains("nevus")
    t = t[keep].copy()
    t["mel"] = mel[keep].astype(int)
    for k, (col, vals) in D7.items():
        t[k] = t[col].isin(vals).astype(int)
    means = {None: {k: (t[t.mel == 0][k].mean(), t[t.mel == 1][k].mean()) for k in C}}
    return P, means, len(fs)


def censored_reference(lab):
    out = lab.copy()
    for c in C:
        opp = (out.y == 0) if c in MEL else (out.y == 1)
        out.loc[opp & (out[c] == 1), c] = 0
    return out


rows, per = [], []
store = {}
for pi in (0, 50, 100):
    P, means, ns = load(pi)
    store[pi] = (P, means)
    say(f"[V2 pi = {pi / 100:.1f}: mean of {ns} seeds]")
    r, pc = recipe(lab, P, means, C, f"V2 pi={pi / 100:.1f} -> blinded EASY")
    rows += r
    per += pc
    if pi in (0, 100):
        r2, _ = recipe(censored_reference(lab), P, means, C, f"V2 pi={pi / 100:.1f} -> EASY censored at 1")
        rows += r2
pd.DataFrame(rows).to_csv(os.path.join(RES, "step11_v2_summary.csv"), index=False)
pd.DataFrame(per).to_csv(os.path.join(RES, "step11_v2_per_concept.csv"), index=False)

# ---- the one pre-specified test: excess(pi=1) - excess(pi=0) against blinded EASY ----
w = 1.0 / lab.groupby("image")["image"].transform("size").values
codes, idx = np.unique(lab.image.values, return_inverse=True)
an = [c for c in C if (lab[c].values * w).sum() >= 20 and ((1 - lab[c].values) * w).sum() >= 20]


def excess(pi, ww):
    P, means = store[pi]
    p = P.p.values
    e = []
    for c in an:
        nul = p * means[None][c][1] + (1 - p) * means[None][c][0]
        e.append(wauc(P[c].values, lab[c].values, ww) - wauc(nul, lab[c].values, ww))
    return np.mean(e)


pt = excess(100, w) - excess(0, w)
rng = np.random.default_rng(2026)
bs = []
for _ in range(1000):
    cnt = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx]
    bs.append(excess(100, w * cnt) - excess(0, w * cnt))
lo, hi = np.percentile(bs, [2.5, 97.5])
say(f"=== THE TEST: excess(pi=1) - excess(pi=0) vs blinded EASY: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
    f"{'DETECTED: censored training lowers excess' if hi < 0 else 'not detected'}")

# within-naevus AUROC of melanoma-side concepts (descriptive)
nev = lab.y.values == 0
for pi in (0, 50, 100):
    P, _ = store[pi]
    a = [wauc(P[c].values[nev], lab[c].values[nev], w[nev]) for c in an if c in MEL
         and (lab[c].values[nev] * w[nev]).sum() >= 10 and ((1 - lab[c].values[nev]) * w[nev]).sum() >= 10]
    say(f"within-naevus AUROC, melanoma-side concepts, pi={pi / 100:.1f}: mean {np.mean(a):.3f} over {len(a)} concepts")
open(os.path.join(RES, "step11_log.txt"), "w").write("\n".join(LOG) + "\n")
