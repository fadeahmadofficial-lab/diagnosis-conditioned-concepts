# Paper 2, fix experiment, Derm7pt part: T1 (fix recovers excess at pi = 1), T3 (ConvNeXt
# replication of V2) and the Derm7pt secondaries, as pre-registered ("Pre-registration of
# the fix", D1 doc, 2026-10-12, before the run).
# Usage: python -I step13_fix_derm7pt.py <step8_recipe.py> <dermaxai_step1_out> <kaggle_out>
#        <p2_dir> <easy_dir> <v2_out_dir> <fix_out_dir> <result_dir>
import glob
import os
import sys

import numpy as np
import pandas as pd

S8, DX, KO, P2, EASY, V2, FIX, RES = sys.argv[1:9]
src = open(S8).read().split("ALL, PER = [], []")[0]
sys.argv = ["step8", DX, KO, P2, EASY, RES]
exec(src)  # wauc, d7_probs, D7, setting_easy, recipe, say, LOG
LOG.clear()
os.makedirs(RES, exist_ok=True)
lab, _, _, C, _ = setting_easy("effb4")
lab = lab.reset_index(drop=True)
MEL = ["PN_atyp", "STR_irr", "PIG_irr", "DaG_irr", "RS", "BWV", "VS_atyp"]
w = 1.0 / lab.groupby("image")["image"].transform("size").values
codes, idx = np.unique(lab.image.values, return_inverse=True)
an = [c for c in C if (lab[c].values * w).sum() >= 20 and ((1 - lab[c].values) * w).sum() >= 20]


def load(pred_glob, label_glob):
    fs = sorted(glob.glob(pred_glob))
    d = pd.concat(pd.read_csv(f) for f in fs).groupby("isic_id").mean(numeric_only=True)
    pr = d7_probs(d.reset_index())
    pr.index = d.index
    P = pr.loc[lab.image.values].reset_index(drop=True)[C + ["p"]]
    t = pd.read_csv(sorted(glob.glob(label_glob))[0])
    dx = t.diagnosis.str.lower()
    mel = dx.str.startswith("melanoma") & ~dx.str.contains("metastasis")
    keep = mel | dx.str.contains("nevus")
    t = t[keep].copy()
    t["mel"] = mel[keep].astype(int)
    for k, (col, vals) in D7.items():
        t[k] = t[col].isin(vals).astype(int)
    means = {None: {k: (t[t.mel == 0][k].mean(), t[t.mel == 1][k].mean()) for k in C}}
    return dict(P=P, means=means, n=len(fs))


M = {
    "effb4 std pi0": load(f"{V2}/easy_preds_v2_pi000_effb4_seed*.csv", f"{V2}/train_labels_v2_pi000_*.csv"),
    "effb4 std pi1": load(f"{V2}/easy_preds_v2_pi100_effb4_seed*.csv", f"{V2}/train_labels_v2_pi100_*.csv"),
    "effb4 aware pi0": load(f"{FIX}/easy_preds_fix_pi000_aware_effb4_seed*.csv", f"{FIX}/train_labels_fix_pi000_aware_*.csv"),
    "effb4 aware pi1": load(f"{FIX}/easy_preds_fix_pi100_aware_effb4_seed*.csv", f"{FIX}/train_labels_fix_pi100_aware_*.csv"),
    "convnext std pi0": load(f"{FIX}/easy_preds_fix_pi000_std_convnext_tiny_seed*.csv", f"{FIX}/train_labels_fix_pi000_std_*.csv"),
    "convnext std pi1": load(f"{FIX}/easy_preds_fix_pi100_std_convnext_tiny_seed*.csv", f"{FIX}/train_labels_fix_pi100_std_*.csv"),
}
rows = []
for name, m in M.items():
    say(f"[{name}: mean of {m['n']} seeds]")
    r, _ = recipe(lab, m["P"], m["means"], C, name + " -> blinded EASY")
    rows += r
pd.DataFrame(rows).to_csv(os.path.join(RES, "step13_summary.csv"), index=False)


def excess(name, ww):
    P, means = M[name]["P"], M[name]["means"]
    p = P.p.values
    return np.mean([wauc(P[c].values, lab[c].values, ww)
                    - wauc(p * means[None][c][1] + (1 - p) * means[None][c][0], lab[c].values, ww)
                    for c in an])


def paired(a, b, label, rule):
    pt = excess(a, w) - excess(b, w)
    rng = np.random.default_rng(2026)
    bs = []
    for _ in range(1000):
        cnt = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx]
        bs.append(excess(a, w * cnt) - excess(b, w * cnt))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ok = (lo > 0) if rule == "above" else (hi < 0)
    say(f"=== {label}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {'PASSED' if ok else 'not passed'} "
        f"(rule: CI {rule} 0)")
    return pt, lo, hi, np.array(bs)


t1 = paired("effb4 aware pi1", "effb4 std pi1", "T1 fix at pi=1: aware - std", "above")
lost = excess("effb4 std pi0", w) - excess("effb4 std pi1", w)
say(f"    recovery share of the excess lost to censoring ({lost:.3f}): {t1[0] / lost:.0%}")
t3 = paired("convnext std pi1", "convnext std pi0", "T3 ConvNeXt replication of V2: pi1 - pi0", "below")
cost = paired("effb4 aware pi0", "effb4 std pi0", "secondary: cost of the fix on uncensored labels", "above")
nev = lab.y.values == 0
for name in ("effb4 std pi1", "effb4 aware pi1", "effb4 std pi0", "effb4 aware pi0"):
    P = M[name]["P"]
    a = [wauc(P[c].values[nev], lab[c].values[nev], w[nev]) for c in an if c in MEL
         and (lab[c].values[nev] * w[nev]).sum() >= 10 and ((1 - lab[c].values[nev]) * w[nev]).sum() >= 10]
    say(f"within-naevus AUROC, melanoma-side concepts, {name}: {np.mean(a):.3f} over {len(a)}")
pd.DataFrame({"T1_boot": t1[3], "T3_boot": t3[3], "cost_boot": cost[3]}).to_csv(
    os.path.join(RES, "step13_boot.csv"), index=False)
open(os.path.join(RES, "step13_log.txt"), "w").write("\n".join(LOG) + "\n")
