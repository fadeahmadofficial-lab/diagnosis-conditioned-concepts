# Paper 2, B3 (pre-registered in the D1 doc 2026-10-13): do the V2 and T1 effects differ between
# EASY images selected as exemplars of our mapped concepts and the remaining images?
# Usage: python -I step16_b3_selection.py <step13_fix_derm7pt.py> <step8_recipe.py> <dermaxai_step1_out>
#        <kaggle_out> <p2_dir> <easy_dir> <v2_out_dir> <fix_out_dir> <result_dir>
import os
import sys

import numpy as np
import pandas as pd

S13 = sys.argv[1]
S8, DX, KO, P2, EASY, V2, FIX, RES = sys.argv[2:10]
src = open(S13).read().split("rows = []")[0]
sys.argv = ["step13", S8, DX, KO, P2, EASY, V2, FIX, RES]
exec(src)  # lab (full EASY), M (models), wauc, EASY_MAP
os.makedirs(RES, exist_ok=True)
M["v2 pi0"] = load(f"{V2}/easy_preds_v2_pi000_effb4_seed*.csv", f"{V2}/train_labels_v2_pi000_*.csv")
M["v2 pi1"] = load(f"{V2}/easy_preds_v2_pi100_effb4_seed*.csv", f"{V2}/train_labels_v2_pi100_*.csv")

meta = pd.read_csv(os.path.join(EASY, "metadata.csv"))
terms = {t for ts in EASY_MAP.values() for t in ts}
selected = set(meta[meta.exemplar.isin(terms)].ISIC_id)
sel = lab.image.isin(selected).values
CC = ["PN_atyp", "DaG_irr", "RS", "VS_atyp"]
w = 1.0 / lab.groupby("image")["image"].transform("size").values
codes, idx = np.unique(lab.image.values, return_inverse=True)
img_sel = pd.Series(sel).groupby(idx).first().values
groups = {g: np.where(img_sel == g)[0] for g in (True, False)}
print("images selected / rest:", len(groups[True]), len(groups[False]))


def excess(name, mask, ww):
    P, means = M[name]["P"], M[name]["means"]
    p = P.p.values[mask]
    return np.mean([wauc(P[c].values[mask], lab[c].values[mask], ww[mask])
                    - wauc(p * means[None][c][1] + (1 - p) * means[None][c][0], lab[c].values[mask], ww[mask])
                    for c in CC])


EFFECTS = {"V2 (pi1 - pi0)": ("v2 pi1", "v2 pi0"), "T1 (aware - std at pi1)": ("effb4 aware pi1", "effb4 std pi1")}


def stat(a, b, ww):
    e_sel = excess(a, sel, ww) - excess(b, sel, ww)
    e_rest = excess(a, ~sel, ww) - excess(b, ~sel, ww)
    return e_sel, e_rest, e_sel - e_rest


lines = []
for lab_, (a, b) in EFFECTS.items():
    pt = stat(a, b, w)
    rng = np.random.default_rng(2026)
    bs = []
    for _ in range(1000):
        cnt = np.zeros(len(codes))
        for g, ids in groups.items():  # resample images within each subset
            cnt += np.bincount(rng.choice(ids, len(ids)), minlength=len(codes))
        bs.append(stat(a, b, w * cnt[idx]))
    bs = np.array(bs)
    ci = np.percentile(bs, [2.5, 97.5], axis=0)
    s = (f"=== B3 {lab_} on {', '.join(CC)}: selected {pt[0]:+.3f} [{ci[0,0]:+.3f}, {ci[1,0]:+.3f}]; "
         f"rest {pt[1]:+.3f} [{ci[0,1]:+.3f}, {ci[1,1]:+.3f}]; difference {pt[2]:+.3f} [{ci[0,2]:+.3f}, {ci[1,2]:+.3f}] -> "
         f"{'concentrated in selected images' if (ci[0,2] > 0 or ci[1,2] < 0) else 'no detectable difference'}")
    print(s)
    lines.append(s)
open(os.path.join(RES, "step16_log.txt"), "w").write("\n".join(lines) + "\n")
