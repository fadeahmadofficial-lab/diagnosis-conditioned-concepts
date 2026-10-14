# Paper 2, B4 (pre-registered in the D1 doc 2026-10-13): leave-one-annotator-out on the
# Derma-XAI aware test labels: coupling (eps2, step 7 code) and S1 excess (step 8 recipe).
# Usage: python -I step17_b4_loao.py <step7_coupling.py> <step8_recipe.py> <dermaxai_step1_out>
#        <easy_dir> <release_v0_dir> <d1_easy_out> <p2_dir> <result_dir>
import os
import sys

import numpy as np
import pandas as pd

S7, S8, DX, EASY, REL, KO_NEW, P2, RES = sys.argv[1:9]
sys.argv = ["step7", DX, EASY, REL, RES]
exec(open(S7).read().split("# ------------------------------------------------------------------ run")[0])
B = 2000
sys.argv = ["step8", DX, KO_NEW, P2, EASY, RES]
exec(open(S8).read().split("ALL, PER = [], []")[0])  # recipe, setting_dermaxai (uses KO = KO_NEW)
LOG.clear()
os.makedirs(RES, exist_ok=True)

aw, sides = load_dermaxai()["Derma-XAI test, aware"]
C = [c for c in aw.columns if c not in ("image", "y", "reader")]
lab_s1, P_s1, means, C18, _ = setting_dermaxai("aware", "effb0")
r = pd.read_csv(os.path.join(DX, "reads_test.csv"))
reader_of_row = r[r.arm == "aware"].reader.values  # same row order as setting_dermaxai's lab
assert len(reader_of_row) == len(lab_s1)

subsets = {f"drop {a}": aw.reader != a for a in sorted(aw.reader.unique(), key=lambda s: int(s[9:]))}
subsets["annotator1 alone"] = aw.reader == "annotator1"
subsets["all except annotator1"] = aw.reader != "annotator1"
subsets["all (reference)"] = aw.reader == aw.reader

rows = []
for name, keep in subsets.items():
    d, an = prep(aw[keep.values].reset_index(drop=True), C)
    s, _, _ = summarise(d, an, sides, d.groupby("image").size().max() > 1, with_null=False)
    row = dict(subset=name, images=d.image.nunique(), reads=len(d), concepts=len(an), eps2=s["eps2"],
               cross_nev_has_mel=s["cross_nev_has_mel"], cross_mel_has_nev=s["cross_mel_has_nev"])
    if name in ("annotator1 alone", "all except annotator1", "all (reference)"):
        bt = boot(d, an, sides, False)
        row["eps2_lo"], row["eps2_hi"] = np.nanpercentile(bt["eps2"], [2.5, 97.5])
    k = (reader_of_row != name[5:]) if name.startswith("drop ") else (
        reader_of_row == "annotator1" if name == "annotator1 alone" else
        reader_of_row != "annotator1" if name == "all except annotator1" else np.ones(len(lab_s1), bool))
    res, _ = recipe(lab_s1[k].reset_index(drop=True), P_s1[k].reset_index(drop=True), means, C18, "S1 " + name)
    allrow = [x for x in res if x.get("subset") == "all"]
    if allrow:
        row.update(s1_excess=allrow[0]["excess"], s1_lo=allrow[0]["lo"],
                   s1_hi=allrow[0]["hi"], s1_concepts=allrow[0]["concepts"])
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(os.path.join(RES, "step17_loao.csv"), index=False)
print(T.round(3).to_string(index=False))
lo = T[T.subset.str.startswith("drop ") | (T.subset == "all except annotator1")]
print(f"min eps2 over drop-one and all-except-annotator1: {lo.eps2.min():.3f} (rule: > 0.043)")
print(f"max S1 excess over the same: {lo.s1_excess.max():+.3f} (rule: < +0.063)")
open(os.path.join(RES, "step17_log.txt"), "w").write("\n".join(LOG) + "\n" + T.round(4).to_string(index=False) + "\n")
