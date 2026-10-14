# Paper 2, fix experiment, Derma-XAI part: T2 (masked vs standard concept models against
# EASY blinded experts, 5 cleanly mapped concepts) and the pre-registered secondary (the
# same models judged against Derma-XAI's own aware test labels). Holm over T1 and T2.
# Usage: python -I step14_fix_dxai.py <step9_dxai_on_easy.py> <step8_recipe.py> <dermaxai_step1_out>
#        <old_kaggle_out> <d1_easy_out> <p2_dir> <easy_dir> <fix_dxai_dir> <step13_boot.csv> <result_dir>
import glob
import os
import sys

import numpy as np
import pandas as pd

S9, S8, DX, KO_OLD, KO_NEW, P2, EASY, FX, BOOT13, RES = sys.argv[1:11]
src9 = open(S9).read()
sys.argv = ["x", S8, DX, KO_OLD, KO_NEW, P2, EASY, RES]
exec(src9.split("# ---------------------------------------------------------------- (A)")[0])
exec(src9[src9.index("MAP_P = "):src9.index("def setting_s5")])
LOG.clear()
import json  # noqa: E402

D = json.load(open(os.path.join(DX, "definitions.json")))
C18 = D["MEL"] + D["NEV"]
tr = pd.read_csv(os.path.join(DX, "train_labels.csv"))
tr = tr[tr.split == "train"]
MEANS = {None: {c: (tr[tr.benign_malignant == 0][c].mean(), tr[tr.benign_malignant == 1][c].mean()) for c in C18}}


def mean_pred(pattern, cols):
    fs = sorted(glob.glob(pattern))
    return sum(pd.read_csv(f).set_index("image_id")[cols] for f in fs) / len(fs), len(fs)


# EASY labels for the primary map (same construction as step 9)
em = pd.read_csv(os.path.join(EASY, "metadata.csv"))
em = em[em.diagnosis.isin(["melanoma", "nevus"])]
ymap = dict(zip(em.ISIC_id, (em.diagnosis == "melanoma").astype(int)))
mk = json.load(open(os.path.join(EASY, "markups.json")))
rows = []
for iid, rd in mk.items():
    if iid in ymap:
        for rid, feats in rd.items():
            present = {k for k, v in feats.items() if len(v)}
            rows.append({"image": iid, "y": ymap[iid],
                         **{c: int(any(t in present for t in ts)) for c, ts in MAP_P.items()}})
elab = pd.DataFrame(rows)
CP = list(MAP_P)

out = {}
for bb in ("effb0", "convnext_tiny"):
    bbx, _ = mean_pred(os.path.join(KO_NEW, f"easy_{bb}_seed*.csv"), ["p_mel_blackbox"])
    for kind, pat in (("standard", os.path.join(KO_NEW, f"easy_{bb}_seed*.csv")),
                      ("masked", os.path.join(FX, f"easy_masked_{bb}_seed*.csv"))):
        P, n = mean_pred(pat, CP)
        P = P.join(bbx).rename(columns={"p_mel_blackbox": "p"}).loc[elab.image.values].reset_index(drop=True)
        label = f"{bb} {kind} -> EASY blinded (primary map)"
        say(f"[{label}: mean of {n} files]")
        recipe(elab, P[CP + ["p"]], MEANS, CP, label)
        out[(bb, kind, "easy")] = P

# own aware test labels (secondary): same recipe as S1, standard vs masked
r = pd.read_csv(os.path.join(DX, "reads_test.csv"))
aw = r[r.arm == "aware"][["image_id", "y"] + C18].rename(columns={"image_id": "image"}).reset_index(drop=True)
for bb in ("effb0", "convnext_tiny"):
    bbx, _ = mean_pred(os.path.join(KO_NEW, f"test_{bb}_seed*.csv"), ["p_mel_blackbox"])
    for kind, pat in (("standard", os.path.join(KO_NEW, f"test_{bb}_seed*.csv")),
                      ("masked", os.path.join(FX, f"test_masked_{bb}_seed*.csv"))):
        P, n = mean_pred(pat, C18)
        P = P.join(bbx).rename(columns={"p_mel_blackbox": "p"}).loc[aw.image.values].reset_index(drop=True)
        label = f"{bb} {kind} -> Derma-XAI aware test labels"
        say(f"[{label}: mean of {n} files]")
        recipe(aw, P[C18 + ["p"]], MEANS, C18, label)

# T2: paired bootstrap, EffB0, masked - standard, against EASY
w = 1.0 / elab.groupby("image")["image"].transform("size").values
codes, idx = np.unique(elab.image.values, return_inverse=True)
an = [c for c in CP if (elab[c].values * w).sum() >= 20 and ((1 - elab[c].values) * w).sum() >= 20]


def excess(P, ww):
    p = P.p.values
    return np.mean([wauc(P[c].values, elab[c].values, ww)
                    - wauc(p * MEANS[None][c][1] + (1 - p) * MEANS[None][c][0], elab[c].values, ww) for c in an])


A, Bs = out[("effb0", "masked", "easy")], out[("effb0", "standard", "easy")]
pt = excess(A, w) - excess(Bs, w)
rng = np.random.default_rng(2026)
bs = []
for _ in range(1000):
    cnt = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx]
    bs.append(excess(A, w * cnt) - excess(Bs, w * cnt))
bs = np.array(bs)
lo, hi = np.percentile(bs, [2.5, 97.5])
say(f"=== T2 Derma-XAI EffB0 masked - standard vs EASY ({len(an)} concepts): {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] "
    f"-> {'PASSED' if lo > 0 else 'not passed'} (rule: CI above 0)")
# Holm over T1 and T2, one-sided bootstrap p = share of replicates <= 0
b13 = pd.read_csv(BOOT13)
p1 = (np.sum(b13.T1_boot.values <= 0) + 1) / (len(b13) + 1)
p2 = (np.sum(bs <= 0) + 1) / (len(bs) + 1)
ps = sorted([("T1", p1), ("T2", p2)], key=lambda t: t[1])
holm = {ps[0][0]: ps[0][1] <= 0.025, ps[1][0]: ps[0][1] <= 0.025 and ps[1][1] <= 0.05}
say(f"=== Holm (one-sided bootstrap p): T1 p={p1:.4f}, T2 p={p2:.4f} -> T1 {'holds' if holm['T1'] else 'fails'}, "
    f"T2 {'holds' if holm['T2'] else 'fails'}")
open(os.path.join(RES, "step14_log.txt"), "w").write("\n".join(LOG) + "\n")
