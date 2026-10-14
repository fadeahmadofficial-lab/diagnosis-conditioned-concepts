# Paper 2, V: censoring simulation on EASY blinded labels. Pre-registered in the D1 doc
# ("Pre-registration of V", 2026-10-10) before any number was computed.
# Usage: python -I step10_censoring_sim.py <step8_recipe.py> <dermaxai_step1_out> <kaggle_out>
#        <p2_dir> <easy_dir> <result_dir>
import os
import sys

import numpy as np
import pandas as pd

S8, DX, KO, P2, EASY, RES = sys.argv[1:7]
src = open(S8).read().split("ALL, PER = [], []")[0]
sys.argv = ["step8", DX, KO, P2, EASY, RES]
exec(src)  # wauc, setting_easy, EASY_MAP, say, LOG ...
LOG.clear()
os.makedirs(RES, exist_ok=True)

MEL = ["PN_atyp", "STR_irr", "PIG_irr", "DaG_irr", "RS", "BWV", "VS_atyp"]
NEV = ["PN_typ", "PIG_reg", "DaG_reg"]
lab0, P, means, C, ns = setting_easy("effb4")
lab0 = lab0.reset_index(drop=True)
P = P.reset_index(drop=True)
y = lab0.y.values
w0 = 1.0 / lab0.groupby("image")["image"].transform("size").values
codes, idx = np.unique(lab0.image.values, return_inverse=True)
n_img = len(codes)
an = [c for c in C if (lab0[c].values * w0).sum() >= 20 and ((1 - lab0[c].values) * w0).sum() >= 20]
side = {c: (1 if c in MEL else 0) for c in an}          # 1 = melanoma-side
p = P.p.values
nullscore = {c: p * means[None][c][1] + (1 - p) * means[None][c][0] for c in an}
wrong_model = ((p >= 0.5).astype(int) != y)

# scenario W: flip the supplied diagnosis on a random 20% of images within each class
rng_w = np.random.default_rng(2026)
img_y = pd.Series(y).groupby(idx).first().values
flip_img = np.zeros(n_img, bool)
for cls in (0, 1):
    ids = np.where(img_y == cls)[0]
    flip_img[rng_w.choice(ids, int(round(0.2 * len(ids))), replace=False)] = True
flip = flip_img[idx]
SUPPLIED = {"T": y.copy(), "W": np.where(flip, 1 - y, y)}


def censor(lab, supplied, pi, rng):
    out = lab.copy()
    for c in an:
        opposite = (supplied == 0) if side[c] == 1 else (supplied == 1)
        drop = opposite & (out[c].values == 1) & (rng.random(len(out)) < pi)
        out.loc[drop, c] = 0
    return out


def eps2_bin(c, g, w):
    mu = np.average(c, weights=w)
    sst = (w * (c - mu) ** 2).sum()
    if sst == 0:
        return np.nan
    ssw = sum((w[g == k] * (c[g == k] - np.average(c[g == k], weights=w[g == k])) ** 2).sum()
              for k in (0, 1))
    n = w.sum()
    return 1 - (ssw / (n - 2)) / (sst / (n - 1))


def alpha(lab, c, sel):
    d = lab[sel]
    g = d.groupby("image")[c].agg(["sum", "count"])
    g = g[g["count"] >= 2]
    n1, m = g["sum"].values.astype(float), g["count"].values.astype(float)
    n0 = m - n1
    n, N1, N0 = m.sum(), n1.sum(), n0.sum()
    if N1 == 0 or N0 == 0:
        return np.nan
    return 1 - (n - 1) * (2 * n1 * n0 / (m - 1)).sum() / (2 * N1 * N0)


def ok(cc, ww):
    return (cc * ww).sum() >= 10 and ((1 - cc) * ww).sum() >= 10


def metrics(lab, w, supplied, with_alpha=True):
    r = {}
    r["eps2_truth"] = np.nanmean([eps2_bin(lab[c].values.astype(float), y, w) for c in an])
    r["eps2_supplied"] = np.nanmean([eps2_bin(lab[c].values.astype(float), supplied, w) for c in an])
    lab_null = []
    for c in an:
        cc = lab[c].values
        if ok(cc, w):
            a = wauc(y.astype(float), cc, w)
            lab_null.append(max(a, 1 - a))
    r["labels_null_auc"] = np.mean(lab_null)
    mel_any = lab[[c for c in an if side[c] == 1]].max(axis=1).values
    nev_any = lab[[c for c in an if side[c] == 0]].max(axis=1).values
    r["cross_nev_has_mel"] = np.average(mel_any[y == 0], weights=w[y == 0])
    r["cross_mel_has_nev"] = np.average(nev_any[y == 1], weights=w[y == 1])
    if with_alpha:
        r["alpha_within_nev"] = np.nanmean([alpha(lab, c, y == 0) for c in an])
        r["alpha_within_mel"] = np.nanmean([alpha(lab, c, y == 1) for c in an])
    subsets = {"all": np.ones(len(y), bool), "within_nev": y == 0, "within_mel": y == 1,
               "model_misdiagnosed": wrong_model, "flipped": flip}
    for sn, sel in subsets.items():
        m1, m2 = [], []
        for c in an:
            cc, ww = lab[c].values[sel], w[sel]
            if not ok(cc, ww):
                continue
            m1.append(wauc(P[c].values[sel], cc, ww))
            m2.append(wauc(nullscore[c][sel], cc, ww))
        r[sn + "_model"] = np.mean(m1) if m1 else np.nan
        r[sn + "_null"] = np.mean(m2) if m2 else np.nan
        r[sn + "_excess"] = r[sn + "_model"] - r[sn + "_null"] if m1 else np.nan
        r[sn + "_k"] = len(m1)
    return r


say(f"V: EASY images {n_img}, reads {len(lab0)}, analysable concepts {len(an)}: {an}; "
    f"model misdiagnoses {int(pd.Series(wrong_model).groupby(idx).first().sum())} images; "
    f"W flips {int(flip_img.sum())} images")
rows = []
for scen in ("T", "W"):
    for pi in (0, 0.25, 0.5, 0.75, 1):
        rng = np.random.default_rng(int(pi * 100) + (0 if scen == "T" else 1000))
        draws = [metrics(censor(lab0, SUPPLIED[scen], pi, rng), w0, SUPPLIED[scen])
                 for _ in range(1 if pi in (0, 1) else 20)]
        r = pd.DataFrame(draws).mean().to_dict()
        r.update(scenario=scen, pi=pi)
        rows.append(r)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(RES, "step10_censoring.csv"), index=False)
show = ["eps2_truth", "eps2_supplied", "labels_null_auc", "cross_nev_has_mel", "cross_mel_has_nev",
        "alpha_within_nev", "alpha_within_mel", "all_model", "all_null", "all_excess",
        "within_nev_excess", "within_nev_k", "within_mel_excess", "model_misdiagnosed_model",
        "model_misdiagnosed_null", "flipped_model", "flipped_null", "flipped_k"]
for scen in ("T", "W"):
    say(f"=== scenario {scen}")
    say(R[R.scenario == scen].set_index("pi")[show].T.round(3).to_string())

# the one pre-specified test: excess(pi=1) - excess(pi=0), scenario T, image bootstrap
lab1 = censor(lab0, SUPPLIED["T"], 1.0, np.random.default_rng(0))
rb = np.random.default_rng(2026)
d = []
for _ in range(1000):
    cnt = np.bincount(rb.integers(0, n_img, n_img), minlength=n_img)[idx]
    ww = w0 * cnt
    d.append(metrics(lab1, ww, SUPPLIED["T"], False)["all_excess"]
             - metrics(lab0, ww, SUPPLIED["T"], False)["all_excess"])
pt = (metrics(lab1, w0, SUPPLIED["T"], False)["all_excess"]
      - metrics(lab0, w0, SUPPLIED["T"], False)["all_excess"])
lo, hi = np.percentile(d, [2.5, 97.5])
say(f"=== THE TEST: excess(pi=1) - excess(pi=0), scenario T: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
    f"{'DETECTED (CI below 0)' if hi < 0 else 'not detected'}")
open(os.path.join(RES, "step10_log.txt"), "w").write("\n".join(LOG) + "\n")
