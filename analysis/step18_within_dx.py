# Paper 2, step 18: diagnosis-adjusted concept AUROC (theta_d) and a p-quality check.
# Written 2026-10-14 BEFORE any number below was computed. Added after the main results
# (proposal "Candidate 1"); SECONDARY analysis, logged as a deviation. The primary metric
# stays the excess of step 8. Already known before writing: the step 8/9 within-class
# means (unweighted over concepts), and a probe of worst-case bounds (cib/probe_bounds.py).
#
# Usage: python -I step18_within_dx.py <step8_recipe.py> <step9_dxai_on_easy.py> <dx_out>
#        <kaggle_old> <kaggle_new> <p2> <easy> <result_dir>
#
# PRE-SPECIFIED
# theta_d (per concept): positives-weighted mean of within-diagnosis AUROCs over the strata
#   (naevus, melanoma) that are identifiable (>= 10 weighted positives and negatives in the
#   stratum); a stratum that is not identifiable is reported, never imputed. Computed for the
#   model score (theta_model) and the diagnosis-only null (theta_null); dtheta = difference.
#   Read weights 1/(reads per image), image bootstrap 2000, as step 8.
# Same-side version: only the stratum of the concept's own side (melanoma-side concepts in
#   melanomas, naevus-side in naevi). This is the only stratum aware labels can identify.
# H1 (headline stress test). Derma-XAI EffB0, 4 shared concepts (TRBL, GP, PV, WLSA, all
#   melanoma-side), melanoma stratum only: dtheta against EASY (F3) minus dtheta against aware
#   reads (F1); independent image bootstraps (different images), as step 9b.
#   CI above 0 -> the F1-vs-F3 gap survives removing the between-class component.
#   CI includes 0 -> the gap is carried by the between-class (opposite-side) component; the
#   paper must then say the near-zero F1 excess comes from labels that carry no opposite-side
#   information, and that within melanomas the model's same-side skill is measurable on both.
# H2 (p-quality confound). The F1 black box has image AUROC 0.874, the F3 one 0.839. Degrade
#   F1's p with Gaussian noise on logit p, sigma chosen (grid 0.05) so the mean image AUROC
#   over 50 draws is closest to 0.839; recompute the F1 excess on the 4 shared concepts
#   (all images) as the mean over 50 draws (draws fixed across bootstrap replicates).
#   Matched F3 - F1 difference with CI above 0 -> the gap is not explained by p quality.
#   Also reported: excess vs sigma in {0, 0.5, 1, 2} for F1 and F3 (descriptive).
import os
import sys

import numpy as np
import pandas as pd

S8, S9, DXO, KO_OLD, KO_NEW, P2D, EASYD, RES = sys.argv[1:9]
src9 = open(S9).read()
sys.argv = ["step9", S8, DXO, KO_OLD, KO_NEW, P2D, EASYD, RES]
exec(src9.split("# ---------------------------------------------------------------- (A)")[0])
exec(src9[src9.index("MAP_P = "):src9.index("for mapping, mname")])
LOG.clear()
B = 2000
D7MEL = ["PN_atyp", "STR_irr", "PIG_irr", "DaG_irr", "RS", "BWV", "VS_atyp"]
MELSIDE = set(D["MEL"]) | set(D7MEL)
COM = ["TRBL", "GP", "PV", "WLSA"]


def nulls(lab, P, means, cs):
    fkey = lab["fold"].values if "fold" in lab else np.full(len(lab), None)
    p = P.p.values
    return {c: p * np.array([means[k][c][1] for k in fkey]) + (1 - p) * np.array([means[k][c][0] for k in fkey])
            for c in cs}


def theta(q, c, y, w, strata):
    num, den, used = 0.0, 0.0, []
    for k in strata:
        s = y == k
        wp, wn = (w[s] * c[s]).sum(), (w[s] * (1 - c[s])).sum()
        if wp >= 10 and wn >= 10:
            num += wp * wauc(q[s], c[s], w[s]); den += wp; used.append(k)
    return (num / den if den else np.nan), used


def analyse(lab, P, means, C, label, cs=None):
    w0 = 1.0 / lab.groupby("image")["image"].transform("size").values
    y = lab.y.values
    an = cs or [c for c in C if (lab[c].values * w0).sum() >= 20 and ((1 - lab[c].values) * w0).sum() >= 20]
    nl = nulls(lab, P, means, an)
    codes, idx = np.unique(lab.image.values, return_inverse=True)

    def stat(w):
        out = {}
        for ver in ("both", "same"):
            tm, tn = [], []
            for c in an:
                st = (0, 1) if ver == "both" else ((1,) if c in MELSIDE else (0,))
                a, _ = theta(P[c].values, lab[c].values, y, w, st)
                b, _ = theta(nl[c], lab[c].values, y, w, st)
                if not np.isnan(a):
                    tm.append(a); tn.append(b)
            out[ver] = (np.mean(tm), np.mean(tn), np.mean(tm) - np.mean(tn), len(tm))
        return out

    pt = stat(w0)
    rng = np.random.default_rng(2026)
    bs = [stat(w0 * np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx])
          for _ in range(B)]
    rows = []
    for ver in ("both", "same"):
        d = np.array([b[ver][2] for b in bs])
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        m, n, dd, k = pt[ver]
        say(f"  {label:58s} {ver:4s} theta_model {m:.3f} theta_null {n:.3f} dtheta {dd:+.3f} [{lo:+.3f}, {hi:+.3f}] (concepts {k})")
        rows.append(dict(setting=label, version=ver, theta_model=m, theta_null=n, dtheta=dd, lo=lo, hi=hi, concepts=k))
    per = []
    for c in an:
        for st in ((0,), (1,)):
            a, u = theta(P[c].values, lab[c].values, y, w0, st)
            b, _ = theta(nl[c], lab[c].values, y, w0, st)
            per.append(dict(setting=label, concept=c, side="mel" if c in MELSIDE else "nev",
                            stratum="melanoma" if st == (1,) else "naevus", identifiable=bool(u),
                            auroc_model=a, auroc_null=b))
    return rows, per, [np.array([b["same"][2] for b in bs]), pt["same"][2]]


say("=== step 18: theta_d (diagnosis-adjusted concept AUROC)")
ROWS, PER = [], []
settings = [
    (lambda: setting_dermaxai("aware", "effb0"), "F1 Derma-XAI EffB0 -> aware reads"),
    (lambda: setting_dermaxai("blind", "effb0"), "F2 Derma-XAI EffB0 -> blind reads"),
    (lambda: setting_s5(MAP_P, "effb0"), "F3 Derma-XAI EffB0 -> EASY"),
    (lambda: setting_derm7pt("effb4"), "F4 Derm7pt EffB4 OOF -> Derm7pt"),
    (lambda: setting_easy("effb4"), "F5 Derm7pt EffB4 -> EASY"),
    (lambda: setting_dermaxai("aware", "convnext_tiny"), "sens F1 ConvNeXt -> aware reads"),
    (lambda: setting_s5(MAP_P, "convnext_tiny"), "sens F3 ConvNeXt -> EASY"),
]
for fn, label in settings:
    lab, P, means, C, _ = fn()
    r, pc, _ = analyse(lab, P, means, C, label)
    ROWS += r; PER += pc

say("\n=== H1: 4 shared concepts, melanoma stratum (same-side), F3 - F1")
l1, P1, m1, _, _ = setting_dermaxai("aware", "effb0")
l3, P3, m3, _, _ = setting_s5(MAP_P, "effb0")
_, _, (b1, a1) = analyse(l1, P1, m1, COM, "H1 F1 shared", COM)
_, _, (b3, a3) = analyse(l3, P3, m3, COM, "H1 F3 shared", COM)
d = b3 - b1
lo, hi = np.nanpercentile(d, [2.5, 97.5])
say(f"  H1 dtheta F3 - F1 = {a3 - a1:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
    f"{'gap survives' if lo > 0 else 'gap NOT shown within melanomas'}")
ROWS.append(dict(setting="H1 F3-F1 shared, melanoma stratum", version="same", dtheta=a3 - a1, lo=lo, hi=hi))

say("\n=== H2: p-quality check (logit noise on the F1 black box)")
def logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6); return np.log(x / (1 - x))
def img_auc(lab, p):
    g = pd.DataFrame({"i": lab.image.values, "p": p, "y": lab.y.values}).groupby("i").first()
    return wauc(g.p.values, g.y.values, np.ones(len(g)))
ND = 50


def excess_noise(lab, P, means, sigma, seed, cs=COM):
    w0 = 1.0 / lab.groupby("image")["image"].transform("size").values
    codes, idx = np.unique(lab.image.values, return_inverse=True)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((ND, len(codes)))  # one draw per image, fixed across bootstrap
    lp = logit(P.p.values)
    draws = []
    for j in range(ND):
        Pj = P.copy(); Pj["p"] = 1 / (1 + np.exp(-(lp + sigma * Z[j][idx])))
        draws.append((Pj, nulls(lab, Pj, means, cs), img_auc(lab, Pj.p.values)))
    def f(w):
        return np.mean([np.mean([wauc(P[c].values, lab[c].values, w) - wauc(nl[c], lab[c].values, w) for c in cs])
                        for _, nl, _ in draws])
    return f, np.mean([a for _, _, a in draws]), w0, codes, idx

target = img_auc(l3, P3.p.values)
grid = np.round(np.arange(0, 1.51, 0.05), 2)
aucs = {s: excess_noise(l1, P1, m1, s, 7)[1] for s in grid}
sig = min(grid, key=lambda s: abs(aucs[s] - target))
say(f"  target image AUROC (F3 black box) {target:.3f}; F1 at sigma 0 {aucs[0.0]:.3f}; matched sigma {sig} -> {aucs[sig]:.3f}")
f1, _, w1, c1, i1 = excess_noise(l1, P1, m1, sig, 7)
f3, _, w3, c3, i3 = excess_noise(l3, P3, m3, 0.0, 7)
rng = np.random.default_rng(11)
bs1 = np.array([f1(w1 * np.bincount(rng.integers(0, len(c1), len(c1)), minlength=len(c1))[i1]) for _ in range(B)])
bs3 = np.array([f3(w3 * np.bincount(rng.integers(0, len(c3), len(c3)), minlength=len(c3))[i3]) for _ in range(B)])
e1, e3 = f1(w1), f3(w3)
lo, hi = np.nanpercentile(bs3 - bs1, [2.5, 97.5])
say(f"  F1 excess (shared, p matched) {e1:+.3f} [{np.percentile(bs1, 2.5):+.3f}, {np.percentile(bs1, 97.5):+.3f}]; "
    f"F3 {e3:+.3f}; F3 - F1 {e3 - e1:+.3f} [{lo:+.3f}, {hi:+.3f}] -> "
    f"{'gap not explained by p quality' if lo > 0 else 'gap NOT robust to p quality'}")
ROWS.append(dict(setting=f"H2 F3-F1 shared, F1 p degraded sigma={sig}", version="all", dtheta=e3 - e1, lo=lo, hi=hi))
for s in (0.0, 0.5, 1.0, 2.0):
    a = excess_noise(l1, P1, m1, s, 7); b = excess_noise(l3, P3, m3, s, 7)
    say(f"  sigma {s}: F1 excess {a[0](a[2]):+.3f} (p AUROC {a[1]:.3f})   F3 excess {b[0](b[2]):+.3f} (p AUROC {b[1]:.3f})")

pd.DataFrame(ROWS).to_csv(os.path.join(RES, "step18_summary.csv"), index=False)
pd.DataFrame(PER).to_csv(os.path.join(RES, "step18_per_concept.csv"), index=False)
open(os.path.join(RES, "step18_log.txt"), "w").write("\n".join(LOG) + "\n")
