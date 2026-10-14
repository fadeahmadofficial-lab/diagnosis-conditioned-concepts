# Paper 2, step 20: worst-case identification bounds for concept AUROC (supplement).
# Written 2026-10-14. Decided after the bounds probe (cib/probe_bounds.py), so DESCRIPTIVE.
# Unweighted read-level AUROC (the sorting result is proved for equal weights only; with
# read weights it fails, see the probe). Uncertain set M_c = reads whose recorded label is 0
# and whose conditioning diagnosis is on the opposite side of c (aware: true diagnosis;
# blind: the reader's own call; Derm7pt: true diagnosis, as if it were explanation labels).
# Bounds: min over k of AUROC with the k lowest-scored members of M set to 1, max over k with
# the k highest. Unit test: exhaustive enumeration on random small instances.
import itertools
import sys
import numpy as np
import pandas as pd
src = open(sys.argv[1]).read()
sys.argv = ["s18"] + sys.argv[2:]
exec(src.split('say("=== step 18')[0])
LOG.clear()


def auc(s, c):
    return wauc(s, c, np.ones(len(s)))


def bounds(q, yt, M):
    o = np.where(M)[0]
    o = o[np.argsort(q[o], kind="stable")]
    L, U = [], []
    for k in range(len(o) + 1):
        a = yt.copy(); a[o[:k]] = 1; L.append(auc(q, a))
        b = yt.copy()
        if k:
            b[o[len(o) - k:]] = 1
        U.append(auc(q, b))
    return np.nanmin(L), np.nanmax(U)


# unit test
rng = np.random.default_rng(1); bad = cases = 0
for _ in range(2000):
    n = rng.integers(4, 12); q = rng.integers(0, 5, n).astype(float)
    yt = (rng.random(n) < .35).astype(int); M = (yt == 0) & (rng.random(n) < .5)
    mi = np.where(M)[0]
    if not len(mi):
        continue
    allv = []
    for k in range(len(mi) + 1):
        for S in itertools.combinations(mi, k):
            y = yt.copy(); y[list(S)] = 1; allv.append(auc(q, y))
    allv = [v for v in allv if not np.isnan(v)]
    lo, hi = bounds(q, yt, M); cases += 1
    bad += (abs(lo - min(allv)) > 1e-12) or (abs(hi - max(allv)) > 1e-12)
say(f"unit test: {cases} random instances, exact-bound mismatches {bad}")

rows = []
r = pd.read_csv(os.path.join(DXO, "reads_test.csv"))
for arm, tag in (("aware", "F1"), ("blind", "F2")):
    lab, P, _, C, _ = setting_dermaxai(arm, "effb0")
    cond = lab.y.values if arm == "aware" else (r[r.arm == "blind"].own_dx.values == 1).astype(int)  # 0.5 = naevus but excise -> naevus side
    for c in C:
        yt = lab[c].values.astype(int)
        if yt.sum() < 20 or (1 - yt).sum() < 20:
            continue
        M = ((cond == 0) if c in MELSIDE else (cond == 1)) & (yt == 0)
        lo, hi = bounds(P[c].values, yt, M)
        rows.append(dict(setting=tag, concept=c, side="mel" if c in MELSIDE else "nev",
                         naive=auc(P[c].values, yt), lower=lo, upper=hi, share_uncertain=M.mean()))
lab, P, _, C, _ = setting_derm7pt("effb4")
for c in C:
    yt = lab[c].values.astype(int)
    M = ((lab.y.values == 0) if c in MELSIDE else (lab.y.values == 1)) & (yt == 0)
    lo, hi = bounds(P[c].values, yt, M)
    rows.append(dict(setting="F4 (if explanation labels)", concept=c, side="mel" if c in MELSIDE else "nev",
                     naive=auc(P[c].values, yt), lower=lo, upper=hi, share_uncertain=M.mean()))
T = pd.DataFrame(rows)
T["contains_0.5"] = (T.lower <= 0.5) & (T.upper >= 0.5)
say(T.round(3).to_string(index=False))
say(T.groupby("setting")[["naive", "lower", "upper", "share_uncertain"]].mean().round(3).to_string())
T.to_csv(os.path.join(RES, "step20_bounds.csv"), index=False)
open(os.path.join(RES, "step20_log.txt"), "w").write("\n".join(LOG) + "\n")
