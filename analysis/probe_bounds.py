# Feasibility probe for the proposal's Candidate 2 (censoring-identification bounds).
# NOT a paper result: a quick look at whether worst-case bounds are informative at all.
import sys, itertools
import numpy as np, pandas as pd
S8 = sys.argv[1]
src = open(S8).read().split("ALL, PER = [], []")[0]
sys.argv = ["step8"] + sys.argv[2:7]
exec(src)

def bounds(q, ytil, M, w):
    """worst-case [L,U] over all k; M boolean (uncertain cells, recorded 0)."""
    idx = np.where(M)[0]
    order = idx[np.argsort(q[idx], kind="stable")]
    L, U = [], []
    for k in range(len(order) + 1):
        y1 = ytil.copy(); y1[order[:k]] = 1; L.append(wauc(q, y1, w))
        y2 = ytil.copy();
        if k: y2[order[-k:]] = 1
        U.append(wauc(q, y2, w))
    return np.nanmin(L), np.nanmax(U), L, U

# lemma check, unweighted and weighted
rng = np.random.default_rng(0); viol = {"unw": 0, "w": 0}; cases = 0
for _ in range(1500):
    n = rng.integers(4, 11); q = rng.integers(0, 4, n).astype(float)
    yt = (rng.random(n) < .3).astype(int); M = (yt == 0) & (rng.random(n) < .5)
    if M.sum() == 0: continue
    for wt, key in ((np.ones(n), "unw"), (rng.choice([1, .5, 1/3], n), "w")):
        mi = np.where(M)[0]
        for k in range(len(mi) + 1):
            vals = []
            for S in itertools.combinations(mi, k):
                y = yt.copy(); y[list(S)] = 1; vals.append(wauc(q, y, wt))
            vals = [v for v in vals if not np.isnan(v)]
            if not vals: continue
            cases += 1
            o = mi[np.argsort(q[mi], kind="stable")]
            y = yt.copy(); y[o[:k]] = 1; lo = wauc(q, y, wt)
            y = yt.copy();
            if k: y[o[-k:]] = 1
            hi = wauc(q, y, wt)
            if lo > min(vals) + 1e-12 or hi < max(vals) - 1e-12: viol[key] += 1
print("lemma check (fixed k):", cases, "cases, violations", viol)

D = json.load(open(os.path.join(DX, "definitions.json"))); MEL = set(D["MEL"])
for arm in ("aware", "blind"):
    lab, P, means, C, _ = setting_dermaxai(arm, "effb0")
    r = pd.read_csv(os.path.join(DX, "reads_test.csv")); r = r[r.arm == arm].reset_index(drop=True)
    cond = r["own_dx"].values if (arm == "blind" and "own_dx" in r) else lab.y.values
    w = 1.0 / lab.groupby("image")["image"].transform("size").values
    print(f"\n=== F{1 if arm=='aware' else 2} Derma-XAI EffB0 -> {arm} reads (conditioning dx = {'own call' if arm=='blind' and 'own_dx' in r else 'truth'})")
    print(f"{'concept':6s} side  naive   L      U     |M|/n")
    for c in C:
        yt = lab[c].values.astype(int)
        if (yt * w).sum() < 20 or ((1 - yt) * w).sum() < 20: continue
        opp = (cond == 0) if c in MEL else (cond == 1)
        M = opp & (yt == 0)
        lo, hi, _, _ = bounds(P[c].values, yt, M, w)
        print(f"{c:6s} {'mel' if c in MEL else 'nev'}  {wauc(P[c].values, yt, w):.3f}  {lo:.3f}  {hi:.3f}  {M.mean():.2f}")
print("\nreads_test columns:", list(pd.read_csv(os.path.join(DX, 'reads_test.csv')).columns)[:30])
