# Paper 2, step 22: F3 and F5 excess under image x reader bootstrap (21 EASY readers).
# Written 2026-10-14 BEFORE computing. Rule: F3/F5 statements stand if CIs stay above 0.
import json, os, sys
import numpy as np, pandas as pd
src = open(sys.argv[1]).read()
sys.argv = ["s18"] + sys.argv[2:]
exec(src.split('say("=== step 18')[0])
LOG.clear()
em = pd.read_csv(os.path.join(EASYD, "metadata.csv"))
keep = set(em[em.diagnosis.isin(["melanoma", "nevus"])].ISIC_id)
mk = json.load(open(os.path.join(EASYD, "markups.json")))
rid = np.array([r for iid, rd in mk.items() if iid in keep for r in rd])
for fn, name in ((lambda: setting_s5(MAP_P, "effb0"), "F3"), (lambda: setting_easy("effb4"), "F5")):
    lab, P, means, C, _ = fn()
    assert len(rid) == len(lab)
    w0 = 1.0 / lab.groupby("image")["image"].transform("size").values
    an = [c for c in C if (lab[c].values * w0).sum() >= 20 and ((1 - lab[c].values) * w0).sum() >= 20]
    nl = nulls(lab, P, means, an)
    f = lambda w: np.mean([wauc(P[c].values, lab[c].values, w) - wauc(nl[c], lab[c].values, w) for c in an])
    codes, idx = np.unique(lab.image.values, return_inverse=True)
    R, j = np.unique(rid, return_inverse=True)
    rng = np.random.default_rng(2026)
    bs = [f(w0 * np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx]
            * np.bincount(rng.integers(0, len(R), len(R)), minlength=len(R))[j]) for _ in range(2000)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    say(f"{name} excess {f(w0):+.3f}; image x reader CI [{lo:+.3f}, {hi:+.3f}] -> {'stands' if lo > 0 else 'NOT robust'}")
open(os.path.join(RES, "step22_log.txt"), "w").write("\n".join(LOG) + "\n")
