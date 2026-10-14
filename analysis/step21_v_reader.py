# Paper 2, step 21: V test under reader clustering. Written 2026-10-14 BEFORE computing.
# PRE-SPECIFIED: the step 10 test (excess at pi=1 minus excess at pi=0, scenario T, EASY,
# Derm7pt EffB4) with (a) image x reader crossed bootstrap (2000 reps; 5 EASY readers) and
# (b) leave-one-reader-out point values. [Note added after running: EASY has 21 readers, 5 per image, not 5 in total; the header line below about "only 5 readers" is wrong.]. V keeps its "detected" status in the paper only if
# the crossed CI stays below 0 and all five leave-one-out values are below 0. Otherwise the
# paper says V is not robust to reader clustering. With only 5 readers the crossed bootstrap
# is coarse; this is stated alongside the result.
import json, os, sys
import numpy as np
import pandas as pd
s10 = open(sys.argv[1]).read()
sys.argv = ["s10"] + sys.argv[2:]
exec(s10.split("say(f\"V: EASY images")[0])
LOG.clear()
em = pd.read_csv(os.path.join(EASY, "metadata.csv"))
em = em[em.diagnosis.isin(["melanoma", "nevus"])]
keep = set(em.ISIC_id)
mk = json.load(open(os.path.join(EASY, "markups.json")))
rid = np.array([r for iid, rd in mk.items() if iid in keep for r in rd])
img = np.array([iid for iid, rd in mk.items() if iid in keep for r in rd])
assert (img == lab0.image.values).all()
R, j = np.unique(rid, return_inverse=True)
say(f"EASY readers: {list(R)}; reads per reader {np.bincount(j).tolist()}")
lab1 = censor(lab0, SUPPLIED["T"], 1.0, np.random.default_rng(0))
def dV(ww):
    return metrics(lab1, ww, SUPPLIED["T"], False)["all_excess"] - metrics(lab0, ww, SUPPLIED["T"], False)["all_excess"]
pt = dV(w0)
rng = np.random.default_rng(2026)
bs = []
for _ in range(2000):
    ci = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)[idx]
    cr = np.bincount(rng.integers(0, len(R), len(R)), minlength=len(R))[j]
    bs.append(dV(w0 * ci * cr))
lo, hi = np.nanpercentile(bs, [2.5, 97.5])
say(f"V delta {pt:+.3f}; image x reader 95% CI [{lo:+.3f}, {hi:+.3f}]")
loro = []
for k, rr in enumerate(R):
    m = j != k
    ww = np.where(m, 1.0, 0.0)
    ww = ww / np.maximum(pd.Series(ww).groupby(idx).transform("sum").values, 1e-12) * (ww > 0)
    loro.append(dV(ww)); say(f"  without {rr}: {loro[-1]:+.3f}")
ok_ = hi < 0 and max(loro) < 0
say(f"RULE: {'V stands' if ok_ else 'V NOT robust to reader clustering'}")
open(os.path.join(RES, "step21_log.txt"), "w").write("\n".join(LOG) + "\n")
