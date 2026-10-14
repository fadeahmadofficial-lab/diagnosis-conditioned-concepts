# Paper 2, step 19: reader x image (crossed) bootstrap for the Derma-XAI settings.
# Written 2026-10-14 BEFORE any number below was computed. Robustness of CIs only.
# Usage: python -I step19_reader_boot.py <step18_within_dx.py> <step8> <step9> <dx_out> <ko_old>
#        <ko_new> <p2> <easy> <result_dir>
# PRE-SPECIFIED
# Each replicate resamples images (2000 reps, as before) AND, independently, readers within
# each arm (aware: 14 annotators; blind: 113 participants). Weight of a read = read weight x
# image count x reader count. Statistics: F1 and F2 overall excess (all analysable concepts),
# and on the 8 concepts analysable in both arms the F2 - F1 overall excess and same-side dtheta.
# Rule: the paper keeps its statements if (a) the F1 excess CI still includes 0 and (b) the
# F2 - F1 overall excess CI stays above 0. If (b) fails, the paper states that the F1 vs F2
# overall difference is not robust to reader clustering. Widths are reported either way.
import sys
import numpy as np
import pandas as pd
src = open(sys.argv[1]).read()
sys.argv = ["s18"] + sys.argv[2:]
exec(src.split('say("=== step 18')[0])
LOG.clear()
r = pd.read_csv(os.path.join(DXO, "reads_test.csv"))
l1, P1, m1, C, _ = setting_dermaxai("aware", "effb0")
l2, P2_, m2, _, _ = setting_dermaxai("blind", "effb0")
rd1 = r[r.arm == "aware"].reader.astype(str).values
rd2 = r[r.arm == "blind"].reader.astype(str).values
assert len(rd1) == len(l1) and len(rd2) == len(l2)
assert (r[r.arm == "aware"].image_id.values == l1.image.values).all()
w1 = 1.0 / l1.groupby("image")["image"].transform("size").values
w2 = 1.0 / l2.groupby("image")["image"].transform("size").values
ok = lambda l, w, c: (l[c].values * w).sum() >= 20 and ((1 - l[c].values) * w).sum() >= 20
an1 = [c for c in C if ok(l1, w1, c)]
an2 = [c for c in C if ok(l2, w2, c)]
an = [c for c in C if ok(l1, w1, c) and ok(l2, w2, c)]
n1, n2 = nulls(l1, P1, m1, an1), nulls(l2, P2_, m2, an2)
codes = np.unique(np.r_[l1.image.values, l2.image.values])
i1, i2 = np.searchsorted(codes, l1.image.values), np.searchsorted(codes, l2.image.values)
R1, j1 = np.unique(rd1, return_inverse=True)
R2, j2 = np.unique(rd2, return_inverse=True)
say(f"aware readers {len(R1)}, blind readers {len(R2)}; concepts F1 {len(an1)}, F2 {len(an2)}, shared {len(an)}")

def ex(l, P, nl, w, cs):
    return np.mean([wauc(P[c].values, l[c].values, w) - wauc(nl[c], l[c].values, w) for c in cs])
def dth(l, P, nl, w, cs):
    v = [theta(P[c].values, l[c].values, l.y.values, w, (1,) if c in MELSIDE else (0,))[0]
         - theta(nl[c], l[c].values, l.y.values, w, (1,) if c in MELSIDE else (0,))[0] for c in cs]
    return np.nanmean(v)
def stat(a, b):
    return np.array([ex(l1, P1, n1, a, an1), ex(l2, P2_, n2, b, an2),
                     ex(l2, P2_, n2, b, an) - ex(l1, P1, n1, a, an),
                     dth(l2, P2_, n2, b, an) - dth(l1, P1, n1, a, an)])
pt = stat(w1, w2)
names = ["F1 excess", "F2 excess", "F2-F1 overall excess (shared)", "F2-F1 same-side dtheta (shared)"]
for mode in ("image only", "image x reader"):
    rng = np.random.default_rng(2026)
    bs = []
    for _ in range(2000):
        ci = np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))
        if mode == "image only":
            c1, c2 = np.ones(len(R1)), np.ones(len(R2))
        else:
            c1 = np.bincount(rng.integers(0, len(R1), len(R1)), minlength=len(R1))
            c2 = np.bincount(rng.integers(0, len(R2), len(R2)), minlength=len(R2))
        bs.append(stat(w1 * ci[i1] * c1[j1], w2 * ci[i2] * c2[j2]))
    bs = np.array(bs)
    say(f"--- {mode}")
    for k, nm in enumerate(names):
        lo, hi = np.nanpercentile(bs[:, k], [2.5, 97.5])
        say(f"  {nm:34s} {pt[k]:+.3f} [{lo:+.3f}, {hi:+.3f}]  width {hi - lo:.3f}")
open(os.path.join(RES, "step19_log.txt"), "w").write("\n".join(LOG) + "\n")
