# Step 18b, EXPLORATORY (added 2026-10-14 after step 18, not pre-registered).
# F1 vs F2 share the images, the model and the black-box p, so their difference is free of the
# p-quality and image-population confounds found in step 18 H2. Paired image bootstrap.
import sys
import numpy as np
src = open(sys.argv[1]).read()
sys.argv = ["s18"] + sys.argv[2:]
exec(src.split('say("=== step 18')[0])
l1, P1, m1, C, _ = setting_dermaxai("aware", "effb0")
l2, P2, m2, _, _ = setting_dermaxai("blind", "effb0")
w1 = 1.0 / l1.groupby("image")["image"].transform("size").values
w2 = 1.0 / l2.groupby("image")["image"].transform("size").values
an = [c for c in C if all((l[c].values * w).sum() >= 20 and ((1 - l[c].values) * w).sum() >= 20
                          for l, w in ((l1, w1), (l2, w2)))]
n1, n2 = nulls(l1, P1, m1, an), nulls(l2, P2, m2, an)
codes = np.unique(np.r_[l1.image.values, l2.image.values])
i1, i2 = np.searchsorted(codes, l1.image.values), np.searchsorted(codes, l2.image.values)

def stat(cnt):
    a, b = w1 * cnt[i1], w2 * cnt[i2]
    ex = lambda l, P, nl, w: np.mean([wauc(P[c].values, l[c].values, w) - wauc(nl[c], l[c].values, w) for c in an])
    th = lambda l, P, nl, w: np.mean([theta(P[c].values, l[c].values, l.y.values, w, (1,) if c in MELSIDE else (0,))[0]
                                      - theta(nl[c], l[c].values, l.y.values, w, (1,) if c in MELSIDE else (0,))[0] for c in an])
    return np.array([ex(l1, P1, n1, a), ex(l2, P2, n2, b), th(l1, P1, n1, a), th(l2, P2, n2, b)])

pt = stat(np.ones(len(codes)))
rng = np.random.default_rng(2026)
bs = np.array([stat(np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))) for _ in range(2000)])
print("concepts analysable in both arms:", an)
for j, name in ((0, "overall excess"), (2, "same-side dtheta")):
    d = bs[:, j + 1] - bs[:, j]
    print(f"{name:17s} F1 {pt[j]:+.3f}  F2 {pt[j+1]:+.3f}  F2-F1 {pt[j+1]-pt[j]:+.3f} "
          f"[{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}]")
