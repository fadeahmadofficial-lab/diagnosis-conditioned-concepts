# Step 19b (added 2026-10-14 after step 19, exploratory): F1 same-side dtheta, image x reader bootstrap.
import sys
import numpy as np
src = open(sys.argv[1]).read()
sys.argv = sys.argv[1:]
exec(src.split("pt = stat(w1, w2)")[0])
f = lambda a: dth(l1, P1, n1, a, an1)
rng = np.random.default_rng(2026)
bs = [f(w1 * np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[i1]
        * np.bincount(rng.integers(0, len(R1), len(R1)), minlength=len(R1))[j1]) for _ in range(2000)]
print(f"F1 same-side dtheta {f(w1):+.3f}; image x reader CI {np.nanpercentile(bs, [2.5, 97.5]).round(3)}")
