# Step 9b (EXPLORATORY, added after step 9): same model (Derma-XAI EffB0), same 4 concepts
# (TRBL, GP, PV, WLSA), two references: aware test reads (S1) vs EASY blinded experts (S5).
# Difference in excess; images differ, so the two bootstraps are independent.
import sys, os
# Usage: python -I step9b_diff.py <step8_recipe.py> <step9_dxai_on_easy.py> <dx_out> <kaggle_old>
#        <kaggle_new> <p2_dir> <easy_dir> <scratch_dir>
S8, S9 = sys.argv[1:3]
argv9 = ["x", S8] + sys.argv[3:9]
os.makedirs(argv9[-1], exist_ok=True)
src9 = open(S9).read().split("# ---------------------------------------------------------------- (A)")[0]
sys.argv = argv9
exec(src9)
src9b = open(S9).read()
exec(src9b[src9b.index("MAP_P = "):src9b.index("for mapping, mname")])
import numpy as np
COM = ["TRBL", "GP", "PV", "WLSA"]

def boot_excess(lab, P, means, seed):
    w = 1.0 / lab.groupby("image")["image"].transform("size").values
    p = P.p.values
    nul = {c: p * means[None][c][1] + (1 - p) * means[None][c][0] for c in COM}
    codes, idx = np.unique(lab.image.values, return_inverse=True)
    f = lambda ww: np.mean([wauc(P[c].values, lab[c].values, ww) - wauc(nul[c], lab[c].values, ww) for c in COM])
    rng = np.random.default_rng(seed)
    pt = f(w)
    bs = np.array([f(w * np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx]) for _ in range(2000)])
    return pt, bs

l1, P1, m1, _, _ = setting_dermaxai("aware", "effb0")
l5, P5, m5, _, _ = setting_s5(MAP_P, "effb0")
a, ba = boot_excess(l1, P1, m1, 1)
b, bb = boot_excess(l5, P5, m5, 2)
d = bb - ba
print(f"S1 aware ref excess {a:+.3f} [{np.percentile(ba,2.5):+.3f}, {np.percentile(ba,97.5):+.3f}]")
print(f"S5 EASY  ref excess {b:+.3f} [{np.percentile(bb,2.5):+.3f}, {np.percentile(bb,97.5):+.3f}]")
print(f"difference S5 - S1  {b-a:+.3f} [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]")
