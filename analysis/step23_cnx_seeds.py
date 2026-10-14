# Paper 2, step 23: ConvNeXt-Tiny on Derma-XAI with 3 seeds (0, 1, 2).
# Written 2026-10-14 BEFORE the seed 1-2 predictions exist (Kaggle run in progress).
# Tested on seed 0 alone, which must reproduce step 14 (F1 +0.036 [+0.007, +0.066]).
#
# Usage: python -I step23_cnx_seeds.py <step18_within_dx.py> <step8> <step9> <dx_out> <ko_old>
#        <ko_cnx_dir> <p2> <easy> <result_dir>
#   <ko_cnx_dir> holds test_convnext_tiny_seed*.csv and easy_convnext_tiny_seed*.csv for the
#   seeds to use (seed 0 from d1_easy_out, seeds 1-2 from d1_cnx_seeds.zip).
#
# PRE-SPECIFIED
# Predictions: mean over the available seeds (concepts and black-box p), as for EfficientNet-B0.
# Statistics (same code as steps 8, 18, 19, 22):
#   F1 (aware), F2 (blind), F3 (EASY, primary map) excess with image bootstrap, and with
#   image x reader bootstrap; F1 and F3 own-side dtheta; per-seed F1/F2/F3 excess (points).
# Reporting rules:
#   R1  The paper's ConvNeXt F1, F2 and F3 values are replaced by the 3-seed means. T2 and B1
#       for ConvNeXt stay single-seed (only seed 0 has a masked model) and are labelled so.
#   R2  The ConvNeXt F1 excess is called "above zero" only if the lower limit of its image x
#       reader interval is above 0; otherwise the paper says its interval includes zero.
#   R3  If the per-seed F1 excess values span more than 0.03, the paper reports the range
#       and calls the ConvNeXt F1 result seed-sensitive.
#   No conclusion of the paper depends on these values; they replace a one-seed sensitivity.
import glob, json, os, shutil, sys
import numpy as np
import pandas as pd
S18 = sys.argv[1]
S8, S9, DXO, KO_OLD, KOC, P2D, EASYD, RES = sys.argv[2:10]
os.makedirs(RES, exist_ok=True)
src = open(S18).read()
sys.argv = ["s18", S8, S9, DXO, KO_OLD, KOC, P2D, EASYD, RES]
exec(src.split('say("=== step 18')[0])
LOG.clear()
seeds = sorted(int(f.split("seed")[-1].split(".")[0]) for f in glob.glob(os.path.join(KOC, "test_convnext_tiny_seed*.csv")))
eseeds = sorted(int(f.split("seed")[-1].split(".")[0]) for f in glob.glob(os.path.join(KOC, "easy_convnext_tiny_seed*.csv")))
assert seeds == eseeds, (seeds, eseeds)
say(f"ConvNeXt seeds found: {seeds}")
r = pd.read_csv(os.path.join(DXO, "reads_test.csv"))
em = pd.read_csv(os.path.join(EASYD, "metadata.csv"))
keep = set(em[em.diagnosis.isin(["melanoma", "nevus"])].ISIC_id)
mk = json.load(open(os.path.join(EASYD, "markups.json")))
easy_rid = np.array([q for iid, rd in mk.items() if iid in keep for q in rd])


def crossed(lab, P, means, C, readers):
    w0 = 1.0 / lab.groupby("image")["image"].transform("size").values
    an = [c for c in C if (lab[c].values * w0).sum() >= 20 and ((1 - lab[c].values) * w0).sum() >= 20]
    nl = nulls(lab, P, means, an)
    f = lambda w: np.mean([wauc(P[c].values, lab[c].values, w) - wauc(nl[c], lab[c].values, w) for c in an])
    codes, idx = np.unique(lab.image.values, return_inverse=True)
    Rr, j = np.unique(readers, return_inverse=True)
    rng = np.random.default_rng(2026)
    bs = [f(w0 * np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[idx]
            * np.bincount(rng.integers(0, len(Rr), len(Rr)), minlength=len(Rr))[j]) for _ in range(B)]
    return f(w0), np.nanpercentile(bs, [2.5, 97.5])


def settings(ko):
    global KO_NEW
    out = {}
    for arm, tag in (("aware", "F1"), ("blind", "F2")):
        files = sorted(glob.glob(os.path.join(ko, "test_convnext_tiny_seed*.csv")))
        P = sum(pd.read_csv(f).set_index("image_id")[C18 + ["p_mel_blackbox"]] for f in files) / len(files)
        P = P.rename(columns={"p_mel_blackbox": "p"})
        rr = r[r.arm == arm].reset_index(drop=True)
        tr = pd.read_csv(os.path.join(DXO, "train_labels.csv")); tr = tr[tr.split == "train"]
        means = {None: {c: (tr[tr.benign_malignant == 0][c].mean(), tr[tr.benign_malignant == 1][c].mean()) for c in C18}}
        lab = rr[["image_id", "y"] + C18].rename(columns={"image_id": "image"})
        out[tag] = (lab, P.loc[lab.image.values].reset_index(drop=True), means, C18, rr.reader.astype(str).values)
    KO_NEW = ko
    lab, P, means, C, _ = setting_s5(MAP_P, "convnext_tiny")
    out["F3"] = (lab, P, means, C, easy_rid)
    return out


S = settings(KOC)
rows = []
for tag in ("F1", "F2", "F3"):
    lab, P, means, C, rd = S[tag]
    res, _ = recipe(lab, P, means, C, f"{tag} ConvNeXt, seeds {seeds}")
    pt, (lo, hi) = crossed(lab, P, means, C, rd)
    say(f"  {tag} image x reader: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    row = dict(setting=tag, seeds=str(seeds), excess=pt, xr_lo=lo, xr_hi=hi,
               img_lo=[x for x in res if x["subset"] == "all"][0]["lo"], img_hi=[x for x in res if x["subset"] == "all"][0]["hi"])
    if tag in ("F1", "F3"):
        rr_, _, _ = analyse(lab, P, means, C, f"{tag} ConvNeXt dtheta")
        row.update(dtheta_same=[x for x in rr_ if x["version"] == "same"][0]["dtheta"])
    rows.append(row)
per = []
for s in seeds:
    d = os.path.join(RES, f"tmp_cnx_seed{s}"); os.makedirs(d, exist_ok=True)
    for kind in ("test", "easy"):
        shutil.copy(os.path.join(KOC, f"{kind}_convnext_tiny_seed{s}.csv"), d)
    Ss = settings(d)
    for tag in ("F1", "F2", "F3"):
        lab, P, means, C, _ = Ss[tag]
        w0 = 1.0 / lab.groupby("image")["image"].transform("size").values
        an = [c for c in C if (lab[c].values * w0).sum() >= 20 and ((1 - lab[c].values) * w0).sum() >= 20]
        nl = nulls(lab, P, means, an)
        e = np.mean([wauc(P[c].values, lab[c].values, w0) - wauc(nl[c], lab[c].values, w0) for c in an])
        img = pd.DataFrame({"i": lab.image.values, "p": P.p.values, "y": lab.y.values}).groupby("i").first()
        per.append(dict(seed=s, setting=tag, excess=e, p_auroc=wauc(img.p.values, img.y.values, np.ones(len(img)))))
        say(f"  seed {s} {tag}: excess {e:+.3f} (black-box AUROC {per[-1]['p_auroc']:.3f})")
KO_NEW = KOC
T = pd.DataFrame(rows); Pp = pd.DataFrame(per)
f1 = T[T.setting == "F1"].iloc[0]; sp = Pp[Pp.setting == "F1"].excess
say(f"R2: F1 ConvNeXt excess {'ABOVE zero' if f1.xr_lo > 0 else 'interval INCLUDES zero'} "
    f"({f1.excess:+.3f} [{f1.xr_lo:+.3f}, {f1.xr_hi:+.3f}])")
say(f"R3: per-seed F1 range {sp.min():+.3f} to {sp.max():+.3f} -> "
    f"{'seed-sensitive' if sp.max() - sp.min() > 0.03 else 'within 0.03'}")
T.to_csv(os.path.join(RES, "step23_summary.csv"), index=False)
Pp.to_csv(os.path.join(RES, "step23_per_seed.csv"), index=False)
open(os.path.join(RES, "step23_log.txt"), "w").write("\n".join(LOG) + "\n")
