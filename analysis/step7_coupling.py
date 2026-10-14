# Paper 2, component C: diagnosis-coupling index across public concept-label sets.
# Written BEFORE any number below was computed (2026-10-08). Descriptive, not confirmatory.
#
# Usage: python -I step7_coupling.py <dermaxai_step1_out> <easy_dir> <release_v0_dir> <result_dir>
#
# PRE-SPECIFIED DEFINITIONS
# Task. Melanocytic sets: primary melanoma (y=1) vs naevus (y=0). Metastases, AIMP and
#   non-melanocytic lesions are excluded. SkinCon: (a) fine diagnosis (multi-class),
#   (b) malignant vs not.
# Unit. A "read" = one reader's labels on one image. Multi-reader sets are analysed at read
#   level with weight 1/(reads on that image), so each image counts once (n = images).
# Analysable concept: >= 20 weighted positives AND >= 20 weighted negatives AND positive
#   share >= 1%; the same rule for every dataset.
# Metric 1, coupling eps2: adjusted share of concept-label variance explained by diagnosis
#   (epsilon-squared; for a binary diagnosis ~ phi^2). 0 = labels unrelated to diagnosis,
#   1 = labels fully determined by diagnosis.
# Metric 2, diagnosis-only null AUROC: AUROC of the score E[c | diagnosis] (class means
#   cross-fitted over 5 image folds) against the concept labels. It is the concept AUROC a
#   "model" that knows only the diagnosis would get. Any concept-AUROC claim should beat it.
# Metric 3, cross-side marking (sided vocabularies only): share of naevus reads with >= 1
#   melanoma-side concept, and of melanoma reads with >= 1 naevus-side concept.
# Metric 4, within-class reliability (multi-reader sets only): Krippendorff alpha (binary)
#   within naevi and within melanomas, mean over analysable concepts.
# Dataset summary = mean over analysable concepts; 95% CI = image bootstrap, 2000 reps.
# Caveats fixed in advance: coupling is NOT bias (real features co-vary with diagnosis);
#   datasets differ in images, case mix, vocabulary and reader count; only Derma-XAI aware
#   vs blind is a same-image contrast.
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DX, EASY, REL, RES = sys.argv[1:5]
os.makedirs(RES, exist_ok=True)
B = 2000
LOG = []


def say(m):
    print(m, flush=True)
    LOG.append(str(m))


# ------------------------------------------------------------------ loaders
# Each loader returns: reads DataFrame with columns image, y, and 0/1 concept columns;
# plus dict(mel_side=[...], nev_side=[...]) or None.
def load_dermaxai():
    D = json.load(open(os.path.join(DX, "definitions.json")))
    C = D["MEL"] + D["NEV"]
    r = pd.read_csv(os.path.join(DX, "reads_test.csv")).drop(columns=["y"])
    t = pd.read_csv(os.path.join(DX, "test_images.csv"))[["image_id", "benign_malignant"]]
    r = r.merge(t, on="image_id").rename(columns={"image_id": "image", "benign_malignant": "y"})
    out = {}
    for arm in ("aware", "blind"):
        out["Derma-XAI test, " + arm] = (r[r.arm == arm][["image", "y", "reader"] + C].copy(),
                                         dict(mel=D["MEL"], nev=D["NEV"]))
    tr = pd.read_csv(os.path.join(DX, "train_labels.csv")).rename(
        columns={"image_id": "image", "benign_malignant": "y"})
    tr["reader"] = "train"
    out["Derma-XAI train labels"] = (tr[["image", "y", "reader"] + C], dict(mel=D["MEL"], nev=D["NEV"]))
    return out


def load_derm7pt():
    m = pd.read_csv(os.path.join(REL, "meta", "meta.csv"))
    dx = m.diagnosis.str.lower()
    mel = dx.str.startswith("melanoma") & ~dx.str.contains("metastasis")
    nev = dx.str.contains("nevus")
    m = m[mel | nev].copy()
    m["y"] = mel[mel | nev].astype(int)
    f = lambda col, vals: m[col].isin(vals).astype(int)
    df = pd.DataFrame({
        "image": m.case_num.values, "y": m.y.values, "reader": "atlas",
        "PN_atyp": f("pigment_network", ["atypical"]),
        "STR_irr": f("streaks", ["irregular"]),
        "PIG_irr": f("pigmentation", ["diffuse irregular", "localized irregular"]),
        "DaG_irr": f("dots_and_globules", ["irregular"]),
        "RS": f("regression_structures", ["blue areas", "white areas", "combinations"]),
        "BWV": f("blue_whitish_veil", ["present"]),
        "VS_atyp": f("vascular_structures", ["dotted", "linear irregular"]),
        "PN_typ": f("pigment_network", ["typical"]),
        "STR_reg": f("streaks", ["regular"]),
        "PIG_reg": f("pigmentation", ["diffuse regular", "localized regular"]),
        "DaG_reg": f("dots_and_globules", ["regular"]),
    })
    return df, dict(mel=["PN_atyp", "STR_irr", "PIG_irr", "DaG_irr", "RS", "BWV", "VS_atyp"],
                    nev=["PN_typ", "STR_reg", "PIG_reg", "DaG_reg"])


EASY_MAP = {  # melanoma-side: Paper-1 mapping (code/easy_common.py MAP); naevus-side added here
    "PN_atyp": ["Network : Atypical pigment network / Reticulation",
                "Network : Broadened pigment network / Reticulation"],
    "STR_irr": ["Lines : Pseudopods", "Lines : Radial streaming"],
    "PIG_irr": ["Structureless : Blotch irregular"],
    "DaG_irr": ["Dots : Irregular", "Globules / Clods : Irregular"],
    "RS": ["Regression structures : Peppering / Granularity",
           "Regression structures : Scarlike depigmentation"],
    "BWV": ["Structureless : Blue-whitish veil"],
    "VS_atyp": ["Vessels : Dotted", "Vessels : Linear irregular", "Vessels : Polymorphous"],
    "PN_typ": ["Network : Typical pigment network / Reticulation",
               "Network : Delicate Pigment Network / Reticulation"],
    "PIG_reg": ["Structureless : Blotch regular"],
    "DaG_reg": ["Dots : Regular", "Globules / Clods : Regular"],
}


def load_easy():
    meta = pd.read_csv(os.path.join(EASY, "metadata.csv"))
    meta = meta[meta.diagnosis.isin(["melanoma", "nevus"])]
    y = dict(zip(meta.ISIC_id, (meta.diagnosis == "melanoma").astype(int)))
    mk = json.load(open(os.path.join(EASY, "markups.json")))
    rows = []
    for iid, rd in mk.items():
        if iid not in y:
            continue
        for rid, feats in rd.items():
            present = {k for k, v in feats.items() if len(v)}
            rows.append({"image": iid, "y": y[iid], "reader": rid,
                         **{c: int(any(t in present for t in ts)) for c, ts in EASY_MAP.items()}})
    df = pd.DataFrame(rows)
    return df, dict(mel=["PN_atyp", "STR_irr", "PIG_irr", "DaG_irr", "RS", "BWV", "VS_atyp"],
                    nev=["PN_typ", "PIG_reg", "DaG_reg"])


def load_ph2():
    rows = []
    for line in open(os.path.join(REL, "PH2", "PH2_dataset.txt")):
        if "IMD" not in line:
            continue
        p = [x.strip() for x in line.strip().strip("|").replace("||", "|").split("|")]
        name, hist, clin, asym, pn, dg, st, rs, bwv = p[:9]
        rows.append(dict(image=name, clin=int(clin), PN=pn, DG=dg, ST=st, RS=rs, BWV=bwv))
    m = pd.DataFrame(rows)
    df = pd.DataFrame({
        "image": m.image, "y": (m.clin == 2).astype(int), "reader": "expert",
        "PN_atyp": (m.PN == "AT").astype(int), "DaG_irr": (m.DG == "AT").astype(int),
        "STR_irr": (m.ST == "P").astype(int),  # PH2 records streaks present, not irregular
        "RS": (m.RS == "P").astype(int), "BWV": (m.BWV == "P").astype(int),
        "PN_typ": (m.PN == "T").astype(int), "DaG_reg": (m.DG == "T").astype(int)})
    return df, dict(mel=["PN_atyp", "DaG_irr", "STR_irr", "RS", "BWV"], nev=["PN_typ", "DaG_reg"])


def load_skincon():
    out = {}
    a = pd.read_csv(os.path.join(REL, "SkinCon", "annotations_fitzpatrick17k.csv"))
    f = pd.read_csv(os.path.join(REL, "SkinCon", "fitzpatrick17k.csv"))
    f["ImageID"] = f.md5hash + ".jpg"
    m = a.merge(f[["ImageID", "label", "three_partition_label"]], on="ImageID")
    C = [c for c in a.columns if c not in ("Unnamed: 0", "ImageID", "Do not consider this image")]
    m = m[m["Do not consider this image"] == 0]
    for nm, ycol in (("fine diagnosis", m.label), ("malignant vs not", (m.three_partition_label == "malignant").astype(int))):
        d = m[["ImageID"] + C].rename(columns={"ImageID": "image"}).copy()
        d["y"] = ycol.values
        d["reader"] = "skincon"
        out["SkinCon-Fitz17k, " + nm] = (d, None)
    a = pd.read_csv(os.path.join(REL, "SkinCon", "annotations_ddi.csv"))
    md = pd.read_csv(os.path.join(REL, "SkinCon", "ddi_metadata.csv"))
    m = a.merge(md, left_on="ImageID", right_on="DDI_file")
    m = m[m["Do not consider this image"] == 0]
    for nm, ycol in (("fine diagnosis", m.disease), ("malignant vs not", m.malignant.astype(int))):
        d = m[["ImageID"] + C].rename(columns={"ImageID": "image"}).copy()
        d["y"] = ycol.values
        d["reader"] = "skincon"
        out["SkinCon-DDI, " + nm] = (d, None)
    return out


# ------------------------------------------------------------------ metrics
def prep(df, concepts):
    df = df.copy()
    df["w"] = 1.0 / df.groupby("image")["image"].transform("size")
    imgs = df.image.unique()
    rng = np.random.default_rng(11)
    fold = dict(zip(imgs, rng.integers(0, 5, len(imgs))))
    df["fold"] = df.image.map(fold)
    an = [c for c in concepts
          if (df[c] * df.w).sum() >= 20 and ((1 - df[c]) * df.w).sum() >= 20
          and (df[c] * df.w).sum() / df.w.sum() >= 0.01]
    return df, an


def eps2(c, g, w):
    n = w.sum()
    K = pd.Series(g).nunique()
    mu = np.average(c, weights=w)
    sst = (w * (c - mu) ** 2).sum()
    gm = pd.Series(w * c).groupby(g).sum() / pd.Series(w).groupby(g).sum()
    ssw = (w * (c - pd.Series(g).map(gm).values) ** 2).sum()
    if sst == 0 or n <= K:
        return np.nan
    return 1 - (ssw / (n - K)) / (sst / (n - 1))


def null_auc(c, g, w, fold):
    s = np.zeros(len(c))
    for k in range(5):
        tr, te = fold != k, fold == k
        gm = (pd.Series(w[tr] * c[tr]).groupby(g[tr]).sum()
              / pd.Series(w[tr]).groupby(g[tr]).sum())
        s[te] = pd.Series(g[te]).map(gm).fillna(np.average(c[tr], weights=w[tr])).values
    if c.min() == c.max():
        return np.nan
    return roc_auc_score(c, s, sample_weight=w)


def alpha_binary(df, c):
    g = df.groupby("image")[c].agg(["sum", "count"])
    g = g[g["count"] >= 2]
    n1, m = g["sum"].values.astype(float), g["count"].values.astype(float)
    n0 = m - n1
    n, N1, N0 = m.sum(), n1.sum(), n0.sum()
    if N1 == 0 or N0 == 0:
        return np.nan
    return 1 - (n - 1) * (2 * n1 * n0 / (m - 1)).sum() / (2 * N1 * N0)


def summarise(df, an, sides, multi_reader, with_null=True):
    c_ = {}
    y = df.y.values
    w = df.w.values
    e = [eps2(df[c].values.astype(float), y, w) for c in an]
    c_["eps2"] = np.nanmean(e)
    if with_null:
        nu = [null_auc(df[c].values, y, w, df.fold.values) for c in an]
        c_["null_auc"] = np.nanmean(nu)
    if sides:
        mel = [c for c in sides["mel"] if c in df]
        nev = [c for c in sides["nev"] if c in df]
        anym = df[mel].max(axis=1)
        anyn = df[nev].max(axis=1)
        c_["cross_nev_has_mel"] = np.average(anym[df.y == 0], weights=w[df.y.values == 0])
        c_["cross_mel_has_nev"] = np.average(anyn[df.y == 1], weights=w[df.y.values == 1])
    if multi_reader:
        for cls, nm in ((0, "alpha_within_nev"), (1, "alpha_within_mel")):
            s = df[df.y == cls]
            c_[nm] = np.nanmean([alpha_binary(s, c) for c in an])
        c_["alpha_all"] = np.nanmean([alpha_binary(df, c) for c in an])
    return c_, dict(zip(an, e)), (dict(zip(an, nu)) if with_null else {})


def boot(df, an, sides, multi_reader):
    imgs = df.image.unique()
    idx = {i: np.where(df.image.values == i)[0] for i in imgs}
    rng = np.random.default_rng(2026)
    res = []
    for b in range(B):
        pick = rng.choice(imgs, len(imgs), replace=True)
        rows = np.concatenate([idx[i] for i in pick])
        d = df.iloc[rows].copy()
        # duplicated images must stay distinct units for alpha
        d["image"] = np.repeat(np.arange(len(pick)), [len(idx[i]) for i in pick])
        res.append(summarise(d, an, sides, multi_reader, with_null=False)[0])
    return pd.DataFrame(res)


# ------------------------------------------------------------------ run
sets = {}
sets.update(load_dermaxai())
sets["Derm7pt (atlas)"] = load_derm7pt()
sets["PH2 (one expert)"] = load_ph2()
sets["EASY (5 blinded experts)"] = load_easy()
sets.update(load_skincon())

summary, per_concept = [], []
for name, (df, sides) in sets.items():
    concepts = [c for c in df.columns if c not in ("image", "y", "reader")]
    df, an = prep(df, concepts)
    multi = df.groupby("image").size().max() > 1
    s, e, nu = summarise(df, an, sides, multi)
    is_binary = df.y.nunique() == 2
    bt = boot(df, an, sides, multi) if is_binary else None
    row = dict(dataset=name, images=df.image.nunique(), reads=len(df),
               classes=df.y.nunique(), mel_share=(df.groupby("image").y.first().mean() if is_binary else np.nan),
               concepts_total=len(concepts), concepts_analysable=len(an), **s)
    if bt is not None:
        for k in bt.columns:
            row[k + "_lo"], row[k + "_hi"] = np.nanpercentile(bt[k], [2.5, 97.5])
    summary.append(row)
    for c in an:
        side = ("mel" if sides and c in sides["mel"] else "nev" if sides and c in sides["nev"] else "")
        wpos = (df[c] * df.w).sum() / df.w.sum()
        per_concept.append(dict(dataset=name, concept=c, side=side, prevalence=wpos,
                                eps2=e[c], null_auc=nu[c]))
    say(f"--- {name}: images {row['images']}, reads {row['reads']}, classes {row['classes']}, "
        f"analysable {len(an)}/{len(concepts)}")
    for k, v in s.items():
        ci = (f" [{row[k + '_lo']:.3f}, {row[k + '_hi']:.3f}]" if (k + "_lo") in row else "")
        say(f"    {k}: {v:.3f}{ci}")

S = pd.DataFrame(summary)
S.to_csv(os.path.join(RES, "coupling_summary.csv"), index=False)
pd.DataFrame(per_concept).to_csv(os.path.join(RES, "coupling_per_concept.csv"), index=False)

# Same-image contrast: Derma-XAI aware - blind, paired image bootstrap
a_df, sides = sets["Derma-XAI test, aware"]
b_df, _ = sets["Derma-XAI test, blind"]
C = [c for c in a_df.columns if c not in ("image", "y", "reader")]
a_df, an_a = prep(a_df, C)
b_df, an_b = prep(b_df, C)
an = [c for c in an_a if c in an_b]
pa = summarise(a_df, an, sides, True)[0]
pb = summarise(b_df, an, sides, True)[0]
imgs = np.intersect1d(a_df.image.unique(), b_df.image.unique())
ia = {i: np.where(a_df.image.values == i)[0] for i in imgs}
ib = {i: np.where(b_df.image.values == i)[0] for i in imgs}
rng = np.random.default_rng(99)
diffs = []
for _ in range(B):
    pick = rng.choice(imgs, len(imgs), replace=True)
    da = a_df.iloc[np.concatenate([ia[i] for i in pick])].copy()
    db = b_df.iloc[np.concatenate([ib[i] for i in pick])].copy()
    da["image"] = np.repeat(np.arange(len(pick)), [len(ia[i]) for i in pick])
    db["image"] = np.repeat(np.arange(len(pick)), [len(ib[i]) for i in pick])
    sa = summarise(da, an, sides, False, with_null=False)[0]
    sb = summarise(db, an, sides, False, with_null=False)[0]
    diffs.append(sa["eps2"] - sb["eps2"])
lo, hi = np.percentile(diffs, [2.5, 97.5])
say(f"=== Same-image contrast, Derma-XAI aware - blind coupling eps2 on {len(an)} common "
    f"concepts: {pa['eps2'] - pb['eps2']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
    f"(aware {pa['eps2']:.3f}, blind {pb['eps2']:.3f}); null AUROC aware {pa['null_auc']:.3f}, "
    f"blind {pb['null_auc']:.3f}")
open(os.path.join(RES, "step7_log.txt"), "w").write("\n".join(LOG) + "\n")
