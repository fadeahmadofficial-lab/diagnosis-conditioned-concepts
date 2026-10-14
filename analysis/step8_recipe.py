# Paper 2, component F: the evaluation recipe applied to trained concept models.
# Written 2026-10-09 BEFORE any number below was computed. Descriptive demonstration.
#
# Usage: python -I step8_recipe.py <dermaxai_step1_out> <dermaxai_kaggle_out> <p2_dir> <easy_dir> <result_dir>
#
# PRE-SPECIFIED
# Settings (model -> reference labels):
#   S1 Derma-XAI EffB0 (5-seed mean of predictions) -> aware test reads
#   S2 Derma-XAI EffB0 -> blind test reads (censored by each reader's own diagnosis)
#   S3 Derm7pt EffB4 (5-seed mean, out-of-fold, epoch 40) -> Derm7pt atlas labels
#   S4 Derm7pt-trained EffB4 (3-seed mean) on EASY -> 5 blinded expert readers (external)
#   Sensitivity: ConvNeXt-T in each setting.
# Images: primary melanoma vs naevus only (as step 7). Reads weighted 1/(reads per image).
# Concepts: both sides; analysable = >= 20 weighted positives and negatives in the
#   reference labels (as step 7). Subset metrics need >= 10 weighted pos and neg.
# R1 concept AUROC: model concept probability vs reference label.
# R2 diagnosis-only null: score = p*E[c|mel] + (1-p)*E[c|naevus], p = the model's own
#   black-box melanoma probability, class means from the model's TRAINING labels (out-of-
#   fold for Derm7pt). It is a "concept predictor" that only knows the diagnosis.
# R3 excess = R1 - R2 (primary recipe number). 95% CI: image bootstrap, 2000 reps.
# R4 within-class: R1 and R2 within naevi and within melanomas.
# R5 correctness-conditioned: R1 and R2 on images the black box misdiagnoses (p >= 0.5
#   vs truth). A predictor that follows its own diagnosis falls below 0.5 here.
# Summary = mean over concepts. Interpretation fixed in advance: excess near 0 means the
#   concept head adds nothing measurable beyond the model's diagnosis on that reference.
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

DX, KO, P2, EASY, RES = sys.argv[1:6]
os.makedirs(RES, exist_ok=True)
B = 2000
LOG = []


def say(m):
    print(m, flush=True)
    LOG.append(str(m))


def wauc(s, c, w):
    """Weighted AUROC with ties counted half."""
    pos, neg = c == 1, c == 0
    wp, wn = w[pos].sum(), w[neg].sum()
    if wp <= 0 or wn <= 0:
        return np.nan
    u, inv = np.unique(s, return_inverse=True)
    np_ = np.bincount(inv, weights=w * pos, minlength=len(u))
    nn_ = np.bincount(inv, weights=w * neg, minlength=len(u))
    below = np.cumsum(nn_) - nn_
    return (np_ * (below + 0.5 * nn_)).sum() / (wp * wn)


# ---------------------------------------------------------------- Derm7pt pieces
D7 = {"PN_atyp": ("pigment_network", ["atypical"]), "STR_irr": ("streaks", ["irregular"]),
      "PIG_irr": ("pigmentation", ["diffuse irregular", "localized irregular"]),
      "DaG_irr": ("dots_and_globules", ["irregular"]),
      "RS": ("regression_structures", ["blue areas", "white areas", "combinations"]),
      "BWV": ("blue_whitish_veil", ["present"]),
      "VS_atyp": ("vascular_structures", ["dotted", "linear irregular"]),
      "PN_typ": ("pigment_network", ["typical"]), "STR_reg": ("streaks", ["regular"]),
      "PIG_reg": ("pigmentation", ["diffuse regular", "localized regular"]),
      "DaG_reg": ("dots_and_globules", ["regular"])}
meta = pd.read_csv(os.path.join(P2, "meta.csv"))
dx = meta.diagnosis.str.lower()
meta["mel"] = (dx.str.startswith("melanoma") & ~dx.str.contains("metastasis")).astype(int)
meta["keep"] = meta.mel.astype(bool) | dx.str.contains("nevus")
for k, (col, vals) in D7.items():
    meta[k] = meta[col].isin(vals).astype(int)
meta["row"] = np.arange(len(meta))
fold = pd.read_csv(os.path.join(P2, "oof", "fold_assignment.csv")).set_index("row").fold


def d7_probs(d):
    out = pd.DataFrame({k: d[[col + "__" + v for v in vals]].sum(axis=1).values
                        for k, (col, vals) in D7.items()})
    out["p"] = d.p_mel_bb.values
    return out


# ---------------------------------------------------------------- settings
def setting_dermaxai(arm, backbone):
    D = json.load(open(os.path.join(DX, "definitions.json")))
    C = D["MEL"] + D["NEV"]
    r = pd.read_csv(os.path.join(DX, "reads_test.csv"))
    r = r[r.arm == arm]
    files = sorted(glob.glob(os.path.join(KO, f"test_{backbone}_seed*.csv")))
    P = sum(pd.read_csv(f).set_index("image_id")[C + ["p_mel_blackbox"]] for f in files) / len(files)
    P = P.rename(columns={"p_mel_blackbox": "p"})
    tr = pd.read_csv(os.path.join(DX, "train_labels.csv"))
    tr = tr[tr.split == "train"]
    means = {c: (tr[tr.benign_malignant == 0][c].mean(), tr[tr.benign_malignant == 1][c].mean()) for c in C}
    lab = r[["image_id", "y"] + C].rename(columns={"image_id": "image"})
    Pr = P.loc[lab.image.values].reset_index(drop=True)
    return lab.reset_index(drop=True), Pr, {k: means for k in [None]}, C, len(files)


def setting_derm7pt(backbone):
    if backbone == "effb4":
        files = [f for f in glob.glob(os.path.join(P2, "oof3", "oof_fold*_frac1.0_seed*.csv"))]
    else:
        files = glob.glob(os.path.join(P2, "oofcnx", "oof_convnext_tiny_fold*_frac1.0_seed*.csv"))
    d = pd.concat(pd.read_csv(f) for f in files)
    d = d[d.epoch == d.epoch.max()]
    nseed = d.seed.nunique()
    num = d.drop(columns=[c for c in ("backbone",) if c in d]).groupby("row").mean(numeric_only=True)
    pr = d7_probs(num.reset_index())
    pr["row"] = num.index.values
    m = meta[meta.keep].merge(pr, on="row", suffixes=("", "_hat"))
    lab = pd.DataFrame({"image": m.row.values, "y": m.mel.values, **{k: m[k].values for k in D7}})
    P = pd.DataFrame({**{k: m[k + "_hat"].values for k in D7}, "p": m.p.values})
    # training-fold class means, per held-out fold
    mm = meta[meta.keep]
    means = {}
    for f in range(5):
        trn = mm[mm.row.map(fold) != f]
        means[f] = {k: (trn[trn.mel == 0][k].mean(), trn[trn.mel == 1][k].mean()) for k in D7}
    lab["fold"] = lab.image.map(fold).values
    return lab, P, means, list(D7), nseed


EASY_MAP = {"PN_atyp": ["Network : Atypical pigment network / Reticulation",
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
            "DaG_reg": ["Dots : Regular", "Globules / Clods : Regular"]}


def setting_easy(backbone):
    em = pd.read_csv(os.path.join(EASY, "metadata.csv"))
    em = em[em.diagnosis.isin(["melanoma", "nevus"])]
    y = dict(zip(em.ISIC_id, (em.diagnosis == "melanoma").astype(int)))
    mk = json.load(open(os.path.join(EASY, "markups.json")))
    rows = []
    for iid, rd in mk.items():
        if iid not in y:
            continue
        for rid, feats in rd.items():
            present = {k for k, v in feats.items() if len(v)}
            rows.append({"image": iid, "y": y[iid],
                         **{c: int(any(t in present for t in ts)) for c, ts in EASY_MAP.items()}})
    lab = pd.DataFrame(rows)
    files = glob.glob(os.path.join(P2, "easy_run", f"easy_preds_{backbone}_seed*.csv"))
    d = pd.concat(pd.read_csv(f) for f in files).groupby("isic_id").mean(numeric_only=True)
    pr = d7_probs(d.reset_index())
    pr.index = d.index
    P = pr.loc[lab.image.values].reset_index(drop=True)[list(EASY_MAP) + ["p"]]
    mm = meta[meta.keep]
    means = {None: {k: (mm[mm.mel == 0][k].mean(), mm[mm.mel == 1][k].mean()) for k in EASY_MAP}}
    return lab, P, means, list(EASY_MAP), len(files)


# ---------------------------------------------------------------- recipe
def recipe(lab, P, means, C, label):
    lab = lab.copy()
    w = 1.0 / lab.groupby("image")["image"].transform("size").values
    y = lab.y.values
    p = P.p.values
    an = [c for c in C if (lab[c].values * w).sum() >= 20 and ((1 - lab[c].values) * w).sum() >= 20]
    fkey = lab["fold"].values if "fold" in lab else np.full(len(lab), None)
    null = {}
    for c in an:
        m0 = np.array([means[k][c][0] for k in fkey])
        m1 = np.array([means[k][c][1] for k in fkey])
        null[c] = p * m1 + (1 - p) * m0
    wrong = (p >= 0.5).astype(int) != y
    subsets = {"all": np.ones(len(y), bool), "within_nev": y == 0, "within_mel": y == 1,
               "misdiagnosed": wrong}
    img_codes, img_idx = np.unique(lab.image.values, return_inverse=True)

    def metrics(wt):
        out = {}
        for sn, sel in subsets.items():
            r1, r2 = [], []
            for c in an:
                cc, ww = lab[c].values[sel], wt[sel]
                if sn != "all" and ((cc * ww).sum() < 10 or ((1 - cc) * ww).sum() < 10):
                    continue
                r1.append(wauc(P[c].values[sel], cc, ww))
                r2.append(wauc(null[c][sel], cc, ww))
            out[sn + "_model"] = np.nanmean(r1) if r1 else np.nan
            out[sn + "_null"] = np.nanmean(r2) if r2 else np.nan
            out[sn + "_excess"] = out[sn + "_model"] - out[sn + "_null"]
            out[sn + "_k"] = len(r1)
        return out

    pt = metrics(w)
    rng = np.random.default_rng(2026)
    bs = []
    n_img = len(img_codes)
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
        bs.append(metrics(w * cnt[img_idx]))
    bs = pd.DataFrame(bs)
    img_wrong = pd.Series(wrong).groupby(img_idx).first()
    say(f"=== {label}: images {n_img}, reads {len(lab)}, analysable concepts {len(an)}; "
        f"black box misdiagnoses {int(img_wrong.sum())} images "
        f"(threshold 0.5; AUROC {wauc(pd.Series(p).groupby(img_idx).first().values, pd.Series(y).groupby(img_idx).first().values, np.ones(n_img)):.3f})")
    rows = []
    for sn in subsets:
        lo, hi = np.nanpercentile(bs[sn + "_excess"], [2.5, 97.5])
        say(f"    {sn:13s} model {pt[sn + '_model']:.3f}  null {pt[sn + '_null']:.3f}  "
            f"excess {pt[sn + '_excess']:+.3f} [{lo:+.3f}, {hi:+.3f}]  (concepts {pt[sn + '_k']})")
        rows.append(dict(setting=label, subset=sn, model=pt[sn + "_model"], null=pt[sn + "_null"],
                         excess=pt[sn + "_excess"], lo=lo, hi=hi, concepts=pt[sn + "_k"],
                         images=n_img, misdiagnosed=int(img_wrong.sum())))
    per = [dict(setting=label, concept=c, model=wauc(P[c].values, lab[c].values, w),
                null=wauc(null[c], lab[c].values, w)) for c in an]
    return rows, per


ALL, PER = [], []
for lab_fn, label in [
        (lambda: setting_dermaxai("aware", "effb0"), "S1 Derma-XAI EffB0 -> aware reads"),
        (lambda: setting_dermaxai("blind", "effb0"), "S2 Derma-XAI EffB0 -> blind reads"),
        (lambda: setting_derm7pt("effb4"), "S3 Derm7pt EffB4 OOF -> atlas labels"),
        (lambda: setting_easy("effb4"), "S4 Derm7pt EffB4 on EASY -> blinded experts"),
        (lambda: setting_dermaxai("aware", "convnext_tiny"), "sens S1 ConvNeXt -> aware reads"),
        (lambda: setting_dermaxai("blind", "convnext_tiny"), "sens S2 ConvNeXt -> blind reads"),
        (lambda: setting_derm7pt("convnext"), "sens S3 ConvNeXt OOF -> atlas labels"),
        (lambda: setting_easy("convnext_tiny"), "sens S4 ConvNeXt on EASY -> blinded experts")]:
    lab, P, means, C, ns = lab_fn()
    say(f"[{label}: prediction mean over {ns} files]")
    r, pc = recipe(lab, P, means, C, label)
    ALL += r
    PER += pc
pd.DataFrame(ALL).to_csv(os.path.join(RES, "recipe_summary.csv"), index=False)
pd.DataFrame(PER).to_csv(os.path.join(RES, "recipe_per_concept.csv"), index=False)
open(os.path.join(RES, "step8_log.txt"), "w").write("\n".join(LOG) + "\n")
