# Paper 2, F decisive test: Derma-XAI-trained concept models on EASY (blinded experts).
# Written 2026-10-09 AFTER unzipping d1_easy.zip but BEFORE reading any prediction value.
#
# Usage: python -I step9_dxai_on_easy.py <step8_recipe.py> <dermaxai_step1_out> <old_kaggle_out>
#        <new_kaggle_out> <p2_dir> <easy_dir> <result_dir>
#
# PRE-SPECIFIED
# (A) Consistency: retrained vs original Derma-XAI models on the 196 test images.
#     Per concept and for p_mel: Pearson r of the 5-seed mean predictions.
#     Then S1/S2 of step 8 are re-run with the retrained models.
# (B) S5: retrained Derma-XAI EffB0 (5-seed mean) -> EASY 5 blinded readers; sens: ConvNeXt.
#     Same recipe as step 8: R1 model AUROC, R2 diagnosis-only null (class means from the
#     Derma-XAI TRAINING labels, p = the Derma-XAI black box on EASY), R3 excess, R4 within
#     class, R5 misdiagnosed images. Primary melanoma vs naevus images only.
# Concept mapping Derma-XAI -> EASY features (fixed before results):
#   PRIMARY (clean): TRBL <- broadened network, branched streaks; GP <- peppering/granularity;
#     PV <- polymorphous vessels; PRL <- pseudopods, radial streaming (EASY does not code
#     circumference); WLSA <- shiny white streaks, scarlike depigmentation;
#     PRLC <- starburst pattern.
#   SECONDARY (approximate, added to primary): ESA <- irregular blotch; BDG <- irregular
#     dots, irregular globules (colour not coded in EASY); OPC <- homogeneous pattern;
#     MVP <- comma vessels.
#   Not mappable: SPC, APC, MS, PLR, PES, PIF, PLF, PDES.
# Interpretation fixed in advance:
#   S5 excess clearly > 0 (CI above 0) -> the Derma-XAI model carries concept information
#     beyond its diagnosis that its own diagnosis-conditioned test labels could not show;
#     the near-zero S1 excess is then a property of the reference labels.
#   S5 excess ~ 0 -> the model itself learned little beyond diagnosis (consistent with
#     training labels that are near-relabels of diagnosis).
import os
import sys

import numpy as np
import pandas as pd

S8, DX, KO_OLD, KO_NEW, P2, EASY, RES = sys.argv[1:8]
src = open(S8).read().split("ALL, PER = [], []")[0]
sys.argv = ["step8", DX, KO_NEW, P2, EASY, RES]
exec(src)   # defines wauc, recipe, setting_* , say, LOG, B ...
LOG.clear()
import json  # noqa: E402

D = json.load(open(os.path.join(DX, "definitions.json")))
C18 = D["MEL"] + D["NEV"]

# ---------------------------------------------------------------- (A) consistency
say("=== (A) retrained vs original Derma-XAI models, 196 test images, 5-seed EffB0 mean")
def mean_preds(ko, tag):
    fs = sorted(glob.glob(os.path.join(ko, f"test_{tag}_seed*.csv")))
    return sum(pd.read_csv(f).set_index("image_id")[C18 + ["p_mel_blackbox"]] for f in fs) / len(fs)
for tag in ("effb0", "convnext_tiny"):
    a, b = mean_preds(KO_OLD, tag), mean_preds(KO_NEW, tag)
    b = b.loc[a.index]
    r = {c: np.corrcoef(a[c], b[c])[0, 1] for c in C18 + ["p_mel_blackbox"]}
    say(f"  {tag}: r(p_mel) {r['p_mel_blackbox']:.3f}; concept r median "
        f"{np.median([r[c] for c in C18]):.3f}, min {min(r[c] for c in C18):.3f} ({min(C18, key=r.get)})")

ALL9, PER9 = [], []
for lab_fn, label in [
        (lambda: setting_dermaxai("aware", "effb0"), "S1-retrained Derma-XAI EffB0 -> aware reads"),
        (lambda: setting_dermaxai("blind", "effb0"), "S2-retrained Derma-XAI EffB0 -> blind reads")]:
    lab, P, means, C, ns = lab_fn()
    r_, pc = recipe(lab, P, means, C, label)
    ALL9 += r_; PER9 += pc

# ---------------------------------------------------------------- (B) S5
MAP_P = {"TRBL": ["Network : Broadened pigment network / Reticulation", "Lines : Branched streaks"],
         "GP": ["Regression structures : Peppering / Granularity"],
         "PV": ["Vessels : Polymorphous"],
         "PRL": ["Lines : Pseudopods", "Lines : Radial streaming"],
         "WLSA": ["Shiny white structures : Shiny white streaks",
                  "Regression structures : Scarlike depigmentation"],
         "PRLC": ["Pattern : Starburst"]}
MAP_S = dict(MAP_P, **{"ESA": ["Structureless : Blotch irregular"],
                       "BDG": ["Dots : Irregular", "Globules / Clods : Irregular"],
                       "OPC": ["Pattern : Homogeneous : NOS"],
                       "MVP": ["Vessels : Comma"]})


def setting_s5(mapping, backbone):
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
                         **{c: int(any(t in present for t in ts)) for c, ts in mapping.items()}})
    lab = pd.DataFrame(rows)
    fs = sorted(glob.glob(os.path.join(KO_NEW, f"easy_{backbone}_seed*.csv")))
    P = sum(pd.read_csv(f).set_index("image_id")[C18 + ["p_mel_blackbox"]] for f in fs) / len(fs)
    P = P.rename(columns={"p_mel_blackbox": "p"}).loc[lab.image.values].reset_index(drop=True)
    tr = pd.read_csv(os.path.join(DX, "train_labels.csv"))
    tr = tr[tr.split == "train"]
    means = {None: {c: (tr[tr.benign_malignant == 0][c].mean(), tr[tr.benign_malignant == 1][c].mean())
                    for c in mapping}}
    return lab, P[list(mapping) + ["p"]], means, list(mapping), len(fs)


for mapping, mname in ((MAP_P, "primary map"), (MAP_S, "primary+approx map")):
    for bb in ("effb0", "convnext_tiny"):
        label = f"S5 Derma-XAI {bb} on EASY -> blinded experts ({mname})"
        lab, P, means, C, ns = setting_s5(mapping, bb)
        say(f"[{label}: prediction mean over {ns} files]")
        r_, pc = recipe(lab, P, means, C, label)
        ALL9 += r_; PER9 += pc
pd.DataFrame(ALL9).to_csv(os.path.join(RES, "step9_summary.csv"), index=False)
pd.DataFrame(PER9).to_csv(os.path.join(RES, "step9_per_concept.csv"), index=False)
open(os.path.join(RES, "step9_log.txt"), "w").write("\n".join(LOG) + "\n")
