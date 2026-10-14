# Shared loaders for the EASY external replication (Paper 2).
# Protocol: PAPER2_PREREGISTRATION.md, "External replication protocol".
import glob
import json
import os

import numpy as np
import pandas as pd

CONC = ["PN", "BWV", "VS", "STR", "PIG", "DaG", "RS"]
W = np.array([2, 2, 2, 1, 1, 1, 1])
# IDS-2016 terms -> 7-point criteria (fixed in the protocol before any
# encoder was run on EASY)
MAP = {
    "PN": ["Network : Atypical pigment network / Reticulation",
           "Network : Broadened pigment network / Reticulation"],
    "BWV": ["Structureless : Blue-whitish veil"],
    "VS": ["Vessels : Dotted", "Vessels : Linear irregular",
           "Vessels : Polymorphous"],
    "STR": ["Lines : Pseudopods", "Lines : Radial streaming"],
    "PIG": ["Structureless : Blotch irregular"],
    "DaG": ["Dots : Irregular", "Globules / Clods : Irregular"],
    "RS": ["Regression structures : Peppering / Granularity",
           "Regression structures : Scarlike depigmentation"],
}
# encoder category -> positive for the binary criterion (same as Derm7pt)
POS = {
    "PN": ("pigment_network", ["atypical"]),
    "BWV": ("blue_whitish_veil", ["present"]),
    "VS": ("vascular_structures", ["dotted", "linear irregular"]),
    "STR": ("streaks", ["irregular"]),
    "PIG": ("pigmentation", ["diffuse irregular", "localized irregular"]),
    "DaG": ("dots_and_globules", ["irregular"]),
    "RS": ("regression_structures",
           ["blue areas", "white areas", "combinations"]),
}
HERE = os.path.dirname(os.path.abspath(__file__))


def _first_dir(env, candidates):
    if os.environ.get(env):
        return os.environ[env]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise SystemExit("folder not found; set " + env)


# Folder layout of the paper-2 bundle: code/, data/easy_preds/,
# external_easy/ (repository files), results/easy/.
EASY_REPO = _first_dir("EASY_REPO", [
    os.path.join(HERE, "..", "external_easy")])
PRED_DIR = _first_dir("EASY_PREDS", [
    os.path.join(HERE, "..", "data", "easy_preds"),
    os.path.join(HERE, "easy_run")])


def _repo_file(*names):
    for n in names:
        p = os.path.join(EASY_REPO, n)
        if os.path.exists(p):
            return p
    raise SystemExit("missing in " + EASY_REPO + ": " + names[0])


def score(C):
    return (np.asarray(C) * W).sum(-1)


def load_readers():
    """Per image: reader-level binary criteria, majority, ambiguity."""
    meta = pd.read_csv(_repo_file("metadata.csv", "easy_metadata.csv"))
    meta = meta.rename(columns={"ISIC_id": "isic_id"})
    mel_dx = ["melanoma", "melanoma metastasis"]
    meta["mel"] = meta["diagnosis"].isin(mel_dx).astype(int)
    marks = json.load(open(_repo_file("markups.json",
                                      "easy_markups.json")))
    rows, reader_rows = [], []
    for iid in meta.isic_id:
        per_reader = marks[iid]
        R = []
        for rid, feats in per_reader.items():
            present = {k for k, v in feats.items() if len(v) > 0}
            c = [int(any(t in present for t in MAP[j])) for j in CONC]
            R.append(c)
            reader_rows.append([iid, rid] + c)
        R = np.array(R)
        n = len(R)
        maj = (2 * R.sum(0) >= n).astype(int)   # ties count as present
        dec = score(R) >= 3
        rows.append({"isic_id": iid, "n_readers": n,
                     "frac_refer": dec.mean(),
                     "unanimous": int(dec.all() or (~dec).all()),
                     "maj_score": int(score(maj)),
                     **{"maj_" + j: int(maj[k])
                        for k, j in enumerate(CONC)}})
    img = pd.DataFrame(rows).merge(
        meta[["isic_id", "mel", "diagnosis", "benign_malignant",
              "group", "exemplar"]], on="isic_id")
    img["ambiguous"] = 1 - img["unanimous"]
    rd = pd.DataFrame(reader_rows, columns=["isic_id", "reader"] + CONC)
    return img, rd


def load_preds():
    """Encoder runs on EASY: dict tag -> (P, C, p_mel), rows aligned
    to load_readers() order is done by the caller via isic_id."""
    out = {}
    for f in sorted(glob.glob(os.path.join(PRED_DIR,
                                           "easy_preds_*.csv"))):
        d = pd.read_csv(f)
        tag = os.path.basename(f)[len("easy_preds_"):-4]
        out[tag] = d
    return out


def concept_probs(d):
    """Binary criterion probability = sum of positive categories."""
    cols = []
    for j in CONC:
        c, pos = POS[j]
        cols.append(d[[c + "__" + k for k in pos]].sum(axis=1).values)
    return np.column_stack(cols)
