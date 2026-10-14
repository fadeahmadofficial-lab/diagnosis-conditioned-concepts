# D1 protocol, Step 1 (signed protocol "D1 Protocol: Review Bias in Concept
# Labels", Sections 2-3). Builds the analysis tables. No hypothesis is tested.
# Public version: builds only the columns Paper 2 uses (no reader-background columns).
#
# Usage:  python -I step1_build_tables.py <Datasets_dir> <metadata_gt.csv>
#                <out_dir>
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

DS, GT_CSV, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)

# Section 3: directions from the published ontology (Chanda et al. 2024, Table 1)
MEL = ["TRBL", "ESA", "GP", "PV", "PRL", "BDG", "WLSA", "PLR", "PES", "PIF"]
NEV = ["OPC", "SPC", "MVP", "PRLC", "PLF", "PDES", "APC", "MS"]
ALL = MEL + NEV
DIR = {**{c: 1 for c in MEL}, **{c: -1 for c in NEV}}

log = []


def say(msg):
    print(msg)
    log.append(msg)


def md5_12(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


say("fingerprint metadata_gt.csv: %s (protocol: 1d64f98a6cb7)" % md5_12(GT_CSV))

study = pd.read_csv(os.path.join(DS, "metadata", "metadata_study.csv"))
gtm = pd.read_csv(GT_CSV)


def read_dir(root, prefix, arm, valid_images):
    rows, bad = [], 0
    for d in sorted(glob.glob(os.path.join(root, prefix + "*"))):
        reader = os.path.basename(d)
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            iid = os.path.basename(f)[:-5]
            if valid_images is not None and iid not in valid_images:
                continue
            try:
                j = json.load(open(f))
                j = json.loads(j) if isinstance(j, str) else j
            except Exception:
                bad += 1
                continue
            unknown = set(j) - set(ALL) - {"None"}
            if unknown:
                say("  note: unknown keys %s in %s" % (sorted(unknown), f))
            rows.append({"arm": arm, "reader": reader, "image_id": iid,
                         **{c: int(c in j) for c in ALL}})
    say("%s reads parsed: %d (unparseable files: %d)" % (arm, len(rows), bad))
    return pd.DataFrame(rows)


# --- test set (Section 2): the 196 images with both protocols, one per lesion
test_imgs = set(study.image_id)
aware_all = read_dir(os.path.join(DS, "ground_truth_annotations"), "annotator",
                     "aware", None)
blind = read_dir(os.path.join(DS, "study_annotations"), "participant", "blind",
                 test_imgs)
both = test_imgs & set(aware_all.image_id) & set(blind.image_id)
say("test images with both protocols: %d" % len(both))
info = gtm.set_index("image_id")[["lesion_id", "benign_malignant"]]
test = sorted(both)
test_les = set(info.loc[test, "lesion_id"])
say("test lesions: %d" % len(test_les))

# --- read-level table on the test images
aw = aware_all[aware_all.image_id.isin(both)].copy()
bl = blind[blind.image_id.isin(both)].copy()
reads = pd.concat([aw, bl], ignore_index=True)
reads["y"] = reads.image_id.map(info.benign_malignant).astype(int)
reads["lesion_id"] = reads.image_id.map(info.lesion_id)
reads["s"] = np.where(reads.arm == "aware", 2 * reads.y - 1, 0)

# own diagnosis (blind arm)
study["reader"] = "participant" + study.participant.astype(int).astype(str)
own = study.set_index(["reader", "image_id"]).prediction
reads["own_dx"] = [own.get((r, i), np.nan) if a == "blind" else np.nan
                   for a, r, i in zip(reads.arm, reads.reader, reads.image_id)]
say("blind reads with own diagnosis: %d of %d"
    % (reads[(reads.arm == "blind")].own_dx.notna().sum(),
       (reads.arm == "blind").sum()))

# analysable characteristics (Section 3): >= 10 marks in each arm on test images
cnt = reads.groupby("arm")[ALL].sum()
analysable = [c for c in ALL if cnt.loc["aware", c] >= 10 and cnt.loc["blind", c] >= 10]
say("marks per arm on test images:\n" + cnt.to_string())
say("analysable characteristics (%d): %s" % (len(analysable), analysable))

# --- training labels (Section 5): released union labels, lesion-disjoint
train = gtm[~gtm.lesion_id.isin(test_les)].copy()
say("training images: %d (lesions %d); dropped for lesion overlap: %d"
    % (len(train), train.lesion_id.nunique(),
       len(gtm) - len(train) - len(both)))
rng = np.random.default_rng(2026)
les = np.array(sorted(train.lesion_id.unique()))
val_les = set(rng.choice(les, int(round(0.15 * len(les))), replace=False))
train["split"] = np.where(train.lesion_id.isin(val_les), "val", "train")
say("train / val images: %s" % train.split.value_counts().to_dict())

# check: does the released union equal the union of the aware JSON reads?
u = aware_all.groupby("image_id")[ALL].max()
g2 = gtm.set_index("image_id")[ALL]
common = u.index.intersection(g2.index)
mism = (u.loc[common] != g2.loc[common]).any(axis=1).mean()
say("released labels vs union of aware JSON reads: %.1f%% of images differ"
    % (100 * mism))

reads.to_csv(os.path.join(OUT, "reads_test.csv"), index=False)
train[["image_id", "lesion_id", "benign_malignant", "split"] + ALL].to_csv(
    os.path.join(OUT, "train_labels.csv"), index=False)
pd.DataFrame({"image_id": test, "lesion_id": [info.lesion_id[i] for i in test],
              "benign_malignant": [int(info.benign_malignant[i]) for i in test]}
             ).to_csv(os.path.join(OUT, "test_images.csv"), index=False)
json.dump({"MEL": MEL, "NEV": NEV, "analysable": analysable},
          open(os.path.join(OUT, "definitions.json"), "w"), indent=1)
open(os.path.join(OUT, "step1_log.txt"), "w").write("\n".join(log) + "\n")
