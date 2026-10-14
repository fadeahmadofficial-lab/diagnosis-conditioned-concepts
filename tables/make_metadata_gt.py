# Builds metadata_gt.csv (one row per image; characteristic = 1 if any training annotator marked it)
# from the Derma-XAI release file metadata/metadata_ground_truth.csv.
# Checked 2026-10-14: with this file, step1_build_tables.py reproduces our train_labels.csv,
# reads_test.csv, test_images.csv and definitions.json exactly.
# Usage: python -I make_metadata_gt.py <Datasets_dir> <out_csv>
import sys
import pandas as pd
MEL = ["TRBL", "ESA", "GP", "PV", "PRL", "BDG", "WLSA", "PLR", "PES", "PIF"]
NEV = ["OPC", "SPC", "MVP", "PRLC", "PLF", "PDES", "APC", "MS"]
g = pd.read_csv(sys.argv[1] + "/metadata/metadata_ground_truth.csv")
first = g.drop_duplicates("image_id").set_index("image_id")
first[MEL + NEV] = g.groupby("image_id")[MEL + NEV].max()
first.reset_index().to_csv(sys.argv[2], index=False)
print("images:", len(first))
