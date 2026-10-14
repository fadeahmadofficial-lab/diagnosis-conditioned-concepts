# Builds the EASY subsets used in the selection sensitivity analyses (Table III, B3).
#   easy_exD7: images NOT submitted as exemplars of any EASY feature in the Derm7pt concept map
#   easy_exDX: images NOT submitted as exemplars of any EASY feature in the primary Derma-XAI map (MAP_P)
#   easy_inD7: images submitted as exemplars of a Derm7pt-map feature (selected only)
# Usage: python -I make_exemplar_subsets.py <step8_recipe.py> <step9_dxai_on_easy.py> <easy_dir> <out_root>
import json, os, shutil, sys
import pandas as pd
S8, S9, EASY, OUT = sys.argv[1:5]
ns = {}
s8 = open(S8).read(); exec(s8[s8.index("EASY_MAP = {"):s8.index("def setting_easy")], ns)
s9 = open(S9).read(); exec(s9[s9.index("MAP_P = "):s9.index("def setting_s5")], ns)
meta = pd.read_csv(os.path.join(EASY, "metadata.csv"))
t7 = {t for v in ns["EASY_MAP"].values() for t in v}
tx = {t for v in ns["MAP_P"].values() for t in v}
for name, keep in (("easy_exD7", ~meta.exemplar.isin(t7)), ("easy_exDX", ~meta.exemplar.isin(tx)),
                   ("easy_inD7", meta.exemplar.isin(t7))):
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    m = meta[keep]; m.to_csv(os.path.join(d, "metadata.csv"), index=False)
    shutil.copy(os.path.join(EASY, "markups.json"), d)  # full markups; analyses keep only images in metadata.csv
    for f in ("LICENSE", "README.md"):
        if os.path.exists(os.path.join(EASY, f)): shutil.copy(os.path.join(EASY, f), d)
    print(name, len(m), "images;", int(m.diagnosis.isin(["melanoma", "nevus"]).sum()), "melanoma/naevus")
