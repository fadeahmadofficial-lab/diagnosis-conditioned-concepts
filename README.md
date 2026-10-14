# Diagnosis-conditioned concept labels in dermoscopy

[![DOI](https://zenodo.org/badge/1418582685.svg)](https://doi.org/10.5281/zenodo.23371805)

Code for the paper *"Diagnosis-Conditioned Concept Labels in Dermoscopy: Measuring the Coupling and Auditing Concept Evaluation"* (Fade Hafez Ahmad and Sathyapriya Loganathan, SRM Institute of Science and Technology). [Add the journal and DOI after acceptance.]

The code measures how strongly concept labels depend on the diagnosis. It scores concept models against a diagnosis-only null and tests the effect of censoring opposite-side features. It also contains the training code for every model used in the paper.

**No data are included.** Every dataset must be downloaded from its original provider under its own licence (see [Data](#data)). Nothing in this repository contains images, labels or reader information from those datasets.

## Folder layout

| Folder | What it contains |
| --- | --- |
| `tables/` | Builds the Derma-XAI label tables from the public release. It also builds the EASY subsets used in the selection analyses. |
| `training/derma_xai/` | Kaggle notebooks for the Derma-XAI concept models (EfficientNet-B0, ConvNeXt-Tiny). They predict on the test images and on EASY. |
| `training/derm7pt/` | Derm7pt training code: shared encoder, one softmax head per criterion, plus a melanoma head. |
| `training/censoring/` | V2 (training on censored labels) and the censoring-aware losses (T1–T3). |
| `analysis/` | `step7` to `step22`: every number in the paper. Each script states its pre-registered rule in its header. |
| `figures/` | `make_figs.py`: Figures 1–4. |
| `run_analysis.sh` | Runs all analysis steps in order. |

## Data

| Dataset | Where to get it | Licence (as we found it) |
| --- | --- | --- |
| Derma-XAI (Chanda et al., *Nat. Commun.* 15:524, 2024) | Figshare item "HAM10000 expert-annotated explanations and pseudonymized reader study data", linked from the paper. The HAM10000 images are on Kaggle as `kmader/skin-cancer-mnist-ham10000`. | CC BY 4.0 |
| EASY (Liopyris et al., *J. Invest. Dermatol.* 2024) | The authors' repository (`metadata.csv`, `markups.json`). Images are ISIC collection 166. | Apache 2.0 |
| Derm7pt (Kawahara et al., *IEEE JBHI* 2019) | Derm7pt website (registration) | CC BY-NC-ND 4.0 |
| PH2 (Mendonça et al., EMBC 2013) | PH2 website | Research use, no redistribution |
| SkinCon (Daneshjou et al., NeurIPS 2022) with Fitzpatrick 17k and DDI | SkinCon website. The images come from the Fitzpatrick 17k and DDI providers. | SkinCon: none stated. Fitzpatrick 17k: CC BY-NC-SA 3.0. DDI: no distribution. |

Our scripts expect these folders (you choose the names and set them in `run_analysis.sh`):

- `REL/meta/meta.csv` (Derm7pt metadata), `REL/PH2/PH2_dataset.txt`, `REL/SkinCon/annotations_fitzpatrick17k.csv`, `REL/SkinCon/fitzpatrick17k.csv`, `REL/SkinCon/annotations_ddi.csv`, `REL/SkinCon/ddi_metadata.csv`
- `EASY/metadata.csv`, `EASY/markups.json`
- `P2/meta.csv` (Derm7pt metadata, the same file as `REL/meta/meta.csv`) and `P2/oof/fold_assignment.csv` (five folds, split by case), plus the prediction folders written by the training code (see step 2).

## How to reproduce

Use Python 3.11 and run `pip install -r requirements.txt`. Training needs a GPU; we used Kaggle T4 notebooks. Analysis runs on a CPU.

### 1. Build the Derma-XAI tables

```bash
python -I tables/make_metadata_gt.py  <Derma-XAI Datasets folder>  metadata_gt.csv
python -I tables/step1_build_tables.py <Derma-XAI Datasets folder>  metadata_gt.csv  dx_out
```

This writes `dx_out/definitions.json`, `train_labels.csv` (3,254 images), `test_images.csv` (196) and `reads_test.csv` (424 aware and 1,560 blind reads). With these inputs we reproduced our own tables exactly. The MD5 prefixes are 5c4002fbf598 for `train_labels.csv` and d91f29aee05c for `test_images.csv`.

### 2. Train the models (Kaggle, GPU)

1. Upload `dx_out/train_labels.csv` and `dx_out/test_images.csv` as a **private** Kaggle dataset. Add it and `kmader/skin-cancer-mnist-ham10000` as notebook inputs, and turn Internet on.
2. Run these notebooks:
   - `training/derma_xai/d1_step3_kaggle.ipynb` writes the test predictions (`KO_OLD`).
   - `training/derma_xai/d1_easy_kaggle.ipynb` retrains the models and predicts on EASY (`KO_NEW`).
   - `d1_cnx_seeds_kaggle.ipynb` adds ConvNeXt seeds 1–2.
3. For Derm7pt, run the five-fold out-of-fold training with the scripts in `training/derm7pt/`. This writes `P2/oof3` (EfficientNet-B4) and `P2/oofcnx` (ConvNeXt-Tiny). Then run the full-data training that predicts on EASY (`P2/easy_run`).
   - The base training recipe is the one from our earlier paper's code, which `kaggle_dump_concepts.py` calls. [Add a link to that repository.]
4. Run `training/censoring/paper2_v2_censored.ipynb` (writes `V2`) and `training/censoring/paper2_fix.ipynb` (writes `FIX`, plus `FIX/dxai` for the masked Derma-XAI loss).

### 3. Run the analyses

```bash
export DX=dx_out KO_OLD=... KO_NEW=... P2=... EASY=... REL=... V2=... FIX=... FX=$FIX/dxai RES=results
bash run_analysis.sh
```

| Paper item | Script and output file |
| --- | --- |
| Table I, Fig. 2 (coupling) | `step7_coupling.py` → `step7_log.txt`, `coupling_summary.csv`. Coupling to the reader's own diagnosis: `step7c_own_dx_binary.py`. Concept-only diagnosis: `step12_consistency.py`. |
| Fig. 3 (F1–F5) | `step8_recipe.py` (F2, F4, F5). `step9_dxai_on_easy.py` (F1, F3). `step9b_diff.py` (shared concepts). |
| Fig. 4a (V) | `step10_censoring_sim.py` → `step10_censoring.csv` |
| Fig. 4b (V2), T3 | `step11_v2_analysis.py`, `step13_fix_derm7pt.py` |
| T1, T2, Holm | `step13_fix_derm7pt.py`, `step14_fix_dxai.py` |
| B1, B3, B4 | `step15_b1_blind.py`, `step16_b3_selection.py`, `step17_b4_loao.py` |
| Diagnosis-adjusted AUROC, H1, H2 | `step18_within_dx.py`, `step18b_paired_f1f2.py` |
| Reader-resampled intervals | `step19_reader_boot.py`, `step19b_f1_dtheta.py`, `step21_v_reader.py`, `step22_easy_reader.py` |
| ConvNeXt with three seeds | `step23_cnx_seeds.py` (set `KO_CNX` to the folder with the three ConvNeXt seeds) |
| Identification bounds (Supp. S4d) | `step20_bounds.py`. It includes an exhaustive unit test of the sorting result. |
| Table III subsets | `tables/make_exemplar_subsets.py`, then steps 7–13 with `EASY` set to each subset folder |
| Drop-one concept (irregular streaks) | Delete the `"STR_irr": [...]` line from `EASY_MAP` in `step7_coupling.py` and `step8_recipe.py`, then re-run steps 7–13. |
| Figures | `python figures/make_figs.py figures results` |

All bootstraps use fixed seeds. On 2026-10-14 we re-ran `run_analysis.sh` from the saved predictions into an empty folder, and every log and CSV file was byte-identical to the files behind the paper. Point estimates also depend on the model predictions; retraining on another GPU can change them slightly.

## Released data

We release the predictions of all our models and per-read concept tables derived from EASY (`paper2_data_release`, [Zenodo DOI to add]). The predictions of the Derma-XAI-trained models are derived from data licensed CC BY 4.0 by Chanda et al. (*Nat. Commun.* 15:524, 2024).

## Privacy

The analyses use the reader identifiers published in the datasets only to weight reads and to resample readers. We do not build, use or release any reader background information (experience, language, profession).

## Citation

See `CITATION.cff`. Please also cite the dataset papers listed above.

## Licence

Code: MIT (see `LICENSE`). [Authors: confirm the licence before publishing.] The licence covers our code only, not the datasets.
