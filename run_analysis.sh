#!/usr/bin/env bash
# Re-runs every Paper 2 analysis (CPU only) from the model predictions and label tables.
# Set the folder variables below (or export them before running), then:  bash run_analysis.sh
# Each step writes its log and CSV files to $RES. Expected total time: roughly 1-2 hours on a laptop (estimate).
set -euo pipefail
A="$(cd "$(dirname "$0")/analysis" && pwd)"
: "${DX:?folder from tables/step1_build_tables.py (definitions.json, train_labels.csv, reads_test.csv, test_images.csv)}"
: "${KO_OLD:?Derma-XAI predictions from training/derma_xai/d1_step3_kaggle.ipynb (test_*_seed*.csv)}"
: "${KO_NEW:?Derma-XAI predictions from training/derma_xai/d1_easy_kaggle.ipynb (test_*, easy_*)}"
: "${P2:?Derm7pt folder: meta.csv, oof/fold_assignment.csv, oof3/, oofcnx/, easy_run/}"
: "${EASY:?EASY repository folder (metadata.csv, markups.json)}"
: "${REL:?folder with meta/meta.csv (Derm7pt), PH2/PH2_dataset.txt, SkinCon/ files}"
: "${V2:?V2 predictions from training/censoring/paper2_v2_censored.ipynb}"
: "${FIX:?fix predictions from training/censoring/paper2_fix.ipynb (Derm7pt part)}"
: "${FX:?fix predictions, Derma-XAI masked part (usually \$FIX/dxai)}"
: "${RES:?output folder}"
mkdir -p "$RES"
py() { python -I "$@"; }
py "$A/step7_coupling.py"      "$DX" "$EASY" "$REL" "$RES"
py "$A/step7b_own_dx.py"       "$DX" "$RES"
py "$A/step7c_own_dx_binary.py" "$DX" "$RES"
py "$A/step8_recipe.py"        "$DX" "$KO_OLD" "$P2" "$EASY" "$RES"
py "$A/step9_dxai_on_easy.py"  "$A/step8_recipe.py" "$DX" "$KO_OLD" "$KO_NEW" "$P2" "$EASY" "$RES"
py "$A/step9b_diff.py"         "$A/step8_recipe.py" "$A/step9_dxai_on_easy.py" "$DX" "$KO_OLD" "$KO_NEW" "$P2" "$EASY" "$RES/tmp9b" | tee "$RES/step9b_log.txt"
py "$A/step10_censoring_sim.py" "$A/step8_recipe.py" "$DX" "$KO_OLD" "$P2" "$EASY" "$RES"
py "$A/step11_v2_analysis.py"  "$A/step8_recipe.py" "$DX" "$KO_OLD" "$P2" "$EASY" "$V2" "$RES"
py "$A/step12_consistency.py"  "$A/step7_coupling.py" "$DX" "$EASY" "$REL" "$RES"
py "$A/step13_fix_derm7pt.py"  "$A/step8_recipe.py" "$DX" "$KO_OLD" "$P2" "$EASY" "$V2" "$FIX" "$RES"
py "$A/step14_fix_dxai.py"     "$A/step9_dxai_on_easy.py" "$A/step8_recipe.py" "$DX" "$KO_OLD" "$KO_NEW" "$P2" "$EASY" "$FX" "$RES/step13_boot.csv" "$RES"
py "$A/step15_b1_blind.py"     "$A/step8_recipe.py" "$DX" "$KO_NEW" "$P2" "$EASY" "$FX" "$RES"
py "$A/step16_b3_selection.py" "$A/step13_fix_derm7pt.py" "$A/step8_recipe.py" "$DX" "$KO_OLD" "$P2" "$EASY" "$V2" "$FIX" "$RES"
py "$A/step17_b4_loao.py"      "$A/step7_coupling.py" "$A/step8_recipe.py" "$DX" "$EASY" "$REL" "$KO_NEW" "$P2" "$RES"
S18=("$A/step8_recipe.py" "$A/step9_dxai_on_easy.py" "$DX" "$KO_OLD" "$KO_NEW" "$P2" "$EASY" "$RES")
py "$A/step18_within_dx.py"    "${S18[@]}"
py "$A/step18b_paired_f1f2.py" "$A/step18_within_dx.py" "${S18[@]}" | tee "$RES/step18b_log.txt"
py "$A/step19_reader_boot.py"  "$A/step18_within_dx.py" "${S18[@]}"
py "$A/step19b_f1_dtheta.py"   "$A/step19_reader_boot.py" "$A/step18_within_dx.py" "${S18[@]}" | tee "$RES/step19b_log.txt"
py "$A/step20_bounds.py"       "$A/step18_within_dx.py" "${S18[@]}"
py "$A/step21_v_reader.py"     "$A/step10_censoring_sim.py" "$A/step8_recipe.py" "$DX" "$KO_OLD" "$P2" "$EASY" "$RES"
py "$A/step22_easy_reader.py"  "$A/step18_within_dx.py" "${S18[@]}"
if [ -n "${KO_CNX:-}" ]; then  # folder with ConvNeXt test_/easy_convnext_tiny_seed{0,1,2}.csv
  py "$A/step23_cnx_seeds.py" "$A/step18_within_dx.py" "$A/step8_recipe.py" "$A/step9_dxai_on_easy.py" "$DX" "$KO_OLD" "$KO_CNX" "$P2" "$EASY" "$RES"
fi
echo "All steps finished. Logs are in $RES"
