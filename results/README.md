# Results behind the paper

These are the CSV and log files produced by `run_analysis.sh` from the released predictions. Each file is cited in the supplement where it is used.

Setting names in the logs differ from the paper:

| In the logs | In the paper |
| --- | --- |
| S1, "S1-retrained" (Derma-XAI EfficientNet-B0, aware reads) | F1 |
| S2 (blind reads) | F2 |
| S5, "primary map" (Derma-XAI model on EASY) | F3 |
| S3 (Derm7pt EfficientNet-B4, out of fold) | F4 |
| S4 (Derm7pt EfficientNet-B4 on EASY) | F5 |

| File | Content |
| --- | --- |
| `coupling_per_concept.csv`, `coupling_summary.csv` | Coupling ε², prevalence and diagnosis-only null AUROC for each concept and label source (Table I, Fig. 2) |
| `step12_consistency.csv` | Concept-only diagnosis AUROC per label source |
| `recipe_per_concept.csv`, `recipe_summary.csv` | Model and null AUROC per concept for F2, F4, F5 and the sensitivity runs |
| `step9_per_concept.csv`, `step9_summary.csv` | The same for F1 (retrained models) and F3 |
| `step18_per_concept.csv`, `step18_summary.csv` | Within-class AUROC per concept and class, diagnosis-adjusted AUROC, H1, H2 |
| `step20_bounds.csv` | Worst-case identification bounds per concept (unweighted reads) |
| `step23_per_seed.csv`, `step23_summary.csv` | ConvNeXt-Tiny on Derma-XAI, three seeds |
| `step10_censoring.csv` | Censoring simulation V at each π |
| `step11_v2_*.csv`, `step13_summary.csv` | Censored training V2, T1, T3 |
| `step17_loao.csv` | Leave-one-annotator-out (B4) |
| `step*_log.txt` | Full printed output of each step, including confidence intervals |
