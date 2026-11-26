# Repo TV4 – SEM (`semopy`), SHAP Explainability & Model Comparison

Deliverables of **Member 4 (TV4)** for **GROUP7_STAT3013.R11** on hospital-level HCAHPS patient experience and 30-day heart failure readmission (`READM_30_HF`).

| Task | Target / Scope | Method | Artifacts |
| :--- | :--- | :--- | :--- |
| **SEM & Diagnostics** | `READM_30_HF` (`n = 2,850` hospitals) | VIF check, standardized regression (`R² = 0.062`), and `semopy` structural modeling | `tv4_part1_sem_analysis.py`, SEM CSVs, path diagram |
| **SHAP & Benchmark** | `READM_HF_High` (`570` test rows, threshold `0.5`) | `rf.pkl` Gini & `SHAP` (`#Test 470` local case) + comparison across tuned models | SHAP plots/CSVs, report sections |

## Repository structure

```text
Repo TV4/
├── README.md
├── tv4_part1_sem_analysis.py
├── sem_correlation_matrix.csv
├── sem_descriptive_stats.csv
├── sem_standardized_paths_8features.csv
├── sem_path_diagram_2850.png
├── shap_bar_test570.png
├── shap_beeswarm_test570.png
├── shap_waterfall_test470.png
└── shap_local_test470.csv
