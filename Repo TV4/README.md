# Repo TV4 – Structural Analysis, SHAP Interpretability & Model Comparison

Assigned deliverables of **Member 4 (TV4)** for **GROUP7_STAT3013.R11** on hospital-level HCAHPS patient experience scores and 30-day heart failure readmission (`READM_30_HF`).

| Task | Target / Scope | Method | Script / Artifacts |
| :--- | :--- | :--- | :--- |
| **Multicollinearity & Path Analysis** | `READM_30_HF` (`n = 2,850` hospitals) | VIF diagnostics, standardized 8-feature regression (`R² = 0.062`), and `semopy` re-estimation | `outputs_tv4_part1_sem_tv4_part1_sem_analysis_3.py`, `outputs_tv4_part1_sem_sem_*_3.csv` |
| **SHAP & Benchmark** | `READM_HF_High` (`570` test rows, threshold `0.5`) | `rf.pkl` Gini & SHAP (`#Test 470` case) + comparison across tuned classifiers | `shap_explanation_part2_2.py`, SHAP figures, `Member 4.docx_3.docx` |

## Repository structure

```text
GROUP7_STAT3013.R11/
├── Member 4.docx_3.docx
└── Repo TV4/
    ├── README.md
    ├── outputs_tv4_part1_sem_tv4_part1_sem_analysis_3.py
    ├── shap_explanation_part2_2.py
    ├── outputs_tv4_part1_sem_sem_correlation_matrix_3.csv
    ├── outputs_tv4_part1_sem_sem_descriptive_stats_3.csv
    ├── outputs_tv4_part1_sem_sem_standardized_paths_8features_3.csv
    ├── SHAP_2.png
    ├── shap_bar_vs_gini_rf_500trees.jpg
    ├── shap_beeswarm_rf_test570_v1.png
    ├── shap_beeswarm_rf_test570_v2.png
    └── shap_local_test470.png
## Data & Conventions
- **Dataset:** `2,850` hospitals (`STAT3013_final_model_data.csv` / `data/processed.csv`, no missing values), with a shared held-out test set of `570` rows (`Split == "test"`, `random_state = 42`).
- **Label vs. Threshold:** `21.3` (training-set median of `READM_30_HF`) is used strictly to define `READM_HF_High = 1` when `READM_30_HF > 21.3` (else `0`); classification models are evaluated at probability threshold **`0.5`** (never use `0.213` as a probability threshold).
- **Features:** Eight HCAHPS linear score variables (`Nurse_Communication`, `Doctor_Communication`, `Medication_Communication`, `Discharge_Information`, `Cleanliness`, `Quietness`, `Overall_Hospital_Rating`, `Recommend_Hospital`).

## Setup
```bash
pip install pandas numpy scikit-learn statsmodels semopy shap matplotlib joblib
```
## Usage
Run scripts from the directory containing `STAT3013_final_model_data.csv` and `rf.pkl`:
```bash
python outputs_tv4_part1_sem_tv4_part1_sem_analysis_3.py
python shap_explanation_part2_2.py
```

## Results
- **Multicollinearity & Standardized Regression (`n = 2,850`):** `Overall_Hospital_Rating` (`VIF = 17.23`), `Recommend_Hospital` (`VIF = 10.49`), `Nurse_Communication` (`VIF = 6.26`), `Doctor_Communication` (`VIF = 3.68`), `Medication_Communication` (`VIF = 4.24`), `Discharge_Information` (`VIF = 2.77`), `Cleanliness` (`VIF = 1.81`), and `Quietness` (`VIF = 1.94`); standardized 8-predictor regression achieves `R² = 0.062`.
- **Tuned Models Comparison (`n = 570` test set):** `Random Forest (tuned)` (`Accuracy = 0.600`, `ROC-AUC = 0.605`), `KNN (tuned, 1-SE, K=241)` (`Accuracy = 0.5912`, `ROC-AUC = 0.6004`), `XGBoost (tuned)` (`Accuracy = 0.5789`, `ROC-AUC = 0.6124`), and `MLP (tuned)` (`Accuracy = 0.5754`, `ROC-AUC = 0.6126`) all achieve `ROC-AUC` within `0.600–0.613` (within an approximate `±0.024` margin of error).

## Notes
- `Member 4.docx_3.docx` contains TV4's individual work compiled per task assignment (not the final combined group report).
- Legacy `n = 1,000` SEM metrics without reproducible code on the team dataset are excluded in favor of `semopy` re-estimation directly on all `2,850` hospitals, and `#Test 470` / SHAP outputs align with the `570` real test records (`Split == "test"`) at probability threshold `0.5`.
