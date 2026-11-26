# Repo TV4 – SEM, SHAP Interpretability & Model Comparison

Deliverables of **Member 4 (TV4)** for **GROUP7_STAT3013.R11** linking hospital-level HCAHPS patient experience scores to 30-day heart failure readmission (`READM_30_HF`).

| Task | Target / Scope | Method | Artifacts |
| :--- | :--- | :--- | :--- |
| **Multicollinearity & Path Analysis** | `READM_30_HF` (`n = 2,850` hospitals) | VIF diagnostics, standardized 8-feature regression (`R² = 0.062`), and `semopy` re-estimation | `outputs_tv4_part1_sem_tv4_part1_sem_analysis.py`, `outputs_tv4_part1_sem_sem_*.csv` |
| **SHAP & Model Comparison** | `READM_HF_High` (`570` test rows, threshold `0.5`) | `rf.pkl` Gini & SHAP (`#Test 470` case) + comparison across tuned classifiers | `shap_explanation_part2.py`, SHAP PNG/JPG plots, `Member 4.docx` |

## Repository structure

```text
GROUP7_STAT3013.R11/
├── Member 4.docx
└── Repo TV4/
    ├── README.md
    ├── outputs_tv4_part1_sem_tv4_part1_sem_analysis.py
    ├── shap_explanation_part2.py
    ├── outputs_tv4_part1_sem_sem_correlation_matrix.csv
    ├── outputs_tv4_part1_sem_sem_descriptive_stats.csv
    ├── outputs_tv4_part1_sem_sem_standardized_paths_8features.csv
    ├── SHAP.png
    ├── shap_bar_vs_gini_rf_500trees.jpg.png
    ├── shap_beeswarm_rf_test570_v1.png.png
    ├── shap_beeswarm_rf_test570_v2.png.png
    └── shap_local_test470.png.png
## Data & Conventions
- **Dataset:** `2,850` hospitals (`STAT3013_final_model_data.csv` / `data/processed.csv`, no missing values), with a shared held-out test set of `570` rows (`Split == "test"`, `random_state = 42`).
- **Label vs. Threshold:** `21.3` (training-set median of `READM_30_HF`) is used strictly to define `READM_HF_High = 1` when `READM_30_HF > 21.3` (else `0`); classification models are evaluated at probability threshold **`0.5`**.
pip install pandas numpy scikit-learn statsmodels semopy shap matplotlib
python outputs_tv4_part1_sem_tv4_part1_sem_analysis.py
python shap_explanation_part2.py
## Results & Notes
- **VIF & Standardized Regression (`n = 2,850`):** `Overall_Hospital_Rating` (`17.23`), `Recommend_Hospital` (`10.49`), `Nurse_Communication` (`6.26`), `Doctor_Communication` (`3.68`), `Medication_Communication` (`4.24`), `Discharge_Information` (`2.77`), `Cleanliness` (`1.81`), and `Quietness` (`1.94`); standardized 8-predictor regression achieves `R² = 0.062`.
- **Tuned Models (`n = 570` test set):** `Random Forest`, `KNN (K=241)`, `XGBoost`, and `MLP` reach `ROC-AUC` in `0.600–0.613` (`0.6004` to `0.6126`), within an approximate `±0.024` margin of error.
- **Audit Note:** Legacy `n = 1,000` SEM figures without reproducible code on the team dataset are excluded; updated `semopy` (`n = 2,850`) and SHAP (`570` real test rows, `#Test 470`) outputs are documented in `Member 4.docx` at the repository root.
