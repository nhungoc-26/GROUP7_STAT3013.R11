# TV1 results summary – Descriptive, Inferential and Random Forest (STAT3013)

Data: `STAT3013_final_model_data.csv` (v2), 2,850 hospitals, fixed split 2,280 train / 570 test. Classification target `READM_HF_High` (READM_30_HF > 21.3, training-set median). 8 HCAHPS linear mean scores as inputs. Notebook: `TV1_Descriptive_Inferential_RandomForest.ipynb`.

## 1. Descriptive statistics
- HCAHPS means 76–91 points, mostly slightly left-skewed. Lowest: Medication_Communication (76.4). Most variable: Quietness (SD 5.4).
- READM_30_HF: mean 21.35%, SD 1.30, range 17.1–28.4, close to normal (skewness 0.37).
- 39–94 IQR outliers per variable, kept because they are real hospital scores.
- Correlation with READM_30_HF is negative and weak: Recommend −0.20, Discharge −0.20, Overall −0.18, Doctor −0.14, Nurse −0.13, Medication −0.13, Quietness −0.09, Cleanliness −0.07.
- Multicollinearity: Overall_Hospital_Rating vs Recommend_Hospital r = 0.94.

## 2. Inferential statistics (α = 0.05)
**Welch t-test, High (n = 1,368) vs Low (n = 1,482), Bonferroni-corrected for 8 tests:** H0 rejected for all 8 scores; the High group always scores lower.

| Variable | Difference (High − Low) | t | p (Bonferroni) | Cohen's d |
|---|---|---|---|---|
| Recommend_Hospital | −1.58 | −9.22 | 4.7e-19 | −0.35 |
| Discharge_Information | −1.34 | −9.15 | 8.5e-19 | −0.35 |
| Overall_Hospital_Rating | −1.07 | −8.18 | 3.6e-15 | −0.31 |
| Doctor_Communication | −0.68 | −6.98 | 2.9e-11 | −0.26 |
| Medication_Communication | −1.04 | −6.32 | 2.5e-09 | −0.24 |
| Nurse_Communication | −0.58 | −6.18 | 6.1e-09 | −0.23 |
| Quietness | −0.91 | −4.52 | 5.1e-05 | −0.17 |
| Cleanliness | −0.49 | −2.96 | 0.025 | −0.11 |

Significant but small effects (|d| ≤ 0.35).

**Chi-square, Recommend_Hospital quartile × READM_HF_High:** χ² = 64.66, df = 3, p = 5.9e-14, Cramér's V = 0.15. High share: Q1 59.0% → Q2 48.9% → Q3 46.1% → Q4 38.0%. Smallest expected frequency 341.8.

## 3. Random Forest
**Method:** default baseline → grid search of 48 combinations (max_depth {None, 4, 8, 12} × min_samples_leaf {1, 10, 30, 60} × max_features {sqrt, 0.5, 1.0}) with RepeatedStratifiedKFold 5×3 → depth check 2–6 around the best value → final model scored with 5×10 CV (same protocol as the other classifiers) → shared `evaluate()` on the test set.

**Final hyper-parameters:** n_estimators = 500, max_depth = 4, min_samples_leaf = 30, max_features = 1.0, random_state = 42.

| Model | CV ROC-AUC (5×10) | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|
| RF default | 0.568 ± 0.020 | 0.540 | 0.498 | 0.471 | 0.484 | 0.566 |
| **RF tuned** | **0.606 ± 0.022** | **0.600** | **0.583** | **0.444** | **0.504** | **0.605** |

Confusion matrix (tuned): TN 226, FP 83, FN 145, TP 116. Train ROC-AUC 0.664.

**Permutation importance (drop in test ROC-AUC, 30 repeats):** Recommend_Hospital 0.063, Discharge_Information 0.027, Quietness 0.006, Nurse 0.003, Cleanliness 0.002, others ≈ 0.

## Key points for the report
- The default RF overfits (test ROC-AUC 0.566); limiting depth and leaf size raises it to 0.605.
- RF performs on par with KNN (0.600) and a logistic reference (~0.60): the 8 HCAHPS scores carry only this much signal.
- Recommend_Hospital and Discharge_Information are the most important variables, consistent with the correlations, t-tests and chi-square test.
- Overall_Hospital_Rating adds almost nothing on its own because it duplicates Recommend_Hospital (r = 0.94).
- Recall for the High class is 0.44: many high-readmission hospitals are missed.

## Hand-over
- `rf.pkl` → SHAP notebook. Load with `joblib.load("rf.pkl")` and use the FEATURES order from `prepare_split_and_standardize.py`.
- `rf_results.json` and `figures/table_rf_test_results.csv` → model comparison table.
