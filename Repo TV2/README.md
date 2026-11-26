# STAT3013 – Patient Experience and 30-Day Heart Failure Readmission

Code for the STAT3013 project that links hospital-level **HCAHPS patient-experience scores** to the **30-day heart failure (HF) readmission rate** of U.S. hospitals, using the merged CMS *HCAHPS* and *Unplanned Hospital Visits* datasets.

Two questions are studied on the same data and the same fixed train/test split:

| Task | Target | Method | Script |
| --- | --- | --- | --- |
| Regression | `READM_30_HF` (readmission rate, %) | OLS with VIF-based multicollinearity treatment and assumption checks | `Regression.py` |
| Classification | `READM_HF_High` (1 = high-readmission hospital) | XGBoost with randomized hyper-parameter search | `Xgboost.py` |

---

## Repository structure

```
.
├── STAT3013_final_model_data.csv        # Shared dataset (2,850 hospitals, no missing values)
├── prepare_split_and_standardize.py     # Shared loader, scaler and evaluation function
├── Regression.py                        # OLS regression + diagnostics
├── Xgboost.py                           # XGBoost classifier + tuning
├── xgb.pkl                              # Saved tuned XGBoost model (joblib)
├── outputs_regression/                  # Regression results (CSV + PNG)
│   ├── vif_full.csv, vif_reduced.csv
│   ├── regression_metrics.csv
│   ├── coefficients_model_B.csv
│   ├── assumption_tests.csv
│   ├── residual_diagnostics.png
│   └── actual_vs_predicted_test.png
└── outputs_xgboost/                     # XGBoost results (CSV, JSON, PNG)
    ├── xgboost_results.csv
    ├── best_params.json
    ├── tuning_top15.csv
    ├── xgboost_feature_importance.csv / .png
    ├── xgboost_roc.png
    └── xgboost_confusion_matrix.png
```

---

## Data

`STAT3013_final_model_data.csv` has one row per hospital (2,850 rows, 16 columns).

| Column | Description |
| --- | --- |
| `Facility ID` | CMS facility identifier (read as **text** to keep leading zeros, e.g. `010001`) |
| `Facility_Name`, `State` | Hospital name and state |
| `Nurse_Communication` | HCAHPS: communication with nurses |
| `Doctor_Communication` | HCAHPS: communication with doctors |
| `Medication_Communication` | HCAHPS: communication about medicines |
| `Discharge_Information` | HCAHPS: discharge information |
| `Cleanliness` | HCAHPS: cleanliness of hospital environment |
| `Quietness` | HCAHPS: quietness of hospital environment |
| `Overall_Hospital_Rating` | HCAHPS: overall hospital rating |
| `Recommend_Hospital` | HCAHPS: willingness to recommend the hospital |
| `READM_30_HF` | 30-day HF readmission rate (%) – **regression target** |
| `READM_HF_High` | 1 if `READM_30_HF` > 21.3 (median of the training set), else 0 – **classification target** |
| `Split` | `train` (2,280 hospitals) or `test` (570 hospitals), `random_state=42` |
| `Start_Date`, `End_Date` | Reporting period (2023-07-01 to 2025-06-30) |

The eight HCAHPS columns are the model features. The share of "High" hospitals is about 0.48 in the full data.

> **Do not re-split the data.** All models use the provided `Split` column so that results are comparable on the same test set.

---

## Setup

Python 3.9+ is recommended.

```bash
pip install pandas numpy scipy scikit-learn statsmodels xgboost matplotlib joblib
```

`xgb.pkl` is a pickled model: load it with the same `xgboost` / `scikit-learn` versions used to create it, or simply re-run `Xgboost.py` to regenerate it.

---

## Usage

Run all scripts **from the repository root** (file paths are relative).

```bash
# 1. Sanity check: loads the data and runs a Random Forest baseline
python prepare_split_and_standardize.py

# 2. OLS regression (writes to outputs_regression/)
python Regression.py

# 3. XGBoost classifier with tuning (writes to outputs_xgboost/ and xgb.pkl)
python Xgboost.py
```

### Shared module – `prepare_split_and_standardize.py`

Other model scripts import from it so that everyone uses identical data and metrics:

```python
from prepare_split_and_standardize import load_data, scale, evaluate, FEATURES

X_train, X_test, y_train, y_test = load_data()      # fixed split, 8 features
X_train_s, X_test_s = scale(X_train, X_test)        # scaler fit on TRAIN only (for KNN / MLP)
evaluate(model, X_test, y_test, "My model")         # Accuracy, Precision, Recall, F1, ROC-AUC
```

Tree ensembles (Random Forest, XGBoost) do not need scaling; KNN and MLP do.

### Regression – `Regression.py`

1. Fits **Model A** (OLS, all 8 predictors) on the training set.
2. Computes VIF and iteratively drops the predictor with the highest VIF until all VIF < 10. `Overall_Hospital_Rating` is removed (VIF = 17.65; it is highly correlated with `Recommend_Hospital`, r ≈ 0.94).
3. Fits **Model B** (OLS, 7 predictors).
4. Checks assumptions on Model B: Shapiro-Wilk, Jarque-Bera, Breusch-Pagan, Rainbow, Durbin-Watson and Cook's distance (> 4/n). Also reports HC3 robust standard errors.
5. Reports R², adjusted R² and p-values on the **train** set, and R², RMSE and MAE on the **test** set.

### XGBoost – `Xgboost.py`

1. Trains a default `XGBClassifier` as a baseline.
2. Tunes hyper-parameters with `RandomizedSearchCV` (80 iterations, 5-fold stratified CV on the training set, scoring = ROC-AUC).
3. Evaluates both models on the held-out test set and saves the ROC curve, confusion matrix, feature importance and the tuned model (`xgb.pkl`).

---

## Results

### Regression (target: `READM_30_HF`)

| Model | Predictors | R² train | Adj. R² train | R² test | RMSE test | MAE test |
| --- | --- | --- | --- | --- | --- | --- |
| A – full OLS | 8 | 0.066 | 0.063 | 0.042 | 1.273 | 0.965 |
| B – after VIF treatment | 7 | 0.065 | 0.062 | 0.044 | 1.272 | 0.963 |

Significant coefficients in Model B (p < 0.001, unchanged under HC3 robust errors): `Nurse_Communication` (+0.098), `Discharge_Information` (−0.067) and `Recommend_Hospital` (−0.068). The other predictors are not significant at the 5% level.

Assumption checks on Model B: residuals deviate from normality (Shapiro-Wilk and Jarque-Bera p < 0.05), heteroscedasticity is present (Breusch-Pagan p < 0.05), no evidence of non-linearity (Rainbow p = 0.99), Durbin-Watson = 1.80, and 102 of 2,280 observations have Cook's D > 4/n.

### Classification (target: `READM_HF_High`)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| XGBoost (default) | 0.539 | 0.496 | 0.517 | 0.507 | 0.572 |
| XGBoost (tuned) | 0.579 | 0.544 | 0.502 | 0.522 | 0.612 |

Best cross-validated ROC-AUC on the training set is 0.614. Tuned hyper-parameters are stored in `outputs_xgboost/best_params.json`. In the tuned model, `Recommend_Hospital`, `Discharge_Information` and `Overall_Hospital_Rating` have the highest feature importance.

### Interpretation

Hospital-level patient-experience scores have **only a weak association** with HF readmission: the regression explains about 4–7% of the variance and the best classifier reaches an ROC-AUC of about 0.61. This is consistent with the literature, which reports small effects for HCAHPS scores and modest discrimination for readmission models. Results should be read as exploratory and not as a clinical prediction tool.

---

## Notes

- Scripts create their output folders automatically.
- The classification threshold (21.3%) is the median of the **training** set only, so the test set does not influence the label definition.
- Random seed is fixed to `42` everywhere for reproducibility.

## Data source

Centers for Medicare & Medicaid Services (CMS), *HCAHPS – Hospital* and *Unplanned Hospital Visits – Hospital*, Provider Data Catalog: <https://data.cms.gov/provider-data/>
