# TV3 – MLP classifier and ARIMA forecasting

Part of the STAT3013 project on patient experience (HCAHPS) and 30-day heart failure (HF) readmission of U.S. hospitals. This folder holds Member 3's work: a multilayer perceptron (deep learning) classifier and an ARIMA forecast on a **simulated** monthly series.

| Task | Target | Method | Script |
|---|---|---|---|
| Classification | `READM_HF_High` (1 = high-readmission hospital) | MLP (scikit-learn) with grid search | `mlp_member3_v2.py` |
| Forecasting | monthly readmission rate (simulated) | ARIMA(p,d,q) with AIC selection | `arima_member3.py` |
| Figure | MLP architecture diagram | matplotlib | `draw_architecture.py` |

## Folder structure
```
Repo TV3/
├── mlp_member3_v2.py          # MLP + tuning + evaluation
├── arima_member3.py           # simulated series + ARIMA
├── draw_architecture.py       # architecture figure
├── results_mlp/               # mlp_results_table.csv, mlp_summary.json, mlp_grid_top10.csv,
│                              # mlp_learning_curve.png, mlp_roc.png, mlp_roc_points.npy,
│                              # mlp_confusion.png, mlp_architecture.png, mlp.pkl
└── results_arima/             # forecast_plot.png, acf_pacf.png, adf_test.csv,
                               # arima_aic_grid.csv, arima_metrics.csv, arima_summary.json,
                               # simulated_series.csv
```

## Data
The scripts use the shared dataset `STAT3013_final_model_data.csv` (2,850 hospitals, 8 HCAHPS features) and the shared module `prepare_split_and_standardize.py` (fixed `Split` column: 2,280 train / 570 test). Do not re-split the data. `READM_HF_High` = 1 if `READM_30_HF` > 21.3 (median of the training set).

## Setup
Python 3.9+ is recommended.
```
pip install pandas numpy scikit-learn statsmodels matplotlib joblib
```
`mlp.pkl` is a pickled model: only load it with the same scikit-learn version, or re-run `mlp_member3_v2.py` to regenerate it.

## Usage
Copy the three scripts into the folder that contains `STAT3013_final_model_data.csv` and `prepare_split_and_standardize.py` (the shared files from the group), then run from that folder:
```
python mlp_member3_v2.py      # writes results_mlp/
python arima_member3.py       # writes results_arima/
python draw_architecture.py   # writes results_mlp/mlp_architecture.png
```
The random seed is 42 everywhere. The ARIMA script reads the CSV only to set the starting level of the simulated series.

## MLP (mlp_member3_v2.py)
- Pipeline: `StandardScaler` (fit on train only) then `MLPClassifier` with early stopping (15% of train held out, patience 20 epochs).
- Tuning: `GridSearchCV`, 5-fold stratified CV on the training set, scoring ROC-AUC, 96 settings (hidden layers, activation, alpha, learning rate).
- Best setting: hidden layers (64, 32, 16), tanh, alpha = 0.1, learning rate 0.005, 3,201 parameters, CV AUC = 0.608.
- The test set is used once, at the end. A 10-seed stability check is also run.

| Model (test n = 570) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| MLP baseline (32,16), ReLU | 0.549 | 0.516 | 0.245 | 0.332 | 0.572 |
| MLP tuned (64,32,16), tanh | 0.575 | 0.541 | 0.475 | 0.506 | 0.613 |

Confusion matrix (TN, FP, FN, TP) = 204, 105, 137, 124. Over 10 random seeds, ROC-AUC = 0.611 ± 0.010 and F1 = 0.518 ± 0.014.

## ARIMA (arima_member3.py)
**The CMS data has one period per hospital, so no real monthly series exists. ARIMA is run on a simulated series, and its results illustrate the method only.**
- Simulated series: 60 months (2021-01 to 2025-12), start level = real mean readmission rate (21.35%), trend −0.03 percentage points per month, AR(1) noise (phi = 0.6, sigma = 0.25), seed 42. Train 48 months, test 12 months.
- ADF test: d = 0 gives p = 0.230, d = 1 gives p < 0.001, so d = 1.
- AIC grid over p, q in {0, 1, 2} selects ARIMA(0,1,2) (AIC = −7.06). Ljung–Box p = 0.454.

| Model (12-month hold-out) | RMSE (pp) | MAE (pp) |
|---|---|---|
| ARIMA(0,1,2) | 0.500 | 0.405 |
| Naive (last value) | 0.547 | 0.456 |
| Drift | 0.384 | 0.301 |

## Interpretation
Survey scores have only a weak link with HF readmission (Spearman correlations −0.06 to −0.19), so the MLP reaches an AUC of about 0.61. For ARIMA, a simple drift baseline beats the model because the simulated series has a trend and the ARIMA forecast stays flat. Results are exploratory, not a clinical prediction tool.
