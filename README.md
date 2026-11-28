# An Integrated Statistical, Machine Learning, and Time-Series Framework for Analyzing Patient Experience and Hospital Readmission Risk

**STAT3013 · Class R11 · Group 07** · University of Information Technology, VNU-HCM

This project links hospital-level patient-experience scores (CMS HCAHPS survey) to the 30-day heart failure readmission rate (`READM_30_HF`) of 2,850 U.S. hospitals. It combines:

- **Statistics:** descriptive statistics, Welch t-tests, chi-square test, OLS regression, structural equation modeling (SEM)
- **Machine learning:** Random Forest, XGBoost, MLP neural network and KNN classifiers for high vs. low readmission, explained with SHAP
- **Time series:** ARIMA forecasting of a simulated monthly readmission series

## Links

| Resource | Link |
|---|---|
| Demo video (15 min) | https://drive.google.com/file/d/1WTAaTS_1jLj_-QAvfdTmVAKn3kbHHz3U/view?usp=sharing |
| Processed dataset | [`Repo TV5/data/processed.csv`](Repo%20TV5/data/processed.csv) |
| Source data: HCAHPS Patient Survey (CMS) | https://data.cms.gov/provider-data/dataset/dgck-syfz |
| Source data: Unplanned Hospital Visits (CMS) | https://data.cms.gov/provider-data/dataset/632h-zaca |

## Dataset

- 2,850 hospitals × 16 columns, no missing values (cleaning steps: `Repo TV5/data/README_cleaning.txt`).
- Inputs: 8 HCAHPS linear mean scores (Nurse, Doctor and Medication communication, Discharge information, Cleanliness, Quietness, Overall rating, Recommend hospital).
- Targets: `READM_30_HF` (regression) and `READM_HF_High` = 1 if `READM_30_HF` > 21.3, the training-set median (classification).
- Fixed `Split` column: 2,280 train / 570 test (`random_state = 42`). All models use this split; do not re-split the data.
- `processed.csv` and `TV1_RandomForest/STAT3013_final_model_data.csv` contain the same data. Read `Facility ID` as text: `pd.read_csv(path, dtype={"Facility ID": str})`.

## Repository structure

| Folder | Content | Member |
|---|---|---|
| `TV1_RandomForest/` | Descriptive and inferential statistics, Random Forest (notebook), shared data and `prepare_split_and_standardize.py` | Nguyễn Như Ngọc |
| `Repo TV2/` | OLS regression (`Regression.py`), XGBoost (`Xgboost.py`) | Trần Thái Thông |
| `Repo TV3/` | ARIMA (`arima_member3.py`), MLP (`mlp_member3_v2.py`) | Phan Ngọc Phúc |
| `Repo TV4/` | SEM / path analysis, SHAP explanations | Võ Trọng Hiếu |
| `Repo TV5/` | Processed data, shared `evaluate()` module (`src/evaluate.py`), KNN (`src/knn.py`) | Lê Hữu Đức |

Each folder has its own `README.md` with details and output files.

## Environment

- Python 3.10 or later (tested with Python 3.12); also runs on Google Colab.
- Install all packages used in the project:

```bash
pip install pandas numpy scipy scikit-learn statsmodels xgboost semopy shap matplotlib seaborn joblib
```

Per-member package lists: `TV1_RandomForest/requirements.txt`, `Repo TV5/requirements.txt`.

## How to run

```bash
git clone https://github.com/nhungoc-26/STAT3013.R11_Group07.git
cd STAT3013.R11_Group07
pip install -r "Repo TV5/requirements.txt"
```

Most scripts expect the two shared files `STAT3013_final_model_data.csv` and `prepare_split_and_standardize.py` (both in `TV1_RandomForest/`) to be in the same folder as the script. Copy them into the folder first, then run:

| Step | Command (run inside the folder) | Output |
|---|---|---|
| Statistics + Random Forest | open `TV1_Descriptive_Inferential_RandomForest.ipynb` and run all cells (~10 min) | `figures/`, `rf_results.json`, `rf.pkl` |
| Regression | `python Regression.py` | `outputs_regression/` |
| XGBoost | `python Xgboost.py` | `outputs_xgboost/` |
| MLP | `python mlp_member3_v2.py` | `results_mlp/` |
| ARIMA | `python arima_member3.py` | `results_arima/` |
| SEM / path analysis | `python outputs_tv4_part1_sem_tv4_part1_sem_analysis.py` | SEM CSV files |
| SHAP (needs `rf.pkl` from step 1) | `python shap_explanation_part2.py` | SHAP plots |
| KNN (from `Repo TV5/`) | `python src/knn.py` | `results/knn_*` |

All randomness is fixed (`random_state = 42`), so results are reproducible up to small differences between library versions. Pickled models (`*.pkl`) should be loaded with the same scikit-learn version they were saved with; otherwise re-run the script to regenerate them.

## Main results (test set, 570 hospitals)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | 0.600 | 0.504 | 0.605 |
| XGBoost | 0.579 | 0.522 | 0.612 |
| MLP | 0.575 | 0.506 | 0.613 |
| KNN | 0.591 | 0.492 | 0.600 |

Willingness to recommend the hospital and discharge information are the patient-experience factors most strongly linked to lower heart failure readmission.

## License

This project is released under the [MIT License](LICENSE). The CMS source data are public U.S. government data.
