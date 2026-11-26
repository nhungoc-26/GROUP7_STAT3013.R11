# TV1 – Descriptive Statistics, Inferential Statistics and Random Forest

STAT3013 group project. Hospital-level analysis of how HCAHPS patient-experience scores relate to the heart failure 30-day readmission rate (CMS Provider Data Catalog).

## Contents

| File | Purpose |
| --- | --- |
| `TV1_Descriptive_Inferential_RandomForest.ipynb` | Main notebook, already executed, with code, outputs and explanations |
| `rf.pkl` | Final Random Forest model (joblib), used by the SHAP notebook |
| `rf_results.json` | All Random Forest metrics, hyper-parameters and the scikit-learn version |
| `TV1_results_summary.md` | One-page summary of every result, for the report and the model comparison |
| `requirements.txt` | Python packages |
| `figures/` | Figures 1–8 (PNG, 200 dpi) and every result table (CSV) |

## How to run

1. Put `STAT3013_final_model_data.csv` and `prepare_split_and_standardize.py` (from `STAT3013_cleaned_v2.zip`) in this folder.
2. `pip install -r requirements.txt` (not needed on Google Colab).
3. Open the notebook and run all cells. A full run takes about 10 minutes, mostly hyper-parameter tuning.

On Google Colab: File → Upload notebook, then upload the two files above with the Files panel before running all cells.

`rf.pkl` was saved with scikit-learn 1.9.1. A pickled model should be loaded with the same scikit-learn version; if the SHAP notebook uses another version (e.g. on Colab), re-run this notebook there to regenerate `rf.pkl`. All randomness is fixed (`random_state = 42`), so results stay the same up to small differences between library versions.

## Figures and tables

| Output | Content |
| --- | --- |
| `fig1_histograms.png` | Distributions of the 8 HCAHPS scores and READM_30_HF |
| `fig2_boxplots_by_group.png` | HCAHPS scores by readmission group (Low/High) |
| `fig3_correlation_heatmap.png` | Pearson correlation matrix |
| `fig4_scatter.png` | Discharge information and recommend-the-hospital vs readmission rate |
| `fig5_chisquare_quartiles.png` | High-readmission share by recommendation quartile |
| `fig6_rf_tuning.png` | Grid-search results |
| `fig7_rf_roc_confusion.png` | ROC curves and confusion matrix on the test set |
| `fig8_rf_importance.png` | Impurity and permutation importance |
| `table_data_dictionary.csv` | Variable definitions and HCAHPS measure IDs |
| `table_descriptive.csv`, `table_outliers_iqr.csv`, `table_correlation.csv` | Descriptive statistics |
| `table_ttest.csv`, `table_chisquare.csv`, `table_chisquare_contingency.csv` | Inferential tests |
| `table_rf_tuning_all.csv`, `table_rf_tuning_top10.csv`, `table_rf_depth_check.csv` | Tuning |
| `table_rf_test_results.csv`, `table_rf_importance.csv` | Final model results |
