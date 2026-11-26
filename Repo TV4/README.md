# Repo TV4 – Structural Analysis, SHAP Interpretability & Model Comparison

Deliverables of **Member 4 (TV4)** for **GROUP7_STAT3013.R11** on hospital-level HCAHPS patient experience and 30-day heart failure readmission (`READM_30_HF`).

| Task | Target / Scope | Method / Focus | Artifacts |
| :--- | :--- | :--- | :--- |
| **Multicollinearity & Path Regression** | `READM_30_HF` (`n = 2,850` hospitals) | VIF diagnostics, standardized 8-feature regression (`R² = 0.062`), and `semopy` structural modeling | `tv4_part1_sem_analysis_2.py`, `sem_*_2.csv` |
| **Interpretability & Benchmark** | `READM_HF_High` (`570` test rows, threshold `0.5`) | `rf.pkl` Gini importance & SHAP (`#Test 470` case) + comparison across tuned classifiers | `anh1_2.png`–`anh5_2.jpg`, report `Member 4.docx` |

## Repository structure

```text
GROUP7_STAT3013.R11/
├── Member 4.docx
└── Repo TV4/
    ├── README.md
    ├── tv4_part1_sem_analysis_2.py
    ├── sem_correlation_matrix_2.csv
    ├── sem_descriptive_stats_2.csv
    ├── sem_standardized_paths_8features_2.csv
    ├── anh1_2.png
    ├── anh2_2.png
    ├── anh3_2.png
    ├── anh4_2.png
    └── anh5_2.jpg
