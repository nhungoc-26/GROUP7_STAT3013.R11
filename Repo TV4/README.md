# Repo TV4 – SEM (`semopy`) & SHAP Explainability

Member 4 (TV4) deliverables for **GROUP7_STAT3013.R11** on 30-day heart failure readmissions (`READM_30_HF`).

## Scope & Conventions
- **Dataset:** `2,850` hospitals (`STAT3013_final_model_data.csv` / `data/processed.csv`, no missing values) and `8` HCAHPS linear score predictors.
- **Label vs. Threshold:** `21.3` (training-set median of `READM_30_HF`) is used strictly to define `READM_HF_High = 1` (`READM_30_HF > 21.3`); classification models are evaluated at probability threshold **`0.5`**.
- **SEM (`semopy`):** Re-estimated directly on all `2,850` hospitals (replacing prior `n = 1,000` figures that lacked reproducible code) with structural path diagram and code.
- **SHAP Explainability:** Recomputed on `570` real test rows (`Split == "test"`, `random_state = 42`) at threshold `0.5` (including local explanation for `#Test 470`).

## Repository Structure
- `README.md` — Overview of TV4 deliverables, scope, and files
- `phan 1 va phan 2.docx` — Draft/compiled TV4 report content and tables
- `outputs_tv4_part1_sem_*` — SEM correlation, descriptive, statistics, and TV4 part 1 output files
- `sem_*` / `shap_*` *(when uploaded)* — Reproducible `semopy` & `SHAP` scripts and generated figures

## Assigned Report Sections
- **Section 2 (Literature Review — DL & XAI contribution with TV2)**
- **Section 3 (Research Methodology — SEM & SHAP parts)**
- **Section 4 (Implementation & Experiments — SEM & SHAP runs on `2,850` / `570` rows)**
- **Section 5 (5.4 SEM Results, 5.7 SHAP Interpretability, 5.8 Model Comparison with TV5)**
- **Section 7 (References — IEEE format with TV2)**
