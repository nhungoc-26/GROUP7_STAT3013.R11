STAT3013 – Final model dataset (v2)

FILE TO USE
STAT3013_final_model_data.csv – 2,850 hospitals x 16 columns, no missing values.
Read it with: pd.read_csv("STAT3013_final_model_data.csv", dtype={"Facility ID": str})

COLUMNS
- Facility ID (6-character text, leading zeros kept), Facility_Name, State
- 8 HCAHPS linear scores (0-100): Nurse_Communication, Doctor_Communication,
  Medication_Communication, Discharge_Information, Cleanliness, Quietness,
  Overall_Hospital_Rating, Recommend_Hospital
- READM_30_HF: heart failure 30-day readmission rate (%) – target for regression
- READM_HF_High: 1 if READM_30_HF > 21.3, else 0 – target for classification.
  21.3 is the median of the TRAIN set only (no leakage). Train 48.6% High, test 45.8% High.
- Split: train (2,280) / test (570), random_state=42. Everyone uses this split, do not re-split.
- Start_Date, End_Date: READM_30_HF measurement period (07/2023 - 06/2025)

CHANGES FROM v1
1. Dropped 289 hospitals that had all 8 HCAHPS features missing (v1 filled them with the median,
   which invented whole rows). 3,139 -> 2,850 hospitals.
2. Fixed Facility ID: the v1 complete-cases file had lost the leading zero for 483 IDs
   (AL, AK, AZ, AR, CA). All IDs are now 6 characters.
3. Added a fixed Split column and the class target so all models use the same test set.
4. Script: removed the duplicate RF fit on data with NaN; added a shared evaluate() that reports
   Accuracy, Precision, Recall, F1, ROC-AUC and the confusion matrix.

NOTES FOR THE REPORT
- Hospital-level analysis, not individual patient prediction.
- VA and military hospitals have no Facility ID in HCAHPS (11,152 rows) and cannot be merged.
- Periods differ: HCAHPS 10/2024-09/2025, readmission 07/2023-06/2025.
- The 8 features correlate only -0.06 to -0.20 with READM_30_HF; expect ROC-AUC about 0.55-0.65.
- Overall_Hospital_Rating and Recommend_Hospital correlate 0.94 – check VIF in regression.
- Only one period per hospital, so ARIMA needs simulated or other time-series data.
- Source cleaned files (HCAHPS_Hospital_cleaned.csv, Unplanned_Hospital_Visits_cleaned.csv) are
  unchanged from v1 and kept separately because of their size.
