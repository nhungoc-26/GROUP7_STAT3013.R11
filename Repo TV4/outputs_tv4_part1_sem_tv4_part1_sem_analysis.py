"""TV4 - Part 1: Structural / Path Analysis & Multicollinearity Check on HCAHPS -> READM_30_HF."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

DATA_FILE = "STAT3013_final_model_data.csv"
FEATURES = [
    "Nurse_Communication", "Doctor_Communication", "Medication_Communication",
    "Discharge_Information", "Cleanliness", "Quietness",
    "Overall_Hospital_Rating", "Recommend_Hospital",
]
TARGET_RATE = "READM_30_HF"

df = pd.read_csv(DATA_FILE, dtype={"Facility ID": str})
print("Sample size (n):", len(df))

# VIF check
X_vif = sm.add_constant(df[FEATURES])
for i, col in enumerate(FEATURES):
    vif = variance_inflation_factor(X_vif.values, i + 1)
    print(f"VIF {col:26s}: {vif:6.2f}")

# Standardized path regression
scaler = StandardScaler()
Z = pd.DataFrame(scaler.fit_transform(df[FEATURES + [TARGET_RATE]]),
                 columns=FEATURES + [TARGET_RATE], index=df.index)
model_ols = sm.OLS(Z[TARGET_RATE], sm.add_constant(Z[FEATURES])).fit()
print(f"\nOLS Standardized R2: {model_ols.rsquared:.4f} | Adj R2: {model_ols.rsquared_adj:.4f}")
print(model_ols.summary().tables[1])
