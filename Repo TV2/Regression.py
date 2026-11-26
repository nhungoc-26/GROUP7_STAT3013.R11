"""STAT3013: Regression predicting READM_30_HF (30-day heart failure readmission rate, %) from HCAHPS patient-experience scores.
Use the provided Split column (train 2,280 / test 570). Do NOT re-split.
- Models are fitted on TRAIN. R^2 / adjusted R^2 / p-values are reported on TRAIN;
  R^2, RMSE and MAE are reported on TEST (same test set as the classification models).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.stattools import durbin_watson
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_FILE = "STAT3013_final_model_data.csv"
OUT_DIR = "outputs_regression"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    "Nurse_Communication", "Doctor_Communication", "Medication_Communication",
    "Discharge_Information", "Cleanliness", "Quietness",
    "Overall_Hospital_Rating", "Recommend_Hospital",
]
TARGET = "READM_30_HF"

df = pd.read_csv(DATA_FILE, dtype={"Facility ID": str})
train, test = df[df["Split"] == "train"], df[df["Split"] == "test"]
y_train, y_test = train[TARGET], test[TARGET]


def vif_table(X):
    """Variance Inflation Factor for each feature, sorted from highest to lowest."""
    Xc = sm.add_constant(X)
    return pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)


def fit_and_report(features, name):
    """Fit OLS on train; return the fitted model and a dict of train/test metrics."""
    X_tr = sm.add_constant(train[features])
    X_te = sm.add_constant(test[features], has_constant="add")
    model = sm.OLS(y_train, X_tr).fit()
    pred = model.predict(X_te)
    metrics = {
        "Model": name,
        "n_features": len(features),
        "R2_train": model.rsquared,
        "Adj_R2_train": model.rsquared_adj,
        "F_pvalue": model.f_pvalue,
        "R2_test": r2_score(y_test, pred),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, pred)),
        "MAE_test": mean_absolute_error(y_test, pred),
    }
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(model.summary())
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})
    return model, metrics


# 1. Full model
print("VIF - full model (8 predictors):")
vif_full = vif_table(train[FEATURES])
print(vif_full.round(2))
vif_full.to_csv(f"{OUT_DIR}/vif_full.csv", index=False)

model_full, m_full = fit_and_report(FEATURES, "Model A - OLS, full (8 predictors)")

# 2. Handle multicollinearity
# Iteratively drop the predictor with the highest VIF until all VIF < 10
# (Overall_Hospital_Rating and Recommend_Hospital are correlated at ~0.94).
reduced = FEATURES.copy()
while True:
    v = vif_table(train[reduced])
    if v["VIF"].max() < 10:
        break
    drop = v.loc[0, "feature"]
    print(f"\nDropping '{drop}' (VIF = {v.loc[0, 'VIF']:.2f})")
    reduced.remove(drop)
print("\nVIF after removing collinear predictors:")
print(vif_table(train[reduced]).round(2))
vif_table(train[reduced]).to_csv(f"{OUT_DIR}/vif_reduced.csv", index=False)

model_red, m_red = fit_and_report(reduced, "Model B - OLS after multicollinearity treatment")

# 3. Assumption checks (Model B)
resid = model_red.resid
fitted = model_red.fittedvalues
infl = OLSInfluence(model_red)

sw_stat, sw_p = stats.shapiro(resid)
jb_stat, jb_p = stats.jarque_bera(resid)
bp_stat, bp_p, _, _ = het_breuschpagan(resid, model_red.model.exog)
rb_stat, rb_p = linear_rainbow(model_red)
dw = durbin_watson(resid)
cooks = infl.cooks_distance[0]
n, k = len(resid), len(reduced) + 1

assumptions = pd.DataFrame([
    ["Normality of residuals", "Shapiro-Wilk", sw_stat, sw_p, "p < 0.05 => residuals deviate from normal"],
    ["Normality of residuals", "Jarque-Bera", jb_stat, jb_p, "p < 0.05 => residuals deviate from normal"],
    ["Homoscedasticity", "Breusch-Pagan", bp_stat, bp_p, "p < 0.05 => heteroscedasticity"],
    ["Linearity", "Rainbow", rb_stat, rb_p, "p < 0.05 => possible non-linearity"],
    ["Independence of errors", "Durbin-Watson", dw, np.nan, "~2 means no autocorrelation"],
    ["Influential points", "Cook's D > 4/n", int((cooks > 4 / n).sum()), np.nan, f"count out of {n} observations"],
], columns=["Assumption", "Test", "Statistic", "p-value", "Note"])
print("\nAssumption checks:")
print(assumptions.to_string(index=False))
assumptions.to_csv(f"{OUT_DIR}/assumption_tests.csv", index=False)

# If residuals are non-normal / heteroscedastic, also report heteroscedasticity-robust (HC3) standard errors
model_robust = sm.OLS(y_train, sm.add_constant(train[reduced])).fit(cov_type="HC3")
print("\nCoefficients with robust standard errors (HC3):")
print(model_robust.summary().tables[1])

# 4. Coefficient table + model comparison
coef = pd.DataFrame({
    "coef": model_red.params, "std_err": model_red.bse,
    "t": model_red.tvalues, "p_value": model_red.pvalues,
    "p_value_HC3": model_robust.pvalues,
    "CI_low": model_red.conf_int()[0], "CI_high": model_red.conf_int()[1],
}).round(4)
coef.to_csv(f"{OUT_DIR}/coefficients_model_B.csv")
pd.DataFrame([m_full, m_red]).to_csv(f"{OUT_DIR}/regression_metrics.csv", index=False)

# 5. Residual plots
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
ax[0, 0].scatter(fitted, resid, s=8, alpha=0.4)
ax[0, 0].axhline(0, color="red", lw=1)
ax[0, 0].set(title="Residuals vs Fitted", xlabel="Fitted values", ylabel="Residuals")

sm.qqplot(resid, line="s", ax=ax[0, 1], markersize=3)
ax[0, 1].set_title("Normal Q-Q plot of residuals")

ax[1, 0].hist(resid, bins=40, edgecolor="white")
ax[1, 0].set(title="Histogram of residuals", xlabel="Residuals", ylabel="Count")

ax[1, 1].scatter(fitted, np.sqrt(np.abs(infl.resid_studentized_internal)), s=8, alpha=0.4)
ax[1, 1].set(title="Scale-Location", xlabel="Fitted values", ylabel="sqrt(|studentized residuals|)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/residual_diagnostics.png", dpi=150)
plt.close()

pred_te = model_red.predict(sm.add_constant(test[reduced], has_constant="add"))
plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_test, pred_te, s=10, alpha=0.5)
lim = [min(y_test.min(), pred_te.min()), max(y_test.max(), pred_te.max())]
plt.plot(lim, lim, "r--", lw=1)
plt.xlabel("Actual READM_30_HF (%)")
plt.ylabel("Predicted READM_30_HF (%)")
plt.title(f"Test set: actual vs predicted (R² = {m_red['R2_test']:.3f})")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/actual_vs_predicted_test.png", dpi=150)
plt.close()

print(f"\nResults saved to folder: {OUT_DIR}/")