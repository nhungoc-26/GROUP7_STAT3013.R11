"""STAT3013 - Member 3 (TV3): ARIMA forecasting on a SIMULATED monthly series.
Why simulated: the CMS data has ONE period per hospital (no time dimension), so a real
time series cannot be built. The brief allows simulated data if it is stated clearly.
The simulated series is anchored on the real data: its starting level = the real mean
30-day HF readmission rate (read from STAT3013_final_model_data.csv if it is present).
Run:  python arima_member3.py      Output folder: results_arima
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

OUT = "results_arima"; os.makedirs(OUT, exist_ok=True)
SEED = 42; rng = np.random.default_rng(SEED)

# ---- 1. Simulate the series (all settings are written down for the report) ----
LEVEL = 21.4
if os.path.exists("STAT3013_final_model_data.csv"):
    LEVEL = pd.read_csv("STAT3013_final_model_data.csv")["READM_30_HF"].mean()
N, H = 60, 12                      # 60 months (01/2021-12/2025); last 12 months = test
SLOPE, PHI, SIGMA = -0.03, 0.6, 0.25   # slow decline (%/month), AR(1) coefficient, noise SD
noise = np.zeros(N)
for t in range(1, N):
    noise[t] = PHI * noise[t - 1] + rng.normal(0, SIGMA)
idx = pd.date_range("2021-01-01", periods=N, freq="MS")
y = pd.Series(LEVEL + 0.9 + SLOPE * np.arange(N) + noise, index=idx, name="readmission_rate")
y.to_csv(f"{OUT}/simulated_series.csv")
train, test = y.iloc[:-H], y.iloc[-H:]
print(f"Level anchor = {LEVEL:.2f}%  | train {len(train)} months, test {len(test)} months")

# ---- 2. Stationarity (ADF): H0 = unit root (non-stationary) ----
def adf(s):
    stat, p = adfuller(s, autolag="AIC")[:2]; return stat, p
d = 0; s = train.copy(); rows = []
while True:
    stat, p = adf(s); rows.append((d, stat, p))
    if p < 0.05 or d == 2: break
    d += 1; s = train.diff(d).dropna()
adf_tbl = pd.DataFrame(rows, columns=["d", "ADF statistic", "p-value"]).round(4)
print("\nADF test:\n", adf_tbl.to_string(index=False), f"\n=> d = {d}")
adf_tbl.to_csv(f"{OUT}/adf_test.csv", index=False)

# ---- 3. ACF / PACF of the (differenced) training series ----
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
plot_acf(s, lags=15, ax=ax[0]); plot_pacf(s, lags=15, ax=ax[1], method="ywm")
plt.tight_layout(); plt.savefig(f"{OUT}/acf_pacf.png", dpi=200); plt.close()

# ---- 4. Choose (p, q) by AIC on the training set ----
res = []
for p in range(3):
    for q in range(3):
        try:
            m = ARIMA(train, order=(p, d, q)).fit()
            res.append((p, d, q, m.aic, m.bic))
        except Exception:
            pass
grid = pd.DataFrame(res, columns=["p", "d", "q", "AIC", "BIC"]).sort_values("AIC").round(2)
print("\nTop 5 by AIC:\n", grid.head(5).to_string(index=False))
grid.to_csv(f"{OUT}/arima_aic_grid.csv", index=False)
p, d, q = (int(v) for v in grid.iloc[0][["p", "d", "q"]])
model = ARIMA(train, order=(p, d, q)).fit()
print(f"\nSelected ARIMA({p},{d},{q})"); print(model.summary().tables[1])

# ---- 5. Residual check (Ljung-Box: H0 = residuals are white noise) ----
lb = acorr_ljungbox(model.resid.iloc[d:], lags=[10], return_df=True)  # skip the first d residuals (initialisation)
print(f"\nLjung-Box (lag 10) p-value = {lb['lb_pvalue'].iloc[0]:.4f}")

# ---- 6. Forecast the 12 held-out months, RMSE / MAE vs naive baselines ----
fc = model.get_forecast(H); mean = fc.predicted_mean; ci = fc.conf_int(alpha=0.05)
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
mae = lambda a, b: float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
naive = np.repeat(train.iloc[-1], H)
drift = train.iloc[-1] + (train.iloc[-1] - train.iloc[0]) / (len(train) - 1) * np.arange(1, H + 1)
tab = pd.DataFrame({
    "Model": [f"ARIMA({p},{d},{q})", "Naive (last value)", "Drift"],
    "RMSE": [rmse(test, mean), rmse(test, naive), rmse(test, drift)],
    "MAE": [mae(test, mean), mae(test, naive), mae(test, drift)]}).round(4)
print("\n", tab.to_string(index=False)); tab.to_csv(f"{OUT}/arima_metrics.csv", index=False)

plt.figure(figsize=(9, 4))
plt.plot(train.index, train, label="Train (simulated)")
plt.plot(test.index, test, color="k", label="Test (actual simulated)")
plt.plot(mean.index, mean, color="C3", label=f"ARIMA({p},{d},{q}) forecast")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="C3", alpha=0.2, label="95% CI")
plt.title("30-day HF readmission rate - simulated monthly series"); plt.ylabel("Readmission rate (%)")
plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/forecast_plot.png", dpi=200); plt.close()

json.dump(dict(level=float(LEVEL), n=N, horizon=H, slope=SLOPE, phi=PHI, sigma=SIGMA, seed=SEED,
               d=d, order=[p, d, q], aic=float(model.aic), ljung_box_p=float(lb["lb_pvalue"].iloc[0]),
               metrics=tab.to_dict(orient="records")), open(f"{OUT}/arima_summary.json", "w"), indent=2)
print("\nSaved to", OUT)
