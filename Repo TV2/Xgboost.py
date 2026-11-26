"""STAT3013: XGBoost classifier for hospitals with HIGH vs LOW 30-day heart failure readmission rate."""

import json
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from prepare_split_and_standardize import load_data, evaluate, FEATURES

OUT_DIR = "outputs_xgboost"
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42

X_train, X_test, y_train, y_test = load_data()
print(f"Train {len(X_train)} | Test {len(X_test)} | High share in train {y_train.mean():.3f}")

# 1. Baseline (default parameters)
base = xgb.XGBClassifier(
    eval_metric="logloss", random_state=SEED, n_jobs=-1,
).fit(X_train, y_train)
res_base = evaluate(base, X_test, y_test, "XGBoost (default)")

# 2. Tuning with CV on train
# (The expected ROC-AUC ~0.55-0.65)
param_dist = {
    "n_estimators": randint(100, 600),
    "learning_rate": loguniform(0.01, 0.2),
    "max_depth": randint(2, 6),
    "min_child_weight": randint(1, 20),
    "subsample": uniform(0.6, 0.4),          # 0.6 - 1.0
    "colsample_bytree": uniform(0.5, 0.5),   # 0.5 - 1.0
    "gamma": uniform(0, 2),
    "reg_alpha": loguniform(1e-3, 10),
    "reg_lambda": loguniform(0.5, 20),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(
    xgb.XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=1),
    param_distributions=param_dist,
    n_iter=80,
    scoring="roc_auc",
    cv=cv,
    random_state=SEED,
    n_jobs=-1,
    refit=True,
    verbose=1,
)
search.fit(X_train, y_train)
best = search.best_estimator_
print(f"\nBest cross-validated ROC-AUC (train): {search.best_score_:.4f}")
print("Best parameters:", search.best_params_)

cv_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
cv_table[["rank_test_score", "mean_test_score", "std_test_score"]
         + [c for c in cv_table.columns if c.startswith("param_")]
         ].head(15).to_csv(f"{OUT_DIR}/tuning_top15.csv", index=False)
with open(f"{OUT_DIR}/best_params.json", "w") as f:
    json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else int(v))
               for k, v in search.best_params_.items()}, f, indent=2)

# 3. Final evaluation on TEST
print("\n=== Evaluation on the TEST SET ===")
res_best = evaluate(best, X_test, y_test, "XGBoost (tuned)")
results = pd.DataFrame([res_base, res_best]).round(4)
results.to_csv(f"{OUT_DIR}/xgboost_results.csv", index=False)
print("\n", results.to_string(index=False))

# 4. ROC curve + confusion matrix + feature importance
proba_base = base.predict_proba(X_test)[:, 1]
proba_best = best.predict_proba(X_test)[:, 1]
plt.figure(figsize=(6, 5.5))
for p, lab in [(proba_base, "default"), (proba_best, "tuned")]:
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"XGBoost {lab} (AUC = {roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve - XGBoost (test set)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/xgboost_roc.png", dpi=150)
plt.close()

ConfusionMatrixDisplay(confusion_matrix(y_test, best.predict(X_test)),
                       display_labels=["Low", "High"]).plot(cmap="Blues")
plt.title("Confusion matrix - XGBoost (tuned, test set)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/xgboost_confusion_matrix.png", dpi=150)
plt.close()

imp = pd.Series(best.feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", figsize=(7, 4.5), title="XGBoost feature importance (tuned model)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/xgboost_feature_importance.png", dpi=150)
plt.close()
imp.sort_values(ascending=False).to_csv(f"{OUT_DIR}/xgboost_feature_importance.csv")

# 5. Save the model
joblib.dump(best, "xgb.pkl")
print(f"\nSaved xgb.pkl (xgboost {xgb.__version__}) and results to {OUT_DIR}/")