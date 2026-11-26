"""STAT3013 - Member 3 (TV3): MLP classifier on the group's shared dataset v2.
Put in ONE folder: STAT3013_final_model_data.csv, prepare_split_and_standardize.py, this file.
Run:  python mlp_member3_v2.py
Output: folder 'results_mlp' (tables, figures, mlp.pkl for TV4).
"""
import os, json, joblib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, ConfusionMatrixDisplay
# shared code from TV5 (same data, same split, same evaluate())
from prepare_split_and_standardize import load_data, evaluate, FEATURES

OUT = "results_mlp"
os.makedirs(OUT, exist_ok=True)
SEED = 42

# 1. Shared data + fixed split (train 2,280 / test 570)
X_train, X_test, y_train, y_test = load_data()
print(f"Train {len(X_train)} | Test {len(X_test)} | High share train {y_train.mean():.3f}, test {y_test.mean():.3f}")

# 2. Pipeline: StandardScaler (fit on TRAIN only, inside the pipeline) -> MLP
def make_pipe(**kw):
    p = dict(max_iter=500, early_stopping=True, validation_fraction=0.15,
             n_iter_no_change=20, random_state=SEED)
    p.update(kw)
    return Pipeline([("sc", StandardScaler()), ("mlp", MLPClassifier(**p))])

# 3. Baseline
base = make_pipe(hidden_layer_sizes=(32, 16), activation="relu", alpha=1e-3,
                 learning_rate_init=1e-3).fit(X_train, y_train)
print("\n=== Baseline MLP (32,16) ===")
r_base = evaluate(base, X_test, y_test, "MLP baseline")

# 4. Tuning: 5-fold stratified CV on TRAIN only, scoring ROC-AUC (96 configurations)
grid = {
    "mlp__hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "mlp__learning_rate_init": [1e-3, 5e-3],
}
gs = GridSearchCV(make_pipe(), grid, scoring="roc_auc", n_jobs=-1, refit=True,
                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
gs.fit(X_train, y_train)
print("\nBest params:", gs.best_params_, "| CV AUC =", round(gs.best_score_, 4))
cv = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
cv[["rank_test_score", "mean_test_score", "std_test_score", "params"]].head(10).to_csv(
    f"{OUT}/mlp_grid_top10.csv", index=False)

# 5. Final evaluation on the shared test set (used ONCE)
best = gs.best_estimator_
print("\n=== Tuned MLP ===")
r_best = evaluate(best, X_test, y_test, "MLP tuned")
pred = best.predict(X_test); proba = best.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, pred)
pd.DataFrame([r_base, r_best]).round(4).to_csv(f"{OUT}/mlp_results_table.csv", index=False)

# 6. Seed stability (10 seeds)
bp = {k.replace("mlp__", ""): v for k, v in gs.best_params_.items()}
aucs, f1s = [], []
from sklearn.metrics import roc_auc_score, f1_score
for s in range(10):
    m = make_pipe(**bp); m.set_params(mlp__random_state=s).fit(X_train, y_train)
    aucs.append(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))
    f1s.append(f1_score(y_test, m.predict(X_test)))
print(f"\nStability (10 seeds): AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f} | F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

# 7. Figures (learning curve, ROC, confusion matrix)
mlp = best.named_steps["mlp"]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(mlp.loss_curve_); ax[0].set(title="Training loss", xlabel="Epoch", ylabel="Log-loss")
ax[1].plot(mlp.validation_scores_); ax[1].set(title="Validation accuracy (15% of train)", xlabel="Epoch", ylabel="Accuracy")
plt.tight_layout(); plt.savefig(f"{OUT}/mlp_learning_curve.png", dpi=200); plt.close()

fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(4.5, 4.5)); plt.plot(fpr, tpr, label=f"MLP (AUC={r_best['ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve - MLP"); plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/mlp_roc.png", dpi=200); plt.close()
np.save(f"{OUT}/mlp_roc_points.npy", np.vstack([fpr, tpr]))  # TV5/TV4 can overlay all 4 models

ConfusionMatrixDisplay(cm, display_labels=["Low", "High"]).plot(cmap="Blues", values_format="d")
plt.title("Confusion matrix - MLP"); plt.tight_layout(); plt.savefig(f"{OUT}/mlp_confusion.png", dpi=200); plt.close()

# 8. Save model for TV4 (SHAP) and a summary
joblib.dump(best, f"{OUT}/mlp.pkl")
json.dump(dict(best_params={k: str(v) for k, v in gs.best_params_.items()}, cv_auc=float(gs.best_score_),
               baseline=r_base, tuned=r_best, confusion_TN_FP_FN_TP=cm.ravel().tolist(),
               seeds_auc_mean=float(np.mean(aucs)), seeds_auc_std=float(np.std(aucs)),
               seeds_f1_mean=float(np.mean(f1s)), seeds_f1_std=float(np.std(f1s)),
               epochs=int(mlp.n_iter_), n_params=int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)),
               n_train=len(X_train), n_test=len(X_test)),
          open(f"{OUT}/mlp_summary.json", "w"), indent=2, default=float)
print("\nSaved to", OUT)
