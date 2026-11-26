"""STAT3013 – TV5: K-Nearest Neighbors classifier.

Dùng chung dữ liệu, split cố định và hàm evaluate() từ prepare_split_and_standardize.py.
- Chuẩn hóa: StandardScaler nằm TRONG Pipeline -> mỗi fold CV fit scaler trên phần train của fold đó
  (không rò rỉ thông tin từ fold validation hay test set).
- Chọn K, weights, metric bằng 5-fold Stratified CV lặp 10 lần trên TRAIN, tiêu chí ROC-AUC,
  rồi áp quy tắc 1-SE (K nhỏ nhất có AUC >= max - 1 SE).
- Test set (570 bệnh viện) chỉ dùng 1 lần ở bước đánh giá cuối.

Chạy: python knn.py --data-dir <thư mục chứa STAT3013_final_model_data.csv và prepare_split_and_standardize.py>
"""
import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=".")
parser.add_argument("--out-dir", default="outputs")
args = parser.parse_args()

sys.path.insert(0, os.path.abspath(args.data_dir))
os.chdir(os.path.abspath(args.data_dir))          # DATA_FILE trong module là đường dẫn tương đối
import prepare_split_and_standardize as shared    # noqa: E402

out = os.path.abspath(os.path.join(os.path.dirname(__file__), args.out_dir))
os.makedirs(out, exist_ok=True)

X_train, X_test, y_train, y_test = shared.load_data()
print(f"Train {len(X_train)} | Test {len(X_test)} | High share train {y_train.mean():.3f}")

# ---------- Tuning ----------
pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
param_grid = {
    "knn__n_neighbors": list(range(1, 100, 4)) + list(range(101, 502, 20)),  # K lẻ để tránh hòa phiếu
    "knn__weights": ["uniform", "distance"],
    "knn__p": [1, 2],                               # 1 = Manhattan, 2 = Euclidean
}
# 5-fold lặp 10 lần: đường AUC theo K rất phẳng, 1 lần 5-fold quá nhiễu để chọn K
N_SPLITS, N_REPEATS = 5, 10
cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
search = GridSearchCV(pipe, param_grid, cv=cv, scoring="roc_auc",
                      n_jobs=-1, return_train_score=True, refit=False)
search.fit(X_train, y_train)
print("Max CV ROC-AUC:", search.best_params_, f"= {search.best_score_:.4f}")

cvres = pd.DataFrame(search.cv_results_)
cvres_small = cvres[["param_knn__n_neighbors", "param_knn__weights", "param_knn__p",
                     "mean_train_score", "mean_test_score", "std_test_score", "rank_test_score"]]
cvres_small = cvres_small.rename(columns={
    "param_knn__n_neighbors": "K", "param_knn__weights": "weights", "param_knn__p": "p",
    "mean_train_score": "train_AUC", "mean_test_score": "cv_AUC", "std_test_score": "cv_AUC_std",
    "rank_test_score": "rank"}).sort_values("rank")
cvres_small.to_csv(os.path.join(out, "knn_cv_results.csv"), index=False)

# Quy tắc 1-SE: đường AUC theo K rất phẳng, nên chọn K NHỎ NHẤT có AUC >= max - 1 SE
# (SE = std giữa 50 fold / sqrt(50)). Hòa thì ưu tiên weights="uniform", Euclidean (p=2) – đơn giản hơn,
# và weights="distance" cho train AUC = 1.0 (thuộc lòng train).
top = cvres_small.iloc[0]
se = top["cv_AUC_std"] / np.sqrt(N_SPLITS * N_REPEATS)
threshold = top["cv_AUC"] - se
cand = cvres_small[cvres_small["cv_AUC"] >= threshold].copy()
cand["w_rank"] = (cand["weights"] != "uniform").astype(int)
cand["p_rank"] = (cand["p"] != 2).astype(int)
chosen = cand.sort_values(["K", "w_rank", "p_rank"]).iloc[0]
best = {"knn__n_neighbors": int(chosen["K"]), "knn__weights": chosen["weights"], "knn__p": int(chosen["p"])}
print(f"1-SE rule: max AUC {top['cv_AUC']:.4f}, SE {se:.4f}, threshold {threshold:.4f}")
print("Chosen params:", best, f"| CV ROC-AUC = {chosen['cv_AUC']:.4f} ± {chosen['cv_AUC_std']:.4f}")

# Baseline: K=5 mặc định (để cho thấy tuning có tác dụng)
baseline = Pipeline([("scaler", StandardScaler()),
                     ("knn", KNeighborsClassifier(n_neighbors=5))]).fit(X_train, y_train)

# ---------- Đánh giá trên test (dùng hàm chung) ----------
model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
model.set_params(**best).fit(X_train, y_train)
max_auc_model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
max_auc_model.set_params(**search.best_params_).fit(X_train, y_train)
print("\n== Baseline K=5 ==")
res_base = shared.evaluate(baseline, X_test, y_test, "KNN (K=5, default)")
print("\n== Tuned KNN ==")
res = shared.evaluate(model, X_test, y_test, "KNN (tuned, 1-SE)")
res["best_params"] = {k.replace("knn__", ""): v for k, v in best.items()}
res["cv_roc_auc"] = float(chosen["cv_AUC"])
res["cv_roc_auc_std"] = float(chosen["cv_AUC_std"])
print("\n== Max-CV-AUC KNN (tham khảo) ==")
res_max = shared.evaluate(max_auc_model, X_test, y_test,
                          f"KNN (max CV AUC, K={search.best_params_['knn__n_neighbors']})")

with open(os.path.join(out, "knn_results.json"), "w", encoding="utf-8") as f:
    json.dump({"tuned_1se": res, "baseline_k5": res_base, "max_cv_auc": res_max,
               "one_se": {"max_cv_auc": float(top["cv_AUC"]), "se": float(se),
                          "threshold": float(threshold)}}, f, indent=2, default=str)
pd.DataFrame([res_base, res, res_max]).drop(columns=["best_params", "cv_roc_auc", "cv_roc_auc_std"],
             errors="ignore").round(4).to_csv(os.path.join(out, "knn_test_metrics.csv"), index=False)
with open(os.path.join(out, "knn.pkl"), "wb") as f:
    pickle.dump(model, f)

# ---------- Hình ----------
w_best, p_best = best["knn__weights"], best["knn__p"]
plt.figure(figsize=(7, 4.2))
for (w, p), g in cvres_small.groupby(["weights", "p"]):
    g = g.sort_values("K")
    name = f"{w}, {'Manhattan' if p == 1 else 'Euclidean'}"
    lw = 2.4 if (w == w_best and p == p_best) else 1.1
    plt.plot(g["K"], g["cv_AUC"], label=name, lw=lw)
plt.axvline(best["knn__n_neighbors"], color="grey", ls="--", lw=1, label=f"K chọn = {best['knn__n_neighbors']} (1-SE)")
plt.axhline(threshold, color="red", ls=":", lw=1, label="max − 1 SE")
plt.xlabel("K (số láng giềng)")
plt.ylabel("ROC-AUC trung bình (5-fold × 10)")
plt.title("Chọn K cho KNN bằng cross-validation (train set)")
plt.legend(fontsize=8)
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(os.path.join(out, "knn_cv_k_curve.png"), dpi=200)

fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Low", "High"],
                                      cmap="Blues", ax=ax, colorbar=False)
ax.set_title("KNN – Confusion matrix (test)")
plt.tight_layout()
plt.savefig(os.path.join(out, "knn_confusion_matrix.png"), dpi=200)

fig, ax = plt.subplots(figsize=(4.8, 4.5))
RocCurveDisplay.from_estimator(model, X_test, y_test, name=f"KNN (K={best['knn__n_neighbors']})", ax=ax)
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_title("KNN – ROC curve (test)")
plt.tight_layout()
plt.savefig(os.path.join(out, "knn_roc_curve.png"), dpi=200)

# Lưu FPR/TPR để TV4 vẽ ROC chung 4 model
from sklearn.metrics import roc_curve  # noqa: E402
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(out, "knn_roc_points.csv"), index=False)
print("\nSaved to", out)
