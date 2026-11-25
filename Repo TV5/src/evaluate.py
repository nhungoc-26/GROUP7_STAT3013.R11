"""STAT3013 – Hàm dùng chung cho 4 model phân loại (RF, XGBoost, MLP, KNN).

Mọi thành viên import từ file này để dùng CÙNG dữ liệu, CÙNG split và CÙNG bộ chỉ số:
    from evaluate import load_data, scale, evaluate, save_roc_points

- Dữ liệu: data/processed.csv (2.850 bệnh viện, không thiếu giá trị)
- Target phân loại: READM_HF_High = 1 nếu READM_30_HF > 21.3 (21.3 = trung vị của TRAIN, không rò rỉ)
- Target hồi quy:  READM_30_HF (tỷ lệ tái nhập viện 30 ngày do suy tim, %)
- Split: cột "Split" (train 2.280 / test 570, random_state=42). KHÔNG chia lại.
"""
import os

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed.csv")

FEATURES = [
    "Nurse_Communication",       # H_COMP_1_LINEAR_SCORE – giao tiếp của điều dưỡng
    "Doctor_Communication",      # H_COMP_2_LINEAR_SCORE – giao tiếp của bác sĩ
    "Medication_Communication",  # H_COMP_5_LINEAR_SCORE – giao tiếp về thuốc
    "Discharge_Information",     # H_COMP_6_LINEAR_SCORE – thông tin khi xuất viện
    "Cleanliness",               # H_CLEAN_LINEAR_SCORE  – mức độ sạch sẽ
    "Quietness",                 # H_QUIET_LINEAR_SCORE  – mức độ yên tĩnh
    "Overall_Hospital_Rating",   # H_HSP_RATING_LINEAR_SCORE – đánh giá tổng thể
    "Recommend_Hospital",        # H_RECMND_LINEAR_SCORE – sẵn lòng giới thiệu
]
TARGET_CLASS = "READM_HF_High"
TARGET_RATE = "READM_30_HF"


def load_data(target=TARGET_CLASS, path=DATA_FILE):
    """Trả về X_train, X_test, y_train, y_test theo split cố định.

    target=TARGET_CLASS cho phân loại, target=TARGET_RATE cho hồi quy.
    Facility ID phải đọc dạng text, nếu không sẽ mất số 0 ở đầu (010001 -> 10001).
    """
    df = pd.read_csv(path, dtype={"Facility ID": str})
    train, test = df[df["Split"] == "train"], df[df["Split"] == "test"]
    return train[FEATURES], test[FEATURES], train[target], test[target]


def scale(X_train, X_test):
    """Chuẩn hóa cho KNN và MLP. Fit scaler trên TRAIN, chỉ transform test.
    RF và XGBoost không cần chuẩn hóa.
    Nếu tuning bằng cross-validation, nên đặt StandardScaler trong Pipeline thay vì dùng hàm này,
    để scaler được fit lại trong từng fold.
    """
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def evaluate(model, X_test, y_test, name="model", verbose=True):
    """Đánh giá 1 model phân loại trên test set với bộ chỉ số chung.

    Model cần có predict() và predict_proba() (sklearn, XGBClassifier, Keras wrapper...).
    Trả về dict: Model, Accuracy, Precision, Recall, F1, ROC-AUC, TN, FP, FN, TP.
    """
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),       # = sensitivity
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if verbose:
        print(pd.Series(result).to_string())
        print("Confusion matrix [[TN, FP], [FN, TP]]:", [[int(tn), int(fp)], [int(fn), int(tp)]])
    return result


def save_roc_points(model, X_test, y_test, path):
    """Lưu tọa độ ROC (fpr, tpr) ra CSV để TV4 vẽ ROC chung của 4 model trong 1 hình."""
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(path, index=False)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train {len(X_train)} | Test {len(X_test)} | "
          f"High train {y_train.mean():.3f} | High test {y_test.mean():.3f}")
