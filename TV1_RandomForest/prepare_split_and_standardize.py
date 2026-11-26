"""STAT3013 – load the shared dataset, use the fixed train/test split, and scale safely.

Everyone on the team uses this file so all models share the same data, target and test set.
- Data: STAT3013_final_model_data.csv (2,850 hospitals, no missing values)
- Target (classification): READM_HF_High = 1 if READM_30_HF > 21.3 (median of TRAIN only)
- Target (regression): READM_30_HF (heart failure 30-day readmission rate, %)
- Split: column "Split" (train 2,280 / test 570, random_state=42). Do not re-split.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

DATA_FILE = "STAT3013_final_model_data.csv"
FEATURES = [
    "Nurse_Communication", "Doctor_Communication", "Medication_Communication",
    "Discharge_Information", "Cleanliness", "Quietness",
    "Overall_Hospital_Rating", "Recommend_Hospital",
]
TARGET_CLASS = "READM_HF_High"
TARGET_RATE = "READM_30_HF"


def load_data():
    # Facility ID must be read as text, otherwise leading zeros are lost (010001 -> 10001)
    df = pd.read_csv(DATA_FILE, dtype={"Facility ID": str})
    train, test = df[df["Split"] == "train"], df[df["Split"] == "test"]
    return train[FEATURES], test[FEATURES], train[TARGET_CLASS], test[TARGET_CLASS]


def scale(X_train, X_test):
    """For KNN and MLP. Fit the scaler on TRAIN only. RF/XGBoost do not need scaling."""
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def evaluate(model, X_test, y_test, name="model"):
    """Shared evaluation – every classifier reports the same metrics on the same test set."""
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    print(pd.Series(result).to_string())
    print("Confusion matrix [[TN, FP], [FN, TP]]:\n", confusion_matrix(y_test, pred))
    return result


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train {len(X_train)} | Test {len(X_test)} | High share in train {y_train.mean():.3f}")

    # Baseline example – replace with each member's model
    rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train)
    evaluate(rf, X_test, y_test, "Random Forest (baseline)")

    X_train_s, X_test_s = scale(X_train, X_test)  # use these for KNN and MLP
