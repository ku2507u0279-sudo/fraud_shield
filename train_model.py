import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, average_precision_score

DATA_PATH = os.getenv("FRAUD_DATA_PATH", "credit_card_fraud_custom.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "fraud_model.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "threshold.json")
TARGET = "IsFraud"

NUMERIC = ["Amount", "MerchantID", "Hour", "DayOfWeek", "Month"]
CATEGORICAL = ["TransactionType", "Location"]
FEATURES = NUMERIC + CATEGORICAL

def make_features(data):
    x = data.copy()
    if "TransactionDate" in x.columns:
        dt = pd.to_datetime(x["TransactionDate"], errors="coerce")
        x["Hour"] = dt.dt.hour
        x["DayOfWeek"] = dt.dt.dayofweek
        x["Month"] = dt.dt.month
    for c in ["Amount", "MerchantID", "Hour", "DayOfWeek", "Month"]:
        if c not in x.columns:
            x[c] = np.nan
    for c in ["TransactionType", "Location"]:
        if c not in x.columns:
            x[c] = "Unknown"
        x[c] = x[c].fillna("Unknown").astype(str)
    return x[FEATURES]

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Put the CSV beside this script.")
    raw = pd.read_csv(DATA_PATH)
    if TARGET not in raw.columns:
        raise ValueError(f"Expected target column '{TARGET}'.")
    raw = raw.dropna(subset=[TARGET]).copy()
    y = raw[TARGET].astype(int)
    if y.nunique() < 2:
        raise ValueError("The target must contain both legitimate (0) and fraud (1) transactions.")

    X = make_features(raw)
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_hold, y_hold, test_size=0.50, random_state=42, stratify=y_hold
    )

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])
    model = Pipeline([
        ("prep", prep),
        ("classifier", ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)

    # Choose a threshold on validation data to favor precision, while requiring
    # some minimum recall and at least a few predicted positives where possible.
    p_valid = model.predict_proba(X_valid)[:, 1]
    candidates = np.unique(np.r_[np.arange(0.10, 0.96, 0.01), p_valid])
    valid_rows = []
    for threshold in candidates:
        pred = (p_valid >= threshold).astype(int)
        positives = int(pred.sum())
        precision = precision_score(y_valid, pred, zero_division=0)
        recall = recall_score(y_valid, pred, zero_division=0)
        f1 = f1_score(y_valid, pred, zero_division=0)
        if positives >= 3 and recall >= 0.20:
            valid_rows.append((precision, f1, threshold, recall, positives))
    if valid_rows:
        # Precision first; F1 and then lower threshold break ties.
        best = max(valid_rows, key=lambda r: (r[0], r[1], -r[2]))
        threshold = float(best[2])
    else:
        # Fallback when the validation sample is too small/sparse.
        threshold = 0.50
        best = (precision_score(y_valid, p_valid >= threshold, zero_division=0),
                f1_score(y_valid, p_valid >= threshold, zero_division=0),
                threshold,
                recall_score(y_valid, p_valid >= threshold, zero_division=0),
                int((p_valid >= threshold).sum()))

    p_test = model.predict_proba(X_test)[:, 1]
    pred_test = (p_test >= threshold).astype(int)
    metrics = {
        "threshold": threshold,
        "validation_precision": float(best[0]),
        "validation_f1": float(best[1]),
        "validation_recall": float(best[3]),
        "test_precision": float(precision_score(y_test, pred_test, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred_test, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred_test, zero_division=0)),
        "test_average_precision": float(average_precision_score(y_test, p_test)),
        "test_confusion_matrix": confusion_matrix(y_test, pred_test).tolist(),
        "test_rows": int(len(y_test)),
        "fraud_rate": float(y.mean()),
        "features": FEATURES
    }
    joblib.dump(model, MODEL_PATH, compress=3)
    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nEvaluation on held-out test data (not used for threshold tuning):")
    print(classification_report(y_test, pred_test, target_names=["Legitimate", "Fraud"], zero_division=0))
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"\nSaved {MODEL_PATH} and {THRESHOLD_PATH}")

if __name__ == "__main__":
    main()
