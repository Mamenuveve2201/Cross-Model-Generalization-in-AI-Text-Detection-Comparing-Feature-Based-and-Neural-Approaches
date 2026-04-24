import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

os.makedirs("models/logistic", exist_ok=True)
os.makedirs("results", exist_ok=True)

meta_cols = ["essay_id", "text", "label", "model_source"]
models_list = ["gpt4o", "claude", "gemini", "llama", "qwen"]
results = []

for i in range(1, 6):
    print(f"Training fold {i}/5...")
    train = pd.read_csv(f"data/folds/fold{i}_train.csv")
    test  = pd.read_csv(f"data/folds/fold{i}_test.csv")

    feature_cols = [c for c in train.columns if c not in meta_cols]

    X_train = train[feature_cols].fillna(0)
    y_train = train["label"]
    X_test  = test[feature_cols].fillna(0)
    y_test  = test["label"]

    # Fit scaler on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="binary", pos_label=1)
    auc = roc_auc_score(y_test, y_prob)
    fpr = sum((y_pred == 1) & (y_test == 0)) / max(sum(y_test == 0), 1)

    results.append({
        "fold": i,
        "test_model": models_list[i-1],
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "fpr": round(fpr, 4),
        "train_size": len(train),
        "test_size": len(test),
    })

    joblib.dump(clf, f"models/logistic/fold{i}_model.pkl")
    print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | FPR: {fpr:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/logistic_results.csv", index=False)
print("\nMean results:")
print(results_df[["accuracy","f1","auc","fpr"]].mean().round(4))
print("\nSaved to results/logistic_results.csv")