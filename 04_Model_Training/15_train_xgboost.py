import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

os.makedirs("models/xgboost", exist_ok=True)
os.makedirs("results", exist_ok=True)

meta_cols = ["essay_id", "text", "label", "model_source"]
models_list = ["gpt4o", "claude", "gemini", "llama", "qwen"]
results = []
all_importances = []

le = LabelEncoder()

for i in range(1, 6):
    print(f"Training fold {i}/5...")
    train = pd.read_csv(f"data/folds/fold{i}_train.csv")
    test  = pd.read_csv(f"data/folds/fold{i}_test.csv")

    feature_cols = [c for c in train.columns if c not in meta_cols]

    X_train = train[feature_cols].fillna(0)
    y_train = le.fit_transform(train["label"])
    X_test  = test[feature_cols].fillna(0)
    y_test  = le.transform(test["label"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = XGBClassifier(
        learning_rate=0.1, max_depth=6, n_estimators=100,
        random_state=42, eval_metric="logloss", n_jobs=-1
    )
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
    })

    imp = pd.Series(clf.feature_importances_, index=feature_cols)
    imp.name = f"fold{i}"
    all_importances.append(imp)

    joblib.dump(clf, f"models/xgboost/fold{i}_model.pkl")
    print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | FPR: {fpr:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/xgboost_results.csv", index=False)

imp_df = pd.DataFrame(all_importances).T
imp_df["mean_importance"] = imp_df.mean(axis=1)
imp_df = imp_df.sort_values("mean_importance", ascending=False)
imp_df.to_csv("results/xgboost_feature_importance.csv")

print("\nMean results:")
print(results_df[["accuracy","f1","auc","fpr"]].mean().round(4))
print("\nTop 10 features:")
print(imp_df["mean_importance"].head(10))
print("\nSaved to results/xgboost_results.csv and results/xgboost_feature_importance.csv")