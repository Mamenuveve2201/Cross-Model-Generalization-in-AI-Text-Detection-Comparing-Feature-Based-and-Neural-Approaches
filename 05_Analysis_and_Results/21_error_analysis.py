import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

os.makedirs("results/analysis", exist_ok=True)

meta_cols = ["essay_id", "text", "label", "model_source"]
models_list = ["gpt4o", "claude", "gemini", "llama", "qwen"]
classifiers = ["logistic", "random_forest", "xgboost"]
clf_names   = ["Logistic Regression", "Random Forest", "XGBoost"]

all_wrong = []
confusion_data = []

for i, test_model in enumerate(models_list, 1):
    test = pd.read_csv(f"data/folds/fold{i}_test.csv")
    train = pd.read_csv(f"data/folds/fold{i}_train.csv")
    feature_cols = [c for c in test.columns if c not in meta_cols]

    if test["label"].dtype == object:
        y_test = (test["label"] == "ai").astype(int)
    else:
        y_test = test["label"]

    X_test = test[feature_cols].fillna(0)
    X_train = train[feature_cols].fillna(0)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)

    wrong_mask = pd.Series([True] * len(test))

    for clf_dir, clf_name in zip(classifiers, clf_names):
        clf = joblib.load(f"models/{clf_dir}/fold{i}_model.pkl")
        y_pred = clf.predict(X_test_scaled)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_data.append({
            "classifier": clf_name,
            "test_model": test_model,
            "TN": cm[0,0], "FP": cm[0,1],
            "FN": cm[1,0], "TP": cm[1,1],
            "precision": round(cm[1,1]/(cm[1,1]+cm[0,1]+1e-9), 4),
            "recall":    round(cm[1,1]/(cm[1,1]+cm[1,0]+1e-9), 4),
        })

        wrong_mask = wrong_mask & (pd.Series(y_pred) != y_test.values)

    # Samples ALL 3 classical models get wrong
    wrong_samples = test[wrong_mask.values].copy()
    wrong_samples["fold"] = i
    wrong_samples["test_model"] = test_model
    all_wrong.append(wrong_samples)

# Save confusion matrices
conf_df = pd.DataFrame(confusion_data)
conf_df.to_csv("results/analysis/confusion_matrices.csv", index=False)
print("Confusion matrices:")
print(conf_df.to_string(index=False))

# Save hard samples
hard_df = pd.concat(all_wrong, ignore_index=True)
hard_df = hard_df[["fold","test_model","label","model_source","text"]].head(50)
hard_df.to_csv("results/analysis/hard_samples.csv", index=False)

print(f"\nSamples ALL models get wrong: {len(hard_df)}")
print("\nBreakdown by test model:")
print(hard_df.groupby(["test_model","label"]).size())
print("\nSaved to results/analysis/")