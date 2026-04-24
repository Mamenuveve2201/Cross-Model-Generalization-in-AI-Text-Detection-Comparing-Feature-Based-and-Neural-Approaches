import pandas as pd
import numpy as np
import os

os.makedirs("results/analysis", exist_ok=True)

# For LOMO-CV, every fold IS out-of-domain by definition.
# We estimate in-domain performance as the mean of the OTHER 4 folds
# (i.e. how well the model does when the test model WAS in training)

models_list = ["gpt4o", "claude", "gemini", "llama", "qwen"]
classifiers = {
    "Logistic Regression": "results/logistic_results.csv",
    "Random Forest":       "results/rf_results.csv",
    "XGBoost":             "results/xgboost_results.csv",
    "RoBERTa-OpenAI":      "results/roberta_results.csv",
    "ChatGPT Detector":    "results/chatgpt_detector_results.csv",
}

gaps = []
for clf_name, path in classifiers.items():
    df = pd.read_csv(path)
    for i, test_model in enumerate(models_list):
        out_of_domain = df[df["test_model"] == test_model]["f1"].values[0]
        in_domain = df[df["test_model"] != test_model]["f1"].mean()
        gap = in_domain - out_of_domain
        gaps.append({
            "classifier": clf_name,
            "test_model": test_model,
            "in_domain_f1": round(in_domain, 4),
            "out_of_domain_f1": round(out_of_domain, 4),
            "generalization_gap": round(gap, 4),
        })

gap_df = pd.DataFrame(gaps)
gap_df.to_csv("results/analysis/generalization_gaps.csv", index=False)

print("Generalization Gaps (in-domain F1 - out-of-domain F1):")
print(gap_df.to_string(index=False))
print("\nMean gap per classifier:")
print(gap_df.groupby("classifier")["generalization_gap"].mean().round(4))
print("\nSaved to results/analysis/generalization_gaps.csv")