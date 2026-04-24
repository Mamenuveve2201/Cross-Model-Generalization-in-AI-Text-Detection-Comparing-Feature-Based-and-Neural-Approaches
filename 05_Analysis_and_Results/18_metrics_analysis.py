import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
import os

os.makedirs("results/analysis", exist_ok=True)

# Load all results
lr  = pd.read_csv("results/logistic_results.csv")
rf  = pd.read_csv("results/rf_results.csv")
xgb = pd.read_csv("results/xgboost_results.csv")
rob = pd.read_csv("results/roberta_results.csv")
cgpt = pd.read_csv("results/chatgpt_detector_results.csv")

lr["model"]  = "Logistic Regression"
rf["model"]  = "Random Forest"
xgb["model"] = "XGBoost"
rob["model"] = "RoBERTa-OpenAI"
cgpt["model"] = "ChatGPT Detector"

all_results = pd.concat([lr, rf, xgb, rob, cgpt], ignore_index=True)
all_results.to_csv("results/analysis/all_results_combined.csv", index=False)

# Summary per classifier
summary = all_results.groupby("model")[["accuracy","f1","auc","fpr"]].agg(["mean","std"]).round(4)
summary.to_csv("results/analysis/summary_per_classifier.csv")

# Summary per test model (fold)
fold_summary = all_results.groupby(["model","test_model"])[["accuracy","f1","auc","fpr"]].mean().round(4)
fold_summary.to_csv("results/analysis/summary_per_fold.csv")

print("All results combined:")
print(all_results[["model","test_model","accuracy","f1","auc","fpr"]].to_string(index=False))
print("\nMean per classifier:")
print(all_results.groupby("model")[["accuracy","f1","auc","fpr"]].mean().round(4))
print("\nSaved to results/analysis/")