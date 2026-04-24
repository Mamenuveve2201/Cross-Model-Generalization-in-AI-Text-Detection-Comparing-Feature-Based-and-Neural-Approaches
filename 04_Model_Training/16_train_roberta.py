import pandas as pd
import numpy as np
import os
from transformers import pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

models_list = ["gpt4o", "claude", "gemini", "llama", "qwen"]
results = []

# Load RoBERTa detector
print("Loading RoBERTa detector...")
detector = pipeline("text-classification", model="roberta-base-openai-detector", truncation=True, max_length=512)
print("Model loaded.\n")

def get_prediction(text):
    try:
        result = detector(text[:1000])[0]
        # Label is either 'LABEL_0' (human) or 'LABEL_1' (AI)
        # or 'Real' / 'Fake' depending on model version
        label = result["label"].lower()
        score = result["score"]
        if label in ["fake", "label_1"]:
            return 1, score       # AI
        else:
            return 0, 1 - score   # Human
    except Exception as e:
        return 0, 0.5

for i in range(1, 6):
    print(f"Evaluating fold {i}/5 ({models_list[i-1]})...")
    test = pd.read_csv(f"data/folds/fold{i}_test.csv")

    # Convert label to binary if needed
    if test["label"].dtype == object:
        test["label_bin"] = (test["label"] == "ai").astype(int)
    else:
        test["label_bin"] = test["label"]

    texts = test["text"].fillna("").tolist()
    y_test = test["label_bin"].tolist()

    preds, probs = [], []
    for text in tqdm(texts, desc=f"  Fold {i}"):
        pred, prob = get_prediction(text)
        preds.append(pred)
        probs.append(prob)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average="binary", pos_label=1)
    auc = roc_auc_score(y_test, probs)
    fpr = sum((np.array(preds) == 1) & (np.array(y_test) == 0)) / max(sum(np.array(y_test) == 0), 1)

    results.append({
        "fold": i,
        "test_model": models_list[i-1],
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "fpr": round(fpr, 4),
        "test_size": len(test),
    })

    print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | FPR: {fpr:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("results/roberta_results.csv", index=False)
print("\nMean results:")
print(results_df[["accuracy","f1","auc","fpr"]].mean().round(4))
print("\nSaved to results/roberta_results.csv")