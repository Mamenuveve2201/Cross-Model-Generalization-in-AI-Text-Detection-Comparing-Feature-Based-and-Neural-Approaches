import pandas as pd
import os

df = pd.read_csv("data/raw/features_all.csv")
os.makedirs("data/folds", exist_ok=True)

models = ["gpt4o", "claude", "gemini", "llama", "qwen"]

# Split human samples into 5 equal chunks
human_df = df[df["model_source"] == "human"].sample(frac=1, random_state=42).reset_index(drop=True)
human_folds = [human_df.iloc[i::5] for i in range(5)]

for i, test_model in enumerate(models, 1):
    test_ai    = df[df["model_source"] == test_model]
    test_human = human_folds[i-1]
    test       = pd.concat([test_ai, test_human])

    train_ai    = df[(df["model_source"] != test_model) & (df["model_source"] != "human")]
    train_human = pd.concat([human_folds[j] for j in range(5) if j != i-1])
    train       = pd.concat([train_ai, train_human])

    train.to_csv(f"data/folds/fold{i}_train.csv", index=False)
    test.to_csv(f"data/folds/fold{i}_test.csv", index=False)

    print(f"Fold {i} — Test: {test_model} ({len(test_ai)} AI + {len(test_human)} human = {len(test)}) | Train: {len(train)}")

print("\nDone. All 5 folds saved to data/folds/")