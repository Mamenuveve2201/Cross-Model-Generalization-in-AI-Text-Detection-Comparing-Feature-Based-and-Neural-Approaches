# save as 11_normalize.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/raw/features_all.csv")
meta_cols = ["essay_id", "text", "label", "model_source"]
feature_cols = [c for c in df.columns if c not in meta_cols]

scaler = StandardScaler()
df_norm = df.copy()
df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
df_norm.to_csv("data/raw/features_normalized.csv", index=False)
print("Done. Shape:", df_norm.shape)