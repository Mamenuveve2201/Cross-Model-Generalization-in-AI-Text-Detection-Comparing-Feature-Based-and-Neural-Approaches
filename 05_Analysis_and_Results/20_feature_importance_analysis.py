import pandas as pd
import numpy as np
import os

os.makedirs("results/analysis", exist_ok=True)

# Load RF and XGBoost feature importances
rf_imp  = pd.read_csv("results/rf_feature_importance.csv", index_col=0)
xgb_imp = pd.read_csv("results/xgboost_feature_importance.csv", index_col=0)

# Average RF and XGBoost importance
combined = pd.DataFrame({
    "rf_importance":  rf_imp["mean_importance"],
    "xgb_importance": xgb_imp["mean_importance"],
})
combined["avg_importance"] = combined.mean(axis=1)
combined = combined.sort_values("avg_importance", ascending=False)
combined.to_csv("results/analysis/combined_feature_importance.csv")

print("Top 20 features (avg RF + XGBoost importance):")
print(combined.head(20)[["rf_importance","xgb_importance","avg_importance"]].round(4))

# Transferable features: consistent importance across all 5 RF folds
fold_cols = [c for c in rf_imp.columns if c.startswith("fold")]
rf_imp["std_across_folds"] = rf_imp[fold_cols].std(axis=1)
rf_imp["mean_across_folds"] = rf_imp[fold_cols].mean(axis=1)
rf_imp["cv"] = rf_imp["std_across_folds"] / rf_imp["mean_across_folds"].replace(0, np.nan)
rf_imp = rf_imp.sort_values("mean_across_folds", ascending=False)

transferable = rf_imp[rf_imp["cv"] < 0.3].head(10)
model_specific = rf_imp[rf_imp["cv"] > 0.7].head(10)

transferable.to_csv("results/analysis/transferable_features.csv")
model_specific.to_csv("results/analysis/model_specific_features.csv")

print("\nMost TRANSFERABLE features (low variance across folds):")
print(transferable[["mean_across_folds","std_across_folds","cv"]].round(4))

print("\nMost MODEL-SPECIFIC features (high variance across folds):")
print(model_specific[["mean_across_folds","std_across_folds","cv"]].round(4))

# Xia et al. key features comparison
xia_features = [
    "passive_voice_freq", "active_voice_freq",
    "past_tense_freq", "present_tense_freq", "future_tense_freq",
    "first_person_freq", "second_person_freq", "third_person_freq"
]
xia_in_data = [f for f in xia_features if f in combined.index]
print("\nXia et al. key features — your importance rankings:")
for f in xia_in_data:
    rank = list(combined.index).index(f) + 1
    imp  = combined.loc[f, "avg_importance"]
    print(f"  {f}: rank #{rank}, importance {imp:.4f}")

print("\nSaved to results/analysis/")