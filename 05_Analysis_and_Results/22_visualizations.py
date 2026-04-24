import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

os.makedirs("results/visualizations", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

# ── DATA ──────────────────────────────────────────────────────────────
all_results = pd.read_csv("results/analysis/all_results_combined.csv")
gap_df      = pd.read_csv("results/analysis/generalization_gaps.csv")
feat_df     = pd.read_csv("results/analysis/combined_feature_importance.csv", index_col=0)
conf_df     = pd.read_csv("results/analysis/confusion_matrices.csv")

models_list  = ["gpt4o", "claude", "gemini", "llama", "qwen"]
clf_order    = ["Logistic Regression", "Random Forest", "XGBoost",
                "RoBERTa-OpenAI", "ChatGPT Detector"]
colors       = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#e67e22"]

# ── 1. PERFORMANCE COMPARISON BAR CHART ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

metrics = ["f1", "accuracy"]
titles  = ["Mean F1 Score", "Mean Accuracy"]

for ax, metric, title in zip(axes, metrics, titles):
    means = [all_results[all_results["model"] == clf][metric].mean() for clf in clf_order]
    stds  = [all_results[all_results["model"] == clf][metric].std()  for clf in clf_order]
    bars  = ax.bar(clf_order, means, color=colors, yerr=stds, capsize=5,
                   edgecolor="white", linewidth=1.2)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticklabels(clf_order, rotation=20, ha="right", fontsize=10)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random chance")
    ax.legend(fontsize=9)

plt.suptitle("AI Text Detection: Performance Comparison Across All Classifiers",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/visualizations/01_performance_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 01_performance_comparison.png")

# ── 2. GENERALIZATION GAP HEATMAP ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, metric, title in zip(axes, ["out_of_domain_f1", "generalization_gap"],
                                    ["Out-of-Domain F1", "Generalization Gap\n(in-domain − out-of-domain F1)"]):
    pivot = gap_df.pivot(index="classifier", columns="test_model", values=metric)
    pivot = pivot.reindex(clf_order).reindex(columns=models_list)

    cmap = "RdYlGn" if metric == "out_of_domain_f1" else "RdYlGn_r"
    sns.heatmap(pivot, ax=ax, annot=True, fmt=".3f", cmap=cmap,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Held-Out Test Model", fontsize=11)
    ax.set_ylabel("Classifier", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.suptitle("Cross-Model Generalization Analysis (LOMO-CV)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/visualizations/02_generalization_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 02_generalization_heatmap.png")

# ── 3. FEATURE IMPORTANCE BAR CHART ───────────────────────────────────
top20 = feat_df.head(20)

fig, ax = plt.subplots(figsize=(13, 8))
x = range(len(top20))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], top20["rf_importance"],
               width, label="Random Forest", color="#3498db", alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], top20["xgb_importance"],
               width, label="XGBoost", color="#9b59b6", alpha=0.85)

ax.set_xticks(list(x))
ax.set_xticklabels(top20.index, rotation=45, ha="right", fontsize=10)
ax.set_ylabel("Feature Importance", fontsize=12)
ax.set_title("Top 20 Features by Importance (Random Forest vs XGBoost)",
             fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=11)

# Highlight Xia et al. features
xia_features = ["passive_voice_freq","active_voice_freq","past_tense_freq",
                "present_tense_freq","future_tense_freq","first_person_freq",
                "second_person_freq","third_person_freq"]
for tick, label in zip(ax.get_xticklabels(), top20.index):
    if label in xia_features:
        tick.set_color("red")
        tick.set_fontweight("bold")

red_patch = mpatches.Patch(color="red", label="Xia et al. key features")
ax.legend(handles=[mpatches.Patch(color="#3498db", label="Random Forest"),
                   mpatches.Patch(color="#9b59b6", label="XGBoost"),
                   red_patch], fontsize=10)

plt.tight_layout()
plt.savefig("results/visualizations/03_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 03_feature_importance.png")

# ── 4. CONFUSION MATRIX HEATMAPS ──────────────────────────────────────
for clf_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    clf_data = conf_df[conf_df["classifier"] == clf_name]

    for ax, model in zip(axes, models_list):
        row = clf_data[clf_data["test_model"] == model].iloc[0]
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, linewidths=0.5, linecolor="white",
                    xticklabels=["Human", "AI"],
                    yticklabels=["Human", "AI"])
        ax.set_title(f"{model}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    plt.suptitle(f"Confusion Matrices — {clf_name}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = clf_name.lower().replace(" ", "_")
    plt.savefig(f"results/visualizations/04_confusion_{fname}.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: 04_confusion_{fname}.png")

# ── 5. PER-MODEL F1 COMPARISON ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
x = np.arange(len(models_list))
width = 0.15

for idx, (clf, color) in enumerate(zip(clf_order, colors)):
    vals = [all_results[(all_results["model"] == clf) &
                        (all_results["test_model"] == m)]["f1"].values[0]
            for m in models_list]
    ax.bar(x + idx * width, vals, width, label=clf, color=color, alpha=0.85)

ax.set_xticks(x + width * 2)
ax.set_xticklabels(models_list, fontsize=12)
ax.set_ylabel("F1 Score", fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title("F1 Score per Held-Out Model × Classifier",
             fontsize=14, fontweight="bold", pad=12)
ax.legend(fontsize=10, loc="lower right")
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="Random chance")

plt.tight_layout()
plt.savefig("results/visualizations/05_f1_per_model.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 05_f1_per_model.png")

# ── 6. TRANSFERABLE vs MODEL-SPECIFIC FEATURES ────────────────────────
rf_imp = pd.read_csv("results/rf_feature_importance.csv", index_col=0)
fold_cols = [c for c in rf_imp.columns if c.startswith("fold")]
rf_imp["std_across_folds"]  = rf_imp[fold_cols].std(axis=1)
rf_imp["mean_across_folds"] = rf_imp[fold_cols].mean(axis=1)
rf_imp["cv"] = rf_imp["std_across_folds"] / rf_imp["mean_across_folds"].replace(0, np.nan)
rf_imp = rf_imp.dropna(subset=["cv"]).sort_values("mean_across_folds", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(13, 7))
bar_colors = ["#2ecc71" if cv < 0.3 else "#e74c3c" if cv > 0.7 else "#f39c12"
              for cv in rf_imp["cv"]]
ax.barh(rf_imp.index[::-1], rf_imp["mean_across_folds"][::-1],
        color=bar_colors[::-1], edgecolor="white")
ax.set_xlabel("Mean Importance Across Folds", fontsize=12)
ax.set_title("Feature Transferability (Random Forest)\nGreen=Transferable | Orange=Mixed | Red=Model-Specific",
             fontsize=13, fontweight="bold", pad=12)

green  = mpatches.Patch(color="#2ecc71", label="Transferable (CV < 0.3)")
orange = mpatches.Patch(color="#f39c12", label="Mixed (0.3 ≤ CV ≤ 0.7)")
red    = mpatches.Patch(color="#e74c3c", label="Model-Specific (CV > 0.7)")
ax.legend(handles=[green, orange, red], fontsize=10)

plt.tight_layout()
plt.savefig("results/visualizations/06_feature_transferability.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: 06_feature_transferability.png")

print("\n✅ All visualizations saved to results/visualizations/")