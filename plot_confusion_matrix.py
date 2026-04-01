"""Generate publication-ready confusion matrix heatmap + feature importance bar chart."""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score

DATA = "/Users/wu.w4/Desktop/User-log/batch_output_k5/all_windows_with_clusters.csv"
OUT  = "/Users/wu.w4/Desktop/User-log/batch_output_k5/"
FEATURES = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time",
    "puzzle_active", "error_count", "time_since_action",
]
FEAT_LABELS = [
    "Gaze Entropy", "Clue Ratio", "Switch Rate",
    "Action Count", "Idle Time",
    "Puzzle Active", "Error Count", "Time Since Action",
]

df = pd.read_csv(DATA)
X_raw = np.nan_to_num(df[FEATURES].values, nan=0.0, posinf=0.0, neginf=0.0)
players = df["player_id"].values
unique_players = sorted(df["player_id"].unique())

y = df["cluster_id"].map({0:2, 1:1, 2:0, 3:2, 4:1}).astype(int).values
names = ["Productive", "Confused", "Transition"]

# ── LOPO-CV for RF ────────────────────────────────────────────────────
all_true, all_pred = [], []
all_imps = []

for held_out in unique_players:
    mask = players == held_out
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[~mask])
    X_te = scaler.transform(X_raw[mask])

    model = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1)
    model.fit(X_tr, y[~mask])

    all_true.extend(y[mask])
    all_pred.extend(model.predict(X_te))
    all_imps.append(model.feature_importances_)

# ── Figure 1: Normalized confusion matrix ─────────────────────────────
cm = confusion_matrix(all_true, all_pred)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

for i in range(3):
    for j in range(3):
        pct = cm_norm[i, j]
        cnt = cm[i, j]
        color = "white" if pct > 0.6 else "black"
        ax.text(j, i, f"{pct:.1%}\n({cnt})", ha="center", va="center",
                fontsize=11, color=color, fontweight="bold" if i == j else "normal")

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(names, fontsize=11)
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
ax.set_ylabel("Actual", fontsize=12, labelpad=8)
ax.set_title("3-Class LOPO-CV Confusion Matrix (RF)", fontsize=13, pad=10)
fig.colorbar(im, ax=ax, shrink=0.8, label="Recall")
fig.tight_layout()
fig.savefig(OUT + "12_confusion_matrix.png", dpi=300, bbox_inches="tight")
print(f"Saved {OUT}12_confusion_matrix.png")

# ── Figure 2: Feature importance ──────────────────────────────────────
imps = np.array(all_imps)
means = imps.mean(axis=0)
stds  = imps.std(axis=0)
order = np.argsort(means)  # ascending for horizontal bar

fig2, ax2 = plt.subplots(figsize=(5, 3.5))
y_pos = np.arange(len(FEATURES))
ax2.barh(y_pos, means[order], xerr=stds[order], color="#5B8DB8",
         edgecolor="white", capsize=3)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([FEAT_LABELS[i] for i in order], fontsize=10)
ax2.set_xlabel("Mean Importance (Gini)", fontsize=11)
ax2.set_title("RF Feature Importance (3-Class, LOPO-CV)", fontsize=12, pad=10)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()
fig2.savefig(OUT + "13_feature_importance.png", dpi=300, bbox_inches="tight")
print(f"Saved {OUT}13_feature_importance.png")
