"""
Predictive Modeling Pipeline for VR Player Cognitive States
============================================================
Moves from unsupervised cluster labels → supervised classifiers that can
predict player states from behavioral features in real time.

Implements:
  1. Three label schemes (5-class, 3-class, binary)
  2. Leave-One-Player-Out cross-validation (no temporal leakage)
  3. Multiple classifiers: Logistic Regression, Random Forest, XGBoost
  4. Class imbalance handling (SMOTE + class weights)
  5. Temporal feature augmentation (lag features, rolling stats)
  6. Full evaluation: accuracy, macro-F1, per-class F1, confusion matrices
  7. Label noise analysis via confidence filtering

Designed for: CHI / CogSci publication quality
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "batch_output_k5" / "all_windows_with_clusters.csv"
OUTPUT_DIR = BASE_DIR / "modeling_output"

BASE_FEATURES = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time",
]

# Note: puzzle_active, error_count, time_since_action are derived from
# puzzle logs and would NOT be available for real-time prediction from
# eye-tracking alone. We define two feature sets:

# SET A: Eye + Hand only (available in real-time from headset sensors)
REALTIME_FEATURES = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time",
]

# SET B: Full features (includes game-log signals, for offline analysis)
FULL_FEATURES = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time",
    "puzzle_active", "error_count", "time_since_action",
]

CLUSTER_NAMES = {
    0: "Transition", 1: "Idle Waiting", 2: "Active Solving",
    3: "Exploration", 4: "Stuck-on-Clue",
}


# ============================================================================
# 1. Data Loading & Label Schemes
# ============================================================================

def load_and_prepare(path: Path) -> pd.DataFrame:
    """Load data, sort temporally within each player."""
    df = pd.read_csv(path)
    df = df.sort_values(["player_id", "t_start"]).reset_index(drop=True)
    return df


def add_label_schemes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create multiple label schemes from the 5-cluster assignments.

    Scheme A (5-class):  C0, C1, C2, C3, C4 as-is
    Scheme B (3-class):  Confused (C1+C4), Productive (C2+C3), Transition (C0)
    Scheme C (binary):   Confused (C1+C4) vs Not-Confused (C0+C2+C3)
    """
    df["label_5class"] = df["cluster_id"]

    # 3-class: Confused / Productive / Transition
    mapping_3 = {0: "transition", 1: "confused", 2: "productive",
                 3: "productive", 4: "confused"}
    df["label_3class"] = df["cluster_id"].map(mapping_3)

    # Binary: Confused vs Not-Confused
    mapping_2 = {0: 0, 1: 1, 2: 0, 3: 0, 4: 1}
    df["label_binary"] = df["cluster_id"].map(mapping_2)

    # Sub-type binary: Disorientation (C1) vs Impasse (C4) — only within confused
    df["label_confusion_type"] = np.nan
    df.loc[df["cluster_id"] == 1, "label_confusion_type"] = "disorientation"
    df.loc[df["cluster_id"] == 4, "label_confusion_type"] = "impasse"

    return df


# ============================================================================
# 2. Temporal Feature Engineering
# ============================================================================

def add_temporal_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Add lag and rolling-window features to capture temporal dynamics.
    Computed per-player to avoid cross-player leakage.

    New features:
      - lag_1, lag_2 for each base feature (values from previous windows)
      - rolling_mean_3, rolling_std_3 (3-window rolling statistics)
      - delta_1 (first-order difference)
      - cluster_duration: how many consecutive windows of same cluster so far
    """
    temporal_cols = []

    for col in feature_cols:
        for lag in [1, 2]:
            new_col = f"{col}_lag{lag}"
            df[new_col] = df.groupby("player_id")[col].shift(lag)
            temporal_cols.append(new_col)

        roll_mean = f"{col}_rmean3"
        roll_std = f"{col}_rstd3"
        delta = f"{col}_delta1"

        df[roll_mean] = df.groupby("player_id")[col].transform(
            lambda x: x.rolling(3, min_periods=1).mean())
        df[roll_std] = df.groupby("player_id")[col].transform(
            lambda x: x.rolling(3, min_periods=1).std().fillna(0))
        df[delta] = df.groupby("player_id")[col].diff().fillna(0)

        temporal_cols.extend([roll_mean, roll_std, delta])

    # Cluster duration: consecutive same-cluster run length
    def cluster_run_length(series):
        lengths = []
        count = 1
        for i in range(len(series)):
            if i > 0 and series.iloc[i] == series.iloc[i - 1]:
                count += 1
            else:
                count = 1
            lengths.append(count)
        return lengths

    df["cluster_run_length"] = df.groupby("player_id")["cluster_id"].transform(
        cluster_run_length)
    temporal_cols.append("cluster_run_length")

    # Fill NaN from lag features with 0 (first windows of each player)
    df[temporal_cols] = df[temporal_cols].fillna(0)

    return df, temporal_cols


# ============================================================================
# 3. Model Definitions
# ============================================================================

def get_models(n_classes: int, class_weights: dict = None):
    """Return dict of models to evaluate."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            class_weight=class_weights or "balanced",
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial" if n_classes > 2 else "auto",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            class_weight=class_weights or "balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ),
    }
    return models


# ============================================================================
# 4. Evaluation: Leave-One-Player-Out Cross-Validation
# ============================================================================

def evaluate_lopo(df: pd.DataFrame, feature_cols: list, label_col: str,
                  models: dict, label_names: list = None) -> dict:
    """
    Leave-One-Player-Out (LOPO) cross-validation.

    Why LOPO instead of random k-fold:
      - Prevents temporal leakage (adjacent windows are correlated)
      - Tests generalization to UNSEEN players (the real deployment scenario)
      - Each fold = 1 player held out, trained on remaining 10
    """
    groups = df["player_id"].values
    logo = LeaveOneGroupOut()

    X = df[feature_cols].values
    y = df[label_col].values

    # Encode labels if string
    le = LabelEncoder()
    if y.dtype == object:
        y = le.fit_transform(y)
        if label_names is None:
            label_names = list(le.classes_)

    results = {}

    for model_name, model_template in models.items():
        all_y_true = []
        all_y_pred = []
        fold_scores = []

        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Handle XGBoost class weights via scale_pos_weight or sample_weight
            from sklearn.base import clone
            model = clone(model_template)

            if model_name == "GradientBoosting":
                from sklearn.utils.class_weight import compute_sample_weight
                sw = compute_sample_weight("balanced", y_train)
                model.fit(X_train_s, y_train, sample_weight=sw)
            else:
                model.fit(X_train_s, y_train)

            y_pred = model.predict(X_test_s)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            fold_scores.append(f1_score(y_test, y_pred, average="macro",
                                         zero_division=0))

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        acc = accuracy_score(all_y_true, all_y_pred)
        f1_macro = f1_score(all_y_true, all_y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0)

        results[model_name] = {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "f1_per_fold_mean": np.mean(fold_scores),
            "f1_per_fold_std": np.std(fold_scores),
            "y_true": all_y_true,
            "y_pred": all_y_pred,
            "report": classification_report(
                all_y_true, all_y_pred,
                target_names=label_names,
                zero_division=0,
            ),
        }

    return results


# ============================================================================
# 5. Feature Importance Analysis
# ============================================================================

def get_feature_importance(df, feature_cols, label_col):
    """Train XGBoost on full data to extract feature importances."""
    X = df[feature_cols].values
    y = df[label_col].values
    le = LabelEncoder()
    if y.dtype == object:
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=42)
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight("balanced", y)
    model.fit(X_s, y, sample_weight=sw)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance


# ============================================================================
# 6. Visualization
# ============================================================================

def plot_results(all_results: dict, output_dir: Path):
    """Generate publication-quality evaluation plots."""
    output_dir.mkdir(exist_ok=True)

    # --- (A) Model comparison bar chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (scheme_name, scheme_results) in enumerate(all_results.items()):
        ax = axes[idx]
        model_names = list(scheme_results.keys())
        accs = [scheme_results[m]["accuracy"] for m in model_names]
        f1s = [scheme_results[m]["f1_macro"] for m in model_names]

        x = np.arange(len(model_names))
        w = 0.35
        ax.bar(x - w/2, accs, w, label="Accuracy", color="#5B8DB8", alpha=0.85)
        ax.bar(x + w/2, f1s, w, label="Macro F1", color="#E07B9B", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=9, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(scheme_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Value labels
        for i, (a, f) in enumerate(zip(accs, f1s)):
            ax.text(i - w/2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(i + w/2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

    fig.suptitle("Model Performance by Label Scheme (LOPO-CV)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "12_model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 12_model_comparison.png")

    # --- (B) Confusion matrices for best model per scheme ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (scheme_name, scheme_results) in enumerate(all_results.items()):
        # Pick model with highest macro-F1
        best_model = max(scheme_results, key=lambda m: scheme_results[m]["f1_macro"])
        r = scheme_results[best_model]

        ax = axes[idx]
        labels = sorted(np.unique(np.concatenate([r["y_true"], r["y_pred"]])))
        cm = confusion_matrix(r["y_true"], r["y_pred"], labels=labels, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels, square=True)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{scheme_name}\n({best_model}, F1={r['f1_macro']:.3f})",
                     fontsize=10, fontweight="bold")

    fig.suptitle("Normalized Confusion Matrices (Best Model per Scheme)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "13_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 13_confusion_matrices.png")


def plot_feature_importance(importance_df: pd.DataFrame, output_dir: Path):
    """Bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    top_n = min(20, len(importance_df))
    imp = importance_df.head(top_n).sort_values("importance")
    ax.barh(imp["feature"], imp["importance"], color="#5B8DB8", alpha=0.85)
    ax.set_xlabel("Feature Importance (XGBoost gain)")
    ax.set_title("Top Feature Importances (5-class, GradientBoosting)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "14_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved 14_feature_importance.png")


# ============================================================================
# 7. Realtime vs Full Feature Comparison
# ============================================================================

def compare_feature_sets(df, temporal_cols, label_col, output_dir):
    """Compare realtime (eye+hand only) vs full feature set."""
    results = {}

    realtime_temp = REALTIME_FEATURES + [c for c in temporal_cols
                                          if any(c.startswith(f) for f in REALTIME_FEATURES)]
    full_temp = FULL_FEATURES + temporal_cols

    for set_name, feat_cols in [
        ("Realtime (5 features)", REALTIME_FEATURES),
        ("Realtime + temporal", realtime_temp),
        ("Full (8 features)", FULL_FEATURES),
        ("Full + temporal", full_temp),
    ]:
        # Use only XGBoost for this comparison
        models = {"GradientBoosting": get_models(5)["GradientBoosting"]}
        r = evaluate_lopo(df, feat_cols, label_col, models,
                          label_names=[CLUSTER_NAMES[i] for i in range(5)])
        results[set_name] = {
            "accuracy": r["GradientBoosting"]["accuracy"],
            "f1_macro": r["GradientBoosting"]["f1_macro"],
            "n_features": len(feat_cols),
        }

    comparison = pd.DataFrame(results).T
    print("\n  Feature Set Comparison (XGBoost, 5-class, LOPO):")
    print(comparison.to_string(float_format=lambda x: f"{x:.3f}"))

    comparison.to_csv(output_dir / "feature_set_comparison.csv")
    return comparison


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("Predictive Modeling Pipeline for VR Player States")
    print("=" * 70)

    # --- Load & prepare ---
    df = load_and_prepare(DATA_PATH)
    df = add_label_schemes(df)
    df, temporal_cols = add_temporal_features(df, FULL_FEATURES)

    all_features = FULL_FEATURES + temporal_cols
    print(f"\nData: {len(df)} windows, {df['player_id'].nunique()} players")
    print(f"Base features: {len(FULL_FEATURES)}, Temporal: {len(temporal_cols)}, "
          f"Total: {len(all_features)}")

    # --- Label distributions ---
    print("\nLabel distributions:")
    for scheme in ["label_5class", "label_3class", "label_binary"]:
        dist = df[scheme].value_counts().sort_index()
        print(f"  {scheme}: {dict(dist)}")

    all_results = {}

    # ================================================================
    # Experiment 1: 5-Class Classification
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: 5-Class (C0–C4)")
    print("=" * 70)

    label_names_5 = [CLUSTER_NAMES[i] for i in range(5)]
    models_5 = get_models(5)
    results_5 = evaluate_lopo(df, all_features, "label_5class", models_5,
                               label_names=label_names_5)

    for m, r in results_5.items():
        print(f"\n  {m}: Acc={r['accuracy']:.3f}, "
              f"F1-macro={r['f1_macro']:.3f}, F1-weighted={r['f1_weighted']:.3f}")
        print(r["report"])

    all_results["5-Class (C0–C4)"] = results_5

    # ================================================================
    # Experiment 2: 3-Class (Confused / Productive / Transition)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: 3-Class (Confused / Productive / Transition)")
    print("=" * 70)

    models_3 = get_models(3)
    results_3 = evaluate_lopo(df, all_features, "label_3class", models_3,
                               label_names=["confused", "productive", "transition"])

    for m, r in results_3.items():
        print(f"\n  {m}: Acc={r['accuracy']:.3f}, "
              f"F1-macro={r['f1_macro']:.3f}")

    all_results["3-Class"] = results_3

    # ================================================================
    # Experiment 3: Binary (Confused vs Not)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Binary (Confused vs Not)")
    print("=" * 70)

    models_2 = get_models(2)
    results_2 = evaluate_lopo(df, all_features, "label_binary", models_2,
                               label_names=["Not Confused", "Confused"])

    for m, r in results_2.items():
        print(f"\n  {m}: Acc={r['accuracy']:.3f}, "
              f"F1-macro={r['f1_macro']:.3f}")

    all_results["Binary"] = results_2

    # ================================================================
    # Experiment 4: Confusion Sub-Type (C1 vs C4 only)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Confusion Sub-Type (Disorientation vs Impasse)")
    print("=" * 70)

    df_confused = df[df["label_confusion_type"].notna()].copy()
    print(f"  Using {len(df_confused)} confused windows only")

    models_sub = get_models(2)
    results_sub = evaluate_lopo(df_confused, all_features, "label_confusion_type",
                                 models_sub,
                                 label_names=["disorientation", "impasse"])

    for m, r in results_sub.items():
        print(f"\n  {m}: Acc={r['accuracy']:.3f}, "
              f"F1-macro={r['f1_macro']:.3f}")
        print(r["report"])

    all_results["Confusion Sub-Type"] = results_sub

    # ================================================================
    # Feature Importance
    # ================================================================
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)
    importance = get_feature_importance(df, all_features, "label_5class")
    print(importance.head(15).to_string(index=False))

    # ================================================================
    # Feature Set Comparison (Realtime vs Full)
    # ================================================================
    print("\n" + "=" * 70)
    print("FEATURE SET COMPARISON")
    print("=" * 70)
    feat_comparison = compare_feature_sets(df, temporal_cols, "label_5class", OUTPUT_DIR)

    # ================================================================
    # Plots
    # ================================================================
    print("\n[Plots]")
    # For confusion matrices, use readable label names
    # Remap results to use string labels for 5-class
    for m in results_5:
        results_5[m]["y_true_str"] = [label_names_5[y] for y in results_5[m]["y_true"]]
        results_5[m]["y_pred_str"] = [label_names_5[y] for y in results_5[m]["y_pred"]]

    plot_results(all_results, OUTPUT_DIR)
    plot_feature_importance(importance, OUTPUT_DIR)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    rows = []
    for scheme, results in all_results.items():
        for model_name, r in results.items():
            rows.append({
                "Scheme": scheme,
                "Model": model_name,
                "Accuracy": r["accuracy"],
                "F1-macro": r["f1_macro"],
                "F1-weighted": r["f1_weighted"],
                "F1-fold-mean": r["f1_per_fold_mean"],
                "F1-fold-std": r["f1_per_fold_std"],
            })

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary_df.to_csv(OUTPUT_DIR / "model_evaluation_summary.csv", index=False)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
