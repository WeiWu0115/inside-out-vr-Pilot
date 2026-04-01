"""
Batch Stuck-State Clustering + Performance Analysis Pipeline
=============================================================
Phase 1: Run clustering on all 11 players (same as v2 single-player pipeline)
Phase 2: Map windows → puzzles, aggregate cluster distributions per puzzle
Phase 3: Merge with puzzle performance, run correlation & group analyses

Inputs:  User-log/User-{id}/ folders
Outputs: User-log/batch_output/
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy, mannwhitneyu, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "batch_output_k5"
PLAYER_IDS = [1, 2, 3, 5, 6, 9, 10, 12, 14, 22, 23]

WINDOW_DURATION_SEC = 5.0

CLUE_OBJECTS = [
    "Left Diary - Pasta Note",
    "Left Diary - Water Puzzle",
    "Left Diary - Amount of Sunlight",
    "Left Diary - Amount of Protien",
    "Right Diary - Pasta Note",
    "Hub Puzzle Hint 1",
    "Hub Puzzle Hint 2",
    "Hub Puzzle Hint Title",
    "Hub Puzzle_ Cooking Pot Whitboard",
    "ChalkBaord Front",
    "ChalkBaord Back",
    "StoneOven",
    "Carrots Plant Box",
    "Potato Plant Box",
    "SeedPacket Pumpkin",
    "JarCarbonaraSauce Lid SnapInteractable(Clone)",
]

# Top-level puzzles to track (normalized names)
MAIN_PUZZLES = [
    "Spoke Puzzle: Pasta in Sauce",
    "Spoke Puzzle: Amount of Protein",
    "Spoke Puzzle: Water Amount",       # note: some logs use _ instead of :
    "Spoke Puzzle: Amount of Sunlight",
    "Hub Puzzle: Cooking Pot",
]

K_RANGE = range(2, 6)
FORCE_K = 5  # Set to None to auto-select by silhouette; set to int to force K

FEATURE_COLS = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time",
    "puzzle_active", "error_count", "time_since_action",
]


# ============================================================================
# Phase 1: Feature Extraction & Clustering (per player)
# ============================================================================

def load_tracking(player_id: int) -> tuple:
    path = BASE_DIR / f"User-{player_id}" / f"User-{player_id}_PlayerTracking.csv"
    df = pd.read_csv(path, low_memory=False)
    df["datetime"] = pd.to_datetime(df["Timestamp"], utc=True)
    t0 = df["datetime"].iloc[0]
    df["elapsed_sec"] = (df["datetime"] - t0).dt.total_seconds()
    df = df[df["SystemStatus"] != "NO_TRACKING"].copy()
    df.dropna(subset=["GazeTarget_ObjectName"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, t0


def load_puzzle_logs(player_id: int, t0) -> tuple:
    path = BASE_DIR / f"User-{player_id}" / f"User-{player_id}_PuzzleLogs.csv"
    pl = pd.read_csv(path)
    pl["datetime"] = pd.to_datetime(pl["TimeStampUTC"], utc=True)
    pl["elapsed_sec"] = (pl["datetime"] - t0).dt.total_seconds()
    pl_events = pl[pl["ElementType"].isin(["Interaction", "Puzzle"])].copy()
    pl_events.reset_index(drop=True, inplace=True)
    return pl, pl_events


# --- Window feature functions (same as v2) ---

def compute_gaze_entropy(group):
    counts = group["GazeTarget_ObjectName"].value_counts()
    probs = counts / counts.sum()
    return shannon_entropy(probs, base=2)


def compute_clue_ratio(group, clue_objects):
    return group["GazeTarget_ObjectName"].isin(clue_objects).mean()


def compute_switch_rate(group):
    targets = group["GazeTarget_ObjectName"].values
    switches = np.sum(targets[1:] != targets[:-1])
    duration = group["elapsed_sec"].iloc[-1] - group["elapsed_sec"].iloc[0]
    return switches / duration if duration > 0 else 0.0


def compute_action_count(group):
    count = 0
    for col in ["LeftHand_Action", "RightHand_Action"]:
        vals = group[col].values
        count += np.sum(vals[1:] != vals[:-1])
    return int(count)


def compute_idle_time(group):
    passive = {"NOT_TRACKED", "OPEN"}
    gaze = group["GazeTarget_ObjectName"].values
    left = group["LeftHand_Action"].values
    right = group["RightHand_Action"].values
    times = group["elapsed_sec"].values
    if len(times) < 2:
        return 0.0
    idle = 0.0
    for i in range(1, len(times)):
        if (gaze[i] == gaze[i - 1]
                and left[i] in passive and right[i] in passive):
            idle += times[i] - times[i - 1]
    return idle


def compute_puzzle_features(pl_events, window_sec, n_windows):
    pl_e = pl_events.copy()
    pl_e["window_id"] = (pl_e["elapsed_sec"] // window_sec).astype(int)
    event_times = np.sort(pl_e["elapsed_sec"].values)
    records = []
    for wid in range(n_windows):
        w_start = wid * window_sec
        mask = pl_e["window_id"] == wid
        w_events = pl_e[mask]
        puzzle_active = 1 if len(w_events) > 0 else 0
        error_count = int((w_events["Outcome"] == "WrongMove").sum())
        past = event_times[event_times <= w_start]
        time_since = w_start - past[-1] if len(past) > 0 else w_start
        records.append({
            "window_id": wid,
            "puzzle_active": puzzle_active,
            "error_count": error_count,
            "time_since_action": time_since,
        })
    return pd.DataFrame(records)


def extract_features_for_player(df, pl_events, window_sec):
    """Extract all 8 features for one player's windows."""
    df["window_id"] = (df["elapsed_sec"] // window_sec).astype(int)
    grouped = df.groupby("window_id")

    tracking_records = []
    for wid, group in grouped:
        tracking_records.append({
            "window_id": wid,
            "gaze_entropy": compute_gaze_entropy(group),
            "clue_ratio": compute_clue_ratio(group, CLUE_OBJECTS),
            "switch_rate": compute_switch_rate(group),
            "action_count": compute_action_count(group),
            "idle_time": compute_idle_time(group),
        })
    tracking_feats = pd.DataFrame(tracking_records)

    # Drop short last window
    last_wid = tracking_feats["window_id"].max()
    last_n = len(df[df["window_id"] == last_wid])
    median_n = df.groupby("window_id").size().median()
    if last_n < 0.5 * median_n:
        tracking_feats = tracking_feats[tracking_feats["window_id"] != last_wid]

    puzzle_feats = compute_puzzle_features(pl_events, window_sec, last_wid + 1)
    features = tracking_feats.merge(puzzle_feats, on="window_id", how="left")
    features.fillna({"puzzle_active": 0, "error_count": 0, "time_since_action": 0},
                     inplace=True)
    return features, df


def run_phase1():
    """Run feature extraction + clustering on all players pooled together."""
    print("=" * 70)
    print("PHASE 1: Feature Extraction & Clustering (all players)")
    print("=" * 70)

    all_features = []

    for pid in PLAYER_IDS:
        print(f"  Processing User-{pid}...", end=" ")
        df, t0 = load_tracking(pid)
        _, pl_events = load_puzzle_logs(pid, t0)
        features, df_windowed = extract_features_for_player(
            df, pl_events, WINDOW_DURATION_SEC)
        features["player_id"] = pid

        # Store window time ranges for puzzle mapping later
        for _, row in features.iterrows():
            wid = int(row["window_id"])
            w_frames = df_windowed[df_windowed["window_id"] == wid]
            features.loc[features["window_id"] == wid, "t_start"] = \
                w_frames["elapsed_sec"].iloc[0]
            features.loc[features["window_id"] == wid, "t_end"] = \
                w_frames["elapsed_sec"].iloc[-1]

        all_features.append(features)
        print(f"{len(features)} windows")

    combined = pd.concat(all_features, ignore_index=True)
    print(f"\n  Total: {len(combined)} windows from {len(PLAYER_IDS)} players")

    # --- Normalize & cluster (pooled across all players) ---
    X = combined[FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  K-means evaluation:")
    print(f"  {'K':>3}  {'Silhouette':>12}  {'Inertia':>12}")
    results = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results[k] = {"model": km, "labels": labels, "silhouette": sil}
        print(f"  {k:>3}  {sil:>12.4f}  {km.inertia_:>12.1f}")

    auto_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"\n  Auto-best K = {auto_k} (silhouette = {results[auto_k]['silhouette']:.4f})")

    if FORCE_K is not None:
        best_k = FORCE_K
        print(f"  → Forcing K = {best_k} (silhouette = {results[best_k]['silhouette']:.4f})")
    else:
        best_k = auto_k
        print(f"  → Using K = {best_k}")

    best_labels = results[best_k]["labels"]

    combined["cluster_id"] = best_labels

    # --- Print cluster summary ---
    print(f"\n  Cluster summary (pooled):")
    summary = combined.groupby("cluster_id")[FEATURE_COLS].mean()
    counts = combined["cluster_id"].value_counts().sort_index()
    summary.insert(0, "n_windows", counts)
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    return combined, X_scaled, best_k, scaler, results


# ============================================================================
# Phase 2: Map Windows → Puzzles
# ============================================================================

def normalize_puzzle_name(name: str) -> str:
    """Normalize inconsistent puzzle naming (e.g., _ vs :)."""
    return (name
            .replace("Spoke Puzzle_ ", "Spoke Puzzle: ")
            .replace("Spoke Puzzle_", "Spoke Puzzle:")
            .strip())


def get_puzzle_time_ranges(player_id: int, t0) -> list:
    """
    Extract time ranges for each main puzzle from PuzzleLogs.
    Returns list of dicts: {puzzle_id, t_start, t_end, error_count, completion_time}
    """
    path = BASE_DIR / f"User-{player_id}" / f"User-{player_id}_PuzzleLogs.csv"
    pl = pd.read_csv(path)
    pl["datetime"] = pd.to_datetime(pl["TimeStampUTC"], utc=True)
    pl["elapsed_sec"] = (pl["datetime"] - t0).dt.total_seconds()
    pl["PuzzleNorm"] = pl["PuzzleUniqueID"].apply(normalize_puzzle_name)
    # Also normalize within ParentChain
    pl["ParentNorm"] = pl["ParentChain"].apply(normalize_puzzle_name)

    puzzles = []
    for puzzle_name in MAIN_PUZZLES:
        # Find all events belonging to this puzzle (in ParentChain)
        mask = pl["ParentNorm"].str.contains(puzzle_name, na=False)
        events = pl[mask]
        if len(events) == 0:
            continue

        t_start = events["elapsed_sec"].min()
        t_end = events["elapsed_sec"].max()
        errors = (events["Outcome"] == "WrongMove").sum()

        # Check if puzzle was completed
        completion_mask = (
            (pl["PuzzleNorm"] == puzzle_name) &
            (pl["IsCompleted"] == "Completed")
        )
        completed = completion_mask.any()

        puzzles.append({
            "player_id": player_id,
            "puzzle_id": puzzle_name,
            "t_start": t_start,
            "t_end": t_end,
            "completion_time": t_end - t_start,
            "success": 1 if completed else 0,
            "error_count": int(errors),
        })

    return puzzles


def map_windows_to_puzzles(combined: pd.DataFrame) -> pd.DataFrame:
    """
    For each window, determine which puzzle it belongs to based on time overlap.
    A window is assigned to a puzzle if its midpoint falls within [t_start, t_end].
    Windows outside any puzzle range → puzzle_id = 'Transition'.
    Also includes the 'exploration' phase before the first puzzle for each player.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Map Windows → Puzzles")
    print("=" * 70)

    all_puzzle_ranges = []
    # We need each player's t0 again
    for pid in PLAYER_IDS:
        path = BASE_DIR / f"User-{pid}" / f"User-{pid}_PlayerTracking.csv"
        df_tmp = pd.read_csv(path, usecols=["Timestamp"], nrows=1)
        t0 = pd.to_datetime(df_tmp["Timestamp"].iloc[0], utc=True)
        ranges = get_puzzle_time_ranges(pid, t0)
        all_puzzle_ranges.extend(ranges)

    puzzle_df = pd.DataFrame(all_puzzle_ranges)
    print(f"  Found {len(puzzle_df)} puzzle attempts across {len(PLAYER_IDS)} players")

    # Assign puzzle_id to each window
    combined["puzzle_phase"] = "Transition"
    for _, pr in puzzle_df.iterrows():
        pid = pr["player_id"]
        mask = (
            (combined["player_id"] == pid) &
            (combined["t_start"] >= pr["t_start"] - WINDOW_DURATION_SEC) &
            (combined["t_end"] <= pr["t_end"] + WINDOW_DURATION_SEC)
        )
        combined.loc[mask, "puzzle_phase"] = pr["puzzle_id"]

    # Summary
    phase_counts = combined.groupby("puzzle_phase").size()
    print(f"  Window distribution by phase:")
    for phase, cnt in phase_counts.items():
        print(f"    {phase}: {cnt} windows")

    return combined, puzzle_df


# ============================================================================
# Phase 3: Aggregate + Performance Analysis
# ============================================================================

def aggregate_cluster_features(combined: pd.DataFrame, best_k: int,
                                puzzle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by (player_id, puzzle_phase), compute cluster proportions
    and merge with puzzle performance data.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Aggregate & Performance Analysis")
    print("=" * 70)

    # Only keep windows assigned to actual puzzles
    puzzle_windows = combined[combined["puzzle_phase"] != "Transition"].copy()

    records = []
    for (pid, puzzle), group in puzzle_windows.groupby(["player_id", "puzzle_phase"]):
        total = len(group)
        row = {
            "player_id": pid,
            "puzzle_id": puzzle,
            "total_windows": total,
        }
        # Cluster proportions
        for c in range(best_k):
            row[f"pct_cluster_{c}"] = (group["cluster_id"] == c).mean()
            row[f"count_cluster_{c}"] = int((group["cluster_id"] == c).sum())

        # Transition count (cluster changes within puzzle)
        clusters = group.sort_values("window_id")["cluster_id"].values
        row["transition_count"] = int(np.sum(clusters[1:] != clusters[:-1]))

        records.append(row)

    agg_df = pd.DataFrame(records)

    # Merge with puzzle performance
    merged = agg_df.merge(
        puzzle_df[["player_id", "puzzle_id", "completion_time", "success", "error_count"]],
        on=["player_id", "puzzle_id"],
        how="left"
    )

    print(f"  Merged dataset: {len(merged)} rows (player × puzzle)")
    print(f"  Columns: {list(merged.columns)}")

    return merged


def run_analysis(merged: pd.DataFrame, best_k: int):
    """Run correlation, group comparison, and success analyses."""

    pct_cols = [f"pct_cluster_{c}" for c in range(best_k)]

    # ----------------------------------------------------------------
    # (A) Correlation Analysis
    # ----------------------------------------------------------------
    print("\n--- (A) Correlation Analysis ---")
    print(f"  {'Feature pair':<45} {'Pearson r':>10} {'p-value':>10} {'Spearman ρ':>10} {'p-value':>10}")
    print("  " + "-" * 90)

    corr_pairs = []
    for c in range(best_k):
        col = f"pct_cluster_{c}"
        for target in ["completion_time", "error_count"]:
            x = merged[col].values
            y = merged[target].values
            if np.std(x) > 0 and np.std(y) > 0:
                pr, pp = pearsonr(x, y)
                sr, sp = spearmanr(x, y)
                label = f"{col} vs {target}"
                sig_p = "**" if pp < 0.05 else ("*" if pp < 0.1 else "")
                sig_s = "**" if sp < 0.05 else ("*" if sp < 0.1 else "")
                print(f"  {label:<45} {pr:>9.3f}{sig_p} {pp:>10.4f} {sr:>9.3f}{sig_s} {sp:>10.4f}")
                corr_pairs.append({
                    "feature": col, "target": target,
                    "pearson_r": pr, "pearson_p": pp,
                    "spearman_r": sr, "spearman_p": sp
                })

    print("  (** p<0.05, * p<0.1)")

    # ----------------------------------------------------------------
    # (B) Group Comparison — high vs low cluster proportion
    # ----------------------------------------------------------------
    print("\n--- (B) Group Comparison (median split) ---")

    group_results = {}
    for c in range(best_k):
        col = f"pct_cluster_{c}"
        median_val = merged[col].median()
        high = merged[merged[col] > median_val]
        low = merged[merged[col] <= median_val]

        if len(high) < 2 or len(low) < 2:
            continue

        for target in ["completion_time", "error_count"]:
            h_vals = high[target].values
            l_vals = low[target].values
            try:
                stat, p = mannwhitneyu(h_vals, l_vals, alternative="two-sided")
            except ValueError:
                stat, p = np.nan, np.nan
            sig = "**" if p < 0.05 else ("*" if p < 0.1 else "")
            print(f"  {col} → {target}: "
                  f"high={np.mean(h_vals):.1f} (n={len(h_vals)}), "
                  f"low={np.mean(l_vals):.1f} (n={len(l_vals)}), "
                  f"U={stat:.0f}, p={p:.4f} {sig}")
            group_results[f"{col}_vs_{target}"] = {
                "high_mean": np.mean(h_vals), "low_mean": np.mean(l_vals), "p": p
            }

    # ----------------------------------------------------------------
    # (C) Success vs Failure comparison
    # ----------------------------------------------------------------
    print("\n--- (C) Success vs Failure ---")
    success = merged[merged["success"] == 1]
    failure = merged[merged["success"] == 0]
    print(f"  Success: {len(success)} attempts, Failure: {len(failure)} attempts")

    if len(failure) > 0 and len(success) > 0:
        print(f"\n  {'Cluster':<20} {'Success mean':>14} {'Failure mean':>14} {'Diff':>10}")
        print("  " + "-" * 60)
        for c in range(best_k):
            col = f"pct_cluster_{c}"
            s_mean = success[col].mean()
            f_mean = failure[col].mean()
            print(f"  {col:<20} {s_mean:>13.3f} {f_mean:>13.3f} {f_mean-s_mean:>+10.3f}")
    else:
        print("  All puzzles were completed successfully — no failure group.")
        print("  Comparing top-half vs bottom-half completion time instead:")
        median_ct = merged["completion_time"].median()
        fast = merged[merged["completion_time"] <= median_ct]
        slow = merged[merged["completion_time"] > median_ct]
        print(f"  Fast (≤{median_ct:.0f}s): {len(fast)}, Slow (>{median_ct:.0f}s): {len(slow)}")
        print(f"\n  {'Cluster':<20} {'Fast mean':>14} {'Slow mean':>14} {'Diff':>10}")
        print("  " + "-" * 60)
        for c in range(best_k):
            col = f"pct_cluster_{c}"
            f_mean = fast[col].mean()
            s_mean = slow[col].mean()
            print(f"  {col:<20} {f_mean:>13.3f} {s_mean:>13.3f} {s_mean-f_mean:>+10.3f}")

    return corr_pairs, group_results


# ============================================================================
# Phase 4: Visualization
# ============================================================================

def generate_plots(merged: pd.DataFrame, combined: pd.DataFrame,
                   X_scaled: np.ndarray, best_k: int, output_dir: Path):
    """Generate all analysis plots."""
    print("\n[Plots]")
    palette = sns.color_palette("Set2", best_k)
    pct_cols = [f"pct_cluster_{c}" for c in range(best_k)]

    # --- 1. PCA of all windows (colored by cluster) ---
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(9, 7))
    for c in range(best_k):
        mask = combined["cluster_id"].values == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[palette[c]],
                   label=f"Cluster {c}", alpha=0.4, s=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"All Players — Window Clusters (K={best_k})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "01_pca_all_players.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  01_pca_all_players.png")

    # --- 2. Cluster centroid comparison (pooled) ---
    df_plot = combined[FEATURE_COLS].copy()
    df_plot = (df_plot - df_plot.mean()) / df_plot.std()
    df_plot["cluster"] = combined["cluster_id"].values
    centroids = df_plot.groupby("cluster")[FEATURE_COLS].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(FEATURE_COLS))
    width = 0.8 / best_k
    for c in range(best_k):
        offset = (c - best_k / 2 + 0.5) * width
        ax.bar(x + offset, centroids.loc[c].values, width,
               label=f"Cluster {c}", color=palette[c])
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_COLS, rotation=25, ha="right")
    ax.set_ylabel("Z-scored mean")
    ax.set_title("Cluster Centroid Feature Comparison (All Players Pooled)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.5)
    fig.savefig(output_dir / "02_centroid_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  02_centroid_chart.png")

    # --- 3. Scatter: pct_cluster vs completion_time (all clusters) ---
    n_rows_sc = (best_k + 2) // 3
    fig, axes = plt.subplots(n_rows_sc, 3, figsize=(16, 5 * n_rows_sc))
    axes = axes.flatten()
    for c in range(min(best_k, n_rows_sc * 3)):
        ax = axes[c]
        col = f"pct_cluster_{c}"
        ax.scatter(merged[col], merged["completion_time"],
                   alpha=0.6, color=palette[c], s=50)
        # Fit line
        if merged[col].std() > 0:
            z = np.polyfit(merged[col], merged["completion_time"], 1)
            xline = np.linspace(merged[col].min(), merged[col].max(), 50)
            ax.plot(xline, np.polyval(z, xline), "--", color="gray", alpha=0.7)
            r, p = pearsonr(merged[col], merged["completion_time"])
            ax.set_title(f"Cluster {c} (r={r:.2f}, p={p:.3f})", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Completion Time (s)")
    # Hide unused axes
    for i in range(best_k, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Cluster Proportion vs Completion Time", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "03_scatter_completion.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  03_scatter_completion.png")

    # --- 4. Scatter: pct_cluster vs error_count ---
    n_rows_sc2 = (best_k + 2) // 3
    fig, axes = plt.subplots(n_rows_sc2, 3, figsize=(16, 5 * n_rows_sc2))
    axes = axes.flatten()
    for c in range(min(best_k, n_rows_sc2 * 3)):
        ax = axes[c]
        col = f"pct_cluster_{c}"
        ax.scatter(merged[col], merged["error_count"],
                   alpha=0.6, color=palette[c], s=50)
        if merged[col].std() > 0:
            z = np.polyfit(merged[col], merged["error_count"], 1)
            xline = np.linspace(merged[col].min(), merged[col].max(), 50)
            ax.plot(xline, np.polyval(z, xline), "--", color="gray", alpha=0.7)
            r, p = pearsonr(merged[col], merged["error_count"])
            ax.set_title(f"Cluster {c} (r={r:.2f}, p={p:.3f})", fontsize=10)
        ax.set_xlabel(col)
        ax.set_ylabel("Error Count")
    for i in range(best_k, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle("Cluster Proportion vs Error Count", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "04_scatter_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  04_scatter_errors.png")

    # --- 5. Boxplot: completion_time by high/low stuck cluster ---
    # Find the cluster with highest time_since_action centroid → "stuck" cluster
    stuck_centroids = combined.groupby("cluster_id")["time_since_action"].mean()
    stuck_cluster = stuck_centroids.idxmax()
    stuck_col = f"pct_cluster_{stuck_cluster}"
    print(f"  (Identified Cluster {stuck_cluster} as primary 'stuck' cluster)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    median_stuck = merged[stuck_col].median()
    merged["stuck_group"] = np.where(
        merged[stuck_col] > median_stuck, "High stuck %", "Low stuck %")
    sns.boxplot(data=merged, x="stuck_group", y="completion_time", ax=axes[0],
                palette=["salmon", "lightblue"])
    axes[0].set_title(f"Completion Time by Cluster {stuck_cluster} Level")
    axes[0].set_xlabel("")

    # Find the cluster with highest error_count centroid → "error" cluster
    err_centroids = combined.groupby("cluster_id")["error_count"].mean()
    err_cluster = err_centroids.idxmax()
    err_col = f"pct_cluster_{err_cluster}"
    print(f"  (Identified Cluster {err_cluster} as primary 'error' cluster)")

    median_err = merged[err_col].median()
    merged["error_group"] = np.where(
        merged[err_col] > median_err, "High error-state %", "Low error-state %")
    sns.boxplot(data=merged, x="error_group", y="error_count", ax=axes[1],
                palette=["salmon", "lightblue"])
    axes[1].set_title(f"Error Count by Cluster {err_cluster} Level")
    axes[1].set_xlabel("")

    plt.tight_layout()
    fig.savefig(output_dir / "05_boxplot_groups.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  05_boxplot_groups.png")

    # --- 6. Stacked bar: cluster distribution by puzzle ---
    fig, ax = plt.subplots(figsize=(12, 6))
    puzzle_agg = merged.groupby("puzzle_id")[pct_cols].mean()
    puzzle_agg.plot(kind="bar", stacked=True, ax=ax, color=palette[:best_k])
    ax.set_ylabel("Proportion")
    ax.set_title("Average Cluster Distribution by Puzzle")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    fig.savefig(output_dir / "06_cluster_by_puzzle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  06_cluster_by_puzzle.png")

    # --- 7. Per-player cluster distribution bar ---
    fig, ax = plt.subplots(figsize=(12, 6))
    player_agg = merged.groupby("player_id")[pct_cols].mean()
    player_agg.plot(kind="bar", stacked=True, ax=ax, color=palette[:best_k])
    ax.set_ylabel("Proportion")
    ax.set_title("Average Cluster Distribution by Player")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("Player ID")
    fig.savefig(output_dir / "07_cluster_by_player.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  07_cluster_by_player.png")

    # --- 8. Heatmap: correlation matrix ---
    corr_cols = pct_cols + ["completion_time", "error_count", "transition_count"]
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = merged[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True)
    ax.set_title("Correlation Matrix: Clusters vs Performance")
    fig.savefig(output_dir / "08_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  08_correlation_heatmap.png")

    return stuck_cluster, err_cluster


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Phase 1: Clustering
    combined, X_scaled, best_k, scaler, km_results = run_phase1()

    # Phase 2: Map windows to puzzles
    combined, puzzle_df = map_windows_to_puzzles(combined)

    # Phase 3: Aggregate and analyze
    merged = aggregate_cluster_features(combined, best_k, puzzle_df)
    corr_pairs, group_results = run_analysis(merged, best_k)

    # Phase 4: Visualization
    stuck_c, err_c = generate_plots(merged, combined, X_scaled, best_k, OUTPUT_DIR)

    # Save outputs
    print("\n[Saving]")
    combined.to_csv(OUTPUT_DIR / "all_windows_with_clusters.csv", index=False)
    merged.to_csv(OUTPUT_DIR / "puzzle_performance_merged.csv", index=False)
    puzzle_df.to_csv(OUTPUT_DIR / "puzzle_level_performance.csv", index=False)
    print(f"  Saved 3 CSVs to {OUTPUT_DIR}/")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Players: {len(PLAYER_IDS)}")
    print(f"  Total windows: {len(combined)}")
    print(f"  Puzzle attempts: {len(merged)}")
    print(f"  Best K: {best_k}")
    print(f"  Stuck cluster: {stuck_c}")
    print(f"  Error cluster: {err_c}")
    print(f"  Outputs: {OUTPUT_DIR}/")
    print()
    print("  Key files:")
    print("    - all_windows_with_clusters.csv  (every window, every player)")
    print("    - puzzle_performance_merged.csv   (cluster % + performance per puzzle)")
    print("    - 01-08 PNG plots")


if __name__ == "__main__":
    main()
