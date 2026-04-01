"""
Adaptive Hint System Simulation
================================
Simulates the effect of two-stage adaptive hints on player stuck states.

Rules:
  - C1 (Idle Waiting) > 15 s  → Attention Guidance Hint → truncate, transition to C3
  - C4 (Stuck-on-Clue) > 20 s → Action Hint            → truncate, transition to C2

Assumptions:
  - After hint, the remaining windows in that stuck run are replaced by the
    target productive state (C3 for attention hints, C2 for action hints).
  - A cooldown of 30 s (6 windows) is enforced between consecutive hints
    to avoid over-prompting.
  - Completion time is re-estimated by summing window durations with the
    assumption that productive states (C2, C3) proceed ~40% faster than
    stuck states (C1, C4) toward puzzle progress.

Outputs (in batch_output_k5/):
  - hint_simulation_summary.csv
  - 10_simulation_comparison.png
  - 11_simulated_timelines.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "batch_output_k5"
OUTPUT_DIR = DATA_DIR

WINDOW_SEC = 5.0

# Hint trigger thresholds
C1_THRESHOLD_SEC = 15.0   # 3 consecutive C1 windows
C4_THRESHOLD_SEC = 20.0   # 4 consecutive C4 windows
C1_THRESHOLD_WIN = int(C1_THRESHOLD_SEC / WINDOW_SEC)
C4_THRESHOLD_WIN = int(C4_THRESHOLD_SEC / WINDOW_SEC)

# After hint, replace remaining stuck windows with:
C1_REPLACEMENT = 3   # C3 (Exploration) — player redirected to search
C4_REPLACEMENT = 2   # C2 (Active Solving) — player given actionable hint

# Cooldown between hints (windows)
COOLDOWN_WIN = 6  # 30 seconds

# For completion time estimation:
# We model that stuck windows contribute 0 progress toward puzzle completion,
# while productive windows contribute proportionally. So replacing N stuck
# windows with productive ones effectively saves N * WINDOW_SEC of wasted time.
# We apply a conservative discount: only 60% of replaced time counts as saved,
# since the player still needs time to process the hint.
SAVE_EFFICIENCY = 0.60

CLUSTER_NAMES = {
    0: "Transition",
    1: "Idle Waiting",
    2: "Active Solving",
    3: "Exploration",
    4: "Stuck-on-Clue",
}

CLUSTER_COLORS = {
    0: "#AAAAAA",
    1: "#F4A259",
    2: "#5B8DB8",
    3: "#E07B9B",
    4: "#7BC67E",
}

PLAYER_IDS = [1, 2, 3, 5, 6, 9, 10, 12, 14, 22, 23]
SHOWCASE_PLAYERS = [1, 6, 9, 22]


# ============================================================================
# Core Simulation
# ============================================================================

def detect_runs(clusters: np.ndarray) -> list:
    """Detect consecutive runs of the same cluster.
    Returns list of (start_idx, end_idx, cluster_id, length)."""
    runs = []
    if len(clusters) == 0:
        return runs
    start = 0
    for i in range(1, len(clusters)):
        if clusters[i] != clusters[start]:
            runs.append((start, i - 1, clusters[start], i - start))
            start = i
    runs.append((start, len(clusters) - 1, clusters[start], len(clusters) - start))
    return runs


def simulate_hints(clusters_orig: np.ndarray) -> dict:
    """
    Simulate adaptive hint interventions on a sequence of cluster IDs.

    Returns dict with:
      - clusters_sim: modified cluster sequence
      - hints: list of (window_idx, hint_type, run_length_before)
      - stats: before/after metrics
    """
    clusters = clusters_orig.copy()
    n = len(clusters)
    hints = []

    last_hint_idx = -COOLDOWN_WIN  # allow first hint immediately

    # Multi-pass: keep scanning until no more triggers
    # (single pass is sufficient since we modify in-place)
    runs = detect_runs(clusters)

    for start, end, cid, length in runs:
        if cid == 1 and length >= C1_THRESHOLD_WIN:
            # Trigger attention hint at the threshold point
            trigger_idx = start + C1_THRESHOLD_WIN
            if trigger_idx - last_hint_idx >= COOLDOWN_WIN and trigger_idx < n:
                # Replace windows after trigger with C3 (Exploration)
                for j in range(trigger_idx, end + 1):
                    clusters[j] = C1_REPLACEMENT
                hints.append((trigger_idx, "attention_guidance", length))
                last_hint_idx = trigger_idx

        elif cid == 4 and length >= C4_THRESHOLD_WIN:
            trigger_idx = start + C4_THRESHOLD_WIN
            if trigger_idx - last_hint_idx >= COOLDOWN_WIN and trigger_idx < n:
                for j in range(trigger_idx, end + 1):
                    clusters[j] = C4_REPLACEMENT
                hints.append((trigger_idx, "action_hint", length))
                last_hint_idx = trigger_idx

    # Compute before/after stats
    orig_c1 = np.sum(clusters_orig == 1) * WINDOW_SEC
    orig_c4 = np.sum(clusters_orig == 4) * WINDOW_SEC
    sim_c1 = np.sum(clusters == 1) * WINDOW_SEC
    sim_c4 = np.sum(clusters == 4) * WINDOW_SEC

    windows_saved = (
        (np.sum(clusters_orig == 1) - np.sum(clusters == 1)) +
        (np.sum(clusters_orig == 4) - np.sum(clusters == 4))
    )
    time_saved = windows_saved * WINDOW_SEC * SAVE_EFFICIENCY

    stats = {
        "orig_c1_time": orig_c1,
        "orig_c4_time": orig_c4,
        "orig_stuck_total": orig_c1 + orig_c4,
        "sim_c1_time": sim_c1,
        "sim_c4_time": sim_c4,
        "sim_stuck_total": sim_c1 + sim_c4,
        "stuck_reduction": orig_c1 + orig_c4 - sim_c1 - sim_c4,
        "stuck_reduction_pct": (1 - (sim_c1 + sim_c4) / (orig_c1 + orig_c4)) * 100
            if (orig_c1 + orig_c4) > 0 else 0,
        "n_hints": len(hints),
        "n_attention_hints": sum(1 for h in hints if h[1] == "attention_guidance"),
        "n_action_hints": sum(1 for h in hints if h[1] == "action_hint"),
        "est_time_saved": time_saved,
        "session_duration": len(clusters_orig) * WINDOW_SEC,
        "est_time_saved_pct": time_saved / (len(clusters_orig) * WINDOW_SEC) * 100
            if len(clusters_orig) > 0 else 0,
    }

    return {
        "clusters_sim": clusters,
        "hints": hints,
        "stats": stats,
    }


# ============================================================================
# Batch Simulation
# ============================================================================

def run_all_players(windows: pd.DataFrame) -> pd.DataFrame:
    """Run simulation for all players, return summary table."""
    records = []

    for pid in PLAYER_IDS:
        pw = windows[windows["player_id"] == pid].sort_values("t_start")
        clusters_orig = pw["cluster_id"].values.copy()
        result = simulate_hints(clusters_orig)
        s = result["stats"]
        s["player_id"] = pid
        records.append(s)

    summary = pd.DataFrame(records)

    # Add totals row
    totals = summary.select_dtypes(include=[np.number]).sum()
    # Recompute percentage columns
    totals["stuck_reduction_pct"] = (
        totals["stuck_reduction"] / totals["orig_stuck_total"] * 100
        if totals["orig_stuck_total"] > 0 else 0
    )
    totals["est_time_saved_pct"] = (
        totals["est_time_saved"] / totals["session_duration"] * 100
        if totals["session_duration"] > 0 else 0
    )
    totals["player_id"] = "TOTAL"
    summary = pd.concat([summary, pd.DataFrame([totals])], ignore_index=True)

    return summary


# ============================================================================
# Visualization
# ============================================================================

def plot_comparison_chart(summary: pd.DataFrame, output_dir: Path):
    """Bar charts comparing original vs simulated metrics."""

    player_rows = summary[summary["player_id"] != "TOTAL"].copy()
    player_rows["player_id"] = player_rows["player_id"].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # --- (a) Stuck time: before vs after ---
    ax = axes[0, 0]
    x = np.arange(len(player_rows))
    w = 0.35
    ax.bar(x - w/2, player_rows["orig_stuck_total"] / 60, w,
           color="#E07B9B", label="Original", alpha=0.85)
    ax.bar(x + w/2, player_rows["sim_stuck_total"] / 60, w,
           color="#5B8DB8", label="Simulated", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"U-{int(p)}" for p in player_rows["player_id"]], fontsize=9)
    ax.set_ylabel("Total Stuck Time (min)")
    ax.set_title("(a) Stuck Time: Original vs. Simulated")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # --- (b) % reduction in stuck time ---
    ax = axes[0, 1]
    colors = ["#2E8B57" if v > 0 else "#ccc" for v in player_rows["stuck_reduction_pct"]]
    ax.bar(x, player_rows["stuck_reduction_pct"], color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"U-{int(p)}" for p in player_rows["player_id"]], fontsize=9)
    ax.set_ylabel("Stuck Time Reduction (%)")
    ax.set_title("(b) Stuck Time Reduction per Player")
    ax.grid(axis="y", alpha=0.3)
    # Add value labels
    for i, v in enumerate(player_rows["stuck_reduction_pct"]):
        ax.text(i, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")

    # --- (c) Estimated time saved ---
    ax = axes[1, 0]
    ax.bar(x, player_rows["est_time_saved"] / 60, color="#5B8DB8", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"U-{int(p)}" for p in player_rows["player_id"]], fontsize=9)
    ax.set_ylabel("Estimated Time Saved (min)")
    ax.set_title("(c) Estimated Session Time Saved")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(player_rows["est_time_saved"] / 60):
        ax.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=8)

    # --- (d) Hint count breakdown ---
    ax = axes[1, 1]
    ax.bar(x - w/2, player_rows["n_attention_hints"], w,
           color="#F4A259", label="Attention Guidance (C1)", alpha=0.85)
    ax.bar(x + w/2, player_rows["n_action_hints"], w,
           color="#7BC67E", label="Action Hint (C4)", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"U-{int(p)}" for p in player_rows["player_id"]], fontsize=9)
    ax.set_ylabel("Number of Hints Triggered")
    ax.set_title("(d) Hint Triggers by Type")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Adaptive Hint System Simulation Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "10_simulation_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_simulated_timelines(windows: pd.DataFrame, output_dir: Path):
    """Side-by-side original vs simulated timelines for showcase players."""

    n = len(SHOWCASE_PLAYERS)
    fig, axes = plt.subplots(n, 2, figsize=(22, 3.0 * n + 1),
                              gridspec_kw={"wspace": 0.05})

    for idx, pid in enumerate(SHOWCASE_PLAYERS):
        pw = windows[windows["player_id"] == pid].sort_values("t_start")
        clusters_orig = pw["cluster_id"].values.copy()
        t_starts = pw["t_start"].values
        max_t = pw["t_end"].max()

        result = simulate_hints(clusters_orig)
        clusters_sim = result["clusters_sim"]
        hint_list = result["hints"]
        stats = result["stats"]

        for col, (clusters, title_suffix) in enumerate([
            (clusters_orig, "Original"),
            (clusters_sim, f"Simulated (−{stats['stuck_reduction_pct']:.0f}% stuck)")
        ]):
            ax = axes[idx, col]

            # Draw cluster bars
            for i, c in enumerate(clusters):
                ax.barh(y=c, width=WINDOW_SEC, left=t_starts[i],
                        height=0.7, color=CLUSTER_COLORS[c],
                        edgecolor="none", alpha=0.85)

            # On simulated panel, mark hint trigger points
            if col == 1:
                for (widx, htype, _) in hint_list:
                    t = t_starts[widx] if widx < len(t_starts) else max_t
                    marker = "v" if htype == "attention_guidance" else "D"
                    color = "#D4380D" if htype == "attention_guidance" else "#1D39C4"
                    ax.scatter(t, 4.8, marker=marker, c=color, s=70, zorder=10,
                               edgecolors="black", linewidths=0.5)

            ax.set_yticks([0, 1, 2, 3, 4])
            if col == 0:
                ax.set_yticklabels([CLUSTER_NAMES[i] for i in range(5)], fontsize=8.5)
            else:
                ax.set_yticklabels([])
            ax.set_ylim(-0.5, 5.5)
            ax.set_xlim(0, max_t + 10)

            # Time labels in minutes
            max_min = int(max_t // 60) + 1
            ticks = [m * 60 for m in range(0, max_min + 1, max(2, max_min // 8))]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(t//60)}:{int(t%60):02d}" for t in ticks], fontsize=7.5)

            ax.set_title(f"User-{pid} — {title_suffix}", fontsize=10, fontweight="bold", loc="left")
            ax.grid(axis="x", alpha=0.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    axes[-1, 0].set_xlabel("Time (mm:ss)", fontsize=10)
    axes[-1, 1].set_xlabel("Time (mm:ss)", fontsize=10)

    # Legend
    legend_elements = [mpatches.Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
                       for c in range(5)]
    legend_elements.append(
        mlines.Line2D([], [], marker="v", color="#D4380D", linestyle="None",
                      markersize=8, label="Attention Hint"))
    legend_elements.append(
        mlines.Line2D([], [], marker="D", color="#1D39C4", linestyle="None",
                      markersize=8, label="Action Hint"))
    fig.legend(handles=legend_elements, loc="upper center", ncol=7,
               fontsize=9, bbox_to_anchor=(0.5, 1.02), frameon=True, edgecolor="gray")

    fig.suptitle("Original vs. Simulated Behavioral Timelines",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / "11_simulated_timelines.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Adaptive Hint System Simulation")
    print("=" * 70)

    windows = pd.read_csv(DATA_DIR / "all_windows_with_clusters.csv")

    # Run simulation for all players
    print("\nRunning simulation...")
    summary = run_all_players(windows)

    # Print summary table
    print("\n" + "=" * 100)
    print("SIMULATION RESULTS")
    print("=" * 100)
    cols_display = [
        "player_id",
        "orig_stuck_total", "sim_stuck_total", "stuck_reduction", "stuck_reduction_pct",
        "n_hints", "n_attention_hints", "n_action_hints",
        "est_time_saved", "session_duration", "est_time_saved_pct",
    ]
    print(summary[cols_display].to_string(
        index=False,
        float_format=lambda x: f"{x:.1f}",
    ))

    # Save
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "hint_simulation_summary.csv", index=False)
    print(f"\nSaved: hint_simulation_summary.csv")

    # Plots
    print("\nGenerating plots...")
    plot_comparison_chart(summary, OUTPUT_DIR)
    plot_simulated_timelines(windows, OUTPUT_DIR)

    # Print headline results
    totals = summary[summary["player_id"] == "TOTAL"].iloc[0]
    print("\n" + "=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)
    print(f"  Total hints triggered:    {int(totals['n_hints'])}")
    print(f"    - Attention guidance:    {int(totals['n_attention_hints'])}")
    print(f"    - Action hints:          {int(totals['n_action_hints'])}")
    print(f"  Total stuck time reduced:  {totals['stuck_reduction']/60:.1f} min "
          f"({totals['stuck_reduction_pct']:.1f}%)")
    print(f"  Estimated time saved:      {totals['est_time_saved']/60:.1f} min "
          f"({totals['est_time_saved_pct']:.1f}% of total session time)")
    print(f"  Avg hints per player:      {totals['n_hints']/11:.1f}")
    print(f"  Avg time saved per player: {totals['est_time_saved']/11/60:.1f} min")


if __name__ == "__main__":
    main()
