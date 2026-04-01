"""
Behavioral Timeline Visualization
===================================
Publication-ready timelines showing cluster transitions overlaid with
puzzle solve/error events for selected players.

Reads from batch_output_k5/ outputs.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
# BrokenBarHCollection removed in newer matplotlib; not needed
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "batch_output_k5"
OUTPUT_DIR = DATA_DIR  # save alongside other outputs

SELECTED_PLAYERS = [1, 6, 9, 22]

CLUSTER_NAMES = {
    0: "Transition",
    1: "Waiting",
    2: "Solving",
    3: "Exploration",
    4: "Stuck-on-clue",
}

CLUSTER_COLORS = {
    0: "#AAAAAA",  # gray
    1: "#F4A259",  # orange
    2: "#5B8DB8",  # blue
    3: "#E07B9B",  # pink
    4: "#7BC67E",  # green
}

PUZZLE_SHORT = {
    "Spoke Puzzle: Pasta in Sauce": "Pasta",
    "Spoke Puzzle: Amount of Protein": "Protein",
    "Spoke Puzzle: Water Amount": "Water",
    "Spoke Puzzle_ Water Amount": "Water",
    "Spoke Puzzle: Amount of Sunlight": "Sunlight",
    "Hub Puzzle: Cooking Pot": "Hub: Cook",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    windows = pd.read_csv(DATA_DIR / "all_windows_with_clusters.csv")
    perf = pd.read_csv(DATA_DIR / "puzzle_level_performance.csv")
    return windows, perf


def load_puzzle_events(player_id: int) -> pd.DataFrame:
    """Load raw PuzzleLogs for a player, return error/solve events with elapsed_sec."""
    path = BASE_DIR / f"User-{player_id}" / f"User-{player_id}_PuzzleLogs.csv"
    pl = pd.read_csv(path)
    pl["datetime"] = pd.to_datetime(pl["TimeStampUTC"], utc=True)
    t0 = pl["datetime"].iloc[0]
    pl["elapsed_sec"] = (pl["datetime"] - t0).dt.total_seconds()

    # Normalize puzzle names
    pl["PuzzleNorm"] = pl["PuzzleUniqueID"].str.replace("Spoke Puzzle_ ", "Spoke Puzzle: ", regex=False).str.strip()
    pl["ParentNorm"] = pl["ParentChain"].str.replace("Spoke Puzzle_ ", "Spoke Puzzle: ", regex=False).str.strip()

    events = []

    # Errors (WrongMove)
    errors = pl[pl["Outcome"] == "WrongMove"]
    for _, row in errors.iterrows():
        events.append({
            "elapsed_sec": row["elapsed_sec"],
            "event_type": "error",
            "label": row["PuzzleNorm"][:25],
        })

    # Puzzle completions (top-level spoke/hub only)
    main_puzzles = [
        "Spoke Puzzle: Pasta in Sauce",
        "Spoke Puzzle: Amount of Protein",
        "Spoke Puzzle: Water Amount",
        "Spoke Puzzle: Amount of Sunlight",
        "Hub Puzzle: Cooking Pot",
    ]
    for pname in main_puzzles:
        mask = (pl["PuzzleNorm"] == pname) & (pl["IsCompleted"] == "Completed")
        for _, row in pl[mask].iterrows():
            short = PUZZLE_SHORT.get(pname, pname[:15])
            events.append({
                "elapsed_sec": row["elapsed_sec"],
                "event_type": "solve",
                "label": short,
            })

    return pd.DataFrame(events)


# ============================================================================
# Timeline Plot
# ============================================================================

def plot_timelines(windows: pd.DataFrame, perf: pd.DataFrame,
                   player_ids: list, output_dir: Path):
    """Create a multi-panel behavioral timeline figure."""

    n_players = len(player_ids)
    fig, axes = plt.subplots(n_players, 1, figsize=(20, 3.2 * n_players + 1.5),
                              sharex=False)
    if n_players == 1:
        axes = [axes]

    for idx, pid in enumerate(player_ids):
        ax = axes[idx]
        pw = windows[windows["player_id"] == pid].copy().sort_values("t_start")
        events = load_puzzle_events(pid)
        max_t = pw["t_end"].max()

        # ---- Background: shaded bars for each window's cluster ----
        window_sec = 5.0
        for _, row in pw.iterrows():
            c = int(row["cluster_id"])
            ax.barh(
                y=c, width=window_sec, left=row["t_start"],
                height=0.7, color=CLUSTER_COLORS[c],
                edgecolor="none", alpha=0.85
            )

        # ---- Highlight long idle (C1) segments with background shading ----
        # Find consecutive C1 runs ≥ 3 windows (15s+)
        c1_mask = pw["cluster_id"].values == 1
        starts = []
        run_start = None
        for i, is_c1 in enumerate(c1_mask):
            if is_c1 and run_start is None:
                run_start = i
            elif not is_c1 and run_start is not None:
                if i - run_start >= 3:
                    starts.append((run_start, i - 1))
                run_start = None
        if run_start is not None and len(c1_mask) - run_start >= 3:
            starts.append((run_start, len(c1_mask) - 1))

        for s, e in starts:
            t_s = pw.iloc[s]["t_start"]
            t_e = pw.iloc[e]["t_end"]
            ax.axvspan(t_s, t_e, color="#F4A259", alpha=0.10, zorder=0)

        # ---- Similarly highlight long C4 (stuck-on-clue) runs ----
        c4_mask = pw["cluster_id"].values == 4
        run_start = None
        for i, is_c4 in enumerate(c4_mask):
            if is_c4 and run_start is None:
                run_start = i
            elif not is_c4 and run_start is not None:
                if i - run_start >= 3:
                    t_s = pw.iloc[run_start]["t_start"]
                    t_e = pw.iloc[i - 1]["t_end"]
                    ax.axvspan(t_s, t_e, color="#7BC67E", alpha=0.10, zorder=0)
                run_start = None
        if run_start is not None and len(c4_mask) - run_start >= 3:
            t_s = pw.iloc[run_start]["t_start"]
            t_e = pw.iloc[len(c4_mask) - 1]["t_end"]
            ax.axvspan(t_s, t_e, color="#7BC67E", alpha=0.10, zorder=0)

        # ---- Overlay error events ----
        if len(events) > 0:
            errs = events[events["event_type"] == "error"]
            for _, ev in errs.iterrows():
                t = ev["elapsed_sec"]
                # Find which cluster the player was in at this time
                match = pw[(pw["t_start"] <= t) & (pw["t_end"] >= t)]
                y = match["cluster_id"].values[0] if len(match) > 0 else 2
                ax.scatter(t, y, marker="X", c="red", s=100, zorder=10,
                           edgecolors="darkred", linewidths=0.8)

            # ---- Overlay solve events ----
            solves = events[events["event_type"] == "solve"]
            for _, ev in solves.iterrows():
                t = ev["elapsed_sec"]
                ax.axvline(t, color="#2E8B57", linewidth=1.8, linestyle="-",
                           alpha=0.8, zorder=5)
                ax.text(t, 4.55, ev["label"], fontsize=7.5, color="#2E8B57",
                        ha="center", va="bottom", fontweight="bold",
                        rotation=35)

        # ---- Formatting ----
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels([CLUSTER_NAMES[i] for i in range(5)], fontsize=10)
        ax.set_ylim(-0.5, 5.3)
        ax.set_xlim(0, max_t + 10)
        ax.set_ylabel("")
        ax.set_title(f"User-{pid}", fontsize=13, fontweight="bold", loc="left")
        ax.grid(axis="x", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Time axis in minutes
        max_min = int(max_t // 60) + 1
        tick_locs = [m * 60 for m in range(0, max_min + 1, 2)]
        tick_labels = [f"{m}:00" for m in range(0, max_min + 1, 2)]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, fontsize=9)

    axes[-1].set_xlabel("Elapsed Time (mm:ss)", fontsize=11)

    # ---- Global legend ----
    legend_elements = []
    for c in range(5):
        legend_elements.append(
            mpatches.Patch(color=CLUSTER_COLORS[c], label=CLUSTER_NAMES[c])
        )
    legend_elements.append(
        mlines.Line2D([], [], marker="X", color="red", linestyle="None",
                      markersize=9, markeredgecolor="darkred", label="Error")
    )
    legend_elements.append(
        mlines.Line2D([], [], color="#2E8B57", linewidth=2, label="Puzzle Solved")
    )
    legend_elements.append(
        mpatches.Patch(facecolor="#F4A259", alpha=0.25, label="Prolonged Waiting")
    )
    legend_elements.append(
        mpatches.Patch(facecolor="#7BC67E", alpha=0.25, label="Prolonged Stuck-on-clue")
    )

    fig.legend(handles=legend_elements, loc="upper center",
               ncol=5, fontsize=9.5, frameon=True,
               bbox_to_anchor=(0.5, 1.01), edgecolor="gray")

    fig.suptitle("Behavioral State Timelines with Puzzle Events",
                 fontsize=15, fontweight="bold", y=1.04)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    path = output_dir / "09_behavioral_timelines.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")

    return path


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading data...")
    windows, perf = load_data()

    print(f"Plotting timelines for Users: {SELECTED_PLAYERS}")
    plot_timelines(windows, perf, SELECTED_PLAYERS, OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
