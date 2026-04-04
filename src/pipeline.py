"""
Main pipeline: load data -> run agents -> negotiate -> suggest support -> save.

Usage:
    python src/pipeline.py
    python src/pipeline.py --input data/windows.csv --output outputs/
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import INPUT_CSV, OUTPUT_DIR
from load_data import load_csv
from agents import attention_agent, action_agent, performance_agent, temporal_agent
from population_agent import population_agent
from negotiation import run_negotiation
from support import run_support


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent VR user-state interpretation pipeline")
    parser.add_argument("--input", default=INPUT_CSV, help="Path to input CSV")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


def _unpack_agent(df, col_prefix, agent_func, apply_args=None):
    """Run an agent and unpack its dict output into separate columns."""
    if apply_args is not None:
        results = apply_args
    else:
        results = df.apply(agent_func, axis=1)

    df[f"{col_prefix}_label"] = results.apply(lambda r: r["label"])
    df[f"{col_prefix}_confidence"] = results.apply(lambda r: r["confidence"])
    df[f"{col_prefix}_reasoning"] = results.apply(lambda r: r["reasoning"])
    return df


def compute_puzzle_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each window, compute how many seconds the player has spent on the
    current puzzle so far (puzzle_elapsed_time) and the ratio vs the population
    median for that puzzle (puzzle_elapsed_ratio).

    A ratio > 1.0 means the player is taking longer than the median player.
    """
    # Population median duration per puzzle (from 18-participant data)
    PUZZLE_MEDIAN_DURATION = {
        "Spoke Puzzle: Amount of Protein": 135,
        "Spoke Puzzle: Pasta in Sauce": 90,
        "Spoke Puzzle: Water Amount": 230,
        "Spoke Puzzle: Amount of Sunlight": 160,
        "Hub Puzzle: Cooking Pot": 605,
        "Transition": 600,  # not meaningful, just a fallback
    }

    elapsed = []
    ratio = []

    for pid in df["participant_id"].unique():
        p_df = df[df["participant_id"] == pid].sort_values("window_start")

        for puzzle in p_df["puzzle_id"].unique():
            pz_df = p_df[p_df["puzzle_id"] == puzzle]
            pz_start = pz_df["window_start"].min()
            median_dur = PUZZLE_MEDIAN_DURATION.get(puzzle, 300)

            for idx, row in pz_df.iterrows():
                et = row["window_start"] - pz_start
                elapsed.append((idx, et, et / median_dur if median_dur > 0 else 0))

    elapsed_df = pd.DataFrame(elapsed, columns=["idx", "puzzle_elapsed_time", "puzzle_elapsed_ratio"])
    elapsed_df = elapsed_df.set_index("idx")
    df["puzzle_elapsed_time"] = elapsed_df["puzzle_elapsed_time"]
    df["puzzle_elapsed_ratio"] = elapsed_df["puzzle_elapsed_ratio"]

    return df


def compute_progress_momentum(df: pd.DataFrame, window_size: int = 4) -> pd.DataFrame:
    """
    Compute progress_momentum for each window: a smoothed trend [-1, 1] over
    the last `window_size` windows indicating whether the player is making
    real progress (+1) or degrading (-1).

    Positive signals (per window):
      - action_count > 0 (doing something)
      - time_since_action decreased from previous window (new meaningful action)
      - error_count == 0 (no errors)

    Negative signals:
      - time_since_action increased (no new action, drifting)
      - error_count > 0 (making mistakes)
      - action_count == 0 AND idle_time > 4.5 (completely idle)
      - puzzle_elapsed_ratio > 2.0 (over time → drags momentum down)

    The raw signal is computed per window, then smoothed with an exponentially
    weighted moving average over `window_size` windows.
    """
    import numpy as np

    df["_raw_momentum"] = 0.0
    df["progress_momentum"] = 0.0

    for pid in df["participant_id"].unique():
        mask = df["participant_id"] == pid
        p_df = df.loc[mask].sort_values("window_start")
        indices = p_df.index.tolist()

        raw_signals = []
        prev_time_since = None

        for i, idx in enumerate(indices):
            row = p_df.loc[idx]
            signal = 0.0

            action_count = row.get("action_count", 0) or 0
            time_since = row.get("time_since_action", 0) or 0
            error_count = row.get("error_count", 0) or 0
            idle_time = row.get("idle_time", 0) or 0
            elapsed_ratio = row.get("puzzle_elapsed_ratio", 0) or 0

            # Positive signals
            if action_count > 0:
                signal += 0.2
            if action_count >= 3:
                signal += 0.1
            if prev_time_since is not None and time_since < prev_time_since:
                # time_since decreased → a meaningful action reset it
                signal += 0.3
            if error_count == 0 and action_count > 0:
                signal += 0.1

            # Negative signals
            if prev_time_since is not None and time_since > prev_time_since + 4:
                signal -= 0.2  # drifting further from last action
            if error_count > 0:
                signal -= 0.25 * min(error_count, 3)
            if action_count == 0 and idle_time > 4.5:
                signal -= 0.3
            if elapsed_ratio > 2.0:
                signal -= min((elapsed_ratio - 2.0) * 0.1, 0.3)

            signal = max(-1.0, min(1.0, signal))
            raw_signals.append(signal)
            prev_time_since = time_since

            # Reset prev_time_since on puzzle change
            if i + 1 < len(indices):
                next_puzzle = p_df.loc[indices[i + 1]].get("puzzle_id", "")
                if next_puzzle != row.get("puzzle_id", ""):
                    prev_time_since = None

        # Exponentially weighted moving average
        alpha = 2.0 / (window_size + 1)
        smoothed = []
        ema = 0.0
        for j, s in enumerate(raw_signals):
            if j == 0:
                ema = s
            else:
                ema = alpha * s + (1 - alpha) * ema
            smoothed.append(round(max(-1.0, min(1.0, ema)), 4))

        # Write back
        for j, idx in enumerate(indices):
            df.at[idx, "_raw_momentum"] = raw_signals[j]
            df.at[idx, "progress_momentum"] = smoothed[j]

    return df


def run_pipeline(input_path: str, output_dir: str):
    # ---- Step 1: Load data ----
    df = load_csv(input_path)

    # ---- Step 1.5: Compute puzzle elapsed time + progress momentum ----
    print("[INFO] Computing puzzle elapsed time...")
    df = compute_puzzle_elapsed(df)
    print("[INFO] Computing progress momentum...")
    df = compute_progress_momentum(df)

    # ---- Step 2: Run single-row agents ----
    print("[INFO] Running AttentionAgent...")
    _unpack_agent(df, "attention", attention_agent)

    print("[INFO] Running ActionAgent...")
    _unpack_agent(df, "action", action_agent)

    print("[INFO] Running PerformanceAgent...")
    _unpack_agent(df, "performance", performance_agent)

    print("[INFO] Running PopulationAgent...")
    _unpack_agent(df, "population", population_agent)

    # ---- Step 3: Run TemporalAgent (needs prior agent outputs) ----
    print("[INFO] Running TemporalAgent...")
    state_columns = {
        "attention": "attention_label",
        "performance": "performance_label",
    }
    temporal_results = pd.Series([
        temporal_agent(idx, df, state_columns) for idx in df.index
    ])
    _unpack_agent(df, "temporal", None, apply_args=temporal_results)

    # ---- Step 4: Negotiation ----
    print("[INFO] Running negotiation layer...")
    df = run_negotiation(df)

    # ---- Step 5: Support suggestions ----
    print("[INFO] Running support layer...")
    df = run_support(df)

    # ---- Step 6: Save outputs ----
    os.makedirs(output_dir, exist_ok=True)

    agent_out = os.path.join(output_dir, "agent_outputs.csv")
    disagree_out = os.path.join(output_dir, "disagreement_summary.csv")
    support_out = os.path.join(output_dir, "support_summary.csv")

    df.to_csv(agent_out, index=False)
    print(f"[INFO] Saved agent outputs -> {agent_out}")

    # Disagreement summary
    disagree_cols = [
        "participant_id", "puzzle_id", "window_start",
        "attention_label", "attention_confidence",
        "action_label", "action_confidence",
        "performance_label", "performance_confidence",
        "temporal_label", "temporal_confidence",
        "disagreement_type", "disagreement_intensity", "dominant_tension",
        "n_contradictions", "n_constructive", "confidence_spread",
    ]
    disagree_cols = [c for c in disagree_cols if c in df.columns]
    df[disagree_cols].to_csv(disagree_out, index=False)
    print(f"[INFO] Saved disagreement summary -> {disagree_out}")

    # Support summary
    support_cols = [
        "participant_id", "puzzle_id", "window_start",
        "dominant_tension", "suggested_support", "support_confidence",
        "support_rationale", "support_category",
    ]
    support_cols = [c for c in support_cols if c in df.columns]
    df[support_cols].to_csv(support_out, index=False)
    print(f"[INFO] Saved support summary -> {support_out}")

    # ---- Summary statistics ----
    print("\n--- Pipeline Summary ---")
    print(f"Total windows: {len(df)}")
    for agent in ["attention", "action", "performance", "temporal", "population"]:
        print(f"\n{agent.title()} Agent:")
        print(f"  Labels: {df[f'{agent}_label'].value_counts().to_dict()}")
        print(f"  Avg confidence: {df[f'{agent}_confidence'].mean():.3f}")

    print(f"\nDisagreement types:\n{df['disagreement_type'].value_counts().to_string()}")
    print(f"\nDominant tensions:\n{df['dominant_tension'].value_counts().to_string()}")
    print(f"\nMean disagreement intensity: {df['disagreement_intensity'].mean():.3f}")

    print(f"\nSupport suggestions:\n{df['suggested_support'].value_counts().to_string()}")
    print(f"\nSupport categories:\n{df['support_category'].value_counts().to_string()}")
    print("--- Done ---")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.output)
