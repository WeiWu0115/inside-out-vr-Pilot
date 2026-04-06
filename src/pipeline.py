"""
V4 Pipeline: gaze-dominant multi-agent interpretation.

Agents: fixation, semantics, motor (all gaze) + behavioral (game logs) + temporal.

Usage:
    python src/pipeline.py
    python src/pipeline.py --input data/windows_enhanced.csv --output outputs/
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import INPUT_CSV, OUTPUT_DIR
from load_data import load_csv
from agents import fixation_agent, semantics_agent, motor_agent, behavioral_agent, temporal_agent
from negotiation import run_negotiation
from support import run_support


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze-focused multi-agent VR pipeline")
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
    df[f"{col_prefix}_reasoning"] = results.apply(lambda r: r.get("reasoning", ""))
    return df


def compute_puzzle_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    """Compute puzzle elapsed time and ratio vs population median."""
    PUZZLE_MEDIAN_DURATION = {
        "Spoke Puzzle: Amount of Protein": 135,
        "Spoke Puzzle: Pasta in Sauce": 90,
        "Spoke Puzzle: Water Amount": 230,
        "Spoke Puzzle: Amount of Sunlight": 160,
        "Hub Puzzle: Cooking Pot": 605,
        "Transition": 600,
    }

    elapsed = []
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


def run_pipeline(input_path: str, output_dir: str):
    # ---- Step 1: Load data ----
    df = load_csv(input_path)

    # ---- Step 1.5: Compute puzzle elapsed time ----
    print("[INFO] Computing puzzle elapsed time...")
    df = compute_puzzle_elapsed(df)

    # ---- Step 2: Run gaze agents ----
    print("[INFO] Running FixationAgent...")
    _unpack_agent(df, "fixation", fixation_agent)

    print("[INFO] Running SemanticsAgent...")
    _unpack_agent(df, "semantics", semantics_agent)

    print("[INFO] Running GazeMotorAgent...")
    _unpack_agent(df, "motor", motor_agent)

    # ---- Step 3: Run behavioral agent (game logs only) ----
    print("[INFO] Running BehavioralAgent...")
    _unpack_agent(df, "action", behavioral_agent)

    # Backward compat: copy fixation → attention, action → performance for support.py
    df["attention_label"] = df["fixation_label"]
    df["attention_confidence"] = df["fixation_confidence"]
    df["performance_label"] = df["action_label"]
    df["performance_confidence"] = df["action_confidence"]

    # ---- Step 4: Run TemporalAgent ----
    print("[INFO] Running TemporalAgent...")
    state_columns = {
        "fixation": "fixation_label",
        "semantics": "semantics_label",
        "behavioral": "action_label",
    }
    temporal_results = pd.Series([
        temporal_agent(idx, df, state_columns) for idx in df.index
    ])
    _unpack_agent(df, "temporal", None, apply_args=temporal_results)

    # ---- Step 5: Negotiation ----
    print("[INFO] Running negotiation layer...")
    df = run_negotiation(df)

    # ---- Step 6: Support suggestions ----
    print("[INFO] Running support layer...")
    df = run_support(df)

    # ---- Step 7: Save outputs ----
    os.makedirs(output_dir, exist_ok=True)

    agent_out = os.path.join(output_dir, "agent_outputs.csv")
    df.to_csv(agent_out, index=False)
    print(f"[INFO] Saved agent outputs -> {agent_out}")

    # ---- Summary statistics ----
    print("\n--- Pipeline Summary ---")
    print(f"Total windows: {len(df)}")
    for agent in ["fixation", "semantics", "motor", "action", "temporal"]:
        col = f"{agent}_label"
        if col in df.columns:
            print(f"\n{agent.title()} Agent:")
            print(f"  Labels: {df[col].value_counts().to_dict()}")
            print(f"  Avg confidence: {df[f'{agent}_confidence'].mean():.3f}")

    print(f"\nDisagreement types:\n{df['disagreement_type'].value_counts().to_string()}")
    print(f"\nTop tensions:\n{df['dominant_tension'].value_counts().head(15).to_string()}")
    print(f"\nMean disagreement intensity: {df['disagreement_intensity'].mean():.3f}")
    print(f"\nSupport categories:\n{df['support_category'].value_counts().to_string()}")
    print("--- Done ---")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.output)
