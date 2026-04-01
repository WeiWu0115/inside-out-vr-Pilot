"""
Main pipeline: load data -> run agents -> negotiate -> suggest support -> save.

Usage:
    python src/pipeline.py
    python src/pipeline.py --input data/windows.csv --output outputs/
"""

import os
import sys
import argparse

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import INPUT_CSV, OUTPUT_DIR, AGENT_OUTPUT_FILE, DISAGREEMENT_FILE, SUPPORT_FILE
from load_data import load_csv
from agents import attention_agent, action_agent, performance_agent, temporal_agent
from negotiation import run_negotiation
from support import run_support


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent VR user-state interpretation pipeline")
    parser.add_argument("--input", default=INPUT_CSV, help="Path to input CSV")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    return parser.parse_args()


def run_pipeline(input_path: str, output_dir: str):
    # ---- Step 1: Load data ----
    df = load_csv(input_path)

    # ---- Step 2: Run single-row agents ----
    print("[INFO] Running AttentionAgent...")
    df["attention_state"] = df.apply(attention_agent, axis=1)

    print("[INFO] Running ActionAgent...")
    df["action_state"] = df.apply(action_agent, axis=1)

    print("[INFO] Running PerformanceAgent...")
    df["performance_state"] = df.apply(performance_agent, axis=1)

    # ---- Step 3: Run TemporalAgent (needs prior agent outputs) ----
    print("[INFO] Running TemporalAgent...")
    state_columns = {
        "attention": "attention_state",
        "performance": "performance_state",
    }
    df["temporal_state"] = [
        temporal_agent(idx, df, state_columns) for idx in df.index
    ]

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

    # Full output: original columns + all derived columns
    df.to_csv(agent_out, index=False)
    print(f"[INFO] Saved agent outputs -> {agent_out}")

    # Disagreement summary
    disagree_cols = [
        "participant_id", "puzzle_id", "window_start",
        "attention_state", "action_state", "performance_state", "temporal_state",
        "disagreement_score", "disagreement_pattern",
    ]
    disagree_cols = [c for c in disagree_cols if c in df.columns]
    df[disagree_cols].to_csv(disagree_out, index=False)
    print(f"[INFO] Saved disagreement summary -> {disagree_out}")

    # Support summary
    support_cols = [
        "participant_id", "puzzle_id", "window_start",
        "disagreement_pattern", "suggested_support",
    ]
    support_cols = [c for c in support_cols if c in df.columns]
    df[support_cols].to_csv(support_out, index=False)
    print(f"[INFO] Saved support summary -> {support_out}")

    # ---- Summary statistics ----
    print("\n--- Pipeline Summary ---")
    print(f"Total windows: {len(df)}")
    print(f"\nAttention states:\n{df['attention_state'].value_counts().to_string()}")
    print(f"\nAction states:\n{df['action_state'].value_counts().to_string()}")
    print(f"\nPerformance states:\n{df['performance_state'].value_counts().to_string()}")
    print(f"\nTemporal states:\n{df['temporal_state'].value_counts().to_string()}")
    print(f"\nDisagreement patterns:\n{df['disagreement_pattern'].value_counts().to_string()}")
    print(f"\nSupport suggestions:\n{df['suggested_support'].value_counts().to_string()}")
    print("--- Done ---")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.input, args.output)
