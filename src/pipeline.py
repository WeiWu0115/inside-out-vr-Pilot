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


def run_pipeline(input_path: str, output_dir: str):
    # ---- Step 1: Load data ----
    df = load_csv(input_path)

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
