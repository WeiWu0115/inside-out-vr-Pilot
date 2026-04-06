"""
Ablation study: test different agent configurations against facilitator benchmark.

Configurations:
  1. behavioral_only  — only game log features (no eye tracking)
  2. gaze_only        — only gaze agents (no game logs)
  3. v4_full          — 3 gaze + 1 behavioral + temporal (current V4)
  4. v3_main          — original V3 architecture (read from main branch outputs)

Each config runs the full pipeline and computes facilitator benchmark metrics.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from load_data import load_csv
from agents import fixation_agent, semantics_agent, motor_agent, behavioral_agent, temporal_agent
from config import TEMPORAL
from negotiation import compute_disagreement, CONTRADICTIONS, CONSTRUCTIVE_PAIRS
from support import suggest_support, PromptAgent


def _unpack(df, prefix, func):
    results = df.apply(func, axis=1)
    df[f"{prefix}_label"] = results.apply(lambda r: r["label"])
    df[f"{prefix}_confidence"] = results.apply(lambda r: r["confidence"])
    df[f"{prefix}_reasoning"] = results.apply(lambda r: r.get("reasoning", ""))
    return df


def _dummy_agent(label="unknown", conf=0.0):
    """Return a dummy agent that always outputs the same label."""
    def agent(row):
        return {"label": label, "confidence": conf, "reasoning": "disabled", "all_scores": {}}
    return agent


def _run_temporal(df, state_cols):
    results = []
    for idx in df.index:
        results.append(temporal_agent(idx, df, state_cols))
    series = pd.Series(results)
    df["temporal_label"] = series.apply(lambda r: r["label"])
    df["temporal_confidence"] = series.apply(lambda r: r["confidence"])
    df["temporal_reasoning"] = series.apply(lambda r: r.get("reasoning", ""))
    return df


def _run_support(df):
    """Run support layer with stateful prompt agent."""
    base_results = df.apply(suggest_support, axis=1)
    final_results = []
    prompt_agents = {}

    for idx in df.index:
        row = df.loc[idx]
        pid = row.get("participant_id", 0)
        if pid not in prompt_agents:
            prompt_agents[pid] = PromptAgent()
        base = base_results[idx]
        final = prompt_agents[pid].decide(base, row)
        final_results.append(final)

    series = pd.Series(final_results)
    df["support_category"] = series.apply(lambda s: s["category"])
    return df


def run_config(df, config_name):
    """Run a specific agent configuration and return the dataframe with support_category."""
    df = df.copy()

    if config_name == "v4_full":
        # All agents active
        _unpack(df, "fixation", fixation_agent)
        _unpack(df, "semantics", semantics_agent)
        _unpack(df, "motor", motor_agent)
        _unpack(df, "action", behavioral_agent)
        # Backward compat aliases
        df["attention_label"] = df["fixation_label"]
        df["attention_confidence"] = df["fixation_confidence"]
        df["performance_label"] = df["action_label"]
        df["performance_confidence"] = df["action_confidence"]
        _run_temporal(df, {"fixation": "fixation_label", "semantics": "semantics_label", "behavioral": "action_label"})

    elif config_name == "gaze_only":
        # Only gaze agents, no behavioral
        _unpack(df, "fixation", fixation_agent)
        _unpack(df, "semantics", semantics_agent)
        _unpack(df, "motor", motor_agent)
        _unpack(df, "action", _dummy_agent("unknown", 0.0))
        df["attention_label"] = df["fixation_label"]
        df["attention_confidence"] = df["fixation_confidence"]
        df["performance_label"] = "unknown"
        df["performance_confidence"] = 0.0
        _run_temporal(df, {"fixation": "fixation_label", "semantics": "semantics_label"})

    elif config_name == "behavioral_only":
        # Only game log agent, no gaze
        _unpack(df, "fixation", _dummy_agent("unknown", 0.0))
        _unpack(df, "semantics", _dummy_agent("unknown", 0.0))
        _unpack(df, "motor", _dummy_agent("unknown", 0.0))
        _unpack(df, "action", behavioral_agent)
        df["attention_label"] = "unknown"
        df["attention_confidence"] = 0.0
        df["performance_label"] = df["action_label"]
        df["performance_confidence"] = df["action_confidence"]
        _run_temporal(df, {"behavioral": "action_label"})

    else:
        raise ValueError(f"Unknown config: {config_name}")

    # Run negotiation
    from negotiation import run_negotiation
    df = run_negotiation(df)

    # Run support
    df = _run_support(df)

    return df


def evaluate_config(df, fac_windows, prompts_df, expert_df, config_name):
    """Evaluate a config against facilitator benchmark. Returns metrics dict."""
    # Map IO support_category to io_cat
    df["io_cat"] = df["support_category"].replace({"consensus_intervene": "intervene"})

    # Merge with facilitator
    merged = df[["participant_id", "window_start", "io_cat"]].copy()
    merged["window_start"] = (merged["window_start"] // 5 * 5).round(2)

    fac = fac_windows[["participant_id", "window_start", "facilitator_cat"]].copy()
    fac["window_start"] = fac["window_start"].round(2)

    merged = merged.merge(fac, on=["participant_id", "window_start"], how="inner")
    merged["facilitator_cat"] = merged["facilitator_cat"].fillna("watch")

    # Add expert_cat for temporal_tolerance_analysis
    exp = expert_df[["participant_id", "window_start", "expert_cat"]].copy()
    merged = merged.merge(exp, on=["participant_id", "window_start"], how="left")
    merged["expert_cat"] = merged["expert_cat"].fillna("watch")

    n = len(merged)
    if n == 0:
        return {"config": config_name, "n_windows": 0}

    # Distribution
    io_watch = (merged["io_cat"] == "watch").sum()
    io_probe = (merged["io_cat"] == "probe").sum()
    io_intervene = (merged["io_cat"] == "intervene").sum()

    # Binary detection
    fac_pos = merged["facilitator_cat"] != "watch"
    io_pos = merged["io_cat"] != "watch"
    tp = (fac_pos & io_pos).sum()
    fp = (~fac_pos & io_pos).sum()
    fn = (fac_pos & ~io_pos).sum()
    tn = (~fac_pos & ~io_pos).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Temporal tolerance ±15s
    from facilitator_benchmark import temporal_tolerance_analysis
    tol_results, tol_detail = temporal_tolerance_analysis(prompts_df, merged, tolerances=(15,))
    tol15 = {}
    if tol_results is not None and len(tol_results) > 0:
        r = tol_results.iloc[0]
        tol15 = {
            "tol15_recall": round(float(r["io_recall"]), 4),
            "tol15_precision": round(float(r["io_precision"]), 4),
            "tol15_f1": round(float(r["io_f1"]), 4),
        }

    return {
        "config": config_name,
        "n_windows": n,
        "watch": io_watch,
        "probe": io_probe,
        "intervene": io_intervene,
        "active_rate": round((io_probe + io_intervene) / n, 4),
        "binary_precision": round(precision, 4),
        "binary_recall": round(recall, 4),
        "binary_f1": round(f1, 4),
        **tol15,
    }


def main():
    print("=== Ablation Study ===\n")

    # Load data
    df = load_csv("data/windows_enhanced.csv")

    # Compute puzzle elapsed
    from pipeline import compute_puzzle_elapsed
    df = compute_puzzle_elapsed(df)

    # Load facilitator data
    from facilitator_benchmark import load_game_starts, load_facilitator_prompts, assign_facilitator_to_windows, build_expert_windows
    game_starts = load_game_starts()
    prompts_df = load_facilitator_prompts(game_starts)

    expert_df = build_expert_windows()
    all_windows = expert_df[["participant_id", "window_start"]].copy()
    fac_windows = assign_facilitator_to_windows(prompts_df, all_windows)

    # Run configs
    configs = ["behavioral_only", "gaze_only", "v4_full"]
    results = []

    for config in configs:
        print(f"\n--- Running {config} ---")
        config_df = run_config(df, config)

        # Show distribution
        cats = config_df["support_category"].value_counts()
        print(f"  Distribution: {cats.to_dict()}")

        metrics = evaluate_config(config_df, fac_windows, prompts_df, expert_df, config)
        results.append(metrics)
        print(f"  Binary: P={metrics['binary_precision']:.3f} R={metrics['binary_recall']:.3f} F1={metrics['binary_f1']:.3f}")
        if "tol15_f1" in metrics:
            print(f"  ±15s:   P={metrics['tol15_precision']:.3f} R={metrics['tol15_recall']:.3f} F1={metrics['tol15_f1']:.3f}")

    # Summary table
    print("\n\n=== ABLATION RESULTS ===\n")
    results_df = pd.DataFrame(results)

    # Add known results
    results_df = pd.concat([results_df, pd.DataFrame([
        {"config": "rule_based", "tol15_recall": 0.377, "tol15_precision": 0.321, "tol15_f1": 0.347,
         "active_rate": 0.047, "binary_f1": 0.083},
        {"config": "v3_main", "tol15_recall": 0.921, "tol15_precision": 0.371, "tol15_f1": 0.529,
         "active_rate": 0.425, "binary_f1": 0.349},
        {"config": "theory_partitioned", "tol15_recall": 0.887, "tol15_precision": 0.379, "tol15_f1": 0.531,
         "active_rate": None, "binary_f1": None},
    ])], ignore_index=True)

    # Order
    order = ["rule_based", "behavioral_only", "gaze_only", "v4_full", "v3_main", "theory_partitioned"]
    results_df["config"] = pd.Categorical(results_df["config"], categories=order, ordered=True)
    results_df = results_df.sort_values("config").reset_index(drop=True)

    print(results_df[["config", "active_rate", "tol15_recall", "tol15_precision", "tol15_f1", "binary_f1"]].to_string(index=False))

    # Save
    os.makedirs("outputs", exist_ok=True)
    results_df.to_csv("outputs/ablation_results.csv", index=False)
    print("\nSaved to outputs/ablation_results.csv")


if __name__ == "__main__":
    main()
