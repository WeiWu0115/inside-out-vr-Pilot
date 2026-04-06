"""
Compare Expert Rule Engine vs Inside Out Multi-Agent System.

Generates a comparison report showing where the two systems agree,
where they disagree, and what those disagreements reveal.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np


def load_both():
    agent_df = pd.read_csv("outputs/agent_outputs.csv")
    expert_df = pd.read_csv("outputs/expert_engine_outputs.csv")

    # Merge on participant_id + window_start
    merged = agent_df.merge(
        expert_df,
        on=["participant_id", "window_start"],
        how="inner",
    )
    return merged


def map_to_action_category(row):
    """Map both systems to comparable categories: intervene / probe / watch."""
    # Inside Out
    io_cat = row.get("support_category", "watch")

    # Expert: R/V/Special prompts → probe, E prompt → intervene
    expert_action = row.get("expert_action", "NONE")
    expert_prompt_type = str(row.get("expert_prompt_type", ""))

    if expert_action == "PROMPT":
        if expert_prompt_type == "E":
            expert_cat = "intervene"
        else:
            expert_cat = "probe"  # R, V, SpecialA/B/C
    else:
        expert_cat = "watch"

    # Normalize IO categories
    io_normalized = {
        "consensus_intervene": "intervene",
        "probe": "probe",
        "watch": "watch",
    }.get(io_cat, "watch")

    return pd.Series({
        "io_category": io_normalized,
        "expert_category": expert_cat,
    })


def run_comparison():
    merged = load_both()
    print(f"Merged dataset: {len(merged)} windows")

    # Map to comparable categories
    cats = merged.apply(map_to_action_category, axis=1)
    merged["io_category"] = cats["io_category"]
    merged["expert_category"] = cats["expert_category"]

    # ── 1. Overall agreement ──
    print("\n" + "=" * 70)
    print("1. OVERALL AGREEMENT")
    print("=" * 70)

    agree = (merged["io_category"] == merged["expert_category"]).sum()
    print(f"\nExact agreement: {agree}/{len(merged)} ({agree/len(merged):.1%})")

    ct = pd.crosstab(
        merged["expert_category"],
        merged["io_category"],
        margins=True,
    )
    print(f"\nCross-tabulation:\n{ct.to_string()}")

    # ── 2. When expert says INTERVENE, what does Inside Out say? ──
    print("\n" + "=" * 70)
    print("2. WHEN EXPERT SAYS INTERVENE")
    print("=" * 70)

    expert_intervene = merged[merged["expert_category"] == "intervene"]
    print(f"\nExpert triggers intervention: {len(expert_intervene)} windows")
    if len(expert_intervene) > 0:
        io_response = expert_intervene["io_category"].value_counts()
        for cat, n in io_response.items():
            pct = n / len(expert_intervene) * 100
            print(f"  Inside Out says '{cat}': {n} ({pct:.0f}%)")

        # What tensions does IO see when expert intervenes?
        print(f"\n  IO tensions during expert interventions:")
        for tension, n in expert_intervene["dominant_tension"].value_counts().head(5).items():
            print(f"    {tension}: {n}")

        # What expert rules triggered?
        print(f"\n  Expert rules that triggered:")
        for rule, n in expert_intervene["expert_rule"].value_counts().items():
            print(f"    {rule}: {n}")

    # ── 3. When Inside Out says INTERVENE, what does expert say? ──
    print("\n" + "=" * 70)
    print("3. WHEN INSIDE OUT SAYS INTERVENE")
    print("=" * 70)

    io_intervene = merged[merged["io_category"] == "intervene"]
    print(f"\nInside Out triggers intervention: {len(io_intervene)} windows")
    if len(io_intervene) > 0:
        expert_response = io_intervene["expert_category"].value_counts()
        for cat, n in expert_response.items():
            pct = n / len(io_intervene) * 100
            print(f"  Expert says '{cat}': {n} ({pct:.0f}%)")

    # ── 4. Disagreement analysis ──
    print("\n" + "=" * 70)
    print("4. DISAGREEMENT ANALYSIS")
    print("=" * 70)

    # IO says watch but expert says intervene
    missed = merged[
        (merged["io_category"] == "watch") &
        (merged["expert_category"] == "intervene")
    ]
    print(f"\nExpert intervenes but IO watches: {len(missed)} windows")
    if len(missed) > 0:
        print(f"  IO tensions in these windows:")
        for t, n in missed["dominant_tension"].value_counts().head(5).items():
            print(f"    {t}: {n}")
        print(f"  Expert rules:")
        for r, n in missed["expert_rule"].value_counts().head(5).items():
            print(f"    {r}: {n}")

    # IO says intervene but expert says watch
    false_alarm = merged[
        (merged["io_category"] == "intervene") &
        (merged["expert_category"] == "watch")
    ]
    print(f"\nIO intervenes but expert watches: {len(false_alarm)} windows")
    if len(false_alarm) > 0:
        print(f"  IO tensions in these windows:")
        for t, n in false_alarm["dominant_tension"].value_counts().head(5).items():
            print(f"    {t}: {n}")
        print(f"  Expert states:")
        for s, n in false_alarm["expert_state"].value_counts().items():
            print(f"    {s}: {n}")

    # IO probes — expert doesn't have this category
    probes = merged[merged["io_category"] == "probe"]
    print(f"\nIO probes (no expert equivalent): {len(probes)} windows")
    if len(probes) > 0:
        print(f"  Expert would say:")
        for cat, n in probes["expert_category"].value_counts().items():
            pct = n / len(probes) * 100
            print(f"    {cat}: {n} ({pct:.0f}%)")

    # ── 5. Per-player comparison ──
    print("\n" + "=" * 70)
    print("5. PER-PLAYER COMPARISON")
    print("=" * 70)

    print(f"\n{'Player':>8} {'Expert':>8} {'IO':>8} {'IO':>8} {'IO':>8} {'Agree':>8}")
    print(f"{'':>8} {'prompts':>8} {'interv':>8} {'probes':>8} {'watch':>8} {'rate':>8}")
    print("-" * 56)

    for pid in sorted(merged["participant_id"].unique()):
        pf = merged[merged["participant_id"] == pid]
        n_expert = (pf["expert_category"] == "intervene").sum()
        n_io_int = (pf["io_category"] == "intervene").sum()
        n_io_probe = (pf["io_category"] == "probe").sum()
        n_io_watch = (pf["io_category"] == "watch").sum()
        agree_rate = (pf["io_category"] == pf["expert_category"]).mean()
        print(f"P{pid:>6} {n_expert:>8} {n_io_int:>8} {n_io_probe:>8} {n_io_watch:>8} {agree_rate:>7.0%}")

    # ── 6. Key insight: what IO sees that expert doesn't ──
    print("\n" + "=" * 70)
    print("6. WHAT INSIDE OUT SEES THAT THE EXPERT DOESN'T")
    print("=" * 70)

    # Expert has 2 states relevant to difficulty: STUCK and EXPLORE
    # IO has nuanced tensions. Show the diversity within expert's STUCK state.
    stuck_windows = merged[merged["expert_state"] == "STUCK"]
    if len(stuck_windows) > 0:
        print(f"\nWithin expert's STUCK state ({len(stuck_windows)} windows):")
        print(f"  IO sees these tensions:")
        for t, n in stuck_windows["dominant_tension"].value_counts().head(7).items():
            pct = n / len(stuck_windows) * 100
            print(f"    {t}: {n} ({pct:.0f}%)")
        print(f"  IO response categories:")
        for cat, n in stuck_windows["io_category"].value_counts().items():
            pct = n / len(stuck_windows) * 100
            print(f"    {cat}: {n} ({pct:.0f}%)")

    explore_windows = merged[merged["expert_state"] == "EXPLORE"]
    if len(explore_windows) > 0:
        print(f"\nWithin expert's EXPLORE state ({len(explore_windows)} windows):")
        print(f"  IO sees these tensions:")
        for t, n in explore_windows["dominant_tension"].value_counts().head(7).items():
            pct = n / len(explore_windows) * 100
            print(f"    {t}: {n} ({pct:.0f}%)")
        print(f"  IO response categories:")
        for cat, n in explore_windows["io_category"].value_counts().items():
            pct = n / len(explore_windows) * 100
            print(f"    {cat}: {n} ({pct:.0f}%)")

    # Save merged comparison
    out_cols = [
        "participant_id", "puzzle_id", "window_start",
        "expert_state", "expert_action", "expert_prompt_type", "expert_rule",
        "attention_label", "attention_confidence",
        "action_label", "action_confidence",
        "performance_label", "performance_confidence",
        "population_label", "population_confidence",
        "disagreement_type", "dominant_tension", "disagreement_intensity",
        "suggested_support", "support_category", "support_rationale",
        "io_category", "expert_category",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv("outputs/comparison_correct.csv", index=False)
    print(f"\nSaved comparison to outputs/comparison_correct.csv")


if __name__ == "__main__":
    run_comparison()
