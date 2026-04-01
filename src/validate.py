"""
Validation script: run the pipeline on real data, then inspect distributions
and flag suspicious patterns.

Usage:
    python3 src/validate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# ── Step 0: Prepare real data ──────────────────────────────────────────────

RAW_PATH = "batch_output/all_windows_with_clusters.csv"
MAPPED_PATH = "data/windows.csv"

print("=" * 60)
print("STEP 0: Preparing real data")
print("=" * 60)

raw = pd.read_csv(RAW_PATH)
print(f"Loaded {len(raw)} rows from {RAW_PATH}")
print(f"Columns: {list(raw.columns)}")

# Map column names to what the pipeline expects
mapped = raw.rename(columns={
    "player_id": "participant_id",
    "t_start": "window_start",
    "puzzle_phase": "puzzle_id",
})
mapped.to_csv(MAPPED_PATH, index=False)
print(f"Saved mapped data -> {MAPPED_PATH}\n")

# ── Step 1: Run the pipeline ──────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Running pipeline on real data")
print("=" * 60)

from pipeline import run_pipeline
run_pipeline(MAPPED_PATH, "outputs")

# ── Step 2: Load outputs and validate ─────────────────────────────────────

df = pd.read_csv("outputs/agent_outputs.csv")
print()

STATE_COLS = ["attention_state", "action_state", "performance_state", "temporal_state"]
PATTERN_COLS = ["disagreement_pattern", "suggested_support"]
DOMINANCE_THRESHOLD = 0.70

# ── Step 3: Frequency tables ─────────────────────────────────────────────

print("=" * 60)
print("STEP 2: State frequencies")
print("=" * 60)

for col in STATE_COLS + PATTERN_COLS:
    counts = df[col].value_counts()
    pcts = df[col].value_counts(normalize=True).mul(100).round(1)
    table = pd.DataFrame({"count": counts, "pct": pcts})
    print(f"\n── {col} ──")
    print(table.to_string())

# ── Step 4: Flag suspicious distributions ────────────────────────────────

print()
print("=" * 60)
print("STEP 3: Suspicious distribution flags")
print("=" * 60)

flags = []

for col in STATE_COLS:
    top_pct = df[col].value_counts(normalize=True).iloc[0]
    top_val = df[col].value_counts().index[0]
    if top_pct > DOMINANCE_THRESHOLD:
        flags.append(
            f"  [FLAG] {col}: '{top_val}' dominates at {top_pct:.1%} "
            f"({df[col].value_counts().iloc[0]}/{len(df)} rows)"
        )

# Check no_clear_pattern dominance
ncp_pct = (df["disagreement_pattern"] == "no_clear_pattern").mean()
if ncp_pct > DOMINANCE_THRESHOLD:
    flags.append(
        f"  [FLAG] disagreement_pattern: 'no_clear_pattern' at {ncp_pct:.1%} — "
        f"most rows unmatched by negotiation rules"
    )

# Check none support dominance
none_pct = (df["suggested_support"] == "none").mean()
if none_pct > DOMINANCE_THRESHOLD:
    flags.append(
        f"  [FLAG] suggested_support: 'none' at {none_pct:.1%} — "
        f"pipeline rarely suggests interventions"
    )

# Check unknown dominance per agent
for col in STATE_COLS:
    unk_pct = (df[col] == "unknown").mean()
    if unk_pct > 0.3:
        flags.append(
            f"  [FLAG] {col}: 'unknown' at {unk_pct:.1%} — "
            f"agent cannot classify many rows"
        )

if flags:
    print(f"\nFound {len(flags)} issue(s):\n")
    for f in flags:
        print(f)
else:
    print("\nNo suspicious distributions found.")

# ── Step 5: Sample rows per disagreement pattern ─────────────────────────

print()
print("=" * 60)
print("STEP 4: Sample rows per disagreement pattern")
print("=" * 60)

INPUT_COLS = [
    "participant_id", "puzzle_id", "window_start",
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time", "time_since_action",
    "error_count", "puzzle_active",
]
AGENT_COLS = STATE_COLS + ["disagreement_score", "disagreement_pattern", "suggested_support"]
DISPLAY_COLS = [c for c in INPUT_COLS + AGENT_COLS if c in df.columns]

patterns = df["disagreement_pattern"].unique()
for pat in sorted(patterns):
    subset = df[df["disagreement_pattern"] == pat]
    n_sample = min(5, len(subset))
    sample = subset.sample(n=n_sample, random_state=42)
    print(f"\n── {pat} ({len(subset)} total rows, showing {n_sample}) ──")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(sample[DISPLAY_COLS].to_string(index=False))

# ── Step 6: Input feature distributions for context ──────────────────────

print()
print("=" * 60)
print("STEP 5: Input feature summary statistics")
print("=" * 60)

FEATURE_COLS = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time", "time_since_action",
    "error_count",
]
existing = [c for c in FEATURE_COLS if c in df.columns]
desc = df[existing].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(4)
print()
print(desc.to_string())

# ── Step 7: Threshold analysis ───────────────────────────────────────────

print()
print("=" * 60)
print("STEP 6: Threshold analysis & recommendations")
print("=" * 60)

from config import ATTENTION, ACTION, PERFORMANCE

recommendations = []

# Analyze gaze_entropy distribution vs thresholds
if "gaze_entropy" in df.columns:
    ge = df["gaze_entropy"].dropna()
    pct_below_low = (ge <= ATTENTION["entropy_low"]).mean()
    pct_above_high = (ge >= ATTENTION["entropy_high"]).mean()
    pct_middle = 1 - pct_below_low - pct_above_high
    print(f"\ngaze_entropy: min={ge.min():.3f}, median={ge.median():.3f}, max={ge.max():.3f}")
    print(f"  <= {ATTENTION['entropy_low']} (low threshold):  {pct_below_low:.1%}")
    print(f"  >= {ATTENTION['entropy_high']} (high threshold): {pct_above_high:.1%}")
    print(f"  middle (unclassified):         {pct_middle:.1%}")
    if pct_below_low < 0.05:
        q25 = ge.quantile(0.25)
        recommendations.append(
            f"  -> entropy_low={ATTENTION['entropy_low']} captures only {pct_below_low:.1%}. "
            f"P25={q25:.3f}. Consider raising to ~{q25:.2f}"
        )
    if pct_above_high < 0.05:
        q75 = ge.quantile(0.75)
        recommendations.append(
            f"  -> entropy_high={ATTENTION['entropy_high']} captures only {pct_above_high:.1%}. "
            f"P75={q75:.3f}. Consider lowering to ~{q75:.2f}"
        )
    if pct_middle > 0.5:
        recommendations.append(
            f"  -> {pct_middle:.1%} of rows fall between thresholds (entropy middle zone). "
            f"Consider narrowing the gap."
        )

# Analyze clue_ratio vs thresholds
if "clue_ratio" in df.columns:
    cr = df["clue_ratio"].dropna()
    pct_high = (cr >= ATTENTION["clue_ratio_high"]).mean()
    print(f"\nclue_ratio: min={cr.min():.3f}, median={cr.median():.3f}, max={cr.max():.3f}")
    print(f"  >= {ATTENTION['clue_ratio_high']} (high threshold): {pct_high:.1%}")
    if pct_high < 0.05:
        q75 = cr.quantile(0.75)
        recommendations.append(
            f"  -> clue_ratio_high={ATTENTION['clue_ratio_high']} captures only {pct_high:.1%}. "
            f"P75={q75:.3f}. Consider lowering to ~{q75:.2f}"
        )
    if pct_high > 0.90:
        q75 = cr.quantile(0.75)
        recommendations.append(
            f"  -> clue_ratio_high={ATTENTION['clue_ratio_high']} captures {pct_high:.1%} — too permissive. "
            f"P75={q75:.3f}. Consider raising."
        )

# Analyze switch_rate vs thresholds
if "switch_rate" in df.columns:
    sr = df["switch_rate"].dropna()
    pct_high = (sr >= ATTENTION["switch_rate_high"]).mean()
    print(f"\nswitch_rate: min={sr.min():.3f}, median={sr.median():.3f}, max={sr.max():.3f}")
    print(f"  >= {ATTENTION['switch_rate_high']} (high threshold): {pct_high:.1%}")
    if pct_high < 0.05:
        q75 = sr.quantile(0.75)
        recommendations.append(
            f"  -> switch_rate_high={ATTENTION['switch_rate_high']} captures only {pct_high:.1%}. "
            f"P75={q75:.3f}. Consider lowering to ~{q75:.2f}"
        )

# Analyze action_count
if "action_count" in df.columns:
    ac = df["action_count"].dropna()
    pct_high = (ac >= ACTION["action_count_high"]).mean()
    pct_low = (ac <= ACTION["action_count_low"]).mean()
    print(f"\naction_count: min={ac.min():.0f}, median={ac.median():.0f}, max={ac.max():.0f}")
    print(f"  >= {ACTION['action_count_high']} (high threshold): {pct_high:.1%}")
    print(f"  <= {ACTION['action_count_low']} (low threshold):  {pct_low:.1%}")
    if pct_low > 0.85:
        q50 = ac.quantile(0.5)
        recommendations.append(
            f"  -> action_count_low={ACTION['action_count_low']} captures {pct_low:.1%}. "
            f"Median={q50:.0f}. Most rows look inactive. Consider lowering to 0."
        )

# Analyze idle_time
if "idle_time" in df.columns:
    it = df["idle_time"].dropna()
    pct_high = (it >= ACTION["idle_time_high"]).mean()
    pct_low = (it <= ACTION["idle_time_low"]).mean()
    print(f"\nidle_time: min={it.min():.2f}, median={it.median():.2f}, max={it.max():.2f}")
    print(f"  >= {ACTION['idle_time_high']} (high threshold): {pct_high:.1%}")
    print(f"  <= {ACTION['idle_time_low']} (low threshold):  {pct_low:.1%}")

# Analyze error_count
if "error_count" in df.columns:
    ec = df["error_count"].dropna()
    pct_high = (ec >= PERFORMANCE["error_count_high"]).mean()
    print(f"\nerror_count: min={ec.min():.0f}, median={ec.median():.0f}, max={ec.max():.0f}")
    print(f"  >= {PERFORMANCE['error_count_high']} (high threshold): {pct_high:.1%}")
    if pct_high < 0.02:
        q90 = ec.quantile(0.9)
        recommendations.append(
            f"  -> error_count_high={PERFORMANCE['error_count_high']} captures only {pct_high:.1%}. "
            f"P90={q90:.0f}. Consider lowering to {max(1, int(q90))}."
        )

# Print recommendations
print()
if recommendations:
    print(f"RECOMMENDED THRESHOLD ADJUSTMENTS ({len(recommendations)}):\n")
    for r in recommendations:
        print(r)
else:
    print("No threshold adjustments needed — distributions look reasonable.")

print()
print("=" * 60)
print("Validation complete.")
print("=" * 60)
