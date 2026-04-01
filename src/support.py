"""
Support layer: maps disagreement patterns to suggested interventions.
"""

import pandas as pd

SUPPORT_MAP = {
    "focused_but_stuck": "procedural_hint",
    "searching_without_grounding": "spatial_hint",
    "searching_and_hesitant": "encouragement_and_spatial_hint",
    "searching_but_progressing": "wait",
    "active_but_unguided": "light_guidance",
    "productive_struggle": "wait",
    "locked_and_idle": "reorientation_prompt",
    "progressing_but_ambiguous": "monitor",
    "no_clear_pattern": "none",
}


def suggest_support(pattern: str) -> str:
    """Return a support action for the given disagreement pattern."""
    return SUPPORT_MAP.get(pattern, "none")


def run_support(df: pd.DataFrame) -> pd.DataFrame:
    """Add suggested_support column based on disagreement_pattern."""
    df["suggested_support"] = df["disagreement_pattern"].apply(suggest_support)
    return df
