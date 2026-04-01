"""
Negotiation layer: computes disagreement across agent states
and assigns interpretive patterns.
"""

import pandas as pd


def compute_disagreement_score(row: pd.Series) -> int:
    """
    Count the number of distinct non-unknown agent states.
    A higher number means more divergent interpretations.
    Range: 0 to 4.
    """
    states = [
        row.get("attention_state"),
        row.get("action_state"),
        row.get("performance_state"),
        row.get("temporal_state"),
    ]
    known = [s for s in states if s is not None and s not in ("unknown", "ambiguous")]
    return len(set(known))


def assign_disagreement_pattern(row: pd.Series) -> str:
    """
    Match agent states against interpretive patterns.
    Returns the first matching pattern name, or 'no_clear_pattern'.
    """
    att = row.get("attention_state", "unknown")
    act = row.get("action_state", "unknown")
    perf = row.get("performance_state", "unknown")
    temp = row.get("temporal_state", "unknown")

    # focused_but_stuck
    if (att == "focused"
            and act in ("inactive", "hesitant")
            and perf in ("stalled", "failing")):
        return "focused_but_stuck"

    # searching_without_grounding
    if (att == "searching"
            and act == "inactive"
            and perf == "stalled"):
        return "searching_without_grounding"

    # searching_and_hesitant: looking around but acting tentatively, no progress
    if (att == "searching"
            and act == "hesitant"
            and perf == "stalled"):
        return "searching_and_hesitant"

    # searching_but_progressing: scattered attention yet making progress
    if (att == "searching"
            and perf == "progressing"):
        return "searching_but_progressing"

    # active_but_unguided
    if (att == "searching"
            and act == "active"
            and perf == "failing"):
        return "active_but_unguided"

    # productive_struggle
    if (att == "focused"
            and act in ("active", "hesitant")
            and perf == "stalled"
            and temp == "transient"):
        return "productive_struggle"

    # locked_and_idle: fixated on clue area, doing nothing
    if (att == "locked"
            and act == "inactive"
            and perf == "stalled"):
        return "locked_and_idle"

    # progressing_but_ambiguous: making progress despite unclear attention
    if (perf == "progressing"
            and att in ("unknown", "ambiguous")
            and act in ("hesitant", "active", "inactive")
            and temp in ("transient", "persistent")):
        return "progressing_but_ambiguous"

    return "no_clear_pattern"


def run_negotiation(df: pd.DataFrame) -> pd.DataFrame:
    """Add disagreement columns to the dataframe."""
    df["disagreement_score"] = df.apply(compute_disagreement_score, axis=1)
    df["disagreement_pattern"] = df.apply(assign_disagreement_pattern, axis=1)
    return df
