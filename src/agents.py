"""
Four interpretive agents, each reading a different slice of the feature space.

Each agent takes a row (pd.Series) and returns a string label.
The TemporalAgent additionally requires context from neighboring rows.
"""

import pandas as pd
from load_data import safe_get
from config import ATTENTION, ACTION, PERFORMANCE, TEMPORAL


# ---------------------------------------------------------------------------
# 1. AttentionAgent
# ---------------------------------------------------------------------------

def attention_agent(row: pd.Series) -> str:
    """Interpret attentional state from gaze features."""
    entropy = safe_get(row, "gaze_entropy")
    clue_ratio = safe_get(row, "clue_ratio")
    switch_rate = safe_get(row, "switch_rate")
    action_count = safe_get(row, "action_count")

    if entropy is None or clue_ratio is None:
        return "ambiguous"

    # Locked: very high clue fixation with almost no action
    if (clue_ratio >= ATTENTION["clue_ratio_very_high"]
            and action_count is not None
            and action_count <= ATTENTION["action_count_low"]):
        return "locked"

    # Focused: low entropy + high clue ratio
    if (entropy <= ATTENTION["entropy_low"]
            and clue_ratio >= ATTENTION["clue_ratio_high"]):
        return "focused"

    # Searching: high entropy + high switch rate
    if (entropy >= ATTENTION["entropy_high"]
            and switch_rate is not None
            and switch_rate >= ATTENTION["switch_rate_high"]):
        return "searching"

    # Fallback: use entropy as a rough discriminator
    if entropy <= ATTENTION["entropy_low"]:
        return "focused"
    if entropy >= ATTENTION["entropy_high"]:
        return "searching"

    return "ambiguous"


# ---------------------------------------------------------------------------
# 2. ActionAgent
# ---------------------------------------------------------------------------

def action_agent(row: pd.Series) -> str:
    """Interpret behavioral engagement from action features."""
    action_count = safe_get(row, "action_count")
    idle_time = safe_get(row, "idle_time")
    time_since = safe_get(row, "time_since_action")

    if action_count is None:
        return "unknown"

    # Active: many actions, little idle time
    if action_count >= ACTION["action_count_high"]:
        if idle_time is not None and idle_time <= ACTION["idle_time_low"]:
            return "active"
        if idle_time is None:
            return "active"

    # Inactive: very few actions + high idle time
    if action_count <= ACTION["action_count_low"]:
        if idle_time is not None and idle_time >= ACTION["idle_time_high"]:
            return "inactive"
        if time_since is not None and time_since >= ACTION["time_since_action_moderate"]:
            return "inactive"

    # Hesitant: moderate activity with noticeable delays
    if (time_since is not None
            and time_since >= ACTION["time_since_action_moderate"]):
        return "hesitant"
    if (idle_time is not None
            and idle_time >= ACTION["idle_time_low"]
            and action_count > ACTION["action_count_low"]):
        return "hesitant"

    # Default for moderate action without clear delay signals
    if action_count >= ACTION["action_count_high"]:
        return "active"

    return "unknown"


# ---------------------------------------------------------------------------
# 3. PerformanceAgent
# ---------------------------------------------------------------------------

def performance_agent(row: pd.Series) -> str:
    """Interpret task performance from error and progress features."""
    error_count = safe_get(row, "error_count")
    puzzle_active = safe_get(row, "puzzle_active")
    action_count = safe_get(row, "action_count")

    if puzzle_active is None and error_count is None:
        return "unknown"

    # Failing: high errors regardless of other signals
    if error_count is not None and error_count >= PERFORMANCE["error_count_high"]:
        return "failing"

    # Progressing: ongoing actions with no/low errors (active puzzle or not)
    if (action_count is not None
            and action_count > PERFORMANCE["action_count_low"]
            and (error_count is None or error_count < PERFORMANCE["error_count_high"])):
        return "progressing"

    # Stalled: no meaningful action
    if (action_count is not None
            and action_count <= PERFORMANCE["action_count_low"]):
        return "stalled"

    return "unknown"


# ---------------------------------------------------------------------------
# 4. TemporalAgent
# ---------------------------------------------------------------------------

def temporal_agent(
    current_idx: int,
    df: pd.DataFrame,
    state_columns: dict,
) -> str:
    """
    Interpret temporal dynamics by looking at neighboring rows
    from the same participant.

    Parameters
    ----------
    current_idx : int
        Index of the current row in df.
    df : pd.DataFrame
        Full dataframe (must already contain agent state columns).
    state_columns : dict
        Mapping like {"attention": "attention_state", "performance": "performance_state", ...}
    """
    row = df.loc[current_idx]
    pid = safe_get(row, "participant_id")
    if pid is None:
        return "unknown"

    # Get rows for this participant, sorted by window_start
    participant_df = df[df["participant_id"] == pid].sort_values("window_start")
    indices = participant_df.index.tolist()

    if current_idx not in indices:
        return "unknown"

    pos = indices.index(current_idx)
    window = TEMPORAL["persistence_window"]

    # Not enough history to judge temporal patterns
    if pos < 1:
        return "transient"

    # Collect recent states (up to `window` preceding rows + current)
    start = max(0, pos - window + 1)
    recent = participant_df.iloc[start:pos + 1]

    # Check persistence: same attention AND performance state for >= window rows
    att_col = state_columns.get("attention", "attention_state")
    perf_col = state_columns.get("performance", "performance_state")

    if len(recent) >= window:
        if att_col in recent.columns and recent[att_col].nunique() == 1:
            if perf_col in recent.columns and recent[perf_col].nunique() == 1:
                # Check for looping: persistent in a failing/stalled state
                perf_val = recent[perf_col].iloc[-1]
                if perf_val in TEMPORAL["looping_keywords"]:
                    return "looping"
                return "persistent"

    # Check for looping even with shorter window
    if len(recent) >= 2 and perf_col in recent.columns:
        perf_vals = recent[perf_col].tolist()
        if all(v in TEMPORAL["looping_keywords"] for v in perf_vals):
            return "looping"

    return "transient"
