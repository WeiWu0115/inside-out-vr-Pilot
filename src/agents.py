"""
Four interpretive agents, each reading a different slice of the feature space.

Each agent returns a dict:
    {
        "label": str,           # interpreted state
        "confidence": float,    # 0.0 to 1.0
        "evidence": dict,       # feature values that drove the interpretation
        "reasoning": str,       # human-readable explanation
    }
"""

import pandas as pd
import numpy as np
from load_data import safe_get
from config import ATTENTION, ACTION, PERFORMANCE, TEMPORAL


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _linear_scale(value, low, high):
    """Map value to 0-1 range between low and high."""
    if high == low:
        return 0.5
    return _clamp((value - low) / (high - low))


# ---------------------------------------------------------------------------
# 1. AttentionAgent — interprets gaze distribution
# ---------------------------------------------------------------------------

def attention_agent(row: pd.Series) -> dict:
    entropy = safe_get(row, "gaze_entropy")
    clue_ratio = safe_get(row, "clue_ratio")
    switch_rate = safe_get(row, "switch_rate")
    action_count = safe_get(row, "action_count")

    if entropy is None or clue_ratio is None:
        return {
            "label": "ambiguous",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Missing gaze data",
        }

    evidence = {
        "gaze_entropy": entropy,
        "clue_ratio": clue_ratio,
        "switch_rate": switch_rate,
    }

    # Score each possible interpretation
    scores = {}

    # Locked: very high clue fixation with almost no action
    if (clue_ratio >= ATTENTION["clue_ratio_very_high"]
            and action_count is not None
            and action_count <= ATTENTION["action_count_low"]):
        locked_conf = (
            _linear_scale(clue_ratio, ATTENTION["clue_ratio_high"], 1.0) * 0.6
            + (1.0 - _linear_scale(action_count, 0, ATTENTION["action_count_low"] + 2)) * 0.4
        )
        scores["locked"] = _clamp(locked_conf, 0.3, 0.95)

    # Focused: low entropy + high clue ratio
    focused_conf = (
        (1.0 - _linear_scale(entropy, 0, ATTENTION["entropy_high"])) * 0.5
        + _linear_scale(clue_ratio, 0, ATTENTION["clue_ratio_high"]) * 0.5
    )
    if entropy <= ATTENTION["entropy_high"]:
        scores["focused"] = _clamp(focused_conf, 0.2, 0.95)

    # Searching: high entropy + high switch rate
    search_conf = _linear_scale(entropy, ATTENTION["entropy_low"], ATTENTION["entropy_high"] * 1.5) * 0.5
    if switch_rate is not None:
        search_conf += _linear_scale(switch_rate, 0, ATTENTION["switch_rate_high"] * 1.2) * 0.5
    else:
        search_conf += 0.2
    if entropy >= ATTENTION["entropy_low"]:
        scores["searching"] = _clamp(search_conf, 0.2, 0.95)

    if not scores:
        return {
            "label": "ambiguous",
            "confidence": 0.2,
            "evidence": evidence,
            "reasoning": f"Entropy={entropy:.2f} in middle range, no clear pattern",
        }

    # Winner takes label, but we keep the full score distribution
    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]

    # Reduce confidence if second-best is close (genuine ambiguity)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.15:
            best_conf *= 0.7  # ambiguity penalty

    reasons = {
        "locked": f"High clue fixation ({clue_ratio:.2f}) with minimal action",
        "focused": f"Low entropy ({entropy:.2f}) with clue engagement ({clue_ratio:.2f})",
        "searching": f"High entropy ({entropy:.2f}), scanning environment",
    }

    return {
        "label": best_label,
        "confidence": round(_clamp(best_conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best_label, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# 2. ActionAgent — interprets behavioral engagement
# ---------------------------------------------------------------------------

def action_agent(row: pd.Series) -> dict:
    action_count = safe_get(row, "action_count")
    idle_time = safe_get(row, "idle_time")
    time_since = safe_get(row, "time_since_action")

    if action_count is None:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Missing action data",
        }

    evidence = {
        "action_count": action_count,
        "idle_time": idle_time,
        "time_since_action": time_since,
    }

    scores = {}

    # Active: many actions, little idle
    active_conf = _linear_scale(action_count, 0, ACTION["action_count_high"] * 1.5) * 0.6
    if idle_time is not None:
        active_conf += (1.0 - _linear_scale(idle_time, 0, 5.0)) * 0.4
    scores["active"] = _clamp(active_conf, 0.1, 0.95)

    # Inactive: no actions + high idle
    inactive_conf = (1.0 - _linear_scale(action_count, 0, ACTION["action_count_high"])) * 0.4
    if idle_time is not None:
        inactive_conf += _linear_scale(idle_time, ACTION["idle_time_low"], 5.0) * 0.3
    if time_since is not None:
        inactive_conf += _linear_scale(time_since, 0, ACTION["time_since_action_moderate"] * 2) * 0.3
    scores["inactive"] = _clamp(inactive_conf, 0.1, 0.95)

    # Hesitant: moderate activity with delays
    hesitant_conf = 0.3
    if action_count > 0 and action_count < ACTION["action_count_high"]:
        hesitant_conf += 0.2
    if time_since is not None and time_since > ACTION["time_since_action_moderate"] * 0.5:
        hesitant_conf += 0.2
    if idle_time is not None and idle_time > ACTION["idle_time_low"]:
        hesitant_conf += 0.15
    scores["hesitant"] = _clamp(hesitant_conf, 0.1, 0.90)

    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.1:
            best_conf *= 0.75

    reasons = {
        "active": f"High action count ({action_count}) with engagement",
        "inactive": f"Low/no actions, idle_time={idle_time:.1f}s" if idle_time else f"No actions detected",
        "hesitant": f"Some actions ({action_count}) but with delays",
    }

    return {
        "label": best_label,
        "confidence": round(_clamp(best_conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best_label, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# 3. PerformanceAgent — interprets task progress
# ---------------------------------------------------------------------------

def performance_agent(row: pd.Series) -> dict:
    error_count = safe_get(row, "error_count")
    puzzle_active = safe_get(row, "puzzle_active")
    action_count = safe_get(row, "action_count")
    time_since = safe_get(row, "time_since_action")

    if puzzle_active is None and error_count is None:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Missing performance data",
        }

    evidence = {
        "error_count": error_count,
        "puzzle_active": puzzle_active,
        "action_count": action_count,
        "time_since_action": time_since,
    }

    scores = {}

    # Failing: errors present
    if error_count is not None:
        fail_conf = _linear_scale(error_count, 0, 3) * 0.8
        if error_count >= PERFORMANCE["error_count_high"]:
            fail_conf = max(fail_conf, 0.6)
        scores["failing"] = _clamp(fail_conf, 0.05, 0.95)
    else:
        scores["failing"] = 0.05

    # Progressing: actions with no/few errors
    # Key fix: penalize "progressing" when time_since_action is high.
    # A few sporadic actions after 2+ minutes of inactivity is not real progress.
    prog_conf = 0.3
    if action_count is not None and action_count > 0:
        prog_conf += _linear_scale(action_count, 0, ACTION["action_count_high"]) * 0.4
    if error_count is not None and error_count == 0:
        prog_conf += 0.2
    # Penalty: if player hasn't acted for a long time, current actions are
    # likely exploratory attempts, not meaningful progress
    if time_since is not None and time_since > ACTION["time_since_action_moderate"]:
        penalty = _linear_scale(time_since,
                                ACTION["time_since_action_moderate"],
                                ACTION["time_since_action_moderate"] * 3) * 0.35
        prog_conf -= penalty
    scores["progressing"] = _clamp(prog_conf, 0.1, 0.95)

    # Stalled: no action, OR sporadic action after long inactivity
    stall_conf = 0.3
    if action_count is not None and action_count == 0:
        stall_conf += 0.4
    elif action_count is not None and action_count <= PERFORMANCE["action_count_low"]:
        stall_conf += 0.2
    # Boost stalled score when time_since_action is high even with some action
    if time_since is not None and time_since > ACTION["time_since_action_moderate"]:
        stall_boost = _linear_scale(time_since,
                                    ACTION["time_since_action_moderate"],
                                    ACTION["time_since_action_moderate"] * 3) * 0.25
        stall_conf += stall_boost
    scores["stalled"] = _clamp(stall_conf, 0.1, 0.95)

    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.1:
            best_conf *= 0.75

    reasons = {
        "failing": f"Errors detected ({error_count})",
        "progressing": f"Active ({action_count} actions) with few/no errors",
        "stalled": f"No meaningful actions in this window",
    }
    # Add context about long inactivity
    if time_since is not None and time_since > ACTION["time_since_action_moderate"]:
        if best_label == "progressing":
            reasons["progressing"] = (
                f"Some actions ({action_count}) but {time_since:.0f}s since last sustained activity"
            )
        elif best_label == "stalled":
            reasons["stalled"] = (
                f"Stalled: {time_since:.0f}s since last action"
                + (f", only {action_count} sporadic actions" if action_count and action_count > 0 else "")
            )

    return {
        "label": best_label,
        "confidence": round(_clamp(best_conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best_label, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# 4. TemporalAgent — interprets patterns over time
# ---------------------------------------------------------------------------

def temporal_agent(current_idx: int, df: pd.DataFrame, state_columns: dict) -> dict:
    row = df.loc[current_idx]
    pid = safe_get(row, "participant_id")
    if pid is None:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Missing participant ID",
        }

    participant_df = df[df["participant_id"] == pid].sort_values("window_start")
    indices = participant_df.index.tolist()

    if current_idx not in indices:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Index not found in participant data",
        }

    pos = indices.index(current_idx)
    window = TEMPORAL["persistence_window"]

    if pos < 1:
        return {
            "label": "transient",
            "confidence": 0.5,
            "evidence": {"history_length": 0},
            "reasoning": "First window, no history available",
        }

    start = max(0, pos - window + 1)
    recent = participant_df.iloc[start:pos + 1]

    att_col = state_columns.get("attention", "attention_label")
    perf_col = state_columns.get("performance", "performance_label")

    evidence = {"history_length": len(recent)}

    # Check persistence
    if len(recent) >= window and att_col in recent.columns and perf_col in recent.columns:
        att_stable = recent[att_col].nunique() == 1
        perf_stable = recent[perf_col].nunique() == 1

        if att_stable and perf_stable:
            perf_val = recent[perf_col].iloc[-1]
            if perf_val in TEMPORAL["looping_keywords"]:
                return {
                    "label": "looping",
                    "confidence": 0.85,
                    "evidence": {**evidence, "repeated_state": perf_val, "streak": len(recent)},
                    "reasoning": f"Stuck in '{perf_val}' for {len(recent)} consecutive windows",
                }
            return {
                "label": "persistent",
                "confidence": 0.75,
                "evidence": {**evidence, "stable_attention": recent[att_col].iloc[-1], "streak": len(recent)},
                "reasoning": f"Stable state for {len(recent)} windows",
            }

    # Check looping even with shorter window
    if len(recent) >= 2 and perf_col in recent.columns:
        perf_vals = recent[perf_col].tolist()
        if all(v in TEMPORAL["looping_keywords"] for v in perf_vals):
            return {
                "label": "looping",
                "confidence": 0.65,
                "evidence": {**evidence, "pattern": perf_vals},
                "reasoning": f"Repeated difficulty states: {perf_vals}",
            }

    return {
        "label": "transient",
        "confidence": 0.4,
        "evidence": evidence,
        "reasoning": "No persistent pattern detected",
    }
