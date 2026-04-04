"""
V2 Agents: Clean perceptual boundaries — each agent owns exclusive features.

Each agent returns a dict:
    {
        "label": str,           # interpreted state
        "confidence": float,    # 0.0 to 1.0
        "evidence": dict,       # feature values that drove the interpretation
        "reasoning": str,       # human-readable explanation
    }

V2 changes from V1:
  - Attention → Perceptual (same features, no more action_count leak)
  - Action → Behavioral (action_count + idle_time only, no time_since_action)
  - Performance → Progress (time_since_action + puzzle_elapsed_ratio, no action_count)
  - NEW: Spatial Agent (stub — uses puzzle_id area heuristic for now)
  - Temporal unchanged
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
# 1. PerceptualAgent (was AttentionAgent)
#    Exclusive features: gaze_entropy, clue_ratio, switch_rate
#    Question: "What is the player looking at?"
# ---------------------------------------------------------------------------

def perceptual_agent(row: pd.Series) -> dict:
    entropy = safe_get(row, "gaze_entropy")
    clue_ratio = safe_get(row, "clue_ratio")
    switch_rate = safe_get(row, "switch_rate")

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

    scores = {}

    # Locked: very high clue fixation
    # V2: no longer checks action_count — that's Behavioral's domain.
    # Instead, locked = extremely high clue_ratio + low entropy (visual fixation).
    if clue_ratio >= ATTENTION["clue_ratio_very_high"] and entropy <= ATTENTION["entropy_low"]:
        locked_conf = (
            _linear_scale(clue_ratio, ATTENTION["clue_ratio_high"], 1.0) * 0.6
            + (1.0 - _linear_scale(entropy, 0, ATTENTION["entropy_low"])) * 0.4
        )
        scores["locked"] = _clamp(locked_conf, 0.3, 0.95)

    # Focused: low entropy + meaningful clue engagement
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

    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.15:
            best_conf *= 0.7

    reasons = {
        "locked": f"High clue fixation ({clue_ratio:.2f}) + low entropy ({entropy:.2f})",
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


# Backward compatibility alias
attention_agent = perceptual_agent


# ---------------------------------------------------------------------------
# 2. BehavioralAgent (was ActionAgent)
#    Exclusive features: action_count, idle_time, error_count
#    Question: "What is the player doing right now?"
#    V2: time_since_action moved to Progress Agent.
#        error_count moved here from Performance (it's instantaneous behavior).
# ---------------------------------------------------------------------------

def behavioral_agent(row: pd.Series) -> dict:
    action_count = safe_get(row, "action_count")
    idle_time = safe_get(row, "idle_time")
    error_count = safe_get(row, "error_count")

    if action_count is None:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Missing behavioral data",
        }

    evidence = {
        "action_count": action_count,
        "idle_time": idle_time,
        "error_count": error_count,
    }

    scores = {}

    # Active: many actions, little idle
    active_conf = _linear_scale(action_count, 0, ACTION["action_count_high"] * 1.5) * 0.6
    if idle_time is not None:
        active_conf += (1.0 - _linear_scale(idle_time, 0, 5.0)) * 0.4
    scores["active"] = _clamp(active_conf, 0.1, 0.95)

    # Inactive: no actions + high idle
    inactive_conf = (1.0 - _linear_scale(action_count, 0, ACTION["action_count_high"])) * 0.5
    if idle_time is not None:
        inactive_conf += _linear_scale(idle_time, ACTION["idle_time_low"], 5.0) * 0.5
    scores["inactive"] = _clamp(inactive_conf, 0.1, 0.95)

    # Hesitant: moderate activity with delays
    hesitant_conf = 0.3
    if action_count > 0 and action_count < ACTION["action_count_high"]:
        hesitant_conf += 0.25
    if idle_time is not None and idle_time > ACTION["idle_time_low"]:
        hesitant_conf += 0.2
    scores["hesitant"] = _clamp(hesitant_conf, 0.1, 0.90)

    # Failing: errors present (V2: moved here from Performance)
    if error_count is not None and error_count >= PERFORMANCE["error_count_high"]:
        fail_conf = _linear_scale(error_count, 0, 3) * 0.7
        fail_conf = max(fail_conf, 0.5)
        scores["failing"] = _clamp(fail_conf, 0.3, 0.95)

    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.1:
            best_conf *= 0.75

    reasons = {
        "active": f"High action count ({action_count}) with engagement",
        "inactive": f"Low/no actions, idle_time={idle_time:.1f}s" if idle_time else "No actions detected",
        "hesitant": f"Some actions ({action_count}) but with idle periods",
        "failing": f"Errors detected ({error_count}) during interaction",
    }

    return {
        "label": best_label,
        "confidence": round(_clamp(best_conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best_label, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# Backward compatibility alias
action_agent = behavioral_agent


# ---------------------------------------------------------------------------
# 3. ProgressAgent (was PerformanceAgent)
#    Exclusive features: time_since_action, puzzle_elapsed_ratio, puzzle_active
#    Question: "How is puzzle-solving progress going?"
#    V2: No longer reads action_count (Behavioral's domain).
#        Uses macro-temporal signals instead.
# ---------------------------------------------------------------------------

def progress_agent(row: pd.Series) -> dict:
    time_since = safe_get(row, "time_since_action")
    elapsed_ratio = safe_get(row, "puzzle_elapsed_ratio")
    puzzle_active = safe_get(row, "puzzle_active")

    if time_since is None and elapsed_ratio is None:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Missing progress data",
        }

    # Defaults
    if time_since is None:
        time_since = 0.0
    if elapsed_ratio is None:
        elapsed_ratio = 0.0

    evidence = {
        "time_since_action": time_since,
        "puzzle_elapsed_ratio": elapsed_ratio,
        "puzzle_active": puzzle_active,
    }

    scores = {}

    # Progressing: recent action + not over time
    prog_conf = 0.5
    # Recent action is a strong positive signal
    if time_since < ACTION["time_since_action_moderate"] * 0.5:
        prog_conf += 0.3
    elif time_since < ACTION["time_since_action_moderate"]:
        prog_conf += 0.15
    # Under median puzzle time is good
    if elapsed_ratio < 1.0:
        prog_conf += 0.1
    elif elapsed_ratio > 2.0:
        prog_conf -= 0.2
    scores["progressing"] = _clamp(prog_conf, 0.1, 0.95)

    # Stalled: long time since action
    stall_conf = 0.2
    if time_since > ACTION["time_since_action_moderate"]:
        stall_conf += _linear_scale(time_since,
                                     ACTION["time_since_action_moderate"],
                                     ACTION["time_since_action_moderate"] * 3) * 0.5
    if time_since > ACTION["time_since_action_moderate"] * 2:
        stall_conf += 0.15
    # Over time on puzzle boosts stall
    if elapsed_ratio > 1.5:
        stall_conf += _linear_scale(elapsed_ratio, 1.5, 3.0) * 0.2
    scores["stalled"] = _clamp(stall_conf, 0.1, 0.95)

    # Struggling: over time AND long since action — compound signal
    if elapsed_ratio > 1.5 and time_since > ACTION["time_since_action_moderate"]:
        struggle_conf = (
            _linear_scale(elapsed_ratio, 1.5, 3.0) * 0.4
            + _linear_scale(time_since,
                            ACTION["time_since_action_moderate"],
                            ACTION["time_since_action_moderate"] * 3) * 0.4
            + 0.2
        )
        scores["struggling"] = _clamp(struggle_conf, 0.3, 0.95)

    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.1:
            best_conf *= 0.75

    reasons = {
        "progressing": f"Recent activity ({time_since:.0f}s ago), puzzle at {elapsed_ratio:.1f}× median",
        "stalled": f"No action for {time_since:.0f}s, puzzle at {elapsed_ratio:.1f}× median",
        "struggling": f"Stalled ({time_since:.0f}s) AND over time ({elapsed_ratio:.1f}× median)",
    }

    return {
        "label": best_label,
        "confidence": round(_clamp(best_conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best_label, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# Backward compatibility alias
performance_agent = progress_agent


# ---------------------------------------------------------------------------
# 4. SpatialAgent (NEW in V2)
#    Exclusive features: puzzle_id (area heuristic), head_position (when available)
#    Question: "Where is the player and where are they going?"
#    NOTE: Stub implementation using puzzle_id. Full version needs head tracking.
# ---------------------------------------------------------------------------

# Puzzle area coordinates (approximate centers from heatmap data)
PUZZLE_AREAS = {
    "Spoke Puzzle: Amount of Protein": (-3.7, 11.9),
    "Spoke Puzzle: Pasta in Sauce": (-4.5, 14.1),
    "Spoke Puzzle: Water Amount": (-6.9, 17.2),
    "Spoke Puzzle: Amount of Sunlight": (-7.0, 16.7),
    "Hub Puzzle: Cooking Pot": (-5.5, 14.5),
}


def spatial_agent(row: pd.Series) -> dict:
    puzzle_id = safe_get(row, "puzzle_id")

    evidence = {"puzzle_id": puzzle_id}

    # Stub: without head tracking, use puzzle_id as a proxy
    if puzzle_id == "Transition":
        return {
            "label": "navigating",
            "confidence": 0.7,
            "evidence": evidence,
            "reasoning": "Between puzzles — player is navigating the environment",
        }

    if puzzle_id and puzzle_id in PUZZLE_AREAS:
        return {
            "label": "stationed",
            "confidence": 0.6,
            "evidence": evidence,
            "reasoning": f"Player is at {puzzle_id}",
        }

    return {
        "label": "unknown",
        "confidence": 0.3,
        "evidence": evidence,
        "reasoning": "Cannot determine spatial state without head tracking",
    }


# ---------------------------------------------------------------------------
# 5. TemporalAgent — interprets patterns over time
#    Reads: labels from Perceptual + Progress agents (sliding window)
#    No raw features — purely meta-agent.
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

    # V2: read perceptual + progress labels (was attention + performance)
    perc_col = state_columns.get("perceptual", state_columns.get("attention", "attention_label"))
    prog_col = state_columns.get("progress", state_columns.get("performance", "performance_label"))

    evidence = {"history_length": len(recent)}

    # Check persistence
    if len(recent) >= window and perc_col in recent.columns and prog_col in recent.columns:
        perc_stable = recent[perc_col].nunique() == 1
        prog_stable = recent[prog_col].nunique() == 1

        if perc_stable and prog_stable:
            prog_val = recent[prog_col].iloc[-1]
            if prog_val in TEMPORAL["looping_keywords"]:
                return {
                    "label": "looping",
                    "confidence": 0.85,
                    "evidence": {**evidence, "repeated_state": prog_val, "streak": len(recent)},
                    "reasoning": f"Stuck in '{prog_val}' for {len(recent)} consecutive windows",
                }
            return {
                "label": "persistent",
                "confidence": 0.75,
                "evidence": {**evidence, "stable_perceptual": recent[perc_col].iloc[-1], "streak": len(recent)},
                "reasoning": f"Stable state for {len(recent)} windows",
            }

    # Check looping even with shorter window
    if len(recent) >= 2 and prog_col in recent.columns:
        prog_vals = recent[prog_col].tolist()
        if all(v in TEMPORAL["looping_keywords"] for v in prog_vals):
            return {
                "label": "looping",
                "confidence": 0.65,
                "evidence": {**evidence, "pattern": prog_vals},
                "reasoning": f"Repeated difficulty states: {prog_vals}",
            }

    return {
        "label": "transient",
        "confidence": 0.4,
        "evidence": evidence,
        "reasoning": "No persistent pattern detected",
    }
