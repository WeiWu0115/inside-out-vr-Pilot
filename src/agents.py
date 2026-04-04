"""
V3 Agents: Label-flow architecture with stateful awareness.

Design principles:
  1. Each raw feature is owned by ONE primary agent
  2. Downstream agents read pre-interpreted LABELS, not raw features
  3. puzzle_elapsed_ratio integrated at agent level, not post-hoc
  4. New state: ineffective_progress (active but not making real progress)

Data flow:
  Raw features → Attention Agent  (gaze_entropy, clue_ratio, switch_rate)
  Raw features → Behavioral Agent (action_count, idle_time, error_count)
  Labels      → Progress Agent    (behavioral_label + time_since_action + puzzle_elapsed_ratio)
  Labels      → Temporal Agent    (attention_label + progress_label history)
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
#    Exclusive raw features: gaze_entropy, clue_ratio, switch_rate
# ---------------------------------------------------------------------------

def attention_agent(row: pd.Series) -> dict:
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

    # Locked: very high clue fixation + low entropy (pure visual fixation)
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


# ---------------------------------------------------------------------------
# 2. BehavioralAgent (was ActionAgent)
#    Exclusive raw features: action_count, idle_time, error_count
#    Outputs label consumed by Progress Agent via label flow.
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

    # Failing: errors during interaction
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


# Backward compat alias — column names stay "action_*" for app.py
action_agent = behavioral_agent


# ---------------------------------------------------------------------------
# 3. ProgressAgent (was PerformanceAgent)
#    LABEL FLOW: reads behavioral_label (not raw action_count)
#    Own raw features: time_since_action, puzzle_elapsed_ratio
#    Key addition: "ineffective_progress" state
# ---------------------------------------------------------------------------

def progress_agent(row: pd.Series) -> dict:
    # Label flow: read Behavioral Agent's pre-interpreted label
    behavioral_label = row.get("action_label", "unknown")
    behavioral_conf = row.get("action_confidence", 0.5)

    # Own exclusive raw features
    time_since = safe_get(row, "time_since_action") or 0.0
    elapsed_ratio = safe_get(row, "puzzle_elapsed_ratio") or 0.0
    puzzle_active = safe_get(row, "puzzle_active")

    evidence = {
        "behavioral_label": behavioral_label,
        "behavioral_confidence": behavioral_conf,
        "time_since_action": time_since,
        "puzzle_elapsed_ratio": elapsed_ratio,
    }

    scores = {}

    # --- Progressing: recent action + not over time ---
    prog_conf = 0.3
    if behavioral_label in ("active", "hesitant"):
        prog_conf += 0.25
    if time_since < ACTION["time_since_action_moderate"] * 0.5:
        prog_conf += 0.25
    elif time_since < ACTION["time_since_action_moderate"]:
        prog_conf += 0.1
    if elapsed_ratio < 1.0:
        prog_conf += 0.1
    # Penalize if over time even with activity
    if elapsed_ratio > 2.0:
        prog_conf -= 0.15
    if elapsed_ratio > 3.0:
        prog_conf -= 0.15
    scores["progressing"] = _clamp(prog_conf, 0.1, 0.95)

    # --- Ineffective Progress (NEW): active but not making real progress ---
    # Player is doing things but has been at this puzzle too long / too long since
    # meaningful action — the actions are likely exploratory or repetitive
    ineff_conf = 0.1
    if behavioral_label in ("active", "hesitant"):
        # Active but time_since_action is high → sporadic actions after long gap
        if time_since > ACTION["time_since_action_moderate"]:
            ineff_conf += _linear_scale(time_since,
                                         ACTION["time_since_action_moderate"],
                                         ACTION["time_since_action_moderate"] * 3) * 0.3
        # Active but over median puzzle time → struggling despite activity
        if elapsed_ratio > 1.5:
            ineff_conf += _linear_scale(elapsed_ratio, 1.5, 3.0) * 0.3
        # Failing behavior boosts ineffective
        if behavioral_label == "failing":
            ineff_conf += 0.2
        # Combined: high elapsed + high time_since = strong signal
        if elapsed_ratio > 2.0 and time_since > ACTION["time_since_action_moderate"]:
            ineff_conf += 0.15
    scores["ineffective_progress"] = _clamp(ineff_conf, 0.05, 0.95)

    # --- Stalled: behavioral inactive + macro signals ---
    stall_conf = 0.2
    if behavioral_label == "inactive":
        stall_conf += 0.35
    elif behavioral_label == "hesitant":
        stall_conf += 0.1
    if time_since > ACTION["time_since_action_moderate"]:
        stall_conf += _linear_scale(time_since,
                                     ACTION["time_since_action_moderate"],
                                     ACTION["time_since_action_moderate"] * 3) * 0.3
    if elapsed_ratio > 1.5:
        stall_conf += _linear_scale(elapsed_ratio, 1.5, 3.0) * 0.15
    if elapsed_ratio > 3.0:
        stall_conf += 0.1
    scores["stalled"] = _clamp(stall_conf, 0.1, 0.95)

    best_label = max(scores, key=scores.get)
    best_conf = scores[best_label]

    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1:
        gap = sorted_scores[0] - sorted_scores[1]
        if gap < 0.1:
            best_conf *= 0.75

    reasons = {
        "progressing": f"Behavioral={behavioral_label}, recent activity ({time_since:.0f}s ago), {elapsed_ratio:.1f}x median",
        "ineffective_progress": (
            f"Behavioral={behavioral_label} but progress doubtful: "
            f"{time_since:.0f}s since sustained action, {elapsed_ratio:.1f}x median puzzle time"
        ),
        "stalled": f"Behavioral={behavioral_label}, {time_since:.0f}s since action, {elapsed_ratio:.1f}x median",
    }

    return {
        "label": best_label,
        "confidence": round(_clamp(best_conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best_label, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# Backward compat alias
performance_agent = progress_agent


# ---------------------------------------------------------------------------
# 4. TemporalAgent — interprets patterns over time
#    V3: uses puzzle_elapsed_ratio to boost looping detection
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
    elapsed_ratio = safe_get(row, "puzzle_elapsed_ratio") or 0.0

    if pos < 1:
        # Even first window: if elapsed_ratio is very high, flag it
        if elapsed_ratio > 3.0:
            return {
                "label": "persistent",
                "confidence": 0.6,
                "evidence": {"history_length": 0, "puzzle_elapsed_ratio": elapsed_ratio},
                "reasoning": f"First window but puzzle already at {elapsed_ratio:.1f}x median",
            }
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

    evidence = {"history_length": len(recent), "puzzle_elapsed_ratio": elapsed_ratio}

    # Check persistence
    if len(recent) >= window and att_col in recent.columns and perf_col in recent.columns:
        att_stable = recent[att_col].nunique() == 1
        perf_stable = recent[perf_col].nunique() == 1

        if att_stable and perf_stable:
            perf_val = recent[perf_col].iloc[-1]
            if perf_val in TEMPORAL["looping_keywords"]:
                conf = 0.85
                # V3: boost if also over time on puzzle
                if elapsed_ratio > 2.0:
                    conf = min(conf + 0.1, 0.95)
                return {
                    "label": "looping",
                    "confidence": conf,
                    "evidence": {**evidence, "repeated_state": perf_val, "streak": len(recent)},
                    "reasoning": f"Stuck in '{perf_val}' for {len(recent)} windows (puzzle at {elapsed_ratio:.1f}x median)",
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
            conf = 0.65
            if elapsed_ratio > 2.0:
                conf = min(conf + 0.15, 0.85)
            return {
                "label": "looping",
                "confidence": conf,
                "evidence": {**evidence, "pattern": perf_vals},
                "reasoning": f"Repeated difficulty states: {perf_vals} (puzzle at {elapsed_ratio:.1f}x median)",
            }

    # V3: if no temporal pattern but elapsed_ratio is very high, flag as persistent
    if elapsed_ratio > 3.0:
        return {
            "label": "persistent",
            "confidence": 0.6,
            "evidence": evidence,
            "reasoning": f"No pattern but puzzle at {elapsed_ratio:.1f}x median — prolonged engagement",
        }

    return {
        "label": "transient",
        "confidence": 0.4,
        "evidence": evidence,
        "reasoning": "No persistent pattern detected",
    }
