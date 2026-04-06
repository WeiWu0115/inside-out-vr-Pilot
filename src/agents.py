"""
V4 Agents: Gaze-dominant architecture.

3 gaze agents + 1 behavioral agent + 1 temporal agent.
Each raw feature is owned by ONE agent — no feature sharing.

Data flow:
  Gaze features  → Fixation Agent     (fixation_count, duration_*, revisit_rate)
  Gaze features  → Semantics Agent    (target_entropy, clue/puzzle/env_dwell, puzzle_object_ratio, n_unique_targets)
  Gaze features  → Gaze-Motor Agent   (gaze_head_coupling, saccade_amplitude_*, gaze_dispersion)
  Game logs      → Behavioral Agent   (action_count, idle_time, error_count, time_since_action)
  Labels         → Temporal Agent     (history of above agent labels)
"""

import pandas as pd
import numpy as np
from load_data import safe_get
from config import FIXATION, SEMANTICS, MOTOR, ACTION, TEMPORAL


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _linear_scale(value, low, high):
    if high == low:
        return 0.5
    return _clamp((value - low) / (high - low))


# ---------------------------------------------------------------------------
# 1. FixationAgent — how is visual attention distributed?
#    Exclusive features: fixation_count, fixation_duration_mean/max/std, revisit_rate
# ---------------------------------------------------------------------------

def fixation_agent(row: pd.Series) -> dict:
    fix_count = safe_get(row, "fixation_count", 0)
    dur_mean = safe_get(row, "fixation_duration_mean", 0.0)
    dur_max = safe_get(row, "fixation_duration_max", 0.0)
    dur_std = safe_get(row, "fixation_duration_std", 0.0)
    revisit = safe_get(row, "revisit_rate", 0.0)

    evidence = {
        "fixation_count": fix_count,
        "fixation_duration_mean": dur_mean,
        "fixation_duration_max": dur_max,
        "fixation_duration_std": dur_std,
        "revisit_rate": revisit,
    }

    if fix_count == 0:
        return {"label": "no_data", "confidence": 0.0, "evidence": evidence,
                "reasoning": "No fixations detected", "all_scores": {}}

    scores = {}

    # Locked: very long single fixation, few fixations total
    if dur_max >= FIXATION["duration_max_locked"] or (dur_mean >= FIXATION["duration_very_long"] and fix_count <= FIXATION["count_low"]):
        locked_conf = (
            _linear_scale(dur_max, FIXATION["duration_long"], 5.0) * 0.5
            + (1.0 - _linear_scale(fix_count, 0, FIXATION["count_high"])) * 0.3
            + _linear_scale(dur_mean, FIXATION["duration_long"], 5.0) * 0.2
        )
        scores["locked"] = _clamp(locked_conf, 0.4, 0.95)

    # Focused: moderate fixation count, long mean duration, low revisit
    focused_conf = (
        _linear_scale(dur_mean, FIXATION["duration_short"], FIXATION["duration_very_long"]) * 0.4
        + (1.0 - _linear_scale(fix_count, FIXATION["count_low"], FIXATION["count_high"] * 1.5)) * 0.3
        + (1.0 - _linear_scale(revisit, 0, FIXATION["revisit_high"])) * 0.3
    )
    if dur_mean >= FIXATION["duration_short"]:
        scores["focused"] = _clamp(focused_conf, 0.2, 0.95)

    # Scanning: many short fixations, high fixation count
    scanning_conf = (
        _linear_scale(fix_count, FIXATION["count_low"], FIXATION["count_high"] * 1.5) * 0.4
        + (1.0 - _linear_scale(dur_mean, 0, FIXATION["duration_long"])) * 0.4
        + _linear_scale(dur_std, 0, 1.0) * 0.2
    )
    if fix_count >= FIXATION["count_low"]:
        scores["scanning"] = _clamp(scanning_conf, 0.2, 0.95)

    # Revisiting: high revisit rate — checking and rechecking
    if revisit >= FIXATION["revisit_low"]:
        revisit_conf = (
            _linear_scale(revisit, FIXATION["revisit_low"], 1.0) * 0.6
            + _linear_scale(fix_count, FIXATION["count_low"], FIXATION["count_high"]) * 0.4
        )
        scores["revisiting"] = _clamp(revisit_conf, 0.2, 0.90)

    if not scores:
        return {"label": "ambiguous", "confidence": 0.2, "evidence": evidence,
                "reasoning": "No clear fixation pattern", "all_scores": {}}

    best = max(scores, key=scores.get)
    conf = scores[best]

    # Reduce confidence if close competition
    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and (sorted_s[0] - sorted_s[1]) < 0.15:
        conf *= 0.7

    reasons = {
        "locked": f"Very long fixation (max={dur_max:.1f}s, mean={dur_mean:.1f}s), {fix_count} fixations",
        "focused": f"Sustained attention (mean={dur_mean:.1f}s), {fix_count} fixations, revisit={revisit:.0%}",
        "scanning": f"Rapid scanning ({fix_count} fixations, mean={dur_mean:.2f}s)",
        "revisiting": f"Frequently rechecking targets (revisit={revisit:.0%}, {fix_count} fixations)",
    }

    return {
        "label": best,
        "confidence": round(_clamp(conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# 2. GazeSemanticsAgent — what is the player looking at?
#    Exclusive features: gaze_target_entropy, clue_dwell, puzzle_dwell,
#                        env_dwell, puzzle_object_ratio, n_unique_targets
# ---------------------------------------------------------------------------

def semantics_agent(row: pd.Series) -> dict:
    entropy = safe_get(row, "gaze_target_entropy", 0.0)
    clue = safe_get(row, "clue_dwell", 0.0)
    puzzle = safe_get(row, "puzzle_dwell", 0.0)
    env = safe_get(row, "env_dwell", 0.0)
    ratio = safe_get(row, "puzzle_object_ratio", 0.0)
    n_targets = safe_get(row, "n_unique_targets", 0)

    evidence = {
        "gaze_target_entropy": entropy,
        "clue_dwell": clue,
        "puzzle_dwell": puzzle,
        "env_dwell": env,
        "puzzle_object_ratio": ratio,
        "n_unique_targets": n_targets,
    }

    scores = {}

    # Fixated on clue: very high clue dwell
    if clue >= SEMANTICS["clue_dwell_high"]:
        clue_conf = (
            _linear_scale(clue, SEMANTICS["clue_dwell_high"], 1.0) * 0.6
            + (1.0 - _linear_scale(entropy, 0, SEMANTICS["target_entropy_high"])) * 0.4
        )
        scores["fixated_on_clue"] = _clamp(clue_conf, 0.3, 0.95)

    # Task-focused: high puzzle_object_ratio, moderate entropy
    if ratio >= SEMANTICS["puzzle_ratio_low"]:
        task_conf = (
            _linear_scale(ratio, SEMANTICS["puzzle_ratio_low"], 1.0) * 0.5
            + (1.0 - _linear_scale(env, 0, 1.0)) * 0.3
            + _linear_scale(puzzle + clue, 0, 1.0) * 0.2
        )
        scores["task_focused"] = _clamp(task_conf, 0.2, 0.95)

    # Environmental scanning: high env_dwell, high entropy, many targets
    env_conf = (
        _linear_scale(env, SEMANTICS["env_dwell_high"], 1.0) * 0.4
        + _linear_scale(entropy, SEMANTICS["target_entropy_low"], SEMANTICS["target_entropy_high"]) * 0.3
        + _linear_scale(n_targets, SEMANTICS["n_targets_low"], SEMANTICS["n_targets_high"]) * 0.3
    )
    if env >= 0.5:
        scores["environmental_scanning"] = _clamp(env_conf, 0.2, 0.95)

    # Unfocused: high entropy + low task relevance
    if entropy >= SEMANTICS["target_entropy_high"] and ratio < SEMANTICS["puzzle_ratio_high"]:
        unfocused_conf = (
            _linear_scale(entropy, SEMANTICS["target_entropy_high"], 4.0) * 0.5
            + (1.0 - _linear_scale(ratio, 0, SEMANTICS["puzzle_ratio_high"])) * 0.5
        )
        scores["unfocused"] = _clamp(unfocused_conf, 0.3, 0.90)

    if not scores:
        return {"label": "ambiguous", "confidence": 0.2, "evidence": evidence,
                "reasoning": f"Entropy={entropy:.2f}, clue={clue:.0%}, env={env:.0%}", "all_scores": {}}

    best = max(scores, key=scores.get)
    conf = scores[best]

    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and (sorted_s[0] - sorted_s[1]) < 0.15:
        conf *= 0.7

    reasons = {
        "fixated_on_clue": f"High clue engagement ({clue:.0%} dwell), entropy={entropy:.2f}",
        "task_focused": f"Looking at task objects ({ratio:.0%}), clue={clue:.0%}, puzzle={puzzle:.0%}",
        "environmental_scanning": f"Mostly viewing environment ({env:.0%}), {n_targets} targets, entropy={entropy:.2f}",
        "unfocused": f"High entropy ({entropy:.2f}) with low task focus ({ratio:.0%})",
    }

    return {
        "label": best,
        "confidence": round(_clamp(conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# 3. GazeMotorAgent — is gaze purposeful or passive?
#    Exclusive features: gaze_head_coupling, saccade_amplitude_mean/max, gaze_dispersion
# ---------------------------------------------------------------------------

def motor_agent(row: pd.Series) -> dict:
    coupling = safe_get(row, "gaze_head_coupling", 0.0)
    sacc_mean = safe_get(row, "saccade_amplitude_mean", 0.0)
    sacc_max = safe_get(row, "saccade_amplitude_max", 0.0)
    dispersion = safe_get(row, "gaze_dispersion", 0.0)

    evidence = {
        "gaze_head_coupling": coupling,
        "saccade_amplitude_mean": sacc_mean,
        "saccade_amplitude_max": sacc_max,
        "gaze_dispersion": dispersion,
    }

    scores = {}

    # Purposeful: low coupling (eyes explore independently), moderate saccades
    purposeful_conf = (
        (1.0 - _linear_scale(coupling, 0, MOTOR["coupling_high"])) * 0.4
        + _linear_scale(sacc_mean, MOTOR["saccade_amp_low"], MOTOR["saccade_amp_high"]) * 0.3
        + _linear_scale(dispersion, MOTOR["dispersion_low"], MOTOR["dispersion_high"]) * 0.3
    )
    scores["purposeful"] = _clamp(purposeful_conf, 0.15, 0.95)

    # Passive scanning: high coupling (eyes follow head), low independent saccades
    passive_conf = (
        _linear_scale(coupling, MOTOR["coupling_high"], 1.0) * 0.5
        + (1.0 - _linear_scale(sacc_mean, 0, MOTOR["saccade_amp_high"])) * 0.3
        + _linear_scale(dispersion, MOTOR["dispersion_low"], MOTOR["dispersion_high"]) * 0.2
    )
    if coupling >= MOTOR["coupling_low"]:
        scores["passive_scanning"] = _clamp(passive_conf, 0.15, 0.95)

    # Erratic: very high saccade amplitude, high dispersion
    if sacc_mean >= MOTOR["saccade_amp_high"] or sacc_max >= 25.0:
        erratic_conf = (
            _linear_scale(sacc_mean, MOTOR["saccade_amp_high"], MOTOR["saccade_amp_very_high"] * 1.5) * 0.4
            + _linear_scale(dispersion, MOTOR["dispersion_high"], 1.0) * 0.3
            + _linear_scale(sacc_max, 20, 40) * 0.3
        )
        scores["erratic"] = _clamp(erratic_conf, 0.2, 0.90)

    # Concentrated: low dispersion, low saccade amplitude — fixated in one spot
    if dispersion <= MOTOR["dispersion_high"] and sacc_mean <= MOTOR["saccade_amp_high"]:
        conc_conf = (
            (1.0 - _linear_scale(dispersion, 0, MOTOR["dispersion_high"])) * 0.5
            + (1.0 - _linear_scale(sacc_mean, 0, MOTOR["saccade_amp_high"])) * 0.5
        )
        scores["concentrated"] = _clamp(conc_conf, 0.15, 0.90)

    best = max(scores, key=scores.get)
    conf = scores[best]

    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and (sorted_s[0] - sorted_s[1]) < 0.12:
        conf *= 0.75

    reasons = {
        "purposeful": f"Independent eye movement (coupling={coupling:.2f}), moderate saccades ({sacc_mean:.1f}°)",
        "passive_scanning": f"Eyes follow head (coupling={coupling:.2f}), low independent saccades",
        "erratic": f"Large saccades (mean={sacc_mean:.1f}°, max={sacc_max:.0f}°), high dispersion ({dispersion:.2f})",
        "concentrated": f"Tight gaze (dispersion={dispersion:.2f}), small saccades ({sacc_mean:.1f}°)",
    }

    return {
        "label": best,
        "confidence": round(_clamp(conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# ---------------------------------------------------------------------------
# 4. BehavioralAgent — what is the player doing?
#    Features: action_count, idle_time, error_count, time_since_action
#    + gaze-action coupling: informed_ratio, blind_ratio, blind_wrong
#    The coupling features let us distinguish action_count=3 (informed)
#    from action_count=3 (blind trial-and-error).
# ---------------------------------------------------------------------------

def behavioral_agent(row: pd.Series) -> dict:
    action_count = safe_get(row, "action_count")
    idle_time = safe_get(row, "idle_time")
    error_count = safe_get(row, "error_count")
    time_since = safe_get(row, "time_since_action", 0.0)

    # Gaze-action coupling features
    informed_ratio = safe_get(row, "informed_ratio", 0.0)
    blind_ratio = safe_get(row, "blind_ratio", 0.0)
    blind_wrong = safe_get(row, "blind_wrong", 0)
    n_actions = safe_get(row, "n_actions", 0)  # actions with coupling data

    if action_count is None:
        return {"label": "unknown", "confidence": 0.0, "evidence": {},
                "reasoning": "Missing behavioral data", "all_scores": {}}

    evidence = {
        "action_count": action_count,
        "idle_time": idle_time,
        "error_count": error_count,
        "time_since_action": time_since,
        "informed_ratio": informed_ratio,
        "blind_ratio": blind_ratio,
    }

    scores = {}
    has_coupling = n_actions > 0

    # Active: many actions, little idle
    active_conf = _linear_scale(action_count, 0, ACTION["action_count_high"] * 1.5) * 0.4
    if idle_time is not None:
        active_conf += (1.0 - _linear_scale(idle_time, 0, 5.0)) * 0.2
    active_conf += (1.0 - _linear_scale(time_since, 0, ACTION["time_since_action_moderate"])) * 0.15
    # Boost if actions are informed (looked before acting)
    if has_coupling:
        active_conf += informed_ratio * 0.25
    else:
        active_conf += 0.1  # no coupling data, assume moderate
    scores["active"] = _clamp(active_conf, 0.1, 0.95)

    # Inactive
    inactive_conf = (1.0 - _linear_scale(action_count, 0, ACTION["action_count_high"])) * 0.4
    if idle_time is not None:
        inactive_conf += _linear_scale(idle_time, ACTION["idle_time_low"], 5.0) * 0.3
    inactive_conf += _linear_scale(time_since, ACTION["time_since_action_moderate"],
                                    ACTION["time_since_action_moderate"] * 3) * 0.3
    scores["inactive"] = _clamp(inactive_conf, 0.1, 0.95)

    # Hesitant
    hesitant_conf = 0.25
    if 0 < action_count < ACTION["action_count_high"]:
        hesitant_conf += 0.2
    if idle_time is not None and idle_time > ACTION["idle_time_low"]:
        hesitant_conf += 0.15
    if time_since > ACTION["time_since_action_moderate"] * 0.5:
        hesitant_conf += 0.1
    # Misguided actions (looked at wrong object then acted) → hesitant
    if has_coupling and blind_ratio > 0.5 and informed_ratio < 0.3:
        hesitant_conf += 0.15
    scores["hesitant"] = _clamp(hesitant_conf, 0.1, 0.90)

    # Failing: errors OR blind wrong actions
    fail_conf = 0.0
    if error_count is not None and error_count >= PERFORMANCE["error_count_high"]:
        fail_conf = _linear_scale(error_count, 0, 3) * 0.5
    # Blind + wrong is a strong failing signal
    if blind_wrong > 0:
        fail_conf += _linear_scale(blind_wrong, 0, 3) * 0.4
    # High blind ratio with actions = trial-and-error
    if has_coupling and blind_ratio >= 0.8 and action_count >= 2:
        fail_conf += 0.2
    if fail_conf >= 0.3:
        scores["failing"] = _clamp(fail_conf, 0.3, 0.95)

    best = max(scores, key=scores.get)
    conf = scores[best]

    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and (sorted_s[0] - sorted_s[1]) < 0.1:
        conf *= 0.75

    coupling_str = ""
    if has_coupling:
        coupling_str = f", informed={informed_ratio:.0%}, blind={blind_ratio:.0%}"

    reasons = {
        "active": f"Actions={action_count}, {time_since:.0f}s since last{coupling_str}",
        "inactive": f"No actions, idle={idle_time:.1f}s, {time_since:.0f}s since last" if idle_time else f"No actions, {time_since:.0f}s since last",
        "hesitant": f"Actions={action_count} but uncertain{coupling_str}",
        "failing": f"Errors={error_count}, blind_wrong={blind_wrong}{coupling_str}",
    }

    return {
        "label": best,
        "confidence": round(_clamp(conf), 3),
        "evidence": evidence,
        "reasoning": reasons.get(best, ""),
        "all_scores": {k: round(v, 3) for k, v in scores.items()},
    }


# Backward compat aliases
action_agent = behavioral_agent
from config import PERFORMANCE


# ---------------------------------------------------------------------------
# 5. TemporalAgent — is this pattern new or persistent?
#    Reads labels from fixation, semantics, motor, behavioral agents.
# ---------------------------------------------------------------------------

def temporal_agent(current_idx: int, df: pd.DataFrame, state_columns: dict) -> dict:
    row = df.loc[current_idx]
    pid = safe_get(row, "participant_id")
    if pid is None:
        return {"label": "unknown", "confidence": 0.0, "evidence": {},
                "reasoning": "Missing participant ID"}

    participant_df = df[df["participant_id"] == pid].sort_values("window_start")
    indices = participant_df.index.tolist()

    if current_idx not in indices:
        return {"label": "unknown", "confidence": 0.0, "evidence": {},
                "reasoning": "Index not found"}

    pos = indices.index(current_idx)
    window = TEMPORAL["persistence_window"]

    if pos < 1:
        return {"label": "transient", "confidence": 0.5,
                "evidence": {"history_length": 0},
                "reasoning": "First window, no history"}

    start = max(0, pos - window + 1)
    recent = participant_df.iloc[start:pos + 1]

    # Check multiple agent columns for persistence
    agent_cols = [
        state_columns.get("fixation", "fixation_label"),
        state_columns.get("semantics", "semantics_label"),
        state_columns.get("behavioral", "action_label"),
    ]
    agent_cols = [c for c in agent_cols if c in recent.columns]

    evidence = {"history_length": len(recent)}

    if len(recent) >= window and agent_cols:
        # Check if any agent has been stable
        stable_agents = []
        looping_detected = False

        for col in agent_cols:
            if col in recent.columns:
                vals = recent[col].tolist()
                if len(set(vals)) == 1:
                    stable_agents.append((col, vals[-1]))
                    if vals[-1] in TEMPORAL["looping_keywords"]:
                        looping_detected = True

        if looping_detected:
            return {
                "label": "looping",
                "confidence": 0.85,
                "evidence": {**evidence, "stable_agents": stable_agents},
                "reasoning": f"Stuck pattern persisting for {len(recent)} windows: {stable_agents}",
            }

        if len(stable_agents) >= 2:
            return {
                "label": "persistent",
                "confidence": 0.75,
                "evidence": {**evidence, "stable_agents": stable_agents},
                "reasoning": f"Stable pattern across {len(stable_agents)} agents for {len(recent)} windows",
            }

    # Check for recent looping keywords even with shorter window
    if len(recent) >= 2 and agent_cols:
        for col in agent_cols:
            if col in recent.columns:
                vals = recent[col].tolist()
                if all(v in TEMPORAL["looping_keywords"] for v in vals):
                    return {
                        "label": "looping",
                        "confidence": 0.65,
                        "evidence": {**evidence, "pattern": vals},
                        "reasoning": f"Repeated difficulty: {vals}",
                    }

    return {"label": "transient", "confidence": 0.4,
            "evidence": evidence,
            "reasoning": "No persistent pattern"}


# ---------------------------------------------------------------------------
# Backward compat aliases for pipeline/app
# ---------------------------------------------------------------------------
attention_agent = fixation_agent
performance_agent = behavioral_agent
progress_agent = behavioral_agent
