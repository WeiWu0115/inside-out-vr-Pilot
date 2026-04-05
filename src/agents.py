"""
Theory-Partitioned Agents: Each agent sees ALL data, interprets through
a different cognitive theory lens.

Unlike V3 (data-partitioned: each agent reads exclusive features),
here agents disagree because they bring different THEORETICAL COMMITMENTS
to the same observation — faithful to the Inside Out metaphor where Joy
and Sadness see the same memory but interpret it differently.

Echo consensus is avoided not by splitting data, but by ensuring each
agent's scoring logic reflects a genuinely different theoretical framework.

Agent → Theory mapping:
  1. Attention Theory Agent   (Posner, Lavie)       → attention allocation
  2. Self-Regulation Agent    (Zimmerman, Pintrich)  → metacognitive monitoring
  3. Flow Theory Agent        (Csikszentmihalyi)     → challenge-skill balance
  4. Cognitive Load Agent     (Sweller)              → working memory capacity
  5. Temporal Agent           (unchanged)            → persistence over time
  + Population Agent          (unchanged)            → population comparison
"""

import pandas as pd
import numpy as np
from load_data import safe_get
from config import ATTENTION, ACTION, PERFORMANCE, TEMPORAL


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _linear_scale(value, low, high):
    if high == low:
        return 0.5
    return _clamp((value - low) / (high - low))


def _read_all_features(row):
    """Read all available features — shared across all theory agents."""
    return {
        "gaze_entropy": safe_get(row, "gaze_entropy") or 0.0,
        "clue_ratio": safe_get(row, "clue_ratio") or 0.0,
        "switch_rate": safe_get(row, "switch_rate") or 0.0,
        "action_count": safe_get(row, "action_count") or 0,
        "idle_time": safe_get(row, "idle_time") or 0.0,
        "error_count": safe_get(row, "error_count") or 0,
        "time_since_action": safe_get(row, "time_since_action") or 0.0,
        "puzzle_elapsed_ratio": safe_get(row, "puzzle_elapsed_ratio") or 0.0,
        "puzzle_active": safe_get(row, "puzzle_active") or 0,
    }


# ---------------------------------------------------------------------------
# 1. Attention Theory Agent (Posner, Lavie)
#    "Is the player's attention appropriately allocated?"
# ---------------------------------------------------------------------------

def attention_theory_agent(row: pd.Series) -> dict:
    f = _read_all_features(row)
    scores = {}

    entropy = f["gaze_entropy"]
    clue_ratio = f["clue_ratio"]
    switch_rate = f["switch_rate"]
    action_count = f["action_count"]
    idle_time = f["idle_time"]

    # Engaged: attention well-directed
    engaged_conf = 0.2
    if entropy < ATTENTION["entropy_high"]:
        engaged_conf += (1.0 - _linear_scale(entropy, 0, ATTENTION["entropy_high"])) * 0.3
    if clue_ratio > 0.2:
        engaged_conf += _linear_scale(clue_ratio, 0, ATTENTION["clue_ratio_high"]) * 0.2
    if action_count > 0:
        engaged_conf += 0.15
    scores["engaged"] = _clamp(engaged_conf, 0.1, 0.95)

    # Overloaded: too many competing stimuli
    overload_conf = 0.1
    if entropy > ATTENTION["entropy_high"]:
        overload_conf += _linear_scale(entropy, ATTENTION["entropy_high"],
                                        ATTENTION["entropy_high"] * 2) * 0.35
    if switch_rate > ATTENTION["switch_rate_high"]:
        overload_conf += _linear_scale(switch_rate, ATTENTION["switch_rate_high"],
                                        ATTENTION["switch_rate_high"] * 2) * 0.25
    if action_count == 0 and entropy > ATTENTION["entropy_low"]:
        overload_conf += 0.15
    scores["overloaded"] = _clamp(overload_conf, 0.05, 0.95)

    # Fixated: attentional capture — locked on one thing
    fixated_conf = 0.1
    if clue_ratio > ATTENTION["clue_ratio_very_high"] and entropy < ATTENTION["entropy_low"]:
        fixated_conf += 0.4
    if action_count == 0 and idle_time > 4.5 and entropy < ATTENTION["entropy_high"]:
        fixated_conf += 0.2
    scores["fixated"] = _clamp(fixated_conf, 0.05, 0.95)

    # Decoupled: acting without looking = attention-action mismatch
    decoupled_conf = 0.1
    if action_count > 0 and entropy > ATTENTION["entropy_high"] and clue_ratio < 0.1:
        decoupled_conf += 0.4
    if action_count >= ACTION["action_count_high"] and entropy > ATTENTION["entropy_high"]:
        decoupled_conf += 0.2
    scores["decoupled"] = _clamp(decoupled_conf, 0.05, 0.95)

    best = max(scores, key=scores.get)
    conf = scores[best]
    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and sorted_s[0] - sorted_s[1] < 0.12:
        conf *= 0.75

    reasons = {
        "engaged": f"Attention well-directed: entropy={entropy:.2f}, clue={clue_ratio:.2f}",
        "overloaded": f"Attentional overload: entropy={entropy:.2f}, switch={switch_rate:.1f}",
        "fixated": f"Attentional capture: fixated (clue={clue_ratio:.2f}) without acting",
        "decoupled": f"Attention-action decoupling: {action_count} actions but gaze unfocused ({entropy:.2f})",
    }
    return {"label": best, "confidence": round(_clamp(conf), 3),
            "evidence": f, "reasoning": reasons.get(best, "")}


attention_agent = attention_theory_agent


# ---------------------------------------------------------------------------
# 2. Self-Regulation Theory Agent (Zimmerman, Pintrich)
#    "Is the player effectively monitoring and adjusting their strategy?"
# ---------------------------------------------------------------------------

def self_regulation_agent(row: pd.Series) -> dict:
    f = _read_all_features(row)
    scores = {}

    action_count = f["action_count"]
    time_since = f["time_since_action"]
    error_count = f["error_count"]
    idle_time = f["idle_time"]
    elapsed_ratio = f["puzzle_elapsed_ratio"]
    entropy = f["gaze_entropy"]

    # Self-regulated: strategic, measured action
    regulated_conf = 0.3
    if 0 < action_count <= ACTION["action_count_high"]:
        regulated_conf += 0.2
    if error_count == 0:
        regulated_conf += 0.15
    if time_since < ACTION["time_since_action_moderate"]:
        regulated_conf += 0.15
    if elapsed_ratio < 1.5:
        regulated_conf += 0.1
    scores["self_regulated"] = _clamp(regulated_conf, 0.1, 0.95)

    # Impulsive: burst after long gap = trial-and-error
    impulsive_conf = 0.1
    if action_count >= 2 and time_since > ACTION["time_since_action_moderate"]:
        impulsive_conf += _linear_scale(time_since, ACTION["time_since_action_moderate"],
                                         ACTION["time_since_action_moderate"] * 3) * 0.3
        impulsive_conf += _linear_scale(action_count, 1, ACTION["action_count_high"] * 2) * 0.2
    if error_count > 0 and action_count > 0:
        impulsive_conf += 0.15
    scores["impulsive"] = _clamp(impulsive_conf, 0.05, 0.95)

    # Disengaged: gave up self-monitoring
    disengaged_conf = 0.1
    if action_count == 0 and idle_time > 4.5:
        disengaged_conf += 0.35
    if time_since > ACTION["time_since_action_moderate"] * 2:
        disengaged_conf += _linear_scale(time_since, ACTION["time_since_action_moderate"] * 2,
                                          ACTION["time_since_action_moderate"] * 4) * 0.25
    if elapsed_ratio > 2.0:
        disengaged_conf += 0.1
    scores["disengaged"] = _clamp(disengaged_conf, 0.05, 0.95)

    # Reflective: productive pause
    reflective_conf = 0.1
    if action_count == 0 and idle_time > 3.0 and entropy < ATTENTION["entropy_high"]:
        reflective_conf += 0.3
    if time_since < ACTION["time_since_action_moderate"] and action_count == 0:
        reflective_conf += 0.2
    scores["reflective"] = _clamp(reflective_conf, 0.05, 0.95)

    best = max(scores, key=scores.get)
    conf = scores[best]
    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and sorted_s[0] - sorted_s[1] < 0.12:
        conf *= 0.75

    reasons = {
        "self_regulated": f"Strategic: {action_count} actions, {error_count} errors, {time_since:.0f}s since last",
        "impulsive": f"Regulation breakdown: {action_count} actions after {time_since:.0f}s gap",
        "disengaged": f"Self-monitoring collapsed: idle {time_since:.0f}s",
        "reflective": f"Productive pause: focused gaze (entropy={entropy:.2f}), evaluating",
    }
    return {"label": best, "confidence": round(_clamp(conf), 3),
            "evidence": f, "reasoning": reasons.get(best, "")}


action_agent = self_regulation_agent


# ---------------------------------------------------------------------------
# 3. Flow Theory Agent (Csikszentmihalyi)
#    "Is the challenge-skill balance right?"
# ---------------------------------------------------------------------------

def flow_theory_agent(row: pd.Series) -> dict:
    f = _read_all_features(row)
    scores = {}

    action_count = f["action_count"]
    error_count = f["error_count"]
    elapsed_ratio = f["puzzle_elapsed_ratio"]
    time_since = f["time_since_action"]
    entropy = f["gaze_entropy"]
    idle_time = f["idle_time"]

    # Flow: balanced engagement
    flow_conf = 0.2
    if 1 <= action_count <= ACTION["action_count_high"] * 2:
        flow_conf += 0.2
    if error_count == 0:
        flow_conf += 0.15
    if 0.3 < elapsed_ratio < 1.5:
        flow_conf += 0.2
    if time_since < ACTION["time_since_action_moderate"] * 0.5:
        flow_conf += 0.15
    scores["flow"] = _clamp(flow_conf, 0.1, 0.95)

    # Anxiety: challenge exceeds skill
    anxiety_conf = 0.1
    if elapsed_ratio > 1.5:
        anxiety_conf += _linear_scale(elapsed_ratio, 1.5, 3.0) * 0.3
    if time_since > ACTION["time_since_action_moderate"]:
        anxiety_conf += _linear_scale(time_since, ACTION["time_since_action_moderate"],
                                       ACTION["time_since_action_moderate"] * 3) * 0.2
    if entropy > ATTENTION["entropy_high"]:
        anxiety_conf += 0.15
    if error_count > 0:
        anxiety_conf += 0.1
    scores["anxiety"] = _clamp(anxiety_conf, 0.05, 0.95)

    # Frustration: sustained effort + repeated failure
    frustration_conf = 0.1
    if action_count > 0 and error_count > 0:
        frustration_conf += 0.3
    if action_count >= ACTION["action_count_high"] and error_count > 0:
        frustration_conf += 0.2
    if elapsed_ratio > 2.0 and action_count > 0:
        frustration_conf += 0.15
    scores["frustration"] = _clamp(frustration_conf, 0.05, 0.95)

    # Boredom: skill exceeds challenge
    boredom_conf = 0.1
    if elapsed_ratio < 0.3 and action_count == 0:
        boredom_conf += 0.3
    if idle_time > 4.5 and elapsed_ratio < 0.5:
        boredom_conf += 0.2
    scores["boredom"] = _clamp(boredom_conf, 0.05, 0.90)

    best = max(scores, key=scores.get)
    conf = scores[best]
    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and sorted_s[0] - sorted_s[1] < 0.12:
        conf *= 0.75

    reasons = {
        "flow": f"Challenge-skill balance: {action_count} actions, {error_count} errors, {elapsed_ratio:.1f}x median",
        "anxiety": f"Challenge exceeds skill: {elapsed_ratio:.1f}x median, {time_since:.0f}s inactive",
        "frustration": f"Repeated failure: {action_count} actions, {error_count} errors at {elapsed_ratio:.1f}x",
        "boredom": f"Under-challenged: {elapsed_ratio:.1f}x median, low engagement",
    }
    return {"label": best, "confidence": round(_clamp(conf), 3),
            "evidence": f, "reasoning": reasons.get(best, "")}


performance_agent = flow_theory_agent


# ---------------------------------------------------------------------------
# 4. Cognitive Load Theory Agent (Sweller)
#    "Is working memory capacity exceeded?"
# ---------------------------------------------------------------------------

def cognitive_load_agent(row: pd.Series) -> dict:
    f = _read_all_features(row)
    scores = {}

    entropy = f["gaze_entropy"]
    switch_rate = f["switch_rate"]
    clue_ratio = f["clue_ratio"]
    action_count = f["action_count"]
    error_count = f["error_count"]
    idle_time = f["idle_time"]
    time_since = f["time_since_action"]
    elapsed_ratio = f["puzzle_elapsed_ratio"]

    # Manageable: can attend + act effectively
    manage_conf = 0.2
    if entropy < ATTENTION["entropy_high"] and action_count > 0 and error_count == 0:
        manage_conf += 0.4
    if time_since < ACTION["time_since_action_moderate"]:
        manage_conf += 0.15
    if elapsed_ratio < 1.5:
        manage_conf += 0.1
    scores["manageable"] = _clamp(manage_conf, 0.1, 0.95)

    # Overloaded: scanning, switching, can't act
    overloaded_conf = 0.1
    if entropy > ATTENTION["entropy_high"]:
        overloaded_conf += _linear_scale(entropy, ATTENTION["entropy_high"],
                                          ATTENTION["entropy_high"] * 2) * 0.25
    if switch_rate > ATTENTION["switch_rate_high"]:
        overloaded_conf += 0.15
    if action_count == 0 and entropy > ATTENTION["entropy_low"]:
        overloaded_conf += 0.2
    if elapsed_ratio > 2.0:
        overloaded_conf += 0.15
    scores["overloaded"] = _clamp(overloaded_conf, 0.05, 0.95)

    # Fragmented: partial understanding — some right, some wrong
    fragmented_conf = 0.1
    if action_count > 0 and error_count > 0:
        fragmented_conf += 0.35
    if clue_ratio > 0.3 and error_count > 0:
        fragmented_conf += 0.2
    scores["fragmented"] = _clamp(fragmented_conf, 0.05, 0.95)

    # Automated: routine, minimal cognitive effort
    auto_conf = 0.1
    if action_count >= ACTION["action_count_high"] and error_count == 0 and entropy < ATTENTION["entropy_low"]:
        auto_conf += 0.5
    scores["automated"] = _clamp(auto_conf, 0.05, 0.90)

    best = max(scores, key=scores.get)
    conf = scores[best]
    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) > 1 and sorted_s[0] - sorted_s[1] < 0.12:
        conf *= 0.75

    reasons = {
        "manageable": f"Load within capacity: focused + acting + no errors",
        "overloaded": f"Working memory exceeded: entropy={entropy:.2f}, switch={switch_rate:.1f}",
        "fragmented": f"Partial understanding: {action_count} actions but {error_count} errors",
        "automated": f"Routine processing: high action + no errors + focused",
    }
    return {"label": best, "confidence": round(_clamp(conf), 3),
            "evidence": f, "reasoning": reasons.get(best, "")}


# ---------------------------------------------------------------------------
# 5. Temporal Agent — reads theory agent labels over time
# ---------------------------------------------------------------------------

def temporal_agent(current_idx: int, df: pd.DataFrame, state_columns: dict) -> dict:
    row = df.loc[current_idx]
    pid = safe_get(row, "participant_id")
    if pid is None:
        return {"label": "unknown", "confidence": 0.0, "evidence": {}, "reasoning": "Missing participant ID"}

    participant_df = df[df["participant_id"] == pid].sort_values("window_start")
    indices = participant_df.index.tolist()
    if current_idx not in indices:
        return {"label": "unknown", "confidence": 0.0, "evidence": {}, "reasoning": "Index not found"}

    pos = indices.index(current_idx)
    window = TEMPORAL["persistence_window"]
    elapsed_ratio = safe_get(row, "puzzle_elapsed_ratio") or 0.0

    if pos < 1:
        if elapsed_ratio > 3.0:
            return {"label": "persistent", "confidence": 0.6,
                    "evidence": {"history_length": 0, "puzzle_elapsed_ratio": elapsed_ratio},
                    "reasoning": f"First window but puzzle at {elapsed_ratio:.1f}x median"}
        return {"label": "transient", "confidence": 0.5,
                "evidence": {"history_length": 0}, "reasoning": "First window"}

    start = max(0, pos - window + 1)
    recent = participant_df.iloc[start:pos + 1]
    att_col = state_columns.get("attention", "attention_label")
    perf_col = state_columns.get("performance", "performance_label")
    evidence = {"history_length": len(recent), "puzzle_elapsed_ratio": elapsed_ratio}

    struggle_keywords = TEMPORAL["looping_keywords"]

    if len(recent) >= window and att_col in recent.columns and perf_col in recent.columns:
        att_stable = recent[att_col].nunique() == 1
        perf_stable = recent[perf_col].nunique() == 1
        if att_stable and perf_stable:
            perf_val = recent[perf_col].iloc[-1]
            if perf_val in struggle_keywords:
                conf = min(0.95, 0.85 + (0.1 if elapsed_ratio > 2.0 else 0))
                return {"label": "looping", "confidence": conf,
                        "evidence": {**evidence, "repeated_state": perf_val, "streak": len(recent)},
                        "reasoning": f"Stuck in '{perf_val}' for {len(recent)} windows ({elapsed_ratio:.1f}x median)"}
            return {"label": "persistent", "confidence": 0.75,
                    "evidence": {**evidence, "streak": len(recent)},
                    "reasoning": f"Stable state for {len(recent)} windows"}

    if len(recent) >= 2 and perf_col in recent.columns:
        perf_vals = recent[perf_col].tolist()
        if all(v in struggle_keywords for v in perf_vals):
            conf = min(0.85, 0.65 + (0.15 if elapsed_ratio > 2.0 else 0))
            return {"label": "looping", "confidence": conf,
                    "evidence": {**evidence, "pattern": perf_vals},
                    "reasoning": f"Repeated difficulty: {perf_vals} ({elapsed_ratio:.1f}x median)"}

    if elapsed_ratio > 3.0:
        return {"label": "persistent", "confidence": 0.6, "evidence": evidence,
                "reasoning": f"No pattern but {elapsed_ratio:.1f}x median"}

    return {"label": "transient", "confidence": 0.4, "evidence": evidence,
            "reasoning": "No persistent pattern detected"}
