"""
Theory-Partitioned Support Layer.

Maps theory-agent disagreement patterns to adaptive interventions.
Each tension is now between cognitive THEORIES, not data streams.
"""

import pandas as pd
from config import ACTION


def classify_transition(row: pd.Series) -> str:
    entropy = row.get("gaze_entropy", 0) or 0
    switch_rate = row.get("switch_rate", 0) or 0
    action_count = row.get("action_count", 0) or 0
    idle_time = row.get("idle_time", 0) or 0
    time_since = row.get("time_since_action", 0) or 0

    if time_since > 120 and entropy > 1.5:
        return "looping_transition"
    if idle_time > 4.5 and action_count == 0 and time_since > 60:
        return "hesitant_transition"
    if entropy > 1.5 and switch_rate > 5.0:
        return "searching_transition"
    return "navigating_normally"


def suggest_support(row: pd.Series) -> dict:
    d_type = row.get("disagreement_type", "unstructured")
    tension = row.get("dominant_tension", "none")
    temporal = row.get("temporal_label", "transient")
    puzzle_id = row.get("puzzle_id", "")
    elapsed_ratio = row.get("puzzle_elapsed_ratio", 0.0) or 0.0
    time_since = row.get("time_since_action", 0) or 0

    # Agent labels (theory-specific)
    att_label = row.get("attention_label", "unknown")   # Attention Theory
    reg_label = row.get("action_label", "unknown")      # Self-Regulation
    flow_label = row.get("performance_label", "unknown") # Flow Theory
    flow_conf = row.get("performance_confidence", 0.0)

    # ── TRANSITION ──────────────────────────────────────────────────────
    if puzzle_id == "Transition":
        subtype = classify_transition(row)
        if subtype == "looping_transition" and temporal in ("looping", "persistent"):
            return _result("spatial_hint", 0.65, "Lost in transition.", "consensus_intervene")
        if subtype in ("hesitant_transition", "looping_transition"):
            return _result("gentle_probe", 0.45, f"Transition: {subtype}.", "probe")
        if subtype == "searching_transition":
            return _result("wait", 0.5, "Searching during transition.", "watch")
        return _result("wait", 0.7, "Normal navigation.", "watch")

    # ── CONSTRUCTIVE: theories agree ────────────────────────────────────
    if d_type == "constructive":

        # All positive: engaged + regulated + flow
        if tension in ("engaged_in_flow", "regulated_flow", "engaged_and_regulated"):
            return _result("wait", 0.85, f"Theories agree: productive ({tension})", "watch")

        if tension == "engaged_and_reflective":
            return _result("wait", 0.8, "Engaged + reflective pause. Productive thinking.", "watch")

        # All negative: converging struggle
        if tension in ("overloaded_and_disengaged", "overloaded_with_anxiety"):
            urgency = 0.85 if temporal in ("looping", "persistent") else 0.6
            return _result("spatial_hint", urgency,
                           f"Theories converge on struggle: {tension} ({temporal})",
                           "consensus_intervene")

        if tension in ("fixated_and_disengaged", "disengaged_with_anxiety"):
            urgency = 0.8 if temporal in ("looping", "persistent") else 0.55
            return _result("reorientation_prompt", urgency,
                           f"Converging: {tension} ({temporal})",
                           "consensus_intervene")

        if tension == "impulsive_frustration":
            return _result("redirect_hint", 0.7,
                           "Impulsive behavior + frustration. Needs new approach.",
                           "consensus_intervene")

        if tension == "pop_confirms_impasse":
            return _result("spatial_hint", 0.7, "Population confirms cognitive impasse.", "consensus_intervene")

        if tension == "pop_confirms_flow":
            return _result("wait", 0.8, "Population confirms flow state.", "watch")

        if tension == "pop_confirms_disorientation":
            return _result("spatial_hint", 0.6, "Population confirms disorientation.", "probe")

    # ── CONTRADICTORY: theories disagree ────────────────────────────────
    if d_type == "contradictory":

        # Attention says OK but regulation/flow says problem
        if tension == "engaged_but_anxious":
            if temporal in ("looping", "persistent") or elapsed_ratio > 2.0:
                return _result("gentle_probe", 0.6,
                               f"Attention engaged but Flow says anxiety ({elapsed_ratio:.1f}x median). Probe.",
                               "probe")
            return _result("wait", 0.55,
                           "Attention engaged, Flow uncertain. May resolve.", "watch")

        if tension == "engaged_but_frustrated":
            return _result("procedural_hint", 0.65,
                           "Player is paying attention but making errors. Needs how-to help.",
                           "consensus_intervene" if temporal in ("looping", "persistent") else "probe")

        if tension == "engaged_but_impulsive":
            return _result("gentle_probe", 0.5,
                           "Attention focused but Self-Regulation reads impulsive. Probe strategy.",
                           "probe")

        if tension == "attending_but_disengaged":
            if temporal in ("persistent", "looping") or time_since > 60:
                return _result("gentle_probe", 0.6,
                               f"Looking but not acting ({temporal}). Probe: thinking or frozen?",
                               "probe")
            if elapsed_ratio > 2.0:
                return _result("monitor_closely", 0.5,
                               f"Attending but disengaged, {elapsed_ratio:.1f}x median.", "probe")
            return _result("wait", 0.55, "Attending + briefly disengaged — may be thinking.", "watch")

        # Regulation says OK but attention/flow says problem
        if tension == "overloaded_but_regulated":
            return _result("monitor_closely", 0.45,
                           "Attention overloaded but self-regulation intact. Monitor.", "probe")

        if tension == "regulated_but_anxious":
            return _result("gentle_probe", 0.55,
                           "Self-regulated behavior but Flow reads anxiety. Probe subtly.", "probe")

        if tension == "regulated_but_frustrated":
            return _result("procedural_hint", 0.6,
                           "Player is trying strategically but hitting walls.", "probe")

        # Contradictory positive: one theory sees flow/regulated, another sees problem
        if tension in ("impulsive_in_flow", "decoupled_in_flow", "overloaded_in_flow"):
            return _result("monitor_closely", 0.45,
                           f"Flow detected but other theory sees issue ({tension}). Monitor.", "probe")

        if tension == "disengaged_from_flow":
            return _result("gentle_probe", 0.55,
                           "Was in flow but now disengaged. Check if stuck or resting.", "probe")

        if tension in ("fixated_and_impulsive", "decoupled_but_regulated"):
            return _result("monitor_closely", 0.5,
                           f"Unusual pattern: {tension}. Monitor closely.", "probe")

        # Population contradictions
        if tension == "pop_exploring_but_engaged":
            return _result("wait", 0.6, "Population says exploring, attention says engaged. Likely OK.", "watch")

        if tension == "pop_stuck_but_regulated":
            return _result("gentle_probe", 0.5,
                           "Population sees stuck pattern but self-regulation intact.", "probe")

        if tension == "pop_solving_but_anxious":
            return _result("monitor_closely", 0.5,
                           "Population sees solving but Flow reads anxiety.", "probe")

    # ── UNSTRUCTURED ────────────────────────────────────────────────────

    # Use Flow Theory as primary signal for unstructured
    if flow_label in ("anxiety", "frustration") and flow_conf > 0.5 and temporal == "looping":
        return _result("spatial_hint", 0.5, f"No clear pattern but Flow says {flow_label} (looping).",
                       "consensus_intervene")

    if flow_label == "flow":
        return _result("wait", 0.6, "No clear pattern but Flow state detected.", "watch")

    if reg_label == "disengaged" and elapsed_ratio > 2.0:
        return _result("gentle_probe", 0.5, f"Disengaged at {elapsed_ratio:.1f}x median.", "probe")

    return _result("monitor", 0.3, "No clear signal from theory agents.", "watch")


def _result(action, confidence, rationale, category):
    return {"action": action, "confidence": confidence,
            "rationale": rationale, "category": category}


# ---------------------------------------------------------------------------
# Stateful Prompt Agent (same as V3)
# ---------------------------------------------------------------------------

class PromptAgent:
    def __init__(self, cooldown_sec=15.0, escalation_threshold=6,
                 max_prompts_per_puzzle=8):
        self.cooldown_sec = cooldown_sec
        self.escalation_threshold = escalation_threshold
        self.max_prompts_per_puzzle = max_prompts_per_puzzle
        self.reset()

    def reset(self):
        self.last_intervene_time = -999.0
        self.last_trigger_time = -999.0
        self.consecutive_struggle = 0
        self.prompt_count = 0
        self.current_puzzle = None

    def decide(self, base_result: dict, row: pd.Series) -> dict:
        window_start = row.get("window_start", 0) or 0
        puzzle_id = row.get("puzzle_id", "")
        category = base_result["category"]

        if puzzle_id != self.current_puzzle:
            self.current_puzzle = puzzle_id
            self.consecutive_struggle = 0
            self.prompt_count = 0

        if category in ("probe", "consensus_intervene"):
            self.consecutive_struggle += 1
        else:
            if self.consecutive_struggle > 0:
                self.consecutive_struggle = max(0, self.consecutive_struggle - 2)

        result = dict(base_result)

        # Cooldown for interventions
        time_since_intervene = window_start - self.last_intervene_time
        if time_since_intervene < self.cooldown_sec and category == "consensus_intervene":
            result["category"] = "probe"
            result["rationale"] += f" [Cooldown: {time_since_intervene:.0f}s since last intervention]"

        # Fatigue
        if self.prompt_count >= self.max_prompts_per_puzzle and category == "consensus_intervene":
            result["category"] = "probe"
            result["rationale"] += f" [Fatigue: {self.prompt_count} interventions on this puzzle]"

        # Escalation
        if self.consecutive_struggle >= self.escalation_threshold and category == "probe":
            result["category"] = "consensus_intervene"
            result["confidence"] = min(float(result.get("confidence", 0.5)) + 0.1, 0.95)
            result["rationale"] += f" [Escalated: {self.consecutive_struggle} consecutive struggle]"

        if result["category"] == "consensus_intervene":
            self.last_intervene_time = window_start
            self.last_trigger_time = window_start
            self.prompt_count += 1
        elif result["category"] == "probe":
            self.last_trigger_time = window_start

        return result


def run_support(df: pd.DataFrame) -> pd.DataFrame:
    base_results = df.apply(suggest_support, axis=1)
    final_results = []
    prompt_agents = {}

    for idx in df.index:
        row = df.loc[idx]
        pid = row.get("participant_id", 0)
        if pid not in prompt_agents:
            prompt_agents[pid] = PromptAgent()
        final = prompt_agents[pid].decide(base_results[idx], row)
        final_results.append(final)

    final_series = pd.Series(final_results)
    df["suggested_support"] = final_series.apply(lambda s: s["action"])
    df["support_confidence"] = final_series.apply(lambda s: s["confidence"])
    df["support_rationale"] = final_series.apply(lambda s: s["rationale"])
    df["support_category"] = final_series.apply(lambda s: s["category"])
    return df
