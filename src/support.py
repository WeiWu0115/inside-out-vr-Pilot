"""
V4 Support layer: maps gaze-dominant disagreement structures to adaptive responses.

Three gaze agents (fixation, semantics, motor) + one behavioral agent.
Tension patterns drive watch / probe / intervene decisions.
"""

import pandas as pd
from config import ACTION


# ---------------------------------------------------------------------------
# Transition subtype classification
# ---------------------------------------------------------------------------

def classify_transition(row: pd.Series) -> str:
    entropy = row.get("gaze_target_entropy", 0) or 0
    revisit = row.get("revisit_rate", 0) or 0
    action_count = row.get("action_count", 0) or 0
    time_since = row.get("time_since_action", 0) or 0
    coupling = row.get("gaze_head_coupling", 0) or 0

    if time_since > 120 and entropy > 1.5:
        return "looping_transition"
    if time_since > 60 and action_count == 0 and coupling > 0.3:
        return "hesitant_transition"
    if entropy > 1.5 and revisit < 0.3:
        return "searching_transition"
    return "navigating_normally"


# ---------------------------------------------------------------------------
# Core support decision
# ---------------------------------------------------------------------------

def suggest_support(row: pd.Series) -> dict:
    d_type = row.get("disagreement_type", "unstructured")
    d_intensity = row.get("disagreement_intensity", 0.0)
    tension = row.get("dominant_tension", "none")
    temporal = row.get("temporal_label", "transient")
    puzzle_id = row.get("puzzle_id", "")
    time_since = row.get("time_since_action", 0) or 0

    fix_label = row.get("fixation_label", "unknown")
    sem_label = row.get("semantics_label", "unknown")
    mot_label = row.get("motor_label", "unknown")
    act_label = row.get("action_label", "unknown")

    # ── TRANSITION ─────────────────────────────────────────────
    if puzzle_id == "Transition":
        subtype = classify_transition(row)
        if subtype == "looping_transition":
            if temporal in ("looping", "persistent"):
                return _result("spatial_hint", 0.65,
                               "Lost in transition: high entropy + long duration.",
                               "consensus_intervene")
            return _result("gentle_probe", 0.5,
                           "Transition taking long. Probe.", "probe")
        if subtype == "hesitant_transition":
            return _result("monitor", 0.45,
                           "Hesitant during transition.", "probe")
        if subtype == "searching_transition":
            return _result("wait", 0.5,
                           "Actively searching during transition.", "watch")
        return _result("wait", 0.7, "Normal navigation.", "watch")

    # ── CONSTRUCTIVE: agents largely agree ──────────────────────

    if d_type == "constructive":

        # --- Stuck agreements ---
        if tension == "frozen_on_clue":
            urgency = 0.8 if temporal == "looping" else 0.6
            return _result("reorientation_prompt", urgency,
                           f"Fixation locked + clue focused — frozen reading ({temporal})",
                           "consensus_intervene")

        if tension in ("lost_and_passive", "disengaged"):
            urgency = 0.8 if temporal in ("looping", "persistent") else 0.55
            return _result("spatial_hint", urgency,
                           f"Agents agree: lost and passive ({temporal})",
                           "consensus_intervene" if temporal in ("looping", "persistent") else "probe")

        if tension == "passive_and_idle":
            urgency = 0.75 if temporal in ("looping", "persistent") else 0.5
            return _result("engagement_prompt", urgency,
                           f"Passive gaze + no actions ({temporal})",
                           "consensus_intervene" if temporal == "looping" else "probe")

        if tension == "panicking":
            return _result("calming_hint", 0.7,
                           "Erratic gaze + errors — panicking.",
                           "consensus_intervene")

        if tension == "uncertain_checking":
            if temporal in ("looping", "persistent"):
                return _result("gentle_probe", 0.6,
                               f"Revisiting + hesitant persistently ({temporal}). Likely stuck.",
                               "probe")
            return _result("monitor", 0.45,
                           "Revisiting targets + hesitant — checking understanding.",
                           "probe")

        # --- Positive agreements ---
        if tension in ("engaged_and_active", "task_engaged", "purposeful_action",
                        "deep_clue_reading"):
            return _result("wait", 0.8,
                           f"Productive engagement ({tension})", "watch")

        if tension in ("active_exploration", "purposeful_scanning", "systematic_search"):
            return _result("wait", 0.75,
                           f"Healthy exploration ({tension})", "watch")

        if tension == "purposeful_task_gaze":
            if act_label == "inactive" and time_since > 60:
                return _result("gentle_probe", 0.5,
                               "Purposeful gaze at task but inactive for a while. Check.",
                               "probe")
            return _result("wait", 0.7,
                           "Purposeful task-focused gaze.", "watch")

    # ── CONTRADICTORY: agents disagree ─────────────────────────

    if d_type == "contradictory":

        # --- Gaze-action contradictions (largest group) ---

        if tension == "focused_but_idle":
            if temporal in ("persistent", "looping") or time_since > 60:
                return _result("gentle_probe", 0.6,
                               f"Focused gaze + no action ({temporal}, {time_since:.0f}s). Probe.",
                               "probe")
            return _result("wait", 0.55,
                           "Focused + briefly idle — likely thinking.", "watch")

        if tension == "frozen":
            if temporal in ("looping", "persistent"):
                return _result("reorientation_prompt", 0.75,
                               f"Locked gaze + inactive — frozen ({temporal})",
                               "consensus_intervene")
            return _result("gentle_probe", 0.55,
                           "Locked gaze + inactive — may be processing or stuck.",
                           "probe")

        if tension == "watching_task_but_idle":
            if temporal in ("looping", "persistent") or time_since > 90:
                return _result("gentle_probe", 0.6,
                               f"Looking at task but not acting ({temporal}, {time_since:.0f}s).",
                               "probe")
            if time_since > 45:
                return _result("monitor", 0.45,
                               "Watching task objects but idle — processing?",
                               "probe")
            return _result("wait", 0.5,
                           "Task-focused gaze, briefly inactive.", "watch")

        if tension == "reading_but_not_acting":
            if temporal in ("looping", "persistent"):
                return _result("procedural_hint", 0.7,
                               f"Reading clues but not acting ({temporal}). May not understand.",
                               "consensus_intervene")
            return _result("gentle_probe", 0.55,
                           "Reading clues but no action. Processing or confused?",
                           "probe")

        if tension == "acting_while_looking_away":
            if temporal in ("looping", "persistent"):
                return _result("refocus_prompt", 0.6,
                               f"Acting on wrong area — eyes on environment ({temporal}). Redirect.",
                               "probe")
            return _result("monitor", 0.4,
                           "Active but looking at environment. May be navigating.",
                           "probe" if time_since > 30 else "watch")

        if tension == "acting_without_looking":
            return _result("monitor", 0.45,
                           "Active without focused gaze. Trial-and-error behavior.",
                           "probe")

        if tension == "scanning_but_passive":
            if temporal in ("looping", "persistent"):
                return _result("gentle_probe", 0.6,
                               f"Scanning environment + inactive ({temporal}). Lost?",
                               "probe")
            return _result("monitor", 0.45,
                           "Scanning + inactive — early exploration or lost.",
                           "probe" if time_since > 60 else "watch")

        # --- Gaze-gaze contradictions ---

        if tension == "locked_on_environment":
            if temporal in ("looping", "persistent"):
                return _result("spatial_hint", 0.65,
                               f"Locked staring at walls/floor ({temporal}). Disoriented.",
                               "consensus_intervene")
            return _result("monitor", 0.45,
                           "Locked gaze on environment. Taking a break or lost?",
                           "probe")

        if tension == "locked_but_unfocused":
            return _result("gentle_probe", 0.55,
                           "Locked gaze but entropy high. Staring blankly?",
                           "probe")

        if tension == "rapid_clue_switching":
            return _result("monitor", 0.5,
                           "Scanning through clues quickly. Processing or skimming?",
                           "probe")

        if tension == "erratic_clue_reading":
            return _result("gentle_probe", 0.6,
                           "Erratic gaze on clues. Overwhelmed by information?",
                           "probe")

        if tension == "concentrated_but_off_task":
            return _result("refocus_prompt", 0.55,
                           "Concentrated gaze but not on task objects.",
                           "probe")

        # --- Motor contradictions ---

        if tension == "passive_gaze_active_hands":
            return _result("monitor", 0.4,
                           "Passive gaze but active hands. Muscle memory or blind trying.",
                           "probe")

        if tension == "erratic_gaze_no_action":
            if temporal in ("looping", "persistent"):
                return _result("calming_hint", 0.65,
                               f"Erratic looking + no action ({temporal}). Overwhelmed.",
                               "consensus_intervene")
            return _result("monitor", 0.5,
                           "Erratic gaze + inactive. Disoriented?",
                           "probe")

        if tension == "concentrated_but_failing":
            return _result("procedural_hint", 0.7,
                           "Concentrated gaze + errors. Understands what to do but not how.",
                           "consensus_intervene")

        if tension == "erratic_motor_focused_fixation":
            return _result("monitor", 0.4,
                           "Motor erratic but fixations focused. Transitioning?",
                           "watch")

        if tension == "concentrated_motor_scanning_fixation":
            if temporal in ("looping", "persistent"):
                return _result("gentle_probe", 0.5,
                               f"Concentrated motor + scanning fixation ({temporal}). Stuck searching?",
                               "probe")
            return _result("wait", 0.45,
                           "Concentrated motor + scanning fixation. Exploring.",
                           "watch")

        if tension == "locked_gaze_but_active":
            return _result("monitor", 0.4,
                           "Locked gaze but physically active. Fixated on one approach?",
                           "probe")

    # ── UNSTRUCTURED ───────────────────────────────────────────

    if act_label in ("inactive",) and temporal == "looping":
        return _result("spatial_hint", 0.5,
                       "No clear pattern but persistent inactivity.",
                       "consensus_intervene")

    if act_label == "failing" and temporal in ("looping", "persistent"):
        return _result("procedural_hint", 0.6,
                       "Repeated errors with no clear gaze pattern.",
                       "consensus_intervene")

    if fix_label == "locked" and act_label == "inactive" and time_since > 60:
        return _result("gentle_probe", 0.5,
                       "Locked gaze + inactive, long time since action.",
                       "probe")

    if sem_label in ("environmental_scanning", "unfocused") and act_label == "inactive":
        if time_since > 90:
            return _result("gentle_probe", 0.45,
                           "Looking at environment, inactive for a while.",
                           "probe")

    if act_label in ("active", "hesitant") and sem_label == "task_focused":
        return _result("wait", 0.6,
                       "Engaged with task.", "watch")

    return _result("monitor", 0.3,
                   "No clear signal from agent negotiation.", "watch")


def _result(action, confidence, rationale, category):
    return {
        "action": action,
        "confidence": confidence,
        "rationale": rationale,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Stateful Prompt Agent
# ---------------------------------------------------------------------------

class PromptAgent:
    def __init__(self, cooldown_sec=15.0, escalation_threshold=6, max_prompts_per_puzzle=8):
        self.cooldown_sec = cooldown_sec
        self.escalation_threshold = escalation_threshold
        self.max_prompts_per_puzzle = max_prompts_per_puzzle
        self.reset()

    def reset(self):
        self.last_prompt_time = -999.0
        self.consecutive_struggle = 0
        self.prompt_count = 0
        self.last_category = "watch"
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
            if self.consecutive_struggle > 0 and category == "watch":
                self.consecutive_struggle = max(0, self.consecutive_struggle - 2)

        result = dict(base_result)

        time_since_prompt = window_start - self.last_prompt_time
        if time_since_prompt < self.cooldown_sec and category == "consensus_intervene":
            result["category"] = "probe"
            result["rationale"] += f" [Downgraded: {time_since_prompt:.0f}s since last]"

        if self.prompt_count >= self.max_prompts_per_puzzle and category == "consensus_intervene":
            result["category"] = "probe"
            result["rationale"] += f" [Downgraded: {self.prompt_count} prompts on puzzle]"

        if self.consecutive_struggle >= self.escalation_threshold and category == "probe":
            result["category"] = "consensus_intervene"
            result["confidence"] = min(result["confidence"] + 0.15, 0.95)
            result["rationale"] += f" [Escalated: {self.consecutive_struggle} consecutive struggle]"

        if result["category"] == "consensus_intervene":
            self.last_prompt_time = window_start
            self.prompt_count += 1

        self.last_category = result["category"]
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

        base = base_results[idx]
        final = prompt_agents[pid].decide(base, row)
        final_results.append(final)

    final_series = pd.Series(final_results)
    df["suggested_support"] = final_series.apply(lambda s: s["action"])
    df["support_confidence"] = final_series.apply(lambda s: s["confidence"])
    df["support_rationale"] = final_series.apply(lambda s: s["rationale"])
    df["support_category"] = final_series.apply(lambda s: s["category"])

    return df
