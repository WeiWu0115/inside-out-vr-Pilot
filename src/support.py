"""
V3 Support layer: maps disagreement structures to adaptive interventions.

V3 changes:
  - Transition subtypes (navigating_normally / searching / hesitant / looping)
  - ineffective_progress tension handling
  - puzzle_elapsed_ratio as first-class signal (no longer post-hoc escalation)
  - Stateful prompt agent with cooldown, escalation, and recovery
"""

import pandas as pd
from config import ACTION


# ---------------------------------------------------------------------------
# Transition subtype classification
# ---------------------------------------------------------------------------

def classify_transition(row: pd.Series) -> str:
    """
    Classify Transition windows into subtypes using available features.
    No head tracking → approximate from gaze + action patterns.
    """
    entropy = row.get("gaze_entropy", 0) or 0
    switch_rate = row.get("switch_rate", 0) or 0
    action_count = row.get("action_count", 0) or 0
    idle_time = row.get("idle_time", 0) or 0
    time_since = row.get("time_since_action", 0) or 0

    # Looping: high entropy + high switch + long time since action
    # Player has been in transition too long, likely lost
    if time_since > 120 and entropy > 1.5:
        return "looping_transition"

    # Hesitant: low activity + pausing during navigation
    if idle_time > 4.5 and action_count == 0 and time_since > 60:
        return "hesitant_transition"

    # Searching: high entropy + high switch rate = actively looking around
    if entropy > 1.5 and switch_rate > 5.0:
        return "searching_transition"

    # Normal navigation: moderate movement with low entropy (going somewhere)
    return "navigating_normally"


# ---------------------------------------------------------------------------
# Core support decision
# ---------------------------------------------------------------------------

def suggest_support(row: pd.Series) -> dict:
    """
    Determine support based on disagreement structure.
    V3: handles ineffective_progress + transition subtypes.
    """
    d_type = row.get("disagreement_type", "unstructured")
    d_intensity = row.get("disagreement_intensity", 0.0)
    tension = row.get("dominant_tension", "none")
    temporal = row.get("temporal_label", "transient")
    puzzle_id = row.get("puzzle_id", "")
    elapsed_ratio = row.get("puzzle_elapsed_ratio", 0.0) or 0.0
    time_since = row.get("time_since_action", 0) or 0

    att_label = row.get("attention_label", "unknown")
    act_label = row.get("action_label", "unknown")
    perf_label = row.get("performance_label", "unknown")
    perf_conf = row.get("performance_confidence", 0.0)

    # ── TRANSITION: subtype-based handling ───────────────────────────────
    if puzzle_id == "Transition":
        subtype = classify_transition(row)

        if subtype == "looping_transition":
            if temporal in ("looping", "persistent"):
                return _result("spatial_hint", 0.65,
                               "Lost in transition: high entropy + long duration. Needs direction.",
                               "consensus_intervene")
            return _result("gentle_probe", 0.5,
                           "Transition taking long with searching behavior. Probe.",
                           "probe")

        if subtype == "hesitant_transition":
            return _result("monitor_closely", 0.45,
                           "Hesitant during transition — pausing, may be unsure where to go.",
                           "probe")

        if subtype == "searching_transition":
            return _result("wait", 0.5,
                           "Actively searching during transition — likely exploring.",
                           "watch")

        # navigating_normally
        return _result("wait", 0.7,
                        "Normal navigation between puzzles.",
                        "watch")

    # ── CONSTRUCTIVE: agents largely agree ──────────────────────────────
    if d_type == "constructive":

        if tension == "frozen_on_clue":
            urgency = 0.8 if temporal == "looping" else 0.6
            return _result("reorientation_prompt", urgency,
                           f"Agents agree: fixated on clue but not acting ({temporal})",
                           "consensus_intervene")

        if tension in ("passive_and_stuck", "passive_and_ineffective"):
            urgency = 0.85 if temporal in ("looping", "persistent") else 0.5
            return _result("spatial_hint", urgency,
                           f"Agents agree: inactive and stalled ({temporal})",
                           "consensus_intervene")

        if tension == "hesitant_and_ineffective":
            return _result("gentle_probe", 0.6,
                           "Hesitant behavior + ineffective progress — probe to understand.",
                           "probe")

        if tension in ("focused_progress", "engaged_and_active", "active_exploration"):
            return _result("wait", 0.8,
                           f"Agents agree: productive engagement ({tension})",
                           "watch")

    # ── CONTRADICTORY: agents disagree ──────────────────────────────────
    if d_type == "contradictory":

        # Scanning but passive
        if tension == "scanning_but_passive":
            if temporal == "looping":
                return _result("gentle_probe", 0.6,
                               "Searching + inactive persistently. Probe: exploration or lost?",
                               "probe")
            return _result("monitor_closely", 0.5,
                           "Searching + inactive — transient. May resolve.",
                           "probe")

        # Focused but failing
        if tension == "focused_but_failing":
            return _result("procedural_hint", 0.7,
                           "Focused but errors occurring. Needs how-to hint.",
                           "consensus_intervene")

        # Focused but idle
        if tension == "focused_but_idle":
            if temporal in ("persistent", "looping") or time_since > 60:
                return _result("gentle_probe", 0.6,
                               f"Focused + no action ({'persistent' if temporal != 'transient' else f'{time_since:.0f}s'}). "
                               "Probe gently.", "probe")
            if elapsed_ratio > 2.0:
                return _result("monitor_closely", 0.5,
                               f"Focused + idle, puzzle at {elapsed_ratio:.1f}x median. Monitor.",
                               "probe")
            return _result("wait", 0.55,
                           "Focused + briefly idle — likely thinking. Wait.",
                           "watch")

        # Scattered but progressing
        if tension == "scattered_but_progressing":
            if temporal in ("looping", "persistent") or time_since > 90:
                return _result("gentle_probe", 0.55,
                               f"Scattered + nominally progressing but "
                               f"{'persistent' if temporal != 'transient' else f'{time_since:.0f}s since action'}. Probe.",
                               "probe")
            return _result("wait", 0.75,
                           "Scattered but IS progressing. Don't interrupt.",
                           "watch")

        # V3: Ineffective progress tensions
        if tension in ("focused_but_ineffective", "scattered_and_ineffective"):
            if temporal in ("looping", "persistent"):
                return _result("procedural_hint", 0.7,
                               f"Activity detected but progress is ineffective ({temporal}). "
                               "Player likely needs a new approach.",
                               "consensus_intervene")
            return _result("gentle_probe", 0.6,
                           "Active but progress seems ineffective. Probe to check.",
                           "probe")

        # Acting without progress
        if tension == "acting_without_progress":
            return _result("light_guidance", 0.55,
                           "Actions but no puzzle progress. May be interacting with wrong elements.",
                           "probe")

        # Active but failing
        if tension == "active_but_failing":
            if temporal in ("looping", "persistent"):
                return _result("redirect_hint", 0.7,
                               "High activity + repeated errors. Needs new direction.",
                               "consensus_intervene")
            return _result("monitor_closely", 0.5,
                           "Active + errors but may self-correct.",
                           "probe")

        # Idle but progressing (rare)
        if tension == "idle_but_progressing":
            return _result("wait", 0.7,
                           "Low activity but progress detected. Efficient player.",
                           "watch")

        # Fixated but acting (rare)
        if tension == "fixated_but_acting":
            return _result("monitor_closely", 0.5,
                           "Locked gaze but active interaction. Unusual.",
                           "probe")

    # ── UNSTRUCTURED: no clear pattern ──────────────────────────────────

    # V3: ineffective_progress as standalone signal
    if perf_label == "ineffective_progress" and perf_conf > 0.5:
        if temporal in ("looping", "persistent"):
            return _result("gentle_probe", 0.6,
                           "No clear agent pattern but persistent ineffective progress.",
                           "probe")
        return _result("monitor_closely", 0.45,
                       "Ineffective progress detected, monitoring.",
                       "probe")

    if perf_label in ("stalled",) and perf_conf > 0.7 and temporal == "looping":
        return _result("spatial_hint", 0.5,
                       "No clear pattern but persistent stalling detected.",
                       "consensus_intervene")

    if perf_label == "progressing":
        return _result("wait", 0.6,
                       "No clear pattern but player is progressing.",
                       "watch")

    return _result("monitor", 0.3,
                   "No clear signal from agent negotiation.",
                   "watch")


def _result(action, confidence, rationale, category):
    return {
        "action": action,
        "confidence": confidence,
        "rationale": rationale,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Stateful Prompt Agent (Part 5)
# ---------------------------------------------------------------------------

class PromptAgent:
    """
    Stateful decision wrapper that adds:
      - Cooldown: suppress new interventions within 20s of last prompt
      - Escalation: consecutive struggle windows → escalate from probe to intervene
      - Recovery: if player recovers, reset escalation state
    """

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
        """Apply stateful policy on top of base support decision."""
        window_start = row.get("window_start", 0) or 0
        puzzle_id = row.get("puzzle_id", "")
        category = base_result["category"]

        # Reset state on puzzle change
        if puzzle_id != self.current_puzzle:
            self.current_puzzle = puzzle_id
            self.consecutive_struggle = 0
            self.prompt_count = 0
            # Don't reset last_prompt_time — cooldown crosses puzzles

        # Track consecutive struggle
        if category in ("probe", "consensus_intervene"):
            self.consecutive_struggle += 1
        else:
            # Recovery: player seems OK → reset escalation
            if self.consecutive_struggle > 0 and category == "watch":
                self.consecutive_struggle = max(0, self.consecutive_struggle - 2)

        result = dict(base_result)

        # Cooldown: only suppress consecutive INTERVENTIONS, not probes
        # Probes are low-cost (just monitoring more closely), interventions are disruptive
        time_since_prompt = window_start - self.last_prompt_time
        if time_since_prompt < self.cooldown_sec and category == "consensus_intervene":
            result["category"] = "probe"  # downgrade to probe, don't suppress entirely
            result["rationale"] += f" [Downgraded: {time_since_prompt:.0f}s since last intervention]"

        # Fatigue: only suppress interventions after many prompts, keep probes
        if self.prompt_count >= self.max_prompts_per_puzzle and category == "consensus_intervene":
            result["category"] = "probe"
            result["rationale"] += f" [Downgraded: {self.prompt_count} prompts already on this puzzle]"

        # Escalation: consecutive struggle → upgrade probe to intervene
        if self.consecutive_struggle >= self.escalation_threshold and category == "probe":
            result["category"] = "consensus_intervene"
            result["confidence"] = min(result["confidence"] + 0.15, 0.95)
            result["rationale"] += (
                f" [Escalated: {self.consecutive_struggle} consecutive struggle windows]"
            )

        # Record prompt timing — only interventions trigger cooldown
        if result["category"] == "consensus_intervene":
            self.last_prompt_time = window_start
            self.prompt_count += 1

        self.last_category = result["category"]
        return result


def run_support(df: pd.DataFrame) -> pd.DataFrame:
    """Run support layer with stateful prompt agent per participant."""
    # First pass: compute base decisions (stateless)
    base_results = df.apply(suggest_support, axis=1)

    # Second pass: apply stateful prompt agent per participant
    final_results = []
    prompt_agents = {}  # one per participant

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
