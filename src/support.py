"""
V3.1 Support layer: Intervention Necessity Score + Stateful Prompt Agent.

Key changes from V3:
  - Intervention Necessity Score (INS) replaces rule-based tension→action mapping
    for final watch/probe/intervene decision
  - Redundancy suppression: tracks episodes, suppresses repeated triggers
  - Pre-collapse detection: momentum trending down + elapsed rising → early probe
  - Progress momentum integrated into scoring
"""

import pandas as pd
from config import ACTION


# ---------------------------------------------------------------------------
# Transition subtype classification (unchanged from V3)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Intervention Necessity Score (Part 3)
# ---------------------------------------------------------------------------

def compute_intervention_score(row: pd.Series) -> float:
    """
    Weighted combination of signals producing a single score [0, 1].
    Higher = more likely intervention is needed.

    Weights calibrated against facilitator ground truth:
      + ineffective/degrading/stalled confidence (struggle signal)
      + temporal persistence (duration signal)
      + elapsed_ratio (macro-time signal)
      - progress_momentum (counter-signal: positive momentum suppresses)
      + pre-collapse bonus (anticipatory signal)
    """
    perf_label = row.get("performance_label", "progressing")
    perf_conf = row.get("performance_confidence", 0.0) or 0.0
    temporal = row.get("temporal_label", "transient")
    elapsed_ratio = row.get("puzzle_elapsed_ratio", 0) or 0.0
    momentum = row.get("progress_momentum", 0) or 0.0
    behavioral = row.get("action_label", "unknown")
    d_intensity = row.get("disagreement_intensity", 0) or 0.0

    # Component 1: Struggle signal from Progress Agent
    struggle_signal = 0.0
    if perf_label in ("ineffective_progress", "degrading", "stalled"):
        struggle_signal = perf_conf
    elif perf_label == "progressing":
        struggle_signal = max(0, 0.2 - perf_conf * 0.3)  # low when truly progressing

    # Component 2: Temporal persistence
    temporal_signal = {"transient": 0.0, "persistent": 0.5, "looping": 0.9}.get(temporal, 0.0)

    # Component 3: Elapsed ratio (scaled 0-1, saturates at 3x)
    elapsed_signal = min(max(elapsed_ratio - 1.0, 0) / 2.0, 1.0)

    # Component 4: Momentum (negative = bad)
    momentum_signal = max(-momentum, 0)  # 0 when positive, up to 1 when very negative

    # Component 5: Pre-collapse detection (Part 4)
    pre_collapse = 0.0
    if momentum < -0.1 and elapsed_ratio > 1.2 and behavioral in ("hesitant", "inactive"):
        pre_collapse = 0.4
    if momentum < -0.2 and elapsed_ratio > 1.5:
        pre_collapse = 0.6

    # Component 6: Disagreement intensity (high = agents conflict = uncertainty)
    disagreement_signal = d_intensity * 0.3

    # Weighted combination
    score = (
        0.30 * struggle_signal
        + 0.20 * temporal_signal
        + 0.15 * elapsed_signal
        + 0.15 * momentum_signal
        + 0.10 * pre_collapse
        + 0.10 * disagreement_signal
    )

    return round(min(max(score, 0), 1), 4)


# ---------------------------------------------------------------------------
# Core support decision
# ---------------------------------------------------------------------------

# INS thresholds — calibrated for F1 optimization
INS_PROBE_THRESHOLD = 0.20
INS_INTERVENE_THRESHOLD = 0.50

def suggest_support(row: pd.Series) -> dict:
    """
    V3.1: Two-stage decision.
      Stage 1: Tension-based reasoning (determines action type and rationale)
      Stage 2: INS threshold (determines category: watch/probe/intervene)
    """
    d_type = row.get("disagreement_type", "unstructured")
    tension = row.get("dominant_tension", "none")
    temporal = row.get("temporal_label", "transient")
    puzzle_id = row.get("puzzle_id", "")
    elapsed_ratio = row.get("puzzle_elapsed_ratio", 0.0) or 0.0
    momentum = row.get("progress_momentum", 0) or 0.0
    perf_label = row.get("performance_label", "unknown")
    behavioral = row.get("action_label", "unknown")

    # ── TRANSITION ──────────────────────────────────────────────────────
    if puzzle_id == "Transition":
        subtype = classify_transition(row)
        if subtype == "looping_transition" and temporal in ("looping", "persistent"):
            return _result("spatial_hint", 0.65,
                           "Lost in transition: prolonged searching.", "consensus_intervene")
        if subtype in ("hesitant_transition", "looping_transition"):
            return _result("gentle_probe", 0.45,
                           f"Transition: {subtype}.", "probe")
        return _result("wait", 0.7, "Normal navigation.", "watch")

    # ── Compute Intervention Necessity Score ─────────────────────────
    ins = compute_intervention_score(row)

    # ── Stage 1: Determine action type from tension (for rationale) ──
    action, rationale = _tension_to_action(tension, d_type, temporal, momentum,
                                            elapsed_ratio, perf_label, behavioral)

    # ── Stage 2: INS determines category ─────────────────────────────
    if ins >= INS_INTERVENE_THRESHOLD:
        category = "consensus_intervene"
    elif ins >= INS_PROBE_THRESHOLD:
        category = "probe"
    else:
        category = "watch"

    # Override: if tension analysis strongly suggests watch (focused_progress,
    # engaged_and_active), don't let a borderline INS trigger probe
    if tension in ("focused_progress", "engaged_and_active", "active_exploration") and ins < INS_INTERVENE_THRESHOLD:
        category = "watch"

    # Override: pre-collapse always at least probe
    if momentum < -0.2 and elapsed_ratio > 1.5 and behavioral in ("hesitant", "inactive"):
        if category == "watch":
            category = "probe"
            rationale += " [Pre-collapse: momentum declining + over time]"

    return _result(action, round(ins, 3), f"{rationale} [INS={ins:.3f}]", category)


def _tension_to_action(tension, d_type, temporal, momentum, elapsed_ratio, perf_label, behavioral):
    """Map tension to action type and rationale (Stage 1)."""

    # Constructive tensions
    if d_type == "constructive":
        if tension == "frozen_on_clue":
            return "reorientation_prompt", "Fixated on clue but not acting"
        if tension in ("passive_and_stuck", "passive_and_ineffective", "passive_and_degrading"):
            return "spatial_hint", f"Inactive and {perf_label}"
        if tension == "hesitant_and_ineffective":
            return "gentle_probe", "Hesitant + ineffective progress"
        if tension in ("focused_progress", "engaged_and_active", "active_exploration"):
            return "wait", f"Productive engagement ({tension})"
        return "wait", f"Constructive: {tension}"

    # Contradictory tensions
    if d_type == "contradictory":
        if tension == "scanning_but_passive":
            return "gentle_probe", "Searching + inactive"
        if tension == "focused_but_failing":
            return "procedural_hint", "Focused but errors occurring"
        if tension == "focused_but_idle":
            return "gentle_probe", f"Focused + no action ({temporal})"
        if tension == "scattered_but_progressing":
            return ("gentle_probe" if temporal != "transient" else "wait"), "Scattered + nominally progressing"
        if tension in ("focused_but_ineffective", "scattered_and_ineffective"):
            return "gentle_probe", f"Active but ineffective ({momentum:+.2f} momentum)"
        if tension in ("focused_but_degrading", "scattered_and_degrading", "active_but_degrading"):
            return "procedural_hint", f"Degrading (momentum={momentum:+.2f})"
        if tension == "acting_without_progress":
            return "light_guidance", "Actions but no puzzle progress"
        if tension == "active_but_failing":
            return "redirect_hint", "Active + repeated errors"
        if tension == "idle_but_progressing":
            return "wait", "Low activity but progressing"
        return "monitor_closely", f"Contradictory: {tension}"

    # Unstructured
    if perf_label in ("degrading", "ineffective_progress", "stalled"):
        return "gentle_probe", f"No clear pattern but {perf_label}"
    if perf_label == "progressing":
        return "wait", "No clear pattern, progressing"
    return "monitor", "No clear signal"


def _result(action, confidence, rationale, category):
    return {
        "action": action,
        "confidence": confidence,
        "rationale": rationale,
        "category": category,
    }


# ---------------------------------------------------------------------------
# Stateful Prompt Agent with Redundancy Suppression (Parts 2 & 5)
# ---------------------------------------------------------------------------

class PromptAgent:
    """
    V3.1 stateful decision wrapper:
      - Episode tracking: assigns windows to struggle episodes
      - Redundancy suppression: suppresses repeated triggers in same episode
      - Escalation: consecutive struggle → upgrade probe to intervene
      - Cooldown: suppress interventions within cooldown window
      - Recovery: reset on improvement
    """

    def __init__(self, cooldown_sec=15.0, escalation_threshold=5,
                 max_prompts_per_puzzle=8, redundancy_window=25.0):
        self.cooldown_sec = cooldown_sec
        self.escalation_threshold = escalation_threshold
        self.max_prompts_per_puzzle = max_prompts_per_puzzle
        self.redundancy_window = redundancy_window
        self.reset()

    def reset(self):
        self.last_intervene_time = -999.0
        self.last_trigger_time = -999.0
        self.consecutive_struggle = 0
        self.prompt_count = 0
        self.current_puzzle = None
        self.last_tension = None
        self.tension_repeat_count = 0
        self.episode_id = 0
        self.in_episode = False

    def decide(self, base_result: dict, row: pd.Series) -> dict:
        window_start = row.get("window_start", 0) or 0
        puzzle_id = row.get("puzzle_id", "")
        category = base_result["category"]
        tension = row.get("dominant_tension", "none")

        # Reset on puzzle change
        if puzzle_id != self.current_puzzle:
            self.current_puzzle = puzzle_id
            self.consecutive_struggle = 0
            self.prompt_count = 0
            self.tension_repeat_count = 0
            self.last_tension = None
            self.in_episode = False

        # Track consecutive struggle for escalation
        if category in ("probe", "consensus_intervene"):
            self.consecutive_struggle += 1
            if not self.in_episode:
                self.in_episode = True
                self.episode_id += 1
        else:
            if self.consecutive_struggle > 0:
                self.consecutive_struggle = max(0, self.consecutive_struggle - 2)
            if self.consecutive_struggle == 0:
                self.in_episode = False

        # Track tension repetition
        if tension == self.last_tension:
            self.tension_repeat_count += 1
        else:
            self.tension_repeat_count = 0
            self.last_tension = tension

        result = dict(base_result)

        # --- Redundancy suppression (Part 2) ---
        time_since_trigger = window_start - self.last_trigger_time

        # Same episode + recent trigger → suppress (but allow escalation)
        if self.in_episode and time_since_trigger < self.redundancy_window:
            if category == "probe" and result.get("_prev_was_probe", False):
                result["category"] = "watch"
                result["rationale"] += f" [Redundancy: {time_since_trigger:.0f}s since last trigger in episode]"
                return result
            # Allow probe→intervene escalation
            if category == "consensus_intervene" and time_since_trigger < self.cooldown_sec:
                result["category"] = "probe"
                result["rationale"] += f" [Cooldown: {time_since_trigger:.0f}s since last trigger]"

        # Same tension pattern > 3 times → suppress unless state changed
        if self.tension_repeat_count > 3 and category == "probe":
            result["category"] = "watch"
            result["rationale"] += f" [Suppressed: same tension '{tension}' repeated {self.tension_repeat_count}x]"
            return result

        # --- Cooldown for interventions ---
        time_since_intervene = window_start - self.last_intervene_time
        if time_since_intervene < self.cooldown_sec and category == "consensus_intervene":
            result["category"] = "probe"
            result["rationale"] += f" [Cooldown: {time_since_intervene:.0f}s since last intervention]"

        # --- Fatigue ---
        if self.prompt_count >= self.max_prompts_per_puzzle and category == "consensus_intervene":
            result["category"] = "probe"
            result["rationale"] += f" [Fatigue: {self.prompt_count} interventions on this puzzle]"

        # --- Escalation ---
        if self.consecutive_struggle >= self.escalation_threshold and category == "probe":
            result["category"] = "consensus_intervene"
            result["confidence"] = min(float(result.get("confidence", 0.5)) + 0.1, 0.95)
            result["rationale"] += f" [Escalated: {self.consecutive_struggle} consecutive struggle windows]"

        # Record timing
        if result["category"] == "consensus_intervene":
            self.last_intervene_time = window_start
            self.last_trigger_time = window_start
            self.prompt_count += 1
        elif result["category"] == "probe":
            self.last_trigger_time = window_start

        return result


def run_support(df: pd.DataFrame) -> pd.DataFrame:
    """Run support with INS scoring + stateful prompt agent."""
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
