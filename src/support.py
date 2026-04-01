"""
Support layer: maps disagreement structures to adaptive interventions.

The key insight: the TYPE of disagreement determines the TYPE of response,
not just whether to respond.
"""

import pandas as pd


def suggest_support(row: pd.Series) -> dict:
    """
    Determine support based on disagreement structure, not just agent labels.

    Returns:
        {
            "action": str,         # what to do
            "confidence": float,   # how sure the system is about this action
            "rationale": str,      # why this action
            "category": str,       # consensus / probe / watch
        }
    """
    d_type = row.get("disagreement_type", "unstructured")
    d_intensity = row.get("disagreement_intensity", 0.0)
    tension = row.get("dominant_tension", "none")
    temporal = row.get("temporal_label", "transient")

    att_label = row.get("attention_label", "unknown")
    att_conf = row.get("attention_confidence", 0.0)
    act_label = row.get("action_label", "unknown")
    act_conf = row.get("action_confidence", 0.0)
    perf_label = row.get("performance_label", "unknown")
    perf_conf = row.get("performance_confidence", 0.0)

    # ── CONSENSUS: agents largely agree ──────────────────────────────────
    if d_type == "constructive":

        # Frozen on clue: locked gaze + inactive → reorientation
        if tension == "frozen_on_clue":
            urgency = 0.8 if temporal == "looping" else 0.6
            return {
                "action": "reorientation_prompt",
                "confidence": urgency,
                "rationale": f"Agents agree: fixated on clue but not acting ({temporal})",
                "category": "consensus_intervene",
            }

        # Passive and stuck: inactive + stalled
        if tension == "passive_and_stuck":
            urgency = 0.85 if temporal in ("looping", "persistent") else 0.5
            return {
                "action": "spatial_hint",
                "confidence": urgency,
                "rationale": f"Agents agree: inactive and stalled ({temporal})",
                "category": "consensus_intervene",
            }

        # Positive consensus: focused + progressing
        if tension in ("focused_progress", "engaged_and_active", "active_exploration"):
            return {
                "action": "wait",
                "confidence": 0.8,
                "rationale": f"Agents agree: productive engagement ({tension})",
                "category": "watch",
            }

    # ── CONTRADICTORY: agents disagree on what's happening ───────────────
    if d_type == "contradictory":

        # Scanning but passive — is it exploration or disorientation?
        if tension == "scanning_but_passive":
            if temporal == "looping":
                return {
                    "action": "gentle_probe",
                    "confidence": 0.6,
                    "rationale": "Attention says searching, Action says inactive — persistent. "
                                 "Probe to distinguish exploration from disorientation.",
                    "category": "probe",
                }
            return {
                "action": "monitor_closely",
                "confidence": 0.5,
                "rationale": "Attention says searching, Action says inactive — transient. "
                             "May resolve on its own.",
                "category": "probe",
            }

        # Focused but failing — understands the goal but can't execute
        if tension == "focused_but_failing":
            return {
                "action": "procedural_hint",
                "confidence": 0.7,
                "rationale": "Attention is focused but errors are occurring. "
                             "Player likely needs a how-to hint, not a what-to-do hint.",
                "category": "consensus_intervene",
            }

        # Focused but idle — thinking or stuck?
        if tension == "focused_but_idle":
            if temporal == "persistent":
                return {
                    "action": "gentle_probe",
                    "confidence": 0.55,
                    "rationale": "Focused attention + no action for extended period. "
                                 "Could be deep thinking or paralysis. Probe gently.",
                    "category": "probe",
                }
            return {
                "action": "wait",
                "confidence": 0.6,
                "rationale": "Focused + briefly idle — likely thinking. Wait.",
                "category": "watch",
            }

        # Scattered attention but progressing
        if tension == "scattered_but_progressing":
            return {
                "action": "wait",
                "confidence": 0.75,
                "rationale": "Attention looks scattered but player IS progressing. "
                             "Don't interrupt what's working.",
                "category": "watch",
            }

        # Active but failing — trying hard but wrong approach
        if tension == "active_but_failing":
            if temporal in ("looping", "persistent"):
                return {
                    "action": "redirect_hint",
                    "confidence": 0.7,
                    "rationale": "High activity + repeated errors. "
                                 "Player is trying but needs a new direction.",
                    "category": "consensus_intervene",
                }
            return {
                "action": "monitor_closely",
                "confidence": 0.5,
                "rationale": "Active + errors but may self-correct. Monitor.",
                "category": "probe",
            }

        # Acting without progress
        if tension == "acting_without_progress":
            return {
                "action": "light_guidance",
                "confidence": 0.55,
                "rationale": "Actions occurring but no puzzle progress. "
                             "Player may be interacting with wrong elements.",
                "category": "probe",
            }

        # Idle but somehow progressing (rare)
        if tension == "idle_but_progressing":
            return {
                "action": "wait",
                "confidence": 0.7,
                "rationale": "Low activity but progress detected. "
                             "Player may be using minimal, efficient actions.",
                "category": "watch",
            }

        # Fixated but acting (rare)
        if tension == "fixated_but_acting":
            return {
                "action": "monitor_closely",
                "confidence": 0.5,
                "rationale": "Locked gaze but active interaction. Unusual pattern.",
                "category": "probe",
            }

    # ── UNSTRUCTURED: no clear inter-agent pattern ───────────────────────

    # Fallback: use individual agent signals
    if perf_label == "stalled" and perf_conf > 0.7 and temporal == "looping":
        return {
            "action": "spatial_hint",
            "confidence": 0.5,
            "rationale": "No clear agent pattern but persistent stalling detected.",
            "category": "consensus_intervene",
        }

    if perf_label == "progressing":
        return {
            "action": "wait",
            "confidence": 0.6,
            "rationale": "No clear pattern but player is progressing.",
            "category": "watch",
        }

    return {
        "action": "monitor",
        "confidence": 0.3,
        "rationale": "No clear signal from agent negotiation.",
        "category": "watch",
    }


def run_support(df: pd.DataFrame) -> pd.DataFrame:
    """Add support columns based on negotiation results."""
    support_results = df.apply(suggest_support, axis=1)

    df["suggested_support"] = support_results.apply(lambda s: s["action"])
    df["support_confidence"] = support_results.apply(lambda s: s["confidence"])
    df["support_rationale"] = support_results.apply(lambda s: s["rationale"])
    df["support_category"] = support_results.apply(lambda s: s["category"])

    return df
