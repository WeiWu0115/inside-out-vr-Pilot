"""
V4 Negotiation layer: gaze-dominant agent tensions.

Agents: fixation, semantics, motor, behavioral (action), temporal.
Detects pairwise tensions between agents and computes disagreement structure.
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Tension definitions: which agent-label pairs conflict or align
# ---------------------------------------------------------------------------

CONTRADICTIONS = {
    # Fixation × Behavioral: gaze pattern contradicts physical action
    ("fixation", "locked", "action", "active"): "locked_gaze_but_active",
    ("fixation", "scanning", "action", "inactive"): "scanning_but_passive",
    ("fixation", "focused", "action", "inactive"): "focused_but_idle",
    ("fixation", "locked", "action", "inactive"): "frozen",

    # Fixation × Semantics: gaze pattern contradicts what's being looked at
    ("fixation", "scanning", "semantics", "fixated_on_clue"): "rapid_clue_switching",
    ("fixation", "locked", "semantics", "environmental_scanning"): "locked_on_environment",
    ("fixation", "locked", "semantics", "unfocused"): "locked_but_unfocused",

    # Semantics × Behavioral: looking at task but not acting (or vice versa)
    ("semantics", "fixated_on_clue", "action", "inactive"): "reading_but_not_acting",
    ("semantics", "task_focused", "action", "inactive"): "watching_task_but_idle",
    ("semantics", "environmental_scanning", "action", "active"): "acting_while_looking_away",
    ("semantics", "unfocused", "action", "active"): "acting_without_looking",

    # Motor × Behavioral: gaze motor pattern contradicts physical action
    ("motor", "passive_scanning", "action", "active"): "passive_gaze_active_hands",
    ("motor", "erratic", "action", "inactive"): "erratic_gaze_no_action",
    ("motor", "concentrated", "action", "failing"): "concentrated_but_failing",

    # Motor × Fixation: motor pattern contradicts fixation pattern
    ("motor", "erratic", "fixation", "focused"): "erratic_motor_focused_fixation",
    ("motor", "concentrated", "fixation", "scanning"): "concentrated_motor_scanning_fixation",

    # Semantics × Motor: what you look at contradicts how you look
    ("semantics", "fixated_on_clue", "motor", "erratic"): "erratic_clue_reading",
    ("semantics", "unfocused", "motor", "concentrated"): "concentrated_but_off_task",
}

CONSTRUCTIVE_PAIRS = {
    # Positive engagement: gaze + action aligned
    ("fixation", "focused", "action", "active"): "engaged_and_active",
    ("semantics", "task_focused", "action", "active"): "task_engaged",
    ("semantics", "fixated_on_clue", "fixation", "focused"): "deep_clue_reading",
    ("motor", "purposeful", "action", "active"): "purposeful_action",
    ("motor", "purposeful", "semantics", "task_focused"): "purposeful_task_gaze",

    # Stuck patterns: multiple agents agree player is stuck
    ("fixation", "locked", "semantics", "fixated_on_clue"): "frozen_on_clue",
    ("semantics", "environmental_scanning", "action", "inactive"): "lost_and_passive",
    ("semantics", "unfocused", "action", "inactive"): "disengaged",
    ("motor", "passive_scanning", "action", "inactive"): "passive_and_idle",
    ("motor", "erratic", "action", "failing"): "panicking",
    ("fixation", "revisiting", "action", "hesitant"): "uncertain_checking",

    # Exploration: healthy searching
    ("fixation", "scanning", "action", "active"): "active_exploration",
    ("semantics", "environmental_scanning", "fixation", "scanning"): "systematic_search",
    ("motor", "purposeful", "fixation", "scanning"): "purposeful_scanning",
}


def _get_agent_output(row, agent_name):
    """Extract agent output from row columns."""
    return {
        "label": row.get(f"{agent_name}_label", "unknown"),
        "confidence": row.get(f"{agent_name}_confidence", 0.0),
    }


def detect_tensions(row: pd.Series) -> list:
    """Find all pairwise tensions between agents."""
    agents = {}
    for name in ["fixation", "semantics", "motor", "action"]:
        agents[name] = _get_agent_output(row, name)

    tensions = []

    for (a1, l1, a2, l2), tension_name in CONTRADICTIONS.items():
        if agents.get(a1, {}).get("label") == l1 and agents.get(a2, {}).get("label") == l2:
            conf_a1 = agents[a1]["confidence"]
            conf_a2 = agents[a2]["confidence"]
            intensity = (conf_a1 + conf_a2) / 2
            tensions.append({
                "type": "contradictory",
                "name": tension_name,
                "agents": (a1, a2),
                "labels": (l1, l2),
                "confidences": (conf_a1, conf_a2),
                "intensity": round(intensity, 3),
            })

    for (a1, l1, a2, l2), align_name in CONSTRUCTIVE_PAIRS.items():
        if agents.get(a1, {}).get("label") == l1 and agents.get(a2, {}).get("label") == l2:
            conf_a1 = agents[a1]["confidence"]
            conf_a2 = agents[a2]["confidence"]
            intensity = (conf_a1 + conf_a2) / 2
            tensions.append({
                "type": "constructive",
                "name": align_name,
                "agents": (a1, a2),
                "labels": (l1, l2),
                "confidences": (conf_a1, conf_a2),
                "intensity": round(intensity, 3),
            })

    return tensions


def compute_disagreement(row: pd.Series) -> dict:
    """Compute full disagreement structure for a time window."""
    agents = {}
    for name in ["fixation", "semantics", "motor", "action"]:
        agents[name] = _get_agent_output(row, name)

    tensions = detect_tensions(row)
    contradictions = [t for t in tensions if t["type"] == "contradictory"]
    constructive = [t for t in tensions if t["type"] == "constructive"]

    confidences = [a["confidence"] for a in agents.values() if a["confidence"] > 0]
    conf_spread = float(np.std(confidences)) if len(confidences) > 1 else 0.0

    if contradictions:
        max_contradiction = max(c["intensity"] for c in contradictions)
        return {
            "disagreement_type": "contradictory",
            "disagreement_intensity": round(max_contradiction, 3),
            "dominant_tension": max(contradictions, key=lambda c: c["intensity"])["name"],
            "n_contradictions": len(contradictions),
            "n_constructive": len(constructive),
            "confidence_spread": round(conf_spread, 3),
            "tensions": tensions,
        }
    elif constructive:
        return {
            "disagreement_type": "constructive",
            "disagreement_intensity": round(1.0 - max(c["intensity"] for c in constructive), 3),
            "dominant_tension": max(constructive, key=lambda c: c["intensity"])["name"],
            "n_contradictions": 0,
            "n_constructive": len(constructive),
            "confidence_spread": round(conf_spread, 3),
            "tensions": tensions,
        }
    else:
        labels = [a["label"] for a in agents.values() if a["label"] not in ("unknown", "ambiguous", "no_data")]
        n_unique = len(set(labels))
        return {
            "disagreement_type": "unstructured",
            "disagreement_intensity": round(min(n_unique / 4.0, 1.0), 3),
            "dominant_tension": "none",
            "n_contradictions": 0,
            "n_constructive": 0,
            "confidence_spread": round(conf_spread, 3),
            "tensions": tensions,
        }


def generate_negotiation_transcript(row: pd.Series) -> str:
    """Generate human-readable negotiation transcript."""
    agents = {}
    for name in ["fixation", "semantics", "motor", "action", "temporal"]:
        agents[name] = {
            "label": row.get(f"{name}_label", "unknown"),
            "confidence": row.get(f"{name}_confidence", 0.0),
            "reasoning": row.get(f"{name}_reasoning", ""),
        }

    lines = ["=== Agent Interpretations ==="]
    for name, a in agents.items():
        lines.append(f"  {name.title()} Agent: \"{a['label']}\" ({a['confidence']:.0%}) — {a['reasoning']}")

    tensions = detect_tensions(row)
    contradictions = [t for t in tensions if t["type"] == "contradictory"]
    constructive = [t for t in tensions if t["type"] == "constructive"]

    if contradictions:
        lines.append("\n=== Conflicts ===")
        for c in sorted(contradictions, key=lambda x: -x["intensity"]):
            lines.append(f"  [{c['name']}]: {c['agents'][0]}.{c['labels'][0]} vs {c['agents'][1]}.{c['labels'][1]} — {c['intensity']:.0%}")

    if constructive:
        lines.append("\n=== Agreements ===")
        for c in constructive:
            lines.append(f"  [{c['name']}]: {c['agents'][0]}.{c['labels'][0]} + {c['agents'][1]}.{c['labels'][1]} — {c['intensity']:.0%}")

    return "\n".join(lines)


def run_negotiation(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full negotiation layer."""
    disagreements = df.apply(compute_disagreement, axis=1)

    df["disagreement_type"] = disagreements.apply(lambda d: d["disagreement_type"])
    df["disagreement_intensity"] = disagreements.apply(lambda d: d["disagreement_intensity"])
    df["dominant_tension"] = disagreements.apply(lambda d: d["dominant_tension"])
    df["n_contradictions"] = disagreements.apply(lambda d: d["n_contradictions"])
    df["n_constructive"] = disagreements.apply(lambda d: d["n_constructive"])
    df["confidence_spread"] = disagreements.apply(lambda d: d["confidence_spread"])

    # Legacy compat
    df["disagreement_score"] = df["n_contradictions"] + (df["disagreement_intensity"] * 4).round().astype(int)
    df["disagreement_pattern"] = df["dominant_tension"]

    return df
