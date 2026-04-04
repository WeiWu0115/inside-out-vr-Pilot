"""
Negotiation layer: agents challenge each other based on evidence and confidence.

Instead of simple pattern matching, the negotiation process:
1. Detects pairwise tensions between agents
2. Classifies disagreement type (contradictory, constructive, consensus)
3. Computes overall disagreement intensity
4. Produces a negotiation transcript showing the "debate"
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Tension definitions: which agent-label pairs conflict
# ---------------------------------------------------------------------------

CONTRADICTIONS = {
    # Attention × Behavioral
    ("attention", "searching", "action", "inactive"): "scanning_but_passive",
    ("attention", "focused", "action", "inactive"): "focused_but_idle",
    ("attention", "locked", "action", "active"): "fixated_but_acting",
    # Attention × Progress
    ("attention", "focused", "performance", "failing"): "focused_but_failing",
    ("attention", "searching", "performance", "progressing"): "scattered_but_progressing",
    ("attention", "focused", "performance", "ineffective_progress"): "focused_but_ineffective",
    ("attention", "searching", "performance", "ineffective_progress"): "scattered_and_ineffective",
    # Behavioral × Progress
    ("action", "active", "performance", "stalled"): "acting_without_progress",
    ("action", "active", "performance", "failing"): "active_but_failing",
    ("action", "inactive", "performance", "progressing"): "idle_but_progressing",
    # Population agent conflicts (kept for backward compat)
    ("population", "exploring", "attention", "focused"): "pop_says_exploring_but_focused",
    ("population", "disoriented", "attention", "searching"): "pop_says_stuck_but_searching",
    ("population", "actively_solving", "performance", "stalled"): "pop_says_solving_but_stalled",
    ("population", "cognitively_stuck", "action", "active"): "pop_says_stuck_but_active",
    ("population", "exploring", "action", "inactive"): "pop_says_exploring_but_inactive",
}

CONSTRUCTIVE_PAIRS = {
    ("attention", "focused", "action", "active"): "engaged_and_active",
    ("attention", "focused", "performance", "progressing"): "focused_progress",
    ("attention", "searching", "action", "active"): "active_exploration",
    ("action", "inactive", "performance", "stalled"): "passive_and_stuck",
    ("action", "inactive", "performance", "ineffective_progress"): "passive_and_ineffective",
    ("action", "hesitant", "performance", "ineffective_progress"): "hesitant_and_ineffective",
    ("attention", "locked", "action", "inactive"): "frozen_on_clue",
    # Population agent agreements
    ("population", "exploring", "attention", "searching"): "pop_confirms_exploration",
    ("population", "cognitively_stuck", "attention", "locked"): "pop_confirms_impasse",
    ("population", "disoriented", "action", "inactive"): "pop_confirms_disorientation",
    ("population", "actively_solving", "performance", "progressing"): "pop_confirms_progress",
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
    for name in ["attention", "action", "performance", "population"]:
        agents[name] = _get_agent_output(row, name)

    tensions = []

    # Check contradictions
    for (a1, l1, a2, l2), tension_name in CONTRADICTIONS.items():
        if agents[a1]["label"] == l1 and agents[a2]["label"] == l2:
            conf_a1 = agents[a1]["confidence"]
            conf_a2 = agents[a2]["confidence"]
            intensity = (conf_a1 + conf_a2) / 2  # both confident = high tension
            tensions.append({
                "type": "contradictory",
                "name": tension_name,
                "agents": (a1, a2),
                "labels": (l1, l2),
                "confidences": (conf_a1, conf_a2),
                "intensity": round(intensity, 3),
            })

    # Check constructive alignments
    for (a1, l1, a2, l2), align_name in CONSTRUCTIVE_PAIRS.items():
        if agents[a1]["label"] == l1 and agents[a2]["label"] == l2:
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
    """
    Compute the full disagreement structure for a time window.
    Returns type, intensity, dominant tension, and confidence spread.
    """
    agents = {}
    for name in ["attention", "action", "performance", "temporal"]:
        agents[name] = _get_agent_output(row, name)

    tensions = detect_tensions(row)

    contradictions = [t for t in tensions if t["type"] == "contradictory"]
    constructive = [t for t in tensions if t["type"] == "constructive"]

    # Confidence spread: how much do agents' confidences vary?
    confidences = [a["confidence"] for a in agents.values() if a["confidence"] > 0]
    conf_spread = float(np.std(confidences)) if len(confidences) > 1 else 0.0

    # Overall disagreement intensity
    if contradictions:
        # Weighted by the confidence of contradicting agents
        max_contradiction = max(c["intensity"] for c in contradictions)
        disagreement_intensity = max_contradiction
        disagreement_type = "contradictory"
        dominant_tension = max(contradictions, key=lambda c: c["intensity"])["name"]
    elif constructive:
        disagreement_intensity = 1.0 - max(c["intensity"] for c in constructive)
        disagreement_type = "constructive"
        dominant_tension = max(constructive, key=lambda c: c["intensity"])["name"]
    else:
        # No recognized pairwise pattern
        labels = [a["label"] for a in agents.values() if a["label"] not in ("unknown", "ambiguous")]
        n_unique = len(set(labels))
        disagreement_intensity = min(n_unique / 4.0, 1.0)
        disagreement_type = "unstructured"
        dominant_tension = "none"

    return {
        "disagreement_type": disagreement_type,
        "disagreement_intensity": round(disagreement_intensity, 3),
        "dominant_tension": dominant_tension,
        "n_contradictions": len(contradictions),
        "n_constructive": len(constructive),
        "confidence_spread": round(conf_spread, 3),
        "tensions": tensions,
    }


def generate_negotiation_transcript(row: pd.Series) -> str:
    """
    Generate a human-readable transcript of the negotiation process.
    Shows how agents "debate" the interpretation.
    """
    agents = {}
    for name in ["attention", "action", "performance", "temporal"]:
        agents[name] = {
            "label": row.get(f"{name}_label", "unknown"),
            "confidence": row.get(f"{name}_confidence", 0.0),
            "reasoning": row.get(f"{name}_reasoning", ""),
        }

    lines = []
    lines.append("=== Agent Interpretations ===")
    for name, a in agents.items():
        lines.append(
            f"  {name.title()} Agent: \"{a['label']}\" "
            f"(confidence: {a['confidence']:.0%}) — {a['reasoning']}"
        )

    tensions = detect_tensions(row)
    contradictions = [t for t in tensions if t["type"] == "contradictory"]
    constructive = [t for t in tensions if t["type"] == "constructive"]

    if contradictions:
        lines.append("")
        lines.append("=== Conflicts Detected ===")
        for c in sorted(contradictions, key=lambda x: -x["intensity"]):
            a1, a2 = c["agents"]
            l1, l2 = c["labels"]
            c1, c2 = c["confidences"]
            lines.append(
                f"  CONFLICT [{c['name']}]: "
                f"{a1.title()} says \"{l1}\" ({c1:.0%}) vs "
                f"{a2.title()} says \"{l2}\" ({c2:.0%}) "
                f"— intensity: {c['intensity']:.0%}"
            )

            # Agent "arguments"
            r1 = agents[a1]["reasoning"]
            r2 = agents[a2]["reasoning"]
            if r1:
                lines.append(f"    {a1.title()}: \"{r1}\"")
            if r2:
                lines.append(f"    {a2.title()}: \"{r2}\"")

    if constructive:
        lines.append("")
        lines.append("=== Agreements ===")
        for c in constructive:
            a1, a2 = c["agents"]
            l1, l2 = c["labels"]
            lines.append(
                f"  ALIGN [{c['name']}]: "
                f"{a1.title()} ({l1}) + {a2.title()} ({l2}) "
                f"— strength: {c['intensity']:.0%}"
            )

    # Resolution
    lines.append("")
    lines.append("=== Negotiation Outcome ===")
    d = compute_disagreement(row)
    support = row.get("suggested_support", "none")
    pattern = row.get("disagreement_pattern", "none")

    if d["disagreement_type"] == "contradictory":
        lines.append(
            f"  Agents cannot agree (type: {d['dominant_tension']}). "
            f"Intensity: {d['disagreement_intensity']:.0%}. "
            f"Decision: {support} (probe rather than assert)"
        )
    elif d["disagreement_type"] == "constructive":
        lines.append(
            f"  Agents align on {d['dominant_tension']}. "
            f"Confidence is convergent. "
            f"Decision: {support}"
        )
    else:
        lines.append(
            f"  No strong inter-agent pattern. "
            f"Decision: {support}"
        )

    return "\n".join(lines)


def run_negotiation(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full negotiation layer on the dataframe."""

    # Compute disagreement structure for each row
    disagreements = df.apply(compute_disagreement, axis=1)

    df["disagreement_type"] = disagreements.apply(lambda d: d["disagreement_type"])
    df["disagreement_intensity"] = disagreements.apply(lambda d: d["disagreement_intensity"])
    df["dominant_tension"] = disagreements.apply(lambda d: d["dominant_tension"])
    df["n_contradictions"] = disagreements.apply(lambda d: d["n_contradictions"])
    df["n_constructive"] = disagreements.apply(lambda d: d["n_constructive"])
    df["confidence_spread"] = disagreements.apply(lambda d: d["confidence_spread"])

    # Keep legacy columns for backward compat
    df["disagreement_score"] = df["n_contradictions"] + (df["disagreement_intensity"] * 4).round().astype(int)
    df["disagreement_pattern"] = df["dominant_tension"]

    return df
