"""
Population Agent: a data-driven agent that interprets the current window
by comparing it against the distribution of all prior players' behavior.

Instead of rules, this agent asks: "Based on what we've seen from 13 players,
what cognitive state does this behavior pattern most closely resemble?"

It uses the K-means cluster centroids from the FDG study as its "memory"
of population-level behavioral patterns.
"""

import os
import numpy as np
import pandas as pd
from load_data import safe_get


# ---------------------------------------------------------------------------
# Cluster profiles from FDG K=5 analysis
# ---------------------------------------------------------------------------

CLUSTER_PROFILES = {
    0: {
        "name": "transition",
        "description": "Moving between areas, moderate activity, no errors",
        "centroid": {
            "gaze_entropy": 1.076, "clue_ratio": 0.248, "switch_rate": 2.791,
            "action_count": 2.166, "idle_time": 0.717, "error_count": 0.0,
            "time_since_action": 93.287,
        },
        "std": {
            "gaze_entropy": 0.681, "clue_ratio": 0.333, "switch_rate": 2.601,
            "action_count": 3.356, "idle_time": 0.867, "error_count": 0.0,
            "time_since_action": 86.466,
        },
        "n": 477,
    },
    1: {
        "name": "waiting",
        "description": "Low entropy, minimal clue engagement, high idle, disoriented",
        "centroid": {
            "gaze_entropy": 0.958, "clue_ratio": 0.089, "switch_rate": 2.328,
            "action_count": 2.213, "idle_time": 4.657, "error_count": 0.0,
            "time_since_action": 79.046,
        },
        "std": {
            "gaze_entropy": 0.577, "clue_ratio": 0.137, "switch_rate": 1.726,
            "action_count": 3.377, "idle_time": 0.431, "error_count": 0.0,
            "time_since_action": 85.843,
        },
        "n": 2109,
    },
    2: {
        "name": "active_solving",
        "description": "Moderate entropy, active interaction, errors present",
        "centroid": {
            "gaze_entropy": 0.998, "clue_ratio": 0.042, "switch_rate": 2.542,
            "action_count": 2.544, "idle_time": 3.677, "error_count": 0.329,
            "time_since_action": 41.371,
        },
        "std": {
            "gaze_entropy": 0.647, "clue_ratio": 0.102, "switch_rate": 2.405,
            "action_count": 2.592, "idle_time": 1.259, "error_count": 0.704,
            "time_since_action": 76.556,
        },
        "n": 562,
    },
    3: {
        "name": "exploration",
        "description": "High entropy, high switch rate, scanning environment",
        "centroid": {
            "gaze_entropy": 2.347, "clue_ratio": 0.153, "switch_rate": 10.437,
            "action_count": 2.896, "idle_time": 4.092, "error_count": 0.0,
            "time_since_action": 98.728,
        },
        "std": {
            "gaze_entropy": 0.708, "clue_ratio": 0.162, "switch_rate": 4.494,
            "action_count": 3.998, "idle_time": 0.514, "error_count": 0.0,
            "time_since_action": 108.744,
        },
        "n": 1199,
    },
    4: {
        "name": "stuck_on_clue",
        "description": "Low entropy, very high clue ratio, long time since action, cognitive impasse",
        "centroid": {
            "gaze_entropy": 0.794, "clue_ratio": 0.781, "switch_rate": 2.294,
            "action_count": 1.246, "idle_time": 4.732, "error_count": 0.0,
            "time_since_action": 158.486,
        },
        "std": {
            "gaze_entropy": 0.585, "clue_ratio": 0.206, "switch_rate": 2.096,
            "action_count": 2.761, "idle_time": 0.393, "error_count": 0.0,
            "time_since_action": 145.557,
        },
        "n": 918,
    },
}

FEATURES = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time", "error_count", "time_since_action",
]


def _mahalanobis_like_distance(values, centroid, std):
    """
    Compute a standardized distance from the current window to a cluster centroid.
    Uses per-feature standard deviation for normalization (diagonal Mahalanobis).
    """
    dist = 0.0
    n_features = 0
    for feat in FEATURES:
        v = values.get(feat)
        c = centroid.get(feat)
        s = std.get(feat, 1.0)
        if v is None or c is None:
            continue
        s = max(s, 0.01)  # avoid division by zero
        dist += ((v - c) / s) ** 2
        n_features += 1
    if n_features == 0:
        return float("inf")
    return np.sqrt(dist / n_features)


def _distances_to_confidences(distances):
    """
    Convert distances to confidence scores using softmin.
    Closer clusters get higher confidence.
    """
    if not distances:
        return {}
    # Negative exponential: closer = higher score
    scores = {cid: np.exp(-d) for cid, d in distances.items()}
    total = sum(scores.values())
    if total == 0:
        return {cid: 0.0 for cid in distances}
    return {cid: s / total for cid, s in scores.items()}


def population_agent(row: pd.Series) -> dict:
    """
    Compare the current window against population-level cluster centroids.
    Returns the closest cluster as the interpretation, with confidence
    based on relative distance to all clusters.
    """
    # Extract features
    values = {}
    for feat in FEATURES:
        v = safe_get(row, feat)
        if v is not None:
            values[feat] = float(v)

    if len(values) < 3:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "evidence": {},
            "reasoning": "Insufficient features for population comparison",
            "all_scores": {},
            "distances": {},
        }

    # Compute distance to each cluster
    distances = {}
    for cid, profile in CLUSTER_PROFILES.items():
        distances[cid] = _mahalanobis_like_distance(
            values, profile["centroid"], profile["std"]
        )

    # Convert to confidence scores
    confidences = _distances_to_confidences(distances)

    # Best match
    best_cid = min(distances, key=distances.get)
    best_profile = CLUSTER_PROFILES[best_cid]
    best_conf = confidences[best_cid]

    # Second best for ambiguity check
    sorted_cids = sorted(distances, key=distances.get)
    if len(sorted_cids) > 1:
        second_cid = sorted_cids[1]
        second_conf = confidences[second_cid]
        gap = best_conf - second_conf
    else:
        gap = 1.0

    # If top two are very close, flag as ambiguous
    ambiguity_note = ""
    if gap < 0.1:
        second_name = CLUSTER_PROFILES[sorted_cids[1]]["name"]
        ambiguity_note = f" (close to {second_name}, gap={gap:.2f})"

    # Map cluster to a cognitive state label
    label_map = {
        "transition": "transitioning",
        "waiting": "disoriented",
        "active_solving": "actively_solving",
        "exploration": "exploring",
        "stuck_on_clue": "cognitively_stuck",
    }

    label = label_map.get(best_profile["name"], best_profile["name"])

    return {
        "label": label,
        "confidence": round(float(best_conf), 3),
        "evidence": {
            "closest_cluster": f"C{best_cid}: {best_profile['name']}",
            "distance": round(distances[best_cid], 3),
            "population_prevalence": f"{best_profile['n']}/5265 ({best_profile['n']/5265:.0%})",
        },
        "reasoning": (
            f"Closest to C{best_cid} ({best_profile['name']}): "
            f"{best_profile['description']}{ambiguity_note}"
        ),
        "all_scores": {
            CLUSTER_PROFILES[cid]["name"]: round(conf, 3)
            for cid, conf in confidences.items()
        },
        "distances": {
            CLUSTER_PROFILES[cid]["name"]: round(d, 3)
            for cid, d in distances.items()
        },
    }
