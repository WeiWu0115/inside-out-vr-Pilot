"""
Gaze-Action Coupling: classify each interaction as informed vs blind.

For each PuzzleLogs interaction, look at what the player was gazing at
in the 3 seconds before the action:
  - informed: gazed at relevant clue or the puzzle object being interacted with
  - blind: gazed at environment/unrelated objects
  - misguided: gazed at a different puzzle object

Then aggregate per 5-second window to produce:
  informed_action_ratio, blind_action_ratio, misguided_action_ratio,
  gaze_action_latency (time between looking at target and acting on it)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


def _parse_ts(ts):
    try:
        if "." in ts:
            base, frac = ts.split(".")
            frac = frac.rstrip("Z")[:6]
            clean = f"{base}.{frac}+00:00"
        else:
            clean = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(clean).timestamp()
    except Exception:
        return None


def _extract_puzzle_name(parent_chain):
    """Extract puzzle name from ParentChain."""
    if parent_chain is None or pd.isna(parent_chain):
        return ""
    pc = str(parent_chain)
    for spoke in ["Pasta in Sauce", "Amount of Protein", "Water Amount", "Amount of Sunlight"]:
        if spoke in pc:
            return spoke
    if "Cooking Pot" in pc or "Hub" in pc:
        return "Cooking Pot"
    return ""


def _extract_object_keywords(parent_chain):
    """Extract keywords from the interacted object's ParentChain."""
    if parent_chain is None or pd.isna(parent_chain):
        return set()
    pc = str(parent_chain).lower()
    # Extract the last element in the chain (the actual object)
    parts = pc.split("->")
    if parts:
        obj = parts[-1].strip()
        # Get meaningful words (skip short ones)
        words = set(w for w in obj.replace("(", " ").replace(")", " ").split() if len(w) > 2)
        return words
    return set()


CLUE_KEYWORDS = {"diary", "hint", "note", "instruction", "whitboard", "chalkboard",
                 "chalkbaord", "seedpacket"}
PUZZLE_KEYWORDS = {"puzzle", "snap", "jar", "pot", "seed", "plant", "carrot", "potato",
                   "tomato", "water", "sunlight", "protein", "pasta", "sauce", "cooking",
                   "cup", "order", "amount", "bean", "beands", "plot", "lid",
                   "spoon", "ladle", "bowl", "socket", "token", "liquid", "fill",
                   "egg", "fish", "tofu", "peanut", "penuts"}
ENV_KEYWORDS = {"wall", "ground", "floor", "glass", "exterior", "greenhouse",
                "ceiling", "roof", "door", "table", "shelf", "cube"}


def _classify_gaze_target(name):
    if name is None or pd.isna(name):
        return "unknown"
    low = str(name).lower()
    if any(k in low for k in CLUE_KEYWORDS):
        return "clue"
    if any(k in low for k in PUZZLE_KEYWORDS):
        return "puzzle"
    if any(k in low for k in ENV_KEYWORDS):
        return "environment"
    return "other"


def _is_gaze_relevant(gaze_targets, action_puzzle_name, action_keywords):
    """
    Check if any gaze target in the pre-action window is relevant to the action.
    Returns: 'informed', 'clue_informed', 'blind', or 'misguided'
    """
    if len(gaze_targets) == 0:
        return "no_data"

    categories = [_classify_gaze_target(t) for t in gaze_targets]
    target_names = [str(t).lower() for t in gaze_targets]

    # Check 1: Did they look at a clue before acting?
    looked_at_clue = "clue" in categories

    # Check 2: Did they look at the same puzzle object they're about to interact with?
    looked_at_target = False
    for name in target_names:
        # Check if any keyword from the action object appears in the gaze target
        if action_keywords and any(kw in name for kw in action_keywords if len(kw) > 3):
            looked_at_target = True
            break
        # Check if puzzle name appears
        if action_puzzle_name and action_puzzle_name.lower() in name:
            looked_at_target = True
            break

    # Check 3: Did they look at any puzzle object at all?
    looked_at_puzzle = "puzzle" in categories

    # Check 4: Were they mostly looking at environment?
    env_ratio = categories.count("environment") / len(categories) if categories else 0

    if looked_at_target or looked_at_clue:
        return "informed"
    elif looked_at_puzzle:
        return "misguided"  # looked at puzzle stuff, just not the right one
    elif env_ratio > 0.7:
        return "blind"
    else:
        return "blind"


def extract_coupling_for_user(user_id, lookback_sec=3.0):
    """
    For one user, classify each interaction as informed/blind/misguided
    based on pre-action gaze, then aggregate per 5-second window.
    """
    tracking_path = f"User-{user_id}/User-{user_id}_PlayerTracking.csv"
    puzzle_path = f"User-{user_id}/User-{user_id}_PuzzleLogs.csv"

    if not os.path.exists(tracking_path) or not os.path.exists(puzzle_path):
        return pd.DataFrame()

    # Load tracking
    tracking = pd.read_csv(tracking_path, low_memory=False,
                           usecols=["Timestamp", "GazeTarget_ObjectName"])
    tracking["ts"] = tracking["Timestamp"].apply(_parse_ts)
    tracking = tracking.dropna(subset=["ts"])
    if len(tracking) == 0:
        return pd.DataFrame()
    t_start = tracking["ts"].min()
    tracking["t_rel"] = tracking["ts"] - t_start

    # Load puzzle logs
    plog = pd.read_csv(puzzle_path)
    interactions = plog[plog["ElementType"] == "Interaction"].copy()
    if len(interactions) == 0:
        return pd.DataFrame()

    interactions["ts"] = interactions["TimeStampUTC"].apply(_parse_ts)
    interactions = interactions.dropna(subset=["ts"])
    interactions["t_rel"] = interactions["ts"] - t_start

    # Classify each interaction
    action_types = []
    for _, action in interactions.iterrows():
        action_time = action["t_rel"]
        puzzle_name = _extract_puzzle_name(action.get("ParentChain"))
        action_kw = _extract_object_keywords(action.get("ParentChain"))

        # Get gaze targets in lookback window
        pre_gaze = tracking[
            (tracking["t_rel"] >= action_time - lookback_sec) &
            (tracking["t_rel"] < action_time)
        ]["GazeTarget_ObjectName"]

        classification = _is_gaze_relevant(
            pre_gaze.tolist(), puzzle_name, action_kw
        )

        action_types.append({
            "t_rel": action_time,
            "outcome": action.get("Outcome", "Unknown"),
            "gaze_action_type": classification,
            "puzzle": puzzle_name,
        })

    actions_df = pd.DataFrame(action_types)

    # Aggregate per 5-second window
    window_size = 5.0
    max_time = tracking["t_rel"].max()
    n_windows = int(max_time / window_size) + 1

    rows = []
    for w in range(n_windows):
        w_start = w * window_size
        w_end = w_start + window_size

        w_actions = actions_df[
            (actions_df["t_rel"] >= w_start) & (actions_df["t_rel"] < w_end)
        ]

        n_total = len(w_actions)
        if n_total == 0:
            rows.append({
                "participant_id": user_id,
                "window_start": round(w_start, 2),
                "n_actions": 0,
                "informed_actions": 0,
                "blind_actions": 0,
                "misguided_actions": 0,
                "informed_ratio": 0.0,
                "blind_ratio": 0.0,
                "misguided_ratio": 0.0,
                "informed_correct": 0,
                "blind_correct": 0,
                "blind_wrong": 0,
            })
            continue

        n_informed = (w_actions["gaze_action_type"] == "informed").sum()
        n_blind = (w_actions["gaze_action_type"] == "blind").sum()
        n_misguided = (w_actions["gaze_action_type"] == "misguided").sum()

        # Cross with outcome
        informed_correct = ((w_actions["gaze_action_type"] == "informed") &
                           (w_actions["outcome"] == "RightMove")).sum()
        blind_correct = ((w_actions["gaze_action_type"] == "blind") &
                        (w_actions["outcome"] == "RightMove")).sum()
        blind_wrong = ((w_actions["gaze_action_type"] == "blind") &
                      (w_actions["outcome"] == "WrongMove")).sum()

        rows.append({
            "participant_id": user_id,
            "window_start": round(w_start, 2),
            "n_actions": n_total,
            "informed_actions": int(n_informed),
            "blind_actions": int(n_blind),
            "misguided_actions": int(n_misguided),
            "informed_ratio": round(n_informed / n_total, 4),
            "blind_ratio": round(n_blind / n_total, 4),
            "misguided_ratio": round(n_misguided / n_total, 4),
            "informed_correct": int(informed_correct),
            "blind_correct": int(blind_correct),
            "blind_wrong": int(blind_wrong),
        })

    return pd.DataFrame(rows)


def extract_all_users():
    """Extract gaze-action coupling for all eye-tracking users."""
    eye_tracking_users = [1, 2, 3, 5, 6, 9, 10, 12, 14, 22, 23]

    all_dfs = []
    for uid in eye_tracking_users:
        print(f"  Processing User-{uid}...")
        cdf = extract_coupling_for_user(uid)
        if len(cdf) > 0:
            active = cdf[cdf["n_actions"] > 0]
            n_informed = active["informed_actions"].sum()
            n_blind = active["blind_actions"].sum()
            n_misguided = active["misguided_actions"].sum()
            n_total = active["n_actions"].sum()
            print(f"    → {len(cdf)} windows, {n_total} actions: "
                  f"{n_informed} informed, {n_blind} blind, {n_misguided} misguided")
            all_dfs.append(cdf)

    if not all_dfs:
        print("No data extracted!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Summary
    active = combined[combined["n_actions"] > 0]
    total = active["n_actions"].sum()
    print(f"\n=== Summary ===")
    print(f"Total actions: {total}")
    print(f"  Informed: {active['informed_actions'].sum()} ({active['informed_actions'].sum()/total:.1%})")
    print(f"  Blind:    {active['blind_actions'].sum()} ({active['blind_actions'].sum()/total:.1%})")
    print(f"  Misguided:{active['misguided_actions'].sum()} ({active['misguided_actions'].sum()/total:.1%})")

    # Cross with outcome
    print(f"\nInformed + RightMove: {active['informed_correct'].sum()}")
    print(f"Blind + RightMove:    {active['blind_correct'].sum()}")
    print(f"Blind + WrongMove:    {active['blind_wrong'].sum()}")

    os.makedirs("data", exist_ok=True)
    out_path = "data/gaze_action_coupling.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    return combined


if __name__ == "__main__":
    print("=== Extracting Gaze-Action Coupling ===\n")
    extract_all_users()
