"""
Run the expert rule engine on ALL 18 players using only PuzzleLogs data
(no eye tracking required).

Builds 5-second windows from PuzzleLogs and derives the expert's input
variables directly from game events.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime


def parse_timestamp(ts):
    """Parse UTC timestamp to seconds from start."""
    try:
        # Handle 7-digit fractional seconds (truncate to 6)
        if "." in ts:
            base, frac = ts.split(".")
            frac = frac.rstrip("Z")[:6]
            clean = f"{base}.{frac}+00:00"
        else:
            clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        return dt.timestamp()
    except:
        return None


def extract_puzzle_area(parent_chain):
    """Extract which puzzle area from ParentChain."""
    if parent_chain is None:
        return "Transition"
    pc = str(parent_chain)
    for spoke in ["Pasta in Sauce", "Amount of Protein", "Water Amount", "Amount of Sunlight"]:
        if spoke in pc:
            return f"Spoke Puzzle: {spoke}"
    if "Cooking Pot" in pc or "Hub" in pc:
        return "Hub Puzzle: Cooking Pot"
    return "Transition"


def build_expert_windows(user_id):
    """Build 5-second windows from PuzzleLogs for expert engine."""
    folder = f"User-{user_id}"
    puzzle_path = f"{folder}/User-{user_id}_PuzzleLogs.csv"

    if not os.path.exists(puzzle_path):
        return pd.DataFrame()

    plog = pd.read_csv(puzzle_path)
    if len(plog) < 2:
        return pd.DataFrame()

    # Parse timestamps
    plog["ts"] = plog["TimeStampUTC"].apply(parse_timestamp)
    plog = plog.dropna(subset=["ts"])
    if len(plog) == 0:
        return pd.DataFrame()

    t_start = plog["ts"].min()
    t_end = plog["ts"].max()
    plog["t_rel"] = plog["ts"] - t_start

    # Track puzzle completion state
    completed_puzzles = set()
    spoke_puzzles = {
        "Spoke Puzzle: Pasta in Sauce",
        "Spoke Puzzle: Amount of Protein",
        "Spoke Puzzle: Water Amount",
        "Spoke Puzzle: Amount of Sunlight",
    }

    # Build 5-second windows
    window_size = 5.0
    windows = []
    n_windows = int((t_end - t_start) / window_size) + 1

    for w in range(n_windows):
        w_start = w * window_size
        w_end = w_start + window_size

        # Events in this window
        events = plog[(plog["t_rel"] >= w_start) & (plog["t_rel"] < w_end)]

        # Events before this window (for time_since_last_action)
        past_events = plog[plog["t_rel"] < w_end]
        interactions = past_events[past_events["ElementType"] == "Interaction"]

        # --- Derive expert variables ---
        # time_since_last_action
        if len(interactions) > 0:
            last_action_time = interactions["t_rel"].max()
            time_since_last_action = w_end - last_action_time
        else:
            time_since_last_action = w_end

        # puzzle_interaction: any interaction in this window
        window_interactions = events[events["ElementType"] == "Interaction"]
        puzzle_interaction = len(window_interactions) > 0

        # action_count
        action_count = len(window_interactions)

        # error_count
        error_count = len(window_interactions[window_interactions["Outcome"] == "WrongMove"])

        # object_in_hand: approximate as having an interaction
        object_in_hand = puzzle_interaction

        # puzzle area from most recent event
        if len(events) > 0:
            puzzle_id = extract_puzzle_area(events.iloc[-1]["ParentChain"])
        elif len(past_events) > 0:
            puzzle_id = extract_puzzle_area(past_events.iloc[-1]["ParentChain"])
        else:
            puzzle_id = "Transition"

        # current_area_type
        if "Hub" in puzzle_id or "Cooking" in puzzle_id or puzzle_id == "Transition":
            current_area_type = "hub"
        else:
            current_area_type = "puzzle"

        # Track completed puzzles
        completed_in_window = events[
            (events["ElementType"] == "Puzzle") &
            (events["IsCompleted"] == "Completed")
        ]
        for _, ev in completed_in_window.iterrows():
            area = extract_puzzle_area(ev["ParentChain"])
            completed_puzzles.add(area)

        # puzzle_state
        if puzzle_id in completed_puzzles:
            puzzle_state = "solved"
        elif current_area_type == "hub":
            spokes_done = len(completed_puzzles.intersection(spoke_puzzles))
            if spokes_done < 4:
                puzzle_state = "unsolvable"
            else:
                puzzle_state = "solvable"
        else:
            puzzle_state = "unsolved"

        # gaze_on_instruction: not available without eye tracking, set to 0
        # (expert system will treat as "not looking at instructions")
        gaze_on_instruction = 0.0

        windows.append({
            "participant_id": user_id,
            "window_start": round(w_start, 2),
            "t_end": round(w_end, 2),
            "puzzle_id": puzzle_id,
            "time_since_action": round(time_since_last_action, 2),
            "action_count": action_count,
            "error_count": error_count,
            "puzzle_interaction": puzzle_interaction,
            "object_in_hand": object_in_hand,
            "current_area_type": current_area_type,
            "puzzle_state": puzzle_state,
            "gaze_on_instruction": gaze_on_instruction,
        })

    return pd.DataFrame(windows)


def run_expert_on_all():
    """Run expert engine on all 18 players."""
    # Find all user folders
    user_ids = []
    for d in os.listdir("."):
        if d.startswith("User-") and os.path.isdir(d):
            try:
                uid = int(d.split("-")[1])
                user_ids.append(uid)
            except ValueError:
                pass
    user_ids = sorted(user_ids)
    print(f"Found {len(user_ids)} users: {user_ids}")

    # Build windows for all users
    all_windows = []
    for uid in user_ids:
        wdf = build_expert_windows(uid)
        if len(wdf) > 0:
            all_windows.append(wdf)
            print(f"  User-{uid}: {len(wdf)} windows")

    all_df = pd.concat(all_windows, ignore_index=True)
    print(f"\nTotal: {len(all_df)} windows across {all_df['participant_id'].nunique()} players")

    # Run expert engine
    from expert_engine import run_expert_engine_raw
    expert_df = run_expert_engine_raw(all_df)

    # Save
    os.makedirs("outputs", exist_ok=True)
    expert_df.to_csv("outputs/expert_all18_outputs.csv", index=False)
    print(f"\nSaved to outputs/expert_all18_outputs.csv")

    # Summary
    print("\n--- Expert Engine on ALL 18 Players ---")
    print(f"States:\n{expert_df['expert_state'].value_counts().to_string()}")
    print(f"\nActions:\n{expert_df['expert_action'].value_counts().to_string()}")
    print(f"\nRules:\n{expert_df['expert_rule'].value_counts().to_string()}")

    print("\nPrompts per player:")
    for pid in sorted(expert_df['participant_id'].unique()):
        pf = expert_df[expert_df['participant_id'] == pid]
        n_prompts = (pf['expert_action'] == 'PROMPT').sum()
        n_windows = len(pf)
        has_eyetracking = pid in [1, 2, 3, 5, 6, 9, 10, 12, 14, 22, 23]
        marker = "* " if has_eyetracking else "  "
        print(f"  {marker}P{pid:2d}: {n_prompts:3d} prompts / {n_windows:4d} windows {'(has eye tracking)' if has_eyetracking else '(game log only)'}")

    return expert_df


if __name__ == "__main__":
    run_expert_on_all()
