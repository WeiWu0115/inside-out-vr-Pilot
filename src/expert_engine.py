"""
Human Expert Rule-Based Prompting Engine

Implements the rule-based system designed by the education expert
who observed 23 participants in the VR escape room study.

Variable mapping from expert rules to our windowed data:
    time_since_last_action  → time_since_action (seconds)
    gaze_on_instruction     → clue_ratio * 5.0 (proportion × window_size)
    puzzle_interaction      → action_count > 0
    object_in_hand          → action_count > 0 AND puzzle_active == 1
    puzzle_state            → derived from puzzle completion timeline
    current_area_type       → "hub" if puzzle_id contains "Hub", else "puzzle"
    no_trigger_entry_time   → tracked statefully across windows
"""

import pandas as pd
from load_data import safe_get


def _map_area_type(puzzle_id):
    """Map puzzle_id to expert's area type."""
    if puzzle_id is None:
        return "hub"
    pid = str(puzzle_id).lower()
    if "hub" in pid or "cooking" in pid:
        return "hub"
    if "transition" in pid:
        return "hub"  # treat transition as hub-like
    return "puzzle"


def run_expert_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the expert rule-based engine on the windowed data.
    Processes each participant independently and maintains state across windows.
    """
    results = []

    for pid in df["participant_id"].unique():
        pdf = df[df["participant_id"] == pid].sort_values("window_start").copy()

        state = "EXPLORE"
        prompt_count = 0
        no_trigger_entry_time = 0.0
        last_prompt_time = -999.0

        for idx, row in pdf.iterrows():
            # --- Map variables ---
            time_since_last_action = safe_get(row, "time_since_action", 0.0)
            clue_ratio = safe_get(row, "clue_ratio", 0.0)
            gaze_on_instruction = clue_ratio * 5.0  # 5-second window
            action_count = safe_get(row, "action_count", 0)
            puzzle_active = safe_get(row, "puzzle_active", 0)
            error_count = safe_get(row, "error_count", 0)
            puzzle_id = safe_get(row, "puzzle_id", "")
            window_start = safe_get(row, "window_start", 0.0)

            puzzle_interaction = action_count > 0
            object_in_hand = action_count > 0 and puzzle_active == 1
            current_area_type = _map_area_type(puzzle_id)

            # puzzle_state approximation:
            # "solved" if this is the last window for this puzzle and no errors
            # "unsolvable" if in hub and not enough spokes done (approximate)
            # otherwise "unsolved"
            puzzle_state = "unsolved"
            if str(puzzle_id).startswith("Spoke") and puzzle_active == 1 and error_count == 0 and action_count > 2:
                puzzle_state = "solvable"

            # Track no_trigger_entry_time (time without entering a puzzle area)
            if current_area_type == "puzzle" and puzzle_interaction:
                no_trigger_entry_time = 0.0
            else:
                no_trigger_entry_time += 5.0  # each window is 5 seconds

            # --- Prompt timing rule ---
            can_prompt = (window_start - last_prompt_time) >= 30.0

            # --- Evaluate rules in priority order ---
            action = "NONE"
            prompt_type = None
            content = ""
            rule_triggered = "RULE_10_DEFAULT"

            # RULE 1: VICTORY
            if puzzle_state == "solved":
                state = "VICTORY"
                rule_triggered = "RULE_1_VICTORY"

            # RULE 2: HUB UNSOLVABLE
            elif current_area_type == "hub" and puzzle_state == "unsolvable":
                state = "EXPLORE"
                if can_prompt:
                    action = "PROMPT"
                    prompt_type = "V"
                    content = "Try to take a look around"
                    last_prompt_time = window_start
                rule_triggered = "RULE_2_HUB_UNSOLVABLE"

            # RULE 3: NO ENTRY PROGRESSION
            elif no_trigger_entry_time >= 30:
                state = "EXPLORE"
                if can_prompt:
                    action = "PROMPT"
                    prompt_type = "V"
                    content = "Try to explore other puzzles"
                    last_prompt_time = window_start
                rule_triggered = "RULE_3_NO_ENTRY"

            # RULE 6: STUCK EXIT (check before stuck detection)
            elif state == "STUCK" and (object_in_hand or puzzle_interaction or gaze_on_instruction >= 3):
                state = "SOLVING"
                prompt_count = 0
                rule_triggered = "RULE_6_STUCK_EXIT"

            # RULE 4 & 5: STUCK DETECTION + ESCALATION
            elif (not object_in_hand and gaze_on_instruction < 3 and time_since_last_action >= 30):
                state = "STUCK"
                rule_triggered = "RULE_4_STUCK"

                # RULE 5: STUCK PROMPT ESCALATION
                if can_prompt:
                    if prompt_count == 0:
                        action = "PROMPT"
                        prompt_type = "R"
                        content = "Maybe check the instructions again?"
                    elif prompt_count == 1:
                        action = "PROMPT"
                        prompt_type = "V"
                        content = "Something here might be useful"
                    else:
                        action = "PROMPT"
                        prompt_type = "E"
                        content = "Use the available clues to proceed"
                    prompt_count += 1
                    last_prompt_time = window_start
                    rule_triggered = "RULE_5_STUCK_ESCALATION"

            # RULE 8: EXPLORE EXIT
            elif state == "EXPLORE" and (gaze_on_instruction >= 3 or puzzle_interaction):
                state = "SOLVING"
                rule_triggered = "RULE_8_EXPLORE_EXIT"

            # RULE 7: EXPLORE DETECTION
            elif (time_since_last_action >= 30 and gaze_on_instruction < 3 and not puzzle_interaction):
                state = "EXPLORE"
                rule_triggered = "RULE_7_EXPLORE"

            # RULE 9: SOLVING STATE
            elif gaze_on_instruction >= 5 and puzzle_interaction:
                state = "SOLVING"
                rule_triggered = "RULE_9_SOLVING"

            # RULE 10: DEFAULT
            else:
                rule_triggered = "RULE_10_DEFAULT"

            results.append({
                "participant_id": pid,
                "window_start": window_start,
                "expert_state": state,
                "expert_action": action,
                "expert_prompt_type": prompt_type,
                "expert_content": content,
                "expert_rule": rule_triggered,
                "expert_prompt_count": prompt_count,
                # Mapped variables for inspection
                "mapped_gaze_on_instruction": round(gaze_on_instruction, 2),
                "mapped_puzzle_interaction": puzzle_interaction,
                "mapped_object_in_hand": object_in_hand,
                "mapped_area_type": current_area_type,
                "mapped_no_trigger_time": no_trigger_entry_time,
            })

    return pd.DataFrame(results)


def run_expert_engine_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run expert engine on data that already has the expert's variables
    (from expert_from_logs.py). No variable mapping needed.
    """
    results = []

    for pid in df["participant_id"].unique():
        pdf = df[df["participant_id"] == pid].sort_values("window_start").copy()

        state = "EXPLORE"
        prompt_count = 0
        no_trigger_entry_time = 0.0
        last_prompt_time = -999.0

        for idx, row in pdf.iterrows():
            time_since_last_action = row.get("time_since_action", 0.0)
            gaze_on_instruction = row.get("gaze_on_instruction", 0.0)
            puzzle_interaction = bool(row.get("puzzle_interaction", False))
            object_in_hand = bool(row.get("object_in_hand", False))
            current_area_type = row.get("current_area_type", "hub")
            puzzle_state = row.get("puzzle_state", "unsolved")
            puzzle_id = row.get("puzzle_id", "")
            window_start = row.get("window_start", 0.0)

            # Track no_trigger_entry_time
            if current_area_type == "puzzle" and puzzle_interaction:
                no_trigger_entry_time = 0.0
            else:
                no_trigger_entry_time += 5.0

            can_prompt = (window_start - last_prompt_time) >= 30.0

            action = "NONE"
            prompt_type = None
            content = ""
            rule_triggered = "RULE_10_DEFAULT"

            if puzzle_state == "solved":
                state = "VICTORY"
                rule_triggered = "RULE_1_VICTORY"
            elif current_area_type == "hub" and puzzle_state == "unsolvable":
                state = "EXPLORE"
                if can_prompt:
                    action = "PROMPT"
                    prompt_type = "V"
                    content = "Try to take a look around"
                    last_prompt_time = window_start
                rule_triggered = "RULE_2_HUB_UNSOLVABLE"
            elif no_trigger_entry_time >= 30:
                state = "EXPLORE"
                if can_prompt:
                    action = "PROMPT"
                    prompt_type = "V"
                    content = "Try to explore other puzzles"
                    last_prompt_time = window_start
                rule_triggered = "RULE_3_NO_ENTRY"
            elif state == "STUCK" and (object_in_hand or puzzle_interaction or gaze_on_instruction >= 3):
                state = "SOLVING"
                prompt_count = 0
                rule_triggered = "RULE_6_STUCK_EXIT"
            elif (not object_in_hand and gaze_on_instruction < 3 and time_since_last_action >= 30):
                state = "STUCK"
                rule_triggered = "RULE_4_STUCK"
                if can_prompt:
                    if prompt_count == 0:
                        action, prompt_type = "PROMPT", "R"
                        content = "Maybe check the instructions again?"
                    elif prompt_count == 1:
                        action, prompt_type = "PROMPT", "V"
                        content = "Something here might be useful"
                    else:
                        action, prompt_type = "PROMPT", "E"
                        content = "Use the available clues to proceed"
                    prompt_count += 1
                    last_prompt_time = window_start
                    rule_triggered = "RULE_5_STUCK_ESCALATION"
            elif state == "EXPLORE" and (gaze_on_instruction >= 3 or puzzle_interaction):
                state = "SOLVING"
                rule_triggered = "RULE_8_EXPLORE_EXIT"
            elif (time_since_last_action >= 30 and gaze_on_instruction < 3 and not puzzle_interaction):
                state = "EXPLORE"
                rule_triggered = "RULE_7_EXPLORE"
            elif gaze_on_instruction >= 5 and puzzle_interaction:
                state = "SOLVING"
                rule_triggered = "RULE_9_SOLVING"

            results.append({
                "participant_id": pid,
                "window_start": window_start,
                "puzzle_id": puzzle_id,
                "expert_state": state,
                "expert_action": action,
                "expert_prompt_type": prompt_type,
                "expert_content": content,
                "expert_rule": rule_triggered,
                "expert_prompt_count": prompt_count,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from load_data import load_csv

    df = load_csv("data/windows.csv")
    expert_df = run_expert_engine(df)

    os.makedirs("outputs", exist_ok=True)
    expert_df.to_csv("outputs/expert_engine_outputs.csv", index=False)
    print(f"Saved {len(expert_df)} rows to outputs/expert_engine_outputs.csv")

    print("\n--- Expert Engine Summary ---")
    print(f"\nStates:\n{expert_df['expert_state'].value_counts().to_string()}")
    print(f"\nActions:\n{expert_df['expert_action'].value_counts().to_string()}")
    print(f"\nRules triggered:\n{expert_df['expert_rule'].value_counts().to_string()}")

    # Prompts per player
    print("\nPrompts per player:")
    for pid in sorted(expert_df['participant_id'].unique()):
        pf = expert_df[expert_df['participant_id'] == pid]
        n_prompts = (pf['expert_action'] == 'PROMPT').sum()
        print(f"  P{pid}: {n_prompts} prompts")
