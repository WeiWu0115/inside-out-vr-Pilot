"""
Human Expert Rule-Based Prompting Engine

Faithful reimplementation of the Unity PromptStateMachine + PromptEscalator.
See: EscapeRoom/Assets/Scripts/AI/PromptAgent/

State machine per puzzle:
  InBetween → TriggerCheck → Explore → Solving ⇄ PossiblyStuck → (prompt) → Solving

Escalation per puzzle (never reset):
  3 Reflective → 2 Vague → unlimited Explicit

Special prompts (bypass escalation):
  SpecialA  — wandering (InBetween timeout: 300s first, 30s repeat)
  SpecialB  — hub blocked (TriggerCheck: 30s repeat while hub unsolvable)
  SpecialC  — grab something (Explore timeout: 90s)

Variable mapping from windowed data:
  isLookingAtInstructions  → gaze_on_instruction >= gazeThreshold (3s)
  areBothHandsEmpty        → not object_in_hand
  puzzle_interaction       → triggers Explore → Solving transition
"""

import pandas as pd

try:
    from load_data import safe_get
except ImportError:
    def safe_get(row, key, default=None):
        val = row.get(key, default)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return default
        return val


# ── Unity timer defaults ──────────────────────────────────────────
IN_BETWEEN_TIMEOUT = 300.0   # first wander prompt
IN_BETWEEN_REPEAT = 30.0     # subsequent wander prompts
HUB_UNSOLVABLE_REPEAT = 30.0
EXPLORE_TIMEOUT = 90.0
GAZE_THRESHOLD = 3.0         # seconds of sustained gaze to count as "looking"
INTERACTION_COOLDOWN = 3.0    # seconds of full disengagement before PossiblyStuck
STUCK_TIMEOUT = 30.0          # seconds in PossiblyStuck before escalated prompt
WINDOW_SIZE = 5.0             # each window is 5 seconds

# ── Escalation config ─────────────────────────────────────────────
MAX_REFLECTIVE = 3
MAX_VAGUE = 2


def _map_area_type(puzzle_id):
    """Map puzzle_id to area type."""
    if puzzle_id is None:
        return "transition"
    pid = str(puzzle_id).lower()
    if "hub" in pid or "cooking" in pid:
        return "hub"
    if "transition" in pid:
        return "transition"
    return "puzzle"


class PuzzleEscalator:
    """Per-puzzle R/V/E escalation counter. Never resets."""
    def __init__(self):
        self.r_count = 0
        self.v_count = 0
        self.e_count = 0

    def fire_next(self):
        if self.r_count < MAX_REFLECTIVE:
            self.r_count += 1
            return "R", "Maybe check the instructions again?"
        elif self.v_count < MAX_VAGUE:
            self.v_count += 1
            return "V", "Something here might be useful"
        else:
            self.e_count += 1
            return "E", "Use the available clues to proceed"

    @property
    def total(self):
        return self.r_count + self.v_count + self.e_count


class PuzzleStateMachine:
    """
    Per-puzzle state machine mirroring Unity's PromptStateMachine.
    Operates on 5-second windows instead of per-frame ticks.
    """
    def __init__(self, is_hub, is_hub_solvable_fn):
        self.phase = "InBetween"
        self.escalator = PuzzleEscalator()
        self.is_hub = is_hub
        self.is_hub_solvable = is_hub_solvable_fn

        # Timers (in seconds, accumulated across windows)
        self.phase_timer = 0.0
        self.gaze_accumulator = 0.0
        self.interaction_cooldown = INTERACTION_COOLDOWN
        self.hub_unsolvable_timer = 0.0
        self.first_special_a_fired = False

    def tick(self, dt, is_looking, hands_empty, had_interaction):
        """
        Advance state machine by dt seconds.
        Returns (prompt_type, prompt_content, phase, rule) or None.
        """
        prompt = None

        if self.phase == "InBetween":
            prompt = self._tick_in_between(dt)
        elif self.phase == "TriggerCheck":
            prompt = self._tick_trigger_check(dt)
        elif self.phase == "Explore":
            prompt = self._tick_explore(dt, is_looking, hands_empty, had_interaction)
        elif self.phase == "Solving":
            prompt = self._tick_solving(dt, is_looking, hands_empty)
        elif self.phase == "PossiblyStuck":
            prompt = self._tick_possibly_stuck(dt, is_looking, hands_empty)

        return prompt

    def on_enter_zone(self):
        if self.phase == "InBetween":
            self._transition_to("TriggerCheck")

    def on_exit_zone(self):
        if self.phase != "Victory":
            self._transition_to("InBetween")

    def on_puzzle_interaction(self):
        if self.phase == "Explore":
            self._transition_to("Solving")

    def on_puzzle_completed(self):
        self._transition_to("Victory")

    # ── Phase tickers ─────────────────────────────────────────

    def _tick_in_between(self, dt):
        self.phase_timer += dt
        threshold = IN_BETWEEN_REPEAT if self.first_special_a_fired else IN_BETWEEN_TIMEOUT
        if self.phase_timer >= threshold:
            self.first_special_a_fired = True
            self.phase_timer = 0.0
            return ("SpecialA", "Take a look around the room", self.phase, "SPECIAL_A_WANDER")
        return None

    def _tick_trigger_check(self, dt):
        if self.is_hub and not self.is_hub_solvable():
            self.hub_unsolvable_timer += dt
            if self.hub_unsolvable_timer >= HUB_UNSOLVABLE_REPEAT:
                self.hub_unsolvable_timer = 0.0
                return ("SpecialB", "Try to take a look around", self.phase, "SPECIAL_B_HUB_BLOCKED")
            return None
        else:
            self._transition_to("Explore")
            return None

    def _tick_explore(self, dt, is_looking, hands_empty, had_interaction):
        # Player grabbed something → Solving
        if not hands_empty:
            self._transition_to("Solving")
            return None

        # Puzzle interaction event → Solving
        if had_interaction:
            self._transition_to("Solving")
            return None

        # Gaze at instructions → Solving
        if is_looking:
            self.gaze_accumulator += dt
            if self.gaze_accumulator >= GAZE_THRESHOLD:
                self._transition_to("Solving")
                return None
        else:
            self._decay_gaze(dt)

        # Explore timeout → SpecialC + Solving
        self.phase_timer += dt
        if self.phase_timer >= EXPLORE_TIMEOUT:
            self._transition_to("Solving")
            return ("SpecialC", "Try grabbing something", self.phase, "SPECIAL_C_GRAB")
        return None

    def _tick_solving(self, dt, is_looking, hands_empty):
        engaged = (not hands_empty) or is_looking
        if engaged:
            # Refill cooldown
            self.interaction_cooldown = min(
                self.interaction_cooldown + dt,
                INTERACTION_COOLDOWN
            )
            return None

        # Fully disengaged — drain cooldown
        self.interaction_cooldown -= dt
        if self.interaction_cooldown <= 0:
            self.interaction_cooldown = 0
            self._transition_to("PossiblyStuck")
        return None

    def _tick_possibly_stuck(self, dt, is_looking, hands_empty):
        # Re-grab → Solving
        if not hands_empty:
            self._transition_to("Solving")
            return None

        # Gaze → Solving
        if is_looking:
            self.gaze_accumulator += dt
            if self.gaze_accumulator >= GAZE_THRESHOLD:
                self._transition_to("Solving")
                return None
        else:
            self._decay_gaze(dt)

        # Stuck timeout → escalated prompt → Solving
        self.phase_timer += dt
        if self.phase_timer >= STUCK_TIMEOUT:
            prompt_type, content = self.escalator.fire_next()
            self._transition_to("Solving")
            return (prompt_type, content, "PossiblyStuck", "ESCALATION")
        return None

    # ── Helpers ────────────────────────────────────────────────

    def _decay_gaze(self, dt):
        if self.gaze_accumulator > 0:
            self.gaze_accumulator = max(0, self.gaze_accumulator - dt)

    def _transition_to(self, new_phase):
        self.phase = new_phase
        self.phase_timer = 0.0
        self.gaze_accumulator = 0.0
        self.interaction_cooldown = INTERACTION_COOLDOWN
        self.hub_unsolvable_timer = 0.0


# ── Main engine functions ─────────────────────────────────────────

def _run_engine(df, get_vars_fn):
    """
    Core engine loop. Runs per-puzzle state machines for each participant.
    get_vars_fn(row) → dict with keys:
        puzzle_id, window_start, is_looking, hands_empty, had_interaction,
        area_type, puzzle_completed
    """
    results = []

    for pid in df["participant_id"].unique():
        pdf = df[df["participant_id"] == pid].sort_values("window_start")

        # Per-puzzle state machines and escalators
        puzzle_sms = {}
        completed_spokes = set()
        spoke_puzzles = {
            "Spoke Puzzle: Pasta in Sauce",
            "Spoke Puzzle: Amount of Protein",
            "Spoke Puzzle: Water Amount",
            "Spoke Puzzle: Amount of Sunlight",
        }

        def is_hub_solvable():
            return len(completed_spokes.intersection(spoke_puzzles)) >= 4

        active_puzzle = None

        for idx, row in pdf.iterrows():
            v = get_vars_fn(row)
            puzzle_id = v["puzzle_id"]
            window_start = v["window_start"]
            area_type = v["area_type"]
            is_looking = v["is_looking"]
            hands_empty = v["hands_empty"]
            had_interaction = v["had_interaction"]
            puzzle_completed = v["puzzle_completed"]

            # Create state machine for new puzzles
            if puzzle_id not in puzzle_sms:
                is_hub = (area_type == "hub")
                puzzle_sms[puzzle_id] = PuzzleStateMachine(is_hub, is_hub_solvable)

            sm = puzzle_sms[puzzle_id]

            # Handle zone transitions
            if puzzle_id != active_puzzle:
                # Exit old zone
                if active_puzzle is not None and active_puzzle in puzzle_sms:
                    puzzle_sms[active_puzzle].on_exit_zone()
                # Enter new zone
                sm.on_enter_zone()
                active_puzzle = puzzle_id

            # Handle puzzle interaction event
            if had_interaction:
                sm.on_puzzle_interaction()

            # Handle puzzle completion
            if puzzle_completed:
                sm.on_puzzle_completed()
                completed_spokes.add(puzzle_id)

            # Tick the active state machine
            prompt_result = sm.tick(WINDOW_SIZE, is_looking, hands_empty, had_interaction)

            # Build output row
            action = "NONE"
            prompt_type = None
            content = ""
            rule = f"PHASE_{sm.phase.upper()}"

            if prompt_result is not None:
                prompt_type, content, phase_at_prompt, rule_name = prompt_result
                action = "PROMPT"
                rule = rule_name

            results.append({
                "participant_id": pid,
                "window_start": window_start,
                "puzzle_id": puzzle_id,
                "expert_state": sm.phase,
                "expert_action": action,
                "expert_prompt_type": prompt_type,
                "expert_content": content,
                "expert_rule": rule,
                "expert_prompt_count": sm.escalator.total,
                "expert_r_count": sm.escalator.r_count,
                "expert_v_count": sm.escalator.v_count,
                "expert_e_count": sm.escalator.e_count,
            })

    return pd.DataFrame(results)


def run_expert_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run expert engine on windowed data (windows.csv, 11 players with eye tracking).
    Maps variables from our feature columns to the Unity SM inputs.
    """
    def get_vars(row):
        clue_ratio = safe_get(row, "clue_ratio", 0.0)
        gaze_on_instruction = clue_ratio * 5.0  # proportion × window_size
        action_count = safe_get(row, "action_count", 0)
        puzzle_active = safe_get(row, "puzzle_active", 0)
        puzzle_id = safe_get(row, "puzzle_id", "")
        error_count = safe_get(row, "error_count", 0)

        puzzle_interaction = action_count > 0
        object_in_hand = action_count > 0 and puzzle_active == 1
        area_type = _map_area_type(puzzle_id)

        # Approximate puzzle_completed
        puzzle_completed = (
            str(puzzle_id).startswith("Spoke") and
            puzzle_active == 1 and error_count == 0 and action_count > 2
        )

        return {
            "puzzle_id": puzzle_id,
            "window_start": safe_get(row, "window_start", 0.0),
            "is_looking": gaze_on_instruction >= GAZE_THRESHOLD,
            "hands_empty": not object_in_hand,
            "had_interaction": puzzle_interaction,
            "area_type": area_type,
            "puzzle_completed": puzzle_completed,
        }

    return _run_engine(df, get_vars)


def run_expert_engine_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run expert engine on data from expert_from_logs.py (all 18 players).
    Variables are already derived — no mapping needed.
    """
    def get_vars(row):
        gaze_on_instruction = row.get("gaze_on_instruction", 0.0)
        puzzle_interaction = bool(row.get("puzzle_interaction", False))
        object_in_hand = bool(row.get("object_in_hand", False))
        area_type = row.get("current_area_type", "hub")
        puzzle_id = row.get("puzzle_id", "")
        puzzle_state = row.get("puzzle_state", "unsolved")

        return {
            "puzzle_id": puzzle_id,
            "window_start": row.get("window_start", 0.0),
            "is_looking": gaze_on_instruction >= GAZE_THRESHOLD,
            "hands_empty": not object_in_hand,
            "had_interaction": puzzle_interaction,
            "area_type": area_type,
            "puzzle_completed": puzzle_state == "solved",
        }

    return _run_engine(df, get_vars)


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
    print(f"\nPrompt types:")
    prompts = expert_df[expert_df["expert_action"] == "PROMPT"]
    if len(prompts) > 0:
        print(prompts["expert_prompt_type"].value_counts().to_string())

    print("\nPrompts per player:")
    for pid in sorted(expert_df['participant_id'].unique()):
        pf = expert_df[expert_df['participant_id'] == pid]
        n_prompts = (pf['expert_action'] == 'PROMPT').sum()
        print(f"  P{pid}: {n_prompts} prompts")
