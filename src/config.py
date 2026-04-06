"""
Configuration for the gaze-focused multi-agent pipeline.

V4: Gaze-dominant architecture — 3 gaze agents + 1 behavioral agent + 1 temporal.
Thresholds derived from data percentiles (5,265 windows, 11 players).
"""

# --- Column names expected in input CSV ---
REQUIRED_COLUMNS = [
    "participant_id", "puzzle_id", "window_start",
]

# Original features (backward compat)
OPTIONAL_COLUMNS = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time", "time_since_action",
    "error_count", "puzzle_active",
]

# New gaze features from gaze_features.py
GAZE_COLUMNS = [
    "fixation_count", "fixation_duration_mean", "fixation_duration_max",
    "fixation_duration_std", "saccade_count", "saccade_amplitude_mean",
    "saccade_amplitude_max", "gaze_target_entropy", "n_unique_targets",
    "clue_dwell", "puzzle_dwell", "env_dwell", "puzzle_object_ratio",
    "gaze_head_coupling", "revisit_rate", "gaze_dispersion",
    "eye_confidence_mean", "blink_proxy",
]

# --- FixationAgent thresholds ---
# fixation_count: median=6, P25=4, P75=9
# fixation_duration_mean: median=0.68, P75=1.20, P90=2.48
# revisit_rate: median=0.50, P75=0.64
FIXATION = {
    "count_low": 4,              # P25: few fixations
    "count_high": 9,             # P75: many fixations
    "duration_short": 0.45,      # P25: short fixations
    "duration_long": 1.2,        # P75: long fixations
    "duration_very_long": 2.5,   # P90: locked gaze
    "duration_max_locked": 4.0,  # single fixation > 4s = locked
    "revisit_low": 0.33,         # P25: rarely revisiting
    "revisit_high": 0.64,        # P75: frequently revisiting
}

# --- GazeSemanticsAgent thresholds ---
# clue_dwell: median=0.04, P75=0.36, P90=0.81
# puzzle_object_ratio: median=0.10, P75=0.41
# gaze_target_entropy: median=1.15, P75=1.79
# env_dwell: median=0.85, P75=0.99
SEMANTICS = {
    "clue_dwell_high": 0.35,          # P75: meaningful clue engagement
    "clue_dwell_very_high": 0.80,     # P90: locked on clues
    "puzzle_ratio_high": 0.40,        # P75: task-focused gaze
    "puzzle_ratio_low": 0.10,         # below P50: not looking at task
    "target_entropy_low": 0.63,       # P25: concentrated gaze
    "target_entropy_high": 1.79,      # P75: scattered gaze
    "env_dwell_high": 0.85,           # P50: mostly looking at walls/floor
    "n_targets_high": 7,              # P75: many targets
    "n_targets_low": 2,               # P25: very few targets
}

# --- GazeMotorAgent thresholds ---
# gaze_head_coupling: median=0.16, P75=0.30, P90=0.42
# saccade_amplitude_mean: median=0.81, P75=1.06
# gaze_dispersion: median=0.20, P75=0.30
MOTOR = {
    "coupling_low": 0.10,             # eyes independent of head
    "coupling_high": 0.30,            # P75: eyes follow head
    "coupling_very_high": 0.42,       # P90: passive gaze
    "saccade_amp_low": 0.62,          # P25: small eye movements
    "saccade_amp_high": 1.06,         # P75: large saccades
    "saccade_amp_very_high": 1.33,    # P90: erratic
    "dispersion_low": 0.12,           # P25: concentrated
    "dispersion_high": 0.30,          # P75: dispersed
}

# --- BehavioralAgent thresholds (unchanged from V3) ---
ACTION = {
    "action_count_high": 3,
    "action_count_low": 0,
    "idle_time_low": 3.5,
    "idle_time_high": 4.8,
    "time_since_action_moderate": 55.0,
}

# --- PerformanceAgent thresholds (kept for backward compat) ---
PERFORMANCE = {
    "error_count_high": 1,
    "action_count_low": 0,
}

# --- TemporalAgent thresholds ---
TEMPORAL = {
    "persistence_window": 3,
    "looping_keywords": ["failing", "inactive", "locked", "fixated_on_clue",
                          "passive_scanning", "erratic"],
}

# --- File paths ---
INPUT_CSV = "data/windows_enhanced.csv"
OUTPUT_DIR = "outputs"
AGENT_OUTPUT_FILE = "outputs/agent_outputs.csv"
DISAGREEMENT_FILE = "outputs/disagreement_summary.csv"
SUPPORT_FILE = "outputs/support_summary.csv"
