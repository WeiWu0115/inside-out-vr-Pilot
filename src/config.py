"""
Configuration for the multi-agent interpretation pipeline.
Thresholds are intentionally transparent and easy to modify.
"""

# --- Column names expected in input CSV ---
REQUIRED_COLUMNS = [
    "participant_id", "puzzle_id", "window_start",
]

OPTIONAL_COLUMNS = [
    "gaze_entropy", "clue_ratio", "switch_rate",
    "action_count", "idle_time", "time_since_action",
    "error_count", "puzzle_active",
]

# --- AttentionAgent thresholds ---
# Real data: gaze_entropy 0–4.2 (median 1.15), switch_rate 0–32.5 (median 2.8)
ATTENTION = {
    "entropy_low": 0.6,          # ~P25: captures focused gaze
    "entropy_high": 1.8,         # ~P75: captures scattered gaze
    "clue_ratio_high": 0.4,      # ~P75: meaningful clue engagement
    "clue_ratio_very_high": 0.8,
    "switch_rate_high": 5.8,     # ~P75: frequent target switching
    "action_count_low": 1,       # used for "locked" check
}

# --- ActionAgent thresholds ---
# Real data: action_count median=1, idle_time 0–5.0 (5s windows), time_since_action median=56
ACTION = {
    "action_count_high": 3,      # ~P75: active within a 5s window
    "action_count_low": 0,       # truly zero actions
    "idle_time_low": 3.5,        # ~P25: little idle in the window
    "idle_time_high": 4.8,       # ~P75: mostly idle (of 5s window)
    "time_since_action_moderate": 55.0,  # ~P50: half a minute+ since last action
}

# --- PerformanceAgent thresholds ---
# Real data: error_count is almost always 0 (P90=0), only 0.2% >= 3
PERFORMANCE = {
    "error_count_high": 1,       # any error is meaningful in this dataset
    "action_count_low": 0,       # truly zero actions for "stalled"
}

# --- TemporalAgent thresholds ---
TEMPORAL = {
    "persistence_window": 3,     # consecutive windows for "persistent"
    "looping_keywords": ["failing", "stalled", "ineffective_progress"],
}

# --- File paths ---
INPUT_CSV = "data/windows.csv"
OUTPUT_DIR = "outputs"
AGENT_OUTPUT_FILE = "outputs/agent_outputs.csv"
DISAGREEMENT_FILE = "outputs/disagreement_summary.csv"
SUPPORT_FILE = "outputs/support_summary.csv"
