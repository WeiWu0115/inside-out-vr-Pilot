"""
Extract rich eye-tracking features from raw PlayerTracking.csv files.

Produces per-5-second-window features that go far beyond the original
gaze_entropy / clue_ratio / switch_rate triple.

Feature groups:
  1. Fixation metrics     — count, duration, max duration
  2. Saccade metrics      — count, amplitude
  3. Semantic gaze        — target entropy, clue dwell, puzzle object ratio
  4. Gaze-head coupling   — how tightly gaze follows head movement
  5. Revisit patterns     — how often targets are re-fixated
  6. Spatial gaze         — gaze dispersion, directional variance
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

# ── Gaze target categories ────────────────────────────────────────

CLUE_KEYWORDS = [
    "diary", "hint", "note", "instruction", "whitboard", "chalkboard",
    "chalkbaord", "seedpacket",
]

PUZZLE_KEYWORDS = [
    "puzzle", "snap", "jar", "pot", "seed", "plant", "carrot", "potato",
    "tomato", "water", "sunlight", "protein", "pasta", "sauce", "cooking",
    "cup", "order", "amount", "bean", "beands", "plot", "lid",
    "spoon", "ladle", "bowl",
]

ENV_KEYWORDS = [
    "wall", "ground", "floor", "glass", "exterior", "greenhouse",
    "ceiling", "roof", "door", "table", "shelf", "cabinet", "drawer",
    "cube", "cigar", "wooden",
]


def _categorize_target(name):
    """Categorize gaze target as clue / puzzle / environment / other."""
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


# ── Timestamp parsing ─────────────────────────────────────────────

def _parse_timestamp(ts):
    """Parse UTC timestamp string to epoch seconds."""
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


# ── Fixation detection ────────────────────────────────────────────

def _detect_fixations(targets, timestamps, min_duration=0.1):
    """
    Detect fixations as consecutive frames on the same gaze target.
    Returns list of (target, start_time, end_time, duration).
    """
    if len(targets) == 0:
        return []

    fixations = []
    current_target = targets.iloc[0]
    start_time = timestamps.iloc[0]

    for i in range(1, len(targets)):
        if targets.iloc[i] != current_target:
            duration = timestamps.iloc[i - 1] - start_time
            if duration >= min_duration:
                fixations.append((current_target, start_time, timestamps.iloc[i - 1], duration))
            current_target = targets.iloc[i]
            start_time = timestamps.iloc[i]

    # Last fixation
    duration = timestamps.iloc[-1] - start_time
    if duration >= min_duration:
        fixations.append((current_target, start_time, timestamps.iloc[-1], duration))

    return fixations


# ── Per-window feature extraction ─────────────────────────────────

def _extract_window_features(window_df):
    """Extract all gaze features for a single 5-second window."""
    n_samples = len(window_df)
    if n_samples < 5:  # too few samples
        return _empty_features()

    ts = window_df["ts"].values
    dt = ts[-1] - ts[0]
    if dt < 0.5:  # less than 0.5s of data
        return _empty_features()

    targets = window_df["GazeTarget_ObjectName"].fillna("unknown")
    categories = targets.apply(_categorize_target)

    # ── 1. Fixation metrics ───────────────────────────────────
    fixations = _detect_fixations(targets, window_df["ts"])
    fix_durations = [f[3] for f in fixations]

    fixation_count = len(fixations)
    fixation_duration_mean = np.mean(fix_durations) if fix_durations else 0.0
    fixation_duration_max = np.max(fix_durations) if fix_durations else 0.0
    fixation_duration_std = np.std(fix_durations) if len(fix_durations) > 1 else 0.0

    # ── 2. Saccade metrics ────────────────────────────────────
    # Saccade = transition between different fixation targets
    saccade_count = max(0, fixation_count - 1)

    # Saccade amplitude: angular change in gaze direction
    gaze_dirs = window_df[["Cyclo_Dir_X", "Cyclo_Dir_Y", "Cyclo_Dir_Z"]].dropna()
    if len(gaze_dirs) > 1:
        dirs = gaze_dirs.values
        # Compute angular differences between consecutive frames
        dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
        dots = np.clip(dots, -1, 1)
        angles = np.arccos(dots)  # radians
        saccade_amplitude_mean = np.degrees(np.mean(angles))
        saccade_amplitude_max = np.degrees(np.max(angles))
    else:
        saccade_amplitude_mean = 0.0
        saccade_amplitude_max = 0.0

    # ── 3. Semantic gaze ──────────────────────────────────────
    cat_counts = categories.value_counts()
    total = len(categories)

    clue_dwell = cat_counts.get("clue", 0) / total
    puzzle_dwell = cat_counts.get("puzzle", 0) / total
    env_dwell = cat_counts.get("environment", 0) / total
    puzzle_object_ratio = (cat_counts.get("clue", 0) + cat_counts.get("puzzle", 0)) / total

    # Target entropy (Shannon entropy over unique target names)
    target_counts = targets.value_counts()
    target_probs = target_counts / target_counts.sum()
    gaze_target_entropy = -np.sum(target_probs * np.log2(target_probs + 1e-10))

    # Unique targets viewed
    n_unique_targets = targets.nunique()

    # ── 4. Gaze-head coupling ─────────────────────────────────
    # How much gaze direction correlates with head rotation
    head_rot = window_df[["Head_Rot_X", "Head_Rot_Y", "Head_Rot_Z"]].dropna()
    gaze_dir = window_df[["Cyclo_Dir_X", "Cyclo_Dir_Y", "Cyclo_Dir_Z"]].dropna()

    if len(head_rot) > 10 and len(gaze_dir) > 10:
        # Use Y rotation (horizontal) as primary
        head_y = head_rot["Head_Rot_Y"].diff().dropna().values
        gaze_x = gaze_dir["Cyclo_Dir_X"].diff().dropna().values
        min_len = min(len(head_y), len(gaze_x))
        if min_len > 5:
            corr = np.corrcoef(head_y[:min_len], gaze_x[:min_len])[0, 1]
            gaze_head_coupling = abs(corr) if not np.isnan(corr) else 0.0
        else:
            gaze_head_coupling = 0.0
    else:
        gaze_head_coupling = 0.0

    # ── 5. Revisit patterns ───────────────────────────────────
    # How often does gaze return to a previously fixated target?
    if len(fixations) > 1:
        fix_targets = [f[0] for f in fixations]
        seen = set()
        revisits = 0
        for t in fix_targets:
            if t in seen:
                revisits += 1
            seen.add(t)
        revisit_rate = revisits / len(fix_targets)
    else:
        revisit_rate = 0.0

    # ── 6. Spatial gaze dispersion ────────────────────────────
    gaze_pos = window_df[["Cyclo_Pos_X", "Cyclo_Pos_Y", "Cyclo_Pos_Z"]].dropna()
    if len(gaze_pos) > 5:
        gaze_dispersion = gaze_pos.std().mean()  # average spatial spread
    else:
        gaze_dispersion = 0.0

    # ── 7. Eye tracking quality ───────────────────────────────
    left_conf = window_df["Left_Confidence"].mean()
    right_conf = window_df["Right_Confidence"].mean()
    eye_confidence_mean = (left_conf + right_conf) / 2
    # Confidence drops may indicate blinks or lost tracking
    conf_combined = (window_df["Left_Confidence"] + window_df["Right_Confidence"]) / 2
    blink_proxy = (conf_combined < 0.5).sum() / len(conf_combined)

    return {
        # Fixation
        "fixation_count": fixation_count,
        "fixation_duration_mean": round(fixation_duration_mean, 4),
        "fixation_duration_max": round(fixation_duration_max, 4),
        "fixation_duration_std": round(fixation_duration_std, 4),
        # Saccade
        "saccade_count": saccade_count,
        "saccade_amplitude_mean": round(saccade_amplitude_mean, 4),
        "saccade_amplitude_max": round(saccade_amplitude_max, 4),
        # Semantic gaze
        "gaze_target_entropy": round(gaze_target_entropy, 4),
        "n_unique_targets": n_unique_targets,
        "clue_dwell": round(clue_dwell, 4),
        "puzzle_dwell": round(puzzle_dwell, 4),
        "env_dwell": round(env_dwell, 4),
        "puzzle_object_ratio": round(puzzle_object_ratio, 4),
        # Gaze-head coupling
        "gaze_head_coupling": round(gaze_head_coupling, 4),
        # Revisit
        "revisit_rate": round(revisit_rate, 4),
        # Spatial
        "gaze_dispersion": round(gaze_dispersion, 4),
        # Quality
        "eye_confidence_mean": round(eye_confidence_mean, 4),
        "blink_proxy": round(blink_proxy, 4),
        # Meta
        "n_gaze_samples": n_samples,
    }


def _empty_features():
    """Return empty feature dict when data is insufficient."""
    return {
        "fixation_count": 0,
        "fixation_duration_mean": 0.0,
        "fixation_duration_max": 0.0,
        "fixation_duration_std": 0.0,
        "saccade_count": 0,
        "saccade_amplitude_mean": 0.0,
        "saccade_amplitude_max": 0.0,
        "gaze_target_entropy": 0.0,
        "n_unique_targets": 0,
        "clue_dwell": 0.0,
        "puzzle_dwell": 0.0,
        "env_dwell": 0.0,
        "puzzle_object_ratio": 0.0,
        "gaze_head_coupling": 0.0,
        "revisit_rate": 0.0,
        "gaze_dispersion": 0.0,
        "eye_confidence_mean": 0.0,
        "blink_proxy": 0.0,
        "n_gaze_samples": 0,
    }


# ── Main extraction pipeline ─────────────────────────────────────

def extract_gaze_features(user_id, window_size=5.0):
    """
    Extract gaze features for one user, windowed at 5-second intervals.
    Returns DataFrame with one row per window.
    """
    tracking_path = f"User-{user_id}/User-{user_id}_PlayerTracking.csv"
    if not os.path.exists(tracking_path):
        print(f"  [SKIP] {tracking_path} not found")
        return pd.DataFrame()

    df = pd.read_csv(tracking_path, low_memory=False)
    if len(df) < 10:
        return pd.DataFrame()

    # Parse timestamps
    df["ts"] = df["Timestamp"].apply(_parse_timestamp)
    df = df.dropna(subset=["ts"])
    if len(df) == 0:
        return pd.DataFrame()

    t_start = df["ts"].min()
    df["t_rel"] = df["ts"] - t_start

    n_windows = int((df["t_rel"].max()) / window_size) + 1
    rows = []

    for w in range(n_windows):
        w_start = w * window_size
        w_end = w_start + window_size

        window_df = df[(df["t_rel"] >= w_start) & (df["t_rel"] < w_end)].copy()

        features = _extract_window_features(window_df)
        features["participant_id"] = user_id
        features["window_start"] = round(w_start, 2)
        rows.append(features)

    return pd.DataFrame(rows)


def extract_all_users():
    """Extract gaze features for all users with eye tracking data."""
    eye_tracking_users = [1, 2, 3, 5, 6, 9, 10, 12, 14, 22, 23]

    all_dfs = []
    for uid in eye_tracking_users:
        print(f"  Processing User-{uid}...")
        gdf = extract_gaze_features(uid)
        if len(gdf) > 0:
            all_dfs.append(gdf)
            print(f"    → {len(gdf)} windows, {gdf['n_gaze_samples'].sum():.0f} gaze samples")

    if not all_dfs:
        print("No gaze data extracted!")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal: {len(combined)} windows across {combined['participant_id'].nunique()} users")

    # Save
    os.makedirs("data", exist_ok=True)
    out_path = "data/gaze_features.csv"
    combined.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

    # Summary stats
    print("\n--- Feature Summary ---")
    feature_cols = [c for c in combined.columns if c not in ("participant_id", "window_start")]
    print(combined[feature_cols].describe().round(3).to_string())

    return combined


if __name__ == "__main__":
    print("=== Extracting Gaze Features from PlayerTracking ===\n")
    extract_all_users()
