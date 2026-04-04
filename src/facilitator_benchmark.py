"""
Facilitator Benchmark: Three-way comparison of IO vs Expert vs Real Facilitator Prompts.

Aligns facilitator prompt timestamps (absolute) to IO's 5-second windows (relative),
producing a ground-truth benchmark for when a human expert actually intervened.

Output:
  - outputs/facilitator_windows.csv          — per-window facilitator labels for all 18 users
  - outputs/three_way_comparison.csv         — merged IO + expert + facilitator (11 overlap users)
  - outputs/benchmark_report.md              — summary statistics and analysis
"""

import os
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime
from collections import defaultdict

BASE = os.path.join(os.path.expanduser("~"), "Desktop", "VR perception", "VR Escape", "User-Log")
JUPYTER = os.path.join(os.path.expanduser("~"), "Desktop", "VR perception", "Jupyter Analysis")
OUTPUTS = os.path.join(BASE, "outputs")


def load_game_starts():
    """Load game start times from TimeLine xlsx files."""
    tl_dir = os.path.join(JUPYTER, "Time_Line")
    starts = {}
    for f in os.listdir(tl_dir):
        if not f.endswith(".xlsx"):
            continue
        wb = openpyxl.load_workbook(os.path.join(tl_dir, f))
        ws = wb.active
        rows = list(ws.iter_rows(max_row=2, values_only=True))
        uid_str = rows[0][1]  # e.g. "User-1"
        uid = int(uid_str.split("-")[1])
        game_start_str = rows[1][1]  # e.g. "start: 2025-06-23T21:30:54.460100400+00:00"
        ts_str = game_start_str.replace("start: ", "")
        # Truncate nanoseconds to microseconds for parsing
        ts_str = _truncate_ns(ts_str)
        starts[uid] = pd.Timestamp(ts_str)
    return starts


def _truncate_ns(ts_str):
    """Truncate nanosecond timestamps to microsecond precision for pd.Timestamp."""
    # Format: 2025-06-23T21:30:54.460100400+00:00
    # Need to keep only 6 decimal places before timezone
    if "." in ts_str:
        dot_idx = ts_str.index(".")
        # Find the timezone part (+00:00 or Z)
        tz_idx = None
        for i in range(dot_idx, len(ts_str)):
            if ts_str[i] in ("+", "-", "Z") and i > dot_idx + 1:
                tz_idx = i
                break
        if tz_idx:
            frac = ts_str[dot_idx + 1 : tz_idx]
            frac = frac[:6].ljust(6, "0")
            ts_str = ts_str[: dot_idx + 1] + frac + ts_str[tz_idx:]
    return ts_str


def load_facilitator_prompts(game_starts):
    """
    Load all prompt_blocks.csv files and convert to relative seconds from game start.

    Returns DataFrame with: participant_id, rel_start_sec, rel_end_sec, prompt_type, puzzle
    """
    prompts_dir = os.path.join(JUPYTER, "Prompts")
    all_prompts = []

    for f in sorted(os.listdir(prompts_dir)):
        if not f.endswith("_prompt_blocks.csv"):
            continue
        uid = int(f.split("-")[1].split("_")[0])
        if uid not in game_starts:
            print(f"  Warning: No game start for User-{uid}, skipping")
            continue

        game_start = game_starts[uid]
        df = pd.read_csv(os.path.join(prompts_dir, f))
        if df.empty:
            continue

        for _, row in df.iterrows():
            block_start = pd.Timestamp(_truncate_ns(row["block_start_time"]))
            block_end = pd.Timestamp(_truncate_ns(row["block_end_time"]))
            rel_start = (block_start - game_start).total_seconds()
            rel_end = (block_end - game_start).total_seconds()
            all_prompts.append({
                "participant_id": uid,
                "prompt_type": row["prompt_type"],
                "rel_start_sec": rel_start,
                "rel_end_sec": rel_end,
                "puzzle": row["matched_puzzle"],
                "block_id": row["block_id"],
            })

    return pd.DataFrame(all_prompts)


def assign_facilitator_to_windows(prompts_df, windows_df):
    """
    For each 5-second window, determine facilitator action:
      - 'explicit' if an explicit prompt occurred in this window
      - 'reflective' if a reflective prompt occurred in this window
      - 'none' if no prompt

    If both types overlap the same window, explicit takes priority.

    Maps to a three-category system:
      - explicit  → intervene
      - reflective → probe
      - none → watch
    """
    results = []

    for pid in windows_df["participant_id"].unique():
        p_windows = windows_df[windows_df["participant_id"] == pid].copy()
        p_prompts = prompts_df[prompts_df["participant_id"] == pid]

        for _, w in p_windows.iterrows():
            w_start = w["window_start"]
            w_end = w_start + 5.0

            # Find prompts that overlap this window
            overlapping = p_prompts[
                (p_prompts["rel_start_sec"] < w_end) & (p_prompts["rel_end_sec"] > w_start)
            ]

            if overlapping.empty:
                fac_type = "none"
            elif "explicit" in overlapping["prompt_type"].values:
                fac_type = "explicit"
            else:
                fac_type = "reflective"

            fac_cat = {"none": "watch", "reflective": "probe", "explicit": "intervene"}[fac_type]
            results.append({
                "participant_id": pid,
                "window_start": w_start,
                "facilitator_prompt_type": fac_type,
                "facilitator_cat": fac_cat,
            })

    return pd.DataFrame(results)


def build_expert_windows():
    """Load expert outputs and build window-level data for all 18 participants."""
    expert_path = os.path.join(OUTPUTS, "expert_all18_outputs.csv")
    df = pd.read_csv(expert_path)
    # Map expert_action to category
    def expert_to_cat(action):
        if action == "NONE":
            return "watch"
        elif action == "PROMPT":
            return "intervene"
        else:
            return "watch"
    df["expert_cat"] = df["expert_action"].apply(expert_to_cat)
    return df


def compute_metrics(col_true, col_pred, label_intervene="intervene"):
    """Compute precision, recall, F1 for a binary intervention detection task."""
    true_pos = ((col_true != "watch") & (col_pred != "watch")).sum()
    false_pos = ((col_true == "watch") & (col_pred != "watch")).sum()
    false_neg = ((col_true != "watch") & (col_pred == "watch")).sum()
    true_neg = ((col_true == "watch") & (col_pred == "watch")).sum()

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_pos + true_neg) / len(col_true)

    return {
        "TP": int(true_pos), "FP": int(false_pos),
        "FN": int(false_neg), "TN": int(true_neg),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
    }


def compute_three_class_metrics(col_true, col_pred):
    """Compute per-class and overall metrics for watch/probe/intervene."""
    cats = ["watch", "probe", "intervene"]
    results = {}
    for cat in cats:
        tp = ((col_true == cat) & (col_pred == cat)).sum()
        fp = ((col_true != cat) & (col_pred == cat)).sum()
        fn = ((col_true == cat) & (col_pred != cat)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[cat] = {"precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
    overall_acc = (col_true == col_pred).mean()
    results["overall_accuracy"] = round(overall_acc, 3)
    return results


def temporal_tolerance_analysis(prompts_df, merged, tolerances=(0, 5, 10, 15, 20, 30)):
    """
    Event-level evaluation with temporal tolerance.

    For each facilitator prompt block, check if IO or Expert made a non-watch
    decision within [prompt_start - tolerance, prompt_end + tolerance].

    Returns:
      - results: list of dicts with tolerance, system, precision, recall, f1
      - per_prompt_detail: DataFrame with per-prompt hit/miss for each system
    """
    io_users = set(merged["participant_id"].unique())
    # Only evaluate prompts for users in the 11-user overlap
    prompts_overlap = prompts_df[prompts_df["participant_id"].isin(io_users)].copy()

    results = []
    detail_rows = []

    for tol in tolerances:
        io_hits = 0
        ex_hits = 0
        total_prompts = 0

        for _, prompt in prompts_overlap.iterrows():
            pid = prompt["participant_id"]
            p_start = prompt["rel_start_sec"]
            p_end = prompt["rel_end_sec"]

            # Windows for this participant
            p_merged = merged[merged["participant_id"] == pid]

            # Windows within tolerance range
            nearby = p_merged[
                (p_merged["window_start"] >= p_start - tol) &
                (p_merged["window_start"] <= p_end + tol)
            ]

            if nearby.empty:
                continue

            total_prompts += 1
            io_hit = (nearby["io_cat"] != "watch").any()
            ex_hit = (nearby["expert_cat"] != "watch").any()

            if io_hit:
                io_hits += 1
            if ex_hit:
                ex_hits += 1

            if tol == 15:  # Save detail for the default tolerance
                detail_rows.append({
                    "participant_id": pid,
                    "prompt_type": prompt["prompt_type"],
                    "prompt_start_sec": p_start,
                    "prompt_end_sec": p_end,
                    "puzzle": prompt["puzzle"],
                    "io_hit": io_hit,
                    "expert_hit": ex_hit,
                    "io_cats_nearby": ",".join(nearby["io_cat"].unique()),
                    "expert_cats_nearby": ",".join(nearby["expert_cat"].unique()),
                })

        # Recall = prompts detected / total prompts
        io_recall = io_hits / total_prompts if total_prompts > 0 else 0
        ex_recall = ex_hits / total_prompts if total_prompts > 0 else 0

        # Precision: of all system "active" windows, how many are near a real prompt?
        # Build per-participant prompt intervals for fast lookup
        io_active = merged[merged["io_cat"] != "watch"]
        ex_active = merged[merged["expert_cat"] != "watch"]

        def count_near_prompt(active_df, prompts_by_pid, tol):
            count = 0
            for pid, group in active_df.groupby("participant_id"):
                if pid not in prompts_by_pid:
                    continue
                p_starts = prompts_by_pid[pid]["starts"] - tol
                p_ends = prompts_by_pid[pid]["ends"] + tol
                for ws in group["window_start"].values:
                    if ((p_starts <= ws) & (p_ends >= ws)).any():
                        count += 1
            return count

        prompts_by_pid = {}
        for pid, grp in prompts_overlap.groupby("participant_id"):
            prompts_by_pid[pid] = {
                "starts": grp["rel_start_sec"].values,
                "ends": grp["rel_end_sec"].values,
            }

        io_tp_windows = count_near_prompt(io_active, prompts_by_pid, tol)
        ex_tp_windows = count_near_prompt(ex_active, prompts_by_pid, tol)

        io_prec = io_tp_windows / len(io_active) if len(io_active) > 0 else 0
        ex_prec = ex_tp_windows / len(ex_active) if len(ex_active) > 0 else 0

        io_f1 = 2 * io_prec * io_recall / (io_prec + io_recall) if (io_prec + io_recall) > 0 else 0
        ex_f1 = 2 * ex_prec * ex_recall / (ex_prec + ex_recall) if (ex_prec + ex_recall) > 0 else 0

        results.append({
            "tolerance_sec": tol,
            "total_prompts": total_prompts,
            "io_hits": io_hits, "io_recall": round(io_recall, 3),
            "io_precision": round(io_prec, 3), "io_f1": round(io_f1, 3),
            "ex_hits": ex_hits, "ex_recall": round(ex_recall, 3),
            "ex_precision": round(ex_prec, 3), "ex_f1": round(ex_f1, 3),
        })
        print(f"  ±{tol:2d}s: IO recall={io_recall:.3f} prec={io_prec:.3f} F1={io_f1:.3f} | "
              f"Expert recall={ex_recall:.3f} prec={ex_prec:.3f} F1={ex_f1:.3f} "
              f"({total_prompts} prompts)")

    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame()
    return pd.DataFrame(results), detail_df


def episode_level_evaluation(prompts_df, merged, tolerance=15):
    """
    Part 6: Group facilitator prompts into 'struggle episodes' (consecutive
    prompts within 60s of each other) and evaluate at the episode level.

    Returns a DataFrame with per-episode metrics:
      - Was the episode detected by IO? (at least one probe/intervene within tolerance)
      - Detection latency: how many seconds before/after the episode start did IO first react?
      - Episode duration
    """
    io_users = set(merged["participant_id"].unique())
    prompts_overlap = prompts_df[prompts_df["participant_id"].isin(io_users)].copy()

    episodes = []
    episode_gap = 60.0  # seconds between prompts to be same episode

    for pid in prompts_overlap["participant_id"].unique():
        p_prompts = prompts_overlap[prompts_overlap["participant_id"] == pid].sort_values("rel_start_sec")
        if len(p_prompts) == 0:
            continue

        # Group into episodes
        ep_start = None
        ep_end = None
        ep_prompts = 0
        ep_types = []

        for _, pr in p_prompts.iterrows():
            if ep_start is None:
                ep_start = pr["rel_start_sec"]
                ep_end = pr["rel_end_sec"]
                ep_prompts = 1
                ep_types = [pr["prompt_type"]]
            elif pr["rel_start_sec"] - ep_end <= episode_gap:
                ep_end = max(ep_end, pr["rel_end_sec"])
                ep_prompts += 1
                ep_types.append(pr["prompt_type"])
            else:
                episodes.append(_eval_episode(pid, ep_start, ep_end, ep_prompts, ep_types,
                                               merged, tolerance))
                ep_start = pr["rel_start_sec"]
                ep_end = pr["rel_end_sec"]
                ep_prompts = 1
                ep_types = [pr["prompt_type"]]

        if ep_start is not None:
            episodes.append(_eval_episode(pid, ep_start, ep_end, ep_prompts, ep_types,
                                           merged, tolerance))

    return pd.DataFrame(episodes)


def _eval_episode(pid, ep_start, ep_end, n_prompts, prompt_types, merged, tolerance):
    """Evaluate a single struggle episode."""
    p_merged = merged[merged["participant_id"] == pid]

    # Windows within the episode ± tolerance
    nearby = p_merged[
        (p_merged["window_start"] >= ep_start - tolerance) &
        (p_merged["window_start"] <= ep_end + tolerance)
    ]

    io_detected = (nearby["io_cat"] != "watch").any() if len(nearby) > 0 else False
    ex_detected = (nearby["expert_cat"] != "watch").any() if len(nearby) > 0 else False

    # Detection latency: first IO non-watch window relative to episode start
    io_latency = None
    if io_detected:
        first_io = nearby[nearby["io_cat"] != "watch"]["window_start"].min()
        io_latency = first_io - ep_start  # negative = early detection

    return {
        "participant_id": pid,
        "episode_start": ep_start,
        "episode_end": ep_end,
        "episode_duration": ep_end - ep_start,
        "n_prompts": n_prompts,
        "has_explicit": "explicit" in prompt_types,
        "io_detected": io_detected,
        "expert_detected": ex_detected,
        "io_latency_sec": io_latency,
    }


def generate_report(merged, fac_windows, prompts_df, tol_results=None, tol_detail=None, episodes_df=None):
    """Generate markdown benchmark report."""
    lines = []
    lines.append("# Facilitator Benchmark Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # --- Overview ---
    lines.append("## 1. Facilitator Prompt Overview (18 users)")
    lines.append("")
    total_prompts = len(prompts_df)
    n_reflective = (prompts_df["prompt_type"] == "reflective").sum()
    n_explicit = (prompts_df["prompt_type"] == "explicit").sum()
    lines.append(f"- Total prompt blocks: **{total_prompts}** ({n_reflective} reflective, {n_explicit} explicit)")

    # Per-puzzle breakdown
    puzzle_counts = prompts_df.groupby(["puzzle", "prompt_type"]).size().unstack(fill_value=0)
    lines.append(f"\n### Prompts by puzzle\n")
    # Manual markdown table
    cols = list(puzzle_counts.columns)
    lines.append("| Puzzle | " + " | ".join(cols) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(cols)) + "|")
    for idx, row in puzzle_counts.iterrows():
        lines.append(f"| {idx} | " + " | ".join(str(int(row[c])) for c in cols) + " |")

    # --- Window-level facilitator distribution ---
    lines.append("\n## 2. Facilitator Window Distribution")
    fac_dist = fac_windows["facilitator_cat"].value_counts()
    total_w = len(fac_windows)
    lines.append("")
    for cat in ["watch", "probe", "intervene"]:
        n = fac_dist.get(cat, 0)
        lines.append(f"- **{cat}**: {n} ({n/total_w*100:.1f}%)")

    # --- Three-way comparison (11 users) ---
    lines.append("\n## 3. Three-Way Comparison (11 users with IO + Expert + Facilitator)")
    n_overlap = merged["participant_id"].nunique()
    lines.append(f"\n- Overlapping participants: **{n_overlap}**")
    lines.append(f"- Total windows: **{len(merged)}**")

    # Distribution comparison
    lines.append("\n### Decision distribution\n")
    lines.append("| Category | IO | Expert | Facilitator |")
    lines.append("|----------|-----|--------|-------------|")
    for cat in ["watch", "probe", "intervene"]:
        io_n = (merged["io_cat"] == cat).sum()
        ex_n = (merged["expert_cat"] == cat).sum()
        fac_n = (merged["facilitator_cat"] == cat).sum()
        lines.append(f"| {cat} | {io_n} ({io_n/len(merged)*100:.1f}%) | {ex_n} ({ex_n/len(merged)*100:.1f}%) | {fac_n} ({fac_n/len(merged)*100:.1f}%) |")

    # --- Binary detection: "did system detect when facilitator intervened?" ---
    lines.append("\n## 4. Intervention Detection (Facilitator as Ground Truth)")
    lines.append("\nBinary task: facilitator != watch → positive (someone needed help)\n")

    io_binary = compute_metrics(merged["facilitator_cat"], merged["io_cat"])
    ex_binary = compute_metrics(merged["facilitator_cat"], merged["expert_cat"])

    lines.append("| Metric | IO | Expert |")
    lines.append("|--------|-----|--------|")
    for m in ["precision", "recall", "f1", "accuracy", "TP", "FP", "FN", "TN"]:
        lines.append(f"| {m} | {io_binary[m]} | {ex_binary[m]} |")

    # --- Three-class evaluation ---
    lines.append("\n## 5. Three-Class Evaluation (watch / probe / intervene)")
    lines.append("\nFacilitator categories: watch=no prompt, probe=reflective, intervene=explicit\n")

    io_3c = compute_three_class_metrics(merged["facilitator_cat"], merged["io_cat"])
    ex_3c = compute_three_class_metrics(merged["facilitator_cat"], merged["expert_cat"])

    lines.append("| Class | IO Precision | IO Recall | IO F1 | Expert Precision | Expert Recall | Expert F1 |")
    lines.append("|-------|-------------|-----------|-------|-----------------|---------------|-----------|")
    for cat in ["watch", "probe", "intervene"]:
        lines.append(f"| {cat} | {io_3c[cat]['precision']} | {io_3c[cat]['recall']} | {io_3c[cat]['f1']} | {ex_3c[cat]['precision']} | {ex_3c[cat]['recall']} | {ex_3c[cat]['f1']} |")
    lines.append(f"\n- IO overall accuracy: **{io_3c['overall_accuracy']}**")
    lines.append(f"- Expert overall accuracy: **{ex_3c['overall_accuracy']}**")

    # --- Per-puzzle analysis ---
    lines.append("\n## 6. Per-Puzzle Breakdown")
    for puzzle in sorted(merged["puzzle_id"].dropna().unique()):
        if puzzle == "Transition":
            continue
        sub = merged[merged["puzzle_id"] == puzzle]
        if len(sub) < 10:
            continue
        fac_rate = (sub["facilitator_cat"] != "watch").mean()
        io_rate = (sub["io_cat"] != "watch").mean()
        ex_rate = (sub["expert_cat"] != "watch").mean()
        lines.append(f"\n### {puzzle}")
        lines.append(f"- Windows: {len(sub)}")
        lines.append(f"- Facilitator intervention rate: {fac_rate:.1%}")
        lines.append(f"- IO intervention/probe rate: {io_rate:.1%}")
        lines.append(f"- Expert intervention rate: {ex_rate:.1%}")

    # --- Key insight: What does IO see when facilitator intervenes? ---
    lines.append("\n## 7. IO Tension Patterns During Facilitator Interventions")
    fac_active = merged[merged["facilitator_cat"] != "watch"]
    if "disagreement_pattern" in fac_active.columns and len(fac_active) > 0:
        pattern_dist = fac_active["disagreement_pattern"].value_counts()
        lines.append(f"\nWhen the facilitator gave a prompt ({len(fac_active)} windows), IO's tension patterns were:\n")
        for pat, cnt in pattern_dist.items():
            lines.append(f"- **{pat}**: {cnt} ({cnt/len(fac_active)*100:.1f}%)")

    # --- What IO category was active when facilitator prompted ---
    lines.append("\n## 8. IO Decision When Facilitator Prompted")
    if len(fac_active) > 0:
        io_at_prompt = fac_active["io_cat"].value_counts()
        lines.append(f"\nIO's decision at facilitator prompt moments:\n")
        for cat, cnt in io_at_prompt.items():
            lines.append(f"- **{cat}**: {cnt} ({cnt/len(fac_active)*100:.1f}%)")

    # --- Expert decision when facilitator prompted ---
    lines.append("\n## 9. Expert Decision When Facilitator Prompted")
    if len(fac_active) > 0:
        ex_at_prompt = fac_active["expert_cat"].value_counts()
        lines.append(f"\nExpert's decision at facilitator prompt moments:\n")
        for cat, cnt in ex_at_prompt.items():
            lines.append(f"- **{cat}**: {cnt} ({cnt/len(fac_active)*100:.1f}%)")

    # --- Temporal tolerance analysis ---
    if tol_results is not None and len(tol_results) > 0:
        lines.append("\n## 10. Temporal Tolerance Analysis (Event-Level)")
        lines.append("\nInstead of exact window matching, check if system acted within ±N seconds of each facilitator prompt.")
        lines.append("This is fairer because prompts are continuous blocks spanning multiple windows.\n")

        lines.append("| Tolerance | IO Recall | IO Precision | IO F1 | Expert Recall | Expert Precision | Expert F1 |")
        lines.append("|-----------|-----------|-------------|-------|--------------|-----------------|-----------|")
        for _, r in tol_results.iterrows():
            lines.append(f"| ±{int(r['tolerance_sec'])}s | {r['io_recall']} | {r['io_precision']} | {r['io_f1']} | {r['ex_recall']} | {r['ex_precision']} | {r['ex_f1']} |")

        # Highlight key finding
        tol15 = tol_results[tol_results["tolerance_sec"] == 15]
        if len(tol15) > 0:
            r = tol15.iloc[0]
            lines.append(f"\nAt **±15s tolerance** (recommended):")
            lines.append(f"- IO detects **{r['io_recall']:.1%}** of facilitator prompts (F1={r['io_f1']})")
            lines.append(f"- Expert detects **{r['ex_recall']:.1%}** of facilitator prompts (F1={r['ex_f1']})")

    # --- Per-prompt detail at ±15s ---
    if tol_detail is not None and len(tol_detail) > 0:
        lines.append("\n## 11. Per-Prompt Detection Detail (±15s tolerance)")

        # By prompt type
        for pt in ["reflective", "explicit"]:
            sub = tol_detail[tol_detail["prompt_type"] == pt]
            if len(sub) == 0:
                continue
            io_rate = sub["io_hit"].mean()
            ex_rate = sub["expert_hit"].mean()
            lines.append(f"\n### {pt.capitalize()} prompts ({len(sub)} total)")
            lines.append(f"- IO detection rate: **{io_rate:.1%}**")
            lines.append(f"- Expert detection rate: **{ex_rate:.1%}**")

        # By puzzle
        lines.append("\n### Detection rate by puzzle (±15s)")
        lines.append("\n| Puzzle | N prompts | IO hit rate | Expert hit rate |")
        lines.append("|--------|-----------|-------------|-----------------|")
        for puzzle in sorted(tol_detail["puzzle"].dropna().unique()):
            sub = tol_detail[tol_detail["puzzle"] == puzzle]
            io_r = sub["io_hit"].mean()
            ex_r = sub["expert_hit"].mean()
            lines.append(f"| {puzzle} | {len(sub)} | {io_r:.1%} | {ex_r:.1%} |")

        # Per-participant
        lines.append("\n### Detection rate by participant (±15s)")
        lines.append("\n| Participant | N prompts | IO hit rate | Expert hit rate |")
        lines.append("|-------------|-----------|-------------|-----------------|")
        for pid in sorted(tol_detail["participant_id"].unique()):
            sub = tol_detail[tol_detail["participant_id"] == pid]
            io_r = sub["io_hit"].mean()
            ex_r = sub["expert_hit"].mean()
            lines.append(f"| User-{pid} | {len(sub)} | {io_r:.1%} | {ex_r:.1%} |")

    # --- Episode-level evaluation ---
    if episodes_df is not None and len(episodes_df) > 0:
        lines.append("\n## 12. Episode-Level Evaluation")
        lines.append(f"\nStruggle episodes: consecutive facilitator prompts grouped within 60s.\n")
        lines.append(f"- Total episodes: **{len(episodes_df)}**")
        lines.append(f"- IO episode recall: **{episodes_df['io_detected'].mean():.1%}**")
        lines.append(f"- Expert episode recall: **{episodes_df['expert_detected'].mean():.1%}**")
        lines.append(f"- Mean episode duration: **{episodes_df['episode_duration'].mean():.0f}s**")

        detected = episodes_df[episodes_df["io_detected"]]
        if len(detected) > 0:
            lines.append(f"- Mean IO detection latency: **{detected['io_latency_sec'].mean():.1f}s** "
                          f"(negative = early detection)")
            early = (detected["io_latency_sec"] < 0).sum()
            lines.append(f"- IO detected **before** episode start: {early}/{len(detected)} ({early/len(detected):.0%})")

        # By episode severity (has explicit prompt = more severe)
        for severity, label in [(True, "Severe (has explicit prompt)"), (False, "Mild (reflective only)")]:
            sub = episodes_df[episodes_df["has_explicit"] == severity]
            if len(sub) > 0:
                lines.append(f"\n### {label} ({len(sub)} episodes)")
                lines.append(f"- IO recall: **{sub['io_detected'].mean():.1%}**")
                lines.append(f"- Expert recall: **{sub['expert_detected'].mean():.1%}**")

    return "\n".join(lines)


def main():
    print("=== Facilitator Benchmark ===\n")

    # Step 1: Load game start times
    print("Loading game start times...")
    game_starts = load_game_starts()
    print(f"  Found {len(game_starts)} users\n")

    # Step 2: Load facilitator prompts and convert to relative time
    print("Loading facilitator prompts...")
    prompts_df = load_facilitator_prompts(game_starts)
    print(f"  Loaded {len(prompts_df)} prompt blocks from {prompts_df['participant_id'].nunique()} users\n")

    # Step 3: Load IO agent outputs (11 users) — these define the 5-sec windows
    print("Loading IO agent outputs...")
    io_df = pd.read_csv(os.path.join(OUTPUTS, "agent_outputs.csv"))
    io_users = set(io_df["participant_id"].unique())
    print(f"  {len(io_df)} windows, {len(io_users)} users\n")

    # Step 4: Load expert outputs (18 users)
    print("Loading expert outputs...")
    expert_df = build_expert_windows()
    print(f"  {len(expert_df)} windows, {expert_df['participant_id'].nunique()} users\n")

    # Step 5: Build a unified window set for all 18 users (from expert, which covers all)
    # For the 11 IO users, use IO windows (they have richer data)
    # For the other 7, use expert windows
    print("Assigning facilitator prompts to 5-second windows...")

    # Use expert windows as the base for all 18 users
    all_windows = expert_df[["participant_id", "window_start"]].copy()
    fac_windows = assign_facilitator_to_windows(prompts_df, all_windows)
    print(f"  Assigned to {len(fac_windows)} windows\n")

    # Save facilitator windows
    fac_out = os.path.join(OUTPUTS, "facilitator_windows.csv")
    fac_windows.to_csv(fac_out, index=False)
    print(f"  Saved: {fac_out}")

    # Step 6: Merge three systems for 11 overlap users
    print("\nBuilding three-way comparison (11 users)...")

    # IO categories — round window_start to 5-sec grid to match expert
    io_cats = io_df[["participant_id", "window_start", "support_category",
                      "disagreement_pattern", "disagreement_intensity",
                      "dominant_tension", "puzzle_id"]].copy()
    io_cats = io_cats.rename(columns={"support_category": "io_cat"})
    io_cats["io_cat"] = io_cats["io_cat"].replace({"consensus_intervene": "intervene"})
    io_cats["window_start_round"] = (io_cats["window_start"] // 5 * 5).astype(float)

    # Expert categories
    expert_cats = expert_df[["participant_id", "window_start", "expert_state",
                              "expert_action", "expert_cat"]].copy()

    # Merge IO + expert on rounded window
    merged = pd.merge(
        io_cats, expert_cats,
        left_on=["participant_id", "window_start_round"],
        right_on=["participant_id", "window_start"],
        how="inner", suffixes=("_io", "_expert"))

    # Use IO's original window_start for facilitator alignment
    merged["window_start"] = merged["window_start_round"]

    # Merge with facilitator (facilitator was assigned on expert's 5-sec grid)
    merged = pd.merge(merged, fac_windows,
                       on=["participant_id", "window_start"], how="left")
    merged["facilitator_cat"] = merged["facilitator_cat"].fillna("watch")
    merged["facilitator_prompt_type"] = merged["facilitator_prompt_type"].fillna("none")

    print(f"  Merged: {len(merged)} windows, {merged['participant_id'].nunique()} users")

    # Save
    three_way_out = os.path.join(OUTPUTS, "three_way_comparison.csv")
    merged.to_csv(three_way_out, index=False)
    print(f"  Saved: {three_way_out}")

    # Step 7: Temporal tolerance analysis
    print("\nTemporal tolerance analysis (event-level)...")
    tol_results, tol_detail = temporal_tolerance_analysis(prompts_df, merged)

    # Save tolerance results
    tol_results.to_csv(os.path.join(OUTPUTS, "tolerance_results.csv"), index=False)
    if len(tol_detail) > 0:
        tol_detail.to_csv(os.path.join(OUTPUTS, "prompt_detection_detail.csv"), index=False)
    print(f"  Saved: tolerance_results.csv, prompt_detection_detail.csv")

    # Step 8: Episode-level evaluation
    print("\nEpisode-level evaluation...")
    episodes_df = episode_level_evaluation(prompts_df, merged)
    if len(episodes_df) > 0:
        episodes_df.to_csv(os.path.join(OUTPUTS, "episode_evaluation.csv"), index=False)
        print(f"  {len(episodes_df)} episodes, IO recall={episodes_df['io_detected'].mean():.1%}, "
              f"Expert recall={episodes_df['expert_detected'].mean():.1%}")
        detected = episodes_df[episodes_df["io_detected"]]
        if len(detected) > 0:
            print(f"  Mean IO latency: {detected['io_latency_sec'].mean():.1f}s")

    # Step 9: Generate report
    print("\nGenerating benchmark report...")
    report = generate_report(merged, fac_windows, prompts_df, tol_results, tol_detail, episodes_df)
    report_path = os.path.join(OUTPUTS, "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    fac_active_rate = (merged["facilitator_cat"] != "watch").mean()
    io_active_rate = (merged["io_cat"] != "watch").mean()
    ex_active_rate = (merged["expert_cat"] != "watch").mean()
    print(f"Facilitator intervention rate: {fac_active_rate:.1%}")
    print(f"IO intervention/probe rate:    {io_active_rate:.1%}")
    print(f"Expert intervention rate:      {ex_active_rate:.1%}")

    # Quick binary detection scores
    io_m = compute_metrics(merged["facilitator_cat"], merged["io_cat"])
    ex_m = compute_metrics(merged["facilitator_cat"], merged["expert_cat"])
    print(f"\nBinary detection (facilitator as ground truth):")
    print(f"  IO:     P={io_m['precision']} R={io_m['recall']} F1={io_m['f1']}")
    print(f"  Expert: P={ex_m['precision']} R={ex_m['recall']} F1={ex_m['f1']}")


if __name__ == "__main__":
    main()
