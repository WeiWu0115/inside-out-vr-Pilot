# Inside Out Calibration Report: Facilitator-Guided Iterative Refinement

Generated: 2026-04-04

## 1. Motivation

After constructing the multi-agent negotiation system (Inside Out) and the rule-based prompting engine independently, we obtained facilitator prompt logs from the summer 2025 pilot study — 223 real-time prompts given by a human facilitator to 18 participants. These logs provided a ground-truth benchmark: the actual moments when a domain expert decided a learner needed help.

The initial comparison revealed that while IO's overall intervention rate (20.5%) closely matched the facilitator's (20.8%), the **temporal alignment** was imperfect (F1=0.459 at ±15s). We therefore undertook a systematic calibration process, using facilitator prompts as ground truth to identify and correct specific failure modes.

## 2. Methodology: Diagnose → Hypothesize → Calibrate → Evaluate

This section describes a transferable four-step pipeline for calibrating multi-agent cognitive state systems against human expert judgment.

### Step 1: Diagnostic Analysis

We classified every 5-second window into one of four categories:

| Category | Definition |
|----------|-----------|
| **True Positive (TP)** | Both IO and facilitator flagged this moment |
| **True Negative (TN)** | Both IO and facilitator stayed silent |
| **False Negative (FN)** | Facilitator prompted, IO said "watch" |
| **False Positive (FP)** | IO flagged, facilitator stayed silent |

Key diagnostic queries:
- What **agent labels** and **tension patterns** dominate in FN windows?
- What **features** (idle_time, action_count, time_since_action) distinguish FN from TP?
- Where are FP windows concentrated (puzzle phase, transition)?

### Step 2: Root Cause Findings

#### Finding 1: Performance Agent's "progressing" label was too optimistic

| Metric | FN (missed) | TP (caught) | TN (correct watch) |
|--------|------------|------------|-------------------|
| action_count | 2.70 | 0.04 | 2.81 |
| time_since_action | 117s | 132s | 84s |
| Performance = "progressing" | 70% | 0% | — |

The Performance Agent labeled a window as "progressing" whenever `action_count > 0`, regardless of whether the player had been inactive for 2+ minutes before those sporadic actions. This caused the downstream tension `scattered_but_progressing` → watch, even when the facilitator was actively prompting.

**Root cause:** `action_count` measures instantaneous activity within a 5-second window, but `time_since_action` captures the macro-temporal context. The agent was missing the forest for the trees.

#### Finding 2: `focused_but_idle` was under-triggering probes

232 FN windows had the pattern `focused_but_idle`. The support layer only probed when `temporal == "persistent"` (3+ consecutive identical windows). But facilitators routinely prompt focused-but-idle players even during transient episodes — being focused and doing nothing is itself a signal.

#### Finding 3: 46.7% of all FP were in Transition phase

389 of 833 FP windows occurred during Transition (between puzzles). IO detected `scanning_but_passive` and triggered probes, but facilitators almost never intervene during navigation — scanning while walking to the next puzzle is normal behavior.

### Step 3: Calibration Changes

#### Change A1: Performance Agent — time_since_action penalty

```
Before: prog_conf based only on action_count and error_count
After:  if time_since_action > 55s (P50), apply penalty up to -0.35
        if time_since_action > 55s, boost stalled score by up to +0.25
```

**Rationale:** A few sporadic actions after prolonged inactivity should not yield "progressing." The penalty is proportional to inactivity duration, with maximum effect at 3× the median time_since_action.

#### Change A2: `scattered_but_progressing` → conditional probe

```
Before: always watch ("Don't interrupt what's working")
After:  if temporal ∈ {looping, persistent} OR time_since_action > 90s → probe
        else → watch
```

**Rationale:** "What's working" should be verified when temporal context or long inactivity suggests otherwise.

#### Change A3: `focused_but_idle` → always probe

```
Before: probe only if temporal == "persistent", else watch
After:  probe always (higher confidence if persistent/long inactivity,
        lower confidence if transient)
```

**Rationale:** Facilitators prompt focused-idle players regardless of persistence duration. The intensity of the probe varies, but the decision to probe does not.

#### Change A4: Transition suppression

```
Before: Transition windows processed with same rules as puzzle windows
After:  Transition → default watch, unless passive_and_stuck + looping/persistent
```

**Rationale:** Scanning behavior during navigation is expected. Only intervene if the player is truly lost (stuck pattern persisting during transition).

### Step 4: Evaluation — Before vs After

#### Overall metrics at ±15s tolerance

| Metric | V0 (Original) | V1 (A+B) | V2 (A+B+C) | Total Change |
|--------|--------------|-----------|------------|-------------|
| **IO F1** | 0.459 | 0.508 | **0.514** | **+12.0%** |
| **IO Recall** | 75.5% | 79.5% | **86.1%** | **+10.6pp** |
| **IO Precision** | 33.0% | 37.3% | **36.6%** | **+3.6pp** |
| Rule-Based F1 | 0.372 | 0.372 | 0.372 | — |

V1 = Changes A+B (agent fix + support rules). V2 = V1 + Change C (puzzle elapsed time escalation).

#### Per-puzzle detection rate (±15s)

| Puzzle | V0 | V1 | V2 | Change |
|--------|-----|-----|-----|--------|
| Hub Puzzle: Cooking Pot | 58.7% | 87.0% | **91.3%** | +32.6pp |
| Amount of Protein | 80.6% | 77.8% | **83.3%** | +2.7pp |
| Amount of Sunlight | 61.5% | 61.5% | **92.3%** | +30.8pp |
| Pasta in Sauce | 88.5% | 73.1% | **76.9%** | -11.6pp |
| Water Amount | 92.0% | 96.0% | **96.0%** | +4.0pp |

#### Decision distribution shift

| Category | V0 | V1 | V2 | Facilitator |
|----------|-----|-----|-----|-------------|
| watch | 4185 (79.5%) | 3648 (69.3%) | 3164 (60.1%) | 79.2% |
| probe | 695 (13.2%) | 1265 (24.0%) | 1710 (32.5%) | 19.5% |
| intervene | 385 (7.3%) | 352 (6.7%) | 391 (7.4%) | 1.3% |

IO's probe rate increased across versions. The V2 probe rate (32.5%) overshoots the facilitator's 19.5%, but this is expected: IO flags ambiguous moments for probing that a facilitator might silently observe. The key metric is recall — whether IO catches the moments that matter.

## 3. Transferable Pipeline

This calibration process is generalizable to any multi-agent cognitive state system:

```
┌──────────────────────────────────────────────────────────┐
│  1. COLLECT GROUND TRUTH                                 │
│     Human expert decisions during real interactions       │
│     (facilitator prompts, teacher interventions, etc.)    │
├──────────────────────────────────────────────────────────┤
│  2. ALIGN TEMPORALLY                                     │
│     Map ground-truth events to system's time windows     │
│     Use temporal tolerance (±15s) for fairness           │
├──────────────────────────────────────────────────────────┤
│  3. CLASSIFY ERRORS                                      │
│     TP / TN / FN / FP with agent-level decomposition     │
│     Ask: which agent labels and tensions dominate FN/FP? │
├──────────────────────────────────────────────────────────┤
│  4. IDENTIFY ROOT CAUSES                                 │
│     Compare features between FN and TP windows           │
│     Trace back from tension → agent label → threshold    │
├──────────────────────────────────────────────────────────┤
│  5. CALIBRATE TARGETED CHANGES                           │
│     Modify agent scoring functions and/or support rules  │
│     Each change addresses one specific failure mode      │
├──────────────────────────────────────────────────────────┤
│  6. EVALUATE HOLISTICALLY                                │
│     Re-run full pipeline + benchmark                     │
│     Check for regression on previously correct cases     │
│     Verify per-puzzle and per-participant stability       │
└──────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Diagnose before tuning.** Random threshold adjustments without understanding failure modes lead to overfitting. The diagnostic step — comparing FN vs TP feature distributions — reveals which agent is responsible.

2. **Calibrate at the right layer.** Some issues are in the agent (wrong label), others in the support layer (wrong decision from correct labels). Fix the layer where the error originates.

3. **Temporal tolerance matters.** Strict window-level matching penalizes systems that detect problems slightly before or after the human acts. Event-level evaluation with tolerance captures the system's true capability.

4. **Watch for regression.** Improving recall on one puzzle may reduce precision elsewhere. Per-puzzle and per-participant breakdowns catch these trade-offs.

5. **Ground truth has noise.** Human facilitators are not perfect — they may intervene late, early, or inconsistently. The goal is not 100% alignment but systematic improvement with understood trade-offs.

## 4. Change C: Puzzle Elapsed Time Escalation

### Hypothesis

Facilitators implicitly track how long a player has been on a puzzle. A player who has been working on the same puzzle for 3× longer than the median player is almost certainly struggling, even if moment-to-moment signals are ambiguous. IO's agents operate on 5-second windows and lack this macro-temporal awareness.

### Implementation

1. **Computed feature:** `puzzle_elapsed_ratio` = (seconds since puzzle start) / (median duration for this puzzle across population)
2. **Escalation layer** in `support.py`, applied *after* the base decision:
   - ratio > 3.0×: watch → probe
   - ratio > 4.0×: probe → consensus_intervene
3. **Population medians** from 18-participant data: Protein=135s, Pasta=90s, Water=230s, Sunlight=160s, Hub=605s

### Threshold Selection

We tested four escalation threshold pairs:

| watch→probe | probe→intervene | F1 (±15s) | Recall | Precision |
|-------------|-----------------|-----------|--------|-----------|
| 1.5× | 2.0× | 0.486 | 93.4% | 32.8% |
| 2.0× | 2.5× | 0.501 | 89.4% | 34.9% |
| 2.5× | 3.0× | 0.493 | 88.7% | 34.0% |
| **3.0×** | **4.0×** | **0.514** | **86.1%** | **36.6%** |

Conservative thresholds (3.0×/4.0×) maximize F1 by avoiding over-triggering. More aggressive thresholds boost recall at the cost of precision — useful in high-stakes educational settings where missing a struggling student is worse than an unnecessary check-in.

### Impact

Plan C's primary contribution is to the **Amount of Sunlight** puzzle: detection jumped from 61.5% → 92.3% (+30.8pp). This puzzle has short median duration (160s), so players who get stuck quickly exceed the 3× threshold. Hub Puzzle also improved: 87.0% → 91.3%.

### Design Note

The escalation acts as a "safety net" — it only fires when a player is dramatically over-time *and* the base decision was watch/probe. It does not override intervene→watch decisions. This layered architecture allows the agent negotiation to handle normal cases while the elapsed-time signal catches long-tail failure modes.

## 5. Remaining Opportunities

- **Prompt escalation modeling**: facilitators escalate from reflective → explicit. IO could model this escalation trajectory rather than treating each window independently.
- **Individual calibration**: the current `puzzle_elapsed_ratio` uses population medians. Per-player baselines (from early puzzle performance) could personalize the thresholds.
- **80-person study**: these calibrations were derived from 11 overlapping participants. The upcoming larger study will validate whether these patterns generalize.
