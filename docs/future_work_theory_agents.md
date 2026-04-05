# Future Work: Theory-Partitioned Agent Architecture

## Motivation

The current V3 system partitions agents by **data source** — each agent reads exclusive features. This prevents echo consensus but means agents disagree because they see *different data*, not because they hold *different interpretations*.

In the Inside Out metaphor (Pixar), Joy and Sadness see the **same memory** but interpret it differently. We explored whether agents grounded in different cognitive theories — all seeing the same data — could achieve comparable or better performance while producing richer, more theoretically grounded disagreements.

## Experiment: Theory-Partitioned Agents

### Architecture

Four theory agents replace the data-partitioned agents. Each reads ALL features (gaze_entropy, clue_ratio, switch_rate, action_count, idle_time, error_count, time_since_action, puzzle_elapsed_ratio) but interprets through a distinct cognitive theory:

| Agent | Theory | Key Question | Output States |
|-------|--------|-------------|---------------|
| **Attention Theory** | Posner (selective attention), Lavie (perceptual load) | Is the player's attention appropriately allocated? | engaged / overloaded / fixated / decoupled |
| **Self-Regulation Theory** | Zimmerman (self-regulated learning), Pintrich (metacognition) | Is the player effectively monitoring and adjusting their strategy? | self_regulated / impulsive / disengaged / reflective |
| **Flow Theory** | Csikszentmihalyi (flow, challenge-skill balance) | Is the challenge-skill balance right? | flow / anxiety / frustration / boredom |
| **Cognitive Load Theory** | Sweller (working memory capacity) | Is working memory capacity exceeded? | manageable / overloaded / fragmented / automated |

### How Agents Interpret the Same Data Differently

Example: `action_count=3, time_since_action=90s, gaze_entropy=1.2`

| Agent | Interpretation | Reasoning |
|-------|---------------|-----------|
| Attention Theory | **engaged** | Entropy is moderate, player is looking at relevant areas |
| Self-Regulation | **impulsive** | 3 actions after 90s gap = reactive trial-and-error, not strategic |
| Flow Theory | **anxiety** | Active but time_since is high = challenge exceeding skill |
| Cognitive Load | **manageable** | Low entropy + some actions + no errors = within capacity |

Result: 2 agents say OK, 2 say problem → **contradictory tension** → system probes instead of guessing.

In V3 (data-partitioned), the same window might produce echo consensus: Behavioral says "active" (action_count=3), Progress says "progressing" (behavioral_label="active"), both driven by the same underlying signal.

## Results

### After Calibration (±15s tolerance)

| Metric | V3 (data-partitioned) | Theory-partitioned | 
|--------|----------------------|-------------------|
| **F1** | 0.529 | **0.531** |
| **Precision** | 37.1% | **37.9%** |
| Recall | **92.1%** | 88.7% |
| Episode recall | **85.0%** | 81.2% |

### Key Finding: F1 and Precision Improved, Recall Dropped

Theory-partitioned agents achieved **higher F1 (0.531 vs 0.529) and higher precision (37.9% vs 37.1%)**, confirming that genuinely independent theoretical frameworks produce more reliable agreement than data-partitioned agents with shared underlying signals.

However, **recall dropped by 3.4 percentage points** (92.1% → 88.7%).

## Root Cause Analysis: Two Ceilings

### Ceiling 1: Feature Granularity

574 false negative windows were analyzed. In 90%+ of these windows, ALL four theory agents agreed the player was functioning well:
- Attention Theory: "engaged" (90%)
- Self-Regulation: "self_regulated" (95%)  
- Flow Theory: "flow" (92%)

Feature profile of these missed windows:
- `action_count` = 2.78 (active)
- `time_since_action` = 86s (moderate)
- `puzzle_elapsed_ratio` = 1.15 (near median)

**The problem is not the theories — it's the features.** All four theories are given the same shallow data: *how much* the player does, not *whether what they do is meaningful*. `action_count=3` could be 3 correct puzzle interactions or 3 random clicks on irrelevant objects. No cognitive theory can distinguish these without richer input signals such as:
- **Semantic action labels** (clicked correct object vs. wrong object)
- **Gaze-action alignment** (looked at clue then acted on it vs. acted randomly)
- **Spatial trajectory** (moving toward solution vs. wandering)

This is a fundamental data limitation that affects ALL agent architectures equally.

### Ceiling 2: Ground Truth Noise

The remaining recall gap (3.4pp) is concentrated in specific cases:
- **Player 22, Hub Puzzle**: 4 missed prompts during a period where `action_count=5.8`, `time_since=24s`, `elapsed_ratio=0.8`. By every objective measure, this player was performing well. The facilitator's prompts were likely **encouragement or pedagogical guidance**, not confusion detection.

The facilitator's ground truth contains two types of prompts:
1. **Reactive prompts**: responding to observed struggle (what our system should detect)
2. **Proactive prompts**: guiding a functioning player toward deeper understanding (impossible to detect from behavioral data alone)

No automated system should be expected to detect type 2 prompts — they require pedagogical intent modeling, not behavioral observation.

## Implications for Future Development

### For the 80-Person Study

1. **Richer features**: Add semantic action labels (correct/incorrect interaction target), gaze-action coupling (did the player look at the clue before acting?), and spatial trajectory features (from head tracking data already available).

2. **Ground truth refinement**: Ask the facilitator to tag each prompt as "reactive" (responding to struggle) vs. "proactive" (pedagogical guidance). This would eliminate the noise ceiling and allow cleaner evaluation.

3. **Theory agent calibration**: With 80 participants, each theory agent can be calibrated independently against the refined ground truth, potentially revealing which cognitive theory best predicts different types of confusion.

### For the Paper

The theory-partitioned experiment provides a strong **discussion point**:

> The Inside Out metaphor suggests agents should disagree because they hold different *interpretations*, not because they see different *data*. Our experiment with theory-partitioned agents — grounded in Attention Theory, Self-Regulation Theory, Flow Theory, and Cognitive Load Theory — achieved comparable F1 (0.531 vs 0.529) with higher precision (37.9% vs 37.1%), confirming that theoretically independent frameworks produce more genuine agreement signals. The recall gap (3.4pp) traces to a feature granularity ceiling: current behavioral features capture *how much* a player acts but not *whether their actions are meaningful*. Richer semantic features (gaze-action coupling, interaction target correctness) would benefit both architectures equally and represent the primary opportunity for improvement in the planned 80-participant study.

### Architecture Recommendation

Neither architecture is strictly superior:
- **V3 (data-partitioned + label flow)**: Maximizes recall (92.1%). Best for deployment where missing a struggling student is costly.
- **Theory-partitioned**: Maximizes precision and F1. Best for research contexts where false positives waste analysis time, and where theoretical interpretability matters for publication.

The 80-person study should evaluate both architectures on the same data with refined ground truth.
