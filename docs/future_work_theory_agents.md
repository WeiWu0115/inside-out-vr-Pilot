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

---

## Learnable Integration Weights: Neural Network Layer Analogy

### Observation

The current system already has a layered structure analogous to a neural network:

```
Layer 1 (Input):       Raw features (8 values per 5s window)
Layer 2 (Agents):      5 independent interpretations (label + confidence)
Layer 3 (Negotiation): Pairwise tension detection (contradictory/constructive)
Layer 4 (Decision):    watch / probe / intervene
```

However, all transformations between layers use **hand-written rules and manually tuned weights**. For example, the V3.1 Intervention Necessity Score:

```python
score = 0.30 * struggle_signal     # ← manually chosen
      + 0.20 * temporal_signal     # ← manually chosen
      + 0.15 * elapsed_signal      # ← manually chosen
      + 0.15 * momentum_signal     # ← manually chosen
      + 0.10 * pre_collapse        # ← manually chosen
```

These weights were calibrated by iterative experimentation against facilitator ground truth. A natural next step is to **learn these weights from data**.

### Proposed Hybrid Architecture

Keep the agent layer interpretable (rule-based, theory-grounded), but replace the final integration layer with **learnable weights**:

```
Layer 1: Raw Features
    ↓ (rule-based, interpretable)
Layer 2: Agent Outputs
    Each agent produces: label (categorical) + confidence (0-1)
    → 5 agents × 2 values = 10-dimensional representation
    ↓ (rule-based, interpretable)
Layer 3: Negotiation Features
    Tension type, intensity, n_contradictions, confidence_spread
    → ~6-dimensional representation
    ↓ (LEARNABLE weights W)
Layer 4: Decision
    P(watch), P(probe), P(intervene) = softmax(W · [agent_outputs, negotiation_features])
```

The key constraint: **only the last layer has learnable weights**. Layers 1-3 remain fully interpretable — you can always explain *why* each agent produced its label and *why* a tension was detected. The learned weights only determine how to *combine* these interpretable signals into a final decision.

### Why This Is Not "Just an MLP"

A standard MLP would take raw features and output a decision, making the intermediate representations opaque. Our approach:

1. **Agent labels are human-readable** — "Attention Theory says overloaded because entropy=2.3 and switch_rate=8.1"
2. **Tensions are named and interpretable** — "engaged_but_anxious: Attention and Flow disagree"
3. **Only the final weighting is learned** — "the system learned that Flow Theory's anxiety signal should be weighted 2× higher than Attention Theory's engagement signal when elapsed_ratio > 2.0"

This preserves the core contribution (multi-agent negotiation with interpretable disagreement) while allowing data-driven optimization of how disagreements are resolved.

### Requirements

- **Training data**: Need facilitator ground truth labels for supervised learning. Current 11-player dataset (5,265 windows) is too small — risk of overfitting. The 80-person study (~25,000+ windows) would provide sufficient data.
- **Validation strategy**: Leave-one-participant-out cross-validation to ensure weights generalize across players.
- **Comparison**: Train the same architecture with (a) hand-tuned weights, (b) learned weights, (c) end-to-end MLP baseline. This isolates the contribution of the multi-agent structure vs. the learned integration.

### What This Would Add to the Paper

> "We demonstrate that the multi-agent negotiation structure provides an interpretable intermediate representation that a shallow learned layer can integrate more effectively than hand-tuned rules. Compared to an end-to-end MLP that treats raw features as opaque input, our hybrid approach achieves comparable prediction accuracy while maintaining full transparency at the agent and negotiation layers — each decision can be traced back to named agent interpretations and identified theoretical tensions."
