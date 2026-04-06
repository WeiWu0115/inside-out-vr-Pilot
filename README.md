# Inside Out VR: Multi-Agent Negotiation of Cognitive States

**CHI 2027 paper + working prototype.** Instead of classifying a VR player into a single cognitive state, multiple agents each produce their own interpretation, and the system acts based on the *pattern of their disagreement*.

Inspired by Pixar's *Inside Out*: Joy and Sadness see the same memory but interpret it differently. That disagreement is the signal, not noise.

## The Core Idea

```
Traditional:  sensor data → classifier → "stuck" → intervene
Inside Out:   sensor data → 5 agents → they disagree → what does the disagreement mean?
```

A player who is **visually focused but physically idle** could be thinking (good) or frozen (bad). A single classifier must pick one. Inside Out keeps both interpretations and uses the *tension* between them to decide: watch, probe, or intervene.

## What's in This Repo

### Multi-Agent Pipeline (`src/`)

Five agents interpret 5-second windows of VR gameplay:

| Agent | Input | Question | Labels |
|-------|-------|----------|--------|
| Perceptual | gaze_entropy, clue_ratio, switch_rate | What is the player looking at? | focused / searching / locked |
| Behavioral | action_count, idle_time, error_count | What is the player doing? | active / inactive / hesitant / failing |
| Progress | behavioral_label + time_since_action | Is the player making progress? | progressing / stalled / ineffective_progress |
| Temporal | history of agent labels | Is this pattern persistent? | transient / persistent / looping |
| Population | distance to K=5 cluster centroids | How does this compare to other players? | exploring / disoriented / actively_solving / cognitively_stuck |

Each agent reads **exclusive features** (no feature sharing) to prevent echo consensus.

The **negotiation layer** detects pairwise tensions between agents:
- `focused_but_idle` — Perceptual says focused, Behavioral says inactive
- `scattered_but_progressing` — Perceptual says searching, Progress says progressing
- `frozen_on_clue` — Perceptual says locked, Behavioral says inactive

These tensions map to three responses:
- **Watch** (79.5%) — agents agree player is OK
- **Probe** (13.2%) — agents disagree, system explores instead of guessing
- **Intervene** (7.3%) — agents agree player needs help

### Rule-Based Baseline (`src/expert_engine.py`)

Reimplementation of the Unity PromptStateMachine from the VR escape room. Per-puzzle state machine with escalating prompts: 3 Reflective → 2 Vague → unlimited Explicit.

### Facilitator Benchmark (`src/facilitator_benchmark.py`)

Compares both systems against **real facilitator prompts** (223 prompts from a human facilitator observing 18 players). At ±15s temporal tolerance:

| System | Recall | Precision | F1 |
|--------|--------|-----------|----|
| Rule-Based | 37.7% | 32.1% | 0.347 |
| Inside Out | 92.1% | 37.1% | 0.529 |

### Streamlit App (`app.py`)

Interactive dashboard with 7 tabs:

```bash
streamlit run app.py
```

| Tab | What it shows |
|-----|--------------|
| Study Overview | VR escape room layout, heatmaps, movement paths |
| Agent Architecture | How the 5 agents work, IO motivation |
| Agent Confidence | Per-agent scatter plots over time |
| Negotiation Timeline | Disagreement intensity with intervention markers |
| Rule-Based vs IO | Side-by-side decision comparison |
| Facilitator Benchmark | Three-way comparison (facilitator vs rule-based vs IO) |
| Future Work | Ablation study, gaze-focused V4, 80-person study plan |

Password protected via Streamlit secrets.

## Data

- **18 participants** in VR escape room (Meta Quest Pro)
- **11 with complete eye tracking** + game logs → used for Inside Out
- **7 with game logs only** → used for rule-based system only
- **5,265 five-second windows** across 11 players
- Raw eye tracking at ~71Hz (PlayerTracking.csv per user)

## Quick Start

```bash
pip install -r requirements.txt

# Run the multi-agent pipeline
python src/pipeline.py

# Run the rule-based expert engine
python -c "import sys; sys.path.insert(0,'src'); from expert_from_logs import run_expert_on_all; run_expert_on_all()"

# Run facilitator benchmark
python -m src.facilitator_benchmark

# Launch the app
streamlit run app.py
```

## Key Results

### Ablation Study (±15s tolerance)

| Configuration | Eye Tracking | Game Logs | F1 |
|---|---|---|---|
| Rule-Based | No | Yes | 0.347 |
| IO — Behavioral Only | No | Yes | 0.446 |
| IO — Gaze Only | Yes | No | 0.422 |
| IO — V4 Full | Yes | Yes | 0.521 |
| IO — V3 (current) | Yes | Yes | **0.529** |
| IO — Theory-Partitioned | Yes | Yes | **0.531** |

**Key finding:** Gaze-only (0.422) and behavioral-only (0.446) are both limited. Combining them is **superadditive** (0.521) because the system detects *tensions between channels* that neither can see alone.

### Two Performance Ceilings

All architectures converge at F1 ≈ 0.52 due to:

1. **Feature granularity** — `action_count=3` could be 3 correct interactions or 3 random clicks. 68% of missed windows show the player genuinely looked fine by every metric.

2. **Ground truth noise** — Facilitator prompts mix reactive (detectable) and proactive (pedagogical, undetectable) types. Player 22 received 8 prompt episodes while performing well.

## Project Structure

```
src/
  config.py              — agent thresholds (data-derived percentiles)
  load_data.py           — CSV loading + safe_get helper
  agents.py              — 4 rule-based agents with confidence scores
  population_agent.py    — data-driven agent using K=5 cluster centroids
  negotiation.py         — pairwise tension detection + disagreement structure
  support.py             — maps disagreement patterns to adaptive responses
  pipeline.py            — main entry: load → agents → negotiate → support → save
  expert_engine.py       — Unity PromptStateMachine reimplementation
  expert_from_logs.py    — builds expert windows from raw PuzzleLogs
  facilitator_benchmark.py — three-way comparison + temporal tolerance
  compare_systems.py     — expert vs IO detailed analysis
  gaze_features.py       — extract 19 eye-tracking features from raw tracking
  gaze_action_coupling.py — classify actions as informed/blind/misguided
  ablation.py            — run all agent configurations and compare
  validate.py            — distribution checks and threshold recommendations
app.py                   — Streamlit visualization (7 tabs)
data/
  windows.csv            — 5,265 windowed features (11 players)
  windows_enhanced.csv   — above + 19 gaze features + coupling
  gaze_features.csv      — standalone gaze feature extraction
  gaze_action_coupling.csv — per-window action quality classification
outputs/
  agent_outputs.csv      — full pipeline output
  expert_all18_outputs.csv — rule-based on all 18 players
  three_way_comparison.csv — merged IO + expert + facilitator
  benchmark_report.md    — detailed benchmark analysis
  ablation_results.csv   — all configurations compared
```

## Branches

| Branch | What |
|--------|------|
| `main` | V3 data-partitioned architecture (current best) |
| `experiment/gaze-focused-agents` | V4 with 3 gaze agents + gaze-action coupling |
| `experiment/theory-partitioned-agents` | 4 cognitive theory agents (shared features) |

## Paper

**Title:** *Inside Out: Multi-Agent Negotiation of Cognitive States in Virtual Reality*

Target: CHI 2027. 18-person pilot complete. 80-person main study planned.
