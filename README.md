# Multi-Agent VR User-State Interpretation Pipeline

A local, rule-based prototype for interpreting VR user states through multi-agent negotiation. Built for a CHI paper exploring the idea that adaptive support should emerge from disagreement among interpretive agents rather than from a single classifier.

## Quick Start

```bash
pip install -r requirements.txt
python src/pipeline.py
```

Or with custom paths:

```bash
python src/pipeline.py --input data/windows.csv --output outputs/
```

## Expected Input

A CSV file (`data/windows.csv` by default) where each row is a 5-second window from a VR escape room session.

**Required columns:**
- `participant_id` — participant identifier
- `puzzle_id` — which puzzle the window belongs to
- `window_start` — timestamp or index for the window

**Optional columns (used by agents):**
- `gaze_entropy` — entropy of gaze distribution (0-1 scale)
- `clue_ratio` — proportion of gaze on clue-relevant areas (0-1)
- `switch_rate` — rate of gaze target switches (0-1)
- `action_count` — number of actions in the window
- `idle_time` — seconds of inactivity
- `time_since_action` — seconds since last action
- `error_count` — number of errors in the window
- `puzzle_active` — boolean, whether puzzle is currently active

Missing optional columns are handled gracefully — agents output `unknown` for states they cannot determine.

## Agents

| Agent | Features Used | Possible States |
|-------|--------------|-----------------|
| **AttentionAgent** | gaze_entropy, clue_ratio, switch_rate | focused, searching, locked, unknown |
| **ActionAgent** | action_count, idle_time, time_since_action | active, hesitant, inactive, unknown |
| **PerformanceAgent** | error_count, puzzle_active, action_count | progressing, failing, stalled, unknown |
| **TemporalAgent** | consecutive window patterns | transient, persistent, looping, unknown |

## Negotiation Layer

After agents run independently, a negotiation module detects disagreement patterns:

| Pattern | Conditions | Suggested Support |
|---------|-----------|-------------------|
| `focused_but_stuck` | focused + inactive/hesitant + stalled/failing | procedural_hint |
| `searching_without_grounding` | searching + inactive + stalled | spatial_hint |
| `active_but_unguided` | searching + active + failing | light_guidance |
| `productive_struggle` | focused + active/hesitant + stalled + transient | wait |
| `no_clear_pattern` | no rule matches | none |

The `disagreement_score` counts how many distinct agent states are present (0-4).

## Output Files

| File | Contents |
|------|----------|
| `outputs/agent_outputs.csv` | All original columns + agent states + disagreement + support |
| `outputs/disagreement_summary.csv` | Agent states, disagreement score, and pattern per window |
| `outputs/support_summary.csv` | Disagreement pattern and suggested support per window |

## Visualization App

Interactive Streamlit app for exploring agent interpretations and intervention timing:

```bash
pip install streamlit plotly
streamlit run app.py
```

Features:
- **Agent Timeline** — 4 agents' interpretations over time + support suggestions
- **Disagreement & Interventions** — disagreement score timeline with intervention markers
- **Cluster vs Agent Analysis** — K-means cluster vs agent disagreement pattern heatmap
- **Playback** — step-by-step scrubbing through each time window

## Configuration

All thresholds are in `src/config.py` — edit them to tune agent sensitivity.

## Project Structure

```
src/
  config.py        — thresholds and file paths
  load_data.py     — CSV loading and validation
  agents.py        — four interpretive agents
  negotiation.py   — disagreement scoring and pattern matching
  support.py       — pattern-to-intervention mapping
  pipeline.py      — main entry point
```
