# Inside Out V2: Clean Agent Boundaries Architecture

## Problem with V1

In V1, feature overlap across agents creates **echo consensus** — multiple agents agree not because they see different signals, but because they read the same `action_count`. This inflates confidence in the negotiation layer without adding information.

| Feature | V1 Agents Using It |
|---------|-------------------|
| action_count | Attention, Action, Performance, Population |
| time_since_action | Action, Performance, Population |
| gaze_entropy | Attention, Population |
| switch_rate | Attention, Population |
| idle_time | Action, Population |

## V2 Design: One Agent, One Perceptual Dimension

Each agent owns an **exclusive** set of raw features. No feature is read by more than one agent.

### Agent Definitions

| Agent | Question | Exclusive Features | Output Labels |
|-------|----------|-------------------|---------------|
| **Perceptual** | "What is the player looking at?" | gaze_entropy, clue_ratio, switch_rate | focused / searching / locked |
| **Behavioral** | "What is the player doing?" | action_count, idle_time, error_count | active / hesitant / inactive |
| **Spatial** | "Where is the player going?" | head_position, area_id, movement_speed, distance_to_puzzle | navigating / stationed / wandering |
| **Progress** | "How is puzzle-solving going?" | puzzle_elapsed_ratio, time_since_action, puzzle_state, steps_completed | progressing / stalled / failing |
| **Temporal** | "How long has this been going on?" | labels from above agents (sliding window) | transient / persistent / looping |

### Key Changes from V1

1. **`action_count` belongs exclusively to Behavioral.** Performance Agent (now Progress Agent) no longer reads it — it uses `time_since_action` and `puzzle_elapsed_ratio` instead.

2. **`time_since_action` moves from Action to Progress.** It's a macro-temporal signal about puzzle engagement, not instantaneous behavior.

3. **New Spatial Agent** uses head tracking data (already available in Jupyter Analysis Path/*.csv). Solves the Transition-phase blind spot.

4. **Population Agent merged into Progress.** The K-means cluster centroids encoded a mix of signals. V2 replaces this with explicit `puzzle_elapsed_ratio` and `puzzle_state`.

### Data Flow

```
Unity VR Runtime
    │
    ├── Eye Tracker ──────→ Perceptual Agent
    │                        (gaze_entropy, clue_ratio, switch_rate)
    │
    ├── Controller/Hand ──→ Behavioral Agent  
    │                        (action_count, idle_time, error_count)
    │
    ├── Head Tracker ─────→ Spatial Agent
    │                        (head_position, area_id, movement_speed)
    │
    ├── Game Engine ──────→ Progress Agent
    │                        (puzzle_elapsed, time_since_action, puzzle_state)
    │
    └── All of above ─────→ Temporal Agent
                             (reads agent labels over sliding window)
                                    │
                                    ▼
                            Negotiation Layer
                            (pairwise tensions)
                                    │
                                    ▼
                            Prompt Agent
                            (escalation + cooldown + fatigue)
```

### Negotiation Changes

V1 tension pairs need updating for new agent names:

| V1 Tension | V2 Equivalent |
|-----------|---------------|
| scanning_but_passive (Attention × Action) | searching_but_inactive (Perceptual × Behavioral) |
| focused_but_failing (Attention × Performance) | focused_but_failing (Perceptual × Progress) |
| scattered_but_progressing (Attention × Performance) | searching_but_progressing (Perceptual × Progress) |
| focused_but_idle (Attention × Action) | focused_but_inactive (Perceptual × Behavioral) |
| NEW | stationed_but_stalled (Spatial × Progress) |
| NEW | wandering_but_active (Spatial × Behavioral) |
| NEW | navigating_but_stuck (Spatial × Progress) |

### Prompt Agent Enhancements (Stateful)

V1's support layer is stateless (each window decided independently). V2 adds:

```python
class PromptAgent:
    def __init__(self):
        self.last_prompt_time = None    # cooldown tracking
        self.prompt_count = 0           # fatigue tracking  
        self.escalation_stage = 0       # 0=none, 1=reflective, 2=explicit
    
    def decide(self, negotiation_result, timestamp):
        # Cooldown: don't prompt within 20s of last prompt
        # Escalation: if reflective didn't help after 30s, try explicit
        # Fatigue: reduce frequency after 5+ prompts
```

## Migration Plan

1. Rename existing agents (Attention→Perceptual, Action→Behavioral)
2. Remove feature overlap (take action_count out of Performance)
3. Create Progress Agent (from Performance + puzzle_elapsed + time_since_action)
4. Create Spatial Agent (stub — uses puzzle_id area heuristic until head tracking is wired)
5. Update negotiation tension pairs
6. Add stateful Prompt Agent
7. Re-run pipeline + benchmark, compare V1 vs V2

## Evaluation

Both versions evaluated against the same facilitator benchmark (±15s tolerance):

| Metric | V1 (main) | V2 (this branch) |
|--------|-----------|-------------------|
| F1 | **0.514** | 0.503 |
| Recall | **86.1%** | 73.5% |
| Precision | 36.6% | **38.2%** |

### Finding: Complete Feature Isolation Hurts Recall

Removing `action_count` from Progress Agent caused recall to drop 12.6pp. The Progress Agent without action_count over-predicts "progressing" (468/659 FN windows) because short `time_since_action` alone is insufficient — a player may have done one random click 5s ago but still be stuck.

**Precision improved** because eliminating the echo-consensus effect means when multiple agents *do* agree, it's genuine agreement from independent signals.

### Recommended Approach for 80-Person Study

A hybrid: keep feature boundaries as clean as possible, but allow **one-directional information flow** — Progress Agent can receive a "behavioral_summary" (active/inactive/hesitant) from Behavioral Agent as a pre-computed signal rather than reading raw `action_count` directly. This preserves the architectural benefit (each agent interprets independently) while acknowledging that progress assessment inherently requires knowing *whether the player is acting*.

```
Behavioral Agent (raw features) → behavioral_label → Progress Agent (uses label, not raw)
```

This is different from V1's echo problem because the Progress Agent receives a *pre-interpreted* signal, not the same raw number.
