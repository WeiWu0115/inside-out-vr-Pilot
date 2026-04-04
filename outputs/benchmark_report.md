# Facilitator Benchmark Report

Generated: 2026-04-04 19:38

## 1. Facilitator Prompt Overview (18 users)

- Total prompt blocks: **223** (154 reflective, 69 explicit)

### Prompts by puzzle

| Puzzle | explicit | reflective |
|--------|--------|--------|
| Hub Puzzle: Cooking Pot | 26 | 42 |
| Spoke Puzzle: Amount of Protein | 22 | 26 |
| Spoke Puzzle: Amount of Sunlight | 3 | 15 |
| Spoke Puzzle: Pasta in Sauce | 9 | 32 |
| Spoke Puzzle: Water Amount | 5 | 30 |
| none | 4 | 9 |

## 2. Facilitator Window Distribution

- **watch**: 6726 (79.2%)
- **probe**: 1690 (19.9%)
- **intervene**: 81 (1.0%)

## 3. Three-Way Comparison (11 users with IO + Expert + Facilitator)

- Overlapping participants: **11**
- Total windows: **5265**

### Decision distribution

| Category | IO | Expert | Facilitator |
|----------|-----|--------|-------------|
| watch | 3028 (57.5%) | 4930 (93.6%) | 4172 (79.2%) |
| probe | 1272 (24.2%) | 0 (0.0%) | 1027 (19.5%) |
| intervene | 965 (18.3%) | 335 (6.4%) | 66 (1.3%) |

## 4. Intervention Detection (Facilitator as Ground Truth)

Binary task: facilitator != watch → positive (someone needed help)

| Metric | IO | Expert |
|--------|-----|--------|
| precision | 0.26 | 0.227 |
| recall | 0.532 | 0.07 |
| f1 | 0.349 | 0.106 |
| accuracy | 0.588 | 0.758 |
| TP | 581 | 76 |
| FP | 1656 | 259 |
| FN | 512 | 1017 |
| TN | 2516 | 3913 |

## 5. Three-Class Evaluation (watch / probe / intervene)

Facilitator categories: watch=no prompt, probe=reflective, intervene=explicit

| Class | IO Precision | IO Recall | IO F1 | Expert Precision | Expert Recall | Expert F1 |
|-------|-------------|-----------|-------|-----------------|---------------|-----------|
| watch | 0.831 | 0.603 | 0.699 | 0.794 | 0.938 | 0.86 |
| probe | 0.211 | 0.262 | 0.234 | 0 | 0.0 | 0 |
| intervene | 0.032 | 0.47 | 0.06 | 0.006 | 0.03 | 0.01 |

- IO overall accuracy: **0.535**
- Expert overall accuracy: **0.744**

## 6. Per-Puzzle Breakdown

### Hub Puzzle: Cooking Pot
- Windows: 1602
- Facilitator intervention rate: 19.1%
- IO intervention/probe rate: 42.1%
- Expert intervention rate: 7.7%

### Spoke Puzzle: Amount of Protein
- Windows: 370
- Facilitator intervention rate: 37.3%
- IO intervention/probe rate: 46.5%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Amount of Sunlight
- Windows: 330
- Facilitator intervention rate: 24.5%
- IO intervention/probe rate: 56.7%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Pasta in Sauce
- Windows: 270
- Facilitator intervention rate: 20.0%
- IO intervention/probe rate: 47.8%
- Expert intervention rate: 9.6%

### Spoke Puzzle: Water Amount
- Windows: 472
- Facilitator intervention rate: 38.1%
- IO intervention/probe rate: 60.6%
- Expert intervention rate: 11.0%

## 7. IO Tension Patterns During Facilitator Interventions

When the facilitator gave a prompt (1093 windows), IO's tension patterns were:

- **focused_but_idle**: 311 (28.5%)
- **scattered_but_progressing**: 284 (26.0%)
- **scanning_but_passive**: 146 (13.4%)
- **focused_progress**: 109 (10.0%)
- **idle_but_progressing**: 32 (2.9%)
- **none**: 31 (2.8%)
- **scattered_and_ineffective**: 23 (2.1%)
- **pop_confirms_progress**: 23 (2.1%)
- **engaged_and_active**: 22 (2.0%)
- **passive_and_stuck**: 21 (1.9%)
- **pop_says_solving_but_stalled**: 17 (1.6%)
- **focused_but_ineffective**: 14 (1.3%)
- **pop_says_stuck_but_active**: 12 (1.1%)
- **pop_says_exploring_but_focused**: 12 (1.1%)
- **pop_confirms_exploration**: 11 (1.0%)
- **frozen_on_clue**: 7 (0.6%)
- **pop_says_stuck_but_searching**: 6 (0.5%)
- **pop_confirms_impasse**: 5 (0.5%)
- **pop_says_exploring_but_inactive**: 5 (0.5%)
- **acting_without_progress**: 2 (0.2%)

## 8. IO Decision When Facilitator Prompted

IO's decision at facilitator prompt moments:

- **watch**: 512 (46.8%)
- **intervene**: 307 (28.1%)
- **probe**: 274 (25.1%)

## 9. Expert Decision When Facilitator Prompted

Expert's decision at facilitator prompt moments:

- **watch**: 1017 (93.0%)
- **intervene**: 76 (7.0%)

## 10. Temporal Tolerance Analysis (Event-Level)

Instead of exact window matching, check if system acted within ±N seconds of each facilitator prompt.
This is fairer because prompts are continuous blocks spanning multiple windows.

| Tolerance | IO Recall | IO Precision | IO F1 | Expert Recall | Expert Precision | Expert F1 |
|-----------|-----------|-------------|-------|--------------|-----------------|-----------|
| ±0s | 0.86 | 0.226 | 0.358 | 0.4 | 0.203 | 0.269 |
| ±5s | 0.821 | 0.286 | 0.424 | 0.331 | 0.251 | 0.285 |
| ±10s | 0.887 | 0.336 | 0.488 | 0.384 | 0.287 | 0.328 |
| ±15s | 0.921 | 0.371 | 0.529 | 0.45 | 0.316 | 0.372 |
| ±20s | 0.94 | 0.407 | 0.568 | 0.47 | 0.352 | 0.403 |
| ±30s | 0.96 | 0.467 | 0.628 | 0.47 | 0.397 | 0.431 |

At **±15s tolerance** (recommended):
- IO detects **92.1%** of facilitator prompts (F1=0.529)
- Expert detects **45.0%** of facilitator prompts (F1=0.372)

## 11. Per-Prompt Detection Detail (±15s tolerance)

### Reflective prompts (97 total)
- IO detection rate: **92.8%**
- Expert detection rate: **50.5%**

### Explicit prompts (54 total)
- IO detection rate: **90.7%**
- Expert detection rate: **35.2%**

### Detection rate by puzzle (±15s)

| Puzzle | N prompts | IO hit rate | Expert hit rate |
|--------|-----------|-------------|-----------------|
| Hub Puzzle: Cooking Pot | 46 | 93.5% | 50.0% |
| Spoke Puzzle: Amount of Protein | 36 | 91.7% | 0.0% |
| Spoke Puzzle: Amount of Sunlight | 13 | 100.0% | 7.7% |
| Spoke Puzzle: Pasta in Sauce | 26 | 88.5% | 73.1% |
| Spoke Puzzle: Water Amount | 25 | 100.0% | 84.0% |
| none | 5 | 40.0% | 80.0% |

### Detection rate by participant (±15s)

| Participant | N prompts | IO hit rate | Expert hit rate |
|-------------|-----------|-------------|-----------------|
| User-1 | 2 | 100.0% | 0.0% |
| User-2 | 6 | 100.0% | 50.0% |
| User-3 | 13 | 84.6% | 53.8% |
| User-5 | 13 | 92.3% | 61.5% |
| User-6 | 31 | 96.8% | 35.5% |
| User-9 | 25 | 96.0% | 60.0% |
| User-10 | 22 | 100.0% | 36.4% |
| User-12 | 9 | 77.8% | 44.4% |
| User-14 | 7 | 100.0% | 57.1% |
| User-22 | 8 | 62.5% | 12.5% |
| User-23 | 15 | 86.7% | 46.7% |

## 12. Episode-Level Evaluation

Struggle episodes: consecutive facilitator prompts grouped within 60s.

- Total episodes: **80**
- IO episode recall: **85.0%**
- Expert episode recall: **45.0%**
- Mean episode duration: **103s**
- Mean IO detection latency: **-5.2s** (negative = early detection)
- IO detected **before** episode start: 53/68 (78%)

### Severe (has explicit prompt) (28 episodes)
- IO recall: **85.7%**
- Expert recall: **39.3%**

### Mild (reflective only) (52 episodes)
- IO recall: **84.6%**
- Expert recall: **48.1%**