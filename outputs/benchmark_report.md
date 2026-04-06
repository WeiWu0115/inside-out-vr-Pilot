# Facilitator Benchmark Report

Generated: 2026-04-06 13:34

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
| watch | 3112 (59.1%) | 5016 (95.3%) | 4172 (79.2%) |
| probe | 1471 (27.9%) | 220 (4.2%) | 1027 (19.5%) |
| intervene | 682 (13.0%) | 29 (0.6%) | 66 (1.3%) |

## 4. Intervention Detection (Facilitator as Ground Truth)

Binary task: facilitator != watch → positive (someone needed help)

| Metric | IO | Expert |
|--------|-----|--------|
| precision | 0.25 | 0.225 |
| recall | 0.493 | 0.051 |
| f1 | 0.332 | 0.083 |
| accuracy | 0.588 | 0.766 |
| TP | 539 | 56 |
| FP | 1614 | 193 |
| FN | 554 | 1037 |
| TN | 2558 | 3979 |

## 5. Three-Class Evaluation (watch / probe / intervene)

Facilitator categories: watch=no prompt, probe=reflective, intervene=explicit

| Class | IO Precision | IO Recall | IO F1 | Expert Precision | Expert Recall | Expert F1 |
|-------|-------------|-----------|-------|-----------------|---------------|-----------|
| watch | 0.822 | 0.613 | 0.702 | 0.793 | 0.954 | 0.866 |
| probe | 0.23 | 0.33 | 0.271 | 0.232 | 0.05 | 0.082 |
| intervene | 0.022 | 0.227 | 0.04 | 0.034 | 0.015 | 0.021 |

- IO overall accuracy: **0.553**
- Expert overall accuracy: **0.766**

## 6. Per-Puzzle Breakdown

### Hub Puzzle: Cooking Pot
- Windows: 1602
- Facilitator intervention rate: 19.1%
- IO intervention/probe rate: 57.9%
- Expert intervention rate: 4.6%

### Spoke Puzzle: Amount of Protein
- Windows: 370
- Facilitator intervention rate: 37.3%
- IO intervention/probe rate: 57.3%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Amount of Sunlight
- Windows: 330
- Facilitator intervention rate: 24.5%
- IO intervention/probe rate: 54.8%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Pasta in Sauce
- Windows: 270
- Facilitator intervention rate: 20.0%
- IO intervention/probe rate: 61.1%
- Expert intervention rate: 6.3%

### Spoke Puzzle: Water Amount
- Windows: 472
- Facilitator intervention rate: 38.1%
- IO intervention/probe rate: 51.9%
- Expert intervention rate: 7.6%

## 7. IO Tension Patterns During Facilitator Interventions

When the facilitator gave a prompt (1093 windows), IO's tension patterns were:

- **watching_task_but_idle**: 166 (15.2%)
- **acting_while_looking_away**: 160 (14.6%)
- **focused_but_idle**: 142 (13.0%)
- **uncertain_checking**: 129 (11.8%)
- **none**: 84 (7.7%)
- **lost_and_passive**: 78 (7.1%)
- **frozen**: 60 (5.5%)
- **purposeful_scanning**: 60 (5.5%)
- **scanning_but_passive**: 50 (4.6%)
- **purposeful_task_gaze**: 47 (4.3%)
- **systematic_search**: 23 (2.1%)
- **concentrated_motor_scanning_fixation**: 15 (1.4%)
- **purposeful_action**: 14 (1.3%)
- **locked_on_environment**: 11 (1.0%)
- **acting_without_looking**: 10 (0.9%)
- **task_engaged**: 9 (0.8%)
- **reading_but_not_acting**: 8 (0.7%)
- **locked_gaze_but_active**: 7 (0.6%)
- **deep_clue_reading**: 6 (0.5%)
- **engaged_and_active**: 5 (0.5%)
- **frozen_on_clue**: 3 (0.3%)
- **concentrated_but_failing**: 2 (0.2%)
- **concentrated_but_off_task**: 2 (0.2%)
- **disengaged**: 2 (0.2%)

## 8. IO Decision When Facilitator Prompted

IO's decision at facilitator prompt moments:

- **watch**: 554 (50.7%)
- **probe**: 358 (32.8%)
- **intervene**: 181 (16.6%)

## 9. Expert Decision When Facilitator Prompted

Expert's decision at facilitator prompt moments:

- **watch**: 1037 (94.9%)
- **probe**: 54 (4.9%)
- **intervene**: 2 (0.2%)

## 10. Temporal Tolerance Analysis (Event-Level)

Instead of exact window matching, check if system acted within ±N seconds of each facilitator prompt.
This is fairer because prompts are continuous blocks spanning multiple windows.

| Tolerance | IO Recall | IO Precision | IO F1 | Expert Recall | Expert Precision | Expert F1 |
|-----------|-----------|-------------|-------|--------------|-----------------|-----------|
| ±0s | 0.82 | 0.218 | 0.345 | 0.3 | 0.189 | 0.232 |
| ±5s | 0.808 | 0.277 | 0.413 | 0.265 | 0.241 | 0.252 |
| ±10s | 0.881 | 0.324 | 0.473 | 0.338 | 0.293 | 0.314 |
| ±15s | 0.907 | 0.362 | 0.517 | 0.377 | 0.321 | 0.347 |
| ±20s | 0.934 | 0.397 | 0.557 | 0.397 | 0.341 | 0.367 |
| ±30s | 0.96 | 0.459 | 0.621 | 0.417 | 0.386 | 0.401 |

At **±15s tolerance** (recommended):
- IO detects **90.7%** of facilitator prompts (F1=0.517)
- Expert detects **37.7%** of facilitator prompts (F1=0.347)

## 11. Per-Prompt Detection Detail (±15s tolerance)

### Reflective prompts (97 total)
- IO detection rate: **90.7%**
- Expert detection rate: **45.4%**

### Explicit prompts (54 total)
- IO detection rate: **90.7%**
- Expert detection rate: **24.1%**

### Detection rate by puzzle (±15s)

| Puzzle | N prompts | IO hit rate | Expert hit rate |
|--------|-----------|-------------|-----------------|
| Hub Puzzle: Cooking Pot | 46 | 95.7% | 32.6% |
| Spoke Puzzle: Amount of Protein | 36 | 91.7% | 0.0% |
| Spoke Puzzle: Amount of Sunlight | 13 | 69.2% | 7.7% |
| Spoke Puzzle: Pasta in Sauce | 26 | 92.3% | 73.1% |
| Spoke Puzzle: Water Amount | 25 | 100.0% | 72.0% |
| none | 5 | 40.0% | 80.0% |

### Detection rate by participant (±15s)

| Participant | N prompts | IO hit rate | Expert hit rate |
|-------------|-----------|-------------|-----------------|
| User-1 | 2 | 100.0% | 0.0% |
| User-2 | 6 | 100.0% | 16.7% |
| User-3 | 13 | 84.6% | 46.2% |
| User-5 | 13 | 92.3% | 61.5% |
| User-6 | 31 | 96.8% | 22.6% |
| User-9 | 25 | 84.0% | 52.0% |
| User-10 | 22 | 81.8% | 31.8% |
| User-12 | 9 | 88.9% | 44.4% |
| User-14 | 7 | 100.0% | 57.1% |
| User-22 | 8 | 100.0% | 12.5% |
| User-23 | 15 | 93.3% | 40.0% |

## 12. Episode-Level Evaluation

Struggle episodes: consecutive facilitator prompts grouped within 60s.

- Total episodes: **80**
- IO episode recall: **86.2%**
- Expert episode recall: **37.5%**
- Mean episode duration: **103s**
- Mean IO detection latency: **-2.4s** (negative = early detection)
- IO detected **before** episode start: 54/69 (78%)

### Severe (has explicit prompt) (28 episodes)
- IO recall: **89.3%**
- Expert recall: **28.6%**

### Mild (reflective only) (52 episodes)
- IO recall: **84.6%**
- Expert recall: **42.3%**