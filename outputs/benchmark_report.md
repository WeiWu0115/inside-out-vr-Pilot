# Facilitator Benchmark Report

Generated: 2026-04-04 23:31

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
| watch | 3337 (63.4%) | 4930 (93.6%) | 4172 (79.2%) |
| probe | 1239 (23.5%) | 0 (0.0%) | 1027 (19.5%) |
| intervene | 689 (13.1%) | 335 (6.4%) | 66 (1.3%) |

## 4. Intervention Detection (Facilitator as Ground Truth)

Binary task: facilitator != watch → positive (someone needed help)

| Metric | IO | Expert |
|--------|-----|--------|
| precision | 0.269 | 0.227 |
| recall | 0.475 | 0.07 |
| f1 | 0.344 | 0.106 |
| accuracy | 0.623 | 0.758 |
| TP | 519 | 76 |
| FP | 1409 | 259 |
| FN | 574 | 1017 |
| TN | 2763 | 3913 |

## 5. Three-Class Evaluation (watch / probe / intervene)

Facilitator categories: watch=no prompt, probe=reflective, intervene=explicit

| Class | IO Precision | IO Recall | IO F1 | Expert Precision | Expert Recall | Expert F1 |
|-------|-------------|-----------|-------|-----------------|---------------|-----------|
| watch | 0.828 | 0.662 | 0.736 | 0.794 | 0.938 | 0.86 |
| probe | 0.243 | 0.293 | 0.266 | 0 | 0.0 | 0 |
| intervene | 0.017 | 0.182 | 0.032 | 0.006 | 0.03 | 0.01 |

- IO overall accuracy: **0.584**
- Expert overall accuracy: **0.744**

## 6. Per-Puzzle Breakdown

### Hub Puzzle: Cooking Pot
- Windows: 1602
- Facilitator intervention rate: 19.1%
- IO intervention/probe rate: 29.3%
- Expert intervention rate: 7.7%

### Spoke Puzzle: Amount of Protein
- Windows: 370
- Facilitator intervention rate: 37.3%
- IO intervention/probe rate: 41.1%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Amount of Sunlight
- Windows: 330
- Facilitator intervention rate: 24.5%
- IO intervention/probe rate: 40.9%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Pasta in Sauce
- Windows: 270
- Facilitator intervention rate: 20.0%
- IO intervention/probe rate: 45.9%
- Expert intervention rate: 9.6%

### Spoke Puzzle: Water Amount
- Windows: 472
- Facilitator intervention rate: 38.1%
- IO intervention/probe rate: 54.9%
- Expert intervention rate: 11.0%

## 7. IO Tension Patterns During Facilitator Interventions

When the facilitator gave a prompt (1093 windows), IO's tension patterns were:

- **regulated_flow**: 276 (25.3%)
- **pop_stuck_but_regulated**: 172 (15.7%)
- **pop_exploring_but_engaged**: 108 (9.9%)
- **attending_but_disengaged**: 89 (8.1%)
- **engaged_and_regulated**: 76 (7.0%)
- **overloaded_but_regulated**: 76 (7.0%)
- **disengaged_from_flow**: 69 (6.3%)
- **decoupled_in_flow**: 36 (3.3%)
- **decoupled_but_regulated**: 35 (3.2%)
- **engaged_but_anxious**: 26 (2.4%)
- **engaged_in_flow**: 20 (1.8%)
- **overloaded_in_flow**: 19 (1.7%)
- **regulated_but_anxious**: 19 (1.7%)
- **engaged_but_impulsive**: 15 (1.4%)
- **pop_confirms_flow**: 11 (1.0%)
- **none**: 10 (0.9%)
- **pop_solving_but_anxious**: 10 (0.9%)
- **engaged_and_reflective**: 8 (0.7%)
- **disengaged_with_anxiety**: 6 (0.5%)
- **impulsive_in_flow**: 5 (0.5%)
- **pop_confirms_impasse**: 2 (0.2%)
- **fixated_and_disengaged**: 2 (0.2%)
- **engaged_but_frustrated**: 1 (0.1%)
- **overloaded_with_anxiety**: 1 (0.1%)
- **regulated_but_frustrated**: 1 (0.1%)

## 8. IO Decision When Facilitator Prompted

IO's decision at facilitator prompt moments:

- **watch**: 574 (52.5%)
- **probe**: 315 (28.8%)
- **intervene**: 204 (18.7%)

## 9. Expert Decision When Facilitator Prompted

Expert's decision at facilitator prompt moments:

- **watch**: 1017 (93.0%)
- **intervene**: 76 (7.0%)

## 10. Temporal Tolerance Analysis (Event-Level)

Instead of exact window matching, check if system acted within ±N seconds of each facilitator prompt.
This is fairer because prompts are continuous blocks spanning multiple windows.

| Tolerance | IO Recall | IO Precision | IO F1 | Expert Recall | Expert Precision | Expert F1 |
|-----------|-----------|-------------|-------|--------------|-----------------|-----------|
| ±0s | 0.84 | 0.236 | 0.368 | 0.4 | 0.203 | 0.269 |
| ±5s | 0.762 | 0.293 | 0.423 | 0.331 | 0.251 | 0.285 |
| ±10s | 0.848 | 0.343 | 0.488 | 0.384 | 0.287 | 0.328 |
| ±15s | 0.874 | 0.375 | 0.525 | 0.45 | 0.316 | 0.372 |
| ±20s | 0.901 | 0.409 | 0.563 | 0.47 | 0.352 | 0.403 |
| ±30s | 0.94 | 0.466 | 0.623 | 0.47 | 0.397 | 0.431 |

At **±15s tolerance** (recommended):
- IO detects **87.4%** of facilitator prompts (F1=0.525)
- Expert detects **45.0%** of facilitator prompts (F1=0.372)

## 11. Per-Prompt Detection Detail (±15s tolerance)

### Reflective prompts (97 total)
- IO detection rate: **91.8%**
- Expert detection rate: **50.5%**

### Explicit prompts (54 total)
- IO detection rate: **79.6%**
- Expert detection rate: **35.2%**

### Detection rate by puzzle (±15s)

| Puzzle | N prompts | IO hit rate | Expert hit rate |
|--------|-----------|-------------|-----------------|
| Hub Puzzle: Cooking Pot | 46 | 76.1% | 50.0% |
| Spoke Puzzle: Amount of Protein | 36 | 91.7% | 0.0% |
| Spoke Puzzle: Amount of Sunlight | 13 | 100.0% | 7.7% |
| Spoke Puzzle: Pasta in Sauce | 26 | 92.3% | 73.1% |
| Spoke Puzzle: Water Amount | 25 | 100.0% | 84.0% |
| none | 5 | 40.0% | 80.0% |

### Detection rate by participant (±15s)

| Participant | N prompts | IO hit rate | Expert hit rate |
|-------------|-----------|-------------|-----------------|
| User-1 | 2 | 100.0% | 0.0% |
| User-2 | 6 | 83.3% | 50.0% |
| User-3 | 13 | 84.6% | 53.8% |
| User-5 | 13 | 92.3% | 61.5% |
| User-6 | 31 | 96.8% | 35.5% |
| User-9 | 25 | 92.0% | 60.0% |
| User-10 | 22 | 86.4% | 36.4% |
| User-12 | 9 | 77.8% | 44.4% |
| User-14 | 7 | 100.0% | 57.1% |
| User-22 | 8 | 37.5% | 12.5% |
| User-23 | 15 | 86.7% | 46.7% |

## 12. Episode-Level Evaluation

Struggle episodes: consecutive facilitator prompts grouped within 60s.

- Total episodes: **80**
- IO episode recall: **80.0%**
- Expert episode recall: **45.0%**
- Mean episode duration: **103s**
- Mean IO detection latency: **-4.5s** (negative = early detection)
- IO detected **before** episode start: 50/64 (78%)

### Severe (has explicit prompt) (28 episodes)
- IO recall: **75.0%**
- Expert recall: **39.3%**

### Mild (reflective only) (52 episodes)
- IO recall: **82.7%**
- Expert recall: **48.1%**