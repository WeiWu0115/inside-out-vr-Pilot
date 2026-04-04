# Facilitator Benchmark Report

Generated: 2026-04-04 17:27

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
| watch | 3648 (69.3%) | 4930 (93.6%) | 4172 (79.2%) |
| probe | 1265 (24.0%) | 0 (0.0%) | 1027 (19.5%) |
| intervene | 352 (6.7%) | 335 (6.4%) | 66 (1.3%) |

## 4. Intervention Detection (Facilitator as Ground Truth)

Binary task: facilitator != watch → positive (someone needed help)

| Metric | IO | Expert |
|--------|-----|--------|
| precision | 0.257 | 0.227 |
| recall | 0.381 | 0.07 |
| f1 | 0.307 | 0.106 |
| accuracy | 0.643 | 0.758 |
| TP | 416 | 76 |
| FP | 1201 | 259 |
| FN | 677 | 1017 |
| TN | 2971 | 3913 |

## 5. Three-Class Evaluation (watch / probe / intervene)

Facilitator categories: watch=no prompt, probe=reflective, intervene=explicit

| Class | IO Precision | IO Recall | IO F1 | Expert Precision | Expert Recall | Expert F1 |
|-------|-------------|-----------|-------|-----------------|---------------|-----------|
| watch | 0.814 | 0.712 | 0.76 | 0.794 | 0.938 | 0.86 |
| probe | 0.238 | 0.293 | 0.263 | 0 | 0.0 | 0 |
| intervene | 0.009 | 0.045 | 0.014 | 0.006 | 0.03 | 0.01 |

- IO overall accuracy: **0.622**
- Expert overall accuracy: **0.744**

## 6. Per-Puzzle Breakdown

### Hub Puzzle: Cooking Pot
- Windows: 1602
- Facilitator intervention rate: 19.1%
- IO intervention/probe rate: 45.1%
- Expert intervention rate: 7.7%

### Spoke Puzzle: Amount of Protein
- Windows: 370
- Facilitator intervention rate: 37.3%
- IO intervention/probe rate: 53.5%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Amount of Sunlight
- Windows: 330
- Facilitator intervention rate: 24.5%
- IO intervention/probe rate: 53.9%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Pasta in Sauce
- Windows: 270
- Facilitator intervention rate: 20.0%
- IO intervention/probe rate: 58.5%
- Expert intervention rate: 9.6%

### Spoke Puzzle: Water Amount
- Windows: 472
- Facilitator intervention rate: 38.1%
- IO intervention/probe rate: 56.8%
- Expert intervention rate: 11.0%

## 7. IO Tension Patterns During Facilitator Interventions

When the facilitator gave a prompt (1093 windows), IO's tension patterns were:

- **scattered_but_progressing**: 260 (23.8%)
- **focused_but_idle**: 234 (21.4%)
- **focused_progress**: 167 (15.3%)
- **scanning_but_passive**: 152 (13.9%)
- **passive_and_stuck**: 85 (7.8%)
- **none**: 57 (5.2%)
- **pop_confirms_exploration**: 49 (4.5%)
- **pop_confirms_progress**: 21 (1.9%)
- **acting_without_progress**: 14 (1.3%)
- **pop_says_solving_but_stalled**: 12 (1.1%)
- **pop_says_exploring_but_focused**: 10 (0.9%)
- **focused_but_failing**: 7 (0.6%)
- **engaged_and_active**: 7 (0.6%)
- **pop_says_exploring_but_inactive**: 7 (0.6%)
- **pop_says_stuck_but_active**: 4 (0.4%)
- **pop_says_stuck_but_searching**: 4 (0.4%)
- **active_but_failing**: 2 (0.2%)
- **frozen_on_clue**: 1 (0.1%)

## 8. IO Decision When Facilitator Prompted

IO's decision at facilitator prompt moments:

- **watch**: 677 (61.9%)
- **probe**: 321 (29.4%)
- **intervene**: 95 (8.7%)

## 9. Expert Decision When Facilitator Prompted

Expert's decision at facilitator prompt moments:

- **watch**: 1017 (93.0%)
- **intervene**: 76 (7.0%)

## 10. Temporal Tolerance Analysis (Event-Level)

Instead of exact window matching, check if system acted within ±N seconds of each facilitator prompt.
This is fairer because prompts are continuous blocks spanning multiple windows.

| Tolerance | IO Recall | IO Precision | IO F1 | Expert Recall | Expert Precision | Expert F1 |
|-----------|-----------|-------------|-------|--------------|-----------------|-----------|
| ±0s | 0.74 | 0.226 | 0.346 | 0.4 | 0.203 | 0.269 |
| ±5s | 0.689 | 0.281 | 0.4 | 0.331 | 0.251 | 0.285 |
| ±10s | 0.795 | 0.335 | 0.472 | 0.384 | 0.287 | 0.328 |
| ±15s | 0.795 | 0.373 | 0.508 | 0.45 | 0.316 | 0.372 |
| ±20s | 0.815 | 0.408 | 0.543 | 0.47 | 0.352 | 0.403 |
| ±30s | 0.854 | 0.472 | 0.608 | 0.47 | 0.397 | 0.431 |

At **±15s tolerance** (recommended):
- IO detects **79.5%** of facilitator prompts (F1=0.508)
- Expert detects **45.0%** of facilitator prompts (F1=0.372)

## 11. Per-Prompt Detection Detail (±15s tolerance)

### Reflective prompts (97 total)
- IO detection rate: **82.5%**
- Expert detection rate: **50.5%**

### Explicit prompts (54 total)
- IO detection rate: **74.1%**
- Expert detection rate: **35.2%**

### Detection rate by puzzle (±15s)

| Puzzle | N prompts | IO hit rate | Expert hit rate |
|--------|-----------|-------------|-----------------|
| Hub Puzzle: Cooking Pot | 46 | 87.0% | 50.0% |
| Spoke Puzzle: Amount of Protein | 36 | 77.8% | 0.0% |
| Spoke Puzzle: Amount of Sunlight | 13 | 61.5% | 7.7% |
| Spoke Puzzle: Pasta in Sauce | 26 | 73.1% | 73.1% |
| Spoke Puzzle: Water Amount | 25 | 96.0% | 84.0% |
| none | 5 | 20.0% | 80.0% |

### Detection rate by participant (±15s)

| Participant | N prompts | IO hit rate | Expert hit rate |
|-------------|-----------|-------------|-----------------|
| User-1 | 2 | 100.0% | 0.0% |
| User-2 | 6 | 66.7% | 50.0% |
| User-3 | 13 | 76.9% | 53.8% |
| User-5 | 13 | 84.6% | 61.5% |
| User-6 | 31 | 96.8% | 35.5% |
| User-9 | 25 | 76.0% | 60.0% |
| User-10 | 22 | 77.3% | 36.4% |
| User-12 | 9 | 44.4% | 44.4% |
| User-14 | 7 | 85.7% | 57.1% |
| User-22 | 8 | 87.5% | 12.5% |
| User-23 | 15 | 66.7% | 46.7% |