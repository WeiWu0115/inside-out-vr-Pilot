# Facilitator Benchmark Report

Generated: 2026-04-04 18:01

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
| watch | 3604 (68.5%) | 4930 (93.6%) | 4172 (79.2%) |
| probe | 1436 (27.3%) | 0 (0.0%) | 1027 (19.5%) |
| intervene | 225 (4.3%) | 335 (6.4%) | 66 (1.3%) |

## 4. Intervention Detection (Facilitator as Ground Truth)

Binary task: facilitator != watch → positive (someone needed help)

| Metric | IO | Expert |
|--------|-----|--------|
| precision | 0.261 | 0.227 |
| recall | 0.397 | 0.07 |
| f1 | 0.315 | 0.106 |
| accuracy | 0.642 | 0.758 |
| TP | 434 | 76 |
| FP | 1227 | 259 |
| FN | 659 | 1017 |
| TN | 2945 | 3913 |

## 5. Three-Class Evaluation (watch / probe / intervene)

Facilitator categories: watch=no prompt, probe=reflective, intervene=explicit

| Class | IO Precision | IO Recall | IO F1 | Expert Precision | Expert Recall | Expert F1 |
|-------|-------------|-----------|-------|-----------------|---------------|-----------|
| watch | 0.817 | 0.706 | 0.757 | 0.794 | 0.938 | 0.86 |
| probe | 0.233 | 0.326 | 0.272 | 0 | 0.0 | 0 |
| intervene | 0.053 | 0.182 | 0.082 | 0.006 | 0.03 | 0.01 |

- IO overall accuracy: **0.625**
- Expert overall accuracy: **0.744**

## 6. Per-Puzzle Breakdown

### Hub Puzzle: Cooking Pot
- Windows: 1602
- Facilitator intervention rate: 19.1%
- IO intervention/probe rate: 38.9%
- Expert intervention rate: 7.7%

### Spoke Puzzle: Amount of Protein
- Windows: 370
- Facilitator intervention rate: 37.3%
- IO intervention/probe rate: 40.3%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Amount of Sunlight
- Windows: 330
- Facilitator intervention rate: 24.5%
- IO intervention/probe rate: 47.0%
- Expert intervention rate: 0.0%

### Spoke Puzzle: Pasta in Sauce
- Windows: 270
- Facilitator intervention rate: 20.0%
- IO intervention/probe rate: 40.7%
- Expert intervention rate: 9.6%

### Spoke Puzzle: Water Amount
- Windows: 472
- Facilitator intervention rate: 38.1%
- IO intervention/probe rate: 51.3%
- Expert intervention rate: 11.0%

## 7. IO Tension Patterns During Facilitator Interventions

When the facilitator gave a prompt (1093 windows), IO's tension patterns were:

- **scattered_but_progressing**: 254 (23.2%)
- **idle_but_progressing**: 180 (16.5%)
- **focused_but_idle**: 127 (11.6%)
- **navigating_but_idle**: 108 (9.9%)
- **navigating_but_stalled**: 100 (9.1%)
- **stationed_and_progressing**: 90 (8.2%)
- **scanning_but_passive**: 58 (5.3%)
- **focused_progress**: 34 (3.1%)
- **acting_without_progress**: 30 (2.7%)
- **none**: 19 (1.7%)
- **stationed_and_struggling**: 12 (1.1%)
- **pop_confirms_exploration**: 11 (1.0%)
- **passive_and_stuck**: 10 (0.9%)
- **active_but_struggling**: 10 (0.9%)
- **pop_confirms_progress**: 9 (0.8%)
- **engaged_and_active**: 9 (0.8%)
- **pop_says_exploring_but_focused**: 8 (0.7%)
- **pop_says_stuck_but_active**: 7 (0.6%)
- **pop_says_solving_but_stalled**: 5 (0.5%)
- **pop_says_stuck_but_searching**: 5 (0.5%)
- **frozen_on_clue**: 4 (0.4%)
- **pop_confirms_impasse**: 2 (0.2%)
- **pop_says_exploring_but_inactive**: 1 (0.1%)

## 8. IO Decision When Facilitator Prompted

IO's decision at facilitator prompt moments:

- **watch**: 659 (60.3%)
- **probe**: 358 (32.8%)
- **intervene**: 76 (7.0%)

## 9. Expert Decision When Facilitator Prompted

Expert's decision at facilitator prompt moments:

- **watch**: 1017 (93.0%)
- **intervene**: 76 (7.0%)

## 10. Temporal Tolerance Analysis (Event-Level)

Instead of exact window matching, check if system acted within ±N seconds of each facilitator prompt.
This is fairer because prompts are continuous blocks spanning multiple windows.

| Tolerance | IO Recall | IO Precision | IO F1 | Expert Recall | Expert Precision | Expert F1 |
|-----------|-----------|-------------|-------|--------------|-----------------|-----------|
| ±0s | 0.7 | 0.228 | 0.343 | 0.4 | 0.203 | 0.269 |
| ±5s | 0.669 | 0.294 | 0.409 | 0.331 | 0.251 | 0.285 |
| ±10s | 0.702 | 0.346 | 0.464 | 0.384 | 0.287 | 0.328 |
| ±15s | 0.735 | 0.382 | 0.503 | 0.45 | 0.316 | 0.372 |
| ±20s | 0.762 | 0.419 | 0.541 | 0.47 | 0.352 | 0.403 |
| ±30s | 0.788 | 0.479 | 0.596 | 0.47 | 0.397 | 0.431 |

At **±15s tolerance** (recommended):
- IO detects **73.5%** of facilitator prompts (F1=0.503)
- Expert detects **45.0%** of facilitator prompts (F1=0.372)

## 11. Per-Prompt Detection Detail (±15s tolerance)

### Reflective prompts (97 total)
- IO detection rate: **78.4%**
- Expert detection rate: **50.5%**

### Explicit prompts (54 total)
- IO detection rate: **64.8%**
- Expert detection rate: **35.2%**

### Detection rate by puzzle (±15s)

| Puzzle | N prompts | IO hit rate | Expert hit rate |
|--------|-----------|-------------|-----------------|
| Hub Puzzle: Cooking Pot | 46 | 73.9% | 50.0% |
| Spoke Puzzle: Amount of Protein | 36 | 66.7% | 0.0% |
| Spoke Puzzle: Amount of Sunlight | 13 | 92.3% | 7.7% |
| Spoke Puzzle: Pasta in Sauce | 26 | 57.7% | 73.1% |
| Spoke Puzzle: Water Amount | 25 | 100.0% | 84.0% |
| none | 5 | 20.0% | 80.0% |

### Detection rate by participant (±15s)

| Participant | N prompts | IO hit rate | Expert hit rate |
|-------------|-----------|-------------|-----------------|
| User-1 | 2 | 100.0% | 0.0% |
| User-2 | 6 | 50.0% | 50.0% |
| User-3 | 13 | 69.2% | 53.8% |
| User-5 | 13 | 84.6% | 61.5% |
| User-6 | 31 | 83.9% | 35.5% |
| User-9 | 25 | 80.0% | 60.0% |
| User-10 | 22 | 77.3% | 36.4% |
| User-12 | 9 | 77.8% | 44.4% |
| User-14 | 7 | 71.4% | 57.1% |
| User-22 | 8 | 25.0% | 12.5% |
| User-23 | 15 | 60.0% | 46.7% |