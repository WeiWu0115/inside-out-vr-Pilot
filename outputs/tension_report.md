# Inside Out: Agent Tension Report
**Dataset:** 13 players, 5265 time windows (5-second each), VR Escape Room  
**Agents:** Attention, Action, Performance, Temporal, Population  
**Generated:** 2026-04-01

---

## 1. Overall Disagreement Summary

| Metric | Value |
|--------|-------|
| Total windows | 5,265 |
| Mean disagreement intensity | 0.576 |
| Contradictory disagreements | 3,613 (68.6%) |
| Constructive alignments | 1,631 (31.0%) |
| Unstructured | 21 (0.4%) |

**System responses:**
| Category | Count | % |
|----------|-------|---|
| Watch (no intervention) | 4,185 | 79.5% |
| Probe (exploratory) | 695 | 13.2% |
| Consensus Intervene (direct) | 385 | 7.3% |

---

## 2. Top Tensions Across All Players

| Tension | Count | % | Type | Meaning |
|---------|-------|---|------|---------|
| scattered_but_progressing | 1,770 | 33.6% | Contradictory | Attention scattered but making progress — don't interrupt |
| focused_progress | 1,069 | 20.3% | Constructive | Focused and progressing — productive engagement |
| focused_but_idle | 952 | 18.1% | Contradictory | Focused attention but no action — thinking or stuck? |
| scanning_but_passive | 690 | 13.1% | Contradictory | Scanning environment but not acting — exploring or lost? |
| passive_and_stuck | 329 | 6.3% | Constructive | Inactive and stalled — clear difficulty |
| pop_confirms_progress | 160 | 3.0% | Constructive | Population data confirms rule agents' progress reading |
| pop_says_exploring_but_focused | 56 | 1.1% | Contradictory | Population says exploring, Attention says focused |
| focused_but_failing | 39 | 0.7% | Contradictory | Focused but making errors — needs procedural help |
| pop_says_solving_but_stalled | 37 | 0.7% | Contradictory | Population says solving, Performance says stalled |
| pop_confirms_exploration | 36 | 0.7% | Constructive | Population confirms exploration pattern |
| pop_says_exploring_but_inactive | 36 | 0.7% | Contradictory | Population says exploring but player isn't acting |
| pop_says_stuck_but_active | 19 | 0.4% | Contradictory | Population says stuck but player is active — atypical |
| frozen_on_clue | 16 | 0.3% | Constructive | Locked gaze + inactive — cognitive impasse |
| engaged_and_active | 14 | 0.3% | Constructive | Focused + active — strong engagement |
| idle_but_progressing | 10 | 0.2% | Contradictory | Low activity but somehow progressing |

---

## 3. Per-Player Profile

| Player | Windows | Contradictory | Constructive | Interventions | Probes | Top Tension |
|--------|---------|--------------|-------------|---------------|--------|-------------|
| P1 | 207 | 78% | 22% | 16 | 43 | scattered_but_progressing |
| P2 | 276 | 62% | 38% | 17 | 28 | scattered_but_progressing |
| P3 | 381 | 61% | 38% | 26 | 28 | scattered_but_progressing |
| P5 | 372 | 70% | 30% | 17 | 35 | scattered_but_progressing |
| P6 | 618 | 73% | 27% | 18 | 92 | scattered_but_progressing |
| P9 | 695 | 69% | 30% | 106 | 121 | scattered_but_progressing |
| P10 | 720 | 66% | 34% | 52 | 103 | scattered_but_progressing |
| P12 | 452 | 69% | 30% | 40 | 63 | scattered_but_progressing |
| P14 | 556 | 66% | 34% | 36 | 47 | scattered_but_progressing |
| P22 | 442 | 62% | 37% | 24 | 40 | scattered_but_progressing |
| P23 | 546 | 78% | 22% | 33 | 95 | scattered_but_progressing |

**Key observations:**
- P9 received the most interventions (106) and probes (121) — longest and most difficult session
- P1 had the highest contradiction rate (78%) despite shortest session — agents struggled to agree
- P2 and P22 had the most constructive alignment (37-38%) — clearer cognitive states

---

## 4. Per-Puzzle Tension Breakdown

### Transition Phase (n=2,221)
- Dominated by `scattered_but_progressing` (39%) — players scanning environment while making progress
- Highest intervention count (142) — many moments of ambiguity during navigation

### Spoke Puzzles

| Puzzle | Top Tension | Interventions | Probes | Character |
|--------|------------|---------------|--------|-----------|
| Pasta in Sauce (n=270) | focused_but_idle (24%) | 33 | 42 | Thinking-heavy, players pause to reason |
| Amount of Protein (n=370) | focused_progress (22%) | 53 | 34 | Most productive, clear engagement |
| Water Amount (n=472) | scattered_but_progressing (40%) | 58 | 100 | Most exploration, high probing |
| Amount of Sunlight (n=330) | scattered_but_progressing (50%) | 28 | 55 | Hardest to interpret, very scattered |

### Hub Puzzle: Cooking Pot (n=1,602)
- `focused_progress` (27%) and `scattered_but_progressing` (26%) almost tied
- Mix of clear engagement and ambiguous scanning — players alternate between understanding and searching
- 33% of Population Agent readings = "disoriented" — hub puzzle is where players get most lost

---

## 5. Population Agent Analysis

### Population Agent Label Distribution
| Label | Count | % | Meaning |
|-------|-------|---|---------|
| exploring | 1,468 | 27.9% | Behavior matches C3: high entropy, scanning |
| disoriented | 1,405 | 26.7% | Behavior matches C1: low engagement, idle |
| cognitively_stuck | 1,007 | 19.1% | Behavior matches C4: fixated on clues, no progress |
| actively_solving | 858 | 16.3% | Behavior matches C2: active interaction |
| transitioning | 527 | 10.0% | Behavior matches C0: moving between areas |

### Population Agent Disagreements with Rule-Based Agents

Total population-related tensions: **351 / 5,265 (6.7%)**

| Tension | Count | What it reveals |
|---------|-------|-----------------|
| pop_confirms_progress (constructive) | 160 | Population data validates rule agents — convergent evidence |
| pop_says_exploring_but_focused (contradictory) | 56 | Population norms say this is exploration, but Attention Agent reads focused attention — player may have a **unique strategy** |
| pop_says_solving_but_stalled (contradictory) | 37 | Population data says this pattern usually means active solving, but Performance Agent sees no progress — player **deviates from population norm** |
| pop_confirms_exploration (constructive) | 36 | Both population and rule agents agree: exploration |
| pop_says_exploring_but_inactive (contradictory) | 36 | Population says exploring but player isn't acting — **passive exploration** or early disorientation |
| pop_says_stuck_but_active (contradictory) | 19 | Population says stuck but player is active — **atypical coping strategy**, acting despite matching stuck profile |
| pop_confirms_impasse (constructive) | 7 | Strong convergence: both data and rules indicate cognitive impasse |

### Why This Matters

The 148 contradictory tensions between Population Agent and rule-based agents represent moments where **the current player's behavior deviates from what 13 prior players typically exhibited**. These are precisely the cases where a single classifier trained on population data would make the wrong call — the player is doing something the population model doesn't expect.

In the Inside Out framework, these disagreements don't force a choice. Instead, they trigger probing interventions that respect the ambiguity: "your behavior looks like exploration to most people, but your attention pattern says otherwise — let's find out which it is."

---

## 6. Implications for the 80-Player Study

1. **Population Agent will become more accurate** — with 80 players, cluster centroids will be more stable and representative
2. **Individual deviation detection** — the contradiction rate between Population and rule agents can serve as a **personalization signal**: players with high deviation rates may need non-standard support
3. **Puzzle-specific profiles** — different puzzles produce different tension profiles (e.g., Water Amount is exploration-heavy, Protein is engagement-heavy). The 80-player study can build per-puzzle population models
4. **Temporal negotiation patterns** — with more data, we can analyze how tension sequences (not just snapshots) relate to outcomes
