# Paper Section Status — Inside Out CHI '27

Last updated: 2026-04-05

## Section Map

| # | Section | File | Status | Key Content |
|---|---------|------|--------|-------------|
| 1 | Introduction | `introduction.tex` | Draft done | Inside Out metaphor, classification pipeline critique, 7 paragraphs with heavy citations |
| 2 | Related Work | `related_work.tex` | Complete | 126 papers, from FDG submission |
| 3 | Framework | `framework.tex` | Draft done | System overview, 5 agents, negotiation model, ~40 citations |
| 4 | Pilot Study | `pilot_study.tex` | **Active — primary work this session** | See detailed status below |
| 5 | Discussion | — | **Not started** | Need to write |
| 6 | Conclusion | — | **Not started** | Need to write |

## Section 4 (Pilot Study) Detailed Status

### 4.1 Study Context ✅
- 18 participants, Meta Quest Pro, VR escape room
- **NEW**: Task structure table (puzzle solve times + prompt frequency)
- **NEW**: Facilitator ground truth description (223 prompts, scaffolding principle)

### 4.2 Two Systems, Two Philosophies ✅
- Rule-based system (10 rules, game logs only)
- **UPDATED**: Inside Out V3 description with:
  - 5 agents with exclusive features listed
  - Label flow architecture explained
  - `ineffective_progress` state
  - Stateful Prompt Agent (cooldown/escalation/recovery)

### 4.3 Results: IO vs Rule-Based ✅
- 75.5% agreement, cross-tabulation table
- When expert intervenes but IO doesn't (Rule 3 accounts for 73%)
- When IO intervenes but expert doesn't (passive_and_stuck, n=293)
- Probe category (695 decisions, no expert equivalent)
- Expert's STUCK label contains 7 IO tension patterns

### 4.4 Summary ✅
- Three empirical properties: richer vocabulary, within-category diversity, complementary evidence

### 4.5 Validation Against Facilitator Ground Truth ✅ **NEW**
- **Temporal alignment**: ±15s window-level + episode-level evaluation
- **Baselines**: random F1=0.208, always-intervene F1=0.344
- **Window-level results table**: IO F1=0.529 (2.5× random), recall=92.1%
- **Episode-level results table**: 85% recall, 78% early detection, -5.2s latency
- **Calibration history table**: V0(0.459) → V1(0.508) → V1+C(0.514) → V3(0.529)
- **Key diagnostic finding**: 70% FN from "progressing" mislabel, echo consensus
- **Architectural improvement**: label flow, ineffective_progress, stateful agent
- **Transferable methodology**: 6-step pipeline generalized

## What Section 5 (Discussion) Should Cover

Based on this session's findings, the discussion should address:

### 5.1 What Multi-Agent Negotiation Adds
- IO detects 92.1% of facilitator prompts vs 45.0% for rule-based (2× recall)
- The probe category (24.2% of decisions) has no rule-based equivalent
- Expert's STUCK label collapses 7 distinct tension patterns → IO preserves this diversity

### 5.2 The Echo Consensus Problem and Label Flow Solution
- V1 had 4 agents reading same `action_count` → fake agreement
- V2 experiment: complete isolation → precision up but recall collapsed
- V3: label flow = information flows as pre-interpreted labels, not raw features
- This is analogous to the perspectivist turn in ML: same data, different interpretive frameworks

### 5.3 Two Performance Ceilings
- **Feature granularity ceiling**: `action_count` doesn't distinguish meaningful vs random actions. All theories, all architectures hit this same wall. Solution: semantic action labels, gaze-action coupling.
- **Ground truth noise ceiling**: facilitator prompts include proactive guidance (undetectable) and reactive intervention (detectable). Solution: tag prompts as reactive vs proactive.

### 5.4 Data-Partitioned vs Theory-Partitioned Agents
- V3 (data-partitioned): F1=0.529, Recall=92.1% — agents disagree because they see different data
- Theory-partitioned: F1=0.531, Precision=37.9% — agents disagree because they hold different theories
- Theory-partitioned is more faithful to Inside Out metaphor
- Neither is strictly superior; trade-off between recall (V3) and precision/interpretability (theory)
- Cite: Posner (attention), Zimmerman (self-regulation), Csikszentmihalyi (flow), Sweller (cognitive load)

### 5.5 The Trigger Frequency Problem
- IO triggers 2× more than facilitator overall
- Probe rates are comparable (24.2% vs 19.5%)
- Intervene rates diverge (18.3% vs 1.3% = 15× more)
- Facilitator is very restrained with explicit prompts (scaffolding principle)
- Future work: calibrate intervention threshold, not detection threshold

### 5.6 Limitations
- N=18 (11 with eye tracking) — pilot, not definitive
- Ground truth from single facilitator — inter-rater reliability unknown
- Features capture quantity not quality of actions
- No head tracking in windowed data (available in raw but not integrated)
- Stateful agent parameters (cooldown=15s, escalation=6) not cross-validated

### 5.7 Future Work
- 80-person study with refined ground truth
- Theory-partitioned agents with richer features
- **Learnable integration weights**: keep agent layer interpretable (rule-based), learn only the final integration layer weights from facilitator ground truth. Hybrid of neural network efficiency + multi-agent interpretability. Requires 80-person data (~25k windows) for training. Compare: hand-tuned vs learned vs end-to-end MLP baseline.
- Unity real-time integration via WebSocket
- LLM-generated prompt content (not just when to prompt, but what to say)
- Per-player calibration using early puzzle performance as baseline

## Files to Create

| File | Content | Priority |
|------|---------|----------|
| `discussion.tex` | Sections 5.1-5.7 above | **High** |
| `conclusion.tex` | Summarize contributions + future | **High** |
| Update `inside_out_full_draft.tex` | Replace scenarios with pilot_study, add discussion + conclusion | **Medium** |

## Key Numbers for the Paper

| Metric | Value | Context |
|--------|-------|---------|
| F1 (±15s) | 0.529 | 2.5× random (0.208), 1.5× always-intervene (0.344) |
| Recall | 92.1% | vs rule-based 45.0% |
| Precision | 37.1% | structural ceiling from early detection |
| Episode recall | 85.0% | 68/80 struggle episodes detected |
| Early detection | 78% | IO detects before facilitator in 53/68 cases |
| Detection latency | -5.2s | IO is 5.2 seconds ahead of facilitator |
| Calibration improvement | +15.3% F1 | V0(0.459) → V3(0.529) |
| Rule-based blind spots | 0% on Protein | IO detects 91.7% on same puzzle |
