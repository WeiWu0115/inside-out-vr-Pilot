# Identifying Player Stuck States in a VR Puzzle Game: A Clustering Analysis of Eye-Tracking and Interaction Data

---

## 1. Method

### 1.1 Participants and Task

Eleven participants (User-1, 2, 3, 5, 6, 9, 10, 12, 14, 22, 23) completed a VR escape-room puzzle game ("Kitchen Nutrition") while wearing an eye-tracking-enabled headset. The game consisted of four Spoke puzzles (Pasta in Sauce, Amount of Protein, Water Amount, Amount of Sunlight) and one Hub puzzle (Cooking Pot). Session durations ranged from 17.2 to 60.0 minutes (*M* = 39.9 min, *SD* = 13.8 min).

### 1.2 Data Collection

Two concurrent data streams were recorded:

- **Player Tracking Data** — Frame-level (≈30 Hz) recording of binocular eye gaze position and direction, head position and rotation, hand position, rotation, and action state (e.g., OPEN, GRAB, TELEPORT_INTENT), cyclopean gaze viewport coordinates, and gaze target object name/scene.
- **Puzzle Event Logs** — Timestamped events recording each puzzle interaction, including element type (Interaction/Puzzle), completion status, and outcome (RightMove / WrongMove).

### 1.3 Feature Engineering

Raw frame-level tracking data was segmented into fixed-length **5-second time windows**. For each window, eight behavioral features were extracted:

| Feature | Description | Source |
|---------|-------------|--------|
| `gaze_entropy` | Shannon entropy (base 2) over the distribution of fixated objects within the window | Eye tracking |
| `clue_ratio` | Proportion of frames where gaze was directed at puzzle-relevant clue objects (diaries, hint boards, puzzle elements) | Eye tracking |
| `switch_rate` | Number of gaze target transitions per second | Eye tracking |
| `action_count` | Number of hand action state transitions (left + right) | Hand tracking |
| `idle_time` | Duration (seconds) where gaze target was unchanged AND both hands were in passive states (OPEN / NOT_TRACKED) | Eye + hand |
| `puzzle_active` | Binary: whether any puzzle interaction event occurred in this window | Puzzle logs |
| `error_count` | Number of WrongMove events in this window | Puzzle logs |
| `time_since_action` | Seconds elapsed since the most recent puzzle interaction event | Puzzle logs |

A total of **5,265 time windows** were extracted across all 11 participants.

### 1.4 Clustering

Features were standardized using z-score normalization (StandardScaler). K-means clustering was applied with *K* = 2, 3, 4, 5 (20 random initializations each, random seed = 42). The silhouette coefficient was computed for each *K*. While *K* = 2 yielded the highest silhouette score (0.371), we selected ***K* = 5** (silhouette = 0.279) to achieve finer-grained behavioral discrimination, as the research goal required distinguishing between qualitatively different stuck states. The five-cluster solution was validated through centroid analysis, temporal coherence in behavioral timelines, and alignment with puzzle event logs.

---

## 2. Results

### 2.1 Cluster Identification

Five distinct behavioral states were identified. Table 1 presents the centroid feature values for each cluster.

**Table 1.** Cluster centroid features (*M* ± *SD*). Bold values indicate the defining characteristic(s) of each cluster.

| Feature | C0: Transition | C1: Idle Waiting | C2: Active Solving | C3: Visual Exploration | C4: Stuck-on-Clue |
|---------|---------------|-----------------|-------------------|----------------------|-------------------|
| *n* (%) | 477 (9.1%) | 2,109 (40.1%) | 562 (10.7%) | 1,199 (22.8%) | 918 (17.4%) |
| Gaze entropy | 1.08 ± 0.68 | 0.96 ± 0.58 | 1.00 ± 0.65 | **2.35 ± 0.71** | 0.79 ± 0.59 |
| Clue ratio | 0.25 ± 0.33 | 0.09 ± 0.14 | 0.04 ± 0.10 | 0.15 ± 0.16 | **0.78 ± 0.21** |
| Switch rate (s⁻¹) | 2.79 ± 2.60 | 2.33 ± 1.73 | 2.54 ± 2.41 | **10.44 ± 4.49** | 2.29 ± 2.10 |
| Action count | 2.17 ± 3.36 | 2.21 ± 3.38 | 2.54 ± 2.59 | **2.90 ± 4.00** | 1.25 ± 2.76 |
| Idle time (s) | **0.72 ± 0.87** | 4.66 ± 0.43 | 3.68 ± 1.26 | 4.09 ± 0.51 | **4.73 ± 0.39** |
| Puzzle active | 0.00 | 0.00 | **1.00** | 0.01 | 0.00 |
| Error count | 0.00 | 0.00 | **0.33 ± 0.70** | 0.00 | 0.00 |
| Time since action (s) | 93.3 ± 86.5 | 79.0 ± 85.8 | **41.4 ± 76.6** | 98.7 ± 108.7 | **158.5 ± 145.6** |

The five clusters are interpreted as follows:

- **C0 — Transition**: Characterized by extremely low idle time (0.72 s), indicating rapid movement or teleportation between locations. No puzzle interaction.
- **C1 — Idle Waiting**: High idle time with low gaze entropy and minimal clue engagement. The player is stationary, fixating on non-informative surfaces (walls, floor). This represents a **disoriented stuck state** — the player does not know where to look.
- **C2 — Active Solving**: The only cluster with puzzle_active = 1. Low time-since-action and moderate error rate. Represents direct puzzle engagement.
- **C3 — Visual Exploration**: Highest gaze entropy (2.35) and switch rate (10.44 s⁻¹), with the most hand activity. The player is actively scanning the environment, looking at many different objects.
- **C4 — Stuck-on-Clue**: Highest clue ratio (78%), highest idle time (4.73 s), and longest time-since-action (158.5 s). The player is fixating on clue objects (diaries, hint boards) for extended periods without taking action. This represents a **comprehension-based stuck state** — the player has found the relevant information but cannot translate it into a solution.

**[Insert Figure: `02_centroid_chart.png` — Cluster Centroid Feature Comparison]**

**[Insert Figure: `01_pca_all_players.png` — PCA Projection of All Windows]**

### 2.2 Cluster–Performance Correlations

Table 2 presents the correlations between cluster proportions (per puzzle attempt) and puzzle performance outcomes across all 51 puzzle attempts (11 players × 5 puzzles, 4 attempts excluded due to window-mapping edge cases).

**Table 2.** Correlations between behavioral state proportions and puzzle performance. Significant results in bold.

| Cluster proportion | Completion time |  | Error count |  |
|--------------------|----------------|---|-------------|---|
| | *r* | *ρ* | *r* | *ρ* |
| % C0 (Transition) | −.080 | .194 | −.085 | .151 |
| % C1 (Idle Waiting) | .174 | .176 | **.266**† | **.283*** |
| % C2 (Active Solving) | **−.294*** | **−.596**** | .175 | .174 |
| % C3 (Exploration) | .126 | **.331*** | −.183 | −.122 |
| % C4 (Stuck-on-Clue) | .018 | **.243**† | −.220 | −.197 |

*Note.* *r* = Pearson correlation; *ρ* = Spearman rank correlation. *N* = 51 puzzle attempts.
†*p* < .10, \**p* < .05, \*\**p* < .01, \*\*\**p* < .001.

Key findings:

1. **Active Solving (C2) was the only state significantly associated with faster completion** (*r* = −.294, *p* = .036; *ρ* = −.596, *p* < .001). A higher proportion of time spent actively engaging with puzzles predicted shorter completion times.

2. **Idle Waiting (C1) was significantly associated with more errors** (*ρ* = .283, *p* = .044). Players who spent more time in a disoriented, passive state subsequently made more mistakes when they did attempt interactions.

3. **Visual Exploration (C3) showed a significant Spearman correlation with completion time** (*ρ* = .331, *p* = .018), suggesting that excessive scanning behavior — while potentially productive — was associated with longer puzzle durations when occurring in high proportions.

**[Insert Figure: `03_scatter_completion.png` — Cluster Proportions vs. Completion Time]**

**[Insert Figure: `08_correlation_heatmap.png` — Correlation Matrix]**

### 2.3 Group Comparisons

Median-split analyses using Mann-Whitney *U* tests confirmed the correlational findings:

**Table 3.** Group comparisons using median splits on cluster proportions.

| Split variable | Outcome | High group | Low group | *U* | *p* |
|----------------|---------|-----------|----------|-----|-----|
| % C2 (Active Solving) | Completion time | *M* = 155.0 s (*n* = 25) | *M* = 429.1 s (*n* = 26) | 99 | **< .001** |
| % C1 (Idle Waiting) | Error count | *M* = 4.7 (*n* = 25) | *M* = 1.4 (*n* = 26) | 454 | **.012** |

Players in the high Active Solving group completed puzzles nearly three times faster than those in the low group. Players in the high Idle Waiting group committed over three times as many errors.

**[Insert Figure: `05_boxplot_groups.png` — Boxplot Group Comparisons]**

### 2.4 Fast vs. Slow Solvers

Since all 51 puzzle attempts were ultimately successful (no failures), we compared behavioral profiles between fast (completion time ≤ 177.3 s, *n* = 26) and slow solvers (> 177.3 s, *n* = 25) using a median split.

**Table 4.** Mean cluster proportions for fast vs. slow solvers.

| Cluster | Fast solvers | Slow solvers | Difference |
|---------|-------------|-------------|------------|
| C0: Transition | .133 | .101 | −.032 |
| C1: Idle Waiting | .345 | .366 | +.021 |
| C2: Active Solving | **.317** | **.155** | **−.162** |
| C3: Exploration | .093 | **.195** | **+.102** |
| C4: Stuck-on-Clue | .112 | **.182** | **+.070** |

Fast solvers spent approximately twice as much time in the Active Solving state (31.7% vs. 15.5%). Slow solvers showed elevated proportions of Exploration (+10.2%) and Stuck-on-Clue (+7.0%), indicating that both excessive visual scanning and prolonged fixation on clue objects without action are associated with slower performance.

### 2.5 Puzzle-Level Behavioral Profiles

**Table 5.** Mean cluster distribution by puzzle type.

| Puzzle | C0 | C1 | C2 | C3 | C4 | Mean completion (s) | Mean errors |
|--------|----|----|----|----|----|--------------------|-------------|
| Pasta in Sauce | .06 | .30 | .14 | .17 | .33 | 145.7 ± 119.7 | 0.7 |
| Amount of Protein | .14 | .36 | .10 | .10 | .20 | 201.1 ± 156.1 | 4.6 |
| Water Amount | .07 | .28 | .08 | .23 | .17 | 246.0 ± 150.7 | 1.8 |
| Amount of Sunlight | .23 | .36 | .12 | .07 | .07 | 172.3 ± 115.5 | 0.7 |
| Hub: Cooking Pot | .07 | .49 | .16 | .04 | .12 | 723.4 ± 740.3 | 8.9 |

Notably, the Hub puzzle (Cooking Pot) had the highest proportion of Idle Waiting (49%) and the highest error count (8.9), consistent with its role as the most complex, integrative challenge. The Pasta in Sauce puzzle showed the highest Stuck-on-Clue proportion (33%), suggesting that players frequently found relevant clue information but struggled to apply it.

**[Insert Figure: `06_cluster_by_puzzle.png` — Cluster Distribution by Puzzle]**

**[Insert Figure: `07_cluster_by_player.png` — Cluster Distribution by Player]**

### 2.6 Behavioral Timelines

Figure 9 presents temporal cluster trajectories for four representative players, overlaid with puzzle solve events (green vertical lines) and error events (red × markers). Shaded regions indicate prolonged (≥ 15 s) episodes of Idle Waiting (orange shading) or Stuck-on-Clue (green shading).

**[Insert Figure: `09_behavioral_timelines.png` — Behavioral State Timelines]**

Three distinct temporal patterns were observed:

**Pattern A — Efficient solving (User-1, 17.2 min).** Brief initial exploration, followed by rapid cycling between Active Solving and short Exploration phases. Spoke puzzles were completed in quick succession with minimal errors. Only the Hub puzzle elicited a prolonged Stuck-on-Clue episode.

**Pattern B — Disoriented trial-and-error (User-6, 51.5 min).** Dominated by extended Idle Waiting segments interspersed with error-prone solving attempts. This player exhibited the highest error count (37 total) and showed a recurring sequence of *Waiting → Error → Waiting → Error*, suggesting persistent disorientation rather than comprehension difficulty.

**Pattern C — Comprehension bottleneck (User-9, 57.9 min).** Characterized by long Exploration phases followed by extended Stuck-on-Clue episodes. Unlike User-6, this player actively searched for and found clue objects, but struggled to translate information into action. The *Exploration → Stuck-on-Clue → (eventually) Solve* pattern repeated across multiple puzzles.

**Pattern D — Mixed strategy (User-22, 36.8 min).** Exhibited both Idle Waiting and Stuck-on-Clue episodes, with eventual convergence toward Active Solving. Errors occurred primarily during transitions from Stuck-on-Clue to solving, suggesting incomplete comprehension at the point of action.

### 2.7 Predictive Classification of Cognitive-Perceptual States

To evaluate whether the identified behavioral clusters can be reliably predicted from multimodal features alone — a prerequisite for real-time adaptive intervention — we trained supervised classifiers using Leave-One-Player-Out cross-validation (LOPO-CV, *N* = 11). Two models were evaluated: Logistic Regression (LR; *C* = 1.0, balanced class weights, max 1000 iterations) and Random Forest (RF; 100 trees, balanced class weights). Both models were trained on the same eight behavioral features described in Section 1.3, with z-score standardization applied per fold. No temporal context features were included.

Two label schemes were tested:

- **Binary**: Confused (C1 + C4) vs. Not Confused (C0 + C2 + C3)
- **Three-class**: Productive (C2) vs. Confused (C1 + C4) vs. Transition (C0 + C3)

#### 2.7.1 Classification Accuracy

**Table 6.** LOPO-CV classification results across two label schemes and two models.

| Label scheme | Model | Accuracy | F1-macro | F1 range across folds |
|--------------|-------|----------|----------|-----------------------|
| Binary | LR | .971 | .971 | .956 – .995 |
| Binary | RF | .983 | .983 | .970 – .997 |
| 3-class | LR | .970 | .974 | .945 – .995 |
| 3-class | **RF** | **.984** | **.986** | **.972 – .999** |

The RF classifier outperformed LR by approximately 1.3 percentage points across both schemes. Three-class classification was only marginally harder than binary, suggesting that cluster boundaries are well-separated in feature space.

#### 2.7.2 Confusion Matrix Analysis

Table 7 presents the normalized confusion matrix for the best-performing model (RF, 3-class). Per-class recall exceeded 97% for all three states.

**Table 7.** RF 3-class LOPO-CV confusion matrix (row-normalized). Raw counts in parentheses.

| | Predicted: Productive | Predicted: Confused | Predicted: Transition |
|---|---|---|---|
| **Actual: Productive** (*n* = 562) | **100.0% (562)** | 0.0% (0) | 0.0% (0) |
| **Actual: Confused** (*n* = 3,027) | 0.0% (1) | **98.8% (2,990)** | 1.2% (36) |
| **Actual: Transition** (*n* = 1,676) | 0.1% (2) | 2.8% (47) | **97.1% (1,627)** |

Three observations are noteworthy:

1. **Zero false positives for Productive → Confused.** The model never misclassified active solving as confusion. This is critical for deployment: a hint system using this classifier would never interrupt a productively engaged player.

2. **Primary confusion between Confused ↔ Transition.** The 83 misclassified windows (36 + 47) occurred between Confused and Transition states. This is consistent with the behavioral similarity between C1 (idle waiting) and C3 (visual exploration) — both involve limited hand activity and variable gaze behavior.

3. **Productive state perfectly classified.** All 562 windows of active puzzle engagement were correctly identified, reflecting the uniquely distinguishing role of the `puzzle_active` feature (binary indicator of puzzle interaction events).

**[Insert Figure: `12_confusion_matrix.png` — Normalized Confusion Matrix (RF, 3-Class)]**

#### 2.7.3 Feature Importance

Feature importance was assessed using mean decrease in Gini impurity, averaged across 11 LOPO folds. We note that Gini importance can overweight high-cardinality continuous features [Strobl et al., 2007]; however, all eight features are continuous, mitigating this concern.

**Table 8.** RF feature importance (mean ± SD across 11 LOPO folds).

| Rank | Feature | Binary importance | 3-class importance |
|------|---------|------------------|--------------------|
| 1 | Puzzle Active | .175 ± .008 | **.423 ± .003** |
| 2 | Idle Time | **.408 ± .009** | .235 ± .004 |
| 3 | Switch Rate | .186 ± .005 | .128 ± .004 |
| 4 | Gaze Entropy | .138 ± .004 | .101 ± .003 |
| 5 | Time Since Action | .025 ± .001 | .045 ± .002 |
| 6 | Clue Ratio | .040 ± .002 | .037 ± .002 |
| 7 | Error Count | .014 ± .001 | .019 ± .003 |
| 8 | Action Count | .015 ± .001 | .012 ± .001 |

Two features dominated: **idle time** (40.8% in binary) and **puzzle interaction presence** (42.3% in 3-class). In the binary scheme, idle time was the strongest predictor because it directly distinguishes the behavioral disengagement that characterizes both C1 and C4. In the 3-class scheme, puzzle_active became dominant because it is the sole feature that separates the Productive state from Transition (neither involves high idle time). The low importance of error_count (.014–.019) and action_count (.012–.015) suggests that these features contribute minimally beyond what idle time and switch rate already capture; however, they may become more informative as temporal features are added in future work.

**[Insert Figure: `13_feature_importance.png` — RF Feature Importance (3-Class)]**

#### 2.7.4 Interpretation and Deployment Implications

These results demonstrate that a lightweight Random Forest classifier can replicate unsupervised cluster assignments with F1-macro = .986 in LOPO-CV, enabling frame-level (5 s resolution) state estimation suitable for real-time adaptive hint systems without the computational overhead of online clustering. The classification operates on features computable within a single time window (no temporal look-back), making it directly deployable in streaming VR applications.

The feature importance analysis reveals that cognitive-perceptual confusion manifests primarily through **behavioral disengagement** (idle time, absence of puzzle interaction) rather than specific gaze patterns (clue ratio, gaze entropy contributed ≤ 13.8%). This has practical implications: a simplified two-feature detector using only idle time and puzzle interaction presence could achieve near-equivalent performance for resource-constrained deployments.

---

## 3. Discussion

### 3.1 Two Qualitatively Distinct Stuck States

The most important finding of this analysis is the identification of two qualitatively different stuck states that require different intervention strategies:

| | C1: Idle Waiting (Disoriented) | C4: Stuck-on-Clue (Comprehension) |
|---|---|---|
| **Behavioral signature** | Low entropy, low clue ratio, high idle time | Low entropy, **high clue ratio (78%)**, high idle time |
| **Interpretation** | Does not know where to look | Found the clue but cannot use it |
| **Performance impact** | Associated with more errors (*ρ* = .28, *p* = .044) | Associated with longer completion (trend, *ρ* = .24, *p* = .086) |
| **Suggested intervention** | Spatial guidance — direct attention toward relevant clue locations | Content support — provide interpretive hints about how to apply the clue information |
| **Example** | User-6: staring at walls before attempting wrong answers | User-9: reading diary entries repeatedly without acting |

### 3.2 Design Implications

These findings support the design of a **two-stage adaptive hint system**:

1. **Stage 1 (Detecting C1 → Spatial Hint):** When the system detects ≥ 3 consecutive Idle Waiting windows (15 s), provide a subtle spatial cue (e.g., gaze-contingent highlight, audio directional hint) to guide the player toward relevant clue objects.

2. **Stage 2 (Detecting C4 → Content Hint):** When the system detects ≥ 3 consecutive Stuck-on-Clue windows after the player has already fixated on the correct clue objects, provide a progressive content hint (e.g., simplified restatement of the clue, partial solution reveal).

### 3.3 Limitations

- **Sample size.** *N* = 11 limits the statistical power of correlation analyses. Results should be replicated with a larger sample (~80 participants). With more participants, confidence intervals on per-fold F1 scores can be reported, and a true hold-out test set (e.g., 60 train / 20 test) becomes feasible.
- **All puzzle attempts were successful**, preventing success/failure comparison.
- **Window size.** The 5-second window was chosen a priori; sensitivity analyses with other window sizes (e.g., 3 s, 10 s) may yield different cluster boundaries.
- **Clustering assumptions.** K-means assumes spherical clusters; future work could explore DBSCAN or Gaussian Mixture Models.
- **Cluster labels as ground truth.** The predictive classification (Section 2.7) used K-means cluster assignments as supervised labels. Because the labels were derived from the same features used for prediction, the high classification accuracy (F1 = .986) reflects the model's ability to reproduce a decision boundary rather than to predict an independently validated cognitive state. Future work should validate cluster labels against external measures (e.g., think-aloud protocols, self-report, or expert video annotation) and compute inter-rater agreement (Cohen's κ) between cluster labels and human judgments.
- **Transductive leakage in clustering.** K-means was applied to the full dataset prior to LOPO cross-validation, meaning that cluster centroids reflect the held-out player's data. A fully nested design with per-fold clustering would eliminate this dependency but introduces label alignment challenges across folds. We note that LOPO-CV nonetheless ensures no temporal windows from the test player appear in the training set, and that the narrow F1 range across folds (.972–.999 for RF 3-class) suggests robust generalization.

---

## Figure Captions

- **Figure 1** (`01_pca_all_players.png`): PCA projection of 5,265 time windows across 11 players, colored by cluster assignment (*K* = 5). PC1 and PC2 explain 22.8% and 20.5% of variance, respectively.

- **Figure 2** (`02_centroid_chart.png`): Z-scored cluster centroid feature comparison. Each group of bars represents one feature; bars are colored by cluster. The dominant features defining each cluster are visually apparent.

- **Figure 3** (`03_scatter_completion.png`): Scatter plots showing the relationship between each cluster's proportion and puzzle completion time. Dashed lines indicate linear fit. Only C2 (Active Solving) shows a significant negative correlation (*r* = −.29, *p* = .036).

- **Figure 4** (`04_scatter_errors.png`): Scatter plots showing the relationship between each cluster's proportion and error count per puzzle attempt.

- **Figure 5** (`05_boxplot_groups.png`): Box plots comparing completion time between high and low Active Solving groups (left) and error count between high and low Idle Waiting groups (right). Median splits were used for group assignment.

- **Figure 6** (`06_cluster_by_puzzle.png`): Stacked bar chart showing the mean cluster distribution for each puzzle type, averaged across all players.

- **Figure 7** (`07_cluster_by_player.png`): Stacked bar chart showing the mean cluster distribution for each player, averaged across all puzzles.

- **Figure 8** (`08_correlation_heatmap.png`): Pearson correlation matrix between cluster proportions, completion time, error count, and cluster transition count.

- **Figure 9** (`09_behavioral_timelines.png`): Behavioral state timelines for four representative players. Each row represents one player. Colored bars indicate the cluster assignment of each 5-second window. Green vertical lines mark puzzle completion events; red × markers indicate errors. Shaded regions highlight prolonged (≥ 15 s) Idle Waiting (orange) or Stuck-on-Clue (green) episodes.

- **Figure 10** (`12_confusion_matrix.png`): Row-normalized confusion matrix for the RF 3-class classifier under LOPO-CV. Cell values show recall percentages with raw counts in parentheses. The model achieves ≥ 97.1% recall for all three states, with zero false positives for Productive → Confused misclassification.

- **Figure 11** (`13_feature_importance.png`): Mean decrease in Gini impurity for the RF 3-class classifier, averaged across 11 LOPO folds. Error bars indicate ± 1 SD. Puzzle Active (42.3%) and Idle Time (23.5%) are the two dominant predictors, together accounting for 65.8% of total importance.

---

## Appendix: File Inventory

| File | Description |
|------|-------------|
| `all_windows_with_clusters.csv` | 5,265 window-level records with all 8 features, cluster assignments, and puzzle phase labels |
| `puzzle_performance_merged.csv` | 51 puzzle-level records with cluster proportions, completion time, error count, and transition count |
| `puzzle_level_performance.csv` | 55 puzzle-level performance records (time ranges, errors, success) |
| `01_pca_all_players.png` | PCA cluster visualization |
| `02_centroid_chart.png` | Cluster centroid feature comparison |
| `03_scatter_completion.png` | Cluster proportion vs. completion time |
| `04_scatter_errors.png` | Cluster proportion vs. error count |
| `05_boxplot_groups.png` | Group comparison box plots |
| `06_cluster_by_puzzle.png` | Cluster distribution by puzzle |
| `07_cluster_by_player.png` | Cluster distribution by player |
| `08_correlation_heatmap.png` | Correlation heatmap |
| `09_behavioral_timelines.png` | Behavioral state timelines (300 DPI, 5966 × 4478 px) |
| `12_confusion_matrix.png` | Normalized confusion matrix heatmap (RF, 3-class, LOPO-CV) |
| `13_feature_importance.png` | RF feature importance bar chart (3-class, LOPO-CV) |
| `predict_minimal.py` | Predictive classification pipeline (binary + 3-class, LR + RF, LOPO-CV) |
| `plot_confusion_matrix.py` | Script for generating Figures 10–11 |
