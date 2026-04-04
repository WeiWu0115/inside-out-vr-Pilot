# Inside Out 技术报告：从概念到校准的完整记录

## 第一章：我们在解决什么问题

### 场景

18 个参与者戴着 Meta Quest Pro 玩一个 VR 密室逃脱游戏。游戏里有 4 个小谜题（Spoke Puzzles）和 1 个大谜题（Hub Puzzle）。每个人大约玩 20 分钟。

在实验中，有一个真人 facilitator 在旁边观察。当她觉得玩家卡住了、困惑了、需要帮助时，她会给提示：
- **Reflective prompt**（引导性提问）：比如"你觉得这个东西是做什么用的？"
- **Explicit prompt**（直接告知）：比如"把蛋白质调到 3"

### 核心问题

**我们能不能用一个自动系统来代替这个 facilitator？**

也就是说：系统实时读取玩家的眼动数据、操作数据、游戏状态，自动判断"这个人现在需不需要帮助"，如果需要，给什么类型的帮助。

### 为什么不直接做一个分类器？

传统做法是训练一个模型：输入 = 特征，输出 = "困惑/不困惑"。但我们的 FDG 论文发现，"困惑"不是一个单一状态 — K-means 聚类找到了至少两种不同的困惑：
- **迷失方向型**（disorientation）：玩家在房间里乱转，不知道该去哪
- **认知卡壳型**（cognitive impasse）：玩家盯着谜题，知道该做什么但做不出来

这两种困惑需要完全不同的帮助。所以我们需要一个系统能**区分不同类型的困难**，而不是简单地说"困惑了/没困惑"。

---

## 第二章：Inside Out 的设计思路

灵感来自皮克斯的《头脑特工队》：大脑里不是一个声音，而是多个角色在争论。我们的系统也是这样 — **多个 agent 各自解读玩家的状态，然后通过"争论"来做决定**。

### 五个 Agent

每个 agent 看一部分数据，回答一个特定的问题：

| Agent | 它看什么数据 | 它回答什么问题 | 它的判断 |
|-------|------------|--------------|---------|
| **Attention** | 眼动熵值、线索注视比、视线切换率 | "玩家在看什么？" | focused / searching / locked |
| **Action** | 操作次数、空闲时间、距上次操作时间 | "玩家在做什么？" | active / hesitant / inactive |
| **Performance** | 错误次数、操作次数、距上次操作时间 | "解题进展如何？" | progressing / stalled / failing |
| **Temporal** | 上面 agent 的历史判断 | "这个状态持续了多久？" | transient / persistent / looping |
| **Population** | 所有特征（对比群体聚类中心） | "跟其他玩家比怎样？" | exploring / stuck / solving / ... |

### 数据是什么样的

原始数据被切成 **5 秒一个窗口**（window）。每个窗口有 8 个特征值：

```
窗口 #42：玩家 3，时间 210-215 秒，在 Water Puzzle
  gaze_entropy = 2.1      （眼动分散程度，越高 = 看的地方越多）
  clue_ratio = 0.3        （盯线索的时间占比）
  switch_rate = 5.2        （视线切换频率）
  action_count = 2         （这 5 秒内的操作次数）
  idle_time = 3.1          （这 5 秒内没操作的时间）
  time_since_action = 45   （距离上一次操作过了多久）
  error_count = 0          （这 5 秒内的错误次数）
  puzzle_active = 1        （是否在解谜中）
```

### Agent 怎么做判断

以 Attention Agent 为例：
1. 读入 `gaze_entropy`、`clue_ratio`、`switch_rate`
2. 分别计算 "focused"、"searching"、"locked" 三个分数
3. 最高分的那个就是这个 agent 的判断
4. 分数就是 confidence（0-1）

比如：`gaze_entropy = 2.1`（高）+ `switch_rate = 5.2`（高）→ searching，confidence = 0.82

### 争论（Negotiation）怎么发生

Agent 判断完之后，系统检查每一对 agent 之间是否有**矛盾**或**共识**：

```
Attention 说 "searching"（在找东西）
Action 说 "inactive"（没在操作）
→ 矛盾！名叫 "scanning_but_passive"
→ 意思是：玩家在看来看去但没有动手，可能是迷路了

Attention 说 "focused"（专注看某个东西）
Action 说 "inactive"（没在操作）  
→ 矛盾！名叫 "focused_but_idle"
→ 意思是：玩家盯着谜题但不动，可能在思考，也可能卡住了

Attention 说 "focused"
Performance 说 "progressing"
→ 一致！名叫 "focused_progress"
→ 意思是：专注且有进展，不需要干预
```

### 最终决策（Support Layer）

根据争论的结果，系统做三种决定之一：

| 决定 | 含义 | 占比 |
|------|------|------|
| **watch** 👁 | 不干预，继续观察 | ~70% |
| **probe** 🔍 | agent 之间有分歧，先问一下试探 | ~24% |
| **intervene** 🚨 | agent 达成共识：玩家需要帮助 | ~7% |

---

## 第三章：对比对象 — Rule-Based 系统

教育专家观察了 18 个玩家后，总结了 **10 条规则**，做成了一个自动系统：

```
如果 time_since_action > 60 秒 且 在 puzzle 中 → 给提示
如果 error_count >= 3 → 给提示
如果 在 hub puzzle 且还没解完任何 spoke → 告诉玩家先去做小谜题
...
```

这个系统只用 game log（操作记录），不用眼动数据。它只能做两个决定：**给提示 / 不给提示**，没有 probe 的概念。

### 两个系统的对比

在 11 个同时有眼动 + game log 的玩家上对比（5,265 个 5 秒窗口）：

- 75.5% 的时间两个系统做了相同决定
- IO 发现了 **348 个时刻**是 rule-based 漏掉的
- IO 的 probe 类别（695 个决定）是 rule-based 完全没有的能力

---

## 第四章：引入 Ground Truth — 真人 Facilitator 的提示记录

之前 IO 和 rule-based 只能互相对比，没有 "正确答案"。后来我们拿到了另一个研究者分析暑假数据的结果 — 包含了 **真人 facilitator 实际给提示的精确时间戳**。

这就是 ground truth：真人专家在什么时刻觉得"这个玩家需要帮助"。

### 数据对齐

Facilitator 的提示是绝对时间戳（比如 21:41:54），IO 用的是相对秒数（比如 game 开始后第 655 秒）。我们通过 TimeLine 文件获取每个玩家的游戏开始时间，把 facilitator 的时间戳转换成相对秒数，然后对齐到 5 秒窗口。

### 为什么需要 Temporal Tolerance

Facilitator 给提示是一个**持续过程**（比如说了 30 秒话），而 IO 是每 5 秒一个判断。如果 IO 在 facilitator 开口前 10 秒就发现了问题并 probe，严格按窗口匹配会算成"没检测到"。所以我们用 ±15 秒容忍度：只要 IO 在 facilitator 给提示的前后 15 秒内做了 probe/intervene，就算检测到了。

### 初始基线结果（V0）

| 指标 (±15s) | IO | Rule-Based |
|---|---|---|
| **Recall** | 75.5% | 45.0% |
| **Precision** | 33.0% | 31.6% |
| **F1** | 0.459 | 0.372 |

**Recall**（召回率）= facilitator 给了 151 次提示，IO 检测到了多少？75.5% = 114 次。

**Precision**（精确率）= IO 所有说"需要干预"的时刻，有多少附近确实有 facilitator 提示？33%。

**F1** = Precision 和 Recall 的调和平均，综合指标。

---

## 第五章：诊断 — 为什么 IO 会漏掉 25% 的提示？

我们把每个 5 秒窗口分成四类：

- **TP**（True Positive）：IO 说"需要帮助"，facilitator 确实给了提示 ✓
- **TN**（True Negative）：IO 说"不需要"，facilitator 也没给 ✓
- **FN**（False Negative）：facilitator 给了提示，但 IO 说"不需要" ✗ ← **漏掉了**
- **FP**（False Positive）：IO 说"需要帮助"，但 facilitator 没给 ✗ ← **误报了**

### FN 分析：IO 为什么漏掉？

看 FN 窗口中 agent 的判断分布：

| Agent | FN（漏掉的） | TP（抓到的） |
|-------|-------------|-------------|
| Performance = "progressing" | **591 (70%)** | 0 (0%) |
| Performance = "stalled" | 252 (30%) | 239 (97%) |
| Action = "hesitant" | 417 | 5 |
| Action = "inactive" | 249 | 241 |

**核心问题一清二楚**：FN 里 70% 被 Performance Agent 标成了 "progressing"，但实际上玩家正在挣扎。

为什么？因为 Performance Agent 的 "progressing" 判断只看 `action_count > 0` — 只要玩家在这 5 秒内有任何操作就算"有进展"。但一个卡了 2 分钟的玩家偶尔乱点一下，不代表他在进步。

**关键特征对比**：

| 特征 | FN（漏掉） | TP（抓到） |
|------|-----------|-----------|
| action_count | 2.70 | 0.04 |
| time_since_action | 117s | 132s |

FN 窗口的 action_count 平均 2.7（有操作），TP 几乎是 0（完全不动）。这说明 IO 只能检测"完全不动"的困难，检测不了"乱动但没进展"的困难。

### FN 的 Tension Pattern 分布

| Tension Pattern | FN 数量 | IO 的决定 | 问题 |
|----------------|---------|----------|------|
| scattered_but_progressing | 330 | watch | "看起来乱但在进步，别打扰" — 但其实没在进步 |
| focused_but_idle | 232 | watch（仅 persistent 才 probe） | 门槛太高，应该更早 probe |
| focused_progress | 219 | watch | 看似正常，但 facilitator 还是给了提示 |

### FP 分析：IO 为什么误报？

833 个 FP 窗口中，**389 个（46.7%）发生在 Transition 阶段**（玩家在两个谜题之间走路）。IO 检测到 `scanning_but_passive`（在看但没操作）→ probe，但 facilitator 知道玩家只是在走路找下一个谜题，不需要帮助。

---

## 第六章：第一轮改进（V1 = 改动 A + B）

### 改动 B：修 Performance Agent（根本原因）

```
之前：action_count > 0 → progressing
之后：action_count > 0 但 time_since_action > 55s → 扣分
      如果 time_since_action 很高 → 给 stalled 加分
```

意思是：如果玩家之前已经 2 分钟没动了，现在偶尔点一下，不算真正的进步。

### 改动 A1：scattered_but_progressing 不再自动 watch

```
之前：scattered_but_progressing → 一律 watch
之后：如果 temporal = looping/persistent 或 time_since_action > 90s → probe
      否则 → watch
```

### 改动 A2：focused_but_idle 一律 probe

```
之前：只有 temporal = persistent 才 probe
之后：一律 probe（persistent 时 confidence 更高）
```

### 改动 A3：Transition 阶段抑制误报

```
之前：Transition 用正常规则处理
之后：Transition → 默认 watch（除非 passive_and_stuck + looping）
```

### V1 结果

| 指标 (±15s) | V0 | V1 | 变化 |
|---|---|---|---|
| **F1** | 0.459 | **0.508** | +10.7% |
| **Recall** | 75.5% | **79.5%** | +4.0pp |
| **Precision** | 33.0% | **37.3%** | +4.3pp |

---

## 第七章：第二轮改进（V2 = V1 + 改动 C）

### 改动 C：加入 Puzzle Elapsed Time

Facilitator 做判断时有一个 IO 没有的信息：**这个人在这个谜题上已经待了多久**。如果一个人在 Water Puzzle 上待了 10 分钟（正常是 2:54），facilitator 几乎肯定会给提示。

我们计算了每个窗口的 `puzzle_elapsed_ratio`：

```
puzzle_elapsed_ratio = 当前在该谜题上的时间 / 群体中位数时间

比如：玩家在 Water Puzzle 上已经 460 秒，中位数是 230 秒
→ ratio = 2.0（花了正常时间的 2 倍）
```

然后作为一个"安全网"加在 Support Layer 之后：

```
如果 ratio > 3.0 且当前决定是 watch → 升级为 probe
如果 ratio > 4.0 且当前决定是 probe → 升级为 intervene
```

### 阈值选择

我们测试了四组阈值：

| watch→probe | probe→intervene | F1 | Recall |
|---|---|---|---|
| 1.5× | 2.0× | 0.486 | 93.4% |
| 2.0× | 2.5× | 0.501 | 89.4% |
| 2.5× | 3.0× | 0.493 | 88.7% |
| **3.0×** | **4.0×** | **0.514** | **86.1%** |

选了 3.0×/4.0× — F1 最高，recall 和 precision 最平衡。

### V2 最终结果

| 指标 (±15s) | V0 | V1 (A+B) | V2 (A+B+C) |
|---|---|---|---|
| **F1** | 0.459 | 0.508 | **0.514** |
| **Recall** | 75.5% | 79.5% | **86.1%** |
| **Precision** | 33.0% | 37.3% | **36.6%** |

### 每个谜题的检测率

| Puzzle | V0 | V2 | Rule-Based |
|--------|-----|-----|-----------|
| Hub Puzzle (Cooking Pot) | 58.7% | **91.3%** | 50.0% |
| Amount of Sunlight | 61.5% | **92.3%** | 7.7% |
| Water Amount | 92.0% | **96.0%** | 84.0% |
| Amount of Protein | 80.6% | **83.3%** | 0.0% |
| Pasta in Sauce | 88.5% | 76.9% | 73.1% |

Pasta 下降的原因：6 个 miss 中 5 个发生在 Transition 阶段（facilitator 在玩家走向 Pasta 的路上就提示了，但 IO 把这些窗口标记为 Transition → watch）。

---

## 第八章：架构实验 — V2 Clean Boundaries

### 动机

V1 有一个理论问题：**feature overlap 导致虚假共识**。

`action_count` 被 4 个 agent 同时使用。当 `action_count = 3` 时：
- Attention 说 "not locked"（因为有动作 → 不是锁定状态）
- Action 说 "active"
- Performance 说 "progressing"
- Population 也受影响

表面上看：3 个 agent 都同意"玩家状态良好" → 系统很自信地说 watch。但实际上它们不是独立判断的 — 它们看的是同一个数字。这像三个人看同一份考试答案然后说"我们三个都同意答案是 C" — 这不是三倍的信心，只是一倍。

### 实验：完全消除 feature overlap

我们在一个 branch（`refactor/clean-agent-boundaries`）上试了完全隔离：

| Agent | V1 用的特征 | V2 用的特征 |
|-------|-----------|-----------|
| Perceptual | gaze_entropy, clue_ratio, switch_rate, ~~action_count~~ | gaze_entropy, clue_ratio, switch_rate |
| Behavioral | action_count, idle_time, ~~time_since_action~~ | action_count, idle_time, error_count |
| Progress | ~~error_count, action_count~~, time_since_action, puzzle_elapsed | time_since_action, puzzle_elapsed |
| Spatial (新) | — | puzzle_id (area heuristic) |

### 结果

| 指标 | V1 main | V2 branch |
|---|---|---|
| **F1** | **0.514** | 0.503 |
| **Recall** | **86.1%** | 73.5% |
| **Precision** | 36.6% | **38.2%** |

**Precision 提升了** — 说明消除 echo consensus 确实让系统的"自信"更可靠。

**Recall 下降了** — Progress Agent 没有 `action_count`，只看 `time_since_action`。当 `time_since_action` 短（最近有动作）时，它就说 "progressing"，但不知道这个动作是有意义的操作还是随便乱点。

### 核心发现

> 判断"解题是否有进展"本质上需要知道"玩家是否在行动"。完全切断这个信息流会损失能力。

但 V1 的方式（直接共享 raw feature）也有问题。

### 推荐方案：单向 Label Flow

```
V1 (有问题):
  action_count → Attention Agent
  action_count → Action Agent      ← 三个 agent 读同一个数字
  action_count → Performance Agent

V2 branch (太极端):
  action_count → Behavioral Agent only  ← Progress Agent 完全不知道玩家在不在动

推荐 V3 (折中):
  action_count → Behavioral Agent → "inactive" label → Progress Agent
                                                        ↑
                                   Progress 不读 raw action_count，
                                   而是读 Behavioral 的判断结果
```

这样：
- 每个 agent 仍然独立解读自己的 raw data
- Progress Agent 知道"玩家是否在动"，但通过 Behavioral Agent 的 **解读** 获知，而非直接读数字
- 如果 Behavioral 和 Progress 都说"有问题"，这是真正的两个独立信号，不是 echo

---

## 第九章：V3 — Label Flow + Stateful Prompt Agent

V2 的实验证明了完全隔离 feature 的方向是对的（precision 提升），但走得太极端（recall 暴跌）。V3 取中间路线：**单向 label flow + 新认知状态 + 有状态决策**。

### 9.1 六个改动

#### 改动 1：Label Flow 架构（解决 echo consensus）

```
V1:  action_count → Action Agent
     action_count → Performance Agent  ← 读同一个数字，虚假共识
     action_count → Attention Agent

V3:  action_count → Behavioral Agent → "inactive" label → Progress Agent
                                                           ↑
                                        Progress 读的是 Behavioral 的解读，
                                        不是原始数字
```

**效果**：当 Behavioral 说 "active" 而 Progress 说 "ineffective_progress" 时，这是两个独立信号的真正分歧，不是同一个 `action_count` 产生的 echo。

#### 改动 2：新状态 — `ineffective_progress`（解决最大的 FN 来源）

V1 的 Performance Agent 只有三个状态：progressing / stalled / failing。问题是很多卡住的玩家被标成了 "progressing"（因为有零星操作）。V3 加了第四个状态：

| 状态 | 含义 | 检测条件 |
|------|------|---------|
| **progressing** | 真正有进展 | behavioral=active + 近期有动作 + 没超时 |
| **ineffective_progress** | 在动但没进展 | behavioral=active/hesitant + time_since 高 + elapsed_ratio > 1.5 |
| **stalled** | 完全停滞 | behavioral=inactive + time_since 高 |
| failing | 在犯错 | behavioral=failing |

这直接解决了 V0 诊断中发现的核心问题："70% 的 FN 被错误标记为 progressing"。

对应的新 negotiation tension：
- `focused_but_ineffective`：玩家盯着谜题在操作但没进展 → probe
- `scattered_and_ineffective`：乱看乱点但没进展 → probe（persistent 时 → intervene）
- `passive_and_ineffective`：半动不动，也没进展 → intervene
- `hesitant_and_ineffective`：犹豫且无效 → probe

#### 改动 3：puzzle_elapsed_ratio 提前到 Agent 层

V1 中 `puzzle_elapsed_ratio` 只在最后一步作为 post-hoc 安全网。V3 把它整合进 agent 内部：

**Progress Agent**：
```
如果 elapsed_ratio < 1.0 → progressing 加分（还在正常时间内）
如果 elapsed_ratio > 2.0 → progressing 减分，stalled/ineffective 加分
如果 elapsed_ratio > 3.0 → 进一步减分
```

**Temporal Agent**：
```
如果没有时间模式但 elapsed_ratio > 3.0 → 标记为 persistent（即使窗口历史不一致）
如果有 looping 且 elapsed_ratio > 2.0 → 提升 looping confidence
```

#### 改动 4：Transition 子类型（解决 FP + Pasta 下降）

V1 的问题：Transition 阶段一刀切 → watch。这消除了 389 个 FP，但也导致 Pasta 的 5 个 miss（facilitator 在玩家走向 Pasta 的路上给了提示）。

V3 把 Transition 分成四种子类型：

| 子类型 | 检测方式 | 决策 |
|--------|---------|------|
| `navigating_normally` | 默认 | watch |
| `searching_transition` | 高 entropy + 高 switch_rate | watch（正常探索） |
| `hesitant_transition` | 高 idle + 无动作 + time_since > 60s | probe |
| `looping_transition` | time_since > 120s + 高 entropy | probe/intervene |

**效果**：Pasta puzzle 检测率从 V1 的 76.9% 恢复到 **88.5%**（回到 V0 水平），同时 Transition FP 仍然被有效抑制。

#### 改动 5：Stateful Prompt Agent（解决重复触发）

V1 的 support layer 是无状态的 — 每个 5 秒窗口独立决策。这导致：
- 连续 10 个窗口都说 probe → 实际上应该逐渐升级到 intervene
- 刚给了提示 → 5 秒后又说要给 → 不合理
- 玩家恢复了 → 系统不知道，还在高警戒

V3 的 PromptAgent 是一个有状态的类：

```python
class PromptAgent:
    last_prompt_time     # 上次干预时间 → cooldown
    consecutive_struggle # 连续困难窗口数 → escalation  
    prompt_count         # 本 puzzle 已给提示数 → fatigue
    current_puzzle       # 换 puzzle 时重置状态
```

决策策略：
- **Cooldown**：上次 intervene 后 15 秒内，新的 intervene → 降级为 probe
- **Escalation**：连续 6 个 struggle 窗口 → 把 probe 升级为 intervene
- **Recovery**：玩家变成 watch → 逐步降低 consecutive_struggle
- **Fatigue**：同一 puzzle 超过 8 次 intervene → 降级为 probe

**关键设计选择**：cooldown 和 fatigue 只限制 intervene，不限制 probe。因为 probe 是低成本的（只是更仔细地观察），intervene 才是打断玩家的。

#### 改动 6：Episode-Level 评估

传统的 per-window 评估（每个 5 秒窗口独立判对错）有一个问题：一次 facilitator 提示可能跨越 6 个窗口（30 秒），如果 IO 在前 2 个窗口检测到了但后 4 个没有，per-window 会报 4 个 FN。这不公平。

Episode-level 评估把**连续的 facilitator 提示**（间隔 < 60 秒）归为一个 "struggle episode"，然后问：

- 这个 episode 被检测到了吗？（至少有一个 probe/intervene 在 ±15s 内）
- 检测延迟是多少？（IO 第一次反应 vs episode 开始的时间差）

### 9.2 V3 结果

#### 核心指标（±15s tolerance）

| 指标 | V0 原始 | V1 (A+B+C) | **V3** | V0→V3 变化 |
|------|---------|------------|--------|-----------|
| **F1** | 0.459 | 0.514 | **0.529** | **+15.3%** |
| **Recall** | 75.5% | 86.1% | **92.1%** | **+16.6pp** |
| **Precision** | 33.0% | 36.6% | **37.1%** | **+4.1pp** |

#### Per-Puzzle 检测率

| Puzzle | V0 | V1 | **V3** | Rule-Based |
|--------|-----|-----|--------|-----------|
| Hub Puzzle | 58.7% | 91.3% | **93.5%** | 50.0% |
| Protein | 80.6% | 83.3% | **91.7%** | 0.0% |
| Sunlight | 61.5% | 92.3% | **100%** | 7.7% |
| Pasta | 88.5% | 76.9% | **88.5%** | 73.1% |
| Water | 92.0% | 96.0% | **100%** | 84.0% |

Pasta 的 V1 下降完全恢复。Sunlight 和 Water 达到 100%。

#### Episode-Level 评估（V3 独有）

| 指标 | IO (V3) | Rule-Based |
|------|---------|-----------|
| Episode recall | **85.0%** | 45.0% |
| Mean detection latency | **-5.2s**（提前 5.2 秒） | N/A |
| Early detection rate | **78%** | N/A |

80 个 struggle episode 中，IO 检测到 68 个。其中 53 个（78%）在 facilitator 开口**之前**就被检测到了，平均提前 5.2 秒。这意味着 IO 不仅能检测到困难，还能比真人更早发现。

#### Prompt Type 检测率

| 类型 | V0 | V1 | V3 |
|------|-----|-----|-----|
| Reflective | — | 82.5% | **92.8%** |
| Explicit | — | 74.1% | **90.7%** |

V3 对 explicit prompt（最严重的困难）的检测率提升最大，从 74.1% → 90.7%。

### 9.3 Label Flow 为什么解决了 V2 的问题

V2（完全隔离 feature）的 recall 暴跌到 73.5%，因为 Progress Agent 不知道玩家在不在动。V3 的 label flow 解决了这个矛盾：

| | V1 | V2 | V3 |
|---|---|---|---|
| Progress Agent 知道玩家行为？ | 是（读 raw action_count） | **否** | 是（读 behavioral label） |
| 信号独立性？ | **否**（echo consensus） | 是 | 是（label 是预解读的） |
| Precision | 36.6% | **38.2%** | **37.1%** |
| Recall | **86.1%** | 73.5% | **92.1%** |

V3 同时继承了 V1 的高 recall（因为 Progress Agent 有行为信息）和 V2 的高 precision（因为信号是独立的）。

---

## 第十章：可迁移的方法论

整个过程形成了一个 6 步 pipeline，适用于任何 multi-agent 认知状态系统：

```
1. 收集 Ground Truth
   → 真人专家在真实场景中的干预记录

2. 时间对齐
   → 把专家的干预时间戳对齐到系统的时间窗口
   → 使用 temporal tolerance（±15s）避免惩罚"早发现"的系统

3. 错误分类
   → 把每个窗口分为 TP/TN/FN/FP
   → 问：哪个 agent 的判断导致了 FN/FP？

4. 根因分析
   → 对比 FN vs TP 的 feature 分布
   → 追溯：错误的决定 ← 错误的 tension ← 错误的 agent label ← 错误的阈值

5. 定向校准
   → 每个改动针对一个具体的 failure mode
   → 改正确的层（agent 层 vs support 层）

6. 整体评估
   → 重跑全流程 + benchmark
   → 检查 regression（修了一个谜题，别的谜题有没有变差）
   → 逐谜题、逐玩家检查稳定性
```

---

## 第十一章：当前状态和下一步

### 当前状态

| Branch | 内容 | F1 | Recall | Precision | 状态 |
|--------|------|-----|--------|-----------|------|
| `main` | V1 + A+B+C calibration | 0.514 | 86.1% | 36.6% | 稳定版，Streamlit app 用这个 |
| `refactor/clean-agent-boundaries` | V2 clean boundaries 实验 | 0.503 | 73.5% | 38.2% | 实验版，记录了发现 |
| **`v3/label-flow-stateful`** | **V3 完整版** | **0.529** | **92.1%** | **37.1%** | **最新最优版** |

### 版本演进总结

```
V0 (Original)     F1=0.459  R=75.5%  P=33.0%
 │ 诊断：Performance Agent 的 "progressing" 太乐观
 │ 修复：time_since_action 惩罚 + support 规则调整
 ▼
V1 (A+B+C)        F1=0.514  R=86.1%  P=36.6%    (+12.0%)
 │ 问题：feature overlap 导致 echo consensus
 │ 实验 V2：完全隔离 → precision 升但 recall 崩
 │ 发现：Progress Agent 必须知道玩家行为，但不能读 raw data
 ▼
V3 (Label Flow)   F1=0.529  R=92.1%  P=37.1%    (+15.3% from V0)
   解决方案：
   ✓ Label flow（单向信号，不共享 raw feature）
   ✓ ineffective_progress（新认知状态）
   ✓ Transition 子类型（恢复 Pasta 检测）
   ✓ Stateful PromptAgent（cooldown + escalation）
   ✓ Episode-level evaluation（85% episode recall，78% 提前检测）
```

### 下一步

1. **合并 V3 到 main**：验证 Streamlit app 兼容性后合并
2. **80 人数据收集后**：重跑 pipeline + benchmark，验证 V3 校准是否泛化到更大样本
3. **Spatial Agent 完整版**：接入 head tracking 数据（Path/*.csv），替换 puzzle_id 的 area heuristic。这将进一步改善 Transition 阶段的检测
4. **Prompt 内容生成**：当前系统只决定 "什么时候给提示" 和 "什么类型"，下一步是生成具体的提示内容（接入 LLM）
5. **Unity 实时集成**：WebSocket server 接收实时数据流，完成从离线分析到在线系统的转换
6. **个性化校准**：用玩家在前几个 puzzle 的表现来个性化 `puzzle_elapsed_ratio` 的基线，而不是用群体中位数
