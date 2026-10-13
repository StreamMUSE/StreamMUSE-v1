# 周报简报：Offline 历史修复实验 + RL 后训练调研

日期：2026-10-08（晚间会议用）
范围：StreamMUSE-v1，本周（10/02–10/08）两条线：①老师布置的 offline bar 级历史修复（方案 2）；②ReaLchords/ReaLJam RL 后训练调研。

---

## 1. Offline 历史修复：设计、设置与本周实验

### 1.1 背景与问题定位

老师本周布置了三条优化方向：①训练 dropout；②实时判断质量不高直接 drop；③**每生成一个小节（4 beats）后，在 bar 单位上测量质量，不合格的历史直接代换而非丢弃**。

先厘清与文献的关系：ReaLchords/ReaLJam（详见 §3）针对的是同一个病——**exposure bias / 错误累积**（训练只见 GT 伴奏、推理却见自己的输出），但它们的手段全部在**训练期**（REINFORCE + 奖励模型 + offline teacher KL 蒸馏）。**推理期的"按小节测量并代换历史"机制不出自这两篇论文**，是本周我们自己设计的——一个更近的概念框架是 speculative decoding 的 draft-verify-accept（见 1.2）。

本周落地的是方向③：当某小节质量不佳时，用同一模型在无 deadline 条件下重采样该小节，经事务校验后替换进条件历史，让后续生成在更好的上下文上继续。

### 1.2 与 spec decoding 的同构性（内部理解框架）

系统级映射：**realtime 流式推理是 drafter**（持续产出条件历史），**offline 分支是验证侧**（无 deadline、算力充足，对应 SD 里的 target 模型），在 bar 边界 prefill 测量、决定接受或代换：

| Speculative decoding | 本系统 |
|---|---|
| drafter 出候选 | realtime 逐拍生成历史 |
| verifier（target 模型）验证 | offline 在 bar 边界测量质量 |
| 接受 → 拼入 KV | 达标 → 历史保留 |
| 拒绝 → 从拒绝点重算 | 不达标 → 代换该 bar |

本周实验是这张图的**单次退化版**：单次替换内部，offline worker 既出候选（重采样该 bar）又是被验对象。三臂对比里 ①≈②（同 seed 8/10 首 CR 逐位一致）说明 realtime 这个 drafter 的接受率天然很高——这是连续刷新版经济性的前提。

两点关键差异：
- **验证器是软门槛**（质量分数）而非 logit 精确匹配，因此本质是有损 draft-accept；
- **offline 分支没有信息优势**：continuation 模型是因果的，offline 跑同一模型能看的仍是过去 melody + 过去伴奏。优势只在**算力预算**（无 deadline，可重采样/best-of-N），是 self-speculation 而非 privileged drafter——因此 accept 无理论兜底，门槛是必需品。

### 1.3 机制实现（全部在独立 worktree，未提交）

代码：`worktrees/bar-quality-history-experiment/StreamMUSE-v1`，分支 `codex/bar-quality-history-experiment`

- `history_repair.py`：事件级交易验证器。重放引擎校验 tick 单调/音高/零长音符/重复 onset；**候选 bar 换入后小节边界的悬挂音集合必须与原历史完全一致**；拼回全曲后重放必须等于原 final active。任一违反 → 整体拒绝、保留原历史（fail-closed）。
- `history_repair_controller.py`：one-shot 编排。生成推进到 `bar_end + 4` 才触发（多等一拍保证 ACC-first 编码地平线）；异步提交私有 worker（进程隔离、独立 seed）；**全程不阻塞生成、从不等待模型计算**。
- 因果性保证：worker 只能拿到触发时刻**实际已收到**的 melody 事件，且截断到 bar 末尾——候选不偷看未来。
- 换的是**事件账本而非 KV cache**：每次 continuation 调用都从事件历史重新编码 32-beat 上下文，因此天然在 bar 边界重锚定，没有脏 KV 拼接问题。
- 实验层另有 `preserve_cross_boundary` 投影：只有完整落在 bar 内部的候选音符才替换，跨边界音符保留原样。

### 1.4 实验设置（frozen design）

- 数据：user10 的 10 首完整曲目 × seed 0/1/2，**baseline 与 repair 配对，30 对 / 60 份全曲输出**。
- 每对共享同一 Rule If-Else N=10 Prompt；采样 temp 1.1 / top-p 0.95 / top-k 50 / rep 1；模型 BPM 80。
- 逐拍同步请求，**无 wall-clock、无 deadline、无 drop**——即 baseline 臂本身就是纯 offline 全曲生成。
- 干预点：生成到 `[80,84)` 后，重采样历史 bar `[64,80)`（seed 101，单次，无质量门槛、无 best-of-N）。
- 修复只影响后续条件，已发出的音符永不改写；被拒的 5 对保留原历史且留在分母内。
- 完整性：3837/3837 配对调用的 melody token/pianoroll 摘要一致；tick 84 之前两组输入与事件完全相等。
- 运行环境：H200 GPU2，tmux `bar_history_offline_user10_20261007`，175 项测试通过。

### 1.5 本周测过的结果

**① 音乐指标（CR/UR，主窗口 `[84, 曲末)`，baseline → repair）**

| 指标 | baseline | repair | 变化 |
|---|---:|---:|---|
| CR（谐和率）↑ | 0.7328 | 0.7198 | −0.013 |
| UR（不谐和率）↓ | 0.0412 | 0.0608 | +0.020 |
| sounding coverage | 0.9277 | 0.9067 | −0.021 |

逐曲：4/10 首 CR 上升、6/10 下降；3/10 首 UR 下降、7/10 上升。**无门槛的固定历史替换未显示整体改善。**

**② 同 checkpoint NLL（事后评分，同 60 份输出）**

原版 **0.419** vs 修复版 **0.454**（+8.4%）——修复版似然更低。注意：同 checkpoint NLL 衡量模型自身偏好，不等于听感，也不独立。

**③ FMD/JSD**：特意未算——user10 谱面 GT 未对齐核实，硬算无意义。

**④ 60 份听感样本已回传本地**：`F:\repos\StreamMUSE-v1\remote_results\00_bar_history_probe_20261007\offline_user10\listening_by_piece\`，按曲目分组、同 seed 配对，可直接 A/B 盲听。

**小结：两个独立口径（音乐指标、模型似然）都指向"当前形态（单次、无门槛、单候选）的修复未占优"。**

### 1.6 三臂对比：Realtime P+C vs Offline P+C vs Offline+修复（10/08 补做）

应老师会前要求，把 9 月的实时正式批次与本周 offline 实验组装成三臂对比并统一打分（同 user10、同 Prompt 机制、同采样参数、同 checkpoint、同窗口 `[84, stop_tick)`、同 evaluator，各 30 份）：

| 臂 | CR ↑ | UR ↓ | onsets/beat |
|---|---:|---:|---:|
| ① Realtime P+C（9/10 正式批次） | 0.7222 | 0.0377 | 0.567 |
| ② Offline P+C（上周六实验 baseline 臂） | 0.7304 | 0.0377 | 0.593 |
| ③ Offline P+C + bar 修复 | 0.7240 | 0.0621 | 0.639 |

关键观察：

- **①≈②是确定性使然**：同 seed 下 10 首中 8 首 realtime 与 offline 的 CR 逐位一致（仅 08/09 因实时 drop 出现分歧）。说明本批实时实验的投递损失很小，offline 上限参照的抬头有限；三臂对比的信息量主要在③。
- **③修复效应双极**：repair 在 01（童话）CR 0.82→0.92 变好，但在 03/08/10 大幅变差；UR 整体 0.038→0.062。再次支持"无门槛单次替换是赌博"，门槛（E3）是必要前提。
- 试听包：`F:\repos\StreamMUSE-v1\remote_results\00_bar_history_probe_20261007\three_arm_bundle_20261008\`（10 曲 × 3 seed × 3 臂 = 90 个 MIDI，同 seed 配对；`three_arm_scores.json` 为逐曲分数；远端 masdar 同名 tar）。
- 校准备注：本表打分脚本与 1.5 的 `analyze_offline_paired.py` 用同一 evaluator，但 note 配对方式略异（UR 取 `unsupported_ratio`），臂间对比内部一致，**勿与 1.5 的表格直接混排**。

---

## 2. 核心疑问与后续路线

### 2.1 疑问一：如何证明 offline 修复比 continuation 更好？

当前的根本困难：**缺乏独立的裁判**。

- 同 checkpoint NLL 是循环论证——模型给自己两种输出打分，只反映模型偏好（且偏好简单重复内容），与"更好听"未建立相关；
- offline 与 realtime 的差距是明确的（主表：offline BEAT+GT FMD 109.8 vs realtime 160+，六项指标全占优），但"bar 级修复能把 realtime 往 offline 方向推多少"目前没有证据。

**建议方向（尚未定，供会议讨论）**：
1. **CLaMP2 嵌入相似度**（成本最低的建议）：FMD 栈里 CLAMP2 已装好有 cache，可直接抽逐曲嵌入做配对比较，当"第二意见"；CLaMP-DPO（NotaGen, arXiv 2502.18008）证明该类嵌入可作偏好信号。
2. **人工盲听**：60 份已就绪，同 seed 配对盲听是最直接的 ground truth，成本是人力。
3. **reward model**：一步到位且服务 RL 线，但需要 GPU 时间和标注对齐工作。

### 2.2 疑问二：断裂点（seam）如何处理？

实验与试听暴露的断裂机制：
- tail-active 校验只保证**音符连续**（悬挂音/跨界音不变），不保证**和声/调性连续**；
- 候选是温度 1.1 的单次采样，无门槛——可能换调、换和声走向；
- 试听印象（小幸运）：修复后段落风格变化明显，且其指标（CR −0.023 / UR +0.039）属于变差组。UR 恶化的方向性与"候选调性错位→与旋律音程冲突增多"假设自洽，待逐对听确认。

若做连续刷新（老师方案的终态），断裂问题会**每个 bar 出现一次**，因此门槛不是可选项而是必需品。候选门槛从轻到重：
1. **调性闸**：Krumhansl 式调性估计，候选与前缀不一致即拒绝（只挡换调）；
2. **嵌入闸**：CLAMP2 候选/前缀相似度阈值；
3. **NLL 闸**（弱门槛，有循环论证风险）；
4. **best-of-N + 门槛**：worker 异步、`bar_end+4` 才应用，时间窗本就允许 N 个候选择优——从"赌一把"变"选优"，架构现成。

### 2.3 下一步实验设计（建议顺序）

| 步骤 | 内容 | 成本 |
|---|---|---|
| E1 | 60 份 CLAMP2 嵌入配对评分（验证嵌入裁判与听感的相关性） | 极低，嵌入已有 |
| E2 | 同 seed 配对人工盲听（小幸运/cruel angel/天空之城优先），确认断裂的听感形态 | 人力 1-2 小时 |
| E3 | 单 bar best-of-N + 调性闸/嵌入闸重跑 30 对——看加门槛后修复能否从负转正 | GPU 数小时 |
| E4 | **Prompt+continuation 的 offline 推理版本（对比对象）**：~~待跑~~ **已于 10/08 完成**——即 1.6 三臂对比的②臂（30 份），并与①③组装成三臂 bundle + 统一打分（见 1.6）；如需扩到 test40 另需一批算力 | 已完成 |
| E5 | 转正后再讨论连续刷新版（多 bar 调度 + 每 bar 门槛 + worker 流水线化） | 待 E3/E4 结论 |
| 并行 | user10 谱面 GT 对齐核实 → 补 FMD/JSD | 与 10/02 谱面审计同源 |

**先 commit**：bar-quality worktree 全部改动与 5a95 的 lekai failcase 改动均未提交，接任何新实验前先固化。

---

## 3. RL 后训练调研（ReaLchords / ReaLJam / NotaGen）

笔记：`docs/research/realchords_rl_posttraining_review_20261007.md`

### 3.1 ReaLchords（ICML 2024, arXiv 2506.14723）

- 目标：让 causal online policy 在训练时反复经历**自己的输出历史**，学习从错误累积、冷启动、分布外 melody 中恢复——正对我们的 exposure bias（训练见 GT 伴奏、推理见自己的伴奏）。
- 训练三件套：**on-policy REINFORCE + 多尺度 Melody–Accompaniment 匹配奖励（contrastive + discriminative，1/16~1/2 多窗口）+ non-causal offline teacher 的 KL 约束**。
- 奖励不是单一乐理指标；KL 项防止 RL 破坏原有能力。
- 未公开：RL batch size、训练步数、采样规模等关键复现参数；官方代码未找到。

### 3.2 ReaLJam（CHI EA 2025, arXiv 2502.21267）

- 无新算法：把 ReaLchords 三种模型放进真人实时合奏系统。
- **对我们的四点用途**：①用户实验证明真人显著偏好 RL 模型（可引用的"RL 值得做"外部依据）；②自认长程乐句结构不足——与我们 bar 级问题同源，可作引证；③实时部署的延迟/体积/容错参照；④主观实验设计模板（我们手里有真人弹奏数据，具备做小规模盲听的条件）。

### 3.3 NotaGen（arXiv 2502.18008）——更轻的备选路线

- 符号音乐 LLM 三段式：预训练 1.6M 曲 → SFT 9K 古典曲 → **CLaMP-DPO**。
- 关键启示：**不需要 reward model / value model / on-policy rollout**，用现成音乐嵌入构造偏好对直接 DPO。对我们：若能组织（realtime vs offline、failcase 正负）偏好对，DPO 可能比 REINFORCE 全家桶更适配我们"四件套全无"的现状。
- 风险：嵌入奖励偏向表面统计特征（与其 NLL 循环论证同构）——只能当第二意见，终审仍靠人工/fail 标签。

### 3.4 仓库现状与推荐路线

现状：Continuation checkpoint 可作 policy 初始化，但 **reward model、value model、offline teacher、on-policy trainer 四件全无**。不能拿 Rule-S/PPL/CR 直接当奖励。

| 路线 | 前提 | 重量 |
|---|---|---|
| A. ReaLchords 式（reward model + REINFORCE + teacher KL） | 先训 reward model + rollout 接口 | 重 |
| B. NotaGen 式（嵌入偏好 + DPO，离线成对） | 组织偏好对 + CLaMP 嵌入（已装） | 轻 |
| C. 先造裁判（reward model / 嵌入验证） | — | 与 E1/E3 共用 |

**建议：2.1 的裁判工作（E1）先行——它同时是 bar 修复的验收官和路线 A/B 的奖励源；裁判验证通过后再在 A/B 之间选型。**

---

## 4. 老师会后反馈与三段式路线（10/11 追加）

老师听取汇报后给出三项方向，按"防新坏 → 治已坏 → 治根"组织：

### 4.1 事情①（治已坏）：更好的 history recovery model

两条方法路线：

- **方法 1：Inpaint model**——把坏小节当掩码、两侧上下文重建。条件最强但需要新模型和新数据管线。
- **方法 2（优先）：无偏移 continuation 重训**——去掉 ACC-first 编码中 melody/伴奏的时间错位（生成第 t 拍 acc 时可见第 t 拍自身 melody），从 base 微调。关键性质：**修复场景下不违反因果**——重采样 bar `[64,80)` 发生在 tick 84 后，该小节 melody 已经收到，多用的是系统已有的信息；相当于"用本小节自己的 melody 重新配伴奏"，接缝与和声对齐同时改善，且完全复用现有修复管线（worker/事务校验/门槛接口）。

### 4.2 事情②（防新坏）：强拍三和弦 logit boost（当日已实现并开跑）

现有引擎从累计 melody 估计调性（掩码式"禁出调"）。实验版升级为引导式：**每小节第 1/3 拍（强拍），对估计调的主三和弦 pitch class（覆盖全部八度）加 logit +τ**，从第 3 小节（tick 32，prompt 之后）生效。

实现：`TonalBeatMask` 扩展 boost 参数 + backend 环境变量窗口（`LEKAI_TONAL_BOOST_TAU/START_TICK/END_TICK`，默认关），含单测（tonal 套件 73 通过）。

实验设计（offline，user10 × 3 seed，共享 Prompt，无修复）：

| 臂 | 配置 |
|---|---|
| A | τ=0（基线，应与上周六 baseline 逐位一致=确定性校验） |
| B | τ=2.0，tick [32,256)（第 3–16 小节） |
| C | τ=2.0，tick [32, 曲末) |

判读：A vs B/C 的 CR/UR + 调性稳定度；若全不动=现有约束已饱和；若只防漂移不提升音乐性=调整预期。

**结果（10/11 晚跑完，n=30/臂）**：

| 臂 | CR ↑ | UR ↓ | 配对 dCR vs A |
|---|---:|---:|---|
| A 基线（τ=0） | 0.7304 | 0.0377 | — |
| B 至 16 小节 | 0.7564 | 0.0484 | +0.026（18 胜 11 负） |
| **C 全曲** | **0.7987** | **0.0369** | **+0.068（20 胜 10 负）** |

- **C 臂（全曲 boost）正信号**：CR +6.8 点且 UR 微降——谐和提升未付不谐和代价；7/10 首 CR 上升。
- **B 臂（中途截断）两头不讨好**：CR 只 +2.6 且 UR 恶化 +1.1 点——疑似中途撤掉 boost 本身制造分布切换，与 bar 修复的"断裂"教训呼应：**干预的进入和退出都需要平滑**。
- 确定性闭环：A 臂 CR 与上周六 30 对 baseline、三臂 bundle 的 offline 臂逐位一致（三批独立运行同 seed 同结果）。
- 诚实标注：单 τ（2.0）未扫强度；显著性未检验；boost 可能降低音高多样性待查。样本包与分数：`F:\repos\StreamMUSE-v1\remote_results\chord_law_20261008_results\`。

### 4.3 事情③（治根）：ReaLchords RL 后训练

老师目标=让现有模型经后训练提高"对未来的预测能力"，即治 exposure bias（训练见 GT、推理见自己的输出，误差累积）。重读论文带四个问题：①rollout 的 action 边界（一拍=一个 action？polyphonic token 怎么截断）；②offline teacher 架构——**方法 2 的无偏移模型可能兼任**（一砖两用）；③multi-scale 滑窗在全曲 offline 场景的切法；④MLE-likelihood 当 reward 会塌缩的教训（与我们 NLL 循环论证同坑）。实施顺序照旧笔记 Phase A（reward 验证）→ B（KD-only）→ C（REINFORCE）。

### 4.4 更新后的实验队列

| 步骤 | 内容 | 状态 |
|---|---|---|
| ②-law | 强拍 boost A/B/C（10 首×3 seed×3 臂，GPU2） | **10/11 已开跑** |
| ①-oracle | GT history 续写臂（天花板证据 + DPO 正例库） | 待跑（改 history 源即可） |
| ①-retrain | 方法 2 无偏移重训（需与老师钉死偏移量语义后动工） | 设计中 |
| E1 | CLaMP2 嵌入评 60 份（裁判验证，②①③共用） | **10/11 已跑完——负结果**：延续一致性余弦对配对条件不敏感（详见 §5） |
| ③-RL | Phase A rollout 审计（policy 起点=①重训后模型） | 排后 |

---

## 5. E1：CLaMP2 嵌入裁判验证（10/11，负结果）

**设计**：把每份输出在干预点 tick 84 切成上下文半段 `[0,84)` 与后半段，各取 CLaMP2 嵌入，算 cos(上下文, 延续) 作为"延续一致性"分数；配对比较各臂（上下文半段在配对内逐位相同，差异全部来自延续半段）。覆盖两组数据：30 对 baseline/repair + 90 份 law A/B/C，共 150 份 × 2 半段。

**结果**：

| 配对 | dCos 均值 | 偏好胜率 |
|---|---:|---|
| bar: repair vs baseline | +0.0051 | 16/30（掷硬币） |
| law: B vs A | −0.0066 | 12/30 |
| law: C vs A | −0.0022 | 16/30（打平） |

**判读**：

1. **CLaMP2 全局嵌入对这个粒度的配对差异不敏感**——绝对余弦 ~0.167，臂间差 ±0.005，低两个数量级；而同期 CR 显示 C 臂 +6.8 点的明确改善。嵌入裁判与和声指标**不一致**（对 repair 反而微偏好、对 C 的改善视而不见）。
2. 与 ReaLchords 的实证互证：**embedding bottleneck 偏全局风格、看不见局部和声细节**——我们 bar/强拍级别的干预恰好都在局部。
3. 对 DPO 路线的含义：NotaGen 式"嵌入排序产偏好对"在我们场景只能分辨**大质量差**（他们设定是 100 候选取两端），**细粒度配对判断不可用**。偏好对来源应回到：人工盲听（E2）或领域内自训 reward model（ReaLchords 路线 A）。
4. 附带确定性校验：bar-baseline 与 law-A 是同一条件，两次独立嵌入的绝对余弦一致到小数点后 4 位（0.1670）。

**注意适用范围**：本负结果只否定"延续一致性余弦"这一种用法；"与 GT 质心相似度"变体（NotaGen 原版）因 user10 谱面未对齐尚未测试，待 GT 对齐后可补测。

脚本：masdar `/data/home/yuanxin/experiments/e1_clamp_scores.py`（独立 venv `~/e1_clamp_venv`，未触碰共享环境）；分数 `e1_clamp_scores.json`。

---

## 附：文件索引

- 修复实现：`worktrees/bar-quality-history-experiment/.../src/streammuse/infrastructure/inference/history_repair.py`、`history_repair_controller.py`
- 实验文档：同 worktree `experiments/bar_quality_history/OFFLINE_PAIRED.md`、`docs/experiments/bar-quality-history-preparation-20261006.md`
- 样本包：`F:\repos\StreamMUSE-v1\remote_results\00_bar_history_probe_20261007\offline_user10\`
- RL 调研：`worktrees/5a95/StreamMUSE-v1/docs/research/realchords_rl_posttraining_review_20261007.md`
- 主表 FMD 数据：`worktrees/5a95/.../remote_results/05_music_metrics_202609/matched_40piece_3seed_trimmed_v1_v2single_beatgt_h200_20260903/`
- 待办：bar-quality worktree 未提交改动、5a95 lekai failcase 未提交改动
