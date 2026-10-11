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

## 附：文件索引

- 修复实现：`worktrees/bar-quality-history-experiment/.../src/streammuse/infrastructure/inference/history_repair.py`、`history_repair_controller.py`
- 实验文档：同 worktree `experiments/bar_quality_history/OFFLINE_PAIRED.md`、`docs/experiments/bar-quality-history-preparation-20261006.md`
- 样本包：`F:\repos\StreamMUSE-v1\remote_results\00_bar_history_probe_20261007\offline_user10\`
- RL 调研：`worktrees/5a95/StreamMUSE-v1/docs/research/realchords_rl_posttraining_review_20261007.md`
- 主表 FMD 数据：`worktrees/5a95/.../remote_results/05_music_metrics_202609/matched_40piece_3seed_trimmed_v1_v2single_beatgt_h200_20260903/`
- 待办：bar-quality worktree 未提交改动、5a95 lekai failcase 未提交改动
