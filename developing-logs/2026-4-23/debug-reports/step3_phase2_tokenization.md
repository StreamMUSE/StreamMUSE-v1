# Step 3 Phase 2: Tokenization 层分析 执行报告

**执行日期**: 2026-04-24  
**执行人**: Claude Code  

---

## 1. 执行内容

- [x] 分析 Offline 模式的 prompt 构造结构（`model.py:generate_accompaniment`）
- [x] 分析 FakeRT 模式的 prompt 构造结构（`lekai_http_backend.py:_generate_with_interleaved_prompt`）
- [x] 对比两者的初始 token 序列差异（Bug 3）
- [x] 分析 delay_beats=-1 机制对 prompt 结构的影响
- [x] 确认 FakeRT 与 Offline 的 prompt 结构是否等价

---

## 2. Prompt 结构对比分析

### 2.1 Offline 模式（`model.py`，delay_beats=-1）

Offline 使用 `delay_beats=-1`，意味着伴奏"领先"旋律 1 拍。  
具体实现：在 `part0_beats_list` 开头插入 1 个 `pad_marker`，使得：

```
part0_beats_list = [pad_marker, bar_token, beat0_mel, beat1_mel, ..., bar_token, beat4_mel, ...]
part1_beats_gt_list = [bar_token, beat0_gt, beat1_gt, ..., pad_marker]
```

生成时的 token 注入顺序（前几步）：

| 步骤 | position | 注入/生成 | 累计序列 |
|------|----------|-----------|---------|
| 0    | part0    | 注入 `pad_marker` | `[BOS, ts, bpm, pad]` |
| 1    | part1    | **生成 acc_{-1}**（从 `[BOS, ts, bpm, pad]` 出发）| `[..., pad, <acc_{-1}>]` |
| 2    | part0    | 注入 `bar_token`（measure 0 开始）| `[..., bar]` |
| 3    | part1    | **生成 acc_{bar}** | `[..., bar, <acc_bar>]` |
| 4    | part0    | 注入 `beat0_mel`（旋律第 0 拍）| `[..., mel_0]` |
| 5    | part1    | **生成 acc_0**（真正的第 0 拍伴奏）| `[..., mel_0, <acc_0>]` |
| ...  | ...      | ...       | ... |

**关键特征**：模型在看到任何旋律 token 之前，先生成了 `acc_{-1}`（基于纯粹的 `[BOS, ts, bpm, pad]`）。这个初始生成"激活"了模型的伴奏状态。

### 2.2 FakeRT 模式（`lekai_http_backend.py`）

FakeRT 的初始序列固定为：

```python
seq = [BOS, ts, bpm, pad]
```

第一次请求（`current_beat=0`，`generation_start_tick=0`）：

| 步骤 | 操作 | 累计序列 |
|------|------|---------|
| 初始化 | 固定初始 seq | `[BOS, ts, bpm, pad]` |
| context loop | `range(0, 0)` = 空 | 无变化 |
| generation, beat 0 | beat%4==0 → `bar, bar` | `[BOS, ts, bpm, pad, bar, bar]` |
| generation, beat 0 | 注入 mel_0 | `[BOS, ts, bpm, pad, bar, bar, mel_0]` |
| generation, beat 0 | **生成 acc_0** | `[..., bar, bar, mel_0, <acc_0>]` |

**关键差异**：FakeRT **跳过了 `acc_{-1}` 的生成**，直接用 melody 上下文生成 acc_0。模型从未见过"先于旋律的伴奏 token"。

### 2.3 差异量化

| 特征 | Offline | FakeRT | 是否等价 |
|------|---------|--------|---------|
| 初始 token 序列 | `[BOS, ts, bpm, pad]` | `[BOS, ts, bpm, pad]` | ✅ 相同 |
| beat 0 前的伴奏 | 生成 `acc_{-1}` | 无 | ❌ 不同 |
| beat 0 context | `[BOS, ts, bpm, pad, acc_{-1}, bar, bar, mel_0]` | `[BOS, ts, bpm, pad, bar, bar, mel_0]` | ❌ 差 1 token |
| 后续每拍 context | 含 `acc_{-1}` | 不含 | ❌ positional 偏移 |

**结论：这是 Bug 3（结构性提示词不匹配）**。FakeRT 的 prompt 结构与模型训练时的结构有 1 个 token 的偏移，导致所有后续 beat 的 positional encoding 不对齐。

---

## 3. Bug 3 影响分析

**根本原因**：PianoLLaMA 使用 `delay_beats=-1` 训练，意味着每个样本都有：
- 位置 0：`pad_marker`（旋律的第 0 个 slot 是空的）
- 位置 1：`acc_{-1}`（在旋律之前生成的伴奏 token）
- 位置 2 开始：`bar_token, mel_0, acc_0, mel_1, acc_1, ...`

FakeRT 跳过了位置 0 和 1 的生成，直接从位置 2 开始。这相当于让模型从一个它从未在训练中见过的 context 分布开始推理。

**对生成质量的影响**：
- 模型 "认为" 自己处于 seq 中间位置，而非开头
- 以 top_k=1 的贪心策略下，模型大概率选择 "empty" token（最安全的默认选项）
- 结果：FakeRT 所有请求中伴奏极度稀疏（Song 1：6 note_on vs Offline 的 233）

---

## 4. 关键发现

### 4.1 Bug 3 详情

- **位置**：`lekai_http_backend.py:_generate_with_interleaved_prompt()`，初始 seq 构造部分
- **症状**：FakeRT 缺少 `acc_{-1}` 的先验生成，导致 prompt 结构与训练时不一致
- **严重程度**：严重（导致模型几乎不生成任何伴奏）
- **修复方向**：在 FakeRT 初始请求（start_beat=0）时，在 context loop 前先生成一次 `acc_{-1}` from `[BOS, ts, bpm, pad]`，并将其加入 `_accompaniment_history`

### 4.2 Bug 4 详情（相关）

- **症状**：Offline Song 1 的伴奏从第 36 拍才开始（前 35 拍全为 empty beat）
- **原因**：模型以 top_k=1 时，在前 36 拍选择了 empty 作为最可能的 token
- **关联**：这与 Bug 3 无关——Offline 在此不受影响因为它有正确的 `acc_{-1}` token
- **修复方向**：可尝试 temperature > 1 或 top_k > 1 来允许更多非空生成（但会牺牲确定性）

---

## 5. 结论与下一步

- **结论**: Phase 2 分析发现 **Bug 3（结构性 prompt 不匹配）** 是 FakeRT 伴奏极度稀疏的根本原因。FakeRT 缺少 `delay_beats=-1` 所需的"先行伴奏 token"，导致模型在错误的 context 分布下运行。
- **下一步行动**: 进入 Step 4（Bug 修复）—— 实现 Bug 3 修复方案
- **阻塞项**: 无（修复方案已确定）

---

## 6. 附件

- Offline 生成代码：`src/streammuse/infrastructure/inference/lekai_model/model.py:generate_accompaniment()`（L61-282）
- FakeRT 生成代码：`src/streammuse/infrastructure/inference/lekai_http_backend.py:_generate_with_interleaved_prompt()`（L422-639）
- Bug 3 相关行：`lekai_http_backend.py:L455-460`（初始 seq 构造）和 `L466-488`（context loop）
