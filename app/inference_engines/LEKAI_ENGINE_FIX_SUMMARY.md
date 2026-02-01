# Lekai Inference Engine 修复总结

## 修复日期
2026-02-01

## 问题背景

`transformer_engine_lekai.py` 能够运行但生成结果较差，经过与参考实现 (`/home/cby/not_use/Advanced/generative_newtoken_improved_1_4_relative_track_RT_Accompaniment/`) 的对比分析，发现了以下关键问题。

---

## 核心问题分析

### 问题 1: 交错顺序错误 (最严重)

**训练数据结构** (`PianoDataset.py`, `delay_beats=-1`):
```
序列顺序: [BOS, ts, bpm, pad, acc_bar, mel_bar, acc0, mel0, acc1, mel1, ...]
交错模式: Acc -> Mel -> Acc -> Mel ...
```

**原推理代码**:
```python
# 错误的顺序
for b in range(start_beat, current_beat):
    seq.append(mel_tokens)   # Mel 在前
    seq.append(acc_tokens)   # Acc 在后
```
```
推理序列: [BOS, ts, bpm, pad, bar, bar, mel0, acc0, mel1, acc1, ...]
交错模式: Mel -> Acc -> Mel -> Acc ...  ❌
```

**结论**: 交错顺序完全相反，导致模型接收到错误的上下文。

### 问题 2: 生成时机错误

**训练数据中的因果关系**:
- 模型学习: 在看到 `mel[b-1]` 后预测 `acc[b]`
- 标签掩码: `part0 (melody)` 位置被掩码，只预测 `part1 (accompaniment)`

**原推理代码**:
```python
# 错误: 在生成 acc[b] 前先注入了 mel[b]
seq.append(mel_tokens_curr)  # 注入 mel[current_beat]
# 然后生成 acc[current_beat]
```

**正确逻辑**:
- 生成 `acc[b]` 时，context 应该以 `mel[b-1]` 结尾
- `mel[b]` 应该在生成 `acc[b]` 之后才加入 context

### 问题 3: 变量未定义 Bug

在空 token 处理分支中，`beat_start_tick` 在使用前未定义:
```python
if not valid_tokens:
    # beat_start_tick 未定义就被使用 ❌
    {"type": "note_off", "pitch": int(pitch), "tick": int(beat_start_tick)}
```

---

## 修复内容

### 修复 1: 调整历史上下文的交错顺序

**文件**: `transformer_engine_lekai.py`
**位置**: `generate_accompaniment()` 方法, sliding window 分支

```python
# 修复后: Acc 在前, Mel 在后
for b in range(start_beat, current_beat):
    if b % 4 == 0:
        seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))  # Acc Bar
        seq.append(torch.tensor([self.config.bar_token_id], dtype=torch.long))  # Mel Bar

    # FIXED ORDER: Acc tokens FIRST, then Mel tokens
    acc_tokens = self._get_tokens_for_beat(acc_notes_b, b, end_marker_id=self.tokenizer.end_marker_part1)
    seq.append(acc_tokens)

    mel_tokens = self._get_tokens_for_beat_pianoroll(mel_pr_b, end_marker_id=self.tokenizer.end_marker_part0)
    seq.append(mel_tokens)
```

### 修复 2: 移除生成前的 mel 注入

**修复前**:
```python
# 注入当前拍的 melody
mel_tokens_curr = self._get_tokens_for_beat_pianoroll(mel_pr_curr, ...)
seq.append(mel_tokens_curr)  # ❌ 不应该在生成前加入
# 然后生成
```

**修复后**:
```python
# 不注入 mel[current_beat]
# context 以 mel[current_beat-1] 结尾
# 直接生成 acc[current_beat]
input_ids = torch.cat(seq).unsqueeze(0).to(device)
```

### 修复 3: Stateful 模式的正确处理

```python
# 在生成 acc[b] 前，先注入 mel[b-1] (来自上一次调用)
prev_beat = current_beat - 1
if prev_beat >= 0:
    mel_pr_prev = self._get_mel_pianoroll_for_beat(prev_start_tick, prev_end_tick)
    mel_tokens_prev = self._get_tokens_for_beat_pianoroll(mel_pr_prev, ...)
    seq.append(mel_tokens_prev)

# Bar tokens if new measure
if current_beat % 4 == 0:
    seq.append(bar_token)
    seq.append(bar_token)

# 不添加 mel[current_beat]，直接生成 acc
```

### 修复 4: 变量定义位置

```python
# 在 if/else 分支前定义 beat_start_tick
pr = None
beat_start_tick = current_beat * self.ticks_per_beat  # 提前定义

if not valid_tokens:
    # 现在可以安全使用 beat_start_tick
    ...
```

---

## 修复后的数据流

### Sliding Window 模式 (beat b)

```
1. 构建历史 context:
   [BOS, ts, bpm, pad, bar, bar, acc0, mel0, acc1, mel1, ..., acc[b-1], mel[b-1]]

2. 添加 bar tokens (如果 b % 4 == 0):
   [..., mel[b-1], bar, bar]

3. 生成 acc[b]:
   模型预测 acc[b] (正确: 在 mel[b-1] 后预测)

4. mel[b] 存入历史，下次调用时使用
```

### Stateful 模式 (beat b, 从 beat b-1 继续)

```
1. KV cache 状态: 以 acc[b-1] 结尾

2. 注入 mel[b-1]:
   context 变为 [..., acc[b-1], mel[b-1]]

3. 添加 bar tokens (如果 b % 4 == 0)

4. 生成 acc[b]

5. mel[b] 存入历史，下次调用时使用
```

---

## 测试结果


### 测试文件: `input/mel/nyan_cat.mid`

| 项目 | 数值 |
|------|------|
| 旋律音符 | 326 notes |
| 生成伴奏 | 205 notes |
| 生成拍数 | 96 beats |

---

## 关键代码位置

| 修复项 | 文件位置 | 行号范围 |
|--------|---------|---------|
| 交错顺序 | `transformer_engine_lekai.py` | 429-453 |
| 生成时机 (sliding) | `transformer_engine_lekai.py` | 455-477 |
| 生成时机 (stateful) | `transformer_engine_lekai.py` | 479-523 |
| beat_start_tick 定义 | `transformer_engine_lekai.py` | 555-556 |

---

## 已知遗留问题

1. **前序空拍**: 旋律输入拍为空 (empty_marker)导致生成效果显著下降

---

---

## 参考文件

- 训练数据处理: `/home/cby/not_use/Advanced/generative_newtoken_improved_1_4_relative_track_RT_Accompaniment/PianoDataset.py`
- 参考推理实现: `/home/cby/not_use/Advanced/generative_newtoken_improved_1_4_relative_track_RT_Accompaniment/model.py` (generate_accompaniment 方法)
- 模型配置: `/home/cby/not_use/A_proj/StreamMUSE/lekai_model/config.py`
