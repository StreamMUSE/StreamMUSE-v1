# Debug Round 2: Fake Realtime 复现 Offline 行为

**日期**: 2026-04-23 ~ 2026-04-24  
**目标**: 让 Fake Realtime 的输出与 Offline 完全一致（即使 Offline 有 Bug）  
**核心原则**: 只关注一致性，不关注正确性  

---

## 第一轮关键发现回顾

### 已确认的 Root Cause

| Bug | 描述 | 修复状态 | 影响 |
|-----|------|---------|------|
| Bug 1 | `start_tick <= 0` 误回退 | ✅ 已修复 | 首次请求走真实模型 |
| Bug 2 | BPM 未传递 | ✅ 已修复 | BPM 正确 |
| Bug 3 | **缺少 `acc_{-1}` 先行 token** | ⚠️ 部分修复 | **prompt 结构偏移** |
| Bug 4 | Offline Song 2/3 全空 | ❌ 未修复 | Offline 本身有问题 |

### Bug 3 的核心问题（来自 Phase 2 Report）

**Offline 的 prompt 结构**（delay_beats=-1）：
```
[BOS, ts, bpm, pad, acc_{-1}, bar, bar, mel_0, acc_0, mel_1, acc_1, ...]
```

**FakeRT 的 prompt 结构**（修复前）：
```
[BOS, ts, bpm, pad, bar, bar, mel_0, acc_0, mel_1, acc_1, ...]
       ↑ 少了 acc_{-1}！
```

**后果**: 所有后续 token 的 positional encoding 偏移 1，模型处于 OOD 状态。

### 第一轮修复后的结果

| 歌曲 | Offline note_on | FakeRT v3 note_on | Match Rate |
|------|-----------------|-------------------|------------|
| 1    | 233             | 14                | 0.0%       |
| 2    | 0               | 37                | 0.0%       |
| 3    | 0               | 17                | 0.0%       |
| 4    | 59              | 16                | 0.0%       |
| 5    | 57              | 70                | 5.93%      |

**结论**: Bug 3 修复方向正确（note_on 增加），但 match rate 仍极低。

---

## Round 2 核心任务

### 任务 1: 完全复刻 Offline 的 prompt 构造

**目标**: 让 FakeRT 的 prompt 结构与 Offline **逐 token 一致**。

#### 1.1 详细分析 Offline 的 prompt 构造

从 `model.py:generate_accompaniment` 提取完整逻辑：

```python
# Step 1: 初始 tokens
initial_tokens = [
    BOS_TOKEN,           # 1
    time_sig_token,      # ts
    bpm_token,           # bpm
]

# Step 2: 处理 delay_beats=-1
if delay_beats < 0:
    # part0 前面加 pad
    part0_padded = [pad_token] + part0_beats  
    # part1 后面加 pad
    part1_padded = part1_beats + [pad_token]

# Step 3: 交错生成
for beat in range(num_beats):
    # position 0: 注入 part0 (melody)
    # position 1: 生成 part1 (acc)
    
# 实际 prompt 序列（前 20 个 token）:
# [BOS, ts, bpm, pad, acc_{-1}, bar, bar, mel_0, acc_0, mel_1, acc_1, ...]
```

#### 1.2 对比 FakeRT 的 prompt 构造

从 `lekai_http_backend.py:_generate_with_interleaved_prompt` 提取：

```python
# 当前实现（Bug 3 部分修复后）:
seq = [BOS, ts, bpm, pad]

# context loop
for beat in range(start_beat, current_beat):
    if beat % 4 == 0:
        seq.append(bar)
        seq.append(bar)
    # encode melody
    seq.append(mel_tokens)
    # encode acc (from history)
    seq.append(acc_tokens)

# generation loop
# ...

# 实际 prompt 序列:
# [BOS, ts, bpm, pad, bar, bar, mel_0, acc_0, ...]
#              ↑ 这里应该是 acc_{-1}，但现在是 bar！
```

#### 1.3 找出差异点

| Position | Offline | FakeRT | 差异 |
|----------|---------|--------|------|
| 3 | `pad` | `pad` | ✅ 相同 |
| 4 | `acc_{-1}` | `bar` | ❌ **不同** |
| 5 | `bar` | `bar` | ✅ 相同 |
| 6 | `bar` | `mel_0` | ❌ **不同** |
| 7 | `mel_0` | `acc_0` | ❌ **不同** |

**问题**: FakeRT 的 `bar` 位置比 Offline 早 1 个 token。

#### 1.4 修复方案

在 FakeRT 的 `_generate_with_interleaved_prompt` 中，当 `start_beat == 0` 时：

```python
# 修复后的逻辑
seq = [BOS, ts, bpm, pad]

# ===== 新增：生成 acc_{-1} =====
if start_beat == 0:
    # 从 [BOS, ts, bpm, pad] 生成 acc_{-1}
    prompt_for_acc_neg1 = torch.cat(seq, dim=0).unsqueeze(0).to(device)
    acc_neg1_tokens = self._generate_part1_tokens_from_prompt(
        prompt_for_acc_neg1,
        temperature=self._get_temperature(),
        top_k=self._get_top_k(),
        top_p=self._get_top_p(),
        repetition_penalty=1.2,
    )
    seq.append(torch.tensor(acc_neg1_tokens, dtype=torch.long))
    # 记录到 accompaniment_history
    self._accompaniment_history.append({
        'tick': -4,  # 第 -1 拍
        'tokens': acc_neg1_tokens,
    })

# ===== 原有 context loop =====
for beat in range(start_beat, current_beat):
    ...
```

---

### 任务 2: 逐 token 对比验证

**目标**: 验证修复后 prompt 是否逐 token 一致。

#### 2.1 添加详细日志

在 `model.py` 和 `lekai_http_backend.py` 中添加完全相同的日志格式：

```python
# 统一日志格式
print(f"[PROMPT_DEBUG] {'='*60}")
print(f"[PROMPT_DEBUG] Mode: {'Offline'/'FakeRT'}")
print(f"[PROMPT_DEBUG] Generation start tick: {gen_start_tick}")
print(f"[PROMPT_DEBUG] Full prompt tokens: {prompt.tolist()}")
print(f"[PROMPT_DEBUG] Prompt length: {len(prompt)}")
print(f"[PROMPT_DEBUG] First 20 tokens: {prompt[:20].tolist()}")
print(f"[PROMPT_DEBUG] Tokens at positions 3-10: {prompt[3:10].tolist()}")
```

#### 2.2 运行并对比

```bash
# 运行 Offline
python scripts/run_lekai_offline.py ... > logs/offline_prompt.log 2>&1

# 运行 FakeRT
python scripts/run_lekai_fake_realtime.py ... > logs/fakert_prompt.log 2>&1

# 提取并对比 prompt
python scripts/extract_and_compare_prompts.py \
  --offline logs/offline_prompt.log \
  --fakert logs/fakert_prompt.log \
  --output logs/prompt_comparison.txt
```

#### 2.3 验证标准

- [ ] Position 0-3: 必须完全一致 (`[BOS, ts, bpm, pad]`)
- [ ] Position 4: 必须包含 `acc_{-1}` token
- [ ] Position 5-6: 必须包含 `bar, bar`
- [ ] Position 7+: melody 和 acc 交替

---

### 任务 3: 借鉴老系统实现

**参考文件**: `/data/home/bowenzheng/mbzuai-projects/StreamMUSE/app/inference_engines/transformer_engine_lekai.py`

#### 3.1 提取老系统的关键逻辑

```python
# 从老系统提取 prompt 构建代码
class LekaiTransformerEngine:
    def generate(self, melody_events, ...):
        # 1. 构建 melody pianoroll
        melody_pr = self._events_to_pianoroll(melody_events)
        
        # 2. 分 beat tokenize
        melody_beats = []
        for beat in range(num_beats):
            beat_pr = melody_pr[:, :, beat*4:(beat+1)*4]
            beat_tokens = self.tokenizer.encode(beat_pr)
            melody_beats.append(beat_tokens)
        
        # 3. 构建 prompt (含 delay_beats=-1 处理)
        prompt = self._build_prompt_with_delay(melody_beats, delay_beats=-1)
        
        # 4. 生成
        output = self.model.generate(prompt)
        
        return output
```

#### 3.2 对比新老系统差异

创建对比表格：

| 步骤 | 老系统 | 新系统 (当前) | 状态 |
|------|--------|---------------|------|
| 初始 seq | `[BOS, ts, bpm, pad]` | 相同 | ✅ |
| acc_{-1} 生成 | ✅ 有 | ⚠️ 部分 | 需完全修复 |
| bar token 插入位置 | pad 后 | acc_{-1} 后 | 需对齐 |
| mel/acc 交错 | 正确 | 偏移 1 | 需修复 |

---

### 任务 4: 验证一致性

#### 4.1 运行修复后的 FakeRT v4

```bash
for f in 1 2 3 4 5; do
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path prompts/inputs_lekai/mel/${f}.mid \
    --output-dir output/debug_round2/fake_rt_v4 \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --max-ticks 256
done
```

#### 4.2 对比结果

| 歌曲 | Offline note_on | FakeRT v4 note_on | 差异 |
|------|-----------------|-------------------|------|
| 1    | 233             | ?                 | ?    |
| 2    | 0               | ?                 | ?    |
| 3    | 0               | ?                 | ?    |
| 4    | 59              | ?                 | ?    |
| 5    | 57              | ?                 | ?    |

**目标**: FakeRT v4 的 note_on 数量与 Offline **完全一致**。

---

## 执行计划

### Phase 1: 代码修复 (2 hours)

- [ ] **Task 1.1**: 在 `_generate_with_interleaved_prompt` 中实现完整的 `acc_{-1}` 生成逻辑
- [ ] **Task 1.2**: 确保 bar token 插入位置与 Offline 一致
- [ ] **Task 1.3**: 添加统一的 `[PROMPT_DEBUG]` 日志格式

### Phase 2: 日志收集 (1 hour)

- [ ] **Task 2.1**: 选择 1 首歌曲（如 Song 1），运行 Offline，收集 prompt 日志
- [ ] **Task 2.2**: 运行 FakeRT v4，收集 prompt 日志
- [ ] **Task 2.3**: 对比两个日志，验证 prompt 是否逐 token 一致

### Phase 3: 迭代修复 (2-4 hours)

- [ ] **Task 3.1**: 如果不一致，分析差异位置
- [ ] **Task 3.2**: 修复差异
- [ ] **Task 3.3**: 重新运行，验证
- [ ] **Task 3.4**: 重复直到 prompt 完全一致

### Phase 4: 全量验证 (1 hour)

- [ ] **Task 4.1**: 运行全部 5 首歌曲
- [ ] **Task 4.2**: 对比 note_on 数量
- [ ] **Task 4.3**: 对比 MIDI 文件（使用 scripts/compare_midi.py）

---

## 报告模板

### Round 2 每日报告

```markdown
# Debug Round 2 Daily Report - YYYY-MM-DD

## 完成任务
- [x] Task X: ...

## 关键发现
### Prompt 对比结果
| Position | Offline | FakeRT | Match |
|----------|---------|--------|-------|
| 0-3      | [1,2,3,4] | [1,2,3,4] | ✅ |
| 4        | acc_{-1} | ? | ? |

### 修复内容
- 修改文件: ...
- 修改内容: ...

## 明日计划
- [ ] Task Y: ...
```

### Round 2 最终报告

```markdown
# Debug Round 2 Final Report

## 修复的 Bug
| Bug | 描述 | 修复方案 | 状态 |
|-----|------|---------|------|
| Bug 3 完整修复 | acc_{-1} 生成 + bar token 位置 | ... | ✅ |

## 验证结果
| 歌曲 | Offline | FakeRT v4 | Match |
|------|---------|-----------|-------|
| 1    | 233     | 233       | 100%  |
| ...  | ...     | ...       | ...   |

## 结论
- FakeRT 与 Offline 一致性: XX%
- 主要差异: ...
```

---

## 关键代码修改位置

### 修改 1: lekai_http_backend.py

```python
# 在 _generate_with_interleaved_prompt 方法中

# 当前代码（部分修复）
seq: List[torch.Tensor] = [
    torch.tensor([bos_token], dtype=torch.long),
    torch.tensor([time_sig_token], dtype=torch.long),
    torch.tensor([bpm_token], dtype=torch.long),
    torch.tensor([pad_token], dtype=torch.long),
]

# ===== 新增：完整实现 delay_beats=-1 逻辑 =====
if start_beat == 0:
    # 1. 生成 acc_{-1}
    prompt_for_acc_neg1 = torch.cat(seq, dim=0)
    acc_neg1_tokens = self._generate_part1_tokens_from_prompt(
        prompt_for_acc_neg1,
        temperature=self._get_temperature(),
        top_k=self._get_top_k(),
        top_p=self._get_top_p(),
        repetition_penalty=1.2,
    )
    seq.append(torch.tensor(acc_neg1_tokens, dtype=torch.long))
    
    # 2. 记录到 history
    # ...

# 3. 继续原有 context loop
for beat in range(start_beat, current_beat):
    # 注意：beat 0 时，前面已经有 acc_{-1} 了
    ...
```

### 修改 2: model.py (添加日志)

```python
# 在 generate_accompaniment 方法中

print(f"[PROMPT_DEBUG] Offline Mode")
print(f"[PROMPT_DEBUG] delay_beats: {delay_beats}")
print(f"[PROMPT_DEBUG] Initial tokens: {[self.bos_token, time_sig_token, bpm_token]}")

# 在生成循环前
print(f"[PROMPT_DEBUG] Full prompt before generation: {generated[0].tolist()}")
```

---

**目标**: FakeRT v4 与 Offline **逐 token 一致**  
**成功标准**: 5 首歌曲的 note_on 数量完全一致  
**时间预估**: 1-2 天
