# Debug Round 2 - Step 1: 代码分析报告

**日期**: 2026-04-24  
**执行人**: Claude Code  

---

## 1. 当前代码状态分析

### 1.1 FakeRT (lekai_http_backend.py) 当前实现

**位置**: `_generate_with_interleaved_prompt` (lines 422-639)

**当前 prompt 构造流程**:

```python
# Line 455-460: 初始序列
seq = [BOS, ts, bpm, pad]

# Line 462-481: 部分修复的 Bug 3
if start_beat == 0 and not self._accompaniment_history:
    # 生成 acc_{-1}
    _primer_prompt = torch.cat(seq, dim=0)
    _acc_neg1_tokens = self._generate_part1_tokens_from_prompt(...)
    seq.append(torch.tensor(_acc_neg1_tokens, dtype=torch.long))
    # ❌ 问题：没有记录到 accompaniment_context_events

# Line 487-509: Context loop
for beat in range(start_beat, current_beat):
    if beat % 4 == 0:
        seq.append(bar)
        seq.append(bar)
    # encode melody
    seq.append(mel_tokens)
    # encode acc
    seq.append(acc_tokens)
```

**发现问题 1**: acc_{-1} 生成后没有正确记录到 accompaniment_context_events，
导致后续 context loop 中的 acc_tokens 可能不包含 acc_{-1} 的信息。

### 1.2 Offline (model.py) 实现

**位置**: `generate_accompaniment` (lines 61-284)

**prompt 构造流程** (delay_beats=-1, gt_prefix_beats=0):

```python
# Line 150-154: 初始序列
initial_tokens = [BOS, ts, bpm]  # 注意：没有 pad！

# Line 140-143: delay_beats=-1 处理
part0_beats_list.insert(0, torch.tensor([pad_marker], ...))
# part0: [pad, bar, mel_0, bar, mel_1, ...]
# part1: [bar, acc_0, acc_1, ..., pad]

# Line 179-230: 交错生成循环
# Iter 0: position=0, delay_beats<0 and part0_idx<=1:
#         注入 pad_token -> seq=[BOS,ts,bpm,pad]
#         position=1
# Iter 1: position=1, part1_idx=0 < gt_prefix=0? No, gt_prefix=0
#         生成 acc_{-1} from [BOS,ts,bpm,pad]
#         -> seq=[BOS,ts,bpm,pad,acc_{-1}]
# Iter 2: position=0, 注入 part0[0]=bar
#         -> seq=[BOS,ts,bpm,pad,acc_{-1},bar]
# Iter 3: position=1, 生成 acc_0 from [BOS,ts,bpm,pad,acc_{-1},bar]
# ...
```

### 1.3 关键差异对比

| Feature | Offline | FakeRT (当前) | 状态 |
|---------|---------|---------------|------|
| 初始 tokens | `[BOS,ts,bpm]` | `[BOS,ts,bpm,pad]` | ❌ 不同 |
| delay_beats 处理 | part0 前加 pad | 直接包含 pad | ⚠️ 等效 |
| acc_{-1} 生成 | ✅ 有 | ✅ 有 | ✅ 都有 |
| acc_{-1} 位置 | 第 4 位 | 第 4 位 | ✅ 相同 |
| bar token 位置 | 第 5 位 | 第 5 位 | ✅ 相同 |
| acc_{-1} 记录到 history | N/A (GT模式) | ❌ 未记录 | ❌ **关键差异** |

### 1.4 根因确认

**关键差异**: Offline 在 `gt_prefix_beats=0` 时不使用 GT，而是生成所有伴奏。
FakeRT 的 acc_{-1} 虽然生成了，但没有被记录到 `accompaniment_context_events`，
导致后续生成时模型看不到这个 token。

---

## 2. 需要修复的问题

### 问题 1: acc_{-1} 未记录到 history

**修复方案**:
```python
if start_beat == 0 and not self._accompaniment_history:
    # 生成 acc_{-1}
    _acc_neg1_tokens = self._generate_part1_tokens_from_prompt(...)
    seq.append(torch.tensor(_acc_neg1_tokens, dtype=torch.long))
    
    # ===== 新增：记录到 accompaniment_context_events =====
    # 创建 fake event 表示 acc_{-1}
    self._accompaniment_history.append({
        "type": "note_on",
        "pitch": -1,  # 标记为虚拟 note
        "tick": -4,   # beat -1
        "_tokens": _acc_neg1_tokens,  # 保存 tokens 供后续使用
        "_is_acc_neg1": True,
    })
```

### 问题 2: 需要添加详细日志

**日志格式**:
```python
print(f"[PROMPT_DEBUG] Mode: FakeRT")
print(f"[PROMPT_DEBUG] start_beat: {start_beat}")
print(f"[PROMPT_DEBUG] Initial seq (4 tokens): {[bos_token, ts, bpm, pad]}")
if start_beat == 0 and not self._accompaniment_history:
    print(f"[PROMPT_DEBUG] Generated acc_{{-1}}: {_acc_neg1_tokens}")
print(f"[PROMPT_DEBUG] Full prompt length: {len(torch.cat(seq, dim=0))}")
print(f"[PROMPT_DEBUG] First 20 tokens: {torch.cat(seq, dim=0)[:20].tolist()}")
```

---

## 3. 下一步行动

1. **修复问题 1**: 修改 lekai_http_backend.py，确保 acc_{-1} 被正确记录
2. **添加日志**: 在 FakeRT 和 Offline 中添加统一的 [PROMPT_DEBUG] 日志
3. **运行测试**: 选择 1 首歌曲，对比 prompt 结构
4. **验证修复**: 确认 note_on 数量与 Offline 一致

---

## 4. 附件

- 代码文件: `lekai_http_backend.py` (lines 422-639)
- 代码文件: `model.py` (lines 61-284)
