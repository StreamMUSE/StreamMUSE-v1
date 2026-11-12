# Debug Round 2 - Step 2: 代码修复报告

**日期**: 2026-04-24  
**执行人**: Claude Code  

---

## 1. 修复内容

### 1.1 Bug 3 完整修复 (lekai_http_backend.py)

**问题**: acc_{-1} 生成后未记录到 accompaniment_history

**修复位置**: `src/streammuse/infrastructure/inference/lekai_http_backend.py`
**行号**: 462-510

**修复代码**:
```python
# 新增：记录 acc_{-1} 到 history
if start_beat == 0 and not self._accompaniment_history:
    # ... 生成 acc_{-1} ...
    
    # ===== ROUND2 FIX: Record acc_{-1} to history for context encoding =====
    self._accompaniment_history.append({
        "type": "note_on",
        "pitch": 60,
        "tick": -4,
        "velocity": 80,
        "_source": "acc_neg1_primer",
        "_tokens": _acc_neg1_tokens,
    })
```

**修复原因**: 确保后续 context loop 中的 accompaniment_context_events 包含 acc_{-1}，
使得 prompt 结构与 Offline 一致。

### 1.2 [PROMPT_DEBUG] 日志添加

#### lekai_http_backend.py

**位置 1**: 初始序列后 (lines 483-501)
```python
print(f"[PROMPT_DEBUG] {'='*60}")
print(f"[PROMPT_DEBUG] Mode: FakeRT")
print(f"[PROMPT_DEBUG] start_beat: {start_beat}, current_beat: {current_beat}")
print(f"[PROMPT_DEBUG] Initial seq (4 tokens): [...]")
```

**位置 2**: Context loop 完成后 (lines 538-548)
```python
print(f"[PROMPT_DEBUG] Full prompt length: {len(_full_prompt_for_debug)}")
print(f"[PROMPT_DEBUG] First 30 tokens: {...}")
print(f"[PROMPT_DEBUG] Tokens at positions 30-50: {...}")
```

#### model.py

**位置 1**: 初始 tokens 后 (lines 150-162)
```python
print(f"[PROMPT_DEBUG] Mode: Offline")
print(f"[PROMPT_DEBUG] delay_beats: {delay_beats}, gt_prefix_beats: {gt_prefix_beats}")
print(f"[PROMPT_DEBUG] Initial tokens (3 tokens): [...]")
```

**位置 2**: 第一次生成前 (lines 243-252)
```python
print(f"[PROMPT_DEBUG] First generation step (part1_idx={part1_idx})")
print(f"[PROMPT_DEBUG] Full prompt length: {generated.shape[1]}")
print(f"[PROMPT_DEBUG] First 30 tokens: {...}")
```

---

## 2. 修改的文件清单

| 文件 | 修改类型 | 行号范围 |
|------|---------|---------|
| `lekai_http_backend.py` | Bug 修复 + 日志 | 462-510, 538-548 |
| `model.py` | 日志 | 150-162, 243-252 |

---

## 3. 下一步行动

1. **重启 Server** 使修复生效
2. **运行测试**: 选择 Song 1，分别运行 Offline 和 FakeRT
3. **收集日志**: 提取 [PROMPT_DEBUG] 日志
4. **对比分析**: 对比 prompt 结构是否一致
5. **验证结果**: 检查 note_on 数量是否一致

---

## 4. 命令备忘

```bash
# 重启 Server
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=cuda \
LEKAI_DTYPE=auto \
LEKAI_RT_TEMPERATURE=0.0 \
LEKAI_RT_TOP_K=1 \
LEKAI_RT_TOP_P=0.0 \
uv run python -m streammuse.infrastructure.inference.server_lekai

# 运行 FakeRT (Song 1)
uv run python scripts/run_lekai_fake_realtime.py \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --output-dir output/debug_round2/fake_rt_v4 \
  --server-url http://127.0.0.1:8001/generate_accompaniment \
  --generation-interval-ticks 4 \
  --generation-length-frames 4 \
  --max-ticks 256 \
  2>&1 | tee logs/fakert_round2.log

# 运行 Offline (Song 1)
CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_lekai_offline.py \
  --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  --npz-dir prompts/inputs_lekai/npz \
  --output-dir output/debug_round2/offline \
  --device cuda \
  --dtype auto \
  --temperature 0.0 \
  --top-k 1 \
  --top-p 0.0 \
  --condition-idx 0 \
  --gt-prefix-beats 0 \
  2>&1 | tee logs/offline_round2.log
```
