# Offline vs Fake Realtime Inference Engine Debug Plan

**日期**: 2026-04-23  
**目标**: 通过对比 Offline 和 Fake Realtime 的生成结果，定位 Inference Engine 的 Bug  
**核心假设**: 在特定设置下，两者应该产生一致的结果

---

## 1. 理论基础：为什么两者应该一致

### 1.1 数据流对比

```
Offline Mode:
NPZ File → PianoDataset.__getitem__ → tokens → model.generate → MIDI
                ↓
            [melody tokens]
            [ground truth acc tokens]

Fake Realtime Mode:
MIDI File → MidiConverter → events → tokens → model.generate → MIDI
                ↓
            [melody events]
            [generated acc events]
```

### 1.2 理想一致性条件

| 条件 | Offline | Fake Realtime | 说明 |
|------|---------|---------------|------|
| 输入 melody 完全相同 | ✅ | ✅ | 前提：从同一来源提取 |
| Tokenizer 参数相同 | ✅ | ✅ | patch_h=1, patch_w=4 |
| 模型 checkpoint 相同 | ✅ | ✅ | 使用同一 .safetensors |
| 生成温度/采样参数相同 | ✅ | ✅ | temperature, top_k, top_p |
| 生成长度等于间隔 (1拍生成1拍) | ✅ | 需设置 | interval=4, length=4 |
| Deterministic 采样配置 | 需设置 | 需设置 | 当前实现不支持 temperature=0；推荐 temperature=1.0, top_k=1, top_p=1.0 |
| 历史累积正确 | N/A (一次性) | 需验证 | fake rt 的 melody history 必须完整 |

### 1.3 关键观察

**如果上述条件都满足，但结果不同 → 存在 Bug**

可能的 Bug 位置：
1. Tokenization 路径不一致 (MIDI → tokens vs NPZ → tokens)
2. History accumulation 错误 (fake rt 漏掉了某些 melody notes)
3. Time alignment 错误 (tick offset 计算错误)
4. Model inference 参数不一致 (hidden state, cache 等)
5. Post-processing 差异 (tokens → MIDI 路径不同)

---

## 2. 测试设计

### 2.1 测试数据准备

#### Step 1: 测试数据确认

测试数据位于 `prompts/inputs_lekai/`，共 **5 首曲子**：

```bash
$ ls prompts/inputs_lekai/mel/
1.mid  2.mid  3.mid  4.mid  5.mid

$ ls prompts/inputs_lekai/npz/
1.npz  2.npz  3.npz  4.npz  5.npz
```

**文件对应关系**：
| MIDI File | NPZ File | 编号 |
|-----------|----------|------|
| `prompts/inputs_lekai/mel/1.mid` | `prompts/inputs_lekai/npz/1.npz` | 001 |
| `prompts/inputs_lekai/mel/2.mid` | `prompts/inputs_lekai/npz/2.npz` | 002 |
| `prompts/inputs_lekai/mel/3.mid` | `prompts/inputs_lekai/npz/3.npz` | 003 |
| `prompts/inputs_lekai/mel/4.mid` | `prompts/inputs_lekai/npz/4.npz` | 004 |
| `prompts/inputs_lekai/mel/5.mid` | `prompts/inputs_lekai/npz/5.npz` | 005 |

**验证文件完整性**：
```bash
# 检查文件存在且非空
for i in 1 2 3 4 5; do
  if [ ! -s "prompts/inputs_lekai/mel/${i}.mid" ]; then
    echo "ERROR: ${i}.mid missing or empty"
  fi
  if [ ! -s "prompts/inputs_lekai/npz/${i}.npz" ]; then
    echo "ERROR: ${i}.npz missing or empty"
  fi
done
echo "All test files verified."
```

#### Step 2: 数据预处理验证

确保 MIDI 和 NPZ 中的 melody 内容一致：

```python
# scripts/verify_test_data.py
import numpy as np
import mido
from pathlib import Path

def verify_pair(midi_path, npz_path):
    """验证 MIDI 和 NPZ 中的 melody 是否一致"""
    # 从 MIDI 提取 notes
    mid = mido.MidiFile(midi_path)
    midi_notes = []
    current_time = 0
    for msg in mid.tracks[0]:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            midi_notes.append((current_time, msg.note))
    
    # 从 NPZ 提取 notes
    save_dict = np.load(npz_path, allow_pickle=True)
    metadata = save_dict['metadata'].item()
    
    # 简化的 note 计数验证
    print(f"\n{Path(midi_path).name} vs {Path(npz_path).name}:")
    print(f"  MIDI notes: {len(midi_notes)}")
    print(f"  NPZ measures: {metadata['num_measures']}")
    print(f"  NPZ BPM: {metadata['bpm']}")
    return len(midi_notes) > 0

# 验证所有 5 首
for i in range(1, 6):
    midi_path = f"prompts/inputs_lekai/mel/{i}.mid"
    npz_path = f"prompts/inputs_lekai/npz/{i}.npz"
    verify_pair(midi_path, npz_path)
```

**预期输出**：
```
1.mid vs 1.npz:
  MIDI notes: XX
  NPZ measures: YY
  NPZ BPM: 120
...
```

#### Step 2: 运行对比实验

**Config A: Offline (Baseline)**

```bash
# 推荐一次性跑 all，并按输出文件 stem 配对（不要假设 idx 与文件名序号一致）
CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_lekai_offline.py \
  --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  --npz-dir prompts/inputs_lekai/npz \
  --output-dir output/debug/offline \
  --device cuda \
  --dtype auto \
  --condition-idx all \
  --gt-prefix-beats 0 \
  --temperature 1.0 \
  --top-k 1 \
  --top-p 1.0 \
  --repetition-penalty 1.2
```

说明：`PianoDataset` 当前通过 `os.listdir()` 构建样本列表，顺序不保证稳定。对齐 Offline/FakeRT 时请按输出文件中的歌曲 stem（例如 `1`、`2`）配对，不要假设 `condition-idx=0` 对应 `1.npz`。

**Config B: Fake Realtime "等价 Offline"模式 (关键配置)**

这是最关键的测试配置：**generation_interval=4, generation_length=4, deterministic sampling (`temperature=1.0, top_k=1, top_p=1.0`)**

```bash
# 先启动 server (启用 deterministic sampling)
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=cuda \
LEKAI_DTYPE=auto \
LEKAI_RT_TEMPERATURE=1.0 \
LEKAI_RT_TOP_K=1 \
LEKAI_RT_TOP_P=1.0 \
LEKAI_RT_REPETITION_PENALTY=1.2 \
uv run python -m streammuse.infrastructure.inference.server_lekai

# 再运行 client (使用 prompts/inputs_lekai/mel/ 下的 5 首曲子)
# 每首曲子运行一次，对比用
for f in 1 2 3 4 5; do
  echo "Processing: prompts/inputs_lekai/mel/${f}.mid"
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path prompts/inputs_lekai/mel/${f}.mid \
    --output-dir output/debug/fake_rt_equivalent \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --max-ticks 256
done

# 输出文件命名: output/debug/fake_rt_equivalent/${f}_fake_realtime_combined.mid
```

**为什么这个配置应该和 Offline 完全一致？**

| 维度 | Offline | Fake Realtime (interval=4, length=4) |
|------|---------|--------------------------------------|
| 生成粒度 | 整首一次生成 | 每 4 ticks (1拍) 生成一次 |
| 每次生成长度 | 全曲 | 4 ticks (1拍) |
| Prompt 累积 | 完整 melody | 逐拍累积 |
| 上下文窗口 | 全曲可见 | 历史 context 累积后等效 |
| 采样参数 | temperature=1.0, top_k=1, top_p=1.0 | temperature=1.0, top_k=1, top_p=1.0 |

**关键逻辑**: 当 generation_length == generation_interval 时，每一拍只生成下一拍的内容，没有冗余和重叠，和 offline 的逐拍生成逻辑一致。

**Config C: Fake Realtime "Overlap"模式 (用于对比)**

```bash
# length > interval，有重叠生成 (使用 prompts/inputs_lekai/mel/ 下的 5 首曲子)
# 这种模式下结果会和 offline 不同（由于增量更新机制）
for f in 1 2 3 4 5; do
  echo "Processing: prompts/inputs_lekai/mel/${f}.mid"
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path prompts/inputs_lekai/mel/${f}.mid \
    --output-dir output/debug/fake_rt_overlap \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 8 \
    --max-ticks 256
done
```

---

## 3. 排查步骤 (从输入到输出)

### Phase 1: 输入层对比 (MIDI vs NPZ)

**目标**: 验证两种模式的输入 melody 是否完全一致

#### Check 1.1: MIDI 文件内容对比

```python
# scripts/compare_inputs.py
import mido
import numpy as np

def midi_to_note_list(mid_path):
    """提取 MIDI 中的 note_on 事件列表"""
    mid = mido.MidiFile(mid_path)
    notes = []
    current_time = 0
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append({
                    'tick': current_time,
                    'pitch': msg.note,
                    'velocity': msg.velocity
                })
    return notes

def npz_to_note_list(npz_path):
    """从 NPZ 中提取 melody notes"""
    save_dict = np.load(npz_path, allow_pickle=True)
    metadata = save_dict['metadata'].item()
    
    # 提取 pianoroll
    num_measures = metadata['num_measures']
    all_measures = []
    for i in range(num_measures):
        measure = save_dict[f'measure_{i}']
        all_measures.append(measure)
    
    full_pianoroll = np.concatenate(all_measures, axis=2)
    melody_pianoroll = full_pianoroll[:2]  # sustain + onset
    
    # pianoroll → notes
    notes = []
    onset_roll = melody_pianoroll[1]
    sustain_roll = melody_pianoroll[0]
    
    for pitch_idx in range(88):
        pitch = pitch_idx + 21
        onset_positions = np.where(onset_roll[pitch_idx] > 0)[0]
        for onset_pos in onset_positions:
            # 找到音符长度
            end_pos = onset_pos + 1
            while end_pos < sustain_roll.shape[1] and sustain_roll[pitch_idx, end_pos] > 0:
                end_pos += 1
            notes.append({
                'tick': int(onset_pos),
                'pitch': int(pitch),
                'duration': int(end_pos - onset_pos)
            })
    
    return sorted(notes, key=lambda x: x['tick'])

# 对比 (使用 prompts/inputs_lekai 下的测试数据)
mid_notes = midi_to_note_list('prompts/inputs_lekai/mel/1.mid')
npz_notes = npz_to_note_list('prompts/inputs_lekai/npz/1.npz')

print(f"MIDI notes: {len(mid_notes)}")
print(f"NPZ notes: {len(npz_notes)}")

# 详细对比
for i, (mn, nn) in enumerate(zip(mid_notes, npz_notes)):
    if mn['pitch'] != nn['pitch'] or abs(mn['tick'] - nn['tick']) > 1:
        print(f"Mismatch at note {i}:")
        print(f"  MIDI: tick={mn['tick']}, pitch={mn['pitch']}")
        print(f"  NPZ:  tick={nn['tick']}, pitch={nn['pitch']}")
```

**预期结果**: 如果提取正确，note 数量和 pitch 应该一致，tick 可能有微小差异（量化误差）。

**如果不同** → **Bug 在输入层**: MIDI 和 NPZ 的提取逻辑不一致

---

### Phase 2: Tokenization 层对比

**目标**: 验证 MIDI → tokens 和 NPZ → tokens 是否一致

#### Check 2.1: 添加 Token 导出日志

修改 `lekai_http_backend.py`:

```python
def _encode_beat_tokens(self, events, beat_start_tick, active_pitches, end_marker):
    """添加调试日志"""
    beat_end_tick = beat_start_tick + TIMESTEPS_PER_BEAT
    beat_pr = self._converter.events_to_pianoroll(
        events=events,
        start_tick=beat_start_tick,
        end_tick=beat_end_tick,
        active_pitches=active_pitches,
    )
    
    # DEBUG: 导出 pianoroll
    print(f"[DEBUG] Beat {beat_start_tick//4} pianoroll shape: {beat_pr.shape}")
    print(f"[DEBUG] pianoroll sum: {beat_pr.sum()}")
    
    tokens_matrix = self._tokenizer.image_to_patch_tokens(beat_pr, strict_mode=True)
    print(f"[DEBUG] tokens_matrix: {tokens_matrix}")
    
    compressed = self._tokenizer.compress_tokens(tokens_matrix, end_marker=end_marker)
    print(f"[DEBUG] compressed tokens: {compressed}")
    
    return torch.tensor(compressed, dtype=torch.long), next_active
```

修改 `inference.py` (offline mode):

```python
# 在 generate_accompaniment 中添加相同日志
print(f"[DEBUG] part0_beat_tokens: {[t.tolist() for t in part0_beats_list]}")
print(f"[DEBUG] part1_beat_tokens (GT): {[t.tolist() for t in part1_beats_gt_list]}")
```

#### Check 2.2: 对比 Token 序列

```python
# 对比同一 beat 的 tokens
# 从 offline 日志提取 beat 0 的 tokens
offline_beat0_tokens = [1, 2, 3, ...]  # 从日志复制

# 从 fake realtime 日志提取 beat 0 的 tokens  
fake_rt_beat0_tokens = [1, 2, 3, ...]  # 从日志复制

if offline_beat0_tokens == fake_rt_beat0_tokens:
    print("✓ Tokens match!")
else:
    print("✗ Tokens mismatch!")
    print(f"Offline: {offline_beat0_tokens}")
    print(f"FakeRT:  {fake_rt_beat0_tokens}")
```

**如果不同** → **Bug 在 Tokenization**: 
- 可能是 pianoroll 转换不一致
- 可能是 patch_h/patch_w 参数不一致
- 可能是 quantization 方式不同

---

### Phase 3: Model Inference 层对比

**目标**: 验证模型输入和输出是否一致

#### Check 3.1: 对比 Model Input

```python
# 在 PianoLLaMA.generate_accompaniment 中添加
print(f"[DEBUG] initial_tokens: {initial_tokens}")
print(f"[DEBUG] part0_beats count: {len(part0_beats_list)}")
print(f"[DEBUG] Full prompt sequence: {generated[0].tolist()[:50]}...")  # 前50个token
```

#### Check 3.2: 对比 Model Output

```python
# 在 generation loop 后添加
print(f"[DEBUG] Generated sequence length: {len(generated[0])}")
print(f"[DEBUG] Generated tokens (first 50): {generated[0][:50].tolist()}")
print(f"[DEBUG] part1_beats_generated: {part1_beats_generated}")
```

#### Check 3.3: 使用 Deterministic Sampling (top_k=1)

**这是保证一致性的关键！**

**原理**: 设置 `top_k=1` 后，每步只保留一个候选 token，采样结果退化为确定性选择。

**注意**: 当前实现中存在 `logits / temperature`，`temperature=0` 会导致数值异常，因此不应使用 `temperature=0`。

**配置方式**:

```python
# 方式 1: 通过环境变量 (server 端)
LEKAI_RT_TEMPERATURE=1.0
LEKAI_RT_TOP_K=1
LEKAI_RT_TOP_P=1.0
LEKAI_RT_REPETITION_PENALTY=1.2

# 方式 2: 修改 lekai_http_backend.py
self._generate_part1_tokens_from_prompt(
    prompt_tokens,
    temperature=1.0,
    top_k=1,
    top_p=1.0,
    repetition_penalty=1.2,
)
```

**验证 deterministic**:
```python
# 同一输入运行两次，结果应该完全一致
result1 = generate(tokens)
result2 = generate(tokens)
assert result1 == result2, "Not deterministic!"
```

**如果 deterministic 模式下输出仍不同** → **Bug 在 Model Inference**:
- 可能是 hidden state 初始化不同
- 可能是 KV cache 使用方式不同
- 可能是 prompt construction 不同

---

### Phase 4: Post-processing 层对比

**目标**: 验证 tokens → MIDI 的转换是否一致

#### Check 4.1: 对比解码后的 Pianoroll

```python
# 在 tokens_to_midi / process_part_beats_to_pianoroll 中添加
print(f"[DEBUG] Decompressed matrix shape: {decompressed_matrix.shape}")
print(f"[DEBUG] Pianoroll shape: {pianoroll.shape}")
print(f"[DEBUG] Pianoroll sum: {pianoroll.sum()}")
```

#### Check 4.2: 对比最终 MIDI Events

```python
# 对比生成的 MIDI 文件
import mido

def extract_accompaniment_events(mid_path):
    mid = mido.MidiFile(mid_path)
    events = []
    for track in mid.tracks:
        current_time = 0
        track_name = ""
        for msg in track:
            current_time += msg.time
            if msg.type == 'track_name':
                track_name = str(getattr(msg, 'name', ''))
                continue

            # 只比较伴奏轨道：Offline 常见为 "Part1 (Accompaniment)"，
            # FakeRT 常见为 "Accompaniment"。
            if msg.type in ['note_on', 'note_off'] and (
                'Accompaniment' in track_name or 'Part1' in track_name
            ):
                events.append({
                    'tick': current_time,
                    'type': msg.type,
                    'pitch': msg.note,
                    'velocity': msg.velocity
                })
    return events

# 对比输出文件 (按歌曲 stem 配对，例如 song=1)
# Offline output: output/debug/offline/xxx_1_generated.mid (idx 前缀不稳定)
# FakeRT output: output/debug/fake_rt_equivalent/1_fake_realtime_combined.mid
offline_events = extract_accompaniment_events('output/debug/offline/xxx_1_generated.mid')
fakert_events = extract_accompaniment_events('output/debug/fake_rt_equivalent/1_fake_realtime_combined.mid')

print(f"Offline events: {len(offline_events)}")
print(f"FakeRT events: {len(fakert_events)}")

# 详细对比
for i, (oe, fe) in enumerate(zip(offline_events, fakert_events)):
    if oe != fe:
        print(f"Mismatch at {i}:")
        print(f"  Offline: {oe}")
        print(f"  FakeRT:  {fe}")
```

**如果不同** → **Bug 在 Post-processing**:
- 可能是 tokenizer.decompress_tokens 不一致
- 可能是 patch_tokens_to_image 不一致
- 可能是 pianoroll_to_events 不一致

---

## 4. 特定 Bug 排查清单

### Bug Type A: Melody History 丢失

**症状**: Fake realtime 生成的伴奏和 melody 不和谐

**检查点**:
```python
# 在 lekai_http_backend._generate_with_interleaved_prompt 中添加
print(f"[DEBUG] Melody history size: {len(self._melody_history)}")
print(f"[DEBUG] Melody history sample: {self._melody_history[:5]}")
```

**修复**: 确保 input worker 正确将 events 添加到 melody_history

### Bug Type B: Tick Offset 错误

**症状**: 伴奏整体提前或延后若干 ticks

**检查点**:
```python
# 检查 generation_start_tick 的计算
print(f"[DEBUG] tick={tick}, generation_start_tick={generation_start_tick}")
print(f"[DEBUG] Returned note ticks: {[e['tick'] for e in generated_notes[:5]]}")
```

**修复**: 检查 tick 对齐逻辑

### Bug Type C: 重复生成/覆盖

**症状**: 某些 beats 的伴奏缺失或被错误覆盖

**检查点**:
```python
# 检查 clear_future_events
print(f"[DEBUG] Clearing future events from tick={generation_start_tick}")
print(f"[DEBUG] Scheduler state before: {scheduler.get_scheduled_ticks()}")
print(f"[DEBUG] Scheduler state after: {scheduler.get_scheduled_ticks()}")
```

### Bug Type D: Tokenization 不一致

**症状**: 相同 melody 产生不同的 tokens

**检查点**:
```python
# 对比两个路径的 pianoroll
print(f"[DEBUG] Offline pianoroll shape: {offline_pr.shape}")
print(f"[DEBUG] FakeRT pianoroll shape: {fakert_pr.shape}")
print(f"[DEBUG] Diff: {(offline_pr != fakert_pr).sum()} pixels differ")
```

---

## 5. 自动化 Debug 脚本

创建 `scripts/debug_inference_consistency.py`:

```python
#!/usr/bin/env python3
"""
Automated debug script to compare Offline and Fake Realtime inference results.
"""

import sys
import json
import mido
from pathlib import Path
from typing import Dict, List, Optional


class InferenceConsistencyDebugger:
    """Debug tool for comparing offline and fake realtime inference."""

    def __init__(self, offline_dir: str, fakert_dir: str):
        self.offline_dir = Path(offline_dir)
        self.fakert_dir = Path(fakert_dir)
        self.report = {
            'input_comparison': {},
            'token_comparison': {},
            'output_comparison': {},
            'per_song': {},
            'bugs_found': []
        }

    def load_track_events(self, mid_path: str, track_keywords: List[str]) -> List[Dict]:
        """Extract note events from tracks whose name contains any keyword."""
        mid = mido.MidiFile(mid_path)
        events = []
        for track in mid.tracks:
            current_time = 0
            track_name = ""
            for msg in track:
                current_time += msg.time
                if msg.type == 'track_name':
                    track_name = str(getattr(msg, 'name', ''))
                    continue

                if msg.type in ['note_on', 'note_off'] and any(k in track_name for k in track_keywords):
                    events.append({
                        'tick': current_time,
                        'type': msg.type,
                        'pitch': msg.note,
                        'velocity': msg.velocity if msg.type == 'note_on' else 0
                    })
        return sorted(events, key=lambda x: (x['tick'], x['type']))

    def compare_event_lists(self, events_a: List[Dict], events_b: List[Dict], 
                           label_a: str = "A", label_b: str = "B") -> Dict:
        """Compare two event lists and return statistics."""
        result = {
            'count_a': len(events_a),
            'count_b': len(events_b),
            'count_diff': len(events_a) - len(events_b),
            'matched': 0,
            'mismatched': 0,
            'only_in_a': [],
            'only_in_b': []
        }
        
        # Simple matching (exact tick and pitch)
        set_a = {(e['tick'], e['pitch'], e['type']) for e in events_a}
        set_b = {(e['tick'], e['pitch'], e['type']) for e in events_b}
        
        result['matched'] = len(set_a & set_b)
        result['only_in_a'] = list(set_a - set_b)
        result['only_in_b'] = list(set_b - set_a)
        result['mismatched'] = len(result['only_in_a']) + len(result['only_in_b'])

        return result

    def compare_outputs(self, offline_mid: str, fakert_mid: str) -> Dict:
        """Compare generated MIDI outputs."""
        offline_acc = self.load_track_events(offline_mid, ['Accompaniment', 'Part1'])
        fakert_acc = self.load_track_events(fakert_mid, ['Accompaniment', 'Part1'])

        if not offline_acc or not fakert_acc:
            raise ValueError(
                'Cannot find accompaniment tracks in one of the MIDI files. '
                'Please verify track names include Accompaniment or Part1.'
            )

        return self.compare_event_lists(offline_acc, fakert_acc, "Offline", "FakeRT")

    def generate_report(self) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("=" * 70)
        lines.append("Inference Consistency Debug Report")
        lines.append("=" * 70)

        # Output comparison
        lines.append("\n## Output Comparison")
        comp = self.report['output_comparison']
        lines.append(f"  Offline events: {comp.get('count_a', 'N/A')}")
        lines.append(f"  FakeRT events:  {comp.get('count_b', 'N/A')}")
        lines.append(f"  Matched: {comp.get('matched', 'N/A')}")
        lines.append(f"  Mismatched: {comp.get('mismatched', 'N/A')}")

        if comp.get('only_in_a'):
            lines.append(f"\n  Only in Offline (first 5):")
            for e in comp['only_in_a'][:5]:
                lines.append(f"    tick={e[0]}, pitch={e[1]}, type={e[2]}")

        if comp.get('only_in_b'):
            lines.append(f"\n  Only in FakeRT (first 5):")
            for e in comp['only_in_b'][:5]:
                lines.append(f"    tick={e[0]}, pitch={e[1]}, type={e[2]}")

        # Bugs found
        lines.append("\n## Bugs Detected")
        if self.report['bugs_found']:
            for bug in self.report['bugs_found']:
                lines.append(f"  - {bug}")
        else:
            lines.append("  No obvious bugs detected (but check details above)")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


def _offline_song_key(path: Path) -> Optional[str]:
    suffix = '_generated.mid'
    if not path.name.endswith(suffix):
        return None
    base = path.name[:-len(suffix)]  # e.g. 000_1
    parts = base.split('_', 1)
    return parts[1] if len(parts) == 2 else None


def _fakert_song_key(path: Path) -> Optional[str]:
    suffix = '_fake_realtime_combined.mid'
    if not path.name.endswith(suffix):
        return None
    return path.name[:-len(suffix)]  # e.g. 1


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_inference_consistency.py <offline_dir> <fakert_dir>")
        sys.exit(1)

    debugger = InferenceConsistencyDebugger(sys.argv[1], sys.argv[2])

    offline_map = {}
    for mid_path in debugger.offline_dir.glob('*_generated.mid'):
        key = _offline_song_key(mid_path)
        if key:
            offline_map[key] = mid_path

    fakert_map = {}
    for mid_path in debugger.fakert_dir.glob('*_fake_realtime_combined.mid'):
        key = _fakert_song_key(mid_path)
        if key:
            fakert_map[key] = mid_path

    common_keys = sorted(set(offline_map) & set(fakert_map))
    if not common_keys:
        raise RuntimeError('No matched song stems found between offline and fake realtime outputs.')

    per_song = {}
    total_a = 0
    total_b = 0
    total_matched = 0
    total_mismatched = 0

    for key in common_keys:
        comp = debugger.compare_outputs(str(offline_map[key]), str(fakert_map[key]))
        per_song[key] = comp
        total_a += int(comp['count_a'])
        total_b += int(comp['count_b'])
        total_matched += int(comp['matched'])
        total_mismatched += int(comp['mismatched'])

    debugger.report['per_song'] = per_song
    debugger.report['output_comparison'] = {
        'count_a': total_a,
        'count_b': total_b,
        'count_diff': total_a - total_b,
        'matched': total_matched,
        'mismatched': total_mismatched,
    }

    report_path = debugger.offline_dir / 'inference_consistency_report.json'
    report_path.write_text(json.dumps(debugger.report, ensure_ascii=False, indent=2), encoding='utf-8')

    # Print report
    print(debugger.generate_report())
    print(f"\nJSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
```

---

## 6. 详细执行计划与报告模板

### 报告记录规范

每个 Step 完成后，必须在 `developing-logs/2026-4-23/debug-reports/` 目录下创建报告文件：

```
developing-logs/2026-4-23/debug-reports/
├── step1_environment_report.md      # Step 1 完成报告
├── step2_baseline_report.md         # Step 2 完成报告
├── step3_phase1_input_layer.md      # Phase 1 完成报告
├── step3_phase2_tokenization.md     # Phase 2 完成报告
├── step3_phase3_model.md            # Phase 3 完成报告
├── step3_phase4_postprocessing.md   # Phase 4 完成报告
├── step4_bugfix_report.md           # Step 4 完成报告
├── step5_verification_report.md     # Step 5 完成报告
└── final_summary_report.md          # 最终总结报告
```

**报告模板格式**：
```markdown
# [Step/Phase 名称] 执行报告

## 1. 执行内容
- [ ] 任务 1
- [ ] 任务 2
...

## 2. 执行结果
| 指标 | 数值 | 状态 |
|------|------|------|
| ... | ... | ... |

## 3. 关键发现
### 3.1 正常情况
...

### 3.2 异常情况 (如有)
...

## 4. 结论与下一步
- **结论**: ...
- **下一步行动**: ...
- **阻塞项**: ... (如有)

## 5. 附件
- 日志文件路径: ...
- 生成的数据文件: ...
```

---

### Step 1: 环境准备与数据验证 (预计 30 min)

#### 1.1 检查测试数据完整性
- [ ] 验证 5 首 MIDI 文件存在且非空: `1.mid` - `5.mid`
- [ ] 验证 5 首 NPZ 文件存在且非空: `1.npz` - `5.npz`
- [ ] 运行 `verify_test_data.py` 验证 MIDI 和 NPZ 对应关系

#### 1.2 创建输出目录结构
- [ ] 创建 `output/debug/offline/`
- [ ] 创建 `output/debug/offline_prefix4/`
- [ ] 创建 `output/debug/fake_rt_equivalent/`
- [ ] 创建 `output/debug/fake_rt_overlap/`
- [ ] 创建 `output/debug/logs/`

#### 1.3 配置 Server 环境
- [ ] 确认 checkpoint 路径: `models/ModelLekai/epoch_4_1104_1204/model.safetensors`
- [ ] 确认 CUDA 设备可用
- [ ] 测试 server 启动命令 (不启动，仅验证命令)

#### 1.4 编写自动化脚本
- [ ] 编写 `scripts/run_all_tests.sh` (批量运行脚本)
- [ ] 编写 `scripts/compare_all.sh` (批量对比脚本)
- [ ] 测试脚本语法正确性

**Step 1 完成报告**: `step1_environment_report.md`
```markdown
# Step 1: 环境准备与数据验证 执行报告

## 1. 执行内容
- [x] 验证测试数据完整性
- [x] 创建输出目录结构
- [x] 配置 Server 环境
- [x] 编写自动化脚本

## 2. 执行结果
| 检查项 | 状态 | 备注 |
|--------|------|------|
| MIDI 1.mid - 5.mid | ✅ | 文件大小: XX KB each |
| NPZ 1.npz - 5.npz | ✅ | 文件大小: XX KB each |
| 输出目录创建 | ✅ | 4 个目录已创建 |
| Checkpoint 路径 | ✅ | 存在并可读 |
| CUDA 设备 | ✅/❌ | 设备号: X |

## 3. 关键发现
### 3.1 正常情况
- 所有测试数据已就绪
- 环境配置正确

### 3.2 异常情况 (如有)
- [ ] 无 / [ ] 数据文件缺失: ... / [ ] CUDA 不可用

## 4. 结论与下一步
- **结论**: 环境准备完成，可以进行测试
- **下一步行动**: 执行 Step 2: 运行 Baseline 测试
- **阻塞项**: 无

## 5. 附件
- 数据验证日志: `output/debug/logs/step1_data_verification.log`
```

---

### Step 2: 运行 Baseline 测试 (预计 1-1.5 hours)

#### 2.1 运行 Offline Mode (基准测试)

**测试 2.1.1: Offline gt-prefix-beats=0**
- [ ] 运行命令:
```bash
CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_lekai_offline.py \
  --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  --npz-dir prompts/inputs_lekai/npz \
  --output-dir output/debug/offline \
  --device cuda \
  --dtype auto \
  --condition-idx all \
  --gt-prefix-beats 0 \
  --temperature 1.0 \
  --top-k 1 \
  --top-p 1.0 \
  --repetition-penalty 1.2
```
- [ ] 验证输出文件生成: `output/debug/offline/*_generated.mid` (5 个文件，按歌曲 stem 配对，不按 idx)
- [ ] 记录每个文件的生成时间和 note 数量

**测试 2.1.2: Offline gt-prefix-beats=4** (用于对比)
- [ ] 运行命令 (同上，改为 `--output-dir output/debug/offline_prefix4 --gt-prefix-beats 4`，其余 deterministic 参数保持一致)
- [ ] 验证输出文件生成: `output/debug/offline_prefix4/*_generated.mid` (5 个文件)

#### 2.2 运行 Fake Realtime "等价 Offline"模式

**前置条件**: Server 已启动且 deterministic sampling 参数已设置

- [ ] 启动 Server (deterministic sampling):
```bash
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=cuda \
LEKAI_DTYPE=auto \
LEKAI_RT_TEMPERATURE=1.0 \
LEKAI_RT_TOP_K=1 \
LEKAI_RT_TOP_P=1.0 \
LEKAI_RT_REPETITION_PENALTY=1.2 \
uv run python -m streammuse.infrastructure.inference.server_lekai
```

- [ ] 运行 5 首曲子的 FakeRT (interval=4, length=4):
```bash
for i in 1 2 3 4 5; do
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path prompts/inputs_lekai/mel/${i}.mid \
    --output-dir output/debug/fake_rt_equivalent \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --max-ticks 256
done
```
- [ ] 验证输出文件: `output/debug/fake_rt_equivalent/*_combined.mid` (5 个文件)
- [ ] 记录每个文件的生成时间和 note 数量

#### 2.3 运行 Fake Realtime "Overlap"模式 (对比用)

- [ ] 运行 5 首曲子的 FakeRT (interval=4, length=8):
```bash
for i in 1 2 3 4 5; do
  uv run python scripts/run_lekai_fake_realtime.py \
    --midi-file-path prompts/inputs_lekai/mel/${i}.mid \
    --output-dir output/debug/fake_rt_overlap \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 8 \
    --max-ticks 256
done
```
- [ ] 验证输出文件: `output/debug/fake_rt_overlap/*_combined.mid` (5 个文件)

#### 2.4 初步结果对比

- [ ] 运行对比脚本，生成初步对比报告:
```bash
uv run python scripts/debug_inference_consistency.py \
  output/debug/offline \
  output/debug/fake_rt_equivalent
```
- [ ] 检查对比报告，记录差异程度（JSON 默认输出到 `output/debug/offline/inference_consistency_report.json`）

**Step 2 完成报告**: `step2_baseline_report.md`
```markdown
# Step 2: 运行 Baseline 测试 执行报告

## 1. 执行内容
- [x] 测试 2.1.1: Offline gt-prefix-beats=0
- [x] 测试 2.1.2: Offline gt-prefix-beats=4
- [x] 测试 2.2: FakeRT equivalent (4,4) deterministic(top_k=1)
- [x] 测试 2.3: FakeRT overlap (4,8)
- [x] 初步结果对比

## 2. 执行结果
| 歌曲 | Offline Notes | FakeRT(4,4) Notes | FakeRT(4,8) Notes | Match % |
|------|---------------|-------------------|-------------------|---------|
| 1    | XX            | XX                | XX                | XX%     |
| 2    | XX            | XX                | XX                | XX%     |
| ...  | ...           | ...               | ...               | ...     |

## 3. 关键发现
### 3.1 正常情况
- [ ] FakeRT(4,4) 和 Offline 完全一致 → 无 Bug，跳到 Step 5
- [ ] FakeRT(4,4) 和 Offline 基本一致 (match > 95%) → 可能是量化误差，继续排查

### 3.2 异常情况 (如发现不一致)
- [ ] 系统性 tick 偏移 (所有 note 差 N ticks)
- [ ] Note 缺失/多余 (match < 90%)
- [ ] 完全不同的伴奏 (match < 50%)
- [ ] 某首歌曲特别差 (其他正常)

## 4. 结论与下一步
- **结论**: 
  - [ ] 无显著 Bug → 结束测试
  - [ ] 有 Bug → 进入 Step 3 Phase X 针对性排查
- **下一步行动**: 根据 Bug 类型选择 Phase
- **阻塞项**: 无

## 5. 附件
- 生成的 MIDI 文件: `output/debug/*/*`
- 对比报告: `output/debug/offline/inference_consistency_report.json`
```

---

### Step 3: 逐层排查 (预计 2-4 hours)

根据 Step 2 的结果，选择性执行以下 Phase。

#### Phase 3.1: 输入层排查 (如果怀疑输入不一致)

**目标**: 验证 MIDI 和 NPZ 提取的 melody 是否完全一致

- [ ] **Task 3.1.1**: 提取并对比 note lists
  - 运行 `scripts/compare_inputs.py`
  - 对比每首歌的 MIDI notes vs NPZ notes
  - 记录 pitch/tick/velocity 差异

- [ ] **Task 3.1.2**: 分析量化误差
  - 计算 tick 差异分布 (统计差 0/1/2 ticks 的比例)
  - 判断是否在接受范围内 (±1 tick)

- [ ] **Task 3.1.3**: 修复输入层 Bug (如有)
  - 如果发现系统性差异，修复提取逻辑
  - 重新运行 Step 2 验证

**Phase 3.1 完成报告**: `step3_phase1_input_layer.md`
```markdown
# Phase 3.1: 输入层排查 执行报告

## 1. 执行内容
- [x] Task 3.1.1: 提取并对比 note lists
- [x] Task 3.1.2: 分析量化误差
- [x] Task 3.1.3: 修复输入层 Bug (如有)

## 2. 执行结果
| 歌曲 | MIDI Notes | NPZ Notes | Match % | Tick 差异分布 |
|------|------------|-----------|---------|---------------|
| 1    | XX         | XX        | XX%     | 0tick: X, 1tick: X |
| ...  | ...        | ...       | ...     | ...           |

## 3. 关键发现
### 3.1 正常情况
- MIDI 和 NPZ 提取的 melody 基本一致 (match > 95%)
- Tick 差异在 ±1 tick 以内 (量化误差)

### 3.2 异常情况
- [ ] 无异常 / [ ] Pitch 不一致 / [ ] Tick 偏移大 / [ ] Note 缺失

## 4. 结论与下一步
- **结论**: 输入层 [无问题 / 已修复]
- **下一步行动**: 
  - [ ] 输入层正常 → 进入 Phase 3.2 Tokenization 层
  - [ ] 已修复 → 重新运行 Step 2 验证
- **阻塞项**: 无

## 5. 附件
- 详细对比日志: `output/debug/logs/phase1_input_comparison.log`
```

---

#### Phase 3.2: Tokenization 层排查 (如果输入层正常)

**目标**: 验证 MIDI → tokens 和 NPZ → tokens 是否一致

- [ ] **Task 3.2.1**: 添加 token 导出日志
  - 修改 `lekai_http_backend.py` 添加 pianoroll/tokens 日志
  - 修改 `inference.py` (offline) 添加相同日志

- [ ] **Task 3.2.2**: 运行带日志的测试
  - 重新运行 1 首歌曲 (选最简单的，如 1.mid/1.npz)
  - 收集两边生成的日志

- [ ] **Task 3.2.3**: 对比 pianoroll 和 tokens
  - 对比每个 beat 的 pianoroll shape 和 sum
  - 对比 compressed tokens 序列
  - 记录第一次出现差异的 beat

- [ ] **Task 3.2.4**: 修复 tokenization Bug (如有)
  - 修复 `MidiConverter` 或 `PianoDataset` 的差异
  - 重新运行验证

**Phase 3.2 完成报告**: `step3_phase2_tokenization.md`
```markdown
# Phase 3.2: Tokenization 层排查 执行报告

## 1. 执行内容
- [x] Task 3.2.1: 添加 token 导出日志
- [x] Task 3.2.2: 运行带日志的测试
- [x] Task 3.2.3: 对比 pianoroll 和 tokens
- [x] Task 3.2.4: 修复 tokenization Bug (如有)

## 2. 执行结果
| Beat | Offline Tokens | FakeRT Tokens | Match | 首次差异位置 |
|------|----------------|---------------|-------|--------------|
| 0    | [X,X,X]        | [X,X,X]       | ✅/❌  | N/A or Beat X |
| ...  | ...            | ...           | ...   | ...          |

## 3. 关键发现
### 3.1 正常情况
- Pianoroll shape 和 sum 完全一致
- Tokens 序列完全一致

### 3.2 异常情况
- [ ] Pianoroll 差异 (像素级) → MidiConverter Bug
- [ ] Tokens 压缩后差异 → Tokenizer Bug
- [ ] 特定 beat 开始不一致 → 时间对齐 Bug

## 4. 结论与下一步
- **结论**: Tokenization 层 [无问题 / 已修复]
- **下一步行动**: 
  - [ ] Tokenization 正常 → 进入 Phase 3.3 Model 层
  - [ ] 已修复 → 重新运行 Step 2 验证
- **阻塞项**: 无

## 5. 附件
- Token 日志: `output/debug/logs/phase2_tokens_*.log`
- 差异分析: `output/debug/logs/phase2_token_diff.txt`
```

---

#### Phase 3.3: Model 层排查 (如果 tokenization 正常)

**目标**: 验证 model input/output 是否一致

- [ ] **Task 3.3.1**: 添加 model I/O 日志
  - 在 `PianoLLaMA.generate_accompaniment` 中添加 prompt/output 日志
  - 在 `lekai_http_backend` 中添加相同日志

- [ ] **Task 3.3.2**: 运行 deterministic 测试
  - 确保 Offline/FakeRT 的采样参数一致（`temperature=1.0, top_k=1, top_p=1.0`）
  - 运行 1 首歌曲
  - 收集 model 输入输出

- [ ] **Task 3.3.3**: 对比 model input
  - 对比 prompt token 序列
  - 确认 prompt construction 一致

- [ ] **Task 3.3.4**: 对比 model output
  - 对比生成的 tokens
  - 如果 input 相同但 output 不同 → model state Bug

- [ ] **Task 3.3.5**: 修复 model 层 Bug (如有)
  - 检查 KV cache, hidden state 使用
  - 修复并验证

**Phase 3.3 完成报告**: `step3_phase3_model.md`
```markdown
# Phase 3.3: Model 层排查 执行报告

## 1. 执行内容
- [x] Task 3.3.1: 添加 model I/O 日志
- [x] Task 3.3.2: 运行 deterministic 测试
- [x] Task 3.3.3: 对比 model input
- [x] Task 3.3.4: 对比 model output
- [x] Task 3.3.5: 修复 model 层 Bug (如有)

## 2. 执行结果
| 检查项 | Offline | FakeRT | Match |
|--------|---------|--------|-------|
| Prompt Length | XX | XX | ✅/❌ |
| Prompt Tokens (前20) | [...] | [...] | ✅/❌ |
| Output Length | XX | XX | ✅/❌ |
| Output Tokens (前20) | [...] | [...] | ✅/❌ |

## 3. 关键发现
### 3.1 正常情况
- Prompt 完全一致
- Output 完全一致 (deterministic)

### 3.2 异常情况
- [ ] Prompt 不同 → history accumulation Bug
- [ ] Prompt 相同但 output 不同 → model cache/state Bug
- [ ] Output length 不同 → generation stopping criteria Bug

## 4. 结论与下一步
- **结论**: Model 层 [无问题 / 已修复]
- **下一步行动**: 
  - [ ] Model 层正常 → 进入 Phase 3.4 Post-processing 层
  - [ ] 已修复 → 重新运行 Step 2 验证
- **阻塞项**: 无

## 5. 附件
- Model I/O 日志: `output/debug/logs/phase3_model_io_*.log`
```

---

#### Phase 3.4: Post-processing 层排查 (如果 model 层正常)

**目标**: 验证 tokens → MIDI 的转换是否一致

- [ ] **Task 3.4.1**: 添加 post-processing 日志
  - 在 `tokens_to_midi` / `beats_to_pianoroll` 中添加形状/数值日志

- [ ] **Task 3.4.2**: 运行并收集日志
  - 运行 1 首歌曲
  - 收集两边的 pianoroll 和 events

- [ ] **Task 3.4.3**: 对比 pianoroll
  - 对比解压后的 pianoroll shape 和数值
  - 统计差异像素数

- [ ] **Task 3.4.4**: 对比最终 MIDI events
  - 对比 note_on/note_off 的时间、音高、力度
  - 记录系统性偏移

- [ ] **Task 3.4.5**: 修复 post-processing Bug (如有)
  - 修复 tokenizer 解码逻辑
  - 修复 events 转换逻辑

**Phase 3.4 完成报告**: `step3_phase4_postprocessing.md`
```markdown
# Phase 3.4: Post-processing 层排查 执行报告

## 1. 执行内容
- [x] Task 3.4.1: 添加 post-processing 日志
- [x] Task 3.4.2: 运行并收集日志
- [x] Task 3.4.3: 对比 pianoroll
- [x] Task 3.4.4: 对比最终 MIDI events
- [x] Task 3.4.5: 修复 post-processing Bug (如有)

## 2. 执行结果
| 检查项 | Offline | FakeRT | Match |
|--------|---------|--------|-------|
| Pianoroll Shape | (X,Y,Z) | (X,Y,Z) | ✅/❌ |
| Pianoroll 差异像素 | 0 | X | ✅/❌ |
| MIDI Events 数量 | XX | XX | ✅/❌ |
| Tick 偏移 (平均) | 0 | X ticks | ✅/❌ |

## 3. 关键发现
### 3.1 正常情况
- Pianoroll 完全一致
- MIDI events 完全一致

### 3.2 异常情况
- [ ] Pianoroll 形状不同 → 解压逻辑 Bug
- [ ] Pianoroll 数值不同 → patch reconstruction Bug
- [ ] Tick 系统性偏移 → time alignment Bug
- [ ] Events 数量不同 → note filtering Bug

## 4. 结论与下一步
- **结论**: Post-processing 层 [无问题 / 已修复]
- **下一步行动**: 
  - [ ] Post-processing 正常 → 所有层排查完毕，进入 Step 4 修复总结
  - [ ] 已修复 → 重新运行 Step 2 验证
- **阻塞项**: 无

## 5. 附件
- Post-processing 日志: `output/debug/logs/phase4_postproc_*.log`
- Pianoroll 对比图: `output/debug/logs/phase4_pianoroll_diff.png`
```

---

### Step 4: Bug 修复与回归测试 (时间不定)

根据 Step 3 定位的 Bug，执行修复。

#### 4.1 Bug 修复
- [ ] **Task 4.1**: 实现 Bug 修复
  - 根据 Phase 3.X 的定位，修改对应代码
  - 编写单元测试覆盖 Bug 场景

- [ ] **Task 4.2**: 本地验证修复
  - 运行相关单元测试
  - 手动验证修复效果

#### 4.2 回归测试
- [ ] **Task 4.3**: 重新运行 Step 2
  - 完整运行 5 首歌曲的对比测试
  - 生成新的对比报告

- [ ] **Task 4.4**: 验证修复效果
  - 对比修复前后的 match rate
  - 确认 Bug 已修复，没有引入新问题

- [ ] **Task 4.5**: 代码审查和提交
  - 代码审查
  - 提交修复 (commit)

**Step 4 完成报告**: `step4_bugfix_report.md`
```markdown
# Step 4: Bug 修复与回归测试 执行报告

## 1. 执行内容
- [x] Task 4.1: 实现 Bug 修复
- [x] Task 4.2: 本地验证修复
- [x] Task 4.3: 重新运行 Step 2
- [x] Task 4.4: 验证修复效果
- [x] Task 4.5: 代码审查和提交

## 2. Bug 修复详情
| Bug ID | 位置 | 根因 | 修复方案 | 状态 |
|--------|------|------|----------|------|
| BUG-1  | 文件:行号 | 简述 | 简述 | ✅ |
| ...    | ...  | ...  | ...      | ...  |

## 3. 回归测试结果
| 歌曲 | 修复前 Match | 修复后 Match | 提升 |
|------|--------------|--------------|------|
| 1    | XX%          | XX%          | +XX% |
| ...  | ...          | ...          | ...  |

## 4. 结论与下一步
- **结论**: 所有定位的 Bug 已修复，回归测试通过
- **下一步行动**: 进入 Step 5 最终验证
- **阻塞项**: 无

## 5. 附件
- 修复的代码变更: `git diff`
- 单元测试: `tests/test_bugfix_*.py`
- 回归测试报告: `output/debug/step4_regression_report.json`
```

---

### Step 5: 最终验证与总结 (预计 30 min)

#### 5.1 完整测试验证
- [ ] **Task 5.1**: 运行全部 5 首歌曲的完整对比
  - Offline (gt-prefix=0)
  - FakeRT equivalent (4,4, deterministic top_k=1)
  - FakeRT overlap (4,8)

- [ ] **Task 5.2**: 生成最终对比报告
  - 统计所有歌曲的 match rate
  - 计算平均 match rate

- [ ] **Task 5.3**: 验证通过标准
  - [ ] FakeRT equivalent match rate >= 95%
  - [ ] FakeRT overlap match rate >= 80% (允许差异)
  - [ ] 无系统性 Bug

#### 5.2 文档总结
- [ ] **Task 5.4**: 编写最终总结报告
  - 汇总所有发现
  - 记录修复的 Bug
  - 给出建议

- [ ] **Task 5.5**: 更新相关文档
  - 更新技术文档 (如有)
  - 更新测试用例

**Step 5 完成报告**: `step5_verification_report.md`
```markdown
# Step 5: 最终验证与总结 执行报告

## 1. 执行内容
- [x] Task 5.1: 运行全部 5 首歌曲的完整对比
- [x] Task 5.2: 生成最终对比报告
- [x] Task 5.3: 验证通过标准
- [x] Task 5.4: 编写最终总结报告
- [x] Task 5.5: 更新相关文档

## 2. 最终测试结果
| 配置 | 平均 Match Rate | 最低 Match Rate | 通过标准 |
|------|-----------------|-----------------|----------|
| FakeRT equiv (4,4) | XX% | XX% | >= 95% ✅/❌ |
| FakeRT overlap (4,8) | XX% | XX% | >= 80% ✅/❌ |

## 3. 发现的 Bug 总结
| Bug ID | 严重程度 | 位置 | 状态 |
|--------|----------|------|------|
| BUG-1  | 高/中/低 | 简述 | 已修复 |
| ...    | ...      | ...  | ...   |

## 4. 结论与建议
- **总体结论**: 测试 [通过 / 未通过]
- **关键发现**: ...
- **后续建议**: 
  - [ ] 继续优化 XXX
  - [ ] 添加自动化测试
  - [ ] 更新文档

## 5. 附件
- 完整测试数据: `output/debug/final_test/`
- 最终对比报告: `output/debug/step5_final_comparison.json`
- 代码变更: `git log`
```

---

### 最终总结报告: `final_summary_report.md`

```markdown
# Offline vs Fake Realtime Debug 最终总结报告

## 项目概述
- **目标**: 通过对比找出 Inference Engine 的 Bug
- **测试数据**: prompts/inputs_lekai 的 5 首曲子
- **测试时间**: YYYY-MM-DD to YYYY-MM-DD
- **参与人员**: XXX

## 执行摘要
- **发现的 Bug 数量**: X 个
- **已修复 Bug 数量**: X 个
- **最终 Match Rate**: XX%
- **状态**: [完成 / 部分完成]

## 详细发现
### Bug 1: XXX
- **位置**: 文件:行号
- **根因**: ...
- **影响**: ...
- **修复**: ...
- **验证**: ...

## 测试数据归档
- 所有生成的 MIDI 文件: `output/debug/`
- 所有报告文件: `developing-logs/2026-4-23/debug-reports/`
- 代码变更: commit hash

## 后续行动
1. ...
2. ...
3. ...
```

---

## 执行计划流程图

```
Step 1: 环境准备
    ↓
Step 2: 运行 Baseline (5 首 × 3 配置)
    ↓
检查结果?
    ├── 完全一致 → Step 5 (结束)
    └── 有差异 → Step 3
            ↓
    Phase 3.1: 输入层排查
            ↓ (如正常)
    Phase 3.2: Tokenization 层排查
            ↓ (如正常)
    Phase 3.3: Model 层排查
            ↓ (如正常)
    Phase 3.4: Post-processing 层排查
            ↓
Step 4: Bug 修复
    ↓
Step 5: 最终验证
```

**每个 Step/Phase 完成后必须写报告！**

---

## 7. 预期结果

### 理想情况

**Config B (interval=4, length=4, deterministic sampling)**:
- Offline 和 Fake Realtime 应该产生 **逐拍完全相同的 tokens**
- 最终 MIDI 文件的伴奏部分应该 **逐 note 一致**

**Config C (overlap 模式)**:
- 由于增量更新机制，结果会和 offline 有差异
- 但和声结构应该保持一致

### 可接受的差异 (Config B 应该无差异)

**Config B (interval=4, length=4, deterministic sampling)**:
- ✅ **应该完全一致** (无差异)
- 如果有差异 → **100% 是 Bug**

**Config C (overlap) 和其他非 deterministic 设置**:
- Tick 级别的量化误差 (±1 tick)
- 随机采样导致的 note 差异 (non-deterministic mode)
- 分段边界处的轻微 discontinuity

### 不可接受的差异 (Bug)
- 大量 note 缺失或 extra
- 系统性的 pitch 偏移
- Tick 偏移 > 4 ticks
- 和声结构完全不同

---

**计划制定者**: Kimi Code  
**更新日期**: 2026-04-23  
**关键更新**: 明确了 interval=4, length=4 + deterministic(top_k=1) 的配置为等价测试基准  
**状态**: 待执行  
**优先级**: 高 (P0)

---

## 附录：核心逻辑推导

### 为什么 interval=4, length=4 + deterministic(top_k=1) 应该和 Offline 完全一致？

**Offline 的工作方式**:
```
输入: [melody_beat_0, melody_beat_1, ..., melody_beat_N]
输出: [acc_beat_0, acc_beat_1, ..., acc_beat_N]
生成: 一次性并行生成所有 beats (或逐 beat 顺序生成)
```

**Fake Realtime (interval=4, length=4) 的工作方式**:
```
tick=0: 触发生成 acc_beat_0
  prompt: [melody_beat_0]
  output: [acc_beat_0]

tick=4: 触发生成 acc_beat_1  
  prompt: [melody_beat_0, acc_beat_0, melody_beat_1]
  output: [acc_beat_1]

tick=8: 触发生成 acc_beat_2
  prompt: [melody_beat_0, acc_beat_0, melody_beat_1, acc_beat_1, melody_beat_2]
  output: [acc_beat_2]
```

**等价性分析**:

当 `top_k=1` 且其余采样参数一致时，模型在每一步只会保留一个候选 token，因此输出是确定性的。

在 Fake Realtime 的每拍生成中：
- prompt 包含了所有历史 melody 和已生成的 accompaniment
- 这和 offline 的逐拍生成使用的 context 是等价的
- 因此生成的结果应该完全一致

**如果不同，说明**:
1. prompt construction 不一致 (漏了某些 notes)
2. tokenization 路径不一致 (MIDI → tokens vs pianoroll → tokens)
3. model state 不一致 (cache, hidden state 等)
4. 时间戳对齐问题 (tick offset)
