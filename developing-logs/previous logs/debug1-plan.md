# Lekai 框架完善实施计划 (Debug1 Plan)

**目标**: 修复所有 placeholder 代码，完成真实的 Lekai 模型推理流程，无需 GPU 但代码必须可运行。

**环境**: macOS (无 GPU)
**输出**: 可工作的 Client-Server 架构，支持真实模型推理（或优雅降级到规则 stub）

---

## 当前问题清单

### 🔴 Critical (阻塞性问题)

1. **`beats_to_pianoroll` 是 placeholder**
   - 位置: `inference_adapter.py:171-198`
   - 现状: 返回 `np.zeros()`，没有实际解码
   - **重要发现**: `PianoRollTokenizer` 已有完整 decode 实现，只需正确调用

2. **`_generate_with_model` 使用 placeholder pianoroll**
   - 位置: `lekai_http_backend.py:224-227`
   - 现状: 生成空的 pianoroll，导致没有音频输出
   - 修复: 使用正确的 `beats_to_pianoroll` 函数

### 🟡 High (功能缺陷)

3. **BPM 硬编码**
   - 位置: `_generate_with_model()` 中 `bpm=120`
   - 现状: 无法从输入动态获取

### 🟢 Medium (优化项)

4. **错误处理不完善**
   - 模型加载失败时 fallback 逻辑已存在，但需要验证
   - 中间步骤异常需要更清晰的错误信息

### ✅ 非问题（澄清）

**Tokenizer decode 已完整实现**
- `my_tokenizer.py` 第 90-107 行: `decode()` 方法 ✅
- `my_tokenizer.py` 第 249-309 行: `decompress_tokens()` ✅
- `my_tokenizer.py` 第 318-366 行: `patch_tokens_to_image()` ✅

**规则生成器 note off 写死**
- 位置: `_generate_rule_based()` 中 `note_off_tick = chord_tick + interval`
- **状态**: ✅ 这是预期行为，不是 bug
- **说明**: 规则 stub 的设计目标就是简单、不崩溃、能响。音符时值固定为 `interval` 是 stub 的固有局限。

---

## 核心数据结构详解

### 1. Pianoroll 定义

```python
shape: (2, 88, T)
- channel 0 (sustain): 音符持续标记 (0/1)
- channel 1 (onset): 音符起始标记 (0/1)，只能在 sustain=1 的位置为 1
- pitch: 88 个音高 (MIDI 21-108)
- time: T 个时间步 (timesteps)
```

**关键理解**:
- `onset=1` 表示该 timestep 有新音符开始
- `sustain=1` 表示该 timestep 有音符在响（包括开始的 timestep）
- 一个音符 = onset 上升沿 + sustain 持续 + sustain 下降沿（note_off）

### 2. Tokenization 流程

```
Pianoroll (2, 88, T)
    ↓ image_to_patch_tokens
Tokens matrix (num_time_patches, num_pitch_patches)
    - patch_h=1, patch_w=4（默认）
    - 三进制编码: 0=无音符, 1=只有sustain, 2=sustain+onset
    ↓ compress_tokens
Compressed tokens (一维数组)
    - 格式: [pos_marker, token_value, pos_marker, token_value, ..., end_marker]
    - pos_marker = relative_position + marker_offset(81)
    - end_marker: part0=170, part1=171
    - empty_measure: 169
```

### 3. Token 解码流程（已实现）

```
Compressed tokens
    ↓ decompress_tokens
Tokens matrix (num_time_patches, num_pitch_patches)
    - 解析 [pos_marker, token_value] 对
    - 重建 token 矩阵
    ↓ patch_tokens_to_image
Pianoroll (2, 88, T)
    - 三进制解码: token_value → (sustain, onset)
    - reshape 回 (2, 88, T)
```

### 4. Measure 和 Beat 结构

```python
# 4 通道 Measure (4, 88, timesteps_per_measure)
- channels 0-1: part0 (melody/高音)
- channels 2-3: part1 (accompaniment/低音)

# 每拍处理 (process_measure_with_beat_interleaving)
for each beat:
    part0_beat = measure[:2]  # (2, 88, 4)
    part1_beat = measure[2:]  # (2, 88, 4)
    
    # 分别编码
    tokens_0 = tokenizer.image_to_patch_tokens(part0_beat)
    compressed_0 = tokenizer.compress_tokens(tokens_0, end_marker=170)
    
    tokens_1 = tokenizer.image_to_patch_tokens(part1_beat)
    compressed_1 = tokenizer.compress_tokens(tokens_1, end_marker=171)
```

---

## 双路径架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    LekaiHttpBackend.generate()                   │
│                                                                  │
│  ┌─────────────────────┐      ┌─────────────────────────────┐   │
│  │   规则 Stub 路径     │      │    真实模型路径              │   │
│  │   (无 checkpoint)    │      │    (有 checkpoint)           │   │
│  │                     │      │                             │   │
│  │  _generate_rule_based│      │  _generate_with_model       │   │
│  │                     │      │                             │   │
│  │  • 音符时值写死 ✅   │      │  • 音符时值由 AI 决定        │   │
│  │  • 音高下移八度      │      │  • 需要 beats→pianoroll     │   │
│  │  • 简单和弦         │      │  • 当前: placeholder ⚠️      │   │
│  └─────────────────────┘      └─────────────────────────────┘   │
│                                                                  │
│  自动选择: checkpoint 存在且加载成功 → 真实模型，否则 → 规则 stub  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 实施计划

### Phase 1: 修复 `beats_to_pianoroll`

**目标**: 使用已有的 tokenizer decode 方法，实现正确的 beat tokens → pianoroll 转换

#### Task 1.1: 分析 `generate_from_beats` 输出格式

- [ ] **1.1.1** 打印 `part1_beats` 的实际结构
  ```python
  print(f"part1_beats length: {len(part1_beats)}")
  print(f"First beat type: {type(part1_beats[0])}")
  print(f"First beat shape/content: {part1_beats[0]}")
  ```
- [ ] **1.1.2** 确认每个 beat 是 `List[int]`（compressed tokens）
- [ ] **1.1.3** 确认 end_marker (171) 的位置

#### Task 1.2: 实现正确的 `beats_to_pianoroll`

- [ ] **1.2.1** 在 `inference_adapter.py` 重写函数:
  ```python
  def beats_to_pianoroll(
      beat_tokens: List[List[int]],
      tokenizer: PianoRollTokenizer,
      timesteps_per_beat: int = 4,
  ) -> np.ndarray:
      """
      将 part1 beat tokens 转换为 pianoroll。
      
      Args:
          beat_tokens: 每个 beat 的 compressed token 列表
          tokenizer: 已配置的 PianoRollTokenizer
          timesteps_per_beat: 每拍时间步数（默认 4）
      
      Returns:
          pianoroll: shape (2, 88, T)，T = len(beat_tokens) * timesteps_per_beat
      """
      all_pianoroll_parts = []
      
      for beat_idx, beat_tokens_list in enumerate(beat_tokens):
          # 跳过 bar token（如果是单个 bar token）
          if len(beat_tokens_list) == 1 and beat_tokens_list[0] == 255:
              # Bar token，生成空的 pianoroll
              beat_pianoroll = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
              all_pianoroll_parts.append(beat_pianoroll)
              continue
          
          # 使用 tokenizer 解码这个 beat
          # beat_tokens_list 是 compressed tokens，需要 decompress 和 decode
          compressed = np.array(beat_tokens_list, dtype=np.int64)
          
          # Decompress tokens
          tokens_matrix = tokenizer.decompress_tokens(compressed, end_marker_id=171)
          
          # Decode to pianoroll
          beat_pianoroll = tokenizer.patch_tokens_to_image(tokens_matrix)
          
          # 确保形状正确
          if beat_pianoroll.shape[2] != timesteps_per_beat:
              # 可能需要 padding 或裁剪
              beat_pianoroll = _adjust_timesteps(beat_pianoroll, timesteps_per_beat)
          
          all_pianoroll_parts.append(beat_pianoroll)
      
      # 拼接所有 beats
      if all_pianoroll_parts:
          full_pianoroll = np.concatenate(all_pianoroll_parts, axis=2)
      else:
          full_pianoroll = np.zeros((2, 88, 0), dtype=np.float32)
      
      return full_pianoroll
  ```
- [ ] **1.2.2** 实现辅助函数 `_adjust_timesteps`（如果需要）
- [ ] **1.2.3** 添加错误处理和边界检查

#### Task 1.3: 更新 `_generate_with_model`

- [ ] **1.3.1** 替换 placeholder:
  ```python
  # 删除:
  # part1_pianoroll = np.zeros((2, 88, num_beats_to_generate * 4), dtype=np.float32)
  
  # 替换为:
  from streammuse.infrastructure.inference.lekai_model.inference_adapter import beats_to_pianoroll
  part1_pianoroll = beats_to_pianoroll(
      part1_beats,
      tokenizer=self._tokenizer,
      timesteps_per_beat=4,
  )
  ```
- [ ] **1.3.2** 添加形状验证:
  ```python
  expected_timesteps = num_beats_to_generate * 4
  assert part1_pianoroll.shape == (2, 88, expected_timesteps), \
      f"Expected shape (2, 88, {expected_timesteps}), got {part1_pianoroll.shape}"
  ```

#### Task 1.4: 测试 tokenizer decode

- [ ] **1.4.1** 创建测试脚本验证 decode 流程:
  ```python
  # 创建简单 pianoroll → encode → decode → 比较
  original = np.zeros((2, 88, 4))
  original[0, 40, :] = 1  # sustain
  original[1, 40, 0] = 1  # onset at tick 0
  
  compressed = tokenizer.encode(original)
  recovered = tokenizer.decode(compressed, end_marker_id=170)
  
  assert np.allclose(original, recovered)
  ```

**Exit Criteria**:
- [ ] E1: `beats_to_pianoroll` 不再使用 placeholder
- [ ] E2: encode → decode roundtrip 通过
- [ ] E3: 生成的 pianoroll 有有效值（非全 zero）

---

### Phase 2: 端到端验证

**目标**: 确保代码能在 CPU 上跑通

#### Task 2.1: Server 启动测试

- [ ] **2.1.1** 启动 server:
  ```bash
  python -m src.streammuse.infrastructure.inference.server_lekai
  ```
- [ ] **2.1.2** 测试 health endpoint:
  ```bash
  curl http://localhost:8000/health
  ```

#### Task 2.2: API 测试

- [ ] **2.2.1** 测试 generate_accompaniment:
  ```bash
  curl -X POST http://localhost:8000/generate_accompaniment \
    -H "Content-Type: application/json" \
    -d '{
      "melody_notes": [{"type": "note_on", "pitch": 60, "tick": 0}],
      "generation_start_tick": 4,
      "generation_length_frames": 8,
      "generation_interval_ticks": 4,
      "model_name": "lekai"
    }'
  ```
- [ ] **2.2.2** 验证响应包含有效的 accompaniment 事件

#### Task 2.3: 客户端集成测试

- [ ] **2.3.1** 启动 CLI:
  ```bash
  uv run streammuse-cli \
    --input-mode keyboard \
    --model-name lekai \
    --generation-interval-ticks 4 \
    --generation-length-frames 20
  ```
- [ ] **2.3.2** 按键测试，验证无崩溃

**Exit Criteria**:
- [ ] E1: Server 能独立启动
- [ ] E2: Client 能连接 Server
- [ ] E3: 按键 → 生成伴奏 → 播放，全流程无崩溃

---

### Phase 3: 文档和清理

#### Task 3.1: 代码注释

- [ ] **3.1.1** 为 `beats_to_pianoroll` 添加详细 docstring
- [ ] **3.1.2** 在关键步骤添加注释说明数据流

#### Task 3.2: 单元测试

- [ ] **3.2.1** `test_tokenizer_decode.py` - 测试 decode 正确性
- [ ] **3.2.2** `test_beats_to_pianoroll.py` - 测试 beat 转换
- [ ] **3.2.3** `test_server_lekai_integration.py` - 测试完整流程

#### Task 3.3: 回退机制验证

- [ ] **3.3.1** 测试无 checkpoint 时的 rule stub
- [ ] **3.3.2** 测试模型加载失败时的错误信息

**Exit Criteria**:
- [ ] E1: 所有新代码有完整 docstring
- [ ] E2: 单元测试覆盖率达标
- [ ] E3: rule stub 作为 fallback 工作正常

---

## 详细实现参考

### Tokenizer 关键参数

```python
tokenizer = PianoRollTokenizer(
    patch_h=1,           # patch 高度（音高方向）
    patch_w=4,           # patch 宽度（时间方向）= 4 timesteps
    marker_offset=81,    # 相对位置标记偏移量
    measures_length=88,  # 音高数量（88键）
    end_marker_part0=170,# part0 结束标记
    end_marker_part1=171,# part1 结束标记
    empty_marker=169,    # 空 measure 标记
    img_h=88             # 图像高度（音高）
)
```

### 使用 Tokenizer 的完整流程

```python
# 1. 编码 pianoroll → compressed tokens
pianoroll = np.zeros((2, 88, 4))  # 1 beat
pianoroll[0, 40, :] = 1  # sustain on pitch 40
pianoroll[1, 40, 0] = 1  # onset at tick 0

tokens_matrix = tokenizer.image_to_patch_tokens(pianoroll)
compressed = tokenizer.compress_tokens(tokens_matrix, end_marker=171)

# 2. 解码 compressed tokens → pianoroll
recovered_tokens = tokenizer.decompress_tokens(compressed, end_marker_id=171)
recovered_pianoroll = tokenizer.patch_tokens_to_image(recovered_tokens)

# 3. 验证
assert recovered_pianoroll.shape == (2, 88, 4)
```

### 数据结构检查清单

- [ ] `part1_beats` 是 `List[List[int]]`，每个元素是一拍的 compressed tokens
- [ ] 每个 beat 的 compressed tokens 以 171 (end_marker_part1) 结尾
- [ ] Bar token (255) 单独作为一个 beat
- [ ] `tokenizer.decode()` 返回 shape (2, 88, T)

---

## 风险和缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| `part1_beats` 格式与预期不符 | 中 | 高 | 打印调试，根据实际格式调整 |
| Decode 后的 pianoroll 形状不对 | 中 | 高 | 添加形状检查和调整逻辑 |
| CPU 推理太慢 | 高 | 中 | 减少 `generation_length_frames` |
| 空 measure 处理不当 | 低 | 中 | 特殊处理 bar token 和 empty_marker |

---

## 提交建议

### Commit 1: Fix beats_to_pianoroll
```
fix(lekai): implement correct beats_to_pianoroll using tokenizer decode

- Use PianoRollTokenizer.decompress_tokens and patch_tokens_to_image
- Replace placeholder in inference_adapter.py
- Update _generate_with_model to use correct function
```

### Commit 2: Integration Tests
```
test(lekai): add integration tests for lekai server

- Add tokenizer decode roundtrip test
- Add beats_to_pianoroll test
- Add server/client integration test
```

---

## Definition of Done

- [ ] `beats_to_pianoroll` 使用正确的 tokenizer decode 方法
- [ ] `_generate_with_model` 不再使用 placeholder
- [ ] Server + Client 能完整跑通（键盘 → 生成 → 播放）
- [ ] 无 checkpoint 时自动 fallback 到 rule stub
- [ ] 所有新代码有完整 docstring 和单元测试
- [ ] 更新 debug-report.md 记录完成状态
