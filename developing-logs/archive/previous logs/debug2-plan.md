# Lekai 框架修复计划 (Debug2 Plan)

**目标**: 修复 debug1 遗留问题，打通真实模型 CLI 链路，修正时长计算逻辑，完善错误处理和测试覆盖。

**执行日期**: 2026-04-01  
**依赖**: debug1 已完成的基础代码

> 执行状态（2026-04-02）: 已完成。实施细节与验证结果见 `informal-docs/debug2-report.md`。

---

## 核心架构修正（重要！）

### 正确的参数职责分离

| 参数 | 所属 | 职责 | 说明 |
|------|------|------|------|
| `generation_interval_ticks` | **Client 专用** | 控制多久触发一次推理请求 | 纯调度参数，Server 不应关心 |
| `generation_length_frames` | **Server 参数** | 控制生成多长的伴奏 | 决定模型输出长度 |

**关键理解**: 这两个参数应该是无关的！Server 不应该用 `interval` 来计算生成长度。

### debug1 中的错误

```python
# lekai_http_backend.py:211 - 错误！
num_beats_to_generate = max(1, generation_length_frames // generation_interval_ticks)
#                                            应该用 4 而不是 interval！
```

**错误分析**:
- Client 每 X ticks 请求一次（interval）
- Server 应该生成 Y ticks 的伴奏（length）
- 两者没有数学关系！

**正确计算**:
```python
num_beats = generation_length_frames // 4  # 模型固定 4 timesteps/beat
```

---

## 问题清单与修复方案

### 🔴 P0 - Critical

#### 问题 1: 时长计算逻辑错误（架构级错误）

**问题描述**:
- 当前：`num_beats = length // interval`
- 正确：`num_beats = length // 4`（模型固定 4 timesteps/beat）
- 后果：interval ≠ 4 时，输出长度与请求不符

**示例**:
```python
# 用户期望生成长度 = 20 ticks
length = 20
interval = 8  # client 每 8 ticks 请求一次

# 错误计算（当前）
beats = 20 // 8  # = 2 beats → 输出 8 ticks ❌

# 正确计算（修复后）
beats = 20 // 4  # = 5 beats → 输出 20 ticks ✅
```

**修复方案**:

```python
# lekai_http_backend.py
# 修改 _generate_with_model 方法

# 删除：
# num_beats_to_generate = max(1, generation_length_frames // generation_interval_ticks)

# 改为：
# 模型固定 4 timesteps/beat，与 generation_interval_ticks 无关
TIMESTEPS_PER_BEAT = 4
num_beats_to_generate = max(1, generation_length_frames // TIMESTEPS_PER_BEAT)

# 处理余数：如果不能整除，向上取整确保覆盖完整长度
if generation_length_frames % TIMESTEPS_PER_BEAT != 0:
    print(f"[Warning] generation_length_frames ({generation_length_frames}) "
          f"not divisible by {TIMESTEPS_PER_BEAT}, rounding up")
```

**Server 端校验简化**:

```python
# server_lekai.py - 删除对 generation_interval_ticks 的校验
# 因为它只是 Client 的调度参数！

@app.post("/generate_accompaniment")
async def generate_accompaniment(request: InferenceRequest) -> AccompanimentResponse:
    # 只校验 model_name
    if request.model_name == "lekai":
        # 只校验 length 是 4 的倍数（模型限制）
        if request.generation_length_frames % 4 != 0:
            raise HTTPException(
                status_code=422,
                detail=f"generation_length_frames must be multiple of 4 "
                       f"(model uses fixed 4-timesteps-per-beat tokenization, "
                       f"got {request.generation_length_frames})"
            )
    
    # 不再校验 generation_interval_ticks！
    # interval 是 client 的调度参数，server 不关心
```

**Client 端同步更新**:

```python
# inference_factory.py - 简化校验
if cfg.model_name == "lekai":
    # 只校验 length
    if int(cfg.generation_length_frames) % 4 != 0:
        raise ValueError(
            f"lekai model requires --generation-length-frames to be a multiple of 4 "
            f"(got {cfg.generation_length_frames})"
        )
    # 删除对 generation_interval_ticks 的校验
```

---

#### 问题 2: 真实模型 CLI 链路未打通

**问题描述**:
- Server 启动时 `backend = LekaiHttpBackend()` 无参初始化
- 不读取 `LEKAI_CHECKPOINT_PATH` 环境变量
- 结果：用户设置环境变量后仍走 rule stub

**修复方案**:

```python
# server_lekai.py
import os

# 启动时读取环境变量
checkpoint_path = os.environ.get("LEKAI_CHECKPOINT_PATH")
backend = LekaiHttpBackend(checkpoint_path=checkpoint_path)

def main() -> None:
    import uvicorn
    
    host = os.environ.get("LEKAI_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("LEKAI_SERVER_PORT", "8000"))
    
    # 打印启动信息
    if checkpoint_path:
        print(f"✓ Loaded checkpoint: {checkpoint_path}")
        print("  Using real PianoLLaMA model")
    else:
        print("! No checkpoint specified (set LEKAI_CHECKPOINT_PATH)")
        print("  Using rule-based stub")
    
    print(f"Listening on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
```

**验证方法**:
```bash
# 测试 checkpoint 加载
LEKAI_CHECKPOINT_PATH=/path/to/model.pt python -m server_lekai
# 应看到: "✓ Loaded checkpoint: /path/to/model.pt"

# 测试无 checkpoint
python -m server_lekai
# 应看到: "! No checkpoint specified"
```

---

#### 问题 3: `beats_to_pianoroll` 空 beat 崩溃

**问题描述**:
- 输入 `beat_tokens = [[]]` 时崩溃
- `decompress_tokens` 内部 `np.stack` 需要至少一个数组

**修复方案**:

```python
def beats_to_pianoroll(
    beat_tokens: List[List[int]],
    tokenizer: PianoRollTokenizer,
    timesteps_per_beat: int = 4,
) -> np.ndarray:
    """..."""
    if not beat_tokens:
        return np.zeros((2, 88, 0), dtype=np.float32)
    
    beat_pianorolls = []
    
    for beat_idx, beat_compressed in enumerate(beat_tokens):
        # ===== 新增：空列表检查 =====
        if not beat_compressed:
            print(f"[beats_to_pianoroll] Warning: empty beat at {beat_idx}, skipping")
            continue
        
        # Handle bar token (255)
        if len(beat_compressed) == 1 and beat_compressed[0] == 255:
            empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
            beat_pianorolls.append(empty_beat)
            continue
        
        # Handle empty marker (169)
        if len(beat_compressed) == 1 and beat_compressed[0] == 169:
            empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
            beat_pianorolls.append(empty_beat)
            continue
        
        # 解码逻辑（带异常保护）
        try:
            compressed_array = np.array(beat_compressed, dtype=np.int64)
            tokens_matrix = tokenizer.decompress_tokens(
                compressed_array, 
                end_marker_id=171
            )
            
            if tokens_matrix.size == 0:
                print(f"[beats_to_pianoroll] Warning: empty decompress at {beat_idx}")
                continue
            
            beat_pianoroll = tokenizer.patch_tokens_to_image(tokens_matrix)
            
            if beat_pianoroll.shape[2] != timesteps_per_beat:
                beat_pianoroll = _adjust_timesteps(beat_pianoroll, timesteps_per_beat)
            
            beat_pianorolls.append(beat_pianoroll)
            
        except Exception as e:
            print(f"[beats_to_pianoroll] Error at beat {beat_idx}: {e}")
            # 生成空 beat 继续
            empty_beat = np.zeros((2, 88, timesteps_per_beat), dtype=np.float32)
            beat_pianorolls.append(empty_beat)
    
    if not beat_pianorolls:
        return np.zeros((2, 88, 0), dtype=np.float32)
    
    full_pianoroll = np.concatenate(beat_pianorolls, axis=2)
    return full_pianoroll
```

---

### 🟡 P1 - High

#### 问题 4: shape 校验改为 exception

**问题描述**:
- 当前是 warning: `print(f"[Warning] Shape mismatch...")`
- 应该 fail-fast

**修复方案**:

```python
# lekai_http_backend.py:233
expected_shape = (2, 88, expected_timesteps)
if part1_pianoroll.shape != expected_shape:
    raise RuntimeError(
        f"Pianoroll shape mismatch: expected {expected_shape}, "
        f"got {part1_pianoroll.shape}. "
        f"This indicates a bug in beats_to_pianoroll."
    )
```

---

#### 问题 5: 补充测试文件

**测试 1: Tokenizer Roundtrip**

```python
# tests/unit/infrastructure/inference/lekai_model/test_tokenizer.py
import numpy as np
import pytest
from streammuse.infrastructure.inference.lekai_model.my_tokenizer import PianoRollTokenizer


@pytest.fixture
def tokenizer():
    return PianoRollTokenizer(
        patch_h=1, patch_w=4, marker_offset=81,
        end_marker_part0=170, end_marker_part1=171,
        empty_marker=169
    )


def test_encode_decode_empty(tokenizer):
    original = np.zeros((2, 88, 4))
    compressed = tokenizer.encode(original)
    recovered = tokenizer.decode(compressed, end_marker_id=170)
    assert recovered.shape == (2, 88, 4)


def test_encode_decode_with_note(tokenizer):
    original = np.zeros((2, 88, 4))
    original[0, 40, :3] = 1
    original[1, 40, 0] = 1
    
    compressed = tokenizer.encode(original)
    recovered = tokenizer.decode(compressed, end_marker_id=170)
    
    assert recovered.shape == (2, 88, 4)
    np.testing.assert_allclose(original, recovered, atol=1e-6)
```

**测试 2: Beats to Pianoroll**

```python
# tests/unit/infrastructure/inference/lekai_model/test_inference_adapter.py
import numpy as np
import pytest
from streammuse.infrastructure.inference.lekai_model.inference_adapter import (
    beats_to_pianoroll, _adjust_timesteps
)
from streammuse.infrastructure.inference.lekai_model.my_tokenizer import PianoRollTokenizer


@pytest.fixture
def tokenizer():
    return PianoRollTokenizer(
        patch_h=1, patch_w=4, marker_offset=81,
        end_marker_part0=170, end_marker_part1=171,
        empty_marker=169
    )


def test_empty_beat_list(tokenizer):
    """Test empty list []."""
    result = beats_to_pianoroll([], tokenizer)
    assert result.shape == (2, 88, 0)


def test_empty_beat(tokenizer):
    """Test beat with empty inner list [[]]."""
    result = beats_to_pianoroll([[]], tokenizer)
    # Should not crash
    assert result.shape[0] == 2
    assert result.shape[1] == 88


def test_bar_token(tokenizer):
    result = beats_to_pianoroll([[255]], tokenizer)
    assert result.shape == (2, 88, 4)
    assert np.all(result == 0)


def test_real_beat(tokenizer):
    # Create and encode a beat
    pianoroll = np.zeros((2, 88, 4))
    pianoroll[0, 40, :3] = 1
    pianoroll[1, 40, 0] = 1
    
    compressed = tokenizer.encode(pianoroll)
    beat_tokens = [compressed.tolist()]
    
    result = beats_to_pianoroll(beat_tokens, tokenizer)
    
    assert result.shape == (2, 88, 4)
    assert np.any(result > 0)
```

---

### 🟢 P2 - Medium

#### 问题 6: BPM 硬编码（可选）

保持默认 120，或从环境变量读取：

```python
bpm = int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))
```

---

## 详细实施计划 (Detailed Todo List)

### Phase 0 — 准备与基线确认

**目标**: 确认当前测试基线，记录修改前状态

- [ ] **P0-1** 确认当前测试基线
  - [ ] P0-1.1 运行 `uv run pytest tests/ -q`，记录当前通过数量（预期 124 passed）
  - [ ] P0-1.2 记录当前代码基线 commit hash

- [ ] **P0-2** 确认问题可复现
  - [ ] P0-2.1 确认 `lekai_http_backend.py:211` 中 `num_beats_to_generate` 使用了 `generation_interval_ticks`
  - [ ] P0-2.2 确认 `lekai_http_backend.py:322` 中 `num_intervals` 使用了 `interval`（同一问题的 rule-based 路径）
  - [ ] P0-2.3 确认 `server_lekai.py:87` 中 `backend = LekaiHttpBackend()` 无参初始化
  - [ ] P0-2.4 确认 `LEKAI_CHECKPOINT_PATH` 在代码中无任何引用

**Exit Criteria**:
- [ ] P0-E1 测试基线数量已记录
- [ ] P0-E2 所有问题位置已确认

---

### Phase 1 — Critical Fixes（架构级修正）

**目标**: 修正时长计算逻辑、打通 checkpoint 加载链路

#### Task 1.1: 问题 1 — 时长计算逻辑修正

核心原则：**`generation_length_frames` = 纯 acc 输出长度（ticks）**，模型固定 4 timesteps/beat，与 `generation_interval_ticks`（client 调度参数）无关。

- [ ] **1.1.1** 修正 `_generate_with_model` 路径
  - [ ] 1.1.1.1 在 `lekai_http_backend.py` 类顶部或方法内定义常量 `TIMESTEPS_PER_BEAT = 4`
  - [ ] 1.1.1.2 修改 `_generate_with_model` 中的 `num_beats_to_generate` 计算：
    - 删除: `num_beats_to_generate = max(1, generation_length_frames // generation_interval_ticks)`
    - 改为: `num_beats_to_generate = max(1, generation_length_frames // TIMESTEPS_PER_BEAT)`
  - [ ] 1.1.1.3 添加余数处理或 warning（如果 `generation_length_frames % 4 != 0`）

- [ ] **1.1.2** 修正 `_generate_rule_based` 路径
  - [ ] 1.1.2.1 分析当前逻辑：`num_intervals = max(1, generation_length_frames // interval)`
    - 当前 `interval = generation_interval_ticks`，同时充当"chord 持续 ticks"
    - 问题：如果 `interval=8, length=20`，生成 `20//8=2` 个 chord = 16 ticks，少了 4 ticks
  - [ ] 1.1.2.2 引入独立的 chord 持续时长概念，不再复用 `generation_interval_ticks`：
    - 方案 A（推荐）：stub 的 chord 持续时长固定为 `TIMESTEPS_PER_BEAT = 4`，与模型对齐
      ```python
      chord_duration = TIMESTEPS_PER_BEAT  # 每个 chord 持续 1 beat = 4 ticks
      num_intervals = max(1, generation_length_frames // chord_duration)
      ```
    - 方案 B：保持 chord 持续时长 = interval，但 num_intervals 用 generation_length_frames 算
  - [ ] 1.1.2.3 更新循环中的 `offset` 和 `note_off_tick` 计算，使用 `chord_duration` 而非 `interval`
  - [ ] 1.1.2.4 确保生成的伴奏总长度 = `num_intervals * chord_duration` ≈ `generation_length_frames`

- [ ] **1.1.3** 修正 `_generate_with_model` 中的 pianoroll 长度
  - [ ] 1.1.3.1 检查 `part1_pianoroll` 的 shape 计算是否也误用了 interval
  - [ ] 1.1.3.2 确保 `part1_pianoroll.shape[2] = num_beats_to_generate * TIMESTEPS_PER_BEAT = generation_length_frames`

- [ ] **1.1.4** Server 端校验简化
  - [ ] 1.1.4.1 修改 `server_lekai.py` 中的 `_validate_lekai_constraints()`：
    - 删除对 `generation_interval_ticks` 的校验
    - 只保留对 `generation_length_frames % 4 != 0` 的校验
    - 更新错误信息，说明原因（模型固定 4 timesteps/beat）
  - [ ] 1.1.4.2 简化函数签名，移除 `generation_interval_ticks` 参数
  - [ ] 1.1.4.3 更新调用处 `generate_accompaniment` endpoint

- [ ] **1.1.5** Client 端校验同步
  - [ ] 1.1.5.1 修改 `inference_factory.py`：
    - 删除对 `generation_interval_ticks % 4` 的校验和 ValueError
    - 只保留对 `generation_length_frames % 4` 的校验
  - [ ] 1.1.5.2 确认 `generation_interval_ticks` 不再有"必须是 4 的倍数"的约束

- [ ] **1.1.6** 文档更新
  - [ ] 1.1.6.1 更新 `docs/reference/cli-reference.md`：
    - 修改 "Lekai 模型参数约束" 章节
    - 删除 `generation_interval_ticks` 的约束行
    - 只保留 `generation_length_frames` 必须是 4 的倍数
  - [ ] 1.1.6.2 更新 `docs/getting-started/configuration.md`：
    - 修改注意事项，只提 `generation_length_frames` 约束
  - [ ] 1.1.6.3 更新 `docs/user-guide/running-realtime.md`：
    - 删除 Lekai 使用说明中关于 interval 必须是 4 的倍数的提示

- [ ] **1.1.7** 回归测试
  - [ ] 1.1.7.1 运行 `uv run pytest tests/ -q`，确认全部通过
  - [ ] 1.1.7.2 手动验证：`generation_length_frames=20, interval=4` → 生成 20 ticks ✓
  - [ ] 1.1.7.3 手动验证：`generation_length_frames=20, interval=8` → 生成 20 ticks ✓（不再是 16）
  - [ ] 1.1.7.4 手动验证：`generation_length_frames=20, interval=2` → 生成 20 ticks ✓

#### Task 1.2: 问题 2 — 打通 checkpoint 加载链路

- [ ] **1.2.1** Server 启动时读取环境变量
  - [ ] 1.2.1.1 在 `server_lekai.py` 模块级别添加 `checkpoint_path = os.environ.get("LEKAI_CHECKPOINT_PATH")`
  - [ ] 1.2.1.2 修改 `backend = LekaiHttpBackend()` 为 `backend = LekaiHttpBackend(checkpoint_path=checkpoint_path)`
  - [ ] 1.2.1.3 确保 `import os` 在模块顶部

- [ ] **1.2.2** 改善启动信息
  - [ ] 1.2.2.1 在 `main()` 中添加 checkpoint 状态打印：
    - 有 checkpoint → 打印路径和模式（real model）
    - 无 checkpoint → 打印提示和模式（rule-based stub）
  - [ ] 1.2.2.2 打印 `backend._has_real_model()` 的结果，确认模型是否实际加载成功

- [ ] **1.2.3** 验证
  - [ ] 1.2.3.1 测试无 checkpoint 启动：`python -m src.streammuse.infrastructure.inference.server_lekai`
  - [ ] 1.2.3.2 测试有 checkpoint 启动（如 checkpoint 可用）
  - [ ] 1.2.3.3 确认 `LEKAI_CHECKPOINT_PATH` 设置无效路径时，graceful fallback 到 stub

#### Phase 1 回归验证

- [ ] **P1-INT-1** 全量测试
  - [ ] P1-INT-1.1 `uv run pytest tests/ -q` 全部通过
  - [ ] P1-INT-1.2 启动 lekai server，运行 CLI `--model-name lekai --generation-interval-ticks 4 --generation-length-frames 20`，确认无崩溃

**Exit Criteria**:
- [ ] P1-E1 `_generate_with_model` 中 `num_beats = length // 4`，不再依赖 interval
- [ ] P1-E2 `_generate_rule_based` 中 chord 持续时长与 interval 解耦
- [ ] P1-E3 Server 不再校验 `generation_interval_ticks`
- [ ] P1-E4 Client factory 不再校验 `generation_interval_ticks`
- [ ] P1-E5 文档中不再声明 `generation_interval_ticks` 需要是 4 的倍数
- [ ] P1-E6 `LEKAI_CHECKPOINT_PATH` 能被正确读取并传入 backend
- [ ] P1-E7 所有现有测试通过

---

### Phase 2 — 鲁棒性修复

**目标**: 修复 `beats_to_pianoroll` 崩溃问题，加强 shape 校验

#### Task 2.1: 问题 3 — `beats_to_pianoroll` 空 beat 崩溃

- [ ] **2.1.1** 实现完整的 `beats_to_pianoroll` 函数
  - [ ] 2.1.1.1 替换 `inference_adapter.py` 中当前的全零占位符实现
  - [ ] 2.1.1.2 添加空列表 `[]` 入口检查，返回 `np.zeros((2, 88, 0))`
  - [ ] 2.1.1.3 添加空内部列表 `[[]]` 检查，跳过并 warning
  - [ ] 2.1.1.4 处理 bar token `[255]`：生成空 beat（全零，shape `(2, 88, 4)`）
  - [ ] 2.1.1.5 处理 empty marker `[169]`：生成空 beat
  - [ ] 2.1.1.6 实现正常 beat 的解码：`compressed → decompress_tokens → patch_tokens_to_image`
  - [ ] 2.1.1.7 添加 try/except 异常保护，单个 beat 解码失败不影响整体
  - [ ] 2.1.1.8 添加 `_adjust_timesteps` 辅助函数（padding/truncation 对齐到 `timesteps_per_beat`）

- [ ] **2.1.2** 同步更新 `_generate_with_model`
  - [ ] 2.1.2.1 将 `_generate_with_model` 中的硬编码 `np.zeros(...)` 替换为调用新的 `beats_to_pianoroll`
  - [ ] 2.1.2.2 移除 `# TODO: Proper token decoding to pianoroll` 注释

#### Task 2.2: 问题 4 — shape 校验改为 exception

- [ ] **2.2.1** 在 `_generate_with_model` 中添加 shape 校验
  - [ ] 2.2.1.1 在 `beats_to_pianoroll` 返回后，校验 `part1_pianoroll.shape`：
    ```python
    expected_timesteps = num_beats_to_generate * TIMESTEPS_PER_BEAT
    expected_shape = (2, 88, expected_timesteps)
    if part1_pianoroll.shape != expected_shape:
        raise RuntimeError(f"Pianoroll shape mismatch: expected {expected_shape}, got {part1_pianoroll.shape}")
    ```
  - [ ] 2.2.1.2 确保此异常会被 server 的错误处理捕获并返回 500

#### Phase 2 回归验证

- [ ] **P2-INT-1** 全量测试
  - [ ] P2-INT-1.1 `uv run pytest tests/ -q` 全部通过
  - [ ] P2-INT-1.2 确认 `beats_to_pianoroll([])` 返回 `(2, 88, 0)` 不崩溃
  - [ ] P2-INT-1.3 确认 `beats_to_pianoroll([[]])` 不崩溃

**Exit Criteria**:
- [ ] P2-E1 `beats_to_pianoroll` 对空输入不崩溃
- [ ] P2-E2 shape 不匹配时 raise RuntimeError 而非静默
- [ ] P2-E3 `_generate_with_model` 使用真正的 token 解码而非全零占位
- [ ] P2-E4 所有现有测试通过

---

### Phase 3 — 测试补充

**目标**: 为新增和修改的代码补充单元测试

#### Task 3.1: Tokenizer roundtrip 测试

- [ ] **3.1.1** 创建 `tests/unit/infrastructure/inference/lekai_model/test_tokenizer.py`
  - [ ] 3.1.1.1 测试 `encode → decode` roundtrip（空 pianoroll）
  - [ ] 3.1.1.2 测试 `encode → decode` roundtrip（有音符的 pianoroll）
  - [ ] 3.1.1.3 测试 `compress → decompress` roundtrip
  - [ ] 3.1.1.4 测试边界情况：全零、全一、单个 patch

#### Task 3.2: `beats_to_pianoroll` 测试

- [ ] **3.2.1** 创建 `tests/unit/infrastructure/inference/lekai_model/test_inference_adapter.py`
  - [ ] 3.2.1.1 测试空列表 `[]` → shape `(2, 88, 0)`
  - [ ] 3.2.1.2 测试空内部列表 `[[]]` → 不崩溃
  - [ ] 3.2.1.3 测试 bar token `[[255]]` → shape `(2, 88, 4)` 全零
  - [ ] 3.2.1.4 测试 empty marker `[[169]]` → shape `(2, 88, 4)` 全零
  - [ ] 3.2.1.5 测试真实 beat：`encode → beats_to_pianoroll` roundtrip 验证 shape 和内容

#### Task 3.3: 时长计算测试

- [ ] **3.3.1** 在现有 `test_lekai_http_backend.py` 中添加（或创建）
  - [ ] 3.3.1.1 测试 `_generate_rule_based`：`length=20, interval=4` → 生成 20 ticks 的伴奏
  - [ ] 3.3.1.2 测试 `_generate_rule_based`：`length=20, interval=8` → 仍生成 20 ticks（不再是 16）
  - [ ] 3.3.1.3 测试 `_generate_rule_based`：`length=20, interval=2` → 仍生成 20 ticks
  - [ ] 3.3.1.4 测试 `_generate_rule_based`：空 melody → 返回 `[]`

#### Task 3.4: 校验逻辑测试

- [ ] **3.4.1** 测试 `inference_factory.py` 校验
  - [ ] 3.4.1.1 测试 `generation_length_frames=20` → ValueError（不是 4 的倍数）
  - [ ] 3.4.1.2 测试 `generation_length_frames=16` → 通过
  - [ ] 3.4.1.3 测试 `generation_interval_ticks=3` → 不再报错（interval 不受约束）

- [ ] **3.4.2** 测试 `server_lekai.py` 校验
  - [ ] 3.4.2.1 测试 `generation_length_frames=17` → HTTP 422
  - [ ] 3.4.2.2 测试 `generation_interval_ticks=3` → 不再 422

#### Phase 3 回归验证

- [ ] **P3-INT-1** 全量测试
  - [ ] P3-INT-1.1 `uv run pytest tests/ -q` 全部通过
  - [ ] P3-INT-1.2 `uv run pytest tests/ --collect-only -q` 确认测试数量增长

**Exit Criteria**:
- [ ] P3-E1 新增测试覆盖所有修改点
- [ ] P3-E2 时长计算测试验证 interval 不影响输出长度
- [ ] P3-E3 所有测试通过

---

### Phase 4 — 回归验证与文档收尾

**目标**: 全面回归，确保修改不影响 Stanley 和其他模式

#### Task 4.1: Stanley 模式回归

- [ ] **4.1.1** 运行 Stanley 相关测试
  - [ ] 4.1.1.1 `uv run pytest tests/ -k "stanley" -v` 全部通过
  - [ ] 4.1.1.2 确认 `inference_factory.py` 的修改不影响 Stanley 路径
  - [ ] 4.1.1.3 确认 client 端 `_last_sent_index`（debug1 修改）在 Stanley 模式下仍正常

#### Task 4.2: Lekai 模式端到端验证

- [ ] **4.2.1** stub 模式
  - [ ] 4.2.1.1 启动 `server_lekai`（无 checkpoint），确认打印 "rule-based stub"
  - [ ] 4.2.1.2 CLI `--model-name lekai --generation-interval-ticks 2 --generation-length-frames 16`
  - [ ] 4.2.1.3 确认输出正常，无崩溃

- [ ] **4.2.2** 参数组合测试
  - [ ] 4.2.2.1 `interval=4, length=16` → 正常
  - [ ] 4.2.2.2 `interval=8, length=16` → 正常（之前这种组合可能生成长度不对）
  - [ ] 4.2.2.3 `interval=2, length=16` → 正常
  - [ ] 4.2.2.4 `interval=3, length=16` → 正常（interval 不再有约束）
  - [ ] 4.2.2.5 `interval=4, length=17` → 报错（length 必须是 4 的倍数）

#### Task 4.3: 文档完整性确认

- [ ] **4.3.1** 确认文档一致性
  - [ ] 4.3.1.1 `docs/reference/cli-reference.md` 中 interval 约束已删除
  - [ ] 4.3.1.2 `docs/getting-started/configuration.md` 中只提 length 约束
  - [ ] 4.3.1.3 `docs/user-guide/running-realtime.md` 中 lekai 说明已更新
  - [ ] 4.3.1.4 文档中的示例命令仍可正常运行

#### Task 4.4: debug2-report 编写

- [ ] **4.4.1** 编写实施报告
  - [ ] 4.4.1.1 创建 `informal-docs/debug2-report.md`
  - [ ] 4.4.1.2 记录所有修改文件和修改内容
  - [ ] 4.4.1.3 记录测试结果
  - [ ] 4.4.1.4 记录已知限制

**Exit Criteria**:
- [ ] P4-E1 所有测试通过（含新增测试）
- [ ] P4-E2 Stanley 模式无回归
- [ ] P4-E3 文档与代码一致
- [ ] P4-E4 debug2-report 完成

---

## 附录 A：测试矩阵

| 测试场景 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---------|---------|---------|---------|---------|
| 时长计算（model 路径）| ✅ 修复 | - | ✅ 测试 | ✅ 回归 |
| 时长计算（stub 路径）| ✅ 修复 | - | ✅ 测试 | ✅ 回归 |
| Server interval 校验移除 | ✅ 修复 | - | ✅ 测试 | ✅ 回归 |
| Client interval 校验移除 | ✅ 修复 | - | ✅ 测试 | ✅ 回归 |
| Checkpoint 环境变量 | ✅ 修复 | - | - | ✅ 回归 |
| beats_to_pianoroll 空输入 | - | ✅ 修复 | ✅ 测试 | ✅ 回归 |
| shape 校验 exception | - | ✅ 修复 | - | ✅ 回归 |
| Tokenizer roundtrip | - | - | ✅ 测试 | ✅ 回归 |
| Stanley 兼容性 | - | - | - | ✅ 回归 |

## 附录 B：修改文件清单（预计）

| 文件 | Phase | 修改类型 |
|------|-------|---------|
| `src/streammuse/infrastructure/inference/lekai_http_backend.py` | 1, 2 | 修正时长计算 + shape 校验 |
| `src/streammuse/infrastructure/inference/server_lekai.py` | 1 | 校验简化 + checkpoint 加载 |
| `src/streammuse/application/factories/inference_factory.py` | 1 | 校验简化 |
| `src/streammuse/infrastructure/inference/lekai_model/inference_adapter.py` | 2 | 实现 beats_to_pianoroll |
| `docs/reference/cli-reference.md` | 1 | 移除 interval 约束 |
| `docs/getting-started/configuration.md` | 1 | 更新注意事项 |
| `docs/user-guide/running-realtime.md` | 1 | 更新 lekai 说明 |
| `tests/unit/.../test_tokenizer.py` | 3 | 新增 |
| `tests/unit/.../test_inference_adapter.py` | 3 | 新增 |
| `tests/unit/.../test_lekai_http_backend.py` | 3 | 新增/修改 |
| `informal-docs/debug2-report.md` | 4 | 新增 |

## 附录 C：关键修复代码片段

### 1. 时长计算修复

```python
# lekai_http_backend.py
TIMESTEPS_PER_BEAT = 4  # 模型固定参数

num_beats_to_generate = max(1, generation_length_frames // TIMESTEPS_PER_BEAT)
```

### 2. Server 校验简化

```python
# server_lekai.py
if request.model_name == "lekai":
    if request.generation_length_frames % 4 != 0:
        raise HTTPException(status_code=422, detail="...")
# 不再校验 generation_interval_ticks
```

### 3. Checkpoint 加载

```python
# server_lekai.py
checkpoint_path = os.environ.get("LEKAI_CHECKPOINT_PATH")
backend = LekaiHttpBackend(checkpoint_path=checkpoint_path)
```
