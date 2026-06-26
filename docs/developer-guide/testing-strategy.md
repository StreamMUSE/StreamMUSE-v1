---
title: 测试策略
description: StreamMUSE 的单元测试和集成测试组织方式
---

# 测试策略

StreamMUSE 的测试分为两层：**单元测试**（快速、无 I/O）和**集成测试**（测试 CLI 入口）。

---

## 运行测试

```bash
# 运行所有测试
uv run pytest tests/

# 简洁输出
uv run pytest tests/ -q --tb=no

# 只运行单元测试
uv run pytest tests/unit/

# 只运行集成测试
uv run pytest tests/integration/
```

---

## 目录结构

```
tests/
├── __init__.py
├── unit/
│   ├── domain/         # Domain 层（纯逻辑）
│   ├── application/    # Application 层（使用 mock）
│   └── infrastructure/ # Infrastructure 层（具体实现）
└── integration/        # CLI 入口测试
```

---

## 单元测试原则

### Domain 层（无依赖）

Domain 层中的类无外部依赖，可直接实例化测试：

```python
# tests/unit/domain/test_tempo.py
from streammuse.domain.timing.tempo import Tempo

def test_tick_to_seconds():
    tempo = Tempo(bpm=120.0, ticks_per_beat=4)
    assert abs(tempo.tick_to_seconds(4) - 0.5) < 1e-6   # 1 拍 = 0.5s
```

### Infrastructure 层（避免 I/O）

输入适配器可通过内部方法直接测试，避免依赖 GUI 或真实设备：

```python
# tests/unit/infrastructure/test_keyboard_input.py
from streammuse.infrastructure.input.keyboard import KeyboardInput

def test_key_down_emits_note_on():
    source = KeyboardInput()
    source._handle_key_down("z")   # 直接调用内部方法
    # 从 _events 队列检查事件
    event = source._events.get_nowait()
    assert event.pitch == 60
    assert event.event_type.value == "note_on"
```

### Application 层（使用 Mock）

通过 Mock 实现 Domain 协议，测试服务逻辑而不依赖实际 I/O：

```python
# tests/unit/application/test_real_time_service.py
from unittest.mock import MagicMock
from streammuse.application.services.real_time_music_service import RealTimeMusicService

def test_service_starts_and_stops():
    input_mock = MagicMock()
    input_mock.read_events.return_value = iter([])
    output_mock = MagicMock()
    engine_mock = MagicMock()
    engine_mock.generate_accompaniment.return_value = ([], make_timing_info())

    service = RealTimeMusicService(
        input_source=input_mock,
        output_sink=output_mock,
        inference_engine=engine_mock,
        tempo=Tempo(120.0, 4),
        scheduler=PlaybackScheduler(),
    )
    service.start(max_ticks=4)
    import time; time.sleep(0.2)
    service.stop()
    assert not service.running
```


---

## Consistency Tests（真实模型，默认跳过）

`tests/consistency/` 是 opt-in 的端到端回归测试，会启动本地 Lekai server、加载真实 checkpoint，并用 MIDI file 输入模拟 realtime。默认不设置 checkpoint env 时会 skip，不影响日常 `uv run pytest tests/`。

单阶段 consistency：

```bash
LEKAI_CHECKPOINT_PATH=<single-stage-checkpoint.safetensors> \
STREAMMUSE_CONSISTENCY_SONGS=4 \
STREAMMUSE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_realtime_offline_consistency.py -q -s
```

Two-stage prompt+continuation consistency：

```bash
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s
```

Two-stage consistency 比较的是 server 端保存的 `prompt_continuation_raw_history.json` / `prompt_continuation_prompt_history.json`，也就是实际用于 continuation 的 inference context；`combined.mid` playback/recording 另有已知调度问题，详见 `developing-logs/reports/2026-06-26-two-stage-consistency-implementation-report.md`。

---

## 集成测试

```python
# tests/integration/test_cli.py
from streammuse.presentation.cli.cli import main
# 通过 CLI 参数测试完整流程（通常使用 list 输入模式）
```

---

## Mock 最佳实践

**InferenceEngine Mock**：

```python
engine_mock = MagicMock()
engine_mock.generate_accompaniment.return_value = (
    [],  # 空伴奏列表
    TimingInfo(
        request_arrival_time=0.0,
        inference_start_time=0.0,
        inference_end_time=0.0,
        response_output_time=0.0,
        preprocess_start_time=0.0,
        postprocess_start_time=0.0,
    )
)
```

**时钟 Mock**（用于确定性测试）：

```python
fake_time = [0.0]
def fake_now(): return fake_time[0]
def fake_sleep(t): fake_time[0] += t

service = RealTimeMusicService(..., now=fake_now, sleep=fake_sleep)
```
