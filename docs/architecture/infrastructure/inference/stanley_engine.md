---
title: StanleyInferenceEngine — 接口适配层
description: InferenceEngine Protocol 实现，衔接 Domain 接口与 LegacyInferenceEngineStanley
---

# StanleyInferenceEngine — 接口适配层

**源文件**：`src/streammuse/infrastructure/inference/stanley_engine.py`

`StanleyInferenceEngine` 是 Clean Architecture 的适配器，将 Domain 的 `InferenceEngine` 协议（事件流接口）与旧有的 `LegacyInferenceEngineStanley`（duration-note 字典接口）连接起来。

---

## `StanleyInferenceConfig`

```python
@dataclass(frozen=True)
class StanleyInferenceConfig:
    checkpoint_path: str
    model_size: str = "0.12B"
    model_max_seq_len_frames: int = 96
    generation_length_frames: int = 20
    max_polyphony: int = 4
```

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `checkpoint_path` | `str` | 必填 | 模型 checkpoint 文件路径 |
| `model_size` | `str` | `"0.12B"` | 模型规模标识，传入 RoFormer 架构选择 |
| `model_max_seq_len_frames` | `int` | `96` | 模型 context window 帧数（96 帧 = 48 ticks = 12 拍） |
| `generation_length_frames` | `int` | `20` | 每次推理生成帧数（20 帧 = 10 ticks = 2.5 拍） |
| `max_polyphony` | `int` | `4` | 最大同时发音音符数 |

---

## `StanleyInferenceEngine`

### `__init__(*, config, legacy_engine=None)`

若未传入 `legacy_engine`，自动创建 `LegacyInferenceEngineStanley`（延迟导入，避免无模型时无法启动服务）。可注入自定义 `legacy_engine` 用于测试（遵循 `_LegacyStanleyLike` Protocol）。

---

### `generate_accompaniment(melody_events, generation_start_tick, generation_length_frames, ...) -> tuple[List[MusicalEvent], TimingInfo]`

**数据流**：

```
List[MusicalEvent]
    │ events_to_notes(horizon_tick=generation_start_tick)
    ▼
List[Note]
    │ 转为 List[dict]：{pitch, tick, duration, velocity, program}
    ▼
LegacyInferenceEngineStanley.generate_accompaniment(...)
    │ 返回 (acc_notes, preprocess_start, inf_start, inf_end, post_start)
    ▼
List[dict] → List[Note] → note.to_events() → List[MusicalEvent]
    │ 构建 TimingInfo（用 legacy 引擎返回的 4 个时间戳）
    ▼
(List[MusicalEvent], TimingInfo)
```

**关键转换**：使用 `events_to_notes(melody_events, horizon_tick=generation_start_tick)` 将事件流转为 duration-note 格式，应用 close-at-horizon 策略（仍在 `generation_start_tick` 时刻未结束的音符在该点截断）。

---

### `inject_history(melody_events, accompaniment_events, injection_length_ticks) -> None`

将旋律和伴奏事件转换为 note 字典，注入到 `_legacy` 引擎的历史记录中。

---

### `set_injection_offset(offset_ticks: int) -> None`

委托给 `_legacy.set_injection_offset(offset_ticks)`。

---

### `clear_history() -> Dict[str, Any]`

委托给 `_legacy.clear_history()` 清空旋律和伴奏历史，并返回统一的清理结果结构（`success/message/melody_history/accompaniment_history`，Stanley 场景下历史列表为空）。

---

## `_LegacyStanleyLike` Protocol

```python
class _LegacyStanleyLike(Protocol):
    generation_length_frames: int

    def generate_accompaniment(self, melody_notes, generation_start_tick, ...) -> tuple: ...
    def clear_history(self) -> None: ...
    def set_injection_offset(self, offset_ticks: int) -> None: ...
```

内部 Protocol，定义了 `StanleyInferenceEngine` 对遗留引擎的最小接口要求，使得测试时可以注入任意 mock 对象。
