# 推理引擎数据结构对比

本文档对比了 **Stanley (RoFormer)** 和 **Lekai (LLaMA)** 两个推理引擎的数据结构和输入格式。

## 1. 高层 API 输入 (完全一致)

两个引擎都实现了相同的 `InferenceEngine` **领域协议**。它们在应用层面上设计为可互换的。

**协议签名 (`domain/interfaces/inference.py`):**
```python
def generate_accompaniment(
    self,
    melody_events: List[MusicalEvent],
    generation_start_tick: int,
    generation_length_frames: int,
    prompt_length_ticks: int | None = None,
) -> tuple[List[MusicalEvent], TimingInfo]:
```

**统一数据格式:**
两个引擎的输入和输出均使用 `MusicalEvent` 对象（共享的领域类型）：

| 字段 | 类型 | 描述 |
| :--- | :--- | :--- |
| `tick` | `int` | 绝对开始时间 (ticks)。 |
| `pitch` | `int` | MIDI 音高 (0-127)。 |
| `event_type` | `EventType` | `NOTE_ON` 或 `NOTE_OFF`。 |
| `velocity` | `int` | 0–127。 |
| `channel` | `int` | MIDI 通道。 |
| `program` | `int` | MIDI 程序号（乐器）。 |
| `source` | `str` | `"user"` 或 `"model"`。 |

**适配器说明:** Stanley 引擎 (`StanleyInferenceEngine`) 包装了一个操作 `dict[pitch, tick, duration]` 格式的遗留实现。适配器在输入时将 `MusicalEvent` 转换为 dict，在输出时将 dict 转换回 `MusicalEvent`。调用方始终只使用 `MusicalEvent`。

**时间分辨率:**
*   两个引擎通常都在 **每拍 4 ticks** (16分音符) 的网格上运行。
*   `tick` 值是基于此分辨率的整数。

---

## 2. 内部数据表示 (主要差异)

虽然输入是相同的，但引擎在内部处理这些数据以准备给各自模型使用的方式非常不同。

### A. Stanley 引擎 (RoFormer / 符号化)

Stanley 引擎将音符列表转换为具有固定维度的 **符号张量 (Symbolic Tensor)**。

*   **表示:** `(时间, 复调数, 特征)` 张量。
*   **形状:** `(序列长度, 最大复调数, 3)`
*   **特征:** 每个音符槽包含 `[乐器程序号, 音高, 时值索引]`。
*   **限制:**
    1.  **最大复调数 (Max Polyphony):** 硬限制 (默认 4)。如果在同一 tick 出现的音符超过 4 个，**多余的音符会被丢弃**。
    2.  **时值模板 (Duration Templates):** 时值不存储为原始整数。它们被量化为预定义 `DURATION_TEMPLATES` 列表中的最近值 (例如 1, 2, 4, 8, 16...)。这意味着不寻常的时值可能会被吸附到标准的音乐长度。

### B. Lekai 引擎 (LLaMA / Piano Roll -> Token)

Lekai 引擎将音符列表转换为 **钢琴卷帘 (Piano Roll)**，然后进行 Token 化。

*   **表示:** 转换为 **Token** 的钢琴卷帘图像切片。
*   **中间形状:** `(2, 88, 时间)` numpy 数组。
    *   通道 0: 延音 (Sustain) (如果音符处于激活状态则为 1)。
    *   通道 1: 起始 (Onset) (如果音符开始则为 1)。
*   **Token 化:** 钢琴卷帘被压缩成整数 Token 序列 (例如 `[BOS, TimeSig, BPM, Bar, Mel_Tokens, Bar, Acc_Tokens...]`)。
*   **限制:**
    1.  **复调数:** 对同时发声的音符数量没有硬性限制 (最多 88 个，每个音高一个)。它 **不会** 基于复调计数丢弃音符。
    2.  **时值:** 时值由钢琴卷帘网格中 "延音" 线条的长度表示。它支持任何符合网格分辨率的整数时值。

## 3. 差异总结

| 特性 | Stanley 引擎 (RoFormer) | Lekai 引擎 (LLaMA) |
| :--- | :--- | :--- |
| **输入格式** | `List[MusicalEvent]`（适配器内部转换为字典） | `List[MusicalEvent]`（适配器内部转换为钢琴卷帘） |
| **复调处理** | **受限** (每 tick 最多 4 个音符)。丢弃多余音符。 | **灵活** (隐式支持最多 88 个)。 |
| **时值处理** | **量化到模板** (吸附到最近的标准时值)。 | **基于网格** (网格上的精确整数 ticks)。 |
| **模型输入** | 稠密张量 `(T, P, 3)` | Token 序列 (整数) |
| **上下文窗口** | 固定帧长度 (例如 384 帧)。 | Token 序列长度 (例如 32 拍)。 |

## 结论

*   **兼容性:** 您可以将完全相同的数据传递给两个引擎，而无需更改应用程序代码。
*   **保真度:** **Lekai** 引擎理论上能够表示更复杂的数据 (更高的复调性，非标准时值)，因为它不会强制将音符放入固定的复调槽或时值模板中。
