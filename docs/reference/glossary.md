---
title: 术语表
description: StreamMUSE 项目中使用的专业术语与缩写
---

# 术语表

---

## 音乐术语

**BPM（Beats Per Minute）**
每分钟拍数，衡量音乐速度。默认 120 BPM = 每秒 2 拍。

**Tick**
系统内部的最细时间单位。默认 `ticks_per_beat=4`，即 1 tick = 1/4 拍。在 120 BPM 下，1 tick = 0.125 秒。

**Beat（拍）**
音乐的基本节拍单位。1 拍 = 4 ticks（默认）。

**Bar / Measure（小节）**
由 `beats_per_bar`（默认 4）个拍组成。

**Polyphony（复音）**
同一时刻发出的音符数量。模型限制 `max_polyphony=4`，即最多 4 个音符同时发声。

**NOTE_ON / NOTE_OFF**
MIDI 事件类型。`NOTE_ON`（velocity > 0）触发音符；`NOTE_OFF` 或 `NOTE_ON`（velocity = 0）释放音符。

**MIDI Program（乐器编号）**
MIDI 标准中的乐器编号（0–127），对应 General MIDI 音色集。如 0=钢琴，40=小提琴。

**Velocity（力度）**
音符的击键力度（0–127）。0 等效于 NOTE_OFF。

---

## 技术术语

**Clean Architecture**
Robert C. Martin 提出的软件架构模式，将系统分为 4 层（Presentation → Application → Domain → Infrastructure），外层依赖内层，内层不依赖外层。

**Protocol（协议）**
Python 中的结构化子类型接口，通过 `typing.Protocol` 定义。StreamMUSE 中的 `InputSource`、`OutputSink`、`InferenceEngine` 均为 Protocol。

**Domain（领域层）**
系统的核心业务规则层，不依赖任何 I/O 或框架。StreamMUSE 的 Domain 层包含 `MusicalEvent`、`Tempo`、`PlaybackScheduler` 等。

**Infrastructure（基础设施层）**
系统的最外层，包含所有具体实现（MIDI 设备、HTTP 客户端、文件 I/O 等）。

**Tick=0 约定**
所有 InputSource 实现发出的 `MusicalEvent` 的 `tick` 字段均设为 0，由 Application 层根据实际时间戳赋值。

**close-at-horizon**
`events_to_notes()` 转换策略：在 `horizon_tick` 时仍未结束的音符，在该 tick 处截断（关闭），而不是丢弃。

**Fan-out（扇出）**
`CompositeOutputSink` 的工作模式：将一个方法调用广播给所有子 Sink。

---

## 系统组件术语

**InferenceEngine**
推理引擎接口，负责根据旋律历史生成伴奏（`generate_accompaniment()`）。

**InputSource**
输入源接口，通过 `read_events()` 生成器提供 `MusicalEvent` 流。

**OutputSink**
输出 Sink 接口，接收事件、tick、统计和状态信息并处理（播放、录制、记录等）。

**PlaybackScheduler**
线程安全的事件调度器，按 tick 安排未来事件，供 `_tick_loop` 在正确时间播放。

**RealTimeMusicService**
系统核心，3 线程编排服务（`_input_worker`、`_tick_loop`、`_inference_worker`）。

**SessionManager**
会话目录管理器，负责创建带时间戳的日志目录和保存会话元数据。

**TimingInfo**
推理时序信息 dataclass，包含请求到达、推理开始/结束、响应发送等 9 个时间戳字段。

---

## 缩写

| 缩写 | 全称 |
|---|---|
| MIDI | Musical Instrument Digital Interface（乐器数字接口） |
| BPM | Beats Per Minute |
| CLI | Command-Line Interface |
| RoFormer | Rotary Position Embedding Transformer |
| API | Application Programming Interface |
| HTTP | HyperText Transfer Protocol |
| JSON | JavaScript Object Notation |
| JSONL | JSON Lines（每行一个 JSON 对象） |
