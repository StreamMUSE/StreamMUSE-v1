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
音乐的基本节拍单位。默认 1 拍 = 4 ticks。

**Bar / Measure（小节）**
由 `beats_per_bar` 个拍组成，默认 4 拍一小节。

**Count-in**
正式演奏前的预备拍。StreamMUSE 中 `--count-in-beats N` 会在正式 tick=0 前空转 N 拍，只输出 metronome，不读取正式输入，不发送推理请求。

**Metronome**
节拍器 click。当前实现默认用 MIDI channel 9（GM percussion channel 10），downbeat note=76 velocity=110，普通 beat note=77 velocity=80。

**Polyphony（复音）**
同一时刻发出的音符数量。

**NOTE_ON / NOTE_OFF**
MIDI 事件类型。`NOTE_ON`（velocity > 0）触发音符；`NOTE_OFF` 或 `NOTE_ON`（velocity = 0）释放音符。

**MIDI Program（乐器编号）**
MIDI 标准中的乐器编号（0–127），对应 General MIDI 音色集。如 0=钢琴。

**Velocity（力度）**
音符的击键力度（0–127）。0 等效于 NOTE_OFF。

---

## 技术术语

**Clean Architecture**
Robert C. Martin 提出的软件架构模式，将系统分为 Presentation → Application → Domain → Infrastructure，外层依赖内层，内层不依赖外层。

**Protocol（协议）**
Python 中的结构化子类型接口，通过 `typing.Protocol` 定义。StreamMUSE 中的 `InputSource`、`OutputSink`、`InferenceEngine` 均为 Protocol。

**Domain（领域层）**
系统的核心业务规则层，不依赖 I/O 或框架。StreamMUSE 的 Domain 层包含 `MusicalEvent`、`Tempo`、`PlaybackScheduler` 等。

**Infrastructure（基础设施层）**
系统的最外层，包含 MIDI 设备、HTTP 客户端、文件 I/O、模型适配器等具体实现。

**Tick=0 约定**
InputSource 发出的原始 `MusicalEvent` 通常使用 `tick=0`，由 Application 层根据正式时间线的 wall-clock 赋值实际 tick。

**Beat-tail trigger**
当前实时推理触发策略：默认 `ticks_per_beat=4` 时，在 tick=3、7、11... 发送下一拍请求，`generation_start_tick=tick+1`。

**Latest-only inference**
推理线程处理队列积压时只保留最新请求 tick，并合并被跳过请求中的 melody 增量，避免旧请求堆积。

**close-at-horizon**
`events_to_notes()` 转换策略：在 `horizon_tick` 时仍未结束的音符，在该 tick 处截断，而不是丢弃。

**Fan-out（扇出）**
`CompositeOutputSink` 的工作模式：将一个方法调用广播给所有子 Sink。

**Injection**
会话开始前把已有 melody/accompaniment 历史注入 server。CLI 通过 `--injection-file` 和 `--injection-length` 支持 MIDI 文件注入。

---

## 系统组件术语

**InferenceEngine**
推理引擎接口，负责根据旋律历史生成伴奏（`generate_accompaniment()`），并可选支持 `inject_history()` / `clear_history()`。

**InputSource**
输入源接口，通过 `read_events()` 生成器提供 `MusicalEvent` 流。

**OutputSink**
输出 Sink 接口，接收事件、tick、统计和状态信息并处理（播放、录制、记录等）。

**PlaybackScheduler**
线程安全的事件调度器，按 tick 安排未来事件，供 `_tick_loop` 在正确时间播放。

**RealTimeMusicService**
系统核心三线程编排服务：`_input_worker`、`_tick_loop`、`_inference_worker`。

**SessionManager**
会话目录管理器，负责创建带时间戳的日志目录和保存会话元数据。

**TimingInfo**
推理时序信息 dataclass，包含请求到达、推理开始/结束、响应发送等时间戳字段。

---

## 缩写

| 缩写 | 全称 |
|---|---|
| MIDI | Musical Instrument Digital Interface |
| BPM | Beats Per Minute |
| CLI | Command-Line Interface |
| API | Application Programming Interface |
| HTTP | HyperText Transfer Protocol |
| JSON | JavaScript Object Notation |
| JSONL | JSON Lines |
| GM | General MIDI |
