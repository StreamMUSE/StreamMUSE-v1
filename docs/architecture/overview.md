---
title: 架构总览
description: StreamMUSE Clean Architecture 四层设计与数据流说明
---

# 架构总览

StreamMUSE 严格遵循 **Clean Architecture（洁净架构）** 原则，将系统划分为四个依赖单向流动的层次。

---

## 四层架构图

```
┌─────────────────────────────────────────────────────┐
│              Presentation 层                        │
│   src/streammuse/presentation/                      │
│   CLI 入口，解析参数，创建所有组件                        │
└────────────────────┬────────────────────────────────┘
                     │ 创建 ApplicationConfig
                     ▼
┌─────────────────────────────────────────────────────┐
│              Application 层                         │
│   src/streammuse/application/                       │
│   配置模型、Factory 工厂、RealTimeMusicService 服务      │
└────────────────────┬────────────────────────────────┘
                     │ 只依赖 Domain 接口（Protocol）
                     ▼
┌─────────────────────────────────────────────────────┐
│              Domain 层                              │
│   src/streammuse/domain/                            │
│   MusicalEvent、Tempo、Protocol 定义、日志领域对象       │
└────────────────────▲────────────────────────────────┘
                     │ Infrastructure 实现 Domain 接口
┌─────────────────────────────────────────────────────┐
│              Infrastructure 层                      │
│   src/streammuse/infrastructure/                    │
│   输入设备、输出 Sink、推理引擎的具体实现                  │
└─────────────────────────────────────────────────────┘
```

---

## 依赖规则

| 层 | 可以依赖 | 不可依赖 |
|---|---|---|
| Presentation | Application、Domain | Infrastructure（通过 Factory 间接使用） |
| Application | Domain | Infrastructure、Presentation |
| Domain | 无外部依赖 | 所有其他层 |
| Infrastructure | Domain | Application、Presentation |

**核心原则**：内层（Domain）完全不知道外层的存在。Application 层通过 Domain 定义的 Protocol（duck typing）与 Infrastructure 解耦，无需任何 `import` 具体实现类。

---

## 数据流：从输入到输出

```mermaid
sequenceDiagram
    participant User as 用户（键盘/MIDI）
    participant Input as InputSource
    participant Service as RealTimeMusicService
    participant Engine as InferenceEngine
    participant Output as OutputSink

    User->>Input: 按键 / MIDI 消息
    Input->>Service: MusicalEvent (source="user")
    Service->>Output: output_event(event, "user")
    
    Note over Service: 每 generation_interval_ticks 触发
    Service->>Engine: generate_accompaniment(melody_events, ...)
    Engine-->>Service: List[MusicalEvent] + TimingInfo
    Service->>Output: output_event(acc_event, "model")
    Service->>Output: output_stats(round_trip_ms, ...)
```

---

## 三线程模型

`RealTimeMusicService` 内部运行三个并发线程，通过队列通信：

| 线程 | 职责 | 输入源 | 输出目标 |
|---|---|---|---|
| `_input_worker` | 从 `InputSource` 读事件，打上 tick 时间戳 | `InputSource.read_events()` | `_event_q`、`_melody_history` |
| `_tick_loop` | 推进音乐时钟，调度播放，触发推理 | `_event_q`、`_inference_response_queue` | `OutputSink`、`_inference_request_queue` |
| `_inference_worker` | 调用推理引擎，将结果放回响应队列 | `_inference_request_queue` | `_inference_response_queue`、`OutputSink` |

```mermaid
graph LR
    IW[_input_worker] -->|MusicalEvent| EQ[_event_q]
    EQ --> TL[_tick_loop]
    TL -->|output_event| OS[OutputSink]
    TL -->|tick, melody_snapshot| IRQ[_inference_request_queue]
    IRQ --> InfW[_inference_worker]
    InfW -->|acc_events| IRSQ[_inference_response_queue]
    IRSQ --> TL
    InfW -->|output_stats| OS
```

---

## 关键设计决策

### Protocol 而非 ABC

Domain 层使用 Python `Protocol`（structural subtyping）而非抽象基类（ABC）。好处：

- Infrastructure 实现类无需继承任何基类，减少耦合
- 便于测试时使用 mock 对象或 `MagicMock`
- `InferenceEngine`、`InputSource`、`OutputSink` 均为 Protocol

### 不使用 asyncio

系统选择多线程而非 async/await，原因：

- `python-rtmidi`、`pynput` 等外部库使用阻塞式回调，不适合 async
- 多线程模型更易于理解推理延迟与音乐时钟对齐关系
- 三线程间通过 `queue.Queue` 通信，线程安全且简洁

### `log_inference()` 非 Protocol 方法

`OutputSink` Protocol 只定义 6 个核心方法。`log_inference()` 是面向日志 Sink 的扩展方法，通过 `hasattr` 检测调用，避免污染核心接口。

---

## 各部分文档导航

| 层 | 文档 |
|---|---|
| Domain | [Domain 层总览](domain/overview.md) |
| Application | [Application 层总览](application/overview.md) |
| Infrastructure | [Infrastructure 层总览](infrastructure/overview.md) |
| Presentation | [Presentation 层总览](presentation/overview.md) |
