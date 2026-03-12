---
title: infrastructure 层总览
description: Infrastructure 层的职责、组成与扩展方式
---

# infrastructure 层总览

**源文件**：`src/streammuse/infrastructure/`

Infrastructure 层是 Clean Architecture 的最外层，包含所有与外部世界交互的具体实现：MIDI 设备、文件系统、HTTP 服务器、ML 模型等。

---

## 职责

- 实现 Domain 接口（`InputSource`、`OutputSink`、`InferenceEngine`）的具体类
- 管理所有外部 I/O（MIDI 端口、文件读写、HTTP 请求、键盘监听）
- **不包含任何业务逻辑**：业务规则留在 Domain 层

---

## 依赖方向

```
Application/Domain  ←（依赖）  Infrastructure
```

Infrastructure 层可以导入 Domain 层（使用 Domain 的接口和类型），但 Domain 层**绝不**导入 Infrastructure 层。

---

## 组成

```
infrastructure/
├── input/            # InputSource 实现（键盘、MIDI 设备、MIDI 文件、列表）
├── output/           # OutputSink 实现（控制台、音频、文件、WebSocket、日志）
└── inference/        # InferenceEngine 实现（HTTP 客户端、Stanley 本地引擎）
```

---

## 各子模块概述

| 子模块 | 实现 | Domain 接口 |
|---|---|---|
| `input/` | `KeyboardInput`, `MidiDeviceInput`, `MidiFileInput`, `ListInput` | `InputSource` |
| `output/` | 7 种 OutputSink 实现 | `OutputSink` |
| `inference/` | `HttpInferenceClient`, `StanleyInferenceEngine` | `InferenceEngine` |

---

## 详细文档

- [input/overview.md](input/overview.md) — 四种输入适配器
- [output/overview.md](output/overview.md) — 七种输出适配器
- [inference/overview.md](inference/overview.md) — 两种推理引擎
