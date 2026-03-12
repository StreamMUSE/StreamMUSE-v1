---
title: StreamMUSE
description: 实时 AI 音乐伴奏生成系统
---

# StreamMUSE

**StreamMUSE** 是一个实时 AI 音乐伴奏生成系统，能够在用户演奏旋律的同时，自动生成与之匹配的伴奏声部。系统基于 Transformer 模型（RoFormer / Stanley 引擎），通过低延迟的 HTTP 推理接口实现毫秒级响应。

---

## 核心特性

- **实时生成**：每半拍触发一次推理，伴奏延迟通常在 50–200 ms 以内
- **多种输入方式**：支持键盘、MIDI 设备、MIDI 文件模拟三种输入模式
- **多种输出方式**：支持控制台打印、实时音频播放、MIDI 文件录制、WebSocket 推送、Session 日志等 7 种输出类型
- **Clean Architecture**：严格的四层架构（Presentation → Application → Domain → Infrastructure），易于扩展
- **完整的 Session 日志**：自动记录事件流、推理请求响应、延迟统计和音乐分析
- **无冷启动**：支持注入 MIDI prompt 预热模型历史

---

## 系统架构概览

```
┌─────────────────────────────────────────┐
│           Presentation (CLI)            │
│  uv run streammuse-cli --input-mode ... │
└─────────────────┬───────────────────────┘
                  │ ApplicationConfig
┌─────────────────▼───────────────────────┐
│         Application Layer               │
│   RealTimeMusicService  (3 threads)     │
│   ┌──────────┐ ┌──────┐ ┌───────────┐  │
│   │  Input   │ │ Tick │ │ Inference │  │
│   │ Worker   │ │ Loop │ │  Worker   │  │
│   └──────────┘ └──────┘ └───────────┘  │
└──────┬──────────────────────────┬───────┘
       │ InputSource / OutputSink │ InferenceEngine
┌──────▼──────────────────────────▼───────┐
│         Infrastructure Layer            │
│  KeyboardInput  →  HttpInferenceClient  │
│  MidiDeviceInput → StanleyInferenceEngine│
│  AudioOutputSink  JsonLoggerOutputSink  │
└──────────────────────────────────────────┘
       │
       ▼  HTTP / local model call
┌──────────────────┐
│  Inference Server│  (fake_inference_server.py / real RoFormer)
└──────────────────┘
```

---

## 系统要求

| 项目 | 要求 |
|---|---|
| Python | ≥ 3.10 |
| 操作系统 | macOS / Linux |
| GPU（可选）| CUDA 12.8+（仅 Stanley 本地推理模式） |
| MIDI 音频输出（可选）| python-rtmidi 虚拟端口 |
| 包管理器 | [uv](https://github.com/astral-sh/uv) |

---

## 快速导航

| 文档 | 内容 |
|---|---|
| [安装](getting-started/installation.md) | 环境搭建与依赖安装 |
| [快速上手](getting-started/quickstart.md) | 5 分钟运行第一个示例 |
| [配置项](getting-started/configuration.md) | 所有 CLI 参数说明 |
| [架构总览](architecture/overview.md) | Clean Architecture 四层设计 |
| [Domain 层](architecture/domain/overview.md) | 核心数据模型与 Protocol |
| [输出类型](user-guide/output-types.md) | 7 种输出类型详细说明 |
| [Session 日志](user-guide/session-logging.md) | 日志文件结构与字段 |
| [扩展输入/输出/引擎](developer-guide/extending-inputs.md) | 开发者扩展指南 |
| [CLI 参考](reference/cli.md) | 完整命令行参数表 |
