---
title: StreamMUSE
description: 实时 AI 音乐伴奏生成系统
---

# StreamMUSE

**StreamMUSE** 是一个实时 AI 音乐伴奏生成系统，能够在用户演奏旋律的同时生成匹配的伴奏声部。系统支持 HTTP 推理（fake server / Lekai server）和本地 Stanley 推理两条路径。

---

## 核心特性

- **实时生成**：客户端按 tick 推进音乐时间线，在 tick=0 和每拍末尾触发下一拍推理请求
- **多种输入方式**：支持键盘、MIDI 设备、MIDI 文件模拟和测试用 list 输入
- **多种输出方式**：支持控制台打印、实时 MIDI 播放、MIDI 文件录制、WebSocket 推送、JSON 日志、Session 日志和组合输出
- **节拍器与 count-in**：支持实时 MIDI click，并可在正式输入和推理前空转若干拍
- **分层架构**：清晰的四层结构（Presentation → Application → Domain → Infrastructure），易于扩展
- **完整的 Session 日志**：自动记录事件流、推理请求响应、延迟统计、音乐分析和 `combined.mid`
- **注入能力**：CLI 支持 `--injection-file` / `--injection-length` 预填充 HTTP server 历史

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
│  Audio/MIDI/JSON/Session/Metronome sinks│
└──────────────────────────────────────────┘
       │
       ▼  HTTP / local model call
┌──────────────────┐
│  Inference Server│  (fake_inference_server.py / server_lekai.py)
└──────────────────┘
```

---

## 系统要求

| 项目 | 要求 |
|---|---|
| Python | ≥ 3.10 |
| 操作系统 | macOS / Linux |
| GPU（可选）| CUDA / MPS / CPU 取决于模型后端 |
| MIDI 音频输出（可选）| python-rtmidi / 系统 MIDI 输出端口 |
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
| [输出类型](user-guide/output-types.md) | 7 种用户可选输出类型与 metronome 说明 |
| [Session 日志](user-guide/session-logging.md) | 日志文件结构与字段 |
| [音乐注入](user-guide/music-injection.md) | CLI/API 注入历史上下文 |
| [交互式游戏语音输入](user-guide/voice-input.md) | faster-whisper 麦克风输入、离线缓存、隐私与故障诊断 |
| [语音输入资格验证](developer-guide/voice-input-qualification.md) | 冻结语料库、准确率/延迟统计与验收门槛 |
| [新增输入源](developer-guide/adding-input-source.md) | 如何扩展 InputSource |
| [新增输出 Sink](developer-guide/adding-output-sink.md) | 如何扩展 OutputSink |
| [新增推理引擎](developer-guide/adding-inference-engine.md) | 如何扩展 InferenceEngine |
| [旋律扰动鲁棒性实验](developer-guide/melody-perturbation-robustness.md) | staging、qualification、formal、analysis 与盲听的可复现工作流 |
| [CLI 参考](reference/cli-reference.md) | 完整命令行参数表 |
