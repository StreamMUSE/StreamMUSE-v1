---
title: presentation 层总览
description: CLI 入口点与配置解析的职责与流程
---

# presentation 层总览

**源文件**：`src/streammuse/presentation/cli/`

Presentation 层是系统的**入口点**，负责将命令行参数转换为 `ApplicationConfig`，并将所有组件组装为可运行的服务。

---

## 职责

- 解析 CLI 参数（`argparse`）
- 从环境变量读取配置（`env_to_config()`）
- 构建 `ApplicationConfig`（`args_to_config()`）
- 创建 `SessionManager`（如有日志需求）
- 通过三个 Factory 创建组件
- 创建 `Tempo`、`PlaybackScheduler`
- 注册清理（`atexit`）和信号处理（`SIGINT`、`SIGTERM`）
- 启动 `RealTimeMusicService`

---

## 启动流程

```
uv run streammuse-cli --input-mode keyboard ...
    │
    ▼ main()
parse_args()
    │
    ├── env_to_config()     # 读取环境变量（当前仅返回 None）
    └── args_to_config()    # CLI 参数 → ApplicationConfig
    │
    ▼
SessionManager（如 output_type 为 json_log/session/composite）
    │
    ▼ 三个 Factory
InputSourceFactory.create()   → InputSource
OutputSinkFactory.create()    → OutputSink
InferenceEngineFactory.create() → InferenceEngine
    │
    ▼
Tempo + PlaybackScheduler
    │
    ▼
RealTimeMusicService.start()
    │
    ▼
while service.running:
    sleep(0.1)
```

---

## 组件

| 文件 | 说明 |
|---|---|
| `cli.py` | `main()` 入口函数，生命周期管理 |
| `config_parser.py` | `parse_args()`、`args_to_config()`、`env_to_config()` |

---

## 详细文档

- [cli.md](cli.md) — `main()` 流程、清理与信号处理
- [config_parser.md](config_parser.md) — CLI 参数说明与配置构建
