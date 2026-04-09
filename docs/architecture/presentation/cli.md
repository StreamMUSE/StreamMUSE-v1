---
title: presentation/cli — 入口点与生命周期
description: main() 函数、cleanup()、信号处理与会话保存
---

# presentation/cli — 入口点与生命周期

源文件：`src/streammuse/presentation/cli/cli.py`

CLI 入口负责：参数解析、组件装配、会话日志初始化、信号处理和服务生命周期管理。

---

## `main() -> int`

当前实现流程（简化）：

```python
def main() -> int:
    args = parse_args()

    config = env_to_config()
    if config is None:
        config = args_to_config(args)
    else:
        config = args_to_config(args)  # 当前实现仍以 CLI 参数为最终配置

    session_manager = None
    session_config = {}
    if config.output.type in ["json_log", "session", "composite"]:
        session_manager = SessionManager(args.log_dir)
        session_manager.create_session_directory()
        session_config = {...}
        session_manager.save_config(session_config)

    input_source = InputSourceFactory.create(config)
    output_sink = OutputSinkFactory.create(config, session_manager)
    inference_engine = InferenceEngineFactory.create(config)

    tempo = Tempo(...)
    scheduler = PlaybackScheduler()

    service = RealTimeMusicService(...)

    def cleanup() -> None:
        output_sink.close()
        if session_manager and isinstance(output_sink, SessionLoggerOutputSink):
            output_sink.save_metrics(session_config)
            session_manager.save_summary({...})

    def signal_handler(sig, frame):
        print("\nShutting down...")
        service.stop()
        sys.exit(0)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    service.start(max_ticks=args.max_ticks)
    while service.running:
        time.sleep(0.1)

    return 0
```

说明：

1. 当前 CLI 没有 `--injection-file` / `--injection-length` 参数。
2. `main()` 不会主动调用 `inference_engine.inject_history()`。

---

## `cleanup() -> None`

`cleanup()` 是 `main()` 内部函数，并通过 `atexit.register(cleanup)` 注册。

当前行为：

1. 调用 `output_sink.close()`。
2. 若 `session_manager` 存在且 `output_sink` 是 `SessionLoggerOutputSink`：
   - 调用 `output_sink.save_metrics(session_config)`（写 `performance.json`、`statistics.csv`）
   - 调用 `session_manager.save_summary(...)`（写 `session_summary.txt`）

`session_config.json` 在服务启动前由 `session_manager.save_config(...)` 写入。

---

## `signal_handler(sig, frame)`

信号处理函数在 `main()` 内部定义，注册到 `SIGINT`（Ctrl+C）和 `SIGTERM`。

行为：

1. 打印 `Shutting down...`
2. 调用 `service.stop()`
3. 调用 `sys.exit(0)` 退出进程

---

## 入口点注册

`pyproject.toml`：

```toml
[project.scripts]
streammuse-cli = "streammuse.presentation.cli.cli:main"
```

安装后可通过 `uv run streammuse-cli` 或 `streammuse-cli` 直接调用。
