---
title: presentation/cli — 入口点与生命周期
description: main() 函数、cleanup()、信号处理与会话保存
---

# presentation/cli — 入口点与生命周期

**源文件**：`src/streammuse/presentation/cli/cli.py`

系统的 CLI 入口点，由 `pyproject.toml` 中的 `[project.scripts]` 注册为 `streammuse-cli`。

---

## `main() -> None`

完整启动流程：

```python
def main() -> None:
    args = parse_args()
    app_config = env_to_config() or args_to_config(args)
    
    session_manager = None
    if args.output_type in ("json_log", "session", "composite") and args.log_dir:
        session_manager = SessionManager(log_dir=Path(args.log_dir))
        session_manager.create_session()
    
    input_source  = InputSourceFactory.create(app_config)
    output_sink   = OutputSinkFactory.create(app_config, session_manager)
    inference_eng = InferenceEngineFactory.create(app_config)
    
    tempo     = Tempo(bpm=app_config.tempo.bpm, ticks_per_beat=app_config.tempo.ticks_per_beat)
    scheduler = PlaybackScheduler()
    
    service = RealTimeMusicService(
        input_source=input_source,
        output_sink=output_sink,
        inference_engine=inference_eng,
        tempo=tempo,
        scheduler=scheduler,
        generation_interval_ticks=app_config.inference.generation_interval_ticks,
        generation_length_frames=app_config.inference.generation_length_frames,
    )
    
    atexit.register(cleanup, service, output_sink, session_manager, app_config)
    signal.signal(signal.SIGINT, signal_handler(service, output_sink, session_manager, app_config))
    signal.signal(signal.SIGTERM, signal_handler(service, output_sink, session_manager, app_config))
    
    service.start(max_ticks=args.max_ticks)
    signal.pause()   # 主线程阻塞
```

**Music Injection**（`--injection-file`）：若指定了 injection 文件，在 `service.start()` 之前调用 `inference_engine.inject_history()`，将预置旋律和伴奏注入模型历史。

---

## `cleanup(service, output_sink, session_manager, app_config) -> None`

清理函数，通过 `atexit.register()` 注册，在进程退出时执行：

1. 调用 `service.stop()`（若正在运行）
2. 调用 `output_sink.close()`
3. 若 `output_sink` 为 `SessionLoggerOutputSink` 且有 `session_manager`：
   - `output_sink.save_metrics(session_config)` → 写入 `performance.json` + `statistics.csv`
   - `output_sink.json_sink.save_inferences()` → 写入 `inferences.json`
4. 若适用，调用 `session_manager.save_session_config(...)` → 写入 `session_config.json`

---

## `signal_handler(service, output_sink, session_manager, app_config)`

返回一个信号处理函数（闭包），注册到 `SIGINT`（Ctrl+C）和 `SIGTERM`：

```python
def handler(sig, frame):
    service.stop()
    cleanup(service, output_sink, session_manager, app_config)
    sys.exit(0)
```

---

## 入口点注册

`pyproject.toml`:
```toml
[project.scripts]
streammuse-cli = "streammuse.presentation.cli.cli:main"
```

安装后可通过 `uv run streammuse-cli` 或 `streammuse-cli` 直接调用。
