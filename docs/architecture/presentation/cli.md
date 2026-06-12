---
title: presentation/cli — 入口点与生命周期
description: main() 函数、injection、cleanup()、信号处理与会话保存
---

# presentation/cli — 入口点与生命周期

**源文件**：`src/streammuse/presentation/cli/cli.py`

CLI 入口负责参数解析、组件装配、可选 injection、会话日志初始化、信号处理和服务生命周期管理。

---

## `main() -> int`

当前实现流程：

```python
def main() -> int:
    args = parse_args()
    config = args_to_config(args)

    if config.input.injection_file:
        validate_injection_args(config)

    if config.output.type != "midi_file":
        session_manager = SessionManager(args.log_dir)
        session_manager.create_session_directory()
        session_manager.save_config(session_config)

    output_sink = OutputSinkFactory.create(config, session_manager)
    inference_engine = InferenceEngineFactory.create(config)

    if config.input.injection_file:
        injected = _perform_injection(inference_engine, config)
        if injected == 0:
            output_sink.close()
            return 1

    input_source = InputSourceFactory.create(config)
    service = RealTimeMusicService(..., count_in_beats=config.count_in_beats)
    service.start(max_ticks=args.max_ticks)
```

关键顺序：

1. 先创建 output sink 和 inference engine。
2. 若指定 `--injection-file`，先执行 `_perform_injection()`。
3. injection 成功后才创建 input source。`InputSourceFactory` 会让 `MidiFileInput` 从 `injection_length_ticks` 后开始。
4. `RealTimeMusicService` 接收 `count_in_beats`，在正式时间线前执行 count-in。

---

## `_perform_injection(...) -> int`

```python
def _perform_injection(inference_engine: InferenceEngine, config: ApplicationConfig) -> int:
    injection_file = config.input.injection_file
    injection_length = int(config.input.injection_length_ticks)
    acc_file = config.input.injection_acc_file
    if acc_file is None:
        acc_file = injection_file.replace("/mel/", "/acc/")

    mel_notes = MidiFileInput._midi_to_notes(..., max_tick=injection_length)
    acc_notes = MidiFileInput._midi_to_notes(..., max_tick=injection_length)
    inference_engine.clear_history()
    inference_engine.inject_history(
        melody_events=mel_events,
        accompaniment_events=acc_events,
        injection_length_ticks=injection_length,
    )
    return injection_length
```

约束：

- 仅支持 `--input-mode midi_file`。
- `--injection-length` 必须大于 0。
- `--injection-file` 必须存在。
- 如果未传 `--inject-acc-file`，CLI 尝试把路径中的 `/mel/` 替换为 `/acc/` 推导伴奏文件；找不到时退化为 melody-only injection。

---

## Session 初始化

当前 CLI 对除 `midi_file` 外的所有 output type 都创建 `SessionManager`：

```python
if config.output.type != "midi_file":
    session_manager = SessionManager(args.log_dir)
    session_manager.create_session_directory()
    session_config = {
        "tempo_bpm": config.tempo.bpm,
        "ticks_per_beat": config.tempo.ticks_per_beat,
        "beats_per_bar": config.tempo.beats_per_bar,
        "metronome_enabled": config.output.metronome_enabled,
        "metronome_port": config.output.metronome_port,
        "metronome_channel": config.output.metronome_channel,
        "count_in_beats": config.count_in_beats,
        ...
    }
    session_manager.save_config(session_config)
```

因此 `console` / `audio` / `websocket` 也会有 session 目录和自动 `combined.mid`。

---

## `cleanup() -> None`

CLI 通过 `atexit.register(cleanup)` 注册清理逻辑：

1. 调用 `inference_engine.clear_history()`。
2. 如果 server 返回历史，则写入 `melody_history.json` 和 `accompaniment_history.json`。
3. 调用 `output_sink.close()`，触发 MIDI 和 JSON 文件落盘。
4. 若有 `session_manager`，写 `session_summary.txt`。

---

## `signal_handler(sig, frame)`

SIGINT / SIGTERM 行为：

1. 打印 `Shutting down...`。
2. 调用 `service.stop()`。
3. `sys.exit(0)`。

---

## 入口点注册

`pyproject.toml`：

```toml
[project.scripts]
streammuse-cli = "streammuse.presentation.cli.cli:main"
```
