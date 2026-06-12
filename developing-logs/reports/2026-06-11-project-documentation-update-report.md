# 2026-06-11 项目文档更新报告

## 背景

本轮目标是按现有 `docs/` 框架重新过一遍当前项目，把最近新增或变更的实时推理、CLI、metronome、count-in、music injection、session logging 等实现同步到文档里，并整理当前项目更新点。

本报告基于当前代码状态整理，重点检查了：

- `src/streammuse/presentation/cli/config_parser.py`
- `src/streammuse/presentation/cli/cli.py`
- `src/streammuse/application/config/models.py`
- `src/streammuse/application/services/real_time_music_service.py`
- `src/streammuse/application/factories/input_factory.py`
- `src/streammuse/application/factories/output_factory.py`
- `src/streammuse/infrastructure/output/midi_file.py`
- `src/streammuse/infrastructure/output/metronome.py`
- `src/streammuse/infrastructure/output/session_logger.py`

---

## 当前项目主要更新点

### 1. 实时推理触发逻辑已从 interval-driven 改为 beat-tail-driven

旧文档里多处写的是“每 `generation_interval_ticks` ticks 触发一次推理”。当前 `RealTimeMusicService._tick_loop()` 实际逻辑已经不是这样。

当前触发点：

1. `tick=0`：如果 `_melody_history` 非空，则发送完整历史请求。这覆盖 injection 或已预填充 history 的场景。
2. 每拍末尾：默认 `ticks_per_beat=4` 时，在 tick=3、7、11... 发送下一拍请求，`generation_start_tick=tick+1`。
3. 每拍末尾即使没有新 melody event，也会发送空增量请求，让 stateful server 可以继续生成。

核心代码：

```python
if tick == 0:
    with self._melody_history_lock:
        notes_for_request = self._melody_history.copy()
    if notes_for_request:
        self._inference_request_queue.put((0, notes_for_request))

if tick > 0 and (tick % ticks_per_beat) == (ticks_per_beat - 1):
    self._inference_request_queue.put((tick + 1, notes_for_next_request))
    notes_for_next_request = []
```

`generation_interval_ticks` 当前仍保留在配置、HTTP payload 和 inference log 中，但不再决定客户端 tick loop 的触发时刻。

### 2. 新增 count-in 功能

`ApplicationConfig` 新增 `count_in_beats`，CLI 暴露 `--count-in-beats`。

当前行为：

1. `start()` 计算正式时间线起点：`timeline_start_time = session_start_time + tempo.tick_to_seconds(count_in_ticks)`。
2. `_tick_loop()` 先执行 `_run_count_in()`。
3. `_input_worker()` 睡到 `timeline_start_time` 才开始读取正式输入。
4. count-in 阶段只输出 metronome，不发正式 tick，不发推理请求。

核心代码：

```python
self._count_in_beats = int(count_in_beats)
self._count_in_ticks = self._count_in_beats * int(self._tempo.ticks_per_beat)

timeline_start_time = session_start_time + self._tempo.tick_to_seconds(self._count_in_ticks)
```

### 3. 新增 metronome 输出和 MIDI 录制

CLI 新增：

- `--enable-metronome`
- `--metronome-port`
- `--metronome-channel`

`MetronomeOutputSink` 默认配置：

```python
channel = 9
beat_note = 77
downbeat_note = 76
velocity = 80
downbeat_velocity = 110
```

实时 click 通过 MIDI drum note 输出。若端口不可用，会打印 warning 并继续运行，不中断主服务。

同时，`MidiFileOutputSink` 新增 metronome 录制能力。开启 `--enable-metronome` 后，写 MIDI 的模式会额外包含 `Metronome` 鼓轨。

### 4. count-in 已支持录进 MIDI 文件

`MidiFileOutputSink` 通过观察负 tick 计算录制偏移：

```python
def _observe_recording_tick(self, tick: int) -> None:
    if int(tick) < 0:
        self._recording_tick_offset = max(self._recording_tick_offset, -int(tick))

def _time(self, tick: int) -> float:
    return float(int(tick) + int(self._recording_tick_offset)) * self._sp_tick
```

因此 count-in 阶段的负 tick click 会出现在 MIDI 文件开头，正式 tick=0 的 Melody/Accompaniment 会在录制文件里整体后移。这个偏移只影响 MIDI 文件，不改变推理 tick。

### 5. CLI injection 已实现，不再只是 HTTP API 能力

旧文档里写“CLI 当前没有 `--injection-file` / `--injection-length` 参数”，这已经过时。

当前 CLI 支持：

- `--injection-file`
- `--injection-length`
- `--inject-acc-file`

限制：

1. 只支持 `--input-mode midi_file`。
2. `--injection-length` 必须大于 0。
3. `--injection-file` 必须存在。
4. 未指定 `--inject-acc-file` 时，尝试把路径中的 `/mel/` 替换成 `/acc/` 推导 accompaniment 文件。

关键流程：

```python
inference_engine.clear_history()
inference_engine.inject_history(
    melody_events=mel_events,
    accompaniment_events=acc_events,
    injection_length_ticks=injection_length,
)
```

随后 `InputSourceFactory` 会让 `MidiFileInput` 从 `injection_length_ticks` 开始：

```python
start_tick=(int(cfg.injection_length_ticks) if cfg.injection_file else 0)
```

### 6. 自动 session / combined.mid 逻辑已扩大

当前 CLI 对除 `midi_file` 外的 output type 都创建 `SessionManager`。

`OutputSinkFactory` 对 `console` / `audio` / `websocket` 自动附加 `MidiFileOutputSink`，写入 session 目录下的 `combined.mid`。

`json_log` 仍只写 JSON，不自动写 `combined.mid`。

`session` / `composite` 通过 `SessionLoggerOutputSink` 原生写 MIDI + JSON。

---

## 本轮已更新的文档

### 顶层和架构文档

- `docs/index.md`
- `docs/architecture/overview.md`
- `docs/architecture/application/service.md`
- `docs/architecture/application/config.md`
- `docs/architecture/application/factories.md`
- `docs/architecture/presentation/config_parser.md`
- `docs/architecture/presentation/cli.md`

主要更新：

- 修正实时推理触发语义。
- 补充 count-in workflow。
- 补充 metronome 扩展方法。
- 补全 `ApplicationConfig` 字段。
- 补全 CLI 参数映射。
- 补充 factory 中自动 MIDI 录制和 metronome 附加逻辑。

### 用户文档

- `docs/getting-started/configuration.md`
- `docs/reference/cli-reference.md`
- `docs/user-guide/running-realtime.md`
- `docs/user-guide/output-types.md`
- `docs/user-guide/session-logging.md`
- `docs/user-guide/music-injection.md`
- `docs/user-guide/input-modes.md`
- `docs/reference/glossary.md`

主要更新：

- 加入 `--count-in-beats`、`--enable-metronome`、`--metronome-port`、`--metronome-channel`。
- 加入 CLI injection 参数说明。
- 修正 `combined.mid` 轨道说明：默认 `Melody` / `Accompaniment`，可选 `Metronome`。
- 明确 count-in 会被 MIDI 录制记录，但不影响推理 tick。
- 明确 `generation_interval_ticks` 当前不是客户端 tick loop 的触发条件。

### Infrastructure/output 文档

- `docs/architecture/infrastructure/output/overview.md`
- `docs/architecture/infrastructure/output/midi_file.md`
- `docs/architecture/infrastructure/output/session_logger.md`
- `docs/architecture/infrastructure/output/metronome.md`

主要更新：

- 新增 `MetronomeOutputSink` 文档。
- 补充 `MidiFileOutputSink` 的 metronome 和 count-in offset 机制。
- 补充 `SessionLoggerOutputSink` 的新初始化参数和 metronome 委托行为。

### README

- `README.md`

主要更新：

- 修正过时命令：`--input-mode midi`、`--midi-file` 等旧参数名已移除。
- 加入 metronome、count-in、CLI injection、Lekai HTTP server 示例。
- 简化旧训练说明，把入口导向 `docs/`。

---

## 当前需要注意的语义变化

### `generation_interval_ticks`

这个参数现在容易误解。当前它仍然存在，但职责主要是：

1. HTTP payload 透传。
2. inference log 记录。
3. 部分 server/backend 可能仍读取它。

它不再控制 `RealTimeMusicService._tick_loop()` 的请求频率。

### `json_log` 与 `combined.mid`

`json_log` 仍不写 `combined.mid`。如果需要 JSON + MIDI，应该使用 `session` 或 `composite`。

### count-in 的记录位置

count-in 不进入正式推理时间线。MIDI 文件中看到的 count-in 是录制层通过负 tick offset 实现的，不代表模型收到负 tick 输入。

---

## 建议后续检查项

1. 用 `uv run streammuse-cli --help` 确认 CLI help 与 docs 参数表一致。
2. 用一个短 MIDI 文件跑：`--enable-metronome --count-in-beats 4 --output-type midi_file`，检查输出 MIDI 是否包含 `Metronome` 轨并包含 count-in click。
3. 用 `--injection-file --injection-length` 跑 HTTP 模式，检查 server history 和正式输入起点是否符合预期。
4. 如果未来要继续保留 `generation_interval_ticks`，建议在代码注释里明确它不驱动客户端 tick loop，避免后续维护者误改。
