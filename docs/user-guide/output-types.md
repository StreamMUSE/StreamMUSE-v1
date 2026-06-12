---
title: 输出类型
description: 七种输出 Sink（console、audio、midi_file、websocket、json_log、session、composite）的使用说明
---

# 输出类型

StreamMUSE 支持七种用户可选输出类型，通过 `--output-type` 参数指定。`--enable-metronome` 是一个额外开关，不是独立 output type。

---

## console — 控制台输出（默认）

将事件和统计信息打印到终端，适合调试和快速验证。当前版本会自动附加 MIDI 录制，输出到 session 目录中的 `combined.mid`。

```bash
uv run streammuse-cli --input-mode keyboard --output-type console
```

---

## audio — 实时 MIDI 播放

将模型伴奏通过 MIDI 设备实时播放，需要系统中有 MIDI 输出端口。当前版本会自动附加 MIDI 录制。

```bash
uv run streammuse-cli --input-mode keyboard --output-type audio --midi-out-port "My Synth"
```

查看可用输出端口：

```python
import mido
print(mido.get_output_names())
```

---

## midi_file — MIDI 文件录制

将演奏录制为单个 MIDI 文件。

```bash
uv run streammuse-cli \
  --input-mode keyboard \
  --output-type midi_file \
  --midi-file-output-path session.mid
```

默认包含 `Melody` 和 `Accompaniment` 两个音轨。开启 `--enable-metronome` 后会额外包含 `Metronome` 鼓轨。

---

## websocket — WebSocket 推送

将事件序列化为 JSON 放入队列，供 WebSocket 服务器推送给前端。当前版本会自动附加 MIDI 录制。

```bash
uv run streammuse-cli --input-mode keyboard --output-type websocket
```

---

## json_log — JSON 日志

将事件和推理记录写入 JSON 文件，用于事后分析。

```bash
uv run streammuse-cli --input-mode keyboard --output-type json_log --log-dir logs
```

输出文件：

- `events.jsonl`：每行一个事件
- `inferences.json`：推理记录
- `session_config.json`：会话配置快照

`json_log` 不自动写 `combined.mid`。

---

## session — 完整会话（MIDI + JSON）

在 JSON 日志基础上增加 MIDI 录制。

```bash
uv run streammuse-cli --input-mode keyboard --output-type session --log-dir logs
```

常见输出：

- `combined.mid`：MIDI 录制
- `events.jsonl`：事件日志
- `inferences.json`：推理日志
- `performance.json`：延迟与事件统计
- `statistics.csv`：摘要指标 CSV
- `session_summary.txt`：会话完成摘要

---

## composite — 组合输出（推荐调试模式）

同时激活控制台输出和完整会话日志。

```bash
uv run streammuse-cli --input-mode keyboard --output-type composite --log-dir logs
```

CLI 中通常等价于 `ConsoleOutputSink + SessionLoggerOutputSink`。

---

## metronome 输出

开启：

```bash
uv run streammuse-cli \
  --input-mode midi_file \
  --midi-file-path prompts/inputs_lekai/mel/1.mid \
  --output-type console \
  --enable-metronome \
  --count-in-beats 4
```

行为：

1. 实时 MIDI click 由 `MetronomeOutputSink` 输出。
2. 写 MIDI 的模式会额外记录 `Metronome` 鼓轨。
3. count-in 阶段的 click 使用负 tick，MIDI 录制会把它平移到文件开头。

默认 click 参数：

| 类型 | note | velocity | channel |
|---|---|---|---|
| downbeat | 76 | 110 | 9 |
| beat | 77 | 80 | 9 |

---

## 各类型特性对比

| 类型 | 实时反馈 | MIDI 录制 | JSON 日志 | 性能分析 |
|---|---|---|---|---|
| `console` | ✓ | ✓（自动 `combined.mid`） | – | – |
| `audio` | MIDI 播放 | ✓（自动 `combined.mid`） | – | – |
| `midi_file` | – | ✓（指定路径） | – | – |
| `websocket` | JSON 推送 | ✓（自动 `combined.mid`） | – | – |
| `json_log` | – | – | ✓ | – |
| `session` | – | ✓ | ✓ | ✓ |
| `composite` | ✓ | ✓ | ✓ | ✓ |
