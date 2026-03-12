---
title: 输出类型
description: 七种输出 Sink（console、audio、midi_file、websocket、json_log、session、composite）的使用说明
---

# 输出类型

StreamMUSE 支持七种输出类型，通过 `--output-type` 参数指定。

---

## console — 控制台输出（默认）

将事件和统计信息打印到终端，适合调试和快速验证。

```bash
uv run streammuse-cli --input-mode keyboard --output-type console
```

输出示例：
```
[status] state=running message=
[tick] tick=0 bar=1 beat=1
[event] source=user tick=0 type=note_on pitch=60
[stats] hit_rate=None round_trip_ms=42.3 server_process_ms=40.1 ...
```

---

## audio — 实时音频播放

将模型伴奏通过 MIDI 设备实时播放，需要系统中有 MIDI 输出端口。

```bash
# 自动选择第一个 MIDI 输出端口
uv run streammuse-cli --input-mode keyboard --output-type audio

# 指定端口
uv run streammuse-cli --input-mode keyboard --output-type audio \
    --midi-out-port "My Synth"
```

查看可用输出端口：
```python
import mido
print(mido.get_output_names())
```

---

## midi_file — MIDI 文件录制

将演奏（用户旋律 + 模型伴奏）录制为 MIDI 文件。

```bash
uv run streammuse-cli --input-mode keyboard --output-type midi_file \
    --midi-file-output session.mid
```

文件包含两个音轨：`User` 和 `Model`。

---

## websocket — WebSocket 推送

将事件序列化为 JSON 放入队列，供 WebSocket 服务器推送给前端。

```bash
uv run streammuse-cli --input-mode keyboard --output-type websocket
```

---

## json_log — JSON 日志

将事件和推理记录写入 JSON 文件，用于事后分析。

```bash
uv run streammuse-cli --input-mode keyboard --output-type json_log \
    --log-dir logs
```

输出文件：
- `events.jsonl` — 每行一个事件
- `inferences.json` — 所有推理记录
- `performance.json` — 延迟百分位和音乐分析
- `statistics.csv` — 摘要指标

---

## session — 完整会话（MIDI + JSON）

在 `json_log` 基础上增加 MIDI 录制。

```bash
uv run streammuse-cli --input-mode keyboard --output-type session \
    --log-dir logs
```

在 `json_log` 文件基础上额外输出：
- `combined.mid` — 包含 User 和 Model 两个音轨的 MIDI 文件

---

## composite — 组合输出（推荐）

同时激活**控制台输出**和**完整会话日志**，是最实用的模式。

```bash
uv run streammuse-cli --input-mode keyboard --output-type composite \
    --log-dir logs
```

等价于同时启用 `console` + `session`。运行时可看到实时输出，结束后有完整日志文件。

若未指定 `--log-dir`，则组合 `console` + `websocket`。

---

## 各类型特性对比

| 类型 | 实时反馈 | MIDI 录制 | JSON 日志 | 性能分析 |
|---|---|---|---|---|
| `console` | ✓ | – | – | – |
| `audio` | 音符播放 | – | – | – |
| `midi_file` | – | ✓ | – | – |
| `websocket` | JSON 推送 | – | – | – |
| `json_log` | – | – | ✓ | ✓ |
| `session` | – | ✓ | ✓ | ✓ |
| `composite` | ✓ | ✓ | ✓ | ✓ |
