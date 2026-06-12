---
title: 音乐注入（music injection）
description: 通过 CLI 或 HTTP API 注入历史上下文
---

# 音乐注入

音乐注入用于在会话开始前预填充模型历史，使模型不必从空上下文启动。当前实现同时支持 CLI injection 和 HTTP API injection。

---

## CLI injection

推荐用法：

```bash
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --injection-file prompts/inputs_lekai/mel/1.mid \
    --injection-length 16 \
    --inference-type http \
    --model-name lekai \
    --server-url http://127.0.0.1:8000/generate_accompaniment
```

### 参数

| 参数 | 说明 |
|---|---|
| `--injection-file` | 用作 prompt history 的 melody MIDI 文件 |
| `--injection-length` | 从 injection 文件开头截取多少 ticks 注入 |
| `--inject-acc-file` | 可选 accompaniment MIDI 文件；不传时尝试由 `/mel/` 替换为 `/acc/` 推导 |

### 限制

1. CLI injection 只支持 `--input-mode midi_file`。
2. `--injection-length` 必须大于 0。
3. `--injection-file` 必须存在。
4. 当前 CLI 通过 `InferenceEngine` 接口执行注入；实际 HTTP 模式会调用 server 的 `/clear_history` 和 `/inject_notes`。

---

## CLI 内部流程

入口在 `src/streammuse/presentation/cli/cli.py::_perform_injection()`。

```python
mel_notes, _resolution, _max_tick = MidiFileInput._midi_to_notes(
    midi_path=injection_file,
    beat_div=config.tempo.ticks_per_beat,
    min_pitch=0,
    max_pitch=127,
    program=None,
    max_tick=injection_length,
)
mel_events = _notes_to_musical_events(mel_notes)

inference_engine.clear_history()
inference_engine.inject_history(
    melody_events=mel_events,
    accompaniment_events=acc_events,
    injection_length_ticks=injection_length,
)
```

随后 `InputSourceFactory` 创建 `MidiFileInput` 时会设置：

```python
start_tick=(int(cfg.injection_length_ticks) if cfg.injection_file else 0)
```

因此正式实时输入会从 `injection_length` 后开始，避免同一段旋律既被注入又被实时发送。

---

## HTTP API injection

以下服务实现了注入相关端点：

1. `scripts/fake_inference_server.py`
2. `src/streammuse/infrastructure/inference/server_lekai.py`

端点：

1. `POST /inject_notes`
2. `GET /injection_status`
3. `POST /clear_history`

### 1) 注入历史

```bash
curl -X POST http://127.0.0.1:8000/inject_notes \
    -H "Content-Type: application/json" \
    -d '{
        "melody_notes": [
            {"type": "note_on", "pitch": 60, "tick": 0},
            {"type": "note_off", "pitch": 60, "tick": 4}
        ],
        "accompaniment_notes": [
            {"type": "note_on", "pitch": 48, "tick": 0},
            {"type": "note_off", "pitch": 48, "tick": 4}
        ],
        "injection_length_ticks": 16
    }'
```

### 2) 查询注入状态

```bash
curl -s http://127.0.0.1:8000/injection_status
```

### 3) 清空历史

```bash
curl -X POST http://127.0.0.1:8000/clear_history
```

`clear_history` 通常会先返回 server 当前累积的 melody/accompaniment history，再清空内部状态。

---

## 数据格式

`/inject_notes` 请求体中的每个事件至少需要：

1. `type`：`note_on` 或 `note_off`
2. `pitch`：MIDI 音高
3. `tick`：绝对 tick

`velocity` 可选。

---

## 会话结束时的历史落盘

CLI 退出时会自动调用一次 `clear_history()`：

1. 将 server 返回的 `melody_history` 写入 `melody_history.json`。
2. 将 `accompaniment_history` 写入 `accompaniment_history.json`。
3. 清空 server 端历史，确保下一次会话从干净上下文开始。
