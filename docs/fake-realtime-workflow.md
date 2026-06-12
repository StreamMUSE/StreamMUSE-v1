---
title: Fake Realtime 工作流程
description: run_lekai_fake_realtime.py 的脚本流程、HTTP 请求和与实时 service 的差异
---

# Fake Realtime 工作流程

本文描述的是 `scripts/run_lekai_fake_realtime.py` 这个**离线驱动的 fake realtime 脚本**，不是 `streammuse-cli` 使用的真实 `RealTimeMusicService` 主循环。

二者都通过 HTTP 调用 Lekai server，但调度语义不同：

| 项目 | `run_lekai_fake_realtime.py` | `RealTimeMusicService` |
|---|---|---|
| 时间推进 | for-loop 扫 MIDI tick，无 wall-clock sleep | 真实 wall-clock tick loop |
| 请求触发 | `generation_interval_ticks` 到期且 `pending_new_events` 非空 | tick=0 history + 每拍末尾，允许空增量请求 |
| beat-tail 处理 | 如果触发点刚好是 beat tail，则把 `generation_start_tick` 改为下一拍 | 每拍末尾固定请求下一拍 |
| 输入来源 | 单个 melody MIDI 文件 | keyboard / MIDI device / MIDI file / list |
| 输出 | 合并 melody + model events 到 MIDI 文件 | 输出 sink 链路，可 console/audio/MIDI/session/metronome |

---

## 系统架构

```text
MIDI melody file
    │
    ▼
run_lekai_fake_realtime.py
    ├── MidiFileInput._midi_to_notes()
    ├── notes_to_events()
    ├── build_tick_index()
    ├── HttpInferenceClient.generate_accompaniment()
    └── MidiFileOutputSink → *_fake_realtime_combined.mid

HTTP server
    └── server_lekai.py / LekaiHttpBackend
```

---

## 主要文件

| 文件 | 职责 |
|---|---|
| `scripts/run_lekai_fake_realtime.py` | 主入口，按 tick 扫描 MIDI 并增量调用 HTTP 推理 |
| `src/streammuse/infrastructure/input/midi_file.py` | 读取 MIDI 并转换 notes |
| `src/streammuse/infrastructure/inference/http_client.py` | HTTP client，调用 `/generate_accompaniment` 等端点 |
| `src/streammuse/infrastructure/output/midi_file.py` | 输出合并 MIDI |
| `src/streammuse/infrastructure/inference/server_lekai.py` | Lekai FastAPI server |
| `src/streammuse/infrastructure/inference/lekai_http_backend.py` | Lekai HTTP 后端逻辑 |

---

## 输入处理

脚本先把 MIDI 文件转为 note dict，再转成 `MusicalEvent`：

```python
notes, _res, max_tick = MidiFileInput._midi_to_notes(
    str(midi_file),
    beat_div=int(args.ticks_per_beat),
    min_pitch=0,
    max_pitch=127,
    program=None,
    max_tick=None,
)
melody_events = notes_to_events(notes)
melody_by_tick = build_tick_index(melody_events)
```

`build_tick_index()` 生成 `tick -> events` 映射：

```python
def build_tick_index(events: list[MusicalEvent]) -> dict[int, list[MusicalEvent]]:
    by_tick: dict[int, list[MusicalEvent]] = defaultdict(list)
    for event in events:
        by_tick[int(event.tick)].append(event)
    return by_tick
```

---

## Tick Loop 和请求触发

当前脚本核心逻辑：

```python
last_generation_tick = -int(args.generation_interval_ticks)

for tick in range(start_tick, max_ticks):
    tick_events = melody_by_tick.get(tick, [])
    if tick_events:
        pending_new_events.extend(tick_events)

    if tick - last_generation_tick >= int(args.generation_interval_ticks) and pending_new_events:
        generation_start_tick = tick
        if (tick % int(args.ticks_per_beat)) == (int(args.ticks_per_beat) - 1):
            generation_start_tick = tick + 1

        accompaniment, _timing = client.generate_accompaniment(
            melody_events=list(pending_new_events),
            generation_start_tick=int(generation_start_tick),
            generation_length_frames=int(args.generation_length_frames),
        )

        model_events = [ev for ev in model_events if ev.tick < generation_start_tick]
        model_events.extend(accompaniment)
        pending_new_events = []
        last_generation_tick = tick
```

关键点：

1. `generation_interval_ticks` 在这个脚本里仍是触发间隔。
2. 请求只有在 `pending_new_events` 非空时才会发送。
3. 如果触发 tick 是 beat tail（默认 tick=3、7、11...），脚本会把 `generation_start_tick` 调整到下一拍。
4. 新响应会替换 `generation_start_tick` 之后的旧 model events。

---

## HTTP 请求结构

`HttpInferenceClient` 会发送类似 payload：

```json
{
  "melody_notes": [
    {"type": "note_on", "pitch": 60, "tick": 0},
    {"type": "note_off", "pitch": 60, "tick": 4}
  ],
  "generation_start_tick": 16,
  "client_request_send_time": 1700000000.123,
  "generation_length_frames": 4,
  "generation_interval_ticks": 4,
  "model_name": "lekai",
  "inference_mode": "sliding_window"
}
```

server 返回：

```json
{
  "accompaniment": [
    {"type": "note_on", "pitch": 48, "tick": 16, "velocity": 80},
    {"type": "note_off", "pitch": 48, "tick": 20, "velocity": 0}
  ],
  "timings": {
    "request_arrival_time": 1700000000.124,
    "inference_start_time": 1700000000.130,
    "inference_end_time": 1700000000.180,
    "response_output_time": 1700000000.181
  }
}
```

---

## 输出文件

脚本输出两个主要文件：

```text
<output-dir>/
├── <stem>_fake_realtime_combined.mid
└── <stem>_fake_realtime_summary.json
```

MIDI 输出使用 `MidiFileOutputSink`，默认包含：

```text
Track: Melody
Track: Accompaniment
```

summary JSON 包含输入文件、server URL、BPM、tick 参数、请求数量、返回事件数量和输出 MIDI 路径。

---

## 与 Offline 模式的区别

| 特性 | Fake Realtime 脚本 | Offline |
|---|---|---|
| 输入 | MIDI melody file | NPZ 或离线脚本指定格式 |
| 生成方式 | 增量 HTTP 请求 | 通常一次性或按离线脚本策略生成 |
| 上下文 | server 累积 history | 离线构造完整 prompt/context |
| 输出 | `*_fake_realtime_combined.mid` | 离线脚本输出 MIDI / token log |

---

## 与真实 realtime service 的差异

`RealTimeMusicService` 当前更接近真实演奏环境：

1. 有 wall-clock tick loop。
2. 支持 count-in。
3. 支持 metronome 输出和录制。
4. 每拍末尾固定发下一拍请求，即使没有新 melody event。
5. 推理线程采用 latest-only drain，慢请求不会无限堆积。

如果要研究真实实时质量问题，应优先看：

- `docs/architecture/application/service.md`
- `docs/user-guide/running-realtime.md`
- `src/streammuse/application/services/real_time_music_service.py`

Fake realtime 脚本更适合做可复现的 HTTP/token 对比实验。
