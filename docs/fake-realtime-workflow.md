# Fake Realtime 工作流程详解

## 1. 系统架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FAKE REALTIME SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Client Side                            Server Side                     │
│  ┌─────────────────────┐               ┌─────────────────────────────┐  │
│  │ run_lekai_fake_     │  HTTP POST    │ server_lekai.py             │  │
│  │ _realtime.py        │ ────────────> │  ┌───────────────────────┐  │  │
│  │                     │  /generate    │  │ LekaiHttpBackend      │  │  │
│  │ ┌───────────────┐   │  _accompaniment│  │  ┌─────────────────┐ │  │  │
│  │ │ MidiFileInput │   │               │  │  │ _generate_with_ │ │  │  │
│  │ └───────┬───────┘   │               │  │  │ interleaved_    │ │  │  │
│  │         │           │               │  │  │ prompt()        │ │  │  │
│  │         v           │               │  │  └────────┬────────┘ │  │  │
│  │ ┌───────────────┐   │               │  │           v          │  │  │
│  │ │ HttpInference │   │ <──────────── │  │  ┌─────────────────┐ │  │  │
│  │ │ Client        │   │  JSON Response│  │  │ PianoLLaMA      │ │  │  │
│  │ └───────┬───────┘   │               │  │  │ .generate()     │ │  │  │
│  │         │           │               │  │  └─────────────────┘ │  │  │
│  │         v           │               │  └───────────────────────┘  │  │
│  │ ┌───────────────┐   │               └─────────────────────────────┘  │
│  │ │ MidiFileOutput│   │                                                 │
│  │ │ Sink          │   │                                                 │
│  │ └───────────────┘   │                                                 │
│  └─────────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. 主要依赖文件

### 2.1 Client Side (Fake Realtime Runner)

| 文件 | 职责 |
|------|------|
| `scripts/run_lekai_fake_realtime.py` | 主入口，协调输入处理和推理请求 |
| `src/streammuse/infrastructure/input/midi_file.py` | MIDI 文件读取和解析 |
| `src/streammuse/infrastructure/inference/http_client.py` | HTTP 客户端，与 Server 通信 |
| `src/streammuse/infrastructure/output/midi_file.py` | MIDI 文件输出 |

### 2.2 Server Side (Inference Engine)

| 文件 | 职责 |
|------|------|
| `src/streammuse/infrastructure/inference/server_lekai.py` | FastAPI HTTP Server |
| `src/streammuse/infrastructure/inference/lekai_http_backend.py` | 核心业务逻辑，处理生成请求 |
| `src/streammuse/infrastructure/inference/lekai_model/model.py` | PianoLLaMA 模型定义 |
| `src/streammuse/infrastructure/inference/lekai_model/PianoDataset.py` | Tokenizer 和数据处理 |

## 3. 数据流详解

### 3.1 阶段 1: 输入处理 (Client)

```
MIDI File
    │
    ▼
┌─────────────────────────────────────────┐
│ MidiFileInput._midi_to_notes()          │
│                                         │
│ 1. 读取 MIDI 文件                       │
│ 2. 提取 note_on/note_off 事件          │
│ 3. 转换为 ticks (beat_div=4)           │
│ 4. 返回: List[{"pitch": X, "tick": Y}]  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ notes_to_events()                       │
│                                         │
│ 将 notes 转换为 MusicalEvent 对象:      │
│ - EventType.NOTE_ON                     │
│ - EventType.NOTE_OFF                    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ build_tick_index()                      │
│                                         │
│ 创建 tick -> events 的映射字典:         │
│ {0: [ev1, ev2], 4: [ev3], ...}         │
└─────────────────────────────────────────┘
```

### 3.2 阶段 2: Tick Loop (Client)

```python
# 伪代码展示核心逻辑
for tick in range(start_tick, max_ticks):
    # 1. 收集当前 tick 的 melody 事件
    tick_events = melody_by_tick.get(tick, [])
    pending_new_events.extend(tick_events)
    
    # 2. 检查是否触发生成
    if tick - last_generation_tick >= generation_interval_ticks:
        # 发送 HTTP 请求到 Server
        accompaniment = client.generate_accompaniment(
            melody_events=pending_new_events,
            generation_start_tick=tick,
            generation_length_frames=4,
        )
        
        # 3. 合并结果（替换旧的事件）
        model_events = [ev for ev in model_events if ev.tick < generation_start_tick]
        model_events.extend(accompaniment)
        
        pending_new_events = []
```

### 3.3 阶段 3: HTTP 请求/响应

**Request (Client -> Server)**:
```json
{
  "melody_notes": [
    {"type": "note_on", "pitch": 60, "tick": 0},
    {"type": "note_off", "pitch": 60, "tick": 4},
    ...
  ],
  "generation_start_tick": 16,
  "generation_length_frames": 4,
  "generation_interval_ticks": 4,
  "model_name": "lekai",
  "bpm": 120
}
```

**Response (Server -> Client)**:
```json
{
  "accompaniment": [
    {"type": "note_on", "pitch": 48, "tick": 16, "velocity": 80},
    {"type": "note_off", "pitch": 48, "tick": 20, "velocity": 0},
    ...
  ],
  "timings": {
    "request_arrival_time": 1234567890.0,
    "inference_start_time": 1234567890.1,
    "inference_end_time": 1234567890.5,
    ...
  }
}
```

### 3.4 阶段 4: Server 端处理

```
HTTP Request
    │
    ▼
┌─────────────────────────────────────────┐
│ server_lekai.py                         │
│ /generate_accompaniment endpoint        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ LekaiHttpBackend.generate()             │
│                                         │
│ 1. 更新 melody_history                  │
│ 2. 调用 _generate_with_model()          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ _generate_with_interleaved_prompt()     │
│                                         │
│ 核心生成逻辑:                           │
│ 1. 构建 prompt tokens                   │
│ 2. 调用模型生成                         │
│ 3. 转换 tokens -> events                │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ _generate_part1_tokens_from_prompt()    │
│                                         │
│ 调用 PianoLLaMA 模型逐 token 生成      │
└─────────────────────────────────────────┘
```

### 3.5 阶段 5: Token 构建流程 (Server)

```python
# Prompt 构建过程（简化版）

# 1. 初始 tokens
seq = [BOS, time_sig, bpm, pad]

# 2. 生成 acc_{-1} (如果 start_beat == 0)
if start_beat == 0 and not accompaniment_history:
    acc_neg1_tokens = generate_from_prompt([BOS, ts, bpm, pad])
    seq.append(acc_neg1_tokens)

# 3. Context loop (历史 context)
for beat in range(start_beat, current_beat):
    if beat % 4 == 0:
        seq.append(bar)
        seq.append(bar)
    
    # Encode melody for this beat
    mel_tokens = encode_beat_tokens(melody_history, beat)
    seq.append(mel_tokens)
    
    # Encode accompaniment for this beat (from history)
    acc_tokens = encode_beat_tokens(accompaniment_history, beat)
    seq.append(acc_tokens)

# 4. Generation loop (生成新伴奏)
for beat in range(num_beats_to_generate):
    if beat % 4 == 0:
        seq.append(bar)
        seq.append(bar)
    
    # Encode current melody
    mel_tokens = encode_beat_tokens(melody_history, target_beat)
    
    # Generate accompaniment
    prompt = concat(seq, mel_tokens)
    acc_tokens = generate_from_prompt(prompt)
    
    # Convert tokens to MIDI events
    events = tokens_to_midi(acc_tokens)
    generated_events.extend(events)
```

### 3.6 阶段 6: 输出合并 (Client)

```
Melody Events (User)     Model Events (Generated)
       │                         │
       │    ┌─────────────┐      │
       └───>│ MidiFile    │<─────┘
            │ OutputSink  │
            └──────┬──────┘
                   │
                   ▼
            combined.mid
            (Track 0: Melody)
            (Track 1: Accompaniment)
```

## 4. 关键数据结构

### 4.1 MusicalEvent
```python
@dataclass
class MusicalEvent:
    tick: int          # 时间位置 (ticks)
    pitch: int         # 音高 (MIDI note number)
    event_type: EventType  # NOTE_ON / NOTE_OFF
    velocity: int = 64
    channel: int = 0
    source: str = "user"  # "user" or "model"
```

### 4.2 Token Sequence 结构
```
[BOS, time_sig, bpm, pad, acc_{-1}, bar, bar, mel_0, acc_0, mel_1, acc_1, ...]
  │     │        │    │     │       │    │    │      │      │      │
  │     │        │    │     │       │    │    │      │      │      └── beat 1 acc
  │     │        │    │     │       │    │    │      │      └───────── beat 1 melody
  │     │        │    │     │       │    │    │      └──────────────── beat 0 acc
  │     │        │    │     │       │    │    └─────────────────────── beat 0 melody
  │     │        │    │     │       │    └──────────────────────────── bar tokens (measure start)
  │     │        │    │     │       └───────────────────────────────── bar token
  │     │        │    │     └───────────────────────────────────────── acc for beat -1
  │     │        │    └─────────────────────────────────────────────── padding
  │     │        └──────────────────────────────────────────────────── BPM token
  │     └───────────────────────────────────────────────────────────── Time signature token
  └─────────────────────────────────────────────────────────────────── Begin of sequence
```

## 5. 工作流程时序图

```
Client                              Server
  │                                    │
  ├─ 1. POST /clear_history ────────>│
  │<─ 200 OK ─────────────────────────┤
  │                                    │
  ├─ 2. POST /generate_accompaniment >│
  │    melody_notes=[...]              │
  │    generation_start_tick=0         │
  │    generation_length_frames=4      │
  │                                    │
  │                                    ├─ 3. Update melody_history
  │                                    ├─ 4. Build prompt tokens
  │                                    ├─ 5. Call model.generate()
  │                                    ├─ 6. Convert tokens to events
  │                                    │
  │<─ 7. JSON Response ───────────────┤
  │    accompaniment=[...]             │
  │                                    │
  ├─ 8. Merge events (replace old)    │
  │                                    │
  ├─ 9. POST /generate_accompaniment >│
  │    (next interval)                 │
  │    ...                             │
```

## 6. 与 Offline 模式的区别

| 特性 | Fake Realtime | Offline |
|------|--------------|---------|
| 输入 | MIDI 文件 (melody only) | NPZ 文件 (melody + GT acc) |
| 生成方式 | 增量生成，分段请求 | 一次性完整生成 |
| 推理触发 | 按 generation_interval_ticks | 一次性 |
| Context | 累积 melody_history | 完整 prompt |
| 输出 | 实时返回伴奏片段 | 完整伴奏 |

## 7. 常见问题调试

### 7.1 检查数据流
```bash
# 查看 Client 发送的请求
grep "melody_notes" logs/fakert.log | head -1

# 查看 Server 接收的请求
grep "generate_accompaniment" logs/server.log

# 查看生成的 tokens
grep "generated tokens" logs/server.log
```

### 7.2 验证 Prompt 结构
在 `lekai_http_backend.py` 中添加 debug 输出:
```python
print(f"[DEBUG] Prompt tokens: {prompt_tokens.tolist()}")
print(f"[DEBUG] Prompt length: {len(prompt_tokens)}")
```

### 7.3 检查 MIDI 输出
```python
import mido
mid = mido.MidiFile("output.mid")
for i, track in enumerate(mid.tracks):
    notes = sum(1 for msg in track if msg.type == 'note_on')
    print(f"Track {i}: {notes} notes")
```

## 8. 总结

Fake Realtime 的核心工作流程:

1. **输入**: MIDI 文件 → MusicalEvent 列表
2. **循环**: 按 tick 遍历，累积 melody 事件
3. **触发**: 到达 generation_interval 时发送 HTTP 请求
4. **生成**: Server 构建 prompt，调用模型生成伴奏
5. **合并**: Client 接收伴奏，替换旧的事件
6. **输出**: 合并 melody 和 accompaniment 到 MIDI 文件

数据流的关键转换:
```
MIDI → Events → HTTP Request → Prompt Tokens → Model → Tokens → Events → MIDI
```
