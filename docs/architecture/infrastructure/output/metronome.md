---
title: MetronomeOutputSink — MIDI 节拍器输出
description: 根据 tick 输出实时 MIDI drum click
---

# MetronomeOutputSink — MIDI 节拍器输出

**源文件**：`src/streammuse/infrastructure/output/metronome.py`

`MetronomeOutputSink` 是 `--enable-metronome` 开关附加的辅助输出 sink。它不对应独立的 `--output-type`，而是被 `OutputSinkFactory` 组合进当前输出链路。

---

## `MetronomeOutputConfig`

```python
@dataclass(frozen=True)
class MetronomeOutputConfig:
    port_name: Optional[str] = None
    ticks_per_beat: int = 4
    beats_per_bar: int = 4
    channel: int = 9
    beat_note: int = 77
    downbeat_note: int = 76
    velocity: int = 80
    downbeat_velocity: int = 110
```

| 字段 | 默认值 | 说明 |
|---|---|---|
| `port_name` | `None` | MIDI 输出端口；`None` 时由 mido 选择默认输出 |
| `ticks_per_beat` | `4` | 每拍 tick 数 |
| `beats_per_bar` | `4` | 每小节拍数 |
| `channel` | `9` | MIDI channel，默认是 General MIDI percussion channel 10 |
| `beat_note` | `77` | 普通拍 click note |
| `downbeat_note` | `76` | 小节第一拍 click note |
| `velocity` | `80` | 普通拍力度 |
| `downbeat_velocity` | `110` | 小节第一拍力度 |

---

## 输出逻辑

```python
def output_metronome_tick(self, tick: int, bar: int, beat: int) -> None:
    self._send_note_off()
    if int(tick) % int(self._config.ticks_per_beat) != 0:
        return

    ticks_per_bar = ticks_per_beat * beats_per_bar
    is_downbeat = ticks_per_bar > 0 and int(tick) % ticks_per_bar == 0
    note = downbeat_note if is_downbeat else beat_note
    velocity = downbeat_velocity if is_downbeat else velocity
    self._port.send(mido.Message("note_on", note=note, velocity=velocity, channel=channel))
```

每个 tick 先关闭上一颗 click note；只有 beat 边界发送新的 note_on。

---

## 端口失败行为

端口通过 `mido.open_output(port_name)` 懒加载打开。如果打开失败，sink 会打印一条 warning，并禁用实时 click；主服务继续运行。

---

## 与 MIDI 录制的关系

实时 click 由 `MetronomeOutputSink` 输出；MIDI 文件里的 `Metronome` 轨由 `MidiFileOutputSink.output_metronome_tick()` 记录。两者共用 service 发出的同一个 metronome tick 回调。
