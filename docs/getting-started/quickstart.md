---
title: 快速上手
description: 5 分钟内运行 StreamMUSE 第一个示例
---

# 快速上手

本文将在 5 分钟内带你启动系统、输入旋律并观察到 AI 生成的伴奏。

> **前提**：已完成[安装指南](installation.md)中的所有步骤。

---

## 第一步：启动 Fake 推理服务器

在第一个终端中运行 fake 服务器（不需要真实模型或 GPU）：

```bash
uv run python scripts/fake_inference_server.py
```

Fake 服务器会监听 `http://localhost:8000`，并将收到的旋律事件回显为伴奏（echo），用于开发和测试。看到以下输出即表示服务器已就绪：

```
INFO:     Started server process [xxxxx]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 第二步：启动 CLI 客户端

在**另一个终端**中，以键盘输入模式、控制台输出启动客户端：

```bash
uv run streammuse-cli --input-mode keyboard
```

看到状态行 `[status] state=running message=` 即表示系统已开始运行。

---

## 第三步：演奏音符

用以下键盘按键输入音符：

| 按键 | MIDI 音高 | 音名 |
|---|---|---|
| `z` | 60 | C4（中央 C）|
| `x` | 62 | D4 |
| `c` | 64 | E4 |
| `v` | 65 | F4 |
| `b` | 67 | G4 |
| `n` | 69 | A4 |
| `m` | 71 | B4 |

- **按住**某个键 = 音符持续（NOTE_ON）
- **松开**该键 = 音符结束（NOTE_OFF）
- 多个键可以同时按下（和弦）

---

## 第四步：观察控制台输出

演奏几个音符后，控制台会打印类似下面的输出：

```
[tick] tick=8 bar=0 beat=2
[event] source=user tick=8 type=note_on pitch=60
[event] source=user tick=10 type=note_on pitch=62
[stats] hit_rate=None avg_backup_level=None round_trip_ms=145.2 server_process_ms=98.3 network_latency_ms=None total_hits=None total_ticks=None
[event] source=model tick=12 type=note_on pitch=45
[event] source=model tick=14 type=note_on pitch=48
```

- `source=user`：你演奏的音符
- `source=model`：AI 生成的伴奏音符
- `[stats]`：当次推理的往返延迟和服务端处理时间

---

## 第五步：退出

按 `Ctrl+C` 优雅地关闭系统。所有资源会被释放，若使用了日志输出模式，文件会在此时写入磁盘。

---

## 下一步

- [配置项](configuration.md)：探索所有可用的 CLI 参数
- [输出类型](../user-guide/output-types.md)：尝试 `--output-type audio` 或 `--output-type composite`
- [音乐注入](../user-guide/music-injection.md)：通过 HTTP API 注入历史（高级用法）
