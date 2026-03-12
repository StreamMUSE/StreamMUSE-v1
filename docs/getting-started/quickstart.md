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

Fake 服务器会监听 `http://localhost:8000`，每次收到推理请求时返回随机的音符序列，用于开发和测试。看到以下输出即表示服务器已就绪：

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

看到状态行 `[RUNNING]` 即表示系统已开始运行。

---

## 第三步：演奏音符

用以下键盘按键输入音符：

| 按键 | MIDI 音高 | 音名 |
|---|---|---|
| `a` | 60 | C4（中央 C）|
| `s` | 62 | D4 |
| `d` | 64 | E4 |
| `f` | 65 | F4 |
| `g` | 67 | G4 |
| `h` | 69 | A4 |
| `j` | 71 | B4 |

- **按住**某个键 = 音符持续（NOTE_ON）
- **松开**该键 = 音符结束（NOTE_OFF）
- 多个键可以同时按下（和弦）

---

## 第四步：观察控制台输出

演奏几个音符后，控制台会打印类似下面的输出：

```
[TICK]  tick=8  bar=1  beat=0
[EVENT] tick=8  pitch=60  type=note_on  src=user
[EVENT] tick=10 pitch=62  type=note_on  src=user
[STATS] round_trip=145.2ms  server=98.3ms
[EVENT] tick=12 pitch=45  type=note_on  src=model
[EVENT] tick=14 pitch=48  type=note_on  src=model
```

- `src=user`：你演奏的音符
- `src=model`：AI 生成的伴奏音符
- `[STATS]`：当次推理的往返延迟和服务端处理时间

---

## 第五步：退出

按 `Ctrl+C` 优雅地关闭系统。所有资源会被释放，若使用了日志输出模式，文件会在此时写入磁盘。

---

## 下一步

- [配置项](configuration.md)：探索所有可用的 CLI 参数
- [输出类型](../user-guide/output-types.md)：尝试 `--output-type audio` 或 `--output-type composite`
- [音乐注入](../user-guide/music-injection.md)：使用 `--injection-file` 预热模型
