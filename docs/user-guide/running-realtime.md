---
title: 运行实时系统
description: 启动推理服务器和 CLI 客户端的完整流程
---

# 运行实时系统

StreamMUSE 实时系统通常由两个进程组成：**推理服务器**（提供模型预测）和 **CLI 客户端**（读取输入、播放输出）。例外是 `--inference-type stanley`，它支持本地单进程运行。

---

## 快速启动（开发模式）

不需要真实模型，使用 fake server 即可体验完整系统：

```bash
# 终端 1：启动 fake 推理服务器
uv run python scripts/fake_inference_server.py

# 终端 2：启动 CLI 客户端（键盘输入）
uv run streammuse-cli --input-mode keyboard
```

按 `Ctrl+C` 停止客户端。

---

## 使用真实模型

当前仓库中，真实模型主要有两条可用路径：

1. **Stanley 本地单进程**：`--inference-type stanley`，不走 HTTP server。
2. **Lekai HTTP server**：启动 `server_lekai.py`，CLI 走 HTTP。

### 方案 A：Stanley 本地单进程

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type stanley \
    --checkpoint-path path/to/model.ckpt
```

### 方案 B：Lekai HTTP server

推理 backend 与 MIDI 设备相互独立。下面的 backend-only launcher 只加载模型并提供 HTTP API，不会枚举或打开 MIDI 输入/输出设备：

```bash
# 终端 1：只启动 Lekai 推理 backend
LEKAI_CHECKPOINT_PATH=path/to/lekai_checkpoint.safetensors \
LEKAI_DEVICE=auto \
LEKAI_DTYPE=auto \
bash scripts/start_lekai_backend.sh
```

Web 是独立进程；启动后保持 idle，只有点击 **Start** 时才创建 session 并打开配置的 MIDI 设备：

```bash
# 终端 2：启动 idle Web
uv run streammuse-web \
    --web-host 0.0.0.0 \
    --web-port 8001 \
    --input-mode midi_device \
    --inference-type http \
    --model-name lekai \
    --server-url http://localhost:8000/generate_accompaniment
```

CLI 客户端也可以单独连接同一个 backend：

```bash
# 终端 2（CLI 方案）
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type http \
    --model-name lekai \
    --server-url http://localhost:8000/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 16
```

在 Web 或 CLI 启动命令中加入 `--mute-melody-output`，可关闭系统对用户 Melody 的实时 MIDI 监听播放。Melody 仍会进入模型，并保留在 Web、session 日志和 `combined.mid` 中。

缺少或断开 MIDI 设备不会阻止 backend 或 idle Web 启动；设备错误只会在用户点击 Start、session 实际打开输入时报告。

---

## MIDI 文件批量测试示例

```bash
for f in 1 2 3 4 5; do
  uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/$f.mid \
    --inference-type http \
    --model-name lekai \
    --server-url http://127.0.0.1:8001/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 4 \
    --max-ticks 256 \
    --output-type console \
    --enable-metronome \
    --count-in-beats 4
 done
```

参数名是 `--count-in-beats`，不是 `--ount-in-beats`。

---

## count-in 与 metronome

`--count-in-beats N` 会在正式音乐时间线开始前空转 N 拍。count-in 阶段：

1. 只输出 metronome click。
2. `_input_worker` 尚未开始读取正式输入。
3. 不发送 `generate_accompaniment` 请求。
4. 结束后正式 tick 从 0 开始。

`--enable-metronome` 会启用实时 MIDI click。默认音色通过 General MIDI percussion channel 输出：

| 类型 | MIDI note | velocity | 说明 |
|---|---|---|---|
| downbeat | 76 | 110 | 小节第一拍，更重 |
| beat | 77 | 80 | 普通拍 |

默认 channel 为 9，即 MIDI channel 10 的鼓通道。

---

## 默认 MIDI 产物与日志目录结构

当前版本中，除了 `--output-type midi_file` 外，CLI 会创建 session 目录：

```text
logs/YYYY-MM-DD/session_HHMMSS/
```

默认 MIDI 输出行为：

1. `console` / `audio` / `websocket`：自动写 `combined.mid`。
2. `session` / `composite`：保持原有 `combined.mid` 行为，并继续输出 JSON 日志。
3. `json_log`：不写 `combined.mid`，只写 JSON。
4. `midi_file`：只写用户指定的 `--midi-file-output-path`，不创建 session 目录。

如果开启 `--enable-metronome`，写 MIDI 的模式会额外包含 `Metronome` 鼓轨。若同时使用 `--count-in-beats`，count-in click 也会被录进 MIDI 文件开头。

---

## 音乐注入

CLI 支持直接 injection：

```bash
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --injection-file prompts/inputs_lekai/mel/1.mid \
    --injection-length 16 \
    --inference-type http \
    --model-name lekai
```

约束：

1. injection 只支持 `--input-mode midi_file`。
2. `--injection-length` 必须大于 0。
3. CLI 会先调用 `clear_history()`，再调用 `inject_history()`。
4. 若未提供 `--inject-acc-file`，CLI 尝试把路径中的 `/mel/` 替换为 `/acc/` 推导 accompaniment 文件。
5. 正式 `MidiFileInput` 从 `injection_length` 后开始播放，避免重复输入已注入片段。

HTTP API 仍可直接使用，详见：[music-injection](music-injection.md)。

---

## Lekai 服务器环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LEKAI_CHECKPOINT_PATH` | `None` | 模型 checkpoint 路径 |
| `LEKAI_DEVICE` | `auto` | 设备选择：`auto` / `mps` / `cpu` / `cuda` |
| `LEKAI_DTYPE` | `auto` | 精度选择：`auto` / `float32` / `float16` |
| `LEKAI_ENABLE_MPS_FALLBACK` | `true` | `mps` 加载失败时是否自动回退 CPU |
| `LEKAI_USE_CACHE` | `true` | 生成时是否使用 KV cache |
| `LEKAI_WARMUP_STEPS` | `1` | 启动后 warmup 的最小生成步数 |
| `LEKAI_MAX_GENERATION_LENGTH_FRAMES` | `None` | 可选上限，限制每次生成长度 |
| `LEKAI_MAX_PROMPT_TICKS` | `None` | 可选上限，限制 prompt 长度 |
| `LEKAI_SERVER_HOST` | `0.0.0.0` | 服务器监听地址 |
| `LEKAI_SERVER_PORT` | `8000` | 服务器监听端口 |

这些环境变量由 server 代码读取。CLI 侧 `env_to_config()` 当前不读取环境变量。

---

## 停止服务

客户端：按 `Ctrl+C`，服务将自动停止线程、清理推理历史并保存日志文件。

推理服务器：按 `Ctrl+C`。

---

## 常见问题

**Q：找不到 MIDI 设备**

```python
import mido
print(mido.get_input_names())
print(mido.get_output_names())
```

**Q：连接推理服务器失败**

检查服务器地址是否正确，可通过 `--server-url` 参数修改。

**Q：latency 过高**

优先降低 `--generation-length-frames`。当前客户端触发由拍点驱动，`--generation-interval-ticks` 不再直接改变 tick loop 的触发频率。
