---
title: 运行实时系统
description: 启动推理服务器和 CLI 客户端的完整流程
---

# 运行实时系统

StreamMUSE 实时系统通常由两个进程组成：**推理服务器**（提供模型预测）和 **CLI 客户端**（读取输入、播放输出）。

例外：`--inference-type stanley` 支持本地单进程模式（无 HTTP server）。

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

当前仓库中，真实模型有两条可用路径：

1. **Stanley 本地单进程**：`--inference-type stanley`（不走 HTTP server）
2. **Lekai HTTP server**：启动 `server_lekai.py`，CLI 走 HTTP

### 方案 A：Stanley 本地单进程

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type stanley \
    --checkpoint-path path/to/model.ckpt
```

### 方案 B：Lekai HTTP server

```bash
# 终端 1：启动 Lekai 推理服务器
LEKAI_CHECKPOINT_PATH=path/to/lekai_checkpoint.safetensors \
LEKAI_DEVICE=auto \
LEKAI_DTYPE=auto \
uv run python -m streammuse.infrastructure.inference.server_lekai

# 终端 2：启动 CLI 客户端（走 HTTP）
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type http \
    --model-name lekai \
    --server-url http://localhost:8000/generate_accompaniment \
    --generation-interval-ticks 4 \
    --generation-length-frames 16
```

---

## 完整 CLI 参数示例

```bash
# MIDI 设备输入 + 实时音频输出
uv run streammuse-cli \
    --input-mode midi_device \
    --midi-device-name "My MIDI Keyboard" \
    --output-type audio \
    --midi-out-port "My MIDI Output"

# MIDI 文件模拟 + 会话日志（含 MIDI 录制和 JSON 日志）
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/C_major/pop909_216_mel.mid \
    --output-type composite \
    --log-dir logs
```

## 默认 MIDI 产物与日志目录结构

当前版本中，除了 `--output-type midi_file` 外，CLI 会创建 session 目录，目录结构为：

```text
logs/YYYY-MM-DD/session_HHMMSS/
```

默认 MIDI 输出行为：
1. `console` / `audio` / `websocket`：自动写 `combined.mid`。
2. `session` / `composite`：保持原有 `combined.mid` 行为（并继续输出 JSON 日志）。
3. `json_log`：仍然不写 `combined.mid`（仅 JSON）。
4. `midi_file`：只写用户指定的 `--midi-file-output-path`，不创建 session 目录。

---

### macOS Apple Silicon 建议

在 Apple Silicon（M1/M2/M3）上建议先使用以下默认值：

1. `LEKAI_DEVICE=auto`（优先 `mps`，失败可回退）。
2. `LEKAI_DTYPE=auto`（当前策略下 `mps` 默认 `float16`）。
3. `generation-length-frames` 从 `8~16` 起步，再根据延迟逐步调高。

可通过以下命令确认 server 运行模式和设备：

```bash
curl -s http://127.0.0.1:8000/runtime_info
```

### Lekai 服务器环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LEKAI_CHECKPOINT_PATH` | `None` | 模型 checkpoint 路径（可选） |
| `LEKAI_DEVICE` | `auto` | 设备选择：`auto` / `mps` / `cpu` / `cuda` |
| `LEKAI_DTYPE` | `auto` | 精度选择：`auto` / `float32` / `float16` |
| `LEKAI_ENABLE_MPS_FALLBACK` | `true` | `mps` 加载失败时是否自动回退到 CPU |
| `LEKAI_USE_CACHE` | `true` | 生成时是否使用 KV cache |
| `LEKAI_WARMUP_STEPS` | `1` | 启动后 warmup 的最小生成步数 |
| `LEKAI_MAX_GENERATION_LENGTH_FRAMES` | `None` | 可选上限，限制每次生成长度 |
| `LEKAI_MAX_PROMPT_TICKS` | `None` | 可选上限，限制 prompt 长度 |
| `LEKAI_SERVER_HOST` | `0.0.0.0` | 服务器监听地址 |
| `LEKAI_SERVER_PORT` | `8000` | 服务器监听端口 |

---

## 本机端到端运行（stanley 引擎）

若想在单进程中运行（无 HTTP 服务器），使用 `--inference-type stanley`：

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type stanley \
    --checkpoint-path path/to/model.ckpt
```

---

## 音乐注入能力说明

当前 CLI 没有 `--injection-file` / `--injection-length` 参数。

如果需要注入历史，请调用 HTTP API：
1. `POST /inject_notes`
2. `GET /injection_status`
3. `POST /clear_history`

详见：[music-injection](music-injection.md)

---

## 停止服务

客户端：按 `Ctrl+C`，服务将自动执行清理（等待线程结束、保存日志文件）。

推理服务器：按 `Ctrl+C`。

---

## 常见问题

**Q：找不到 MIDI 设备**

```python
import mido
print(mido.get_input_names())   # 查看可用输入设备
print(mido.get_output_names())  # 查看可用输出设备
```

**Q：连接推理服务器失败**

检查服务器地址是否正确（默认 `http://localhost:8000`），可通过 `--server-url` 参数修改。

**Q：latency 过高**

降低 `--generation-length-frames`（如设为 10）可以减少每次推理的计算量；降低 `--generation-interval-ticks` 则会增加推理频率。
