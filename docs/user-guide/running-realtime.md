---
title: 运行实时系统
description: 启动推理服务器和 CLI 客户端的完整流程
---

# 运行实时系统

StreamMUSE 实时系统由两个进程组成：**推理服务器**（提供模型预测）和 **CLI 客户端**（读取输入、播放输出）。

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

```bash
# 终端 1：启动真实推理服务器（需要有模型 checkpoint）
CHECKPOINT_PATH=path/to/model.ckpt \
    uvicorn src.streammuse.infrastructure.inference.server:app \
    --host 0.0.0.0 --port 8000

# 终端 2：
uv run streammuse-cli --input-mode keyboard
```

### 推理服务器可选环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `CHECKPOINT_PATH` | 必填 | 模型 checkpoint 路径 |
| `MODEL_MAX_SEQ_LEN_FRAMES` | `96` | context window 帧数 |
| `GENERATION_LENGTH_FRAMES` | `20` | 每次推理生成帧数 |
| `MODEL_SIZE` | `"0.12B"` | 模型规模 |

---

## 完整 CLI 参数示例

```bash
# MIDI 设备输入 + 实时音频输出
uv run streammuse-cli \
    --input-mode midi \
    --midi-device-name "My MIDI Keyboard" \
    --output-type audio \
    --midi-out-port "My MIDI Output"

# MIDI 文件模拟 + 会话日志（含 MIDI 录制和 JSON 日志）
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file prompts/C_major/pop909_216_mel.mid \
    --output-type composite \
    --log-dir logs

# 使用注入预置历史（冷启动更好的初始伴奏）
uv run streammuse-cli \
    --input-mode keyboard \
    --injection-file prompts/C_major/pop909_216_mel.mid \
    --injection-length 50
```

---

## 使用 Lekai 模型

Lekai 是另一种基于 LLaMA 架构的伴奏生成模型。使用 Lekai 需要启动专门的推理服务器：

```bash
# 终端 1：启动 Lekai 推理服务器
# 注意：需要设置 checkpoint 路径（可选，未设置则使用规则 stub）
LEKAI_CHECKPOINT_PATH=path/to/lekai_checkpoint.pt \
    uvicorn src.streammuse.infrastructure.inference.server_lekai:app \
    --host 0.0.0.0 --port 8000

# 或使用直接启动方式
python -m src.streammuse.infrastructure.inference.server_lekai

# 终端 2：启动 CLI 客户端
# 注意：lekai 要求 generation-length-frames 为 4 的倍数（generation-interval-ticks 无此约束）
uv run streammuse-cli \
    --input-mode keyboard \
    --model-name lekai \
    --generation-interval-ticks 4 \
    --generation-length-frames 20
```

### Lekai 服务器环境变量

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LEKAI_CHECKPOINT_PATH` | `None` | 模型 checkpoint 路径（可选） |
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
