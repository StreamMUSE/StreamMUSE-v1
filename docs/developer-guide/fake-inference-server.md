---
title: fake_inference_server — 假推理服务器
description: 无需真实模型即可测试系统的 fake HTTP 服务器
---

# fake_inference_server — 假推理服务器

**源文件**：`scripts/fake_inference_server.py`

一个轻量级 FastAPI 服务器，将请求中的 `melody_notes` 回显为伴奏事件（echo 模式），无需加载真实 ML 模型。适合开发阶段快速验证系统功能。

---

## 启动

```bash
uv run python scripts/fake_inference_server.py
```

默认监听 `http://localhost:8000`，与生产推理服务器使用相同的 API 路径。

---

## 支持的端点

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/generate_accompaniment` | 回显 melody 事件作为伴奏 |
| `POST` | `/inject_notes` | 接受注入请求，返回成功（不实际处理） |
| `POST` | `/clear_history` | 返回成功 |
| `GET` | `/injection_status` | 返回固定状态 |

---

## 响应格式

`/generate_accompaniment` 返回与真实服务器相同的 JSON 结构：

```json
{
  "accompaniment": [
    {"type": "note_on", "pitch": 60, "tick": 12, "velocity": 80},
    {"type": "note_off", "pitch": 60, "tick": 16}
  ],
  "timings": {
    "request_arrival_time": 1700000000.0,
    "inference_start_time": 1700000000.001,
    "inference_end_time": 1700000000.010,
    "response_output_time": 1700000000.011,
    "preprocess_start_time": 1700000000.0,
    "postprocess_start_time": 1700000000.010
  }
}
```

---

## 使用场景

**场景 1：开发 UI/WebSocket 集成**

```bash
# 启动 fake 服务器
uv run python scripts/fake_inference_server.py &

# 启动 WebSocket 输出模式
uv run streammuse-cli --input-mode keyboard --output-type websocket
```

**场景 2：测试会话日志功能**

```bash
uv run python scripts/fake_inference_server.py &
uv run streammuse-cli --input-mode keyboard --output-type composite --log-dir test_logs
```

**场景 3：HTTP 客户端单元测试**

可以在测试中使用 `fake_inference_server` 作为真实的 HTTP 端点，验证 `HttpInferenceClient` 的序列化/反序列化逻辑。

---

## 与 ListInput 配合

对于全自动化测试（不需要按键），结合 `ListInput`（list 输入模式）和 fake server：

```python
# 在测试代码中直接构建组件，不通过 CLI
input_source = ListInput(events=[...])
output_sink = ConsoleOutputSink()
# 使用 HttpInferenceClient 连接 fake server
engine = HttpInferenceClient(HttpInferenceClientConfig("http://localhost:8000/generate_accompaniment"))
```
