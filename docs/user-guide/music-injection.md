---
title: 音乐注入（music injection）
description: 通过 HTTP API 注入历史上下文（/inject_notes、/injection_status、/clear_history）
---

# 音乐注入

音乐注入用于在会话开始前预填充模型历史，使模型不必从空上下文启动。

---

## 当前实现状态

1. CLI 当前**没有** `--injection-file` / `--injection-length` 参数。
2. 注入能力仍然存在于推理接口层（`InferenceEngine.inject_history`）和 HTTP server API。
3. 若要使用注入，请直接调用 HTTP API。

---

## 适用服务

以下服务实现了注入相关端点：

1. `scripts/fake_inference_server.py`（用于开发联调）
2. `src/streammuse/infrastructure/inference/server_lekai.py`

端点：

1. `POST /inject_notes`
2. `GET /injection_status`
3. `POST /clear_history`

---

## API 用法示例

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

---

## 数据格式说明

`/inject_notes` 请求体中的每个事件至少需要：

1. `type`：`note_on` 或 `note_off`
2. `pitch`：MIDI 音高
3. `tick`：绝对 tick

`velocity` 可选。

---

## 与 CLI 的关系

如果你使用 `streammuse-cli --inference-type http`，CLI 本身不会自动发注入请求。

常见做法：

1. 先用 `curl` 调 `/inject_notes`
2. 再启动或继续运行 CLI 进行实时推理

---

## Python 调用示例

```python
import requests

payload = {
        "melody_notes": [{"type": "note_on", "pitch": 60, "tick": 0}, {"type": "note_off", "pitch": 60, "tick": 4}],
        "accompaniment_notes": [{"type": "note_on", "pitch": 48, "tick": 0}, {"type": "note_off", "pitch": 48, "tick": 4}],
        "injection_length_ticks": 16,
}

resp = requests.post("http://127.0.0.1:8000/inject_notes", json=payload, timeout=10)
resp.raise_for_status()
print(resp.json())
```

