---
title: infrastructure/inference — 推理引擎适配器总览
description: 两种 InferenceEngine 实现的架构与选择指南
---

# infrastructure/inference — 推理引擎适配器总览

**源文件**：`src/streammuse/infrastructure/inference/`

所有推理引擎适配器均实现 `InferenceEngine` 协议（见 [domain/interfaces](../../domain/interfaces.md)）。

---

## 三种实现对比

| 类 | 模式 | 说明 |
|---|---|---|
| `HttpInferenceClient` | 远端 HTTP 服务器 | 将推理请求通过 REST API 发送到独立服务器进程 |
| `StanleyInferenceEngine` | 本地进程内 | 直接在当前进程中运行 RoFormer 模型 |
| `server_lekai + LekaiHttpBackend` | 远端 HTTP 服务器 | FastAPI server 内使用 Lekai backend，并复用相同 HTTP 协议 |

---

## 部署架构

**HTTP 模式**（默认、生产环境）：

```
CLI 进程
  └── HttpInferenceClient
          │ HTTP POST /generate_accompaniment
          ▼
        推理服务器（FastAPI）
                ├── Stanley backend
                └── Lekai backend
```

**Stanley 本地模式**：

```
CLI 进程
  └── StanleyInferenceEngine
          └── LegacyInferenceEngineStanley
                  └── RoFormerSymbolicTransformer
```

---

## 两层适配器模式（Stanley）

Stanley 引擎采用两层适配器隔离 Domain/Clean Architecture 模型与旧有 ML 代码：

1. **`StanleyInferenceEngine`**（接口适配层）：
   - 实现 `InferenceEngine` Protocol
   - 负责 `MusicalEvent` ↔ `{pitch, tick, duration}` 字典的转换
   - 遵循 close-at-horizon 策略

2. **`LegacyInferenceEngineStanley`**（ML 引擎层）：
   - 操作 `{pitch, tick, duration}` 字典和 piano-roll 张量
   - 无 Domain 依赖，保持 ML 代码的独立性

---

## 文件结构

```
inference/
├── http_client.py         # HttpInferenceClient
├── server_lekai.py        # Lekai HTTP server (FastAPI)
├── lekai_http_backend.py  # Lekai backend used by server_lekai
├── serialization.py       # event_to_dict / event_from_dict / timing_info_from_dict
├── stanley_engine.py      # StanleyInferenceEngine（接口适配层）
├── stanley_legacy.py      # LegacyInferenceEngineStanley（ML 引擎层）
├── stanley_stack/         # RoFormer 模型定义与预处理
│   ├── m2a_transformer.py
│   └── preprocess/
├── config/                # ML 配置 Schema（OmegaConf/Hydra 格式）
└── tokenization/          # MIDI tokenizer
```

---

## 详细文档

- [http_client.md](http_client.md) — `HttpInferenceClient`
- [overview.md](overview.md) — `server_lekai` / `LekaiHttpBackend` HTTP-first 架构说明
- [serialization.md](serialization.md) — 序列化/反序列化辅助函数
- [stanley_engine.md](stanley_engine.md) — `StanleyInferenceEngine`（接口适配层）
- [stanley_legacy.md](stanley_legacy.md) — `LegacyInferenceEngineStanley`（ML 引擎层）
