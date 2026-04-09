---
title: Application 层总览
description: StreamMUSE Application 层的职责、组件构成与组装流程
---

# Application 层总览

**源文件**：`src/streammuse/application/`

Application 层是系统的**编排层**，负责将各组件组装为可运行的服务。

当前实现中：
1. `services/` 主要依赖 Domain 协议与类型。
2. `factories/` 作为组合入口，会直接导入 Infrastructure 的具体实现类并完成装配。

---

## 职责

- 定义配置数据模型（`ApplicationConfig` 及其子配置）
- 通过 Factory 将配置转换为 Domain 接口的具体实现
- 运行核心实时服务（`RealTimeMusicService`）

---

## 组件构成

```
application/
├── config/
│   └── models.py          # TempoConfig、InputConfig、OutputConfig、InferenceConfig、ApplicationConfig
├── factories/
│   ├── input_factory.py   # InputSourceFactory
│   ├── output_factory.py  # OutputSinkFactory
│   └── inference_factory.py  # InferenceEngineFactory
└── services/
    └── real_time_music_service.py  # RealTimeMusicService
```

---

## 组装流程

```
CLI args
    │
    ▼
ApplicationConfig（由 config_parser 构建）
    │
    ├──▶ InputSourceFactory.create(config)  ──▶ InputSource（实现类）
    ├──▶ OutputSinkFactory.create(config)   ──▶ OutputSink（实现类）
    └──▶ InferenceEngineFactory.create(config) ──▶ InferenceEngine（实现类）
                                  │
                                  ▼
                    RealTimeMusicService(
                        input_source=...,
                        output_sink=...,
                        inference_engine=...,
                        tempo=...,
                        scheduler=...
                    )
                                  │
                                  ▼
                          service.start()
```

**关键原则**：`RealTimeMusicService` 持有的是 Domain 接口（`InputSource`、`OutputSink`、`InferenceEngine`），而非任何具体类。这使得在测试中可以轻松用 mock 替换任何组件。

---

## 详细文档

- [config.md](config.md) — 配置数据模型
- [factories.md](factories.md) — 三个 Factory 的实现细节
- [service.md](service.md) — `RealTimeMusicService` 三线程架构
