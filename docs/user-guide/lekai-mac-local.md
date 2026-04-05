---
title: Lekai 在 macOS 本地运行
description: Apple Silicon 上的离线推理与本地实时服务指南
---

# Lekai 在 macOS 本地运行

本文面向 Apple Silicon（M1/M2/M3）环境，提供从离线验证到实时服务的完整路径。

## 1. 前置检查

```bash
# 1) 激活环境并确认依赖
uv sync

# 2) 检查 MPS 能力
uv run python -c "import torch; print('mps_built=', torch.backends.mps.is_built(), 'mps_available=', torch.backends.mps.is_available())"
```

准备以下资源：

1. checkpoint（推荐 `.safetensors`）
2. NPZ 数据目录（例如 `prompts/inputs_lekai/npz`）

## 2. 离线推理（Offline First）

先验证模型本体可运行，再进入实时模式。

```bash
uv run python scripts/run_lekai_offline.py \
  --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  --npz-dir prompts/inputs_lekai/npz \
  --output-dir output/lekai_offline \
  --device auto \
  --dtype auto \
  --condition-idx 0
```

成功后会输出：

1. 生成 MIDI：`*_generated.mid`
2. GT 参考 MIDI：`*_gt.mid`
3. 控制台统计：device、dtype、单曲耗时

## 3. 启动本地实时服务

```bash
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=auto \
LEKAI_DTYPE=auto \
LEKAI_ENABLE_MPS_FALLBACK=true \
python -m streammuse.infrastructure.inference.server_lekai
```

另开一个终端启动 CLI：

```bash
uv run streammuse-cli \
  --input-mode keyboard \
  --model-name lekai \
  --generation-interval-ticks 4 \
  --generation-length-frames 16
```

## 4. 运行状态验证

```bash
# 服务存活
curl -s http://127.0.0.1:8000/health

# 运行时状态（推荐）
curl -s http://127.0.0.1:8000/runtime_info
```

`runtime_info` 重点字段：

1. `mode`：`real_model` / `rule_stub`
2. `resolved_device`：最终设备（`mps` / `cpu` / `cuda`）
3. `resolved_dtype`：最终精度
4. `fallback_reason`：发生回退时的原因

## 5. 参数建议（M1 16GB）

1. low-latency：`generation-length-frames=8~12`, `generation-interval-ticks=4~8`
2. balanced：`generation-length-frames=16`, `generation-interval-ticks=4`
3. quality-first：`generation-length-frames=20`（可能需要提高 interval）

建议先保证稳态 p95 延迟低于触发预算，再逐步加大生成长度。

## 6. 排障

1. 模型未加载（进入 `rule_stub`）：检查 `LEKAI_CHECKPOINT_PATH` 是否正确。
2. MPS 报错：保留 `LEKAI_ENABLE_MPS_FALLBACK=true`，或临时强制 `LEKAI_DEVICE=cpu`。
3. 延迟过高：优先降低 `generation-length-frames`，再提高 `generation-interval-ticks`。
4. checkpoint key mismatch：优先使用与当前模型结构一致的 `.safetensors`。
