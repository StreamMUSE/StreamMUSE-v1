# Mac M1 Baseline Report

日期：2026-04-02
目标：建立 Lekai 本地运行（offline + server）的基线证据

## 1. 环境信息

- 机器：Apple Silicon M1，16GB Unified Memory
- 操作系统：macOS 15.1 (24B83)
- Python：3.10.18（项目 `.venv`）
- PyTorch：2.7.1
- CUDA：不可用
- MPS：`is_built=True`, `is_available=True`

## 2. 资产检查

- checkpoint：`models/ModelLekai/epoch_4_1104_1204/model.safetensors`（存在）
- NPZ 数据：`prompts/inputs_lekai/npz`（存在，5 个样本）

## 3. Offline 基线验证

执行命令：

```bash
uv run python scripts/run_lekai_offline.py \
  --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  --npz-dir prompts/inputs_lekai/npz \
  --output-dir output/lekai_offline_smoke \
  --device auto \
  --dtype auto \
  --condition-idx 0
```

关键结果：

1. 设备解析：`mps`
2. 精度：`float16`
3. 模型加载成功，参数量：`170,309,376`
4. 推理耗时（单样本）：`~82.9s`
5. 生成产物：
   - `output/lekai_offline_smoke/000_5_generated.mid`
   - `output/lekai_offline_smoke/000_5_gt.mid`
6. 生成 MIDI 为双轨（melody + accompaniment），可播放。

## 4. Server 基线验证

启动命令：

```bash
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=auto \
LEKAI_DTYPE=auto \
LEKAI_SERVER_PORT=8016 \
uv run python -m streammuse.infrastructure.inference.server_lekai
```

关键结果：

1. 启动模式：`real_model`
2. 设备：`mps`
3. 精度：`float16`
4. checkpoint 格式：`safetensors`
5. `/runtime_info` 返回完整字段（mode/device/dtype/checkpoint/fallback/warmup/load info）
6. `/generate_accompaniment` 返回 `200` 且包含非空伴奏事件。

## 5. 当前观察

1. 功能正确性已跑通（offline + server）。
2. 本机实时性需依赖参数控制（详见 benchmark 报告）。
3. 仍有 `pretty_midi` 的 `pkg_resources` 弃用 warning（不影响功能）。
