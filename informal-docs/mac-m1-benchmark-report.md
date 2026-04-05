# Mac M1 Benchmark Report (Lekai HTTP)

日期：2026-04-02
数据源：`informal-docs/mac-m1-benchmark-raw.json`

## 1. 测试配置

- 接口：`POST /generate_accompaniment`
- 服务地址：`http://127.0.0.1:8016`
- 模型模式：`real_model`
- 设备：`mps`（float16）
- 每组请求：`num_requests=2`（含 warmup 前置）

说明：当前样本量较小（每组 2 次），结果用于“方向性判断”和“异常点定位”，不是最终统计学结论。

## 2. 结果总览（p50 / p95, ms）

| generation_length_frames | generation_interval_ticks | p50 (ms) | p95 (ms) | 观察 |
|---|---:|---:|---:|---|
| 8  | 2 | 295.1 | 394.2 | 可用 |
| 8  | 4 | 286.8 | 287.5 | 较稳定 |
| 8  | 8 | 251.8 | 292.9 | 较稳定 |
| 12 | 2 | 532.5 | 619.4 | 接近 0.5s+ |
| 12 | 4 | 422.1 | 492.3 | 可用 |
| 12 | 8 | 404.2 | 496.7 | 可用 |
| 16 | 2 | 696.9 | 992.1 | 长尾明显 |
| 16 | 4 | 596.4 | 674.4 | 较可控 |
| 16 | 8 | 579.0 | 654.0 | 较可控 |
| 20 | 2 | 591.1 | 716.6 | 尚可 |
| 20 | 4 | 14954.9 | 27169.1 | **异常长尾（严重）** |
| 20 | 8 | 957.0 | 959.1 | 稳定但偏慢 |

## 3. 关键结论

1. `length=8/12` 在 M1 + MPS 上整体可用，延迟落在约 250~620ms。
2. `length=16` 开始出现较明显尾部抖动，尤其 `interval=2`。
3. `length=20, interval=4` 出现秒级到十秒级异常长尾，是当前最关键风险点。
4. `length=20, interval=8` 虽稳定，但接近 1s，实时交互体验会明显下降。

## 4. 建议的默认参数

面向实时体验，建议优先：

1. `generation_length_frames=8`
2. `generation_interval_ticks=4`（首选）或 `8`（更稳）

若追求更长上下文，可尝试：

1. `generation_length_frames=12`
2. `generation_interval_ticks=4`

暂不建议默认采用：

1. `generation_length_frames=20`
2. 特别是 `generation_interval_ticks=4`

## 5. 下一步 Benchmark 强化

1. 每组请求提升到 `n>=30`，统计分位更可靠。
2. 增加 end-to-end round-trip（含客户端节拍循环）测量。
3. 分离“模型推理时间”和“序列化/HTTP 时间”。
4. 记录运行时状态快照（temperature, use_cache, prompt length）用于异常追踪。
