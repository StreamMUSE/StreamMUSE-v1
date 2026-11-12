# Mac 本地 Lekai 服务实现报告（mac-local-server-report）

日期：2026-04-02
对应计划：`informal-docs/mac-local-server-plan.md`
目标机器：Apple Silicon M1（16GB）

## 1. 结论摘要

本次已按计划完成 Lekai 在 Mac 本地的核心落地：

1. 完成 Offline-First 到 Real-Time Server 的全链路实现。
2. 完成统一 device/dtype 策略、MPS 优先与 MPS->CPU fallback。
3. 完成可观测接口 `/runtime_info` 与启动诊断日志增强。
4. 完成 checkpoint 格式 hardening（safetensors + pt/pth/ckpt 兼容加载路径）。
5. 完成 benchmark 工具与报告产物。
6. 完成测试扩展与文档收口。

交付报告产物：

1. `informal-docs/mac-m1-baseline-report.md`
2. `informal-docs/mac-m1-benchmark-report.md`
3. `informal-docs/mac-local-server-report.md`（本文件）

## 2. 实现范围与代码变更

### 2.1 Runtime 设备策略（Phase 1 + 2）

新增统一模块：

- `src/streammuse/infrastructure/inference/runtime_device.py`

实现内容：

1. `resolve_device(preference)`：支持 `auto|mps|cpu|cuda`，默认 `cuda -> mps -> cpu`。
2. `resolve_dtype(device, preference)`：支持 `auto|float16|float32`，避免仅 CUDA 语义。
3. `is_mps_available()`、`dtype_to_name()`、`parse_env_bool()`。

接入改造：

1. `inference.py`、`inference_v2.py` 改为统一调用 runtime_device。
2. `lekai_http_backend.py` 在模型加载时统一解析 device/dtype。

### 2.2 Lekai 模型加载与兼容性（Phase 1 + 3）

改造文件：

- `src/streammuse/infrastructure/inference/lekai_model/inference.py`
- `src/streammuse/infrastructure/inference/lekai_model/inference_v2.py`
- `src/streammuse/infrastructure/inference/lekai_model/inference_adapter.py`

实现内容：

1. 删除 `sys.path.append` hack，统一相对导入（`.xxx`）。
2. `load_model()` 增加 `dtype`、`use_cache` 参数。
3. 新增 checkpoint 解析：
   - 主路径：`.safetensors`
   - 兼容：`.pt/.pth/.ckpt`
4. 新增 strict key adapter 策略，处理 key 前缀不一致（如 `model.`）。

### 2.3 Server Backend 与可观测性（Phase 2 + 3）

改造文件：

- `src/streammuse/infrastructure/inference/lekai_http_backend.py`
- `src/streammuse/infrastructure/inference/server_lekai.py`

实现内容：

1. `LekaiHttpBackend` 增加 runtime metadata：
   - `resolved_device`
   - `resolved_dtype`
   - `fallback_reason`
   - `load_time_ms`
   - `warmup_time_ms`
2. 新增 MPS fallback：`LEKAI_ENABLE_MPS_FALLBACK=true|false`。
3. 新增 warmup 与缓存开关：
   - `LEKAI_WARMUP_STEPS`
   - `LEKAI_USE_CACHE`
4. 新增长度限制开关：
   - `LEKAI_MAX_GENERATION_LENGTH_FRAMES`
   - `LEKAI_MAX_PROMPT_TICKS`
5. 新增 `GET /runtime_info`（mode/device/dtype/checkpoint/fallback/load/warmup）。
6. 启动日志增强（mode/device/dtype/checkpoint/use_cache/fallback）。

### 2.4 工具脚本（Phase 1 + 4）

新增脚本：

1. `scripts/run_lekai_offline.py`
   - 支持单曲或 `--condition-idx all` 批量。
   - 输出 generated MIDI + GT MIDI。
2. `scripts/benchmark_lekai_http.py`
   - 固定 payload 循环请求。
   - 统计并输出 p50/p95/p99（JSON）。

### 2.5 测试与文档（Phase 5）

新增/改造测试：

1. `tests/unit/infrastructure/inference/test_runtime_device.py`（新增）
2. `tests/integration/test_lekai_runtime_info_integration.py`（新增）
3. `tests/unit/infrastructure/inference/test_lekai_http_backend.py`（扩展）
4. `tests/unit/infrastructure/inference/test_server_lekai.py`（扩展）

更新文档：

1. `docs/getting-started/installation.md`
2. `docs/user-guide/running-realtime.md`
3. `docs/reference/cli-reference.md`
4. `docs/user-guide/lekai-mac-local.md`（新增）

## 3. 验证结果

### 3.1 自动化测试

执行结果（定向）：

1. `tests/unit/infrastructure/inference/`：41 passed
2. 新增/相关测试集合：22 passed

结论：本次改造覆盖点（runtime_device、backend runtime_info、fallback、server 协议）均通过。

### 3.2 Offline 实机验证（M1 + MPS）

结论：成功加载真实 checkpoint 并输出 MIDI。

产物：

1. `output/lekai_offline_smoke/000_5_generated.mid`
2. `output/lekai_offline_smoke/000_5_gt.mid`

### 3.3 Online 实机验证（FastAPI）

结论：`real_model` 模式可启动并正常响应。

关键观察：

1. `/runtime_info` 返回 200，字段完整。
2. `/generate_accompaniment` 返回 200 且含模型事件（非 stub 空行为）。

## 4. Benchmark 结果与默认参数

raw 数据：

1. `informal-docs/mac-m1-benchmark-raw.json`

分析报告：

1. `informal-docs/mac-m1-benchmark-report.md`

结论摘要：

1. 推荐默认：`generation_length_frames=8`, `generation_interval_ticks=4`。
2. 可选 balanced：`length=12`, `interval=4`。
3. 当前不建议默认 `length=20, interval=4`（出现明显异常长尾）。

## 5. 与计划项对照（DoD）

按计划核心 DoD 判定：

1. M1 离线推理可稳定生成可播放 MIDI：已完成。
2. 本地 server 可在 real-model 模式处理请求：已完成。
3. `runtime_info` 可展示 mode/device/dtype/checkpoint/fallback：已完成。
4. Mac 专项测试、基准报告、用户文档齐全：已完成。
5. 默认参数建议已给出并有 p95 依据：已完成（见 benchmark 报告）。

## 6. 已知风险与建议

1. `length=20` 在部分组合上长尾明显，建议默认配置保守。
2. 目前 benchmark 单组样本量偏小，建议后续扩大到 `n>=30`。
3. 建议将 `runtime_info` 打点接入 session logging，便于线上归因。

## 7. 交付清单（本轮）

代码层：runtime/device/backend/server/model-loader/脚本/测试/文档均已提交改造。
文档层：baseline + benchmark + final report 已补齐。

本轮“implement all + report”目标已完成。
