# Mac 本地 Lekai 服务计划（Review 修订版）

更新时间：2026-04-02
目标机器：macOS Apple Silicon（M1，16GB Unified Memory）
执行策略：Offline-First，再接入 Real-Time Server

## 0. 审阅结论（针对当前仓库代码）

本计划已基于当前代码状态做修订，以下是关键审阅结论：

1. 已实现，不应再作为待修复项：`LEKAI_CHECKPOINT_PATH` 已在 `server_lekai.py` 启动时读取并传入 backend。
2. 已实现，不应再作为待修复项：`_generate_with_model()` 已使用真实 `beats_to_pianoroll` 解码路径，不再是全零占位。
3. 已实现，不应再作为待修复项：`generation_length_frames` 与 `generation_interval_ticks` 的职责分离已经修复（length 决定生成时长，interval 仅调度）。
4. 仍需重点实现：MPS 设备路径、MPS->CPU fallback 策略、可观测 runtime API、Mac 专项验证与文档。

## 1. 功能定义

### 1.1 目标（Goal）

在 M1 16GB 上支持 Lekai 本地运行，分两阶段完成：

1. Stage 1（Offline）：在本机加载 checkpoint，基于 NPZ 调用 `generate_accompaniment`，输出 MIDI。
2. Stage 2（Real-Time）：将离线已验证的模型路径接入 `server_lekai.py`，稳定服务 CLI 实时请求。

### 1.2 非目标（Non-goals）

1. 不做训练链路优化。
2. 不改模型结构。
3. 不承诺高负载参数在 M1 上都能实时。

## 2. 当前基线（As-Is）

### 2.1 已具备能力

1. `lekai_model/` 下模型、tokenizer、数据转换与 MIDI 导出链路存在。
2. `prompts/inputs_lekai/npz` 数据目录可用。
3. `models/ModelLekai/epoch_4_1104_1204/model.safetensors` 存在。
4. `server_lekai.py` + `lekai_http_backend.py` 已有可用服务框架与 stub fallback。

### 2.2 主要缺口（必须补齐）

1. `inference.py` 仍有 `sys.path.append` 和 `from lekai_model...` 导入方式，模块化调用兼容性差。
2. 设备选择仍是 CUDA/CPU 二选一，未显式支持 `mps`。
3. `load_model()` 的 dtype 策略偏 CUDA 语义，未形成统一 `device + dtype` 决策。
4. 缺少干净的离线单曲/批量推理脚本（当前 `__main__` 逻辑硬编码路径且偏实验脚本）。
5. 服务端缺少 `runtime_info` 类可观测接口（当前仅 `/health`）。
6. 缺少 Mac 专项测试矩阵与基准报告。

## 3. 目标架构（To-Be）

### 3.1 Runtime 策略

新增统一环境变量策略：

1. `LEKAI_DEVICE=auto|mps|cpu|cuda`
2. `LEKAI_DTYPE=auto|float32|float16`
3. `LEKAI_ENABLE_MPS_FALLBACK=true|false`
4. `LEKAI_USE_CACHE=true|false`
5. `LEKAI_WARMUP_STEPS=<int>`

推荐默认策略：

1. `auto` 按 `cuda -> mps -> cpu` 选择。
2. Apple Silicon 上优先 `mps`。
3. `mps` warmup 失败且 fallback 打开时自动降级到 `cpu`，并记录 `fallback_reason`。

### 3.2 可观测性（Observability）

新增 `GET /runtime_info`，返回：

1. `mode`（`real_model` / `rule_stub`）
2. `device`
3. `dtype`
4. `checkpoint_path`
5. `checkpoint_format`
6. `fallback_reason`
7. `load_time_ms`、`warmup_time_ms`

## 4. 分阶段实施计划（Phased Plan）

## Phase 0：预检与基线固化

目标：先把输入资产与环境固定，避免后续问题归因不清。

任务：

- [ ] P0-1 记录环境信息：macOS 版本、Python 版本、PyTorch 版本、MPS 是否可用。
- [ ] P0-2 固化测试资产：确认 `checkpoint`、`npz`、输出目录。
- [ ] P0-3 建立基线报告模板：记录首推理时延、steady-state p50/p95、失败日志。

产物：

1. `informal-docs/mac-m1-baseline-report.md`

验收：

- [ ] P0-E1 资产路径与版本信息完整可复现。

---

## Phase 1：离线链路可运行（Offline First）

目标：在 Mac 上一条命令完成“加载模型 -> 生成 -> 输出 MIDI”。

### Task 1.1 修复 `inference.py` 模块化导入

- [ ] 1.1.1 将 `from lekai_model.xxx` 改为相对导入 `from .xxx`。
- [ ] 1.1.2 删除 `sys.path.append(...)` hack。
- [ ] 1.1.3 同步检查 `inference_v2.py` 是否存在相同问题。
- [ ] 1.1.4 验证：
  - `uv run python -c "from streammuse.infrastructure.inference.lekai_model.inference import load_model"`

### Task 1.2 引入设备解析（支持 MPS）

- [ ] 1.2.1 在 `lekai_model/inference.py` 增加 `resolve_device(preference="auto")`。
- [ ] 1.2.2 新增 `resolve_dtype(device, preference="auto")`，统一 dtype 策略。
- [ ] 1.2.3 将 `load_model()` 的精度逻辑改为基于 `device` 与 `dtype`，不再只依赖 CUDA 条件。
- [ ] 1.2.4 在 M1 上验证：`auto` 能优先选择 `mps`（若可用）。

### Task 1.3 新建离线入口脚本

- [ ] 1.3.1 新建 `scripts/run_lekai_offline.py`。
- [ ] 1.3.2 参数支持：
  - `--checkpoint`
  - `--npz-dir`
  - `--output-dir`
  - `--device`
  - `--dtype`
  - `--condition-idx`
  - `--gt-prefix-beats`
  - `--temperature` / `--top-k` / `--top-p`
- [ ] 1.3.3 支持 `--condition-idx all` 批量运行。
- [ ] 1.3.4 输出：生成 MIDI + GT MIDI，并打印每首耗时。

Phase 1 验收：

- [ ] P1-E1 单曲推理成功，输出 MIDI 可播放。
- [ ] P1-E2 5 首 NPZ 批量推理无崩溃。
- [ ] P1-E3 `cpu` 与 `mps` 两种路径均可运行（若 `mps` 不可用需有明确提示）。

---

## Phase 2：Server 接入与设备统一

目标：将离线验证通过的模型路径接入实时服务，形成 Mac 本地可用 server。

### Task 2.1 抽取统一设备模块

- [ ] 2.1.1 新建 `src/streammuse/infrastructure/inference/runtime_device.py`。
- [ ] 2.1.2 提供：
  - `resolve_device(preference: str) -> str`
  - `resolve_dtype(device: str, preference: str) -> torch.dtype`
  - `is_mps_available() -> bool`
- [ ] 2.1.3 `inference.py` 与 `lekai_http_backend.py` 统一调用该模块。

### Task 2.2 Backend 设备与 fallback 策略

- [ ] 2.2.1 `LekaiHttpBackend._load_model()` 使用 `LEKAI_DEVICE/LEKAI_DTYPE`。
- [ ] 2.2.2 新增 `LEKAI_ENABLE_MPS_FALLBACK`：MPS warmup 失败自动降级 CPU。
- [ ] 2.2.3 保存 runtime metadata：`resolved_device`、`resolved_dtype`、`fallback_reason`、`load_time_ms`。

### Task 2.3 现有已修复项回归确认（不重复开发）

以下内容已存在于当前主线代码，本阶段只做回归验证：

- [ ] 2.3.1 `LEKAI_CHECKPOINT_PATH` 启动注入链路回归。
- [ ] 2.3.2 `beats_to_pianoroll` 真正解码路径回归。
- [ ] 2.3.3 length/interval 职责分离回归。

Phase 2 验收：

- [ ] P2-E1 Mac 上 server 可在 real-model 模式启动。
- [ ] P2-E2 `/generate_accompaniment` 返回非空模型伴奏（不是 stub 行为）。
- [ ] P2-E3 CLI `--model-name lekai` 端到端可跑通。

---

## Phase 3：可观测性与可靠性

目标：让运行状态“可看见、可定位、可回退”。

### Task 3.1 API 与日志增强

- [ ] 3.1.1 新增 `GET /runtime_info`。
- [ ] 3.1.2 启动 banner 打印：mode/device/dtype/checkpoint/fallback。
- [ ] 3.1.3 stub 模式下打印明确 warning（避免误判为真实模型）。

### Task 3.2 Checkpoint format hardening

- [ ] 3.2.1 `load_model()` 支持 `.safetensors`（主路径）。
- [ ] 3.2.2 可选支持 `.pt/.pth`（定义明确 key contract）。
- [ ] 3.2.3 不支持格式时给出可操作错误信息。

### Task 3.3 Warmup 与稳定性开关

- [ ] 3.3.1 增加 `LEKAI_WARMUP_STEPS`。
- [ ] 3.3.2 增加 `LEKAI_USE_CACHE`。
- [ ] 3.3.3 增加 `LEKAI_MAX_GENERATION_LENGTH_FRAMES`、`LEKAI_MAX_PROMPT_TICKS`。

Phase 3 验收：

- [ ] P3-E1 `curl /runtime_info` 可直接确认运行模式与设备。
- [ ] P3-E2 30 分钟连续请求无崩溃。
- [ ] P3-E3 首次请求时延显著下降或稳定可解释。

---

## Phase 4：性能评估（M1 16GB）

目标：给出可用参数档位与真实 latency 数据。

### Task 4.1 基准脚本

- [ ] 4.1.1 新建 `scripts/benchmark_lekai_http.py`。
- [ ] 4.1.2 固定 payload 循环请求并统计 p50/p95/p99。

### Task 4.2 参数矩阵

- [ ] 4.2.1 `generation_length_frames` 测试集：`{8, 12, 16, 20}`。
- [ ] 4.2.2 `generation_interval_ticks` 测试集：`{2, 4, 8}`。
- [ ] 4.2.3 设备维度：`cpu` 与 `mps`（可用时）。

### Task 4.3 输出报告

- [ ] 4.3.1 生成 `informal-docs/mac-m1-benchmark-report.md`。
- [ ] 4.3.2 给出推荐 profile（low-latency / balanced / quality-first）。

Phase 4 验收：

- [ ] P4-E1 每组参数均有可复现实测数据。
- [ ] P4-E2 给出明确默认参数建议。

---

## Phase 5：测试与文档收口

目标：保证可维护性，避免后续回归。

### Task 5.1 测试

- [ ] 5.1.1 新增 `tests/unit/infrastructure/inference/test_runtime_device.py`。
- [ ] 5.1.2 扩展 `test_lekai_http_backend.py`：覆盖 MPS fallback 行为（mock）。
- [ ] 5.1.3 扩展 `test_server_lekai.py`：覆盖 `/runtime_info` 协议。
- [ ] 5.1.4 增加一条 Mac 关注的集成测试（可 mock 设备能力，不依赖真实硬件）。

### Task 5.2 文档

- [ ] 5.2.1 更新 `docs/user-guide/running-realtime.md`（Apple Silicon 小节）。
- [ ] 5.2.2 更新 `docs/getting-started/installation.md`（Mac 依赖与排障）。
- [ ] 5.2.3 更新 `docs/reference/cli-reference.md`（Lekai runtime env vars）。
- [ ] 5.2.4 新增 `docs/user-guide/lekai-mac-local.md`。

Phase 5 验收：

- [ ] P5-E1 新增测试通过且无 flaky。
- [ ] P5-E2 文档可复制命令一键跑通。

## 5. M1 16GB 参数建议（初版）

1. low-latency：`generation_length_frames=8~12`, `generation_interval_ticks=4~8`
2. balanced：`generation_length_frames=16`, `generation_interval_ticks=4`
3. quality-first：`generation_length_frames=20`（必要时提高 interval 或降低 BPM）

实时预算公式：

1. `seconds_per_tick = 60 / (BPM * 4)`
2. `inference_budget = generation_interval_ticks * seconds_per_tick`

要求：稳态 `p95` 推理延迟应低于 `inference_budget`。

## 6. 推荐启动命令（修订）

### 6.1 Offline

```bash
uv run python scripts/run_lekai_offline.py \
  --checkpoint models/ModelLekai/epoch_4_1104_1204/model.safetensors \
  --npz-dir prompts/inputs_lekai/npz \
  --output-dir output/lekai_offline \
  --device auto \
  --dtype auto
```

### 6.2 Real-Time Server + CLI

```bash
# Terminal 1
LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
LEKAI_DEVICE=auto \
LEKAI_DTYPE=auto \
python -m streammuse.infrastructure.inference.server_lekai

# Terminal 2
uv run streammuse-cli \
  --input-mode keyboard \
  --model-name lekai \
  --generation-interval-ticks 4 \
  --generation-length-frames 16
```

## 7. 风险与回滚

1. MPS op 不兼容：默认启用 fallback 到 CPU，并记录 `fallback_reason`。
2. 内存压力导致抖动：限制 generation/prompt 上限并调整 interval。
3. checkpoint 格式不匹配：在启动期 fail-fast，给出格式与 key 诊断。
4. 误跑到 stub：通过 `runtime_info` 与启动日志双确认。

回滚策略：

1. 立即设置 `LEKAI_DEVICE=cpu` 保守运行。
2. 暂时关闭新参数开关（回退默认路径）。
3. 保留 stub 模式可用性，确保 CLI 不中断。

## 8. Definition of Done（DoD）

全部满足才算完成：

1. M1 16GB 上离线推理可稳定生成可播放 MIDI。
2. 本地 server 可在 real-model 模式处理实时请求。
3. `runtime_info` 可完整展示 mode/device/dtype/checkpoint/fallback。
4. Mac 专项测试、基准报告、用户文档齐全。
5. 默认参数在目标机器上具备可接受稳定性（以 p95 指标验证）。
