# Lekai Integration Plan — HTTP Client-Server First

## 0. 核心目标（按最新 comment）

本项目里 Lekai 集成的**主路径**是：

- Client 只走 `http`（`HttpInferenceClient`）
- Server 可以是 Stanley 或 Lekai，只要遵守同一协议即可被 Client 复用

本地 in-process `lekai` 模式不作为主交付，仅作为可选调试路径。

此外保留你指定的约束：

1. 当后端是 Lekai 时，`generation_interval_ticks` 与 `generation_length_frames` 必须是 4 的倍数
2. 每次请求都传完整 `melody_history`
3. Server 运行时不从环境变量读取推理参数（interval/length/mode 等），这些参数全部由 client 请求显式传入

---

## 1. 协议优先设计

### 1.1 Client 不改协议

沿用当前 `HttpInferenceClient` 协议：

- `POST /generate_accompaniment`
- `POST /inject_notes`
- `POST /clear_history`
- `GET /injection_status`

Client 侧只关心 `--server-url`，不关心 server 内部是 Stanley 还是 Lekai。

### 1.2 Server 兼容性契约

Lekai server 必须返回与现有 `serialization.timing_info_from_dict()` 兼容的字段：

- `request_arrival_time`
- `response_output_time`
- `preprocess_start_time`
- `inference_start_time`
- `inference_end_time`
- `postprocess_start_time`

以及 `accompaniment` 事件列表（event-stream）。

同时，server 的推理参数来源约束如下：
- `generation_length_frames`、`generation_interval_ticks`、`prompt_length_ticks`、`inference_mode` 等运行时参数均来自请求体
- server 不通过环境变量决定这些推理行为参数

---

## 2. 实施范围（HTTP 路径为主）

### 2.1 必做文件

- `src/streammuse/infrastructure/inference/server_lekai.py`（新增）
- `src/streammuse/infrastructure/inference/lekai_http_backend.py`（新增，server 内部调用 Lekai 模型）
- `src/streammuse/infrastructure/inference/http_client.py`（小改：可选增强错误信息，不改协议）
- `src/streammuse/presentation/cli/config_parser.py`（仅文案：强调 `http` 是推荐生产模式）
- `docs/` 相关文档（后续）

### 2.2 可选文件（非主路径）

- `src/streammuse/infrastructure/inference/lekai_engine.py`（本地调试用，可后置）

---

## 3. 分阶段实现计划

## Phase A — 新增 Lekai HTTP Server

### A1. FastAPI server skeleton

```python
# src/streammuse/infrastructure/inference/server_lekai.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="StreamMUSE Lekai Inference Server")

class MelodyNoteEvent(BaseModel):
    type: str
    pitch: int
    tick: int

class InferenceRequest(BaseModel):
    melody_notes: list[MelodyNoteEvent]
    generation_start_tick: int
    generation_length_frames: int
    generation_interval_ticks: int
    prompt_length_ticks: int | None = None
    inference_mode: str = "sliding_window"
    model_name: str = "lekai"
    client_request_send_time: float | None = None
```

### A2. 端点实现（协议兼容）

```python
@app.post("/generate_accompaniment")
async def generate_accompaniment(req: InferenceRequest):
    # 1) 仅在 req.model_name == "lekai" 时做 4 倍数校验（见 A3）
    # 2) 用完整 req.melody_notes 调后端
    # 3) 返回 accompaniment + timings
    ...

@app.post("/inject_notes")
async def inject_notes(req: DirectInjectionRequest):
    ...

@app.post("/clear_history")
async def clear_history():
    ...

@app.get("/injection_status")
async def injection_status():
    ...
```

### A3. 服务端强校验（必做）

```python
def _validate_lekai_constraints(model_name: str, generation_interval_ticks: int, generation_length_frames: int) -> None:
    if model_name != "lekai":
        return
    if generation_interval_ticks % 4 != 0:
        raise ValueError("lekai requires generation_interval_ticks to be a multiple of 4")
    if generation_length_frames % 4 != 0:
        raise ValueError("lekai requires generation_length_frames to be a multiple of 4")
```

说明：
- 这条校验放 server 端，保证任何 client 都不会绕过
- `generation_interval_ticks` / `generation_length_frames` 均从请求读取，不从环境变量读取
- “仅 Lekai 校验”通过 `model_name` 分支控制；如果部署的是专用 `server_lekai`，`model_name` 可固定为 `lekai`

---

## Phase B — 实现 Lekai backend（server 内部）

新增 `lekai_http_backend.py`，职责是把 HTTP 请求对象转换为 Lekai 引擎输入。

关键策略（按你的要求）：

- 每次 `generate_accompaniment()` 都接收完整 melody history
- 不做 client-side 增量协议改造

```python
class LekaiHttpBackend:
    def generate(
        self,
        melody_events: list[dict],
        generation_start_tick: int,
        generation_length_frames: int,
        generation_interval_ticks: int,
        prompt_length_ticks: int | None,
        inference_mode: str,
    ):
        # melody_events 是完整 history
        # 这里调用 lekai_model 进行推理
        # 返回 event-stream accompaniment
        ...
```

---

## Phase C — Client 侧保持 HTTP 不变

`HttpInferenceClient` 不改 endpoint 协议，仅确保错误可诊断：

```python
resp = requests.post(...)
try:
    resp.raise_for_status()
except requests.HTTPError as e:
    raise RuntimeError(f"HTTP inference failed: {resp.status_code} {resp.text}") from e
```

这样当 server 报“非 4 倍数”错误时，CLI 能直接看到明确原因。

---

## Phase D — Tests（以 HTTP 路径为中心）

### D1. server_lekai 单测

新增：`tests/unit/infrastructure/inference/test_server_lekai.py`

覆盖：
- `generate_accompaniment` 正常返回结构
- 非 4 倍数时报 4xx/5xx + 明确错误信息
- `inject_notes` / `clear_history` / `injection_status` 协议正确

### D2. client-server 兼容回归

扩展：`tests/unit/infrastructure/inference/test_http_inference_client.py`

覆盖：
- client 可解析 Lekai server timings
- server 返回错误时 client 抛出可诊断异常

---

## 4. 配置与启动

### 4.1 server 启动（主路径）

```bash
uvicorn src.streammuse.infrastructure.inference.server_lekai:app --host 0.0.0.0 --port 8000
```

> 注：模型 checkpoint 采用 server 启动参数或配置文件传入（非环境变量）；推理行为参数由 client 每次请求传入

### 4.2 client 启动（不关心 server 类型）

```bash
uv run streammuse-cli \
  --inference-type http \
  --server-url http://localhost:8000/generate_accompaniment \
    --generation-interval-ticks 4 \
  --generation-length-frames 20 \
    --lekai-inference-mode sliding_window
```

---

## 5. 关键风险与对策

1. **Server 与 client timings 字段不一致**
- 对策：server response 严格对齐 `timing_info_from_dict()`

2. **非 4 倍数配置被放行（Lekai）**
- 对策：仅在 `model_name=lekai` 时 server 端 hard-fail；其他模型不做该校验

3. **完整 history 导致性能压力**
- 对策：第一阶段先保证协议正确；第二阶段再考虑 server 内部窗口裁剪（不改 HTTP 协议）

---

## 6. Definition of Done（HTTP 优先版）

- Client 使用 `--inference-type http` 可无缝连接 Lekai server
- Lekai server 完整实现四个 endpoint，并与现有 client 协议兼容
- 仅当 `model_name=lekai` 且参数非 4 倍数时被拒绝并返回明确错误
- 每次推理请求传完整 melody history，server 正常处理
- client-server 回归测试通过

---

## 7. 里程碑顺序

1. `server_lekai.py` + `lekai_http_backend.py`
2. request schema 扩展（client 显式传 interval/length/mode/model_name）
3. server 约束校验（仅 lekai 做 4 的倍数检查）
4. HTTP client 错误诊断增强
5. client-server 测试补齐
6. 文档与运行指引更新

---

## 8. Detailed Todo List (Phases + Individual Tasks)

> 说明：以下清单已按本轮实现完成并勾选。

### Phase 0 — Baseline & Scope Lock

- [x] P0-1 确认 HTTP 协议字段最终版本（request/response JSON schema）
- [x] P0-2 确认 `model_name` 的枚举约束（至少包含 `lekai`，可扩展 `stanley`）
- [x] P0-3 确认“运行时参数只来自 client request”的边界（哪些参数绝不允许 env 覆盖）
- [x] P0-4 记录兼容目标：必须兼容现有 `HttpInferenceClient` 的 endpoint 路径与反序列化字段

**Exit criteria**
- [x] P0-E1 协议文档冻结，可作为后续实现唯一依据

### Phase 1 — `server_lekai.py` skeleton + schema

- [x] P1-1 新建 `server_lekai.py` 与 FastAPI app
- [x] P1-2 定义 `MelodyNoteEvent` / `AccompanimentNoteEvent`
- [x] P1-3 定义 `InferenceRequest`（包含 `generation_interval_ticks`、`generation_length_frames`、`inference_mode`、`model_name`）
- [x] P1-4 定义 `Timings` 与 `AccompanimentResponse`
- [x] P1-5 定义 `DirectInjectionRequest` / `DirectInjectionResponse`
- [x] P1-6 定义 `InjectionStatus` 响应结构

**Exit criteria**
- [x] P1-E1 4 个 endpoint 路径可访问（即使先返回 stub）

### Phase 2 — Lekai backend 实现

- [x] P2-1 新建 `lekai_http_backend.py`，封装 Lekai 模型加载逻辑
- [x] P2-2 backend `generate()` 接口支持完整 history 输入
- [x] P2-3 backend 内部事件流转换（`note_on/note_off` ↔ 模型内部结构）
- [x] P2-4 backend 输出统一 event-stream（供 HTTP 响应）
- [x] P2-5 backend 支持 `inject_history()`
- [x] P2-6 backend 支持 `clear_history()`
- [x] P2-7 backend 支持 `injection_status()`

**Exit criteria**
- [x] P2-E1 在不接 CLI 的前提下，单独调用 backend 可生成合法 accompaniment 事件列表

### Phase 3 — Server endpoint 业务接线

- [x] P3-1 `/generate_accompaniment` 接入 backend.generate
- [x] P3-2 在 endpoint 中记录 timing 并映射到 `Timings`
- [x] P3-3 `/inject_notes` 接入 backend.inject_history
- [x] P3-4 `/clear_history` 接入 backend.clear_history
- [x] P3-5 `/injection_status` 接入 backend.injection_status
- [x] P3-6 为所有 endpoint 增加统一错误返回结构（建议包含 error code + message）

**Exit criteria**
- [x] P3-E1 端到端请求可以返回与 `timing_info_from_dict()` 兼容的数据

### Phase 4 — 约束校验（仅 Lekai）

- [x] P4-1 实现 `_validate_lekai_constraints(model_name, interval, length)`
- [x] P4-2 在 `/generate_accompaniment` 中调用校验器
- [x] P4-3 确保 `model_name != lekai` 时完全跳过 4 倍数校验
- [x] P4-4 设计错误码（如 `E_LEKAI_BAD_MULTIPLE_OF_4`）并返回可读消息
- [x] P4-5 校验器单元测试覆盖 interval 与 length 的两类失败分支

**Exit criteria**
- [x] P4-E1 非 Lekai 请求不受 4 倍数限制；Lekai 请求严格限制

### Phase 5 — Client request 参数透传（不改协议路径）

- [x] P5-1 在 `HttpInferenceClient.generate_accompaniment()` payload 中增加 `generation_interval_ticks`
- [x] P5-2 在 payload 中增加 `inference_mode` / `model_name`（若由配置决定）
- [x] P5-3 保持 endpoint 不变：仍为 `/generate_accompaniment`
- [x] P5-4 保障旧 server 兼容（新增字段不应破坏旧 server 解析）
- [x] P5-5 增强 HTTP 错误信息透传（保留 status code + response text）

**Exit criteria**
- [x] P5-E1 client 每次请求都显式带上 server 所需运行时参数

### Phase 6 — 配置层与 CLI 对齐（仅参数组织，不改架构）

- [x] P6-1 在 `InferenceConfig` 中增加 `model_name`（默认可为 `lekai` 或由 CLI 指定）
- [x] P6-2 在 `InferenceConfig` 中增加 `lekai_inference_mode`
- [x] P6-3 在 `config_parser.py` 添加/整理 CLI 参数并映射到 `InferenceConfig`
- [x] P6-4 更新 `--help` 文案，明确“HTTP 主路径 + 参数由 request 传入”
- [x] P6-5 保证已有 `http`/`stanley` 使用方式不被破坏

**Exit criteria**
- [x] P6-E1 CLI 产生的配置可驱动 client 发出完整参数请求

### Phase 7 — 测试完善（HTTP 优先）

- [x] P7-1 新增 `test_server_lekai.py`：正常生成路径
- [x] P7-2 新增 `test_server_lekai.py`：`model_name=lekai` 且 interval 非 4 倍数失败
- [x] P7-3 新增 `test_server_lekai.py`：`model_name=lekai` 且 length 非 4 倍数失败
- [x] P7-4 新增 `test_server_lekai.py`：`model_name=stanley`（或其他）不触发该校验
- [x] P7-5 扩展 `test_http_inference_client.py`：新增 payload 字段透传断言
- [x] P7-6 扩展 `test_http_inference_client.py`：服务端错误信息可诊断断言
- [x] P7-7 回归现有 inference 测试，确保无破坏

**Exit criteria**
- [x] P7-E1 HTTP 路径相关单元测试全部通过

### Phase 8 — 文档与运行手册

- [x] P8-1 更新架构文档：强调 HTTP client-server 优先
- [x] P8-2 更新 CLI reference：新增/调整请求参数说明
- [x] P8-3 更新 user guide：Lekai server 启动与 client 调用示例
- [x] P8-4 更新 developer guide：`model_name` 分支与校验策略说明
- [x] P8-5 补充故障排查：4 倍数报错、timings 字段不匹配、history 过长

**Exit criteria**
- [x] P8-E1 文档可让新成员独立完成部署与联调

### Phase 9 — Final Validation & Release Checklist

- [x] P9-1 本地联调：启动 Lekai server + CLI http client
- [x] P9-2 验证完整 melody history 请求在长会话下稳定
- [x] P9-3 验证注入流程（inject/clear/status）
- [x] P9-4 验证仅 Lekai 才触发 4 倍数限制
- [x] P9-5 记录已知限制（性能/显存/长上下文）
- [x] P9-6 整理发布说明与回滚方案

**Exit criteria**
- [x] P9-E1 满足本计划第 6 节 Definition of Done