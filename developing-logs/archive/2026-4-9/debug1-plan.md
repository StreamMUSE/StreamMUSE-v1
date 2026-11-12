# Debug1 Plan

## 需求概述

1. 每次跑一个 client 的时候，调用 clean，这样能够确保我们 server 那边可以一直开着，但是每次一首新歌的时候，server 那边的 history 是干净的。**改进**：clean 不是直接清空，而是先把 server 的 acc_history 和 mel_history 返回给 client，client 保存为两个 log 文件后，server 再清空。
2. 把推理那边改为 latest-only（丢弃过期请求），这样确保不会排队。
3. 修复 composite 模式下 performance.json 和 statistics.csv 没有生成的问题。

---

## Feature 1：Clean with History Return

### 现状分析

**Server 端** (`lekai_http_backend.py`):
- `clear_history()` 方法（line 500-505）直接清空 `_melody_history`、`_accompaniment_history`、`_injection_length_ticks`、`_active_pitches`，返回 `{"success": True, "message": "History cleared"}`。
- 两个 history 的数据格式是 `List[EventPayload]`，其中 `EventPayload = Dict[str, int | str]`，每个 event 包含 pitch、tick、duration、velocity 等字段。

**Server API** (`server_lekai.py`):
- `/clear_history` 端点（line 182-185）调用 `backend.clear_history()`，返回 `ClearHistoryResponse(success, message)`。
- 目前的 response model 只有 `success: bool` 和 `message: str`，不包含 history 数据。

**Client 端** (`http_client.py`):
- `clear_history()` 方法（line 96-99）发 POST 请求到 `/clear_history`，只检查 status code，不解析返回数据。

**CLI** (`cli.py`):
- 目前没有在 session 结束时自动调用 `clear_history` 的逻辑。

### 改动方案

#### 1.1 修改 Server 端 `clear_history()` — 先返回再清空

**文件**: `lekai_http_backend.py`

修改 `clear_history()` 方法：在清空之前，先把当前的 `_melody_history` 和 `_accompaniment_history` 拷贝出来，作为返回值的一部分返回给 client。

```python
def clear_history(self) -> Dict[str, Any]:
    # 先保存当前 history
    mel_history = list(self._melody_history)
    acc_history = list(self._accompaniment_history)
    
    # 再清空
    self._melody_history = []
    self._accompaniment_history = []
    self._injection_length_ticks = 0
    self._active_pitches = set()
    
    return {
        "success": True,
        "message": "History cleared",
        "melody_history": mel_history,
        "accompaniment_history": acc_history,
    }
```

#### 1.2 修改 Server API response model

**文件**: `server_lekai.py`

扩展 `ClearHistoryResponse` 模型，增加 history 数据字段：

```python
class ClearHistoryResponse(BaseModel):
    success: bool
    message: str
    melody_history: List[Dict[str, Any]] = []
    accompaniment_history: List[Dict[str, Any]] = []
```

修改 `/clear_history` 端点，把 backend 返回的 history 数据传递到 response：

```python
@app.post("/clear_history", response_model=ClearHistoryResponse)
async def clear_history() -> ClearHistoryResponse:
    result = backend.clear_history()
    return ClearHistoryResponse(
        success=bool(result["success"]),
        message=str(result["message"]),
        melody_history=result.get("melody_history", []),
        accompaniment_history=result.get("accompaniment_history", []),
    )
```

#### 1.3 修改 Client 端 — 接收并返回 history

**文件**: `http_client.py`

修改 `clear_history()` 方法，解析 server 返回的 history 数据并返回给调用者：

```python
def clear_history(self) -> Dict[str, Any]:
    url = self._endpoint("/clear_history")
    resp = requests.post(url, timeout=float(self._config.timeout_s))
    resp.raise_for_status()
    data = resp.json()
    return {
        "melody_history": data.get("melody_history", []),
        "accompaniment_history": data.get("accompaniment_history", []),
    }
```

同时需要更新 `InferenceEngine` protocol 中 `clear_history` 的签名，让返回值从 `None` 变为 `Dict[str, Any]`。

**文件**: `domain/interfaces/inference.py`

```python
def clear_history(self) -> Dict[str, Any]: ...
```

#### 1.4 CLI 端 — session 结束时调用 clean 并保存 log

**文件**: `cli.py`

在 `cleanup()` 函数中，添加调用 `inference_engine.clear_history()` 的逻辑，并把返回的 history 保存到 session 目录下：

```python
def cleanup() -> None:
    output_sink.close()
    
    # 调用 server clean，获取 history 并保存
    if session_manager:
        try:
            history_data = inference_engine.clear_history()
            session_dir = session_manager.get_session_dir()
            
            mel_path = session_dir / "melody_history.json"
            acc_path = session_dir / "accompaniment_history.json"
            
            with open(mel_path, "w") as f:
                json.dump(history_data.get("melody_history", []), f, indent=2)
            with open(acc_path, "w") as f:
                json.dump(history_data.get("accompaniment_history", []), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save history logs: {e}")
    
    if session_manager and isinstance(output_sink, SessionLoggerOutputSink):
        output_sink.save_metrics(session_config)
        session_manager.save_summary(...)
```

保存的文件：
- `melody_history.json`：server 端累积的所有 melody event 记录
- `accompaniment_history.json`：server 端累积的所有生成的 accompaniment event 记录

---

## Feature 2：Latest-Only Inference（丢弃过期请求）

### 现状分析

**当前推理队列机制** (`real_time_music_service.py`):

1. `_tick_loop`（line 205-213）每隔 `generation_interval_ticks` 往 `_inference_request_queue` 放一个推理请求 `(generation_start_tick, melody_events)`。
2. `_inference_worker`（line 247-310）从 `_inference_request_queue` 阻塞取请求，调用 `generate_accompaniment()`，把结果放到 `_inference_response_queue`。
3. `_tick_loop` 在每个 tick 轮询 `_inference_response_queue` 取结果。

**问题**：当推理速度慢于请求产生速度时，`_inference_request_queue` 会积压。inference_worker 会按 FIFO 顺序依次处理每个请求，导致：
- 越来越旧的请求被处理，生成的 accompaniment 已经过时
- 延迟不断累积，实时性完全丧失

### 改动方案

#### 2.1 inference_worker 取请求时丢弃旧请求

**文件**: `real_time_music_service.py`

修改 `_inference_worker` 的取请求逻辑：从队列中取出请求后，继续检查队列中是否还有更新的请求。如果有，丢弃当前请求，取最新的那个。

```python
def _inference_worker(self) -> None:
    while self._running:
        try:
            generation_start_tick, melody_events = self._inference_request_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Latest-only: 丢弃队列中所有旧请求，只保留最新的
        while not self._inference_request_queue.empty():
            try:
                generation_start_tick, melody_events = self._inference_request_queue.get_nowait()
            except queue.Empty:
                break

        if not self._running:
            break

        # ... 继续处理最新的请求（后续代码不变）
```

**核心逻辑**：每次准备推理前，用 `get_nowait()` 把队列里剩余的请求全部取出来，只保留最后一个（最新的）。这样无论积压了多少请求，只有最新的那个会被实际推理。

#### 2.2 需要注意的问题

**melody_events 累积问题**：当前 `_tick_loop` 每次只发送上次请求后新增的 events（Bug #4 fix，`_last_sent_index` 机制）。如果中间的请求被丢弃，那些请求携带的 melody_events 也会被丢弃。

但这**不是问题**，因为：
- Server 端的 `generate_accompaniment()` 接收的是增量 melody events，server 自己维护完整的 `_melody_history`
- 每次请求发送的 `melody_events` 会被 server append 到 `_melody_history` 中
- 如果中间请求被丢弃，这些 melody events 确实不会被发送到 server

**所以需要合并被丢弃请求的 melody_events**：

```python
def _inference_worker(self) -> None:
    while self._running:
        try:
            generation_start_tick, melody_events = self._inference_request_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # Latest-only: 丢弃旧请求，但合并所有 melody_events
        while not self._inference_request_queue.empty():
            try:
                newer_tick, newer_events = self._inference_request_queue.get_nowait()
                # 合并 melody events（保证不丢失中间的 melody 数据）
                melody_events = melody_events + newer_events
                generation_start_tick = newer_tick  # 使用最新的 tick
            except queue.Empty:
                break

        if not self._running:
            break

        # ... 继续用合并后的 melody_events 和最新的 generation_start_tick 推理
```

这样做的好处：
- `generation_start_tick` 使用最新的（最准确反映当前播放进度）
- `melody_events` 包含了所有被跳过请求的 melody 数据（不丢失信息）
- Server 收到的还是完整的增量 melody 数据

---

## Feature 3：修复 Composite 模式下缺失 performance.json 和 statistics.csv

### 现状分析（Bug Root Cause）

**`cli.py` line 70-79** 的 `cleanup()` 函数：

```python
def cleanup() -> None:
    output_sink.close()
    if session_manager and isinstance(output_sink, SessionLoggerOutputSink):
        output_sink.save_metrics(session_config)
        session_manager.save_summary(...)
```

**问题所在**：`isinstance(output_sink, SessionLoggerOutputSink)` 检查。

在 composite 模式下（`output_factory.py` line 105-117），`output_sink` 的类型是 `CompositeOutputSink`，而不是 `SessionLoggerOutputSink`。所以 `isinstance` 检查失败，`save_metrics()` **永远不会被调用**。

虽然 `CompositeOutputSink.close()` 会正确地调用内部 `SessionLoggerOutputSink.close()`，从而生成 `events.jsonl` 和 `inferences.json`（因为这些是在 `JsonLoggerOutputSink.close()` 里写入的），但 `performance.json` 和 `statistics.csv` 是由 `save_metrics()` 方法单独生成的，不在 `close()` 的调用链中。

**总结**：
- `events.jsonl` ✅ — 在 `output_event()` 时实时写入
- `inferences.json` ✅ — 在 `close()` → `save_inferences()` 写入
- `performance.json` ❌ — 需要 `save_metrics()` 调用，但 isinstance 检查阻止了
- `statistics.csv` ❌ — 同上

### 改动方案

#### 3.1 方案：让 `save_metrics` 在 `close()` 中自动调用

最简洁的修复方式：把 `save_metrics()` 的调用挪到 `SessionLoggerOutputSink.close()` 中，而不是依赖外部 CLI 代码来调用。

**文件**: `session_logger.py`

```python
def close(self, session_config: Optional[Dict[str, Any]] = None) -> None:
    if self.json_sink:
        if session_config:
            self.json_sink.save_metrics(session_config)
        self.json_sink.close()
    if self.midi_sink:
        self.midi_sink.close()
```

但这种方式有个问题：`close()` 的调用者（`CompositeOutputSink`）不知道要传 `session_config` 参数，因为 `OutputSink` protocol 的 `close()` 没有参数。

#### 3.2 更好的方案：在 SessionLoggerOutputSink 内部缓存 session_config

让 `SessionLoggerOutputSink` 在创建时或通过 `output_config()` 方法接收 `session_config`，在 `close()` 时自动调用 `save_metrics()`。

**文件**: `session_logger.py`

在 `output_config()` 方法中缓存 config：

```python
class SessionLoggerOutputSink:
    def __init__(self, ...):
        ...
        self._session_config: Optional[Dict[str, Any]] = None

    def output_config(self, config: dict) -> None:
        self._session_config = config
        # 原有逻辑...

    def close(self) -> None:
        # 自动保存 metrics
        if self.json_sink and self._session_config:
            self.json_sink.save_metrics(self._session_config)
        
        if self.midi_sink:
            self.midi_sink.close()
        if self.json_sink:
            self.json_sink.close()
```

**文件**: `cli.py`

现在 CLI 的 `output_config()` 已经会被调用（service 启动时会调用 `output_sink.output_config()`）。确认 `RealTimeMusicService` 或 CLI 在启动时会调用 `output_sink.output_config(session_config)`。

查看当前代码：CLI 在 line 47 调用了 `session_manager.save_config(session_config)`，但没有调用 `output_sink.output_config(session_config)`。需要添加：

```python
# 在 service.start() 之前
output_sink.output_config(session_config)
```

同时简化 `cleanup()`：

```python
def cleanup() -> None:
    output_sink.close()  # SessionLoggerOutputSink.close() 现在自动 save_metrics
    if session_manager:
        session_manager.save_summary(
            {
                "status": "completed",
                "session_id": session_manager.get_session_id(),
            }
        )
```

这样不再需要 `isinstance` 检查。`CompositeOutputSink.close()` 会遍历调用每个子 sink 的 `close()`，其中 `SessionLoggerOutputSink.close()` 会自动保存 metrics。

---

## Feature 4：修复首轮推理 `tick=0` 触发的 Zero-Prompt 崩溃

### 现状分析（触发链路）

这个 bug 的核心不是模型随机失败，而是首轮请求在 `generation_start_tick=0` 时，prompt 时间窗长度为 0，导致后续张量长度与预期不一致。

触发路径如下：

1. `_generate_with_model()` 里计算 `prompt_start = max(0, generation_start_tick - generation_length_frames)`。
   - 当 `generation_start_tick=0` 时，`prompt_start=0`，`end_tick=0`。
2. `events_to_pianoroll(start_tick=0, end_tick=0)` 得到 `T=end-start=0`，返回形状 `(2, 88, 0)`。
3. `num_measures = melody_pianoroll.shape[2] // 16 = 0`，因此 `part0_beats=[]`。
4. `PianoLLaMAAdapter.generate_from_beats()` 在 `part0_beats` 为空时直接 `break`，返回空 `part1_beats`。
5. `beats_to_pianoroll([])` 返回 `(2, 88, 0)`。
6. backend 仍然期望 `(2, 88, num_beats*4)`，最终抛出：
   `RuntimeError: Pianoroll shape mismatch: expected (2, 88, 8), got (2, 88, 0)`。

这会直接导致 `/generate_accompaniment` 返回 500。

### 改动方案

#### 4.1 Backend 入口防御：零窗口请求不进入易崩路径

**文件**: `infrastructure/inference/lekai_http_backend.py`

在 `_generate_with_model()` 中加入显式守卫：如果 `generation_start_tick <= 0` 或 `prompt_end <= prompt_start`，不走当前 model prompt 编码路径，改为可控 fallback（推荐调用 `_generate_rule_based(...)` 并打印 warning）。

建议形式：

```python
prompt_start = max(0, generation_start_tick - generation_length_frames)
prompt_end = generation_start_tick
if prompt_end <= prompt_start:
    print("[LekaiHttpBackend] zero prompt window, fallback to rule-based")
    return self._generate_rule_based(
        generation_start_tick=generation_start_tick,
        generation_interval_ticks=generation_interval_ticks,
        generation_length_frames=generation_length_frames,
    )
```

目标：首轮请求不再触发 500。

#### 4.2 Adapter 逻辑修复：`part0_beats` 为空时仍允许生成

**文件**: `infrastructure/inference/lekai_model/inference_adapter.py`

当前 `generate_from_beats()` 在 `position == 0` 且 `part0_idx >= len(part0_beats)` 时直接 `break`，导致返回空输出。改为切换到 `position=1`，基于 BOS/time/bpm 前缀继续生成 `part1`。

建议修改点（伪代码）：

```python
if position == 0:
    if part0_idx < len(part0_beats):
        ...  # 原逻辑
    else:
        # 不再 break，转入 part1 生成
        position = 1
        continue
```

目标：即使 prompt 为空，也能产出长度正确的 beat token 序列，不返回空列表。

#### 4.3 Shape mismatch 降级策略：避免对用户抛 500

**文件**: `infrastructure/inference/lekai_http_backend.py`

在 shape 校验阶段将“直接 raise”改为“可观测降级”：

1. 记录详细告警（expected/got/generation_start_tick/generation_length_frames）。
2. 自动回退到 `_generate_rule_based(...)` 返回可用结果。
3. 保留调试信息，便于后续继续优化模型路径。

建议：仅对 `time axis == 0` 或可恢复场景降级；其他严重不一致可保留异常以避免静默错误。

#### 4.4 测试补齐（防回归）

**文件**: `tests/unit/infrastructure/inference/test_lekai_http_backend.py`

新增用例：

1. `generation_start_tick=0` 且 model path 可用时，`generate()` 不抛异常（HTTP 语义上应可返回 200）。
2. 返回结果应为 list（允许为空或非空，但不能触发 RuntimeError）。

**文件**: `tests/unit/infrastructure/inference/lekai_model/test_inference_adapter.py`

新增用例：

1. `part0_beats=[]` 且 `num_beats_to_generate>0` 时，`generate_from_beats()` 不应直接返回空（至少应产生 beat 结束标记序列）。

#### 4.5 运行时验证方法（手工）

1. 启动 `server_lekai`。
2. 发送一个 `generation_start_tick=0` 的最小请求到 `/generate_accompaniment`。
3. 验证：
   - 不返回 500；
   - 日志中出现可追踪的 fallback 或空窗口告警；
   - 服务进程继续稳定运行。

---

### 验收标准

1. 首轮推理（`generation_start_tick=0`）不会再触发 `Pianoroll shape mismatch` 导致的 500。
2. `/generate_accompaniment` 在该场景下返回 200。
3. 新增单元测试覆盖 zero-prompt 路径并通过。
4. 回归测试中不影响已有非零窗口推理行为。

---

## 涉及的文件总结

| 文件 | Feature 1 | Feature 2 | Feature 3 | Feature 4 |
|------|-----------|-----------|-----------|-----------|
| `infrastructure/inference/lekai_http_backend.py` | ✅ 修改 `clear_history()` 返回 history | | | ✅ 增加 zero-prompt 守卫与 shape mismatch 降级 |
| `infrastructure/inference/server_lekai.py` | ✅ 修改 response model 和端点 | | |
| `infrastructure/inference/http_client.py` | ✅ 修改 `clear_history()` 解析返回值 | | |
| `domain/interfaces/inference.py` | ✅ 修改 `clear_history` 签名 | | |
| `presentation/cli/cli.py` | ✅ cleanup 中调用 clean 并保存 | | ✅ 简化 cleanup，添加 output_config 调用 |
| `application/services/real_time_music_service.py` | | ✅ 修改 inference_worker 丢弃旧请求 | |
| `infrastructure/output/session_logger.py` | | | ✅ close() 中自动 save_metrics |
| `infrastructure/inference/lekai_model/inference_adapter.py` | | | | ✅ 修复 `part0_beats=[]` 时提前退出 |
| `tests/unit/infrastructure/inference/test_lekai_http_backend.py` | | | | ✅ 新增 `generation_start_tick=0` 防回归测试 |
| `tests/unit/infrastructure/inference/lekai_model/test_inference_adapter.py` | | | | ✅ 新增空 prompt 生成路径测试 |

---

## Detailed Todo List（Phases + Tasks）

### Phase 0 — Baseline 冻结与改动边界确认

- [ ] P0-1 记录当前 `/generate_accompaniment` 在 `generation_start_tick=0` 时的复现日志（作为回归基线）。
- [ ] P0-2 确认 `InferenceEngine.clear_history` 改签名的影响面（`HttpInferenceClient`、`StanleyInferenceEngine`、测试 mock）。
- [ ] P0-3 确认 `composite` 模式下当前产物清单（缺失 `performance.json`、`statistics.csv`）并保存样本目录。
- [ ] P0-4 明确本次改动优先级：先止住 500（Feature 4），再处理排队（Feature 2），最后收敛日志一致性（Feature 1+3）。

**Exit Criteria**
- [ ] P0-E1 基线复现脚本与日志样本可重复执行。
- [ ] P0-E2 所有受影响文件列表和接口变更点已冻结。

### Phase 1 — Feature 4（Zero-Prompt 崩溃修复）

- [ ] P1-1 在 `lekai_http_backend.py::_generate_with_model()` 添加 zero-window 守卫（`prompt_end <= prompt_start`）。
- [ ] P1-2 zero-window 分支改为可控 fallback（调用 `_generate_rule_based(...)`），避免直接抛 500。
- [ ] P1-3 在 shape mismatch 检查处增加可观测日志（expected/got/start_tick/gen_len）。
- [ ] P1-4 仅对可恢复场景（time axis = 0）做降级回退；非可恢复场景保留异常。
- [ ] P1-5 调整 `inference_adapter.py::generate_from_beats()`：`part0_beats` 为空时不提前 break，允许继续 part1 生成。
- [ ] P1-6 新增/更新单测：`generation_start_tick=0` 不崩溃；空 prompt 下 adapter 不直接返回空序列。

**Exit Criteria**
- [ ] P1-E1 `generation_start_tick=0` 请求返回 200，不再触发 `Pianoroll shape mismatch` 500。
- [ ] P1-E2 新增单测通过且不会影响非零窗口路径。

### Phase 2 — Feature 2（Latest-Only 推理队列）

- [ ] P2-1 在 `_inference_worker()` 中实现 latest-only 取队列逻辑（drain 旧请求，仅保留最新 tick）。
- [ ] P2-2 对被丢弃请求的 `melody_events` 做合并，确保 server 端 melody history 不丢数据。
- [ ] P2-3 确认 stop 流程下 dummy item 与 latest-only 逻辑兼容，不引入死循环或异常退出。
- [ ] P2-4 增加调试日志：每轮推理被合并/丢弃请求数量，便于观察排队缓解效果。
- [ ] P2-5 新增单测：队列积压时仅最新请求推进推理，但 melody events 累积完整。

**Exit Criteria**
- [ ] P2-E1 压测下 `_inference_request_queue` 不再持续增长。
- [ ] P2-E2 生成起点滞后显著下降（请求时序更贴近当前 tick）。

### Phase 3 — Feature 1（Clean 返回历史并落盘）

- [ ] P3-1 修改 `lekai_http_backend.py::clear_history()`：先拷贝 `melody/accompaniment` history，再清空状态。
- [ ] P3-2 扩展 `server_lekai.py::ClearHistoryResponse`，增加 `melody_history`、`accompaniment_history` 字段。
- [ ] P3-3 修改 `/clear_history` 端点返回结构，透传 history 数据。
- [ ] P3-4 修改 `http_client.py::clear_history()`：解析响应并返回 history 字典。
- [ ] P3-5 更新 `domain/interfaces/inference.py` 的 `clear_history` 协议返回类型，并处理实现类兼容。
- [ ] P3-6 在 `cli.py::cleanup()` 中调用 `inference_engine.clear_history()`，将 history 落盘为 `melody_history.json` / `accompaniment_history.json`。
- [ ] P3-7 处理异常降级：若 clear/history 保存失败，仅告警，不影响主退出流程。

**Exit Criteria**
- [ ] P3-E1 每次 session 结束后都能在 session 目录看到两份 history 文件。
- [ ] P3-E2 server 端 history 在 clean 后确实清空，下一首歌从干净上下文开始。

### Phase 4 — Feature 3（Composite 模式指标文件修复）

- [ ] P4-1 在 `SessionLoggerOutputSink` 增加 `_session_config` 缓存字段。
- [ ] P4-2 在 `output_config()` 中接收并缓存 session 配置。
- [ ] P4-3 在 `close()` 中自动触发 `json_sink.save_metrics(...)`（当配置存在时）。
- [ ] P4-4 在 `cli.py` 中补充 `output_sink.output_config(session_config)` 调用（`service.start()` 前）。
- [ ] P4-5 简化 `cleanup()`，去掉 `isinstance(output_sink, SessionLoggerOutputSink)` 分支依赖。
- [ ] P4-6 新增单测：`composite` 关闭后自动产生 `performance.json` 与 `statistics.csv`。

**Exit Criteria**
- [ ] P4-E1 `composite` 模式目录稳定包含 `events.jsonl`、`inferences.json`、`performance.json`、`statistics.csv`、`combined.mid`。
- [ ] P4-E2 `session` 模式行为无回归。

### Phase 5 — 联调验证与文档同步

- [ ] P5-1 运行单元测试子集：inference、service、output 相关测试。
- [ ] P5-2 手工联调场景 A：首轮 `tick=0` 请求（验证无 500）。
- [ ] P5-3 手工联调场景 B：长会话 + latest-only（验证 queue 不积压、伴奏回填率改善）。
- [ ] P5-4 手工联调场景 C：`composite` 输出（验证指标文件完整）。
- [ ] P5-5 更新用户/开发文档：clear_history 返回结构、latest-only 行为、composite 产物变化。
- [ ] P5-6 记录前后对比报告：延迟分布、model events 数量、MIDI 伴奏轨音符数。

**Exit Criteria**
- [ ] P5-E1 四个 Feature 的验收标准全部通过。
- [ ] P5-E2 形成可复现的 baseline vs final 对比结论。
