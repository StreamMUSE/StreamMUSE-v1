# 交接：合并计划第 1 到 3 阶段的实施方案（2026-11-11）

对应计划：[2026-11-11 合并计划](../plans/2026-11-11-rap-inference-optimization-and-carryover-plan.md)。这份文档写给接手实施的人，记录已经核实的事实、定下来的设计和还没核实的点。

## 0. 现在的状态

- **代码一行没改。** 只读了代码、跑了基线测试、写了这份文档。
- 分支 `feature/rap_optimization`，HEAD `d67bb48b`，和 origin 一致。
- 基线测试（`uv run python -m pytest tests/ -q --tb=no`）：3400 通过，5 跳过，5 失败。5 个失败是已知的（FastPitch/timing，缺语料），和这次工作无关。
- **H200 连不上**（`ssh mbzuai-masdar` 超时，11-11 起一直如此）。所以只能先写代码和单元测试，验收等连接恢复。
- 库版本：websockets 16.0，httpx 0.28.1，uvicorn 0.35.0。
- 计划里说基线 trace 在 `output/inference_opt_20261111/baseline-cycle/`，但这台机器上没有这个目录。

## 1. 先要知道的事实

### 1.1 隧道的一次往返约 0.5 s

基线里"复用连接的 GET"中位数 507 ms，新建连接再多约 0.55 s。所以 WAIT 主要由往返次数决定，服务器计算（写词 0.15 s，合成加对齐 0.47 s）只占小头。

现在 Cyclic 一个回合在用户敲完之后依次做：

1. `stream.get_capabilities()`：独立的 `httpx.get`，新建连接，约 1.05 s。
2. `stream.open()`：WebSocket 握手约 1.07 s，然后发 `start`。
3. 等 `lyric`：一次往返加写词。
4. `source.fetch()`：新建的 `httpx.Client`，又多 0.55 s。

合成和"歌词到 Mac、Mac 发取音频请求"这次往返是重叠的，所以服务器把音频主动推过来省不了时间，不用做。

第 1 阶段做完后的估算：`start` 单程 0.25 + 写词 0.15 + 歌词回程和取音频去程 0.5 + 音频回程 0.25 + Mac 准备 0.08，约 1.25 s，加上音频包的传输时间，预计 1.3 到 1.6 s。

### 1.2 相关代码的行为（都读过源码）

**`RealInferenceBackend`**（`infrastructure/rap/inference_backend.py`）

- `submit` 只入队；`_run` 在线程池里按"连续的非静音 bar"切段，每段调一次 `_run_session`。
- `_run_session` 每次都：`self._stream_factory()` → `get_capabilities()` → `self._source_factory()` → `stream.open(session_id, payload, handler)`，结束时发 `stop` 并关闭 stream 和 source。
- `start` 里 `lyric_lead` 和 `audio_lead` 都等于这一段的 bar 数；`history` 是 `request.history[-4:]`，bar 号排在 `start` 之前。
- 取音频用 `accept_older=False`，3 个线程并行。

**`LyricStreamConnection`**（`infrastructure/rap/lyric_stream_client.py`）

- `open()` 调 `_connect()`，而 `_connect()` 把"握手"和"发 `start`"合在一起做。断线重连也走 `_connect()`，会重发 `start`。
- **会话 ID 在 WebSocket 的 URL 里**（`/v1/rap/sessions/{session_id}`），所以提前建连时就要定好会话 ID。

**服务器**（`presentation/rap_lyric_stream_server.py`）

- `websocket.accept()` 之后一直等消息，没有超时；收到 `start` 才建会话。所以提前建连不用改服务器。
- 一条连接只对应一个会话，`stop` 后循环结束。`_MAX_SESSIONS = 8`。
- 可选能力的校验模式：`start` 里用了有界会话字段但没声明 `bounded_turns_v1` 就报错。新能力照这个写。

**续写**（`application/rap/lyric_session.py`、`bar_audio.py`）

- `_render_loop` 里 `previous = self._bars.get(bar - 1).latest`。新会话的第一个 bar，前一个 bar 只有 `history` 给的 `committed_text`，`latest` 是 `None`，所以不续写，`diagnostics.continuation.reason` 是 `no_previous_line`。
- `MossBarAudioRenderer._sources` 是所有会话共用的缓存，键是 `(lyric_id, tempo_bpm)`，保留最近 64 条，值是 `SpeechPrefix(text, source_wav)`。`_prefix_for` 只用到 `previous.lyric_id`。

**`InferenceService`**（`application/rap/inference/service.py`）

- `_begin_user()` 在音频回调里、持有锁时被调用。从这里触发的任何后端调用都只能入队，不能做网络操作。
- Steering 的请求节奏：`_extend_steering` 在"当前版本已请求的范围只剩不到一半"时发下一个请求。默认 `model_beats=8`（2 个 bar）时，每 2 个 bar 发一个 2 bar 的请求，发出时正在播的是请求起点前 2 个 bar。一般地，请求 `[s, s+m)` 发出时正在播 bar `s - floor(m/2) - 1`。
- `steer()`：版本号加一，`_cancel_pending()` 取消所有未结束的请求，然后从"播放位置和已锁定位置之后的第一个 bar 边界"发新请求。这时正在播的是新请求起点的前一个 bar。
- `accept_span` 只接受：请求行还活着，且 `span.input_version` 等于当前版本。旧版本的音频只有在 steer 之前已经进了 `SpanBuffer` 才能作为 stale 播出。
- `_account_playback` 在播放越过某个请求的终点时调 `backend.cancel(request_id)`。**所以后端单看 `cancel` 分不清"播完了"和"被 steer 取消"。**
- `epoch` 在每次 Start 和 Reset 时加一；每次 Start 后 tick 从 0 重新计。`_history`（最多 16 句已播出的歌词）跨 Stop/Start 保留，Reset 清空。

## 2. 第 1 阶段：Cyclic 回合之间保持连接

### 改动

1. **`LyricStreamConnection`**：加 `connect(session_id)`，只握手、不发 `start`。`open()` 发现已有同一个 `session_id` 的连接就只发 `start`，否则照旧。重连路径（`_connect()` 重发 `start`）保持不变。
2. **`HttpBarAudioSource`**：
   - 默认的 `httpx.Client` 加 `limits=httpx.Limits(keepalive_expiry=...)`，httpx 默认 5 s 就丢弃空闲连接。
   - 加 `warm(n)`：并发发 n 个便宜的 GET（`/v1/rap/capabilities`），把连接池里的连接建好。bar 0 和 bar 1 会并行取，所以预热 2 条。
3. **render server**：uvicorn 的 `timeout_keep_alive` 默认也是 5 s，回合间隔远大于它。加一个启动参数把它调大（建议 75 s）。**还没看 `rap_render_server.py` 是怎么启动 uvicorn 的，动手前先确认。** 旧服务器不改也不会出错：`fetch` 已有"连接失效就重试一次"的逻辑，只是享受不到复用。
4. **`RealInferenceBackend`**：
   - 能力只查一次，缓存起来；会话出错时清掉缓存。
   - `source` 改成整个后端共用一个，`close()` 时才关；网络错误时丢弃重建。
   - 加 `prepare()`：只入队。后台线程做三件事：查能力（没缓存时）、`source.warm()`、建一条备用的 WebSocket（握手完成、未发 `start`），记下它的会话 ID 和建立时间。
   - `_run_session` 优先用备用连接：年龄没超过上限（建议 30 s）就直接发 `start`；发送失败或太旧，就关掉它、新建连接重试一次。
   - `close()`、出错时关掉备用连接。
   - 测试里的假 stream 没有 `connect`，用 `getattr` 探测，没有就走老路径。
5. **`InferenceService`**：`_begin_user()` 里调一次 `getattr(self.backend, "prepare", None)`。只动 cyclic 路径。

### 和计划的出入（要记进实施记录）

- 计划写"用户没敲出节奏时，提前建好的连接直接关掉"。更好的做法是留着给下一回合用，靠年龄上限淘汰。服务器上一条只握手没 `start` 的连接不占会话名额。
- 计划没提两边的 keep-alive 都是 5 s，这是连接复用能否生效的前提。

### 测试

- 现有 `tests/unit/infrastructure/rap/test_inference_backend.py` 的 `Harness` 每次会话都新建并关闭 source，有几个测试断言了这一点（如 `test_first_bar_emits_before_remaining_turn_and_closes_session_on_completion`、`test_cleanup_closes_socket_even_when_http_source_close_raises`），要跟着改。
- 新增：两个回合只查一次能力、共用一个 source；备用连接被使用；备用连接失效后重建；`close()` 关掉备用连接；`prepare()` 不阻塞。
- 建议加一个真实套接字的集成测试：在线程里起 uvicorn，`mount_lyric_stream_routes` 配假的 writer 和 renderer，用真的 `LyricStreamConnection` 和 `HttpBarAudioSource` 连。这样提前建连、keep-alive 这些线上行为不靠 H200 也能验证。

## 3. 第 2 阶段：续写跨过请求边界

### 改动

1. **协议**（`domain/rap/lyric_stream.py`）：新能力 `continuation_handoff_v1`，加进 `LYRIC_STREAM_CAPABILITIES`。`start` 可带 `continue_from: {"lyric_id": "<64 位十六进制>"}`。
2. **服务器**：
   - `LyricStreamService.start`：带了 `continue_from` 但没声明能力就报错。
   - `LyricStreamSession` 加参数 `continue_from`。`_render_loop` 渲染 `start_bar` 且 `previous` 为 `None` 时，把这个 ID 传给渲染器。
   - `BarAudioRenderer.render` 加可选关键字 `prefix_lyric_id`。**只在确实有交接 ID 时才传**，这样测试里签名不带它的假渲染器不受影响。
   - `MossBarAudioRenderer._prefix_for`：`previous` 为空但有 `prefix_lyric_id` 时查 `self._sources[(prefix_lyric_id, tempo_bpm)]`。命中就续写，`diagnostics.continuation` 里标 `handoff: true`；没命中写 `reason: "handoff_not_cached"`。
3. **客户端**（`RealInferenceBackend`）：
   - 维护一个有上限的映射：已交付歌词的文本 → `lyric_id`。
   - 开会话时用 `request.history[-1]`（服务层记的最后一句实际播出的歌词）查 ID，查到且服务器能力里有 `continuation_handoff_v1` 才带 `continue_from`。
   - 这样 `InferenceRequest` 和 `AudioSpan` 的契约不用改。

### 注意

- 缓存键带 `tempo_bpm`，两个回合之间换了 BPM 就接不上，属于预期。
- 续写接上后声音问题是否消失，只有 H200 上听了才知道。计划里已写：如果没消失，按 Andrew 11-04 笔记的隔离方法查。

### 测试

- 服务器：带交接 ID 的会话，第一个 bar 的渲染收到 `prefix_lyric_id`；不带时调用方式和现在完全一样；没声明能力时报错。
- 渲染器：命中、没命中两种 `diagnostics`。
- 客户端：第二个请求的 `start` 带上了第一个请求最后一句的 ID；旧服务器（没这个能力）不带。

## 4. 第 3 阶段：Steering 改用长会话

### 设计

新模块 `infrastructure/rap/inference_steering.py`，类 `SteeringSessionBackend`，对外接口和现在一样（`submit` / `cancel` / `close` / `capabilities`）。`RealInferenceBackend.submit` 按 `request.config["mode"] == "steering"` 转给它；`cancel` 两边都问。`InferenceService` 和页面不改。

- **会话：** 每个 `(session_id, epoch)` 开一个长会话，不带 `turn_id` 和 `end_bar_exclusive`。bar 号直接用 `request.start_tick // 16`。新 epoch 来了就关旧会话、开新会话。
- **同一版本的后续请求：** 不开新会话，只登记"这个请求覆盖哪些 bar"。
- **版本变了（steer）：** 发一条 `update`，`from_bar` 等于新请求的起始 bar，`edit_seq` 递增。**`PlanEdit` 的字段格式还没核对，动手前看 `domain/rap/lyric_stream.py`。**
- **交付：** 每收到一条 `lyric` 就取音频、准备，然后归到"当前版本、覆盖这个 bar、还活着"的那个请求名下，用它的 `on_span` 交付。服务器提前渲染出来、但还没有请求覆盖的 bar 先留着，等请求来了立刻交付。
- **取消：** 请求被取消且还没交付完，就发 `canceled` 事件（和现在的后端一致），之后不再往它名下交付。
- **关闭：** 没有任何活着的请求超过一小段宽限时间（建议 0.5 s）就发 `stop` 关会话。steer 时"全部取消后立刻发新请求"发生在同一次持锁内，不会误关。
- **旧版本音频：** 不需要服务器的 `older_version`。长会话提前渲染，steer 之前旧版本的 span 已经在 `SpanBuffer` 里了，服务层自己会把它当 stale 播。取音频仍用 `accept_older=False`。
- **第 2 阶段的交接在这里也用：** 新 epoch 的会话带上 `continue_from`。

### 最需要小心的地方：怎么向服务器报播放头

服务器靠 `committed` 推进播放头 P，只为 P+1 到 P+提前量 的 bar 写词和渲染；`update` 的生效位置是 `max(from_bar, P+1)`。所以：

- **P 报得比实际靠前，会推迟 steer 生效。** 必须保证 P 不超过实际正在播的 bar。
- P 报得靠后是安全的，但提前量要相应加大，否则窗口盖不住已请求的范围。

后端拿不到逐 bar 的播放位置（见 1.2：`cancel` 有歧义）。两个方案：

| 方案 | 做法 | 代价 |
|---|---|---|
| A（建议） | 只在请求到达时报。开始和 steer 的请求：P = 起始 bar − 1（准确）。延长请求 `[s, s+m)`：P = `s − floor(m/2) − 1`。提前量相应加大：默认 2 bar 的请求用 `audio_lead=3`、`lyric_lead=4` | 依赖服务层"剩余不到一半时延长"这条规则（接手文档里写明了）。提前量比 v1.0 的 3 和 2 各多 1，steer 后要重写的 bar 多一个，但按 bar 号从小到大处理，不拖慢第一个新 bar |
| B | 组装时给后端注入一个读播放位置的函数，后端起一个轮询线程逐 bar 报 | 和 v1.0 的提前量完全一致，但要读服务层的私有状态或给它加一个访问器，和"服务层不改"冲突 |

`committed` 的 `lyric_id` 和 `text` 填这个 bar 最后一次被服务层接受的那条记录。不带 `turn_id` 的会话对它们没有严格校验，`text` 会进写词的历史。

### 上线方式

加一个命令行参数（建议 `--inference-steering-adapter session|bounded`），**默认先保持 `bounded`**，H200 上验收通过后再把默认改成 `session`。新路径没经过真实服务器验证之前，不应该让别人拉代码就用上。

### 测试

- 两个同版本请求只开一个会话；第二个请求触发 `committed`。
- steer：发出 `update`，`from_bar` 正确；新歌词的 span 带新请求的 ID 和新版本号；旧请求收到 `canceled`。
- 提前渲染的 bar 在请求到达后立即交付。
- 全部取消后超过宽限时间，会话被关闭；新 epoch 开新会话。
- 现有 steering 的服务层测试全部照跑。

## 5. 验证

**不需要 H200**

- 上面各阶段的单元测试，加第 2 节提到的真实套接字集成测试。
- 可选：用 `LyricStreamService` 配假 writer 和假渲染器起一个本地替身服务器，再加一个带固定延迟的 TCP 转发，模拟隧道的 0.25 s 单程延迟，量第 1 阶段前后的 WAIT。这只能验证连接开销的变化，不能替代真实验收。

**需要 H200**（按计划的验收标准）

- 第 0 阶段剩下的：steering 基线（60 和 90 BPM 各 6 次修改）、逐 bar 的续写记录。第 3 阶段的验收要和这个基线比，最好在切换适配器之前测。
- 第 1 阶段：同条件 5 个回合，WAIT 中位数低于 2 s，缺失为 0。
- 第 2 阶段：除整场第一个 bar 外续写比例 95% 以上；回合边界的说话人相似度不低于回合内部；请 Andrew 再听。
- 第 3 阶段：前半 bar 的修改在下一个 bar 播出，后半的最多晚一个 bar，缺失为 0。
- 每次测量前后各记 5 次隧道往返；记录两张卡的占用。

**还要写的工具：** `scripts/analyze_inference_trace.py`（计划第 0 阶段），输入页面导出的 trace，输出每个请求各阶段的耗时。

## 6. 约定

- 每个阶段一个提交，提交信息末尾带 `Co-Authored-By` 行（按当前会话的归属提示）。
- 同步方式：push 到 `origin/feature/rap_optimization`，H200 上 pull。
- H200 是共享机器：用 GPU 前先看 `nvidia-smi`；只按记录的 PID 或进程组停自己的进程，不按名字杀。
- 大文件和缓存放 `/Volumes/ZBW-SSD1/cache`；Python 环境用 uv。
- 不改旧页面和旧模式的行为；改到共用代码要保住旧行为并补回归测试。
- 每个阶段做完，在计划的"实施记录"里追加日期、提交、实测数字和偏离之处。
- 这台 Mac 没有设置 git 邮箱，提交用的是 `bowenzheng@Bowens-Mac-Studio.local`，GitHub 不会算到账号上。用哪个邮箱还没定。
