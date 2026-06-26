# Two-Stage (Prompt+Continuation) 一致性测试计划（2026-06-26）

分支：`integrate-prompt-continuation-switch`。目标和之前的单阶段 final consistency test 一样——**offline fix-output 与 realtime fix-output 逐拍一致**——但对象换成 **two-stage（prompt + continuation）** 路径。

---

## 前置背景：单阶段 final consistency test（实现 two-stage 前必读）

> 这一节是给没有上下文的实现者的。本计划大量复用单阶段那套已建好、已跑通的测试基建。先理解它，再看后面 two-stage 的增量改动。

### 它是什么

系统的"金标准"端到端回归测试：在**确定性（贪心）采样**下，真实的实时链路（`streammuse-cli --input-mode midi_file` → `RealTimeMusicService` 3 线程 → HTTP → Lekai server）与离线一次性生成（`scripts/run_lekai_offline.py`）产出的**伴奏应逐拍完全一致**。它守护"实时和离线是同一个系统"这个不变量。已实测 song 4 × tempo{15,120} 全绿（6m25s）。

完整设计见 `developing-logs/plans/2026-06-12-consistency-final-test-plan.md`，执行报告见 `developing-logs/reports/2026-06-12-consistency-final-test-report.md`，用户文档见 `docs/developer-guide/consistency-test.md`。

### ⚠️ 本分支现状：这些资产的源码**不在当前分支**

`tests/consistency/` 在当前 `integrate-prompt-continuation-switch` 分支上**只有 `__pycache__`，没有任何 .py 源码**（`git ls-files tests/consistency/` 为空）。这套测试代码 commit 在 **`new_system_stanley`** 分支（5 个文件）。

→ **不能假设"直接复用"，在写 two-stage 正式测试前必须先把它们弄到本分支**：
- 先 diff 再取：`git diff new_system_stanley -- tests/consistency/`，确认当前分支没有需要保留的测试源码改动后，再 `git checkout new_system_stanley -- tests/consistency/`。`pyproject.toml` 不要整文件 checkout，避免覆盖当前分支的新依赖/配置；只手动合并 pytest marker 等必要小改动；
- 当前分支已确认 `scripts/run_lekai_offline.py --bpm` 和 `model.py::generate_accompaniment(bpm_override=...)` **不存在**。若要先跑单阶段 consistency 基线，必须先 cherry-pick `new_system_stanley` 对应改动，或手动补最小 `--bpm` / `bpm_override` 改动；否则单阶段基线会因为 BPM 桶不一致而无法 green。
- 或按下面的描述重建。
- 取来后先跑一遍单阶段测试确认基线绿，再做 two-stage 增量。

下面这节描述的是这套代码的**设计**（照着理解/重建），不代表它现在就在你手边。

### 已建好的资产（在 `new_system_stanley` 分支的 `tests/consistency/`，需先取到本分支）

- **`midi_pianoroll.py`**：核心对比。`compare_accompaniment(realtime_midi, offline_midi, max_beat) -> PianorollComparison`（带 `.is_consistent` / `.match_rate` / `.summary()`）。把伴奏轨（track 名含 `Accompaniment`/`Part1`）的每个音符**展开成它覆盖的 `(beat, pitch)` 格子集合**再比，`max_beat` 截断公共窗口。
- **`conftest.py`**：
  - 门控：`LEKAI_CHECKPOINT_PATH` 未设或文件不存在 → 整目录 skip（日常 `uv run pytest tests/` 不受影响）。
  - `lekai_server` fixture（session 级）：找空闲端口 → subprocess 起 `server_lekai`（注入确定性 env）→ 轮询 `/health` → 预热请求 → yield → teardown kill + 失败时 dump server log。
  - `SongSpec`：每首歌的 `condition_idx`、`melody_last_beat`（从 mel MIDI 读）、`max_ticks`（= `(melody_last_beat + TAIL_BEATS=24) * 4`）。two-stage 只借用旋律长度/歌号/max_ticks，不使用 `condition_idx`。
  - 常量：`CONDITION_BPM=120`、`DETERMINISTIC_SERVER_ENV`、`SONG_TO_CONDITION_IDX`、`NON_EMPTY_SONGS`、`artifacts_dir`（`output/consistency/<timestamp>/`）。
- **`runners.py`**：`run_realtime(server, song, tempo, out_dir)`（驱动 CLI，返回 session 目录）、`run_offline(checkpoint, song, out_dir)`（驱动离线脚本）、`count_dropped_requests(session_dir)`（从 `inferences.json` 的 `generation_start_tick` 是否连续步进检测丢弃）。
- **`test_realtime_offline_consistency.py`**：按歌参数化、歌内遍历 tempo；零丢弃前置 + 断言一（vs offline）+ 断言二（跨 tempo 一致）。
- **`pyproject.toml`** 已注册 `consistency` marker。
- **`scripts/run_lekai_offline.py`** 在 `new_system_stanley` 分支已加 `--bpm` 直传（配套 `model.py::generate_accompaniment` 的 `bpm_override`）；当前分支尚未包含这两个改动。

跑法：`LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors STREAMMUSE_CONSISTENCY_GPU=1 uv run pytest tests/consistency/ -v`。可调 env：`STREAMMUSE_CONSISTENCY_SONGS`（默认 `4`）、`_TEMPOS`（默认 `15,120`）、`_GPU`。

### 八条用血泪换来的关键经验（two-stage 全部适用）

1. **只用非空歌**：贪心（top_k=1, temp=0）下很多歌生成**全空伴奏**（已知模型行为，非 bug），空对空比较无意义。单阶段实测非空程度：song 4（56/76 非空拍，默认）> 5（19/116）> 2（9/96）> 3（1，几乎空）> 1（0，全空）。**别用 song 1。**
2. **`condition_idx` 与歌号不相等**：`PianoDataset` 按 `[2,5,3,1,4].npz` 排序，映射 `{1:3, 2:0, 3:2, 4:4, 5:1}`。这只对单阶段 NPZ offline 有意义；two-stage offline driver 直接从 MIDI 读 melody events，不使用 `condition_idx`，最多复用 `SongSpec` 的歌号、旋律长度和 `max_ticks`。
3. **在 pianoroll `(beat,pitch)` 层比，绝不在 raw MIDI note 层比**：实时把持续音**每拍重触发**（note_off+note_on），离线保持**一个长音符**。raw `(tick,pitch,type)` 对比会有 ~78% 假性 mismatch；pianoroll 归一化后才 100%。
4. **截断到旋律窗口 + max-ticks 给够尾部余量**：旋律 MIDI 通常比 max-ticks 短，实时跑过旋律末尾后会在无旋律区继续挂音；对比要截断到 `melody_last_beat`。且 max-ticks 尾部余量要够（24 拍）——太紧（2 拍）会丢窗口内最后一格。
5. **BPM 是分桶条件 token**：`encode_bpm` 把 BPM 分 `<90 慢 / 90–200 中 / >200 快` 三桶（+UNK）。两侧必须**同桶**，否则开头第一个 token 就不同、后面全分叉（历史真踩过）。统一钉 `LEKAI_DEFAULT_BPM=120`（server）+ `--bpm 120`（offline），与各歌原生 BPM（110/74/120/57/92）无关。
6. **tempo 阶梯**：`--tempo` 只控 wall-clock 节奏、不影响生成内容。tempo 15 = 金标准档（慢到推理绝不可能掉队 → 零丢弃）。诊断语义：tempo 15 红 = 真一致性回归；tempo 15 绿 + 120 红 = 推理太慢（性能问题，非 bug）。**零丢弃前置**把这两种红区分开。
7. **对比对象 = session 的 `combined.mid` 的 Accompaniment 轨**（`--output-type session`）。
8. **确定性 = 贪心**：`generation_utils.sample_token` 在 `temperature==0` 时直接 `argmax` 提前 return，所以 top_k/top_p/repetition_penalty 全失效。server 起测时注入 `LEKAI_RT_TEMPERATURE=0.0 / TOP_K=1 / TOP_P=0.0`。判别力已验证（破坏确定性 → 红，单格差异 → 红）。

其它：server 与 offline 同卡共存即可（显存够，无需先停 server）。

---

前置阅读（two-stage 侧）：`developing-logs/2026-6-25/2026-06-26-two-stage-prompt-continuation-research.md`（two-stage 架构详解）。

## 0. 目标与可复用资产

**命题**：在确定性参数下，two-stage 的真实时链路（`streammuse-cli --continuation-mode prompt_continuation`，midi_file 输入）与离线一次性生成，产出的伴奏应在 pianoroll 层逐拍一致。

取回后复用单阶段测试已建好的东西（`tests/consistency/`）：
- `midi_pianoroll.py`：pianoroll (beat,pitch) 归一化对比 + 窗口截断（持续音重触发/长音符差异已归一化）。
- `conftest.py`：门控（checkpoint env 未设即 skip）、`lekai_server` server fixture（空闲端口/health 轮询/预热/teardown）、`SongSpec`（单阶段 condition_idx 映射、旋律长度、max-ticks、`TAIL_BEATS=24`）、工件目录。two-stage 只需要借用旋律长度/歌号/max-ticks，不使用 condition_idx。
- tempo 阶梯（15/120）诊断语义、零丢弃前置、BPM 钉桶、非空歌选择（song 4/5/2）等结论。

素材已确认全在本机：
- **prompt checkpoint**：`~/mbzuai-projects/models/lekai_prompt_model/model.safetensors`（Stage 1）
- **continuation checkpoint**：`~/mbzuai-projects/models/lekai_continuation_model/model.safetensors`（Stage 2）
- 旋律/NPZ：`prompts/inputs_lekai/{mel,npz}/{1..5}`

## 1. 与单阶段的根本不同 —— 三个新风险（plan 的核心）

### 风险 A：prompt 阶段无法靠贪心确定，必须用固定 seed

`prompt_model.generate_music` 用的是 HuggingFace `model.generate(do_sample=True, …)`，**`do_sample=True` 是硬编码的**——即使 temperature 设 0 也不是干净的 argmax 贪心（HF 在 do_sample 下对 temp→0 行为不可靠）。所以 Stage 1 的确定性主要靠固定随机种子：`LekaiPromptEngine._seed_if_configured()` 在每次生成前读 `LEKAI_PROMPT_SEED`（或 `LEKAI_SEED`）并 `torch.manual_seed`。

- 设 `LEKAI_PROMPT_SEED=<固定值>` → 同输入同种子 → 同输出。offline 和 realtime 用**同一个 seed**即可对齐。
- ⚠️ **fp16 + GPU 的 RNG/算子非确定性**：torch 在 GPU 上即便固定种子，某些算子（尤其 fp16）也可能逐次有微小差异，进而采样分叉。Phase 0 必须先验证"同 seed 同输入跑两次，输出伴奏在 pianoroll (beat,pitch) 层是否 100% 一致"；不要把 token sequence bit 级一致作为第一版通过标准。若 pianoroll 不稳，退路：① `LEKAI_PROMPT_DTYPE=float32`；② `torch.use_deterministic_algorithms(True)`（可能需要环境变量 `CUBLAS_WORKSPACE_CONFIG`）；③ 固定到 CPU 跑 prompt（慢但稳）。

> 注意：Stage 1 只在每个 session 调用**一次**（对 prompt 窗口），所以只要这一次可复现，整条链就稳。

### 风险 B：offline 的 history 里躺着"未来旋律"，续写 context 是否正确把它滤掉

**先澄清一个容易混淆的点**：按 predict-ahead 设计（delay_beats=-1），生成 acc[k] 本就**只该用 melody[≤k−1]**，不看 k 及以后——所以**理论上 offline 和 realtime 必然一致，这个风险本不该成立**。真正要担心的不是"该不该看未来"，而是下面这个时机差导致的实现细节：

**offline 和 realtime 喂旋律的时机不同，使得续写 backend 内部 `_melody_history` 在生成 acc[k] 那一刻"持有"的旋律范围不同：**
- **realtime**：旋律随 wall-clock 增量 `append`，生成 acc[k] 时内部 history 里大致**只有 melody[≤k]**。
- **offline**（benchmark_scheduler 模式）：`start`(prompt 窗) 后其余旋律**一次性 append**，所以生成 acc[k] 时内部 history 里**已经躺着 melody[k..结尾] 的全部未来旋律**。

→ 于是问题变成：**当 history 里明明有未来旋律时，续写 context 的构造代码是否严格按 tick 切片把它滤掉？**
- 严格切片（`for beat in range(start_beat, current_beat)`，只迭代到 current_beat 前一拍）→ offline 多出来的未来旋律被滤掉，两者输出相同 → 风险不成立（即上面的"理论"）。
- 若有 off-by-one、或哪里图省事用了 `max(history)` 之类 → offline 会"偷看"到未来、realtime 看不到 → 分叉。

单阶段 `LekaiHttpBackend` 已确认是按 tick 严格切片、不偷看；**但 two-stage 多套了一层 catch-up scheduler 重新驱动这个 backend**（`generation_start_tick = accompaniment_history_beats × 4`，melody 只传增量），这层封装有没有把切片边界/增量拼接搞错，没逐行确认过。所以**保留为风险，但本质是"确认 two-stage 封装没破坏单阶段那条 tick 切片"的快速验证**（Phase 0.2：同一条旋律"一次性 append" vs "分两段 append"，对比续写产出是否逐拍一致）。

### 风险 C：一堆 realtime 行为开关必须钉死

`PromptContinuationRealtimeService` 有若干会改变播放行为的 env，offline 侧没有这些概念。

**核心是"晚到事件抢救"那组（决定要不要这组 + 怎么对齐）：**
realtime 的默认调度 `_schedule_playable` 走 "paired_future_only"——**只排 tick ≥ 当前 wall-clock tick 的伴奏事件，已经过去的直接丢弃（`dropped_past`）**。tempo 越快、推理越跟不上，丢得越多。`RECOVER_LATE_EVENTS` 这组就是把本会丢的晚到事件挪到"现在"补播（会改 tick / 补音）。

**决定（2026-06-26，已确认）：测试时这组全部关闭，不靠抢救、靠降 tempo 保证不丢。**
- `RECOVER_LATE_EVENTS=0`、`REHYDRATE_ACTIVE_NOTES=0`、`BOUND_LATE_RECOVERY` 不设、`RECOVER_LATE_MAX_TICKS` 不设。
- 理由：late recovery 本质是给"赶不上"打的补丁，会挪 tick / 补音，污染 realtime 录出来的内容。测试不该有它。
- "赶不上导致 realtime≠offline" 的正解是**降 tempo**：tempo 足够低（如 15）→ realtime 一定追得上 → **零丢弃** → 播出来的伴奏 = 生成的全量 = offline。**关掉抢救 + 零丢弃 = 干净正确的对比**。
- two-stage 额外好处：prompt 段是一次性生成前 N 拍（几秒），慢 tempo 下 prompt 窗口本身很长（tempo 15 时 8 拍 = 32 秒），prompt 模型从容跑完、不会迟到。

**其余开关：**
- `LEKAI_PROMPT_CONTINUATION_ENGINE=standard`（别混 prompt_extension/bridge 变体；`EXTENSION/BRIDGE_*` 不设）。
- `LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP=1`（client/server 解码 digest 一致性校验，顺带验证；注意它测 client↔server，不是 offline↔realtime）。
- `LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS=1`（两个模型必须真加载，否则静默 fallback 返回空 → 会把"空对空"误判成一致）。
- `observed_until_tick`：realtime 靠它处理边界休止推进 catch-up（`_protocol_worker` 按 `(observed − prompt_length) % interval == 0` 推进）；offline driver 必须把 start/append 的 observed tick 设准，否则 `melody_history_beats` 会少算、catch-up 提前结束。尾部最后几个 tick 不是核心比较对象，可通过对比窗口截断规避停止边界差异。

## 2. 确定性参数设定（fix-output for two-stage）

模型/采样 env（server 端和 offline driver 保持一致；realtime trace 是 CLI/client 侧 env）：

| 阶段 | env | 值 | 说明 |
|---|---|---|---|
| 公共 | `LEKAI_PROMPT_CHECKPOINT_PATH` | `~/mbzuai-projects/models/lekai_prompt_model/model.safetensors` | Stage 1 |
| 公共 | `LEKAI_CONTINUATION_CHECKPOINT_PATH` | `~/mbzuai-projects/models/lekai_continuation_model/model.safetensors` | Stage 2 |
| 公共 | `LEKAI_PROMPT_CONTINUATION_REQUIRE_REAL_MODELS` | `1` | 两个模型必须真加载，禁止静默 fallback 空 |
| Stage 1 | `LEKAI_PROMPT_SEED` | 固定（如 `12345`） | prompt 确定性（do_sample=True，靠种子） |
| Stage 1 | `LEKAI_PROMPT_BPM` / `LEKAI_DEFAULT_BPM` | 钉同一桶（如 `120`） | prompt 的 BPM token |
| Stage 1 | `LEKAI_PROMPT_DTYPE` | 先 `auto`，不稳则 `float32` | 见风险 A 退路 |
| Stage 1 | `LEKAI_PROMPT_TEMPERATURE/TOP_K/TOP_P/REPETITION_PENALTY` | `1.1 / 0 / 0.95 / 1.0` | prompt 采样参数显式 pin，避免外部 env/默认值漂移 |
| Stage 1 | `LEKAI_PROMPT_TIME_SIGNATURE_INDEX` | `4`（默认） | 默认 `prompt_length_ticks=32` 时对应 4/4 下 2 个完整小节；若改 prompt length 要检查小节向上取整 |
| Stage 2 | `LEKAI_RT_TEMPERATURE/TOP_K/TOP_P` | `0.0 / 1 / 0.0` | 续写贪心（同单阶段） |
| Stage 2 | `LEKAI_DEFAULT_BPM` | `120` | 续写 BPM token，与 Stage 1 同桶 |
| 晚到抢救 | `LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS` | `0`（关） | 不补救晚到事件，靠降 tempo 保证不丢 |
| 晚到抢救 | `LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES` | `0`（关） | 不补按住的持续音 |
| 晚到抢救 | `_BOUND_LATE_RECOVERY` / `_RECOVER_LATE_MAX_TICKS` | 不设 | 抢救已关，无意义 |
| 变体 | `LEKAI_PROMPT_CONTINUATION_ENGINE` | `standard` | 不测 bridge 变体 |
| 校验 | `LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP` | `1` | client↔server 解码一致性，顺带验证 |
| Realtime trace | `LEKAI_PROMPT_CONTINUATION_TRACE_PATH` | 每个 realtime run 一个唯一路径 | **传给 `streammuse-cli` 子进程，不是 server env**；零丢弃零裁剪前置要从此 trace 读（two-stage 无 inferences.json） |

注意：`LEKAI_PROMPT_CONTINUATION_TRACE_PATH` 是 `PromptContinuationRealtimeService` 读取的，也就是 CLI/client 进程读取；server fixture 不要靠它产 trace。每个 realtime run 用唯一 trace 文件，或运行前清空旧文件，避免跨 case 累加。

tempo（wall-clock）只影响 realtime 节奏、不影响生成。tempo 阶梯（15 金标准 + 120）的语义：**抢救已全关**，所以 tempo 15 这档零丢弃 → realtime 播出 = 生成全量 = offline，必须一致；某个快档若出现丢弃 → 那是"推理太慢"（性能问题）不是一致性回归。**零丢弃前置检查在 two-stage 里是正确性保证，不只是诊断。**

## 3. 两条路径怎么跑

### Realtime（two-stage）
⚠️ **不要传 `--model-name lekai_prompt_continuation`**：CLI 的 `--model-name` 只接受 `stanley`/`lekai`（`config_parser.py`），会被 argparse 拒。two-stage 是靠 **`--continuation-mode prompt_continuation`** 进入分支的，model_name 在 `cli.py` 内部硬编码为 `lekai_prompt_continuation`。正确命令：
```
streammuse-cli --input-mode midi_file --midi-file-path <song>.mid \
  --inference-type http \
  --continuation-mode prompt_continuation --prompt-length-ticks 32 \
  --server-url http://127.0.0.1:<port> --generation-interval-ticks 4 \
  --tempo <rung> --max-ticks <旋律长+TAIL> \
  --output-type session --log-dir <tmp>
```
伴奏录在 session 的 `combined.mid` Accompaniment 轨（同单阶段）。`--server-url` 给 base 即可（如 `http://127.0.0.1:<port>`），`PromptContinuationHttpClient` 会规整到 `/prompt_continuation/*`。另外启动 CLI 子进程时给 `LEKAI_PROMPT_CONTINUATION_TRACE_PATH=<该 run 的唯一 trace 路径>`，零丢弃前置要从这个 trace 读（见第 4 节）。不要在 prompt-continuation 命令里依赖 `--count-in-beats`：当前 CLI 只把它传给标准 `RealTimeMusicService`，不会传给 `PromptContinuationRealtimeService`。

### Offline（two-stage）—— 需要新写一个小 driver
当前没有 two-stage 的 `run_lekai_offline`。按 `benchmark_lekai_spark.py::benchmark_scheduler` 模式写一个最小离线脚本 `scripts/run_lekai_prompt_continuation_offline.py`：
```python
backend = LekaiPromptContinuationBackend(prompt_checkpoint_path=…, continuation_checkpoint_path=…)
all_melody_events = load_midi_events(<song>.mid)
melody_end_tick = max([int(e.get("tick", 0)) for e in all_melody_events] or [0])
# observed_until_tick 用 beat 边界，避免 melody_history_beats 少算。尾部多出的几 tick 后面对比时可截掉。
final_observed_until_tick = ((melody_end_tick + 3) // 4) * 4

backend.start_prompt_catchup(melody=events_until(all_melody_events, 0, prompt_length_ticks),
                             prompt_length_ticks=…, generation_interval_ticks=4,
                             observed_until_tick=prompt_length_ticks, …)
backend.append_melody_events(events_until(all_melody_events, prompt_length_ticks, final_observed_until_tick),
                             observed_until_tick=final_observed_until_tick)
# 等待要更硬：is_failed / timeout / 终态强断言，别只看 is_playback_ready
deadline = time.perf_counter() + timeout_s
while True:
    status = backend.scheduler_status()
    if status["is_failed"]:
        raise RuntimeError(f"offline scheduler failed: {status['error']}")
    if status["is_playback_ready"]:
        break
    if time.perf_counter() >= deadline:
        raise TimeoutError(f"offline catch-up not ready in {timeout_s}s: {status}")
    time.sleep(0.05)
# 终态断言：伴奏确实追到了"observed melody + lookahead"（is_playback_ready 语义，catchup_state.py:60）
assert status["accompaniment_history_beats"] >= status["target_playable_accompaniment_beats"]
acc = backend.raw_accompaniment_history()      # 离线结果（完整生成，未经播放裁剪）
# acc 转 MIDI（复用 MidiFileOutputSink）或直接转 (tick,pitch) 供对比
```
旋律来源：用和 realtime 完全相同的输入 MIDI（`MidiFileInput._midi_to_notes` 抽出 notes → events），保证两侧 melody 逐音符相同（参考单阶段的 frame 级对齐验证）。默认 `prompt_length_ticks=32` 在 4/4 下刚好是 2 个完整小节；prompt engine 会把 prompt window 向上取整到完整小节再构造 token，若未来改成 20/24 等非整小节长度，需要重新检查窗口取整和截断。

> - offline driver 进程内直接调 backend（不走 HTTP），也可选择走 HTTP `/prompt_continuation/*` 同步驱动——优先**进程内**，少一层网络，确定性更可控。但要保证 env 与 realtime server 一致。
> - `observed_until_tick` 要按 beat/grid 边界精确设置，至少不能小于旋律实际最大 tick；推荐 `ceil(max_tick / 4) * 4`。最后几个 tick 不是本测试重点；如果尾部终点差一小段，可以在对比窗口里截掉尾部，只比较共同有效的旋律窗口。

## 4. 对比方法（沿用单阶段三条铁律）

比的是 realtime 实际播放/录制的 `combined.mid`（不是 raw history）——因为 late recovery 全关 + 慢 tempo 零丢弃零裁剪，播出来的本就 = 生成全量。这样测的是含播放调度的**完整管线**。最后几个 tick 可以不比，对比窗口截到共同有效的旋律窗口即可。

1. **只用非空歌**：先验证 two-stage 在贪心+seed 下哪些歌伴奏非空（prompt 模型可能让更多歌非空——这正是 two-stage 的卖点，Phase 0 顺带量一下）。默认 song 4，辅以 5/2。
2. **pianoroll (beat,pitch) 层对比**，不用 raw MIDI note。直接 `tests/consistency/midi_pianoroll.compare_accompaniment(realtime_combined, offline_acc_midi, max_beat=melody_last_beat)`。
3. **截断到旋律窗口** + max-ticks 给够尾部余量（TAIL_BEATS=24）。尾部最后几个 tick 若只是停止边界差异，不纳入一致性窗口。

断言：
- **前置（正确性保证）**：realtime **零丢弃且零裁剪**——`dropped_past == 0 && clipped_sustains == 0`。
  - ⚠️ **不能复用单阶段的 `count_dropped_requests`**：那是从 `inferences.json` 的 `generation_start_tick` 连续性推断的，而 **prompt-continuation service 根本不写 inferences.json**（不走 `log_inference` 那条路径）。
  - 正确来源：启动 `streammuse-cli` 子进程时设 `LEKAI_PROMPT_CONTINUATION_TRACE_PATH=<path>`，从 trace 的 `schedule_playable` 记录里读 `dropped_past` 和 `clipped_sustains`（两者都在该 trace 行里，见 `_schedule_playable`）。
  - 为什么也要 `clipped_sustains == 0`：late recovery 虽关，但默认 `_clip_playable_to_current_tick` 仍会把"已开始、note_off 在未来"的**持续音在 current_tick 重新触发**（挪了 note_on 的 tick），同样污染录制结果。只有 dropped 和 clipped 都为 0，才真正"播出 = 生成全量"。
  - 任一不为 0 = 该 tempo 推理太慢 → 报 "run invalid: inference too slow"，不是一致性失败。
- **断言一**：每 (歌, tempo) realtime == offline，`is_consistent`（窗口内 mismatched==0）。
- **断言二**：同歌不同 tempo 的 realtime 两两一致（时钟不泄漏进生成；前提是各 tempo 都零丢弃零裁剪）。
- **新增断言三（暂不作为第一版自动断言）**：prompt 段单独对比——realtime 的前 prompt_length 拍伴奏 == offline 的 prompt 段。注意如果 realtime 通过 CLI 子进程跑，父进程等 CLI 退出后再拉 `/prompt_continuation/prompt_history` 已经来不及，因为 CLI 的 `atexit cleanup` 会先调用 `/clear_history`。第一版先用主对比 + trace 定位；若后续要自动化断言三，需要改 CLI 在 clear 前保存 prompt/raw history，或让 runner 在 CLI 仍存活时并发 fetch。

## 5. 测试落位

- **前提**：先按 Phase 0.5 把单阶段基线依赖补齐，把 `tests/consistency/` 从 `new_system_stanley` 取到本分支，并确认单阶段基线能 green。
- 新建 `tests/consistency/test_two_stage_consistency.py`，复用 `conftest.py` 的 fixture，但需要一个 **two-stage 版 server fixture**（注入两个 checkpoint + `LEKAI_PROMPT_SEED` + Stage 1/2 采样参数 + 晚到抢救全关 + `REQUIRE_REAL_MODELS=1`）和 **two-stage runner**（CLI `--continuation-mode prompt_continuation`（不传 `--model-name`）+ 每个 run 的 `TRACE_PATH` CLI env + offline driver）。
- 把 two-stage 专属常量（两个 checkpoint 路径、seed、行为开关 env）集中在新 fixture 里，门控用 `LEKAI_PROMPT_CHECKPOINT_PATH` 存在与否（与单阶段的 `LEKAI_CHECKPOINT_PATH` 区分）。
- **丢弃/裁剪检测换实现**：写一个 `count_dropped_and_clipped(trace_path)` 从 `LEKAI_PROMPT_CONTINUATION_TRACE_PATH` 的 `schedule_playable` 行累加 `dropped_past` / `clipped_sustains`；**不要**用单阶段的 `count_dropped_requests`（two-stage 无 inferences.json）。
- **raw/prompt history 的自动保存先不作为第一版要求**：CLI 退出时 `cleanup()` 走 `/clear_history`，server 端会**顺手清掉 two-stage 的 scheduler history**（`server_lekai.py` 的 `clear_history` 里 `prompt_continuation_backend.clear_history()`）。如果后续要做断言三，需要改 CLI 在 clear 前保存 `prompt_history/raw_history`，或让 runner 在 CLI 仍存活时并发 fetch。主对比走 `combined.mid` 不受此影响。
- marker 复用 `consistency`（或加 `two_stage`）。日常 `uv run pytest tests/` 仍 skip。
- `/prompt_continuation/runtime_info` 返回 plain dict；测试里断言字段时优先用 `.get()`，避免缺字段时直接 KeyError 掩盖真实启动错误。

## 6. 执行 Todo List

### Phase 0：把三个新风险逐一验证（写正式测试前必须全绿）
- [ ] **0.0 起 two-stage server + 兼容性 sanity**：双 checkpoint + `LEKAI_PROMPT_SEED` + Stage 1 采样参数 pin + 贪心续写 + `REQUIRE_REAL_MODELS=1` + 晚到抢救全关。验证查 **`GET /prompt_continuation/runtime_info`**（不是 standard 的 `/runtime_info`），用 `.get()` 断言 continuation 的 `has_real_model == true` **且** prompt 的 `prompt_has_real_model == true`。随后手动调一次 `/prompt_continuation/start`，等 scheduler 完成，确认 `scheduler_is_failed/is_failed` 为 false、没有 representation digest mismatch，并且 `prompt_accompaniment_history` 与 `raw_accompaniment_history` 都非空。
- [ ] **0.1 prompt 确定性（风险 A）**：同 seed 同输入，连续调两次 `start_prompt_catchup`→`prompt_accompaniment_history`，把两次输出转成 pianoroll (beat,pitch) 后断言 100% 一致。不要把 token sequence bit 级一致作为第一版通过标准。不一致→试 `LEKAI_PROMPT_DTYPE=float32` / 确定性算子 / CPU，记录哪种稳。
- [ ] **0.2 续写不看未来（风险 B）**：构造同一条旋律，一次"全 append"、一次"分两段 append"，对比续写产出是否一致（验证 acc[k] 不依赖未来旋律）。不一致→深入 `LekaiHttpBackend.generate` 的 context 取值，记录真实依赖范围。
- [ ] **0.3 选非空歌**：对 song 1..5 跑 offline two-stage（贪心+seed），统计各歌伴奏非空拍数，挑默认歌（预期 two-stage 比单阶段更多歌非空）。
- [ ] **0.4 手动 offline vs realtime（song 默认）**：在正式 offline driver 之前，可以用临时 driver / inline script 先跑一组 offline two-stage + CLI realtime，做 pianoroll 对比。melody 先做 frame 级对齐 sanity gate（同单阶段），再比 acc，目标 100%。**不到 100% 停下排查**（先看 prompt 段还是续写段；断言三暂不自动化），记录原因。

### Phase 0.5：补齐单阶段基线依赖
- [ ] 0.5.1 先 diff：`git diff new_system_stanley -- tests/consistency/`，以及 `git diff new_system_stanley -- scripts/run_lekai_offline.py src/streammuse/infrastructure/inference/lekai_model/model.py`。
- [ ] 0.5.2 当前分支已确认没有 `scripts/run_lekai_offline.py --bpm` / `model.py bpm_override`。先 cherry-pick `new_system_stanley` 对应改动，或手动补最小 `--bpm` / `bpm_override`，否则单阶段 consistency 基线无法 green。
- [ ] 0.5.3 取回测试基建：`git checkout new_system_stanley -- tests/consistency/`；`pyproject.toml` 只手动合并 pytest marker，避免覆盖当前分支配置。
- [ ] 0.5.4 跑通单阶段基线确认绿，再做 two-stage 正式测试增量。

### Phase 1：offline driver
- [ ] 1.1 写 `scripts/run_lekai_prompt_continuation_offline.py`（进程内驱动 backend，输出 acc MIDI + raw_history JSON）。
- [ ] 1.2 与 0.4 手动结果一致性自检；`uv run pytest tests/ -q` 不受影响。

### Phase 2：测试基建
- [ ] 2.1 `conftest` 加 two-stage server fixture（双 checkpoint/seed/Stage 1 采样参数/晚到抢救全关/REQUIRE_REAL_MODELS）+ runner（CLI two-stage，不传 `--model-name`；CLI env 传每个 run 的 `TRACE_PATH`；offline driver）。
- [ ] 2.2 写 `count_dropped_and_clipped(trace_path)`（从 `schedule_playable` trace 累加 `dropped_past`/`clipped_sustains`）；raw/prompt history 自动保存留到后续需要断言三时再做。
- [ ] 2.3 复用 `SongSpec` / pianoroll / 工件目录；门控用 `LEKAI_PROMPT_CHECKPOINT_PATH`。two-stage 不用 `condition_idx`。

### Phase 3：测试本体
- [ ] 3.1 `test_two_stage_consistency.py`：按歌参数化、歌内遍历 tempo，**零丢弃零裁剪前置**（`dropped_past==0 && clipped_sustains==0`，从 trace 读）+ 断言一/二；prompt 段单独断言暂留到需要 CLI history 保存时再做。

### Phase 4：真跑 + 判别力
- [ ] 4.1 默认配置（默认歌 × 15+120）跑全绿，记耗时。
- [ ] 4.2 判别力：① 改 `LEKAI_PROMPT_SEED` 让 offline/realtime 用**不同 seed** → 应红（prompt 段分叉）；② 续写 `LEKAI_RT_TOP_K=2`+temp 0.8 → 应红；③ 恢复→绿。
- [ ] 4.3 日常套件不受影响（skip）。

### Phase 5：收尾
- [ ] 5.1 docs/research 补一段"two-stage 一致性测试怎么跑"。
- [ ] 5.2 勾 todo + 写 report 到 `developing-logs/2026-6-25/`。

## 7. 待你确认的决策点
1. **prompt 确定性**：接受"固定 seed"作为 two-stage 的 fix-output 定义吗？（do_sample=True 硬编码，没有干净贪心；这是 two-stage 与单阶段最大的语义差别。）
2. **offline driver 走进程内还是 HTTP**？推荐进程内（更可控），但要保证 env 与 realtime server 完全一致。
3. **prompt_length_ticks** 用默认 32（8 拍）还是别的？这决定 prompt 段覆盖多长。
4. 默认歌 + tempo 阶梯沿用单阶段（默认 1 首 + 15/120），还是想直接多跑几首？
5. 是否要顺带验证 `prompt_extension`（bridge）变体，还是只测 standard？
