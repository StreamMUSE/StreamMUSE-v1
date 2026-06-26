# Two-Stage Prompt+Continuation Consistency Test 实现报告（2026-06-26）

## 1. 核心结论

本轮已经把 `integrate-prompt-continuation-switch` 分支上的 two-stage consistency test 基建搭起来了，包括：

- 从 `new_system_stanley` 取回单阶段 consistency 测试基建；
- 给单阶段 offline 补上 `--bpm` / `bpm_override`，让单阶段基线具备 BPM 桶对齐能力；
- 新增 two-stage offline driver：`scripts/run_lekai_prompt_continuation_offline.py`；
- 新增 prompt 阶段 token log 能力；
- 新增 two-stage server fixture、runner、trace 统计、pianoroll 对比测试；
- 跑了轻量验证，并跑了一次真模型最小端到端验证。

更新结论：**two-stage inference consistency 已切换到 server raw-history/context 对比，并且真模型默认 tempo 阶梯验证已 green**。现在测试不再用 `combined.mid` 判断 two-stage inference 是否一致，因为 prompt stage 生成的是过去窗口伴奏，realtime playback/schedule 层天然会丢掉这部分；这属于单独的 playback/recording 问题。

当前完整默认 tempo 阶梯验证结果：

```text
命令：
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s

结果：1 passed in 395.73s (0:06:35)
```

测试现在比较：

- realtime CLI session 退出前保存的 `prompt_continuation_prompt_history.json`；
- realtime CLI session 退出前保存的 `prompt_continuation_raw_history.json`；
- offline driver 输出的 `stage1_prompt_history.json`；
- offline driver 输出的 `stage2_raw_history.json`。

比较方式：先确认 prompt context 一致；再用 `raw_history - prompt_history` 得到 continuation context，并截断到共同旋律窗口内比较。这样测的是 server 端实际用于 continuation 的 acc context，而不是 client schedule 是否来得及把它播出来。

历史发现仍然保留：第一版用 `combined.mid` 对比时失败，原因是 realtime playback 路径没有把 generated history 写进 Accompaniment 轨。对应 artifact：

```text
output/consistency/20260626-055750/two_stage/two_stage_song4_tempo15/prompt_continuation_trace.jsonl
output/consistency/20260626-055750/two_stage/two_stage_song4_tempo15/2026-06-26/session_055821/combined.mid
output/consistency/20260626-055750/two_stage/two_stage_offline_song4/001_4/final.mid
output/consistency/20260626-055750/two_stage/two_stage_offline_song4/001_4/summary.json
```

## 2. 已完成改动

### 2.1 单阶段 consistency 基线依赖

按 plan 的 Phase 0.5，先把当前分支缺的单阶段基建补齐：

- 恢复 `tests/consistency/` 测试源码；
- `pyproject.toml` 增加 `consistency` pytest marker；
- `scripts/run_lekai_offline.py` 增加 `--bpm`；
- `src/streammuse/infrastructure/inference/lekai_model/model.py` 增加 `bpm_override`。

核心代码：

```python
# scripts/run_lekai_offline.py
parser.add_argument(
    "--bpm",
    type=int,
    default=None,
    help="Override conditioning BPM (default: read from NPZ metadata). "
    "Used to pin BPM token identical to the realtime side.",
)

result = model.generate_accompaniment(
    dataset,
    condition_idx=idx,
    ...,
    bpm_override=args.bpm,
)
```

```python
# src/streammuse/infrastructure/inference/lekai_model/model.py
bpm_value = bpm_override if bpm_override is not None else metadata["bpm"]
```

这保证 restored 的单阶段 runner 里这条命令是可跑的：

```python
# tests/consistency/runners.py
"--bpm", str(CONDITION_BPM),
```

### 2.2 Two-stage offline driver

新增文件：`scripts/run_lekai_prompt_continuation_offline.py`。

它支持四种输入：

```text
--midi-file <path>
--midi-dir <dir>
--npz-file <path>
--npz-dir <dir>
```

NPZ 输入会自动找对应 melody MIDI：

```python
if npz_path.parent.name == "npz":
    candidates.append(npz_path.parent.parent / "mel" / f"{stem}.mid")
candidates.append(npz_path.with_suffix(".mid"))
candidates.append(Path.cwd() / "prompts" / "inputs_lekai" / "mel" / f"{stem}.mid")
```

默认 checkpoint 路径遵循当前项目约定：

```python
DEFAULT_MODELS_DIR = Path.home() / "mbzuai-projects" / "models"
DEFAULT_PROMPT_CHECKPOINT = DEFAULT_MODELS_DIR / "lekai_prompt_model" / "model.safetensors"
DEFAULT_CONTINUATION_CHECKPOINT = DEFAULT_MODELS_DIR / "lekai_continuation_model" / "model.safetensors"
```

offline driver 的核心流程：

```python
melody_end_tick = max([int(event.get("tick", 0)) for event in melody_events] or [0])
final_observed_until_tick = max(
    int(args.prompt_length_ticks),
    ((int(melody_end_tick) + TIMESTEPS_PER_BEAT - 1) // TIMESTEPS_PER_BEAT) * TIMESTEPS_PER_BEAT,
)
prompt_events = events_until(melody_events, 0, int(args.prompt_length_ticks))
append_events = events_until(melody_events, int(args.prompt_length_ticks), final_observed_until_tick)

backend = LekaiPromptContinuationBackend(
    prompt_checkpoint_path=str(prompt_checkpoint),
    continuation_checkpoint_path=str(continuation_checkpoint),
)
backend.start_prompt_catchup(
    melody_events=prompt_events,
    prompt_length_ticks=int(args.prompt_length_ticks),
    generation_interval_ticks=int(args.generation_interval_ticks),
    inference_mode="sliding_window",
    model_name="lekai_prompt_continuation",
    checkpoint_path=None,
    observed_until_tick=int(args.prompt_length_ticks),
)
backend.append_melody_events(
    append_events,
    observed_until_tick=final_observed_until_tick,
)
final_status, scheduler_samples = wait_for_scheduler(backend, timeout_s=float(args.timeout_s))
```

输出内容包括：

```text
final.mid
stage1_prompt.mid
stage2_raw_history.mid
melody_events.json
stage1_prompt_history.json
stage1_token_log.json
stage2_raw_history.json
stage2_playable_history.json
scheduler_status_log.json
summary.json
stage2_token_logs/*.json
batch_summary.json
```

推荐离线批量运行命令：

```bash
uv run scripts/run_lekai_prompt_continuation_offline.py \
  --npz-dir prompts/inputs_lekai/npz/ \
  --output-dir outputs/lekai_prompt_continuation_offline/inputs_lekai \
  --device cuda:0 \
  --prompt-fp16 \
  --rt-fp16
```

### 2.3 Prompt stage token log

修改文件：`src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_engine.py`。

新增 latest generation diagnostics：

```python
self._last_prompt_token_ids: list[int] = []
self._last_generated_token_ids: list[int] = []
self._last_new_token_ids: list[int] = []
self._last_generation_metadata: dict[str, Any] = {}
```

导出方法：

```python
def last_generation_log(self) -> dict[str, Any]:
    return {
        "prompt_tokens": list(self._last_prompt_token_ids),
        "prompt_token_count": len(self._last_prompt_token_ids),
        "generated_tokens": list(self._last_generated_token_ids),
        "generated_token_count": len(self._last_generated_token_ids),
        "new_tokens": list(self._last_new_token_ids),
        "new_token_count": len(self._last_new_token_ids),
        "generated_acc_beats": int(self._last_generated_acc_beats),
        **dict(self._last_generation_metadata),
    }
```

offline driver 通过 backend 内部 prompt engine 拉取该 log，写入 `stage1_token_log.json`。这是诊断脚本用途，暂未改变 realtime 推理逻辑。

### 2.4 Two-stage consistency fixture 和 runner

新增/扩展文件：

```text
tests/consistency/conftest.py
tests/consistency/two_stage_runners.py
tests/consistency/test_two_stage_prompt_continuation_consistency.py
```

two-stage fixture 的 opt-in 规则：

```python
def _env_path_or_default(env_name: str, default: Path) -> Path | None:
    raw = os.environ.get(env_name)
    if raw:
        path = Path(raw).expanduser()
        return path if path.exists() else None
    if os.environ.get("STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS", "").lower() in {"1", "true", "yes", "on"}:
        return default if default.exists() else None
    return None
```

也就是说默认日常测试不会加载大模型；只有显式提供 checkpoint env 或允许使用本机默认模型时才运行。

two-stage server 启动后会检查真实模型是否加载：

```python
info = requests.get(f"{base_url}/prompt_continuation/runtime_info", timeout=30).json()
if not bool(info.get("has_real_model")) or not bool(info.get("prompt_has_real_model")):
    raise RuntimeError(...)
```

realtime runner 命令重点：

```python
cmd = [
    sys.executable,
    "-m",
    "streammuse.presentation.cli.cli",
    "--input-mode", "midi_file",
    "--midi-file-path", str(song.mel_path),
    "--inference-type", "http",
    "--continuation-mode", "prompt_continuation",
    "--prompt-length-ticks", str(PROMPT_LENGTH_TICKS),
    "--server-url", server.base_url,
    "--generation-interval-ticks", str(GENERATION_INTERVAL_TICKS),
    "--max-ticks", str(song.max_ticks),
    "--tempo", str(tempo),
    "--output-type", "session",
    "--log-dir", str(log_dir),
]
```

注意这里没有传 `--model-name`，也没有传 `--count-in-beats`。`--count-in-beats` 当前不传给 `PromptContinuationRealtimeService`，在 prompt-continuation 模式下不生效。

trace 统计现在保留这些字段：

```python
return {
    "dropped_past": dropped_past,
    "clipped_sustains": clipped,
    "dropped_too_late_note_on": dropped_too_late,
    "skipped_unpaired": skipped_unpaired,
    "scheduled_events": scheduled_events,
    "schedule_rows": schedule_rows,
}
```

测试断言现在不再要求 schedule 零丢弃；schedule trace 只作为诊断。核心断言改为 server raw-history context：

```python
realtime_prompt_context = event_counter(realtime_prompt_events, max_tick=PROMPT_LENGTH_TICKS)
realtime_continuation_context = continuation_counter(
    realtime_raw_events,
    realtime_prompt_events,
    max_tick=comparison_end_tick,
)

assert realtime_prompt_context == offline_prompt_context
assert realtime_continuation_context == offline_continuation_context
```

其中 `continuation_counter(raw, prompt)` 会先做 `raw_history - prompt_history`，再截断到共同旋律窗口，只比较 continuation 实际使用的 acc context。

## 3. 怎么运行

### 3.1 日常轻量检查

```bash
uv run pytest tests/consistency -q --collect-only
uv run pytest tests/consistency -q
```

在没有 checkpoint env 且没有 default-model opt-in 时，应该只收集测试但全部 skip。

### 3.2 Two-stage 真模型 consistency

使用本机默认模型：

```bash
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s
```

显式 checkpoint：

```bash
LEKAI_PROMPT_CHECKPOINT_PATH=~/mbzuai-projects/models/lekai_prompt_model/model.safetensors \
LEKAI_CONTINUATION_CHECKPOINT_PATH=~/mbzuai-projects/models/lekai_continuation_model/model.safetensors \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s
```

### 3.3 单阶段 consistency 基线

```bash
LEKAI_CHECKPOINT_PATH=<single-stage-checkpoint.safetensors> \
STREAMMUSE_CONSISTENCY_SONGS=4 \
STREAMMUSE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_realtime_offline_consistency.py -q -s
```

## 4. 已跑验证

轻量验证：

```text
python3 -m py_compile scripts/run_lekai_prompt_continuation_offline.py ...
结果：通过

uv run python scripts/run_lekai_prompt_continuation_offline.py --help
结果：CLI help 正常输出

uv run pytest tests/unit/application/test_prompt_continuation_realtime_service.py -q
结果：12 passed in 0.30s

uv run pytest tests/unit/infrastructure/test_tokenization_imports.py \
  tests/unit/infrastructure/test_config_imports.py \
  tests/unit/presentation/test_cli_config_parser.py -q
结果：14 passed in 1.74s

uv run pytest tests/consistency -q --collect-only
结果：2 tests collected

uv run pytest tests/consistency -q
结果：2 skipped
```

真模型最小验证：

```text
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s

结果：1 passed in 395.73s (0:06:35)
```

此前 `combined.mid` playback 对比失败不是模型文件缺失。模型文件确认存在：

```text
~/mbzuai-projects/models/lekai_prompt_model/model.safetensors       578M
~/mbzuai-projects/models/lekai_continuation_model/model.safetensors 650M
```

## 5. Playback/recording 失败的证据链（非当前 inference consistency 判定）

### 5.1 server 和模型路径正常

测试 fixture 成功启动 server，并通过：

```text
GET /health
GET /prompt_continuation/runtime_info
```

如果两个真实模型没有加载，fixture 会直接失败；这次不是这个问题。

### 5.2 Stage 1 prompt history 在 realtime 里生成了，但调度时已经全在过去

trace 第一条 `playable_fetch` 显示 server/client representation digest 是一致的，说明 client/server 解码不是 mismatch：

```text
playable_representation_match = true
client/server event_count = 52
min_tick = 12
max_tick = 32
```

但第一次真正 `schedule_playable` 时：

```json
{
  "kind": "schedule_playable",
  "current_tick": 36,
  "input_event_count": 52,
  "input_min_tick": 12,
  "input_max_tick": 32,
  "past_event_count": 52,
  "future_event_count": 0,
  "dropped_past": 52,
  "scheduled_event_count": 0
}
```

也就是说 prompt stage 生成的伴奏范围是 tick 12-32，但 realtime 服务到 tick 36 才拿它来调度，所以它作为“过去事件”全部被丢掉。这个现象和 plan 里的预期不同：plan 假设 tempo 15 足够慢就能“零丢弃”，但当前实现里 prompt 段的时间语义天然会让它在 prompt 窗口结束后才可用。

### 5.3 Stage 2 continuation 也没有被写进 `combined.mid`

trace 汇总：

```text
schedule_rows = 72
dropped_past = 8712
clipped_sustains = 0
dropped_too_late_note_on = 0
skipped_unpaired = 72
scheduled_event_count = 0
```

典型 continuation log 里生成的是同 tick 的 note_on/note_off 事件：

```text
Start tick: 208
Acc events: 8
min_event_tick: 208
max_event_tick: 208
note_on_count: 4
```

而 realtime 的 `_schedule_playable` 会先 `_clip_playable_to_current_tick`，再 `_pair_playable_events`。当事件落在当前 tick 或已经过去时，会被丢；当 note_on/note_off 同 tick 且排序/配对不支持这种零时长表示时，会进入 `skipped_unpaired`。最终没有任何 model event 被 schedule 到 session output。

### 5.4 MIDI 对比确认 realtime accompaniment 为空

用现有 pianoroll helper 比较：

```text
window 80:
  realtime cells = 0
  offline cells = 209
  matched = 0
  match_rate = 0.0%
```

这说明当前失败不是“最后几个 tick 停止边界差异”，而是实时录制输出整体没有写入伴奏。

## 6. 当前状态和剩余 blocker

Two-stage consistency test 现在守护的是 **server-side inference context**，也就是 continuation 实际使用的 acc history。这个目标已经通过最小真模型验证。

剩余 blocker 是另一个问题：**two-stage realtime 的 actual playback/recording 路径没有把 prompt+continuation generated history 写进 `combined.mid` 的 Accompaniment 轨**。如果之后要做“用户实际听到/录到”的端到端测试，还需要单独修 `PromptContinuationRealtimeService` 的调度语义：

- prompt 段是否应该延迟播放、count-in、或作为历史伴奏写入 session；
- continuation 同 tick note_on/note_off 是否要归一成至少一拍 sustain；
- `_pair_playable_events` 和 `_clip_playable_to_current_tick` 如何处理 tick == current_tick 的事件。

因此当前建议是把两类测试分开：

- **two-stage inference consistency**：比较 `prompt_continuation_raw_history.json` / `prompt_continuation_prompt_history.json`，已落地；
- **two-stage playback regression**：未来再比较 `combined.mid`，当前仍是 known failing/blocker。

## 7. 工作树备注

`git status` 里还会看到 `uv.lock` 和 `src/streammuse.egg-info/*` 的变化。这些是 `uv run` 同步项目环境时带出的生成/锁文件变化，核心 consistency 实现不依赖这些文件。

另外当前 workspace 本来就有一些 untracked 目录，例如 `docs/.vitepress/`、`node_modules/`、`output/`、`outputs/`。本轮没有清理这些目录，避免误删用户已有状态。

## 8. Realtime Scheduling 修复实现（2026-06-26 追加）

本轮根据 `developing-logs/plans/2026-06-26-prompt-continuation-streaming-scheduling-fix-plan.md` 完成了 prompt+continuation realtime playback scheduler 修复。核心结论更新为：之前 `combined.mid` 空/缺拍主要不是 server inference context 问题，而是 client scheduling 层把 streaming event history 强行配成 note pair，导致边界事件和跨 chunk note 被跳过。

### 8.1 核心代码改动

修改文件：`src/streammuse/application/services/prompt_continuation_realtime_service.py`。

新增默认 scheduler mode：

```python
raw_scheduling_mode = os.environ.get(
    "LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE",
    "streaming_events",
).strip().lower()
if raw_scheduling_mode not in {"streaming_events", "paired_future_only"}:
    raw_scheduling_mode = "streaming_events"
self._scheduling_mode = raw_scheduling_mode
```

`_schedule_playable()` 现在默认走 streaming event scheduler，旧 pair path 只保留为显式诊断模式：

```python
def _schedule_playable(self, accompaniment: list[MusicalEvent], *, current_tick: int) -> None:
    if self._scheduling_mode == "paired_future_only":
        self._schedule_playable_paired_future_only(accompaniment, current_tick=int(current_tick))
        return
    self._schedule_playable_streaming_events(accompaniment, current_tick=int(current_tick))
```

streaming scheduler 的关键语义：

```python
for event in [*rehydrated_events, *usable_events]:
    event_key = self._event_key(event)
    seen_in_payload[event_key] += 1
    occurrence = seen_in_payload[event_key]
    if self._handled_model_event_counts[event_key] >= occurrence:
        skipped_duplicate += 1
        continue

    event_tick = int(event.tick)
    schedule_tick = event_tick
    if event_tick < current_tick:
        late_event_count += 1
        if not self._recover_late_events:
            dropped_past += 1
            self._ensure_event_count(self._handled_model_event_counts, event_key, occurrence)
            continue
        if self._would_drop_late_note_on(event, current_tick=current_tick):
            dropped_too_late_note_on += 1
            self._ensure_event_count(self._handled_model_event_counts, event_key, occurrence)
            continue
        schedule_tick = current_tick

    model_event = self._to_model_event(event, current_tick=current_tick)
    self._scheduler.schedule(model_event, schedule_tick)
```

这带来几个行为变化：

- 不再要求 `note_on` / `note_off` 在同一次 playable response 内形成完整 pair。
- 同 tick 的 `note_off -> note_on` 可以按 event stream 顺序 schedule。
- 跨 chunk 的 `note_on` 会先播放，后续 `note_off` 到来时再独立 schedule。
- `/playable` 重复返回 full history 时，用 `Counter[EventKey]` 做 occurrence-level 去重。
- event key 扩展为 `(tick, pitch, event_type, velocity, channel, program)`，避免不同 channel/program 被误判重复。

### 8.2 Sustain note rehydrate

新增 `_rehydrate_sustaining_notes()`，处理 `note_on.tick < current_tick < note_off.tick` 的 active span：

```python
if not (note_on_tick < current_tick < note_off_tick):
    continue
span_key = self._note_span_key(note_on, event)
if span_key in self._rehydrated_model_note_span_keys:
    continue
note_on_key = self._event_key(note_on)
if self._played_model_event_counts[note_on_key] >= note_on_occurrence:
    continue
rehydrated.append(self._clone_event_at_tick(note_on, current_tick))
```

实现上区分了两个概念：

- `handled_model_event_counts`：用于 full history duplicate skip。
- `played_model_event_counts`：表示原始 note_on 已经实际播放或被 rehydrated clone 代表。

这个区分很重要：如果一个 late note_on 先被看到但当时还没有 future note_off，不能因为它被 drop/handled 过，就永久禁止后续 sustain rehydrate。

### 8.3 Consistency runner 更新

修改文件：

```text
tests/consistency/two_stage_runners.py
tests/consistency/conftest.py
tests/consistency/test_two_stage_prompt_continuation_consistency.py
```

runner 现在显式设置：

```text
LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE=streaming_events
LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS=0
LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES=1
```

并且 consistency test 会检查 trace：

```python
assert schedule_counts["paired_future_only_rows"] == 0
if schedule_counts["schedule_rows"]:
    assert schedule_counts["streaming_event_rows"] == schedule_counts["schedule_rows"]
```

这样可以防止测试 silent 跑回旧 pair path。

### 8.4 文档更新

更新文件：`src/streammuse/infrastructure/inference/lekai_prompt_continuation/README.md`。

文档里的 scheduling policy 已改成：

- 默认 `streaming_events`；
- `RECOVER_LATE_EVENTS` 只控制 late event recovery，不再控制 pair/streaming 分支；
- sustain rehydrate 默认开启；
- `paired_future_only` 只作为 legacy/diagnostic mode。

### 8.5 验证结果

Unit tests：

```text
uv run pytest tests/unit/application/test_prompt_continuation_realtime_service.py -q
结果：16 passed in 0.31s
```

Two-stage consistency 真模型验证：

```text
STREAMMUSE_CONSISTENCY_USE_DEFAULT_MODELS=1 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS=4 \
STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS=15,120 \
uv run pytest tests/consistency/test_two_stage_prompt_continuation_consistency.py -q -s

结果：1 passed in 395.11s (0:06:35)
```

Prompt-extension realtime sweep：

```text
output/prompt_extension_sweep/20260626-105843/summary.json
```

配置：song 4、tempo 120、prompt extension 1/2/3/4 beats、`max_ticks=128`、`SCHEDULING_MODE=streaming_events`、`RECOVER_LATE_EVENTS=0`。

结果汇总：

| extension | schedule_rows | streaming_rows | paired_rows | scheduled_events | rehydrated | skipped_unpaired | Accompaniment note_on |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 beat | 23 | 23 | 0 | 158 | 0 | 0 | 79 |
| 2 beats | 23 | 23 | 0 | 96 | 7 | 0 | 49 |
| 3 beats | 23 | 23 | 0 | 140 | 15 | 0 | 76 |
| 4 beats | 23 | 23 | 0 | 147 | 24 | 0 | 81 |

四组 sweep 的 `combined.mid` 都包含非空 `Accompaniment` 轨，不再出现旧版本里 `skipped_unpaired` 持续累积且 accompaniment 全空的问题。

### 8.6 剩余注意点

这次验证完成的是事件级 scheduling 和 MIDI artifact 非空验证。我没有做主观听感确认；建议后续人工播放下面这些文件至少一组：

```text
output/prompt_extension_sweep/20260626-105843/ext1_beats/cli/2026-06-26/session_105855/combined.mid
output/prompt_extension_sweep/20260626-105843/ext2_beats/cli/2026-06-26/session_105924/combined.mid
output/prompt_extension_sweep/20260626-105843/ext3_beats/cli/2026-06-26/session_105953/combined.mid
output/prompt_extension_sweep/20260626-105843/ext4_beats/cli/2026-06-26/session_110021/combined.mid
```
