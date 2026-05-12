# Lekai Prompt-Continuation Realtime 功能研究报告

分支：`merge-lekai-prompt-stanley-system`  
研究对象：新加入的 `lekai_prompt_continuation` 两阶段实时伴奏功能  
日期：2026-05-12  

## 结论摘要

这个分支新增的核心功能不是 Stanley 系统本身，而是一条新的 Lekai 两阶段实时路径：

1. 前 `prompt_length_ticks` 的旋律先交给 Prompt 模型，一次性生成开头伴奏。
2. Prompt 伴奏被注入 Continuation 模型历史。
3. Continuation 模型逐拍追赶用户后续旋律。
4. 客户端轮询后端，等 catch-up ready 后再把可播放伴奏排进本地 `PlaybackScheduler`。

服务确实能跑起来，但“听感/结果和 offline version 差很多”是可以从代码解释的。最主要的问题不是单点崩溃，而是 realtime 路径在几个关键语义上没有和 offline 对齐：

- `combined.mid` 是前端调度后的可听输出，会丢弃或截断大量已经过去的 prompt/continuation 事件；offline 输出则是完整生成历史。直接比较二者会非常差。
- continuation 采样没有使用 offline 的 seeded `torch.Generator`，且 demo 默认参数和 offline 对比参数不一致。
- BPM / time signature 主要靠环境变量传入，`start/append` API 没有携带 piece-level metadata；默认 `CONTINUATION_META_MODE=legacy` 会让 continuation 使用 4/4、120 BPM 的历史行为。
- realtime continuation 每一拍都从“已解码 events”重新编码上下文；offline schedule 则保留真实 token 序列。只要 decode/re-encode 不是完全 token-idempotent，第二拍之后就可能发散。
- 当前测试主要证明 prompt stage 可对齐，没有真正证明 continuation raw history 或 audible output 与 offline 对齐。

所以，这个功能的方向是合理的，但目前更像“能跑的 realtime demo + prompt-stage 对齐验证”，还不是一个 offline-equivalent 的实时推理实现。

## 新功能范围

相对 `new_system_stanley`，该分支新增/修改的主要模块包括：

- `src/streammuse/application/services/prompt_continuation_realtime_service.py`
- `src/streammuse/infrastructure/inference/lekai_prompt_continuation/`
- `src/streammuse/infrastructure/inference/prompt_continuation_http_client.py`
- `src/streammuse/infrastructure/inference/server_lekai.py`
- `src/streammuse/infrastructure/inference/lekai_http_backend.py`
- `scripts/run_lekai_prompt_continuation_realtime_demo.sh`
- `scripts/run_cli_prompt_alignment_batch.sh`
- `scripts/prepare_and_compare_lekai_prompt_alignment.py`
- `scripts/compare_lekai_offline_realtime_raw.py`

CLI 入口用 `--model-name lekai_prompt_continuation` 切到新服务：

```python
# src/streammuse/presentation/cli/cli.py
if config.inference.model_name == "lekai_prompt_continuation":
    prompt_client = PromptContinuationHttpClient(...)
    service = PromptContinuationRealtimeService(
        input_source=input_source,
        prompt_client=prompt_client,
        output_sink=output_sink,
        tempo=tempo,
        scheduler=scheduler,
        prompt_length_ticks=int(config.inference.prompt_length_ticks),
        generation_interval_ticks=config.inference.generation_interval_ticks,
    )
else:
    service = RealTimeMusicService(...)
```

## 核心架构

整体调用链：

```text
streammuse-cli
  -> PromptContinuationRealtimeService
  -> PromptContinuationHttpClient
  -> server_lekai.py /prompt_continuation/*
  -> LekaiPromptContinuationBackend
  -> LekaiPromptContinuationEngine
  -> LekaiPromptContinuationScheduler
     -> LekaiPromptEngine
     -> LekaiContinuationEngine
        -> LekaiHttpBackend
```

服务端新增了专用 polling API：

```python
# src/streammuse/infrastructure/inference/server_lekai.py
@app.post("/prompt_continuation/start")
async def prompt_continuation_start(...):
    status = prompt_continuation_backend.start_prompt_catchup(...)

@app.post("/prompt_continuation/append_melody")
async def prompt_continuation_append_melody(...):
    status = prompt_continuation_backend.append_melody_events(...)

@app.get("/prompt_continuation/status")
async def prompt_continuation_status():
    return _scheduler_status_response(prompt_continuation_backend.scheduler_status())

@app.get("/prompt_continuation/playable")
async def prompt_continuation_playable():
    status = prompt_continuation_backend.scheduler_status()
    return PromptContinuationPlayableResponse(
        accompaniment=_accompaniment_response_events(
            prompt_continuation_backend.playable_accompaniment()
        ),
        status=_scheduler_status_response(status),
    )
```

## 客户端实时流程

客户端在 prompt window 结束时发送 `start`，之后每个 `generation_interval_ticks` 发送一次 append：

```python
# src/streammuse/application/services/prompt_continuation_realtime_service.py
def _maybe_enqueue_start(self, observed_until_tick: int) -> None:
    if self._start_enqueued:
        return
    if int(observed_until_tick) < self._prompt_length_ticks:
        return
    self._control_q.put(
        _ControlAction(
            kind="start",
            melody_events=list(self._prompt_events),
            observed_until_tick=self._prompt_length_ticks,
        )
    )
    self._start_enqueued = True
    self._last_append_observed_tick = self._prompt_length_ticks

def _maybe_enqueue_append(self, observed_until_tick: int) -> None:
    if not self._start_enqueued:
        return
    observed_until_tick = int(observed_until_tick)
    if observed_until_tick <= self._last_append_observed_tick:
        return
    if (observed_until_tick - self._prompt_length_ticks) % self._generation_interval_ticks != 0:
        return
    self._control_q.put(
        _ControlAction(
            kind="append",
            melody_events=list(self._pending_append_events),
            observed_until_tick=observed_until_tick,
        )
    )
```

值得注意：protocol worker 只有在 prompt 后至少发过一次 append 后才轮询 playable：

```python
# src/streammuse/application/services/prompt_continuation_realtime_service.py
if self._protocol_started and self._append_sent_after_prompt:
    status = self._client.status()
    if status.get("is_playback_ready"):
        accompaniment, playable_status = self._client.playable()
        self._playable_q.put((accompaniment, playable_status))
```

这会让最早的可播放窗口天然滞后，也会让短片段或纯 prompt 场景拿不到已经 ready 的结果。测试里甚至把这个行为固化为 `test_protocol_worker_does_not_fetch_playable_before_first_append`。

## 后端调度状态机

调度器用单 worker 线程串行执行模型调用，HTTP 线程可以继续 append melody：

```python
# src/streammuse/infrastructure/inference/lekai_prompt_continuation/scheduler.py
def _run_prompt_then_catchup(self, run_id: int) -> None:
    prompt_accompaniment = self._prompt_engine.generate_prompt_accompaniment(
        melody_events=prompt_melody_input,
        prompt_start_tick=0,
        prompt_length_ticks=prompt_length_ticks,
    )
    generated_acc_beats = self._prompt_engine.last_generated_acc_beats()
    actual_prompt_length_ticks = min(
        prompt_length_ticks,
        max(0, int(generated_acc_beats) * TIMESTEPS_PER_BEAT),
    )

    self._continuation_engine.inject_history(
        melody_events=melody_snapshot,
        accompaniment_events=prompt_accompaniment,
        injection_length_ticks=actual_prompt_length_ticks,
    )

    self._run_catchup_loop(run_id)
```

catch-up 条件是“伴奏比已观察旋律多一拍”：

```python
# src/streammuse/infrastructure/inference/lekai_prompt_continuation/catchup_state.py
def target_playable_accompaniment_beats(self) -> int:
    return self.melody_history_beats + self.playable_lookahead_beats

def beats_needed_for_playback(self) -> int:
    return max(0, self.target_playable_accompaniment_beats() - self.accompaniment_history_beats)
```

这个规则本身合理，因为伴奏只追到当前旋律长度时还没有“下一拍”可播放。

## Prompt 模型桥

Prompt engine 复刻了 Lekai prompt offline 的 `prepare_condition` 格式：

```python
# src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_engine.py
ts_token = self._tokenizer.encode_time_sig(metadata["time_signature_idx"])
bpm_token = self._tokenizer.encode_bpm(metadata["bpm"])
_, _, _, measure_beats = self._tokenizer._encode_measures(measures, metadata)

v = self._tokenizer.vocab
mel_parts = []
for beats in measure_beats:
    mel_parts.append(torch.tensor([v.bar_token_id], dtype=torch.long))
    for mel, _acc in beats:
        mel_parts.append(torch.tensor([v.beat_marker], dtype=torch.long))
        mel_parts.append(mel)

prefix = torch.tensor([v.bos_token_id, ts_token, bpm_token], dtype=torch.long)
prompt_tokens = torch.cat([prefix, torch.cat(mel_parts)])
```

现有脚本只严格检查了 prompt stage：

```bash
# scripts/run_cli_prompt_alignment_batch.sh
python scripts/prepare_and_compare_lekai_prompt_alignment.py \
  --npz-id "$id" \
  --output-root "$OUT_ROOT" \
  --compare \
  --cli-prompt-json "$session_dir/prompt_continuation_prompt_history.json"
```

这能证明 prompt history 的事件 SHA 可以和 RT prompt reference 对齐，但不能证明 continuation 或最终 audible output 对齐。

## Continuation 模型桥

Continuation realtime 没有直接执行 offline 的完整 schedule，而是在每个目标 beat 前重新构造 prompt：

```python
# src/streammuse/infrastructure/inference/lekai_http_backend.py
def build_standard_offline_prompt(target_beat: int) -> torch.Tensor:
    seq: List[torch.Tensor] = list(prefix)
    ...
    for beat in range(start_beat, target_beat):
        if beat == start_beat or beat % measure_beats == 0:
            seq.append(torch.tensor([LEKAI_BAR_TOKEN], dtype=torch.long))
        seq.append(torch.tensor([LEKAI_BEAT_TOKEN], dtype=torch.long))

        acc_tokens, accompaniment_active = self._encode_beat_tokens(
            events=accompaniment_context_events,
            beat_start_tick=beat_start_tick,
            active_pitches=accompaniment_active,
            end_marker=LEKAI_ACC_END_TOKEN,
        )
        seq.append(acc_tokens)

        mel_tokens, melody_active = self._encode_beat_tokens(
            events=self._melody_history,
            beat_start_tick=beat_start_tick,
            active_pitches=melody_active,
            end_marker=LEKAI_MEL_END_TOKEN,
        )
        seq.append(mel_tokens)

    if target_beat == start_beat or target_beat % measure_beats == 0:
        seq.append(torch.tensor([LEKAI_BAR_TOKEN], dtype=torch.long))
    seq.append(torch.tensor([LEKAI_BEAT_TOKEN], dtype=torch.long))
    return torch.cat(seq, dim=0)
```

offline continuation 的 schedule 则是：

```python
# src/streammuse/infrastructure/inference/lekai_continuation_model/my_tokenizer.py
for beats in measure_beats:
    steps.append(GenerationStep("inject", torch.tensor([v.bar_token_id], dtype=torch.long)))

    for acc, mel in beats:
        steps.append(GenerationStep("inject", torch.tensor([v.beat_marker], dtype=torch.long)))
        if beat_idx < gt_prefix_beats:
            steps.append(GenerationStep("inject_gt", acc))
        else:
            steps.append(GenerationStep("generate"))
        steps.append(GenerationStep("inject", mel))
        beat_idx += 1
```

两者结构看起来接近，但 realtime 通过 events 反复 decode/re-encode 历史；offline 在同一次 `generated` tensor 中保留真实 token 序列。这个差异非常重要。

## 为什么 realtime 和 offline 差很多

### 1. `combined.mid` 不是 offline-equivalent 输出

`combined.mid` 记录的是前端调度后真正输出过的事件，不是后端生成的完整 raw history。Prompt-continuation 在 catch-up ready 前保持静默，所以第一次拿到 playable 时，大量 prompt 事件已经在当前 tick 之前。

默认 strict 模式会丢掉过去的事件，只保留未来或持续到现在的音：

```python
# src/streammuse/application/services/prompt_continuation_realtime_service.py
events_to_schedule, dropped_past, clipped_sustains = self._clip_playable_to_current_tick(
    accompaniment,
    current_tick=int(current_tick),
)
```

recover-late 模式也会按策略丢弃太旧的 `note_on`：

```python
# src/streammuse/application/services/prompt_continuation_realtime_service.py
if (
    self._recover_late_max_ticks is not None
    and current_tick - event_tick > self._recover_late_max_ticks
    and event.event_type == EventType.NOTE_ON
    and int(event.velocity) > 0
):
    dropped_too_late_note_on += 1
    self._scheduled_model_event_keys.add(event_key)
    continue
```

demo 默认：

```bash
# scripts/run_lekai_prompt_continuation_realtime_demo.sh
export RECOVER_LATE_EVENTS=${RECOVER_LATE_EVENTS:-1}
export RECOVER_LATE_MAX_TICKS=${RECOVER_LATE_MAX_TICKS:-4}
```

这意味着：如果 first playable 在 tick 36 之后才到，tick 0-31 的 prompt note_on 基本都会被认为太旧而丢掉。这样听起来当然会比 offline 少很多开头伴奏，甚至只剩零碎 note_off 或后续短音。

正确比较对象应该是：

- prompt stage：`prompt_continuation_prompt_history.json/mid`
- prompt + continuation raw：`prompt_continuation_raw_history.json/mid`
- live audible policy：`combined.mid`

如果拿 `combined.mid` 去和 offline full output 比，就混入了“前端实时播放策略”的损失。

### 2. continuation 采样随机性没有和 offline 对齐

offline continuation 使用显式 `torch.Generator`：

```python
# src/streammuse/infrastructure/inference/lekai_continuation_model/model.py
def _sample_token(..., generator: Optional[torch.Generator] = None):
    ...
    return torch.multinomial(probs, num_samples=1, generator=generator)
```

对比脚本也显式创建并重置 generator：

```python
# scripts/compare_lekai_offline_realtime_raw.py
generator = torch.Generator(device=device if str(device).startswith("cuda") else "cpu")
generator.manual_seed(args.seed)
...
acc_beats, generated = offline_model.generate_accompaniment(
    ...
    generator=generator,
)
```

realtime continuation 走的是旧 `sample_token`，没有 generator 参数：

```python
# src/streammuse/infrastructure/inference/lekai_model/generation_utils.py
probs = torch.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

而 `LEKAI_SEED` 在 `LekaiHttpBackend` continuation 里没有被使用。Prompt engine 会在 prompt 生成前重置全局 seed，prompt 生成又会消耗随机数；offline 则通常在 prompt 和 continuation 阶段分别 reset seed/generator。只要 `RT_TOP_K > 1` 或 `top_p` 允许多个 token，raw continuation 就很难稳定复现 offline。

### 3. demo 参数和 offline 对齐参数不一致

demo 当前默认：

```bash
# scripts/run_lekai_prompt_continuation_realtime_demo.sh
export RT_TEMPERATURE=${RT_TEMPERATURE:-0.8}
export RT_TOP_K=${RT_TOP_K:-50}
export RT_TOP_P=${RT_TOP_P:-0.98}
export RT_REPETITION_PENALTY=${RT_REPETITION_PENALTY:-1.2}
```

raw offline comparison 脚本默认：

```python
# scripts/compare_lekai_offline_realtime_raw.py
parser.add_argument("--rt-temperature", type=float, default=0.8)
parser.add_argument("--rt-top-k", type=int, default=1)
parser.add_argument("--rt-top-p", type=float, default=0.95)
parser.add_argument("--rt-repetition-penalty", type=float, default=1.2)
```

`top_k=50/top_p=0.98` 和 `top_k=1/top_p=0.95` 是完全不同的 continuation 采样行为。前者更随机，用来听 demo；后者更适合 deterministic alignment。不能用前者的 `combined.mid` 期待和后者的 offline reference 接近。

### 4. BPM / time signature 没有作为请求级 metadata 传递

Prompt API schema 只带 melody、prompt length、interval 等信息：

```python
# src/streammuse/infrastructure/inference/server_lekai.py
class PromptContinuationStartRequest(BaseModel):
    melody_notes: List[MelodyNoteEvent]
    prompt_length_ticks: int = Field(gt=0)
    generation_interval_ticks: int = Field(gt=0)
    observed_until_tick: Optional[int] = Field(default=None, ge=0)
    inference_mode: str = "sliding_window"
    model_name: str = "lekai_prompt_continuation"
    checkpoint_path: Optional[str] = None
```

Prompt engine 用环境变量取 BPM/拍号：

```python
# src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_engine.py
metadata = {
    "time_signature_idx": time_signature_idx,
    "bpm": self._env_int("LEKAI_PROMPT_BPM", self._env_int("LEKAI_DEFAULT_BPM", 120)),
    "num_measures": num_bars,
}
```

Continuation backend 也用环境变量：

```python
# src/streammuse/infrastructure/inference/lekai_http_backend.py
effective_bpm = self._request_bpm if self._request_bpm is not None else int(os.environ.get("LEKAI_DEFAULT_BPM", "120"))
time_signature_idx = int(os.environ.get("LEKAI_TIME_SIGNATURE_INDEX", "4"))
```

批量脚本默认的 continuation metadata mode 是 `legacy`：

```bash
# scripts/run_cli_prompt_alignment_batch.sh
CONTINUATION_META_MODE=${CONTINUATION_META_MODE:-legacy}
```

legacy 模式注释说明就是使用后端默认的 `LEKAI_TIME_SIGNATURE_INDEX=4` 和 `LEKAI_DEFAULT_BPM=120`。如果 piece 的真实 NPZ metadata 不是这个组合，continuation prefix 的 `[TS, BPM]` token 就和 offline 不一致，模型输出会立刻偏离。

`CONTINUATION_META_MODE=offline` 能让脚本给服务端设置 raw NPZ 的 `time_signature_idx` 和 `bpm`，但这是通过“每首歌重启 server + 环境变量”实现的，不是 API 设计本身保证的。普通用户直接跑 CLI 时很容易漏掉这一步。

### 5. realtime continuation 丢掉了 exact token history

offline continuation 在同一条 `generated` tensor 中持续追加 token：

```python
# src/streammuse/infrastructure/inference/lekai_continuation_model/model.py
generated = initial_tokens.unsqueeze(0).to(device)
...
elif step.action == "generate":
    beat_tokens, generated = self._generate_one_beat(...)
    acc_beats.append(beat_tokens)
```

realtime 每拍生成后会 decode 成 pianoroll，再变成 events，然后下一拍重新 encode：

```python
# src/streammuse/infrastructure/inference/lekai_http_backend.py
beat_pianoroll = self._decode_acc_beat_tokens(generated_beat_tokens)
beat_events, next_active = self._converter.pianoroll_to_events(...)
generated_events.extend(normalized_beat_events)
accompaniment_context_events.extend(normalized_beat_events)
```

这在音乐内容层面可能接近，但不保证 token 层面一致。只要模型生成了 tokenizer 可解码但非规范的 token 序列，或者 sustain/boundary event 在 decode/re-encode 后改变压缩表示，下一拍 prompt token 就和 offline `generated` tensor 不同，后续会递归放大差异。

如果目标是严格对齐 offline，realtime continuation 应该保存并复用 exact acc beat tokens，而不应该只保存 events。

### 6. prompt length 对 3/4 等拍号依赖外部脚本修正

Prompt engine 会按 time signature 把 prompt window 扩展到完整小节：

```python
# src/streammuse/infrastructure/inference/lekai_prompt_continuation/prompt_engine.py
beats_per_bar = self._measure_beats_from_time_signature_idx(time_signature_idx)
timesteps_per_bar = TIMESTEPS_PER_BEAT * beats_per_bar
num_bars = max(1, int(np.ceil(prompt_length_ticks / timesteps_per_bar)))
window_ticks = num_bars * timesteps_per_bar
```

批量脚本里 `PROMPT_BEATS=auto` 才会对 3/4 使用 6 beats：

```bash
# scripts/run_cli_prompt_alignment_batch.sh
if [[ "$prompt_beats" == "auto" ]]; then
  if [[ "$beats_per_bar" == "3" ]]; then
    prompt_beats=6
  else
    prompt_beats=8
  fi
fi
```

但 CLI 默认仍是 `--prompt-length-ticks 32`。如果用户直接对 3/4 曲目用默认值，prompt 模型会构造 9 beats 的完整 3 小节条件，再只解码/注入 8 beats，和 offline `prepare_condition(num_bars=2)` 的 6 beats 设定不一致。

## 次要代码风险

### `/generate_accompaniment` 对 prompt backend 的签名不匹配

`server_lekai.py` 的通用路由会把 `bpm`、`input_file` 传给 selected backend：

```python
# src/streammuse/infrastructure/inference/server_lekai.py
accompaniment, timings = selected_backend.generate(
    ...
    checkpoint_path=request.checkpoint_path,
    bpm=request.bpm,
    input_file=request.input_file,
)
```

但 `LekaiPromptContinuationBackend.generate()` 没有 `bpm` 和 `input_file` 参数：

```python
# src/streammuse/infrastructure/inference/lekai_prompt_continuation/backend.py
def generate(
    self,
    melody_events: list[EventPayload],
    generation_start_tick: int,
    generation_length_frames: int,
    generation_interval_ticks: int,
    prompt_length_ticks: Optional[int],
    inference_mode: str,
    model_name: str,
    checkpoint_path: Optional[str],
) -> tuple[list[EventPayload], TimingPayload]:
```

这不影响专用 `/prompt_continuation/*` realtime path，但会影响旧 `/generate_accompaniment` + `model_name=lekai_prompt_continuation` 的兼容路径。现有测试声称这条路由成功，但我没有执行 pytest；从静态代码看这里会触发 unexpected keyword argument。

### Engine 里有两套 catch-up state

`LekaiPromptContinuationEngine` 自己有 `_catchup_state`，scheduler 也有 `_catchup_state`。专用 polling API 用 scheduler status，因此 realtime path 主要没问题；但 `engine.catchup_status()` / `runtime_info()` 里的 `catchup_*` 可能和 scheduler 的真实状态不一致，容易误导调试。

## 现有验证的边界

新增测试和脚本覆盖了不少 plumbing：

- start/append/control queue 行为；
- scheduler catch-up 规则；
- prompt stage SHA 对齐；
- recover-late 策略；
- HTTP endpoint schema。

但缺少这些关键验证：

- continuation raw history 和 offline continuation 的严格对齐测试；
- seed/generator 对齐测试；
- metadata 通过 API 传递的测试；
- `combined.mid` 相对 raw history 的 dropped/recovered coverage 指标；
- 多拍号、多 BPM、不重启 server 的多曲目测试；
- exact token history vs event re-encode 的回归测试。

## 建议修复路线

### P0：先明确比较目标

不要再直接用 offline full output 对比 `combined.mid`。建议分三层指标：

- `prompt_continuation_prompt_history.*` 对比 prompt offline reference；
- `prompt_continuation_raw_history.*` 对比 prompt+continuation offline raw；
- `combined.mid` 只评估 realtime audible policy，例如 dropped note_on 数、late recovered 数、首次出声 tick、静默时长。

### P1：实现 offline-compatible continuation mode

为了验证模型桥是否正确，需要一个 deterministic 模式：

- `RT_TOP_K=1` 或 `temperature=0`；
- prompt 和 continuation 分别使用显式 `torch.Generator(seed)`；
- continuation metadata 使用 raw NPZ `time_signature_idx` 和 raw BPM；
- 每个 session 保存 exact generated acc beat tokens，下一拍直接注入 token history，而不是 events decode/re-encode。

这个模式不一定是最终 live demo 参数，但必须存在，否则很难判断差异来自模型、采样还是播放策略。

### P2：把 metadata 放进 API，而不是靠环境变量

给 `/prompt_continuation/start` 增加字段：

- `bpm`
- `time_signature_idx`
- `beats_per_bar`
- `prompt_beats`
- 可选 `seed`

然后把这些字段传到 `LekaiPromptEngine` 和 `LekaiHttpBackend`。这样 server 不需要每首歌重启，也不会因默认 4/4、120 BPM 让输出偏离 offline。

### P3：重做 playable contract

当前 `/playable` 返回完整 accompaniment history，客户端再在本地丢旧事件。更好的 contract 是后端或客户端显式维护 cursor：

- `playable_since_tick`
- `already_sent_event_count`
- `schedule_tick`
- `raw_tick`

后端可以只返回“新增且还值得播放”的 chunk；raw history 仍保留完整生成结果用于 debug。

### P4：分离 debug render 和 live audible output

建议 session 输出明确分两类：

- `raw_history.mid`：模型实际生成的 offline-style 历史；
- `combined_live.mid`：真实前端调度后听到/输出的内容；
- `combined_rendered.mid`：把 raw history 和 user melody 按原 tick 渲染，仅用于离线听感对比。

现在的 `combined.mid` 名字容易让人误以为它应该等同 offline generated result。

## 最终判断

这个分支的 prompt-continuation 方向是对的：用 prompt 模型解决开头上下文不足，再用 continuation 模型接续，是合理的架构。但当前实现还没有把 offline 的关键语义完整搬进 realtime：

- Prompt stage 相对成熟，并且已经有 SHA 对齐脚本。
- Continuation stage 只是结构近似 offline，随机性、metadata、token history 都可能导致发散。
- Audible scheduling 还会额外丢掉大量旧事件，让听感和 raw/offline 进一步拉开。

因此，“服务能跑但效果很差”不是偶然现象，而是当前设计自然会出现的结果。下一步应先做 deterministic raw-history 对齐，再谈 live playback policy 的音乐性。
