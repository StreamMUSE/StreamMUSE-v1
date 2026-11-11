# Inference Optimization Handoff

Final source-inspected handoff: 2026-11-11. **Software acceptance passed; user confirmed good playback but reported voice changes between parts.**
See the [execution ledger](../reports/2026-11-11-inference-milestone-progress.md) for iteration history,
the [milestone contract](../research/2026-11-10-inference-page-milestone.md#13-required-final-inference-engineer-handoff),
and the [short user guide](inference-page-guide.md) for operation.

## Navigation

- [Runtime and ownership](#runtime-and-ownership)
- [Source map and editing scope](#source-map-and-editing-scope)
- [Execution flows](#execution-flows)
- [Contracts and parameters](#contracts-and-parameters)
- [Measurements and exports](#measurements-and-exports)
- [Tests and deployment](#tests-and-deployment)
- [Verification status](#verification-status)

## Runtime and ownership

`http://127.0.0.1:8018/inference` exposes cyclic exchange and continuous steering; `/` remains Rap Lab.
Both attach to the same FastAPI app/runtime/session; opening a page does not Start, Reset, or create a session.
These staging controllers can become official here: **no separate branch, server, or application session is required**.

```text
Browser /inference -> Mac FastAPI :8018 -> RuntimeInferenceBridge -> InferenceService
  -> RealInferenceBackend -> SSH localhost :18721 -> H200 renderer / v3 service
       -> Qwen :18711 (physical GPU 1): candidate generation
       -> MOSS :18730 + MMS alignment (physical GPU 2): source vocals / onsets
  <- WebSocket selected lyrics + HTTP audio package
  -> Mac decode -> local warp / resample / mix -> AudioSpan -> SpanBuffer
  -> Mac PortAudio callback -> selected Mac output device
```

Mac Python owns audio; the browser is a controller/projector, not an audio player or musical clock.
SSH socket: `/tmp/streammuse-inference-20261111.sock`; Mac forwards: 18711/18721. MOSS 18730 stays H200-local.
Current H200 root: `/data/home/Andrew.Yang/StreamMUSE/deploy/rap_inference_20261111`.
Services were left running and healthy at 2026-11-11T10:08:02Z; this is a point-in-time check, not a health guarantee.

`RuntimeInferenceBridge` shares `runtime._lifecycle_lock` with the service. Its claim
requires stopped legacy playback and no legacy startup/settings mutation; it resets
the legacy controller internally on first claim, without creating another app session.
Legacy Start cannot compete with active inference; releasing inference suspends its
native stream outside the lifecycle lock. Global Stop/Reset route to the actual owner.
Mode-specific legacy writes are guarded while inference is active. Device opening and
settings transitions are fenced too; page separation alone is not an ownership guarantee.

## Source map and editing scope

**Everything serving either inference-page mode is editable**: UI, state machines,
prompts, generation and candidate budgets, validation/ranking, model/serving integration,
TTS, alignment, local warp/mixing, transport, buffering, scheduling, audio output,
configuration, and telemetry. There is no protected backend or audio black box.
The adapter interface supports focused experiments; it does not restrict edit scope.
Shared changes must preserve legacy behavior or use a mode-scoped guard and corresponding
regressions. Update contracts, capability declarations, metrics documentation, and tests together.

Legend: **I** = both new modes; **C** = cyclic; **S** = steering; **L** = also legacy, requiring guard/regressions.
Paths are repository-relative; symbols describe the inspected implementation.

| Scope | File / symbols | Responsibility and callers |
| --- | --- | --- |
| I | [domain/rap/inference.py](../../src/streammuse/domain/rap/inference.py): `InferenceRequest`, `AudioSpan` | Versioned musical request/decoded PCM contracts; used by service, buffer, adapter. |
| I/C/S | [inference/service.py](../../src/streammuse/application/rap/inference/service.py): `InferenceService` | `configure`, `start`, `stop`, `reset`; C: `_begin_user`, `tap`, `_lease_check`; S: `steer`, `_extend_steering`; shared `_request`, `accept_span`, `audio_callback`, snapshots/exports. |
| I | [inference/delivery.py](../../src/streammuse/application/rap/inference/delivery.py): `SpanBuffer` | Validates/holds PCM; `prefix_ticks`, `select`, `discard_before`; immutable safe-span choice at callback commitment. |
| I | [inference/metrics.py](../../src/streammuse/application/rap/inference/metrics.py): `latency_summary`, `coverage_summary` | Pure percentile and duration-weighted formulas, called by service projection. |
| I/L | [inference/runtime_bridge.py](../../src/streammuse/application/rap/inference/runtime_bridge.py): `RuntimeInferenceBridge`, `install_inference` | Exclusive shared-runtime routing and stopped-only audio capability projection. |
| I | [inference_backend.py](../../src/streammuse/infrastructure/rap/inference_backend.py): `RealInferenceBackend` | `submit`/`cancel`, `_flows`, `_run_session`, `_fetch_prepare`, `_deliver`; bounded v3 adapter, not the legacy playback controller. |
| I/L | [inference_output.py](../../src/streammuse/infrastructure/rap/inference_output.py): `InferenceAudioOutput`, `TimedInferenceStream` | Native callback and asynchronous recording; WAV-only clock driver. Reuses native factory/lock in [audio_output.py](../../src/streammuse/infrastructure/rap/audio_output.py). |
| I/L | [runtime.py](../../src/streammuse/application/rap/runtime.py): `RapAudioDemoDependencies` | Shared lifecycle, settings, Start/Stop/Reset and close ownership. |
| I/L | [rap_demo/cli.py](../../src/streammuse/presentation/rap_demo/cli.py): `inference_mix`, `inference_preparer`, `inference_silence`, `prepare_inference` | Composition installs inference only for `moss_aligned_remote` plus protocol `v3`; creates request-local preparers and shared device/mix snapshots. |
| I/L | [server.py](../../src/streammuse/presentation/rap_demo/server.py), [inference_api.py](../../src/streammuse/presentation/rap_demo/inference_api.py): `create_inference_router` | Same app `/inference`; `/api/inference/*` resolves runtime service dynamically and offloads synchronous methods with `asyncio.to_thread`. |
| I/C/S | [inference.html](../../src/streammuse/presentation/rap_demo/static/inference.html), [rap-inference.js](../../src/streammuse/presentation/rap_demo/static/js/rap-inference.js), [CSS](../../src/streammuse/presentation/rap_demo/static/css/rap-inference.css) | `mount`, `createOwner`, `readConfig`, `layoutSpans`: polling, controls, capture, versions, spans, metrics. No browser playback scheduler. |
| I/L | [rap-inference-status.js](../../src/streammuse/presentation/rap_demo/static/js/rap-inference-status.js), [index.html](../../src/streammuse/presentation/rap_demo/static/index.html) | Original-page inference ownership/status integration; retain original controls. |
| L | [lyric_stream_client.py](../../src/streammuse/infrastructure/rap/lyric_stream_client.py): `LyricStreamConnection`, `HttpBarAudioSource.fetch` | Capability negotiation, selected-lyric events, bounded audio retrieval and package decoding. |
| L | [rap_lyric_stream_server.py](../../src/streammuse/presentation/rap_lyric_stream_server.py): `LyricStreamService`, `mount_lyric_stream_routes` | Creates bounded remote sessions; WS `/v1/rap/sessions/{session_id}`, GET bar audio. These are request-owned remote sessions, not separate UI sessions. |
| L | [lyric_session.py](../../src/streammuse/application/rap/lyric_session.py): `LyricStreamSession` | `_writer_loop` and `_render_loop` prepare bounded future lyrics/audio; `fetch_audio` serves selected revision. |
| L | [lyric_writer.py](../../src/streammuse/application/rap/lyric_writer.py): `SequentialLyricWriter.write`, `_record_batch`, `LyricWriterPolicy` | Candidate waves, filtering, prosody checks, ranking, selected lyric/provenance. |
| L | [generators.py](../../src/streammuse/infrastructure/rap/generators.py): `IndependentChoiceCandidateGenerator.generate`, `_build_independent_messages`, `_parse_candidate_lines`; [flow_prompt.py](../../src/streammuse/infrastructure/rap/flow_prompt.py) | Actual prompt construction and response parsing; not an untouchable external prompt service. |
| L | [local_chat_client.py](../../src/streammuse/infrastructure/inference/local_chat_client.py): `LocalChatModelClient`; [prosody.py](../../src/streammuse/infrastructure/rap/prosody.py): `CmuProsodyAnalyzer`; [scoring.py](../../src/streammuse/application/rap/scoring.py): `rank_candidates` | Qwen HTTP choices, syllable analysis, validity/score selection. |
| L | [rap_render_server.py](../../src/streammuse/presentation/rap_render_server.py): `_compose_real_worker` | H200 model clients, writer factory, MOSS/MMS warmup, renderer, producer identity and service composition. |
| L | [bar_audio.py](../../src/streammuse/application/rap/bar_audio.py): `MossBarAudioRenderer.render`, `bar_render_request`, `BarAudioPreparer.prepare` | H200 source synthesis/alignment package; Mac target warp, leading-rest preservation, resampling, drums/vocal mix. |
| L | [moss_aligned_phrase.py](../../src/streammuse/infrastructure/rap/moss_aligned_phrase.py), [sglang_moss_tts.py](../../src/streammuse/infrastructure/rap/sglang_moss_tts.py), [mms_forced_alignment.py](../../src/streammuse/infrastructure/rap/mms_forced_alignment.py) | `MossAlignedPhraseRenderer.render_source`, `SglangMossSynthesizer.synthesize`, `MmsForcedAligner.load`: GPU 2 synthesis/alignment. |
| L | [phrase_warp.py](../../src/streammuse/infrastructure/rap/phrase_warp.py), [chunk_audio.py](../../src/streammuse/application/rap/chunk_audio.py), [bar_package.py](../../src/streammuse/infrastructure/rap/bar_package.py) | `RubberBandPhraseWarper.warp`, `mix_audio_stems`, `decode_bar_package`: local warp, conversion/mixing, Opus packaging. |

## Execution flows

### Startup and a cyclic response

1. [launch_mac.py](../../scripts/experiments/inference_milestone/launch_mac.py) launches the existing
   CLI with v3, Opus, composite output and a real device. It refuses an existing recorded
   client or occupied port. CLI creates one session and installs the bridge/service/backend.
2. Page attachment polls state. Explicit Start claims output, advances epoch/input version,
   clears old capture/queued audio, and enters USER. Its first callback establishes musical
   position zero and plays drums; default duration is four beats, not a browser timer.
3. Focused Space events include owner/session/epoch/cycle/event IDs and estimated phase time.
   Leading rests are retained. At USER end, captured ticks repeat over the MODEL duration
   and truncate at its boundary. Empty capture repeats USER with `No rhythm captured`, no request.
4. Nonempty capture enters silent WAIT and releases `_request(0, model_ticks, flow)`.
   Example geometry at 60 BPM: USER 16 ticks, MODEL 32 ticks; taps `(2, 6, 10, 14)`
   become `(2, 6, 10, 14, 18, 22, 26, 30)`. WAIT has no configured length or beat quantization.
5. `submit` snapshots the request and queues bounded worker work. `_flows` makes one captured
   template per bar; an explicitly empty bar gets local drums-only declared-silence PCM.
   Each contiguous non-silent run opens a separate `bounded_turns_v1` WS session with
   `start_bar`, `end_bar_exclusive`, `turn_id`, segments/history, and full bounded lyric/audio lead.
6. H200 writes sequential selected lyrics using Qwen choices; validity includes text shape,
   duplicate/recent-word filtering, prosody and ranking. The writer publishes `BarLyricRecord`
   before rendering finishes. MOSS produces source vocals, MMS estimates onsets, and the server
   packages source audio. Continuation uses the previous source when available within that session.
7. Mac receives WS `lyric` events, verifies identity/topic/flow/revision, and fetches HTTP audio
   with `accept_older=False`. Fetch workers may finish out of order. A URL or selected lyric is
   not readiness: package decode, identity checks, local warp/resample/mix must finish first.
8. `_deliver` emits a 16-tick `AudioSpan`, e.g. output `<request_id>:bar:0`, with current
   epoch/version, mixed float32 PCM, syllables, stage durations and no interior safe splits.
   `accept_span` validates request coverage and decoded frames. Only a contiguous current-version
   prefix from tick zero satisfies startup; a later bar alone cannot start MODEL.
9. The next schedulable callback enters MODEL, resets its musical frame cursor, and starts drums
   at zero even with a later vocal onset. Default real prefix is one bar, not the whole two-bar turn.
   `SpanBuffer.select` commits the chosen safe range. Late future vocals leave drums-only missing
   coverage; time never stretches and expired unsplittable bars never join midway through a phoneme.
10. At MODEL end, remaining work is canceled/fenced and the next USER starts automatically.
    Response audio cannot spill into USER. No manual rearming or Reset is required.

### Continuous steering

Start enters silent PREPARING, then MODEL on the same decoded-prefix/callback rule; startup
latency is separate from edit latency. `_extend_steering` requests more coverage as the current
version's remaining horizon reaches half `model_beats * ticks_per_beat`.
`steer({topic, template_id})` updates intent/version, cancels pending requests and targets the
first quantum boundary at or beyond both the playback cursor and `frozen_until`.
During PREPARING it replaces the prefix from zero. During MODEL it keeps accepted old PCM.

At a mutable safe boundary, the newest accepted input version/revision wins. Already committed
audio is retained to its safe endpoint. If replacement is late, compatible old prepared output
fills its original slot and is explicitly **stale context**; it is not relabeled as new-context
success, implicitly retimed to a new flow, or looped forever. Without either version, drums-only
missing coverage is counted. Tempo/meter/format stay fixed while active, and new spans are
epoch/coverage/format validated. Requested versus playing versions expose actual input use,
not semantic compliance with the prompt.

### Concurrency, cancellation and browser ownership

- HTTP methods execute off the ASGI loop. The service and callback share a lifecycle lock;
  snapshots freeze mutable containers under that lock and deep-copy outside it.
- Real backend defaults: four request workers, three fetch/preparation workers, 256 queued
  messages per remote session. Network, decoding and generation run outside the audio callback;
  the callback may submit/cancel bounded work but does not wait for its remote result.
- Stop in WAIT/PREPARING is immediate locally. USER/MODEL Stop finishes the current bar or
  phase boundary. Start after Stop creates a new epoch, preserves configuration/latest intent
  and performed lyric history, and does not reuse invalidated audio/capture. Reset clears
  inference history, queues, requests and metrics; it is not required for normal restart.
- Late/superseded spans and backend events are fenced by request state, epoch, input version
  and phase. Cancel sets a local event; cleanup sends best-effort WS Stop and closes owned
  transports. `client_fenced` is **not confirmed remote/GPU cancellation**; close waits for workers.
- Cyclic capture has one browser owner: sessionStorage token plus browser-lock arbitration
  (BroadcastChannel fallback), heartbeat every 2 seconds and a 6-second lease. Tap retries
  deduplicate by event ID; wrong owner/session/epoch/cycle and invalid time are rejected.
- Owner loss stops WAIT immediately or completes the current USER/MODEL phase before stopping.
  Continuous playback has no capture lease and survives navigation/disconnection. Refresh
  reattaches without Start or replay; another tab can monitor and use global Stop/Reset.

## Contracts and parameters

Turn length, generation, delivery and commitment units are independent. Generic contracts use explicit resolution:
`frame(tick) = round(tick * sample_rate * 60 / bpm / ticks_per_beat)`.
Packets/tokens are not musical ticks. Fine-grained fixtures exercise the same contract, **not real beat/tick generation**.

| Surface | Current implementation / restriction |
| --- | --- |
| Request | Session/epoch/request/input version, start/duration ticks, BPM/meter/resolution, topic/template, captured flow, history and config. Request IDs are `<session>:<epoch>:<sequence>`. |
| Wire schemas | [lyric_stream.py](../../src/streammuse/domain/rap/lyric_stream.py): `streammuse.rap_lyric_stream.v3`, `streammuse.rap_bar_audio.v3`; real adapter requires `bounded_turns_v1`. |
| Audio result | Request/output IDs, epoch/version/revision, coverage, sample rate/channels/nonempty float32 PCM, declared silence, safe split offsets, text/syllables, stage timings, final flag and first intended vocal tick. |
| Validation | Positive integer geometry, finite JSON/PCM, complete interleaved frames, 64 MiB PCM bound, exact mapped frame count, ordered interior safe splits, stable revision coverage, duplicate identity idempotence. `final` alone is not complete contiguous readiness. |
| Real adapter | 4/4, four ticks/beat; bar-aligned starts/durations only. Generation/delivery/commit/minimum prefix are one bar = 16 ticks; guard 0 ms, maximum immutable lookahead 16 ticks, no interior splits. Maximum request 64 bars. |
| Service settings | `mode=cycle` or `steering`; BPM 30-240; integer beat/resolution values 1-64. Real turns must be whole bars. Defaults: 60 BPM, USER four beats, MODEL eight beats. |
| UI controls | Mode/BPM/turn settings apply while inactive via Apply settings. Topic/prompt and flow use Apply steering, including live steering mode. Candidate budget/concurrency are disabled because the adapter does not advertise them as configurable. |
| Audio controls | Shared output-device/drum/vocal settings are available only while stopped and capability-supported. They apply at next Start; no live inference gain/device editing. |
| Backend budget | Constructor timeout 90 s; CLI overrides it with `--rap-render-startup-timeout`, default 120 s. Fetch uses 1000 ms server waits plus bounded retry polling. Timeout is an error, not an initial missed musical deadline. |
| Writer/serving | Deployment uses 32 initial candidates; policy defaults allow three rescue waves of 16, minimum three valid, minimum score 0, couplet rhyme. Realtime generator: temperature 1.0, maximum 32 tokens/choice. |
| Audio deployment | Opus transfer, 48 kHz Mac output, `all_onsets_r3` local warp, drums 0.5/vocals 0.8, chords off in launcher. H200 source renderer baseline uses `gentle_sparse_r3`; do not confuse it with local warp selection. |

The service rejects partial-bar turns/BPM outside 30-240 despite broader HTML limits; there is no beat-generation toggle.
Control bodies are bounded to 64 KiB strict JSON; invalid values map to 422, ownership/state
conflicts to 409, connection errors to 503. Exports are GET `/api/inference/export?format=json|csv`.

## Measurements and exports

Local event times are Mac monotonic milliseconds; server durations and PortAudio clock fields are separate.
Never subtract H200 monotonic time from Mac/browser time. Fetch/decode includes wait/transport/decode,
not pure network time. Warp is a preparation substage; timings are not additive by default.

| Field | Exact interpretation / caveat |
| --- | --- |
| `ttft_ms`, `llm_ttft_ms` | Null/unavailable: v3 observes selected complete lyrics, not first content tokens. Total request time is not TTFT. |
| `first_valid_candidate_ms` | Request release to Mac receipt of the first server decision reporting valid candidates; not the server's precise first-valid instant. |
| `latency_ms` / first audio ready | First accepted decoded span minus release; may be out of order and is not necessarily a playable prefix. |
| `playable_prefix_ready_ms` | Release to minimum contiguous decoded coverage from request start; initial MODEL needs current-version coverage from zero. |
| `complete_latency_ms` | Release to full contiguous requested coverage, independent of packet count or final-event arrival order. |
| `response_start_latency_ms` | Cyclic request release at capture closure to MODEL callback start. No initial deadline exists; this is WAIT latency. |
| `startup_latency_ms` | Continuous initial request release to MODEL callback start, not a steering edit response. |
| `ready_to_start_overhead_ms` | MODEL callback start minus client prefix-ready time. |
| `steering_playback_latency_ms` | Accepted edit to first callback-performed span using that version. Not a semantic-quality score. |
| `first_vocal_onset_ms` | Intended onset mapped onto callback musical time, relative to explicit per-request `first_vocal_origin_ms` (`steering_edit` or `request_released`); not acoustically detected. |
| `deadline_margin_ms` | Future committed span deadline minus ready time. Initial frame-zero deadline is null; late/missing coverage must not acquire a fictional completion time. |
| Candidate counts | Requested/valid counts come from selected-lyric provenance, keyed by decision ID; they are not total LLM work/cost. Parsed/evaluated counts remain null when absent; the full candidate ledger is not exposed here. |

Scheduled/observed start fields currently share a callback timestamp, not independent scheduling/DAC measurements.
`audio.last_callback` exposes PortAudio current/DAC time/status where supplied; neither that nor a WAV proves acoustic output.

Eligible coverage is MODEL frames only. Missing rate = missing duration / eligible duration;
stale rate = retained old-context duration / eligible duration. USER and silent WAIT/PREPARING
are not missing MODEL vocals. Declared intentional silence is valid coverage. This adapter
declares no generated fallback, so fallback coverage is zero by policy, separate from missing
drums-only coverage; adapters without measured fallback policy project null instead.
On-time musical seconds / wall seconds = `(eligible_ms - missing_ms) / active_wall_ms`;
active wall time includes USER/WAIT/PREPARING/MODEL, excludes stopped idle, and includes stale audio.
It measures nonmissing coverage, not semantic correctness or new-context compliance.

Per-request accounting is different: `_account_playback` increments `played_frames` only when
the selected audio belongs to that request. Covered frames played from another request, including
retained old/stale audio, increment that request's `missing_frames`. At the requested range's end,
`completed_missing` means the request's own output was not performed throughout its range; it does
**not** imply global missing vocals. Do not sum request `missing_frames` to infer global missing duration.
In the final-source steering trace, request `:5:14` ended `completed_missing` with 192000 played and 192000 missing
frames, while global missing duration stayed zero because retained old audio covered the other frames.

`latency_summary` reports finite nonnegative sample count and linearly interpolated p50/p95/max.
Means of valid candidates are per known selected decision, never per audio packet. `SequentialLyricWriter.write`
starts a fresh ledger/counter set each invocation; selected provenance includes that successful invocation's
initial/rescue waves, but **omits entirely failed writer invocations before eventual success**.
`LyricStreamSession` tracks/reports failures separately and does not add their candidate counts to the
selected record. Requested/valid counts and means therefore cannot measure total generation attempts,
tokens, compute or cost. Retry warnings remain separate request/event trace entries; canceled requests
remain rows within bounded history, but neither supplies the omitted candidate-work denominator. Compare like
mode/config/model/warm-state/shared-load runs, retain canceled/error/completed-missing cases,
and report counts as well as success latency. Timeout/no-valid errors are not separate dedicated
counters here; inspect error traces rather than inventing unavailable aggregates.

JSON schema v1 includes `clock_domain`, `units`, `snapshot`, `events`; CSV mixes `record_type=response`
and `record_type=span` with IDs/versions, latency/candidate fields, coverage frames/ticks,
commit/deadline fields. History is bounded: 128 request rows internally, 64 in snapshots,
4096 trace events, 512 span metric rows, and 16 performed lyric lines. Export before evidence
ages out. Clear-metrics-without-Reset is a specification goal, not an implemented page control.

## Tests and deployment

The [user guide](inference-page-guide.md#launch-only-when-needed) gives exact Mac/H200 launch/SSH commands.
[H200 evidence](../reports/2026-11-11-inference-h200.md) records GPU UUIDs, model/runtime revisions, voice and ownership checks.
Qwen2.5-7B-Instruct uses physical GPU 1 (vLLM memory utilization 0.25, context 2048, max sequences 48).
MOSS-TTS-Local-Transformer-v1.5 and MMS use physical GPU 2 (TTS fractions 0.30, not a whole-stack cap).
Both GPUs have other workloads: these are shared-load measurements. Do not touch unrelated jobs.
The deployment report's initial source copy is historical; its [final refresh](../reports/2026-11-11-inference-h200.md#final-integration-refresh)
records matching local/deployed hashes after source synchronization and renderer restart.
Editing files alone does not reload a resident process.

Focused tests, run from the worktree with its existing environment (no install needed):

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/domain/rap/test_inference_contracts.py tests/unit/application/rap/test_inference*.py tests/unit/infrastructure/rap/test_inference*.py tests/unit/presentation/rap_demo/test_inference*.py
node --test tests/integration/rap-inference.test.cjs
```

- [Service tests](../../tests/unit/application/rap/test_inference_service.py) use `Clock`/`Producer`
  fixtures for non-beat readiness, leading rests, packet invariance, missing/stale coverage,
  commitment races, owner expiry, Stop/Start fencing and metric origins.
- [Delivery tests](../../tests/unit/application/rap/test_inference_delivery.py) exercise beat/tick/
  uneven spans, frame rounding, revisions, safe splits, immutable ranges and incompatible PCM.
- [Backend tests](../../tests/unit/infrastructure/rap/test_inference_backend.py) mock transport,
  test bounded requests, partial/out-of-order delivery, silent bars, timeout/cleanup and bar-only rejection.
- [Runtime tests](../../tests/unit/application/rap/test_inference_runtime.py), [output tests](../../tests/unit/infrastructure/rap/test_inference_output.py),
  [API tests](../../tests/unit/presentation/rap_demo/test_inference_api.py), [CLI tests](../../tests/unit/presentation/rap_demo/test_inference_cli.py)
  cover ownership, device transitions, callback failures, recording and composition.
- [UI tests](../../tests/integration/rap-inference.test.cjs) cover fixtures, controls, owner arbitration,
  reconnect/reset races and original-page status; [metrics tests](../../tests/unit/application/rap/test_inference_metrics.py) cover formulas.
- Shared edits also require legacy regression coverage for lyric session/writer, bar audio/stream,
  audio output/devices, timeline, rap lyric server and original controls; use explicit `tests/`
  scope, not vendored `transformers/`. Final scoped results and limitations are listed below.

Live acceptance commands below change the running session; run only when it is available for testing.
The steering script also tests legacy playback and finishes with Reset. Results are scoped below.

```sh
.venv/bin/python scripts/experiments/inference_milestone/live_acceptance.py --url http://127.0.0.1:8018 --cycles 5 --timeout 600 --output output/inference_20261111/acceptance
.venv/bin/python scripts/experiments/inference_milestone/steering_acceptance.py --url http://127.0.0.1:8018 --output output/inference_20261111/steering
node scripts/experiments/inference_milestone/browser_acceptance.cjs
```

`browser_acceptance.cjs` is playback-read-only; `browser_cycle.cjs` starts/captures/stops real playback.
`PLAYWRIGHT_MODULE`/`CHROMIUM_PATH` can select existing Playwright/Chromium.
Launcher output: `output/inference_20261111/client/`, containing `client.log`, PID metadata and session directories.
`inference.wav` records active callbacks including silent WAIT, omits stopped periods, and is
finalized on output close; check recording errors/dropped frames before using it as evidence.
Legacy `mixed.wav` is distinct. H200 `render_artifacts/`, service logs and ownership manifests
live under the deployment root; do not publish private launch tokens wholesale.

## Verification status

**Final code regression, compatibility review, final-source cyclic/steering acceptance and browser checks passed.
Subsequent user listening confirmed good playback with a remaining voice-consistency issue; no acoustic timing measurement was made.**

- Latest [test/review results](../reports/2026-11-11-inference-milestone-progress.md#final-verification):
  **2055 scoped rap tests passed (35.28 s), 88 legacy GUI checks and 81 inference/research Node tests passed**.
  Final review: **10 targeted tests passed, no remaining findings**; targeted counts are not additional
  disjoint suite totals. Optional `writer_error` metadata preserves legacy listener signatures.
  Known WAV fixture warnings are not failures. Five unrelated Lekai/Transformers collection failures
  still prevent a whole-repository pass claim; iteration history remains in the ledger.
- [Final deployment refresh](../reports/2026-11-11-inference-h200.md#final-integration-refresh):
  renderer **2948248**, physical GPU 2, warm-ready **2026-11-11T10:04:46Z**, with matching source hashes.
  Mac client **93588**, session `rap-20261111T095919Z-6e9d91ae`, remains running. PIDs are observations,
  not shutdown instructions. At **10:08:02Z**, H200 verification found healthy services, owned GPU work
  on physical GPUs 1/2 only, and a real Qwen completion `Ready`. The demo was reset after exports
  and left stopped: cycle mode, 60 BPM, USER one bar / MODEL two bars, topic `space`.
- Primary final-source cyclic run: [summary](../../output/inference_20261111/final-source-cycle/summary.json),
  [trace](../../output/inference_20261111/final-source-cycle/trace.json), [final state](../../output/inference_20261111/final-source-cycle/final.json),
  [samples](../../output/inference_20261111/final-source-cycle/samples.json). **Five exchanges / ten MODEL bars / 40 s eligible coverage**;
  all five requests `played`; zero missing/stale/fallback duration, errors and cancellations.
  Response-start latency (n=5): p50 **5337.111417 ms**, p95 **5973.098933 ms**, max **6132.089000 ms**.
  Ten selected decisions averaged **6.2 valid candidates**; per-request requested counts were
  **64, 64, 96, 64, 64** (two selected bar decisions per request, including rescue waves).
  These provenance counts omit any entirely failed writer attempts and are not total LLM work/cost.
- Recording from `final.audio.recording_path`: [inference.wav](../../output/inference_20261111/client/sessions/rap-20261111T095919Z-6e9d91ae/inference.wav).
  Mac callback warnings/errors and recording drops were zero.
  No recording drops/errors were reported. Automated artifacts retain `acoustic_verification: false`;
  subsequent listening feedback is recorded in the ledger, not retroactively inserted into those traces. Stopped idle is
  excluded from active-wall metrics. TTFT remains null for this real backend.
- [Final ready-state browser check](../../output/inference_20261111/browser/summary.json): desktop
  1440x1000 and mobile 390x844 reported no overflow/page errors; refresh/duplicate tab retained
  session `rap-20261111T095919Z-6e9d91ae`, epoch **7**, stopped.
- [Browser-cycle evidence](../../output/inference_20261111/browser-cycle/evidence.json) and
  [MODEL screenshot](../../output/inference_20261111/browser-cycle/model-playing.png): actual UI Start,
  seven successful Space taps, leading tick **2**, and a populated real MODEL syllable grid (seven syllables).
  Refresh preserved session/epoch/owner and MODEL playback; Stop reached stopped. The
  [browser test](../../scripts/experiments/inference_milestone/browser_cycle.cjs) also verified Rap Lab's
  Stop control enabled during inference and disabled afterward, with no page errors.
  Cross-page playback commands and exclusive ownership were also exercised in the steering run below.
- Final-source [steering evidence](../../output/inference_20261111/final-source-steering/evidence.json) and
  [trace](../../output/inference_20261111/final-source-steering/inference-trace.json): both topic and flow changed;
  first new-version playback arrived after **7480.467000 ms**, with **7465.291667 ms stale coverage** and
  **zero global missing duration** at transition, plus zero callback errors/warnings. Old audio was
  explicitly retained; earliest eligible replacement does not mean immediate or zero-latency steering.
  Request `:5:14` later became `completed_missing`; this records incomplete replacement performance,
  not a vocal gap. The steering snapshot retains earlier cycle counters: do not treat its cumulative
  coverage/candidate totals as a steering-only run or replace the cyclic snapshot's metrics with them.
  Stop/Start preserved performed history without Reset; original-page Stop stopped inference and
  inference-page Stop stopped legacy playback. Legacy reached bar 1 with zero fallback/underrun;
  Reset retained the application session. Competing starts and legacy mode-specific edits were rejected.
- The earlier [steering failure](../../output/inference_20261111/steering-failure.json) exposed fatal
  handling of a retryable lyric failure. The adapter now emits `retry_warning` for recognized
  `lyric_write_failed`, `retryable: true`, valid in-range attempts below the advertised limit;
  retries remain bounded to three server attempts without extending the request deadline.
  Generic/exhausted failures remain fatal. **No `retry_warning` occurred in the successful live run**;
  recovery/deadline behavior is covered by deterministic [backend tests](../../tests/unit/infrastructure/rap/test_inference_backend.py),
  not demonstrated by that live run. Old-listener-signature compatibility is covered by the final review/tests above.
