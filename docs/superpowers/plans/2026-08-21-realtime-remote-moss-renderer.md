# Realtime Remote MOSS Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a selectable realtime mode in which one H200 request generates and selects two bars of lyrics, renders them through persistent MOSS + MFA + Rubber Band R3, and returns exact-duration vocals for deadline-safe Mac playback with local eSpeak fallback.

**Architecture:** Preserve the current bar-oriented eSpeak path. Add a parallel two-bar `RapChunkPreparationStrategy`: the remote implementation calls a private H200 orchestrator that composes existing lyric generation/ranking and offline aligned-MOSS components, while a new rolling chunk controller validates, mixes, and commits returned audio through the existing Mac playback service. The external operation is one request and one binary ZIP response; internal generation, evaluation, synthesis, alignment, and warping remain separate replaceable components.

**Tech Stack:** Python 3.10+, frozen dataclasses, FastAPI/Uvicorn, httpx, NumPy/SciPy, MOSS-TTS-v1.5, Montreal Forced Aligner, Rubber Band R3, pytest, existing StreamMUSE rap domain and monitoring infrastructure.

**Spec:** `docs/superpowers/specs/2026-08-21-realtime-remote-moss-renderer-design.md`

## Global Constraints

- The Mac remains the authoritative 48 kHz stereo float32 playback clock.
- Remote output is exactly two bars of 24 kHz mono PCM16 vocal WAV; drums are rendered and mixed on the Mac.
- `--rap-audio-renderer espeak` preserves the current implementation and behavior.
- `--rap-audio-renderer moss_aligned_remote` uses one external request per two-bar chunk.
- The H200 request includes both materialized flow schedules; the response exposes selected lyrics, score diagnostics, resolved budgets, stage timing, alignment data, and audio identity.
- The accepted first remote alignment mode is plain `continuous_onset_r3`, without the experimental stress gain envelope.
- Every remote failure or deadline miss retains a ready local eSpeak fallback and cannot stretch the musical clock.
- The H200 service binds to loopback and is reached through SSH forwarding.
- Use a persistent MOSS runtime. Reloading the checkpoint for each chunk is not an acceptable realtime implementation.
- Select unused GPUs with `nvidia-smi`; do not terminate or interfere with unrelated processes.
- Work with existing uncommitted files and never revert changes not made for this plan.

## Priority And Cut Line

**P0, required working path:** Tasks 1-6 plus the real H200 smoke and continuous WAV run in Task 8.

**P1, complete after P0:** Full website projection, detailed artifact retention, broader profile sweeps, and listening comparison in Task 7 and the latter half of Task 8.

If execution time becomes constrained, reduce candidate-profile breadth and UI polish. Do not omit exact-duration validation, fallback, real MOSS/MFA/R3 execution, H200 testing, or eSpeak regression coverage.

## File Structure

### New Files

- `src/streammuse/domain/rap/remote_chunk.py`: immutable request, selected-bar, manifest, package, and prepared-chunk values.
- `src/streammuse/application/rap/chunk_orchestration.py`: budget-aware candidate planning and backend-neutral H200 orchestration.
- `src/streammuse/application/rap/chunk_audio.py`: client-side remote vocal validation, resampling, bar splitting, drum mixing, and `PreparedRapChunk` construction.
- `src/streammuse/application/rap/chunk_realtime.py`: two-bar rolling controller, primary/fallback commitment, lifecycle, and deadline policy.
- `src/streammuse/infrastructure/rap/chunk_package.py`: canonical JSON and safe ZIP encode/decode.
- `src/streammuse/infrastructure/rap/remote_chunk_client.py`: cancellable httpx client for health and render operations.
- `src/streammuse/infrastructure/rap/moss_tts.py`: persistent MOSS runtime and one-phrase waveform generation extracted from the offline backend.
- `src/streammuse/infrastructure/rap/moss_aligned_phrase.py`: realtime adapter combining MOSS, MFA, and `continuous_onset_r3` warping.
- `src/streammuse/presentation/rap_render_server.py`: dependency-injected FastAPI app and H200 server CLI.
- Matching unit tests under `tests/unit/domain/rap/`, `tests/unit/application/rap/`, `tests/unit/infrastructure/rap/`, and `tests/unit/presentation/`.

### Modified Files

- `src/streammuse/domain/rap/__init__.py`: export remote chunk contracts.
- `src/streammuse/domain/rap/events.py`: add chunk lifecycle and remote fallback events.
- `src/streammuse/application/rap/audio_service.py`: add controller and chunk-preparation protocols.
- `src/streammuse/application/rap/runtime.py`: type the audio runtime against the common controller lifecycle.
- `src/streammuse/presentation/rap_demo/cli.py`: renderer flags and mode-specific assembly.
- `src/streammuse/application/rap/monitoring_payloads.py`: bounded chunk diagnostic payloads.
- `src/streammuse/application/rap/monitoring.py` and presentation projector/terminal files: display remote state without changing existing events.
- `scripts/rap_audio_backends/moss_backend.py`: delegate shared runtime/generation behavior to `moss_tts.py` while retaining the offline CLI.
- `scripts/rap_audio_backends/aligned_moss_backend.py`: reuse the common aligned phrase operation without changing offline output semantics.
- `pyproject.toml`: add `streammuse-rap-render-server` entry point.
- `docs/developer-guide/rap-demo-quickstart.md`: H200 service, two-port tunnel, renderer commands, and artifact locations.

---

### Task 1: Remote Chunk Contracts And Binary Package Codec [P0]

**Files:**
- Create: `src/streammuse/domain/rap/remote_chunk.py`
- Create: `src/streammuse/infrastructure/rap/chunk_package.py`
- Modify: `src/streammuse/domain/rap/__init__.py`
- Test: `tests/unit/domain/rap/test_remote_chunk.py`
- Test: `tests/unit/infrastructure/rap/test_chunk_package.py`

**Interfaces:**
- Consumes: `FlowTemplate`, `ScheduledSyllable`, `PcmAudio`, `PreparedRapBar`, and canonical timing rules.
- Produces: `RemoteCandidatePolicy`, `RemoteRapBarRequest`, `RemoteRapChunkRequest`, `RemoteSelectedBar`, `RemoteRapChunkManifest`, `DecodedRapChunkPackage`, `PreparedRapChunk`, `encode_chunk_package()`, and `decode_chunk_package()`.

- [ ] **Step 1: Write failing domain validation tests**

Cover exactly two consecutive bars, 4/4 flow compatibility, positive finite BPM and budget, deterministic request identity, expected 24 kHz frame count, selected-bar identity, immutable diagnostic mappings, and exactly two bars in `PreparedRapChunk`.

```python
def test_remote_request_requires_two_consecutive_bars(flow):
    with pytest.raises(ValueError, match="two consecutive bars"):
        RemoteRapChunkRequest.create(
            session_id="session-1",
            chunk_index=0,
            bars=(bar_request(0, flow), bar_request(2, flow)),
            tempo_bpm=90.0,
            remaining_budget_ms=5_000,
            policy=RemoteCandidatePolicy.realtime_default(),
            context_lines=(),
            seed=7,
        )
```

- [ ] **Step 2: Run the contract tests and confirm they fail**

Run: `uv run pytest tests/unit/domain/rap/test_remote_chunk.py -q`

Expected: collection fails because `streammuse.domain.rap.remote_chunk` does not exist.

- [ ] **Step 3: Implement the immutable contracts**

Use these stable public names:

```python
REMOTE_CHUNK_SCHEMA_VERSION = "streammuse.rap_chunk.v1"

@dataclass(frozen=True)
class RemoteCandidatePolicy:
    profile: str
    initial_candidates: int
    rescue_candidates: int
    maximum_candidates: int
    minimum_valid_candidates: int
    minimum_score: float
    render_reserve_ms: int

@dataclass(frozen=True)
class RemoteRapBarRequest:
    bar: int
    topic: str
    flow_template: FlowTemplate

@dataclass(frozen=True)
class RemoteRapChunkRequest:
    schema_version: str
    session_id: str
    request_id: str
    chunk_index: int
    bars: tuple[RemoteRapBarRequest, RemoteRapBarRequest]
    tempo_bpm: float
    output_sample_rate_hz: int
    expected_frame_count: int
    remaining_budget_ms: int
    policy: RemoteCandidatePolicy
    context_lines: tuple[str, ...]
    seed: int

@dataclass(frozen=True)
class PreparedRapChunk:
    request_id: str
    chunk_index: int
    renderer: str
    bars: tuple[PreparedRapBar, PreparedRapBar]
    diagnostics: Mapping[str, object]
```

Provide strict `to_payload()` / `from_payload()` methods. Derive `request_id` from canonical JSON excluding `remaining_budget_ms`, so a transport retry has the same idempotency identity while a changed lyric/flow request does not.

- [ ] **Step 4: Write failing package-codec tests**

Test deterministic member names, canonical manifest bytes, SHA-256 validation, duplicate members, path traversal names, unexpected files, oversized packages, malformed JSON, non-PCM16/non-mono WAV, wrong sample rate, wrong frame count, silence, and non-finite decoded samples.

```python
def test_package_round_trip(manifest, vocal_wav_bytes):
    encoded = encode_chunk_package(manifest, vocal_wav_bytes)
    decoded = decode_chunk_package(encoded, expected_request_id=manifest.request_id)
    assert decoded.manifest == manifest
    assert decoded.vocal_wav == vocal_wav_bytes
```

- [ ] **Step 5: Implement safe ZIP encoding and decoding**

The archive contains only `manifest.json` and `vocals.wav`, uses media type `application/vnd.streammuse.rap-chunk+zip`, and is rejected above 4 MiB. Parse WAV through `wave` or SciPy rather than manual byte slicing. Hash the exact WAV member bytes.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/unit/domain/rap/test_remote_chunk.py tests/unit/infrastructure/rap/test_chunk_package.py -q`

Commit only task files with: `git commit -m "add remote rap chunk contracts"`.

### Task 2: Budget-Aware H200 Candidate Planning And Selection [P0]

**Files:**
- Create: `src/streammuse/application/rap/chunk_orchestration.py`
- Test: `tests/unit/application/rap/test_chunk_orchestration.py`
- Modify if required: `src/streammuse/infrastructure/inference/local_chat_client.py`
- Test if modified: `tests/unit/infrastructure/inference/test_local_chat_client.py`

**Interfaces:**
- Consumes: `RemoteRapChunkRequest`, existing `CandidateGenerator`, `ProsodyAnalyzer`, `rank_candidates()`, `align_exact()`, and `ScoreWeights`.
- Produces: `ChunkCandidatePlanner.plan(request) -> ChunkLyricPlan`, `RapChunkOrchestrator.render(request) -> RemoteChunkRenderArtifact`, and injected `PhraseVocalRenderer` protocol.

- [ ] **Step 1: Write failing tests for candidate waves and pair selection**

Use deterministic fake generators and a fake monotonic clock. Prove that the planner evaluates each completed wave immediately, stops after the configured valid count, never exceeds `maximum_candidates`, reserves `render_reserve_ms`, selects each bar through the existing hard gate, and applies a deterministic pair tie-break based on mean score followed by continuity/rhyme and source order.

```python
def test_planner_stops_after_minimum_valid_candidates(request, fake_generator):
    planner = ChunkCandidatePlanner(fake_generator, analyzer, ScoreWeights(), monotonic=clock)
    plan = planner.plan(request)
    assert plan.stats.generated_candidates == 16
    assert plan.stats.valid_candidates >= request.policy.minimum_valid_candidates
    assert fake_generator.requests == [16]
```

- [ ] **Step 2: Run the planning tests and verify failure**

Run: `uv run pytest tests/unit/application/rap/test_chunk_orchestration.py -q`

- [ ] **Step 3: Implement `ChunkCandidatePlanner`**

For each bar, construct an ordinary `CandidateRequest` containing the exact supplied `FlowTemplate`. Analyze and rank every unique candidate with existing production functions. Preserve all score components and rejection reasons. Build `TwoBarRenderRequest` from the two selected schedules with chunk-relative target seconds.

The plan result must include:

```python
@dataclass(frozen=True)
class ChunkLyricPlan:
    request: RemoteRapChunkRequest
    selected_bars: tuple[RemoteSelectedBar, RemoteSelectedBar]
    render_request: TwoBarRenderRequest
    candidate_stats: Mapping[str, object]
    top_candidates: tuple[Mapping[str, object], ...]
    stage_timings_ms: Mapping[str, float]
```

Limit response diagnostics to the best eight candidates and eight representative rejections per bar; retain the complete ledger in the server artifact directory.

- [ ] **Step 4: Add independent-choice support when the current client cannot expose all vLLM choices**

Extend the OpenAI-compatible client with a typed `generate_choices(messages, *, n, max_tokens, temperature)` method returning each decoded choice plus aggregate usage and latency. Preserve `generate()` behavior exactly. Test malformed individual choices, partial choices, abort, timeout, and `n=1` compatibility.

- [ ] **Step 5: Implement orchestration against injected boundaries**

```python
class PhraseVocalRenderer(Protocol):
    def render(self, request: TwoBarRenderRequest, workspace: Path) -> PhraseRenderResult: ...

@dataclass(frozen=True)
class RemoteChunkRenderArtifact:
    manifest: RemoteRapChunkManifest
    vocal_wav: bytes
    candidate_ledger: tuple[Mapping[str, object], ...]
    workspace: Path

class RapChunkOrchestrator:
    def render(self, request: RemoteRapChunkRequest) -> RemoteChunkRenderArtifact:
        lyric_plan = self._planner.plan(request)
        phrase = self._renderer.render(lyric_plan.render_request, self._workspace(request))
        return self._package_artifact(request, lyric_plan, phrase)
```

Make stage timings monotonic and fail with typed errors such as `NoValidCandidates`, `RenderBudgetExpired`, and `PhraseRenderFailed`. Do not emit a successful silent artifact.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/unit/application/rap/test_chunk_orchestration.py tests/unit/infrastructure/inference/test_local_chat_client.py -q`

Commit with: `git commit -m "add server-side rap chunk planning"`.

### Task 3: Persistent MOSS, MFA, And R3 Phrase Renderer [P0]

**Files:**
- Create: `src/streammuse/infrastructure/rap/moss_tts.py`
- Create: `src/streammuse/infrastructure/rap/moss_aligned_phrase.py`
- Modify: `scripts/rap_audio_backends/moss_backend.py`
- Modify: `scripts/rap_audio_backends/aligned_moss_backend.py`
- Test: `tests/unit/infrastructure/rap/test_moss_tts.py`
- Test: `tests/unit/infrastructure/rap/test_moss_aligned_phrase.py`
- Test: existing `tests/unit/scripts/test_rap_audio_moss_backend.py`
- Test: existing `tests/unit/scripts/test_rap_audio_aligned_moss_backend.py`

**Interfaces:**
- Consumes: `TwoBarRenderRequest`, `PhraseVocalRenderer`, existing MOSS generation parameters, MFA models, and `continuous_pitch_preserving_warp()`.
- Produces: `PersistentMossSynthesizer`, `MossAlignedPhraseRenderer`, and `PhraseRenderResult` containing WAV bytes/path, alignment metrics, versions, hashes, and stage timing.

- [ ] **Step 1: Write failing persistent-runtime tests**

Prove that model construction happens once, consecutive phrase calls reuse the runtime, each request gets an isolated workspace, MOSS retries do not produce a successful silence result, and model/configuration metadata remains identical to the offline backend.

- [ ] **Step 2: Extract the reusable MOSS runtime without changing offline behavior**

Move the reusable runtime and one-phrase generation logic from the script into `moss_tts.py`. Keep the script's public constants and functions available by importing/delegating so existing offline tests and campaign commands continue to work.

```python
class PersistentMossSynthesizer:
    @classmethod
    def load(cls, *, model_id: str, device: str, reference_wav: Path) -> "PersistentMossSynthesizer": ...

    def synthesize(self, request: TwoBarRenderRequest, output_wav: Path) -> MossPhraseResult: ...
```

- [ ] **Step 3: Write failing alignment-adapter tests**

Inject fake MOSS, MFA, and full-chunk stretcher implementations. Verify transcript staging, strict phone matching, documented word fallback, onset promotion, plain `continuous_onset_r3`, exact frame count, hash propagation, stage timings, and cleanup. MFA failure and impossible/non-monotonic anchor maps must raise `PhraseRenderFailed`.

- [ ] **Step 4: Implement `MossAlignedPhraseRenderer`**

Reuse the proven functions from `streammuse.experiments.rap_audio_protocols.warp` and the aligned offline backend. Run MFA in a request-private corpus/output directory. Return 24 kHz mono PCM16 in the service package even when internal processing uses float32 WAV.

- [ ] **Step 5: Run local compatibility tests**

Run:

```bash
uv run pytest \
  tests/unit/infrastructure/rap/test_moss_tts.py \
  tests/unit/infrastructure/rap/test_moss_aligned_phrase.py \
  tests/unit/scripts/test_rap_audio_moss_backend.py \
  tests/unit/scripts/test_rap_audio_aligned_moss_backend.py -q
```

- [ ] **Step 6: Perform the first real H200 worker smoke before building the client**

SSH to `Andrew.Yang@masdar`, inspect `nvidia-smi`, choose an unused physical GPU, and use the existing MOSS/MFA/Rubber Band environments and reference voice documented in `docs/developer-guide/rap-audio-protocol-comparison.md`. Render one known 90 BPM two-bar request twice in the same process. Record cold/warm MOSS, MFA, R3, and total latency plus GPU assignment. This is an early feasibility gate; retain output and logs under a new request-specific H200 artifact directory.

- [ ] **Step 7: Commit the reusable real renderer**

Commit with: `git commit -m "add persistent aligned MOSS phrase renderer"`.

### Task 4: Private H200 Chunk Service And Idempotent Packaging [P0]

**Files:**
- Create: `src/streammuse/presentation/rap_render_server.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/presentation/test_rap_render_server.py`
- Test: `tests/unit/application/rap/test_chunk_orchestration.py`

**Interfaces:**
- Consumes: `RapChunkOrchestrator`, request payload parsing, `encode_chunk_package()`, persistent MOSS worker, local vLLM URL, and artifact root.
- Produces: `create_rap_render_app(orchestrator, health)`, `GET /health`, `POST /v1/rap/chunks/render`, and `streammuse-rap-render-server`.

- [ ] **Step 1: Write failing FastAPI tests with a fake orchestrator**

Test health compatibility fields, valid binary response and content type, malformed schema as HTTP 422, known typed failures as bounded JSON errors, uncaught failures as sanitized HTTP 500, duplicate in-flight idempotency keys, completed-cache hits, and conflicting request bodies.

- [ ] **Step 2: Implement the dependency-injected app**

Run blocking orchestration outside the event-loop thread. A successful response has media type `application/vnd.streammuse.rap-chunk+zip`, `X-StreamMUSE-Request-ID`, `Content-Length`, and `Server-Timing`. Errors never contain secrets or raw authorization headers.

- [ ] **Step 3: Implement atomic idempotent storage**

Store canonical request JSON, full candidate ledger, source WAV, TextGrid, aligned WAV, response package, and failure JSON under `<artifact-root>/<request-id>/`. An identical completed request returns the cached package; a conflicting body returns HTTP 409. Use atomic rename for the final package marker.

- [ ] **Step 4: Add the server CLI**

Expose explicit options for host, port, artifact root, vLLM URL/model, MOSS model/device/reference WAV, MFA dictionary/acoustic model, and candidate profile. Default host is `127.0.0.1`; reject `0.0.0.0` unless `--allow-public-bind` is supplied.

- [ ] **Step 5: Test and commit**

Run: `uv run pytest tests/unit/presentation/test_rap_render_server.py tests/unit/application/rap/test_chunk_orchestration.py -q`

Commit with: `git commit -m "serve aligned rap chunks from H200"`.

### Task 5: Mac Remote Client And Exact Bar Audio Conversion [P0]

**Files:**
- Create: `src/streammuse/infrastructure/rap/remote_chunk_client.py`
- Create: `src/streammuse/application/rap/chunk_audio.py`
- Modify: `src/streammuse/application/rap/audio_service.py`
- Test: `tests/unit/infrastructure/rap/test_remote_chunk_client.py`
- Test: `tests/unit/application/rap/test_chunk_audio.py`

**Interfaces:**
- Consumes: HTTP package endpoint, `DecodedRapChunkPackage`, original chunk request, local `DrumRenderer`, `AudioFormat`, and `ProsodyAnalyzer`.
- Produces: `RemoteChunkClient.health()`, `RemoteChunkClient.prepare(request, timeout_seconds)`, `RemoteMossChunkPreparationStrategy.prepare()`, and exact `PreparedRapChunk` values.

The application boundary used by Task 6 is:

```python
class RapChunkPreparationStrategy(Protocol):
    def prepare(
        self,
        request: RemoteRapChunkRequest,
        *,
        deadline_monotonic: float,
    ) -> PreparedRapChunk: ...

    def abort(self) -> None: ...
    def close(self) -> None: ...
```

`abort()` cancels useful client waiting and closes the active response without
making the strategy permanently unusable. `close()` is final and idempotent.

- [ ] **Step 1: Write failing HTTP client tests**

Use `httpx.MockTransport` to verify canonical request bodies, binary media type, timeout, cancellation/abort, response-size limits, error sanitization, health schema mismatch, idempotent retry, transfer timing, and clean close.

- [ ] **Step 2: Implement the cancellable client**

Keep one persistent `httpx.Client` for connection reuse. Return decoded package plus request/first-byte/download timing. Never retry after the caller's useful musical deadline.

- [ ] **Step 3: Write failing audio-conversion tests**

Verify mono PCM16 decode, 24-to-48 kHz resampling, deterministic two-bar splitting with no dropped/duplicated frame, stereo conversion, local drum rendering with each bar's actual template, exact final `PcmAudio` length, peak limiting, selected-text reanalysis, schedule equality, and manifest mismatch rejection.

```python
def test_remote_chunk_splits_and_mixes_exact_bars(strategy, request):
    prepared = strategy.prepare(request, deadline_monotonic=10.0)
    assert tuple(bar.bar for bar in prepared.bars) == (0, 1)
    assert all(bar.audio.frame_count == 128_000 for bar in prepared.bars)
    assert prepared.renderer == "moss_aligned_remote"
```

- [ ] **Step 4: Implement `RemoteMossChunkPreparationStrategy`**

Reanalyze returned text on the Mac and rerun exact alignment against the original templates. Reject any selected-text, syllable-count, schedule, hash, sample-format, duration, or request-identity disagreement. Convert anchor diagnostics into per-syllable `SyllablePlacementDiagnostic` records with pronunciation source `moss_aligned_remote`.

- [ ] **Step 5: Test and commit**

Run: `uv run pytest tests/unit/infrastructure/rap/test_remote_chunk_client.py tests/unit/application/rap/test_chunk_audio.py -q`

Commit with: `git commit -m "prepare remote MOSS chunks on Mac"`.

### Task 6: Rolling Two-Bar Scheduling, Fallback, And CLI Switch [P0]

**Files:**
- Create: `src/streammuse/application/rap/chunk_realtime.py`
- Modify: `src/streammuse/application/rap/audio_service.py`
- Modify: `src/streammuse/application/rap/runtime.py`
- Modify: `src/streammuse/presentation/rap_demo/cli.py`
- Modify: `src/streammuse/domain/rap/events.py`
- Test: `tests/unit/application/rap/test_chunk_realtime.py`
- Test: `tests/unit/application/rap/test_runtime.py`
- Test: `tests/unit/presentation/rap_demo/test_cli.py`

**Interfaces:**
- Consumes: `RapChunkPreparationStrategy`, existing `DeterministicRapBarRenderer` for fallback, `RapPlaybackService.enqueue()`, scenario/templates/fallback catalog, monotonic clock, and event publisher.
- Produces: `RollingRapChunkController` implementing the audio controller lifecycle and mode-specific CLI assembly.

- [ ] **Step 1: Write failing rolling-controller tests with a manual executor and clock**

Cover startup pre-roll, bars 0-1 remote success, chunk N+1 submission while N plays, exact two-bar commit, on-time primary preference, late primary discard, exception/invalid primary fallback, no queue gap, context advancement only from committed lyrics, finite and looping scenarios, stop-after-bar, reset epoch, close cancellation, and stale result rejection.

- [ ] **Step 2: Define the common controller lifecycle protocol**

```python
class RapAudioController(Protocol):
    @property
    def scenario(self) -> RapScenario: ...
    def start(self) -> None: ...
    def on_tick(self, tick: int) -> None: ...
    def request_stop(self, *, successor_bar: int | None) -> None: ...
    def resume_audio(self, bar: int) -> None: ...
    def resume_after_stop(self) -> None: ...
    def reset(self) -> int: ...
    def close(self) -> None: ...
```

Type `RapAudioDemoDependencies` against this protocol. Do not change existing `RollingRapController` behavior.

- [ ] **Step 3: Implement `RollingRapChunkController`**

Reserve and render local eSpeak fallbacks before every remote request. `start()` may block for the configured startup timeout because the musical clock has not started. Once running, submit one future two-bar request, poll it from tick callbacks without blocking, and commit both bars at the chunk boundary. The Mac's deadline decision is final.

Emit dedicated events for request submission, remote completion, remote rejection, chunk commitment, and chunk fallback. Continue emitting ordinary bar-frozen, audio-ready, audio-committed, playback, tick, and syllable events so current monitors remain useful.

- [ ] **Step 4: Add CLI configuration and validation**

Add:

```text
--rap-audio-renderer espeak|moss_aligned_remote
--rap-render-url http://127.0.0.1:8020
--rap-render-profile realtime
--rap-render-startup-timeout 120
--rap-render-rolling-timeout 5.0
```

Remote mode requires audio output, an available local eSpeak executable for fallback, even two-bar planning, and successful health/schema compatibility. It does not instantiate the Mac-side vLLM client. eSpeak mode preserves current defaults and option validation.

- [ ] **Step 5: Run affected runtime tests**

Run:

```bash
uv run pytest \
  tests/unit/application/rap/test_chunk_realtime.py \
  tests/unit/application/rap/test_runtime.py \
  tests/unit/application/rap/test_playback.py \
  tests/unit/application/rap/test_audio_coordination.py \
  tests/unit/presentation/rap_demo/test_cli.py -q
```

- [ ] **Step 6: Run a local fake-service realtime WAV smoke**

Start the dependency-injected fake chunk service, run six bars at 90 BPM with `--audio-output wav --no-web`, and verify exact WAV duration, alternating remote/fallback scripted cases, zero playback underruns, and visible renderer decisions.

- [ ] **Step 7: Commit the working vertical slice**

Commit with: `git commit -m "switch realtime rap between eSpeak and remote MOSS"`.

### Task 7: Research Monitoring, Website Projection, And Operations Guide [P1]

**Files:**
- Modify: `src/streammuse/application/rap/monitoring_payloads.py`
- Modify: `src/streammuse/application/rap/monitoring.py`
- Modify: `src/streammuse/presentation/rap_demo/terminal_state.py`
- Modify: `src/streammuse/presentation/rap_demo/terminal_dashboard.py`
- Modify: `src/streammuse/presentation/rap_demo/terminal_stream.py`
- Modify: `src/streammuse/presentation/rap_demo/static/js/rap-demo-state.js`
- Modify only if required: `src/streammuse/presentation/rap_demo/static/index.html`
- Modify only if required: `src/streammuse/presentation/rap_demo/static/css/rap-demo.css`
- Modify: `docs/developer-guide/rap-demo-quickstart.md`
- Create: `docs/developer-guide/realtime-remote-moss-acceptance-2026-08-21.md`
- Test: corresponding existing monitoring, terminal, server, and JS tests.

**Interfaces:**
- Consumes: chunk events and manifest diagnostics from Tasks 4-6.
- Produces: bounded live diagnostic projection, recorded manifests, exact setup commands, and acceptance report.

- [ ] **Step 1: Write failing projector and terminal tests**

Assert display of renderer, request/chunk state, two selected lines, both flow schedules, candidate counts, selected component scores, generation/MOSS/MFA/R3/transfer/total timing, remaining slack, warp/fallback warnings, and final MOSS/eSpeak commitment. Preserve dense two-column behavior and current eSpeak snapshots.

- [ ] **Step 2: Implement bounded monitoring payloads**

Do not place raw WAV bytes, unbounded candidate lists, or full TextGrid content on the event bus. Store full artifacts by request ID and expose paths/hashes in events.

- [ ] **Step 3: Update terminal and website projections**

Use existing layout and colors. Add no controls beyond Start, Stop, and Reset. Renderer identity and remote warnings are status data, not interactive controls.

- [ ] **Step 4: Update the quickstart**

Document H200 vLLM and render-server commands, explicit unused-GPU selection, the two-port tunnel:

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L 8001:127.0.0.1:8001 \
  -L 8020:127.0.0.1:8020 \
  Andrew.Yang@masdar
```

Document Mac eSpeak and remote-MOSS commands, health checks, session artifacts, shutdown, and the exact legacy command for temporarily returning to eSpeak.

- [ ] **Step 5: Run monitoring tests and commit**

Run: `uv run pytest tests/unit/application/rap/test_monitoring*.py tests/unit/presentation/rap_demo/test_terminal*.py tests/unit/presentation/rap_demo/test_server.py -q`

Commit with: `git commit -m "expose remote rap render diagnostics"`.

### Task 8: Real H200 Deployment, Latency Tuning, And Final Verification [P0 then P1]

**Files:**
- Modify: `docs/developer-guide/realtime-remote-moss-acceptance-2026-08-21.md`
- Create runtime artifacts under: `logs/rap/remote_moss_acceptance_20260821/`
- Modify implementation/tests only for defects demonstrated by evidence.

**Interfaces:**
- Consumes: committed P0 implementation, SSH host `Andrew.Yang@masdar`, unused H200 GPUs, real Qwen vLLM, persistent MOSS, MFA, Rubber Band, and Mac WAV sink.
- Produces: verified commands, stage latency distributions, candidate/fallback statistics, exact-duration audio, and a documented recommendation.

- [ ] **Step 1: Run the complete local regression gate**

Run: `uv run pytest tests/ -q --tb=no`

Capture the exact pass/fail count. Fix only failures caused by this feature and preserve unrelated worktree changes.

- [ ] **Step 2: Deploy the committed revision to the existing H200 checkout**

Use the established `/data/home/Andrew.Yang/StreamMUSE/real_rap` checkout. Confirm branch/revision, fetch the new commits, and install only the required environment changes. Do not overwrite H200 experiment artifacts or terminate unrelated services.

- [ ] **Step 3: Select GPUs and start persistent services**

Inspect `nvidia-smi` immediately before launch. Assign vLLM and MOSS to unused physical GPUs, record mappings, start both loopback-bound services, and verify `/v1/models` plus `/health`. Keep PIDs and logs in the acceptance artifact directory.

- [ ] **Step 4: Exercise one complete real chunk request**

Send a real 90 BPM request containing two actual flow schedules. Verify selected lyrics against both syllable counts, decode the returned package, confirm 24 kHz mono PCM16 and exact frames, inspect TextGrid/anchor diagnostics, and listen to the returned vocals plus local drum mix.

- [ ] **Step 5: Measure warm stage and transfer latency**

Run at least ten warm requests across the three existing flow templates. Report p50/p95/max for candidate generation, evaluation, MOSS, MFA, R3, packaging, SSH transfer, Mac validation/mix, and total. Report candidate requested/parseable/valid/selectable counts and failures by stage.

- [ ] **Step 6: Tune the realtime profile inside the H200 boundary**

Use measurements to adjust candidate wave sizes, maximum candidates, minimum valid count, generation cutoff, persistent worker allocation, and stage reserve. Keep 90 BPM fixed. Prefer a profile that leaves positive deadline slack and several selectable candidates; do not hide poor candidate validity by labeling server fallback as primary.

- [ ] **Step 7: Run the 20-bar 90 BPM Mac client through SSH**

Use `moss_aligned_remote`, `--audio-output wav`, and the real H200 service. Record the exact mixed WAV, event log, per-chunk manifests, primary acceptance rate, fallback reason distribution, deadline slack, and playback underruns. The run succeeds operationally only if the WAV has exact duration and no application-level gap; fallback usage remains a measured quality result.

- [ ] **Step 8: Compare the same selected lyrics against eSpeak**

Render at least three accepted chunks with both remote MOSS and local eSpeak, retain files, and describe whether the remote path reflects the coherent-word quality of the offline MOSS reference. Do not infer perceptual quality only from alignment metrics.

- [ ] **Step 9: Perform final targeted and full verification after tuning**

Run the affected suites, then `uv run pytest tests/ -q --tb=no`. Check `git diff --check`, inspect every changed file, and verify that no service or SSH tunnel started for testing remains unintentionally running.

- [ ] **Step 10: Complete the acceptance report and commit**

Record assumptions, implementation choices, omitted P1 work, exact commands, revisions, GPU assignments, latency tables, fallback rates, artifact paths, known limitations, and recommended next experiment.

Commit with: `git commit -m "verify realtime remote MOSS rendering"`.

## Subagent Review Protocol

For every implementation task:

1. Dispatch a fresh implementation subagent with only the task text, spec path, current revision, and warning about unrelated worktree changes.
2. Run a specification-compliance review subagent after implementation.
3. Correct all requirement gaps before proceeding.
4. Run a code-quality review subagent after compliance passes.
5. Correct substantive quality findings and rerun focused tests.
6. Commit only the task's files and update this plan's checkboxes.

Tasks 1 and the read-only H200 environment reconnaissance may overlap. Tasks that modify shared Python files execute sequentially.
