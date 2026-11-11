# Inference Page Milestone: Requirements And Delivery Plan

Status: approved for implementation November 11, 2026. The user authorized
subagents, Mac/H200 integration testing, and physical GPUs 1 and 2 only.
The proposed defaults below were accepted with milestone approval; they remain
explicit implementation choices rather than universal research assumptions.

Delivered November 11: see the [implementation handoff](../developer-guide/inference-optimization-handoff.md)
and [verification ledger](../reports/2026-11-11-inference-milestone-progress.md).
This document remains the design contract; the handoff records actual limits.

## 1. Goal

Deliver a clean `/inference` page for optimizing two real, audible rap modes:

1. Cyclic exchange: USER -> WAIT -> MODEL -> USER, repeating automatically.
2. Continuous steering: ongoing model rap with live topic/prompt and flow updates.

These are future official implementations, temporarily exposed on a second page.
Use the same application, session, models, and existing audio pipeline. A separate
branch, server, or independent session is not required. The original page and
legacy modes remain available during development and promotion.

The cyclic optimization trade-off is shorter initial WAIT versus reliable ongoing
MODEL playback: starting from a smaller ready prefix can improve responsiveness
but expose later underruns. Measure both; do not impose a fictitious initial
deadline or improve latency by silently sacrificing playable coverage.

## 2. Confirmed Requirements And Proposed Defaults

Confirmed:

- USER and MODEL follow music. WAIT does not: it has no configured duration and
  no beat/bar quantization. It ends when the client can start the response.
- WAIT pauses music and is silent for this milestone; no filler/replay is added.
- Response production/delivery may use bars, beats, ticks, or variable spans.
- The new page contains only controls and diagnostics relevant to these modes.
- Modes have explicit optimization boundaries; do not fork the whole rap stack.
- Continuous steering may retain compatible previously prepared output when a
  replacement is late. During MODEL, missing material uses drums-only for the
  missed interval rather than moving the musical timeline.

Proposed defaults, requiring review rather than treated as prior user decisions:

- USER initially captures keyboard rhythm using the existing exchange input.
  Microphone topic input is deferred from this milestone.
- Initial demo: 60 BPM, 4/4, USER one bar, MODEL two bars. Durations are configurable;
  four ticks/beat is the existing adapter's resolution, not a universal constant.
- Mode, tempo, turn length, and backend-policy settings change only while stopped
  on this new page. Live steering is limited to topic/prompt and flow initially.
- An empty USER capture repeats USER with a visible "No rhythm captured" status;
  it does not silently invent a response or make a pointless inference request.

## 3. Granularity: Four Independent Concepts

| Concept | Definition | Requirement |
| --- | --- | --- |
| Turn length | How long USER or MODEL performs | Musical duration, independent of response chunks |
| Generation unit | What one inference/validation decision produces | Backend-specific; may be a line, bar, beat, or another span |
| Delivery unit | What arrives in one message/audio chunk | May differ from generation and playback units |
| Commitment unit | How much output becomes immutable at once | Explicit policy/capability, not inferred from packet size |

Transport packets and LLM tokens are not musical ticks. Dividing a fully generated
bar into four packets does not establish beat-level generation or lower latency.

Use exact musical positions, for example integer ticks with an explicit ticks-per-
beat resolution, plus sample/frame positions for actual audio scheduling. Do not
hard-code `bar * 16` into the new mode interface or metrics. The old adapter may
retain its existing bar assumptions behind that interface.

MVP backend: adapt the working bar-oriented pipeline honestly. Acceptance also
requires deterministic beat/tick/uneven-chunk test producers through the same
controller/UI. Real fine-grained generation is an optimization extension, not a
claim this milestone makes. Display supported/active granularity and startup
buffer policy; do not offer unsupported granularity as a working setting.

## 4. Cyclic Exchange State Machine

```text
STOPPED --Start--> USER --capture closes--> WAIT --playable prefix--> MODEL
                     ^                                               |
                     +--------------- turn complete -----------------+
```

### USER

- Start a fresh musical downbeat and play drums for the configured duration.
- Capture input only inside this phase, mapped onto its musical timeline.
- Preserve leading rests; a first tap after tick zero must not be shifted to zero.
- Proposed mapping for unequal turn lengths: repeat the captured U-bar flow over
  the M-bar response and truncate only at the response boundary, keeping all
  relative positions/rests. This is a flow-conditioning policy, not a transport
  requirement; confirm it with the other proposed defaults.
- At the end, freeze capture, assign a cycle/input version, and release response
  work. Cross-cycle input must not leak into another request.
- No prerequisite to generate all MODEL output before the response can begin.

### WAIT

- Stop the musical clock and drums. Display elapsed wait, not a countdown.
- Start MODEL once the client has a decoded, schedulable, contiguous playable
  prefix from response position zero. Honor explicit leading silence/rests.
- Metadata, a token, a URL, or a later out-of-order chunk is not playable readiness.
- Use the earliest available audio callback/output boundary after readiness.
  Do not wait for the next wall-clock beat, a full bar, or a fixed sleep timer.
- The startup buffering policy must be explicit and recorded. Default to the
  minimum prefix supported by the adapter; a larger buffer is a visible optimization
  trade-off, not a hidden delay. Network arrival and client readiness are separate.
- Start drums at MODEL position zero, even if the first vocal onset is later.
- There is no initial response deadline to miss: longer WAIT is response latency.
  Existing bounded request timeouts/errors transition to a visible recoverable
  error, not fake successful playback. Stop must always remain available.

### MODEL

- Establish a fresh musical epoch at response start. Play the configured duration
  independently of the size/count of delivered chunks.
- Accept incremental future spans while playing. Once started, musical time does
  not stretch or pause to accommodate a late result.
- For an unavailable span, continue drums without its intended vocals, record
  the missed coverage, and resume eligible vocals at a future safe boundary.
- Never play a stale chunk halfway through its expired interval. The adapter must
  declare safe split points; do not blindly cut a phoneme because transport chunks
  are small. Without a safe split, discard expired material and use the next safe span.
- At MODEL end, drop/cancel work beyond the turn and begin the next USER phase
  on the following musical boundary. Do not spill vocals into USER.

### Stop, Resume, Reset

- Stop in WAIT cancels local waiting immediately and fences off late responses.
- Proposed preservation of existing stop behavior: Stop during USER/MODEL finishes
  the current musical bar or phase, whichever ends first; no new cycle begins.
  This stop policy is independent of backend output granularity.
- Start after Stop begins a fresh USER cycle without requiring Reset; preserve
  configuration and selected steering inputs, not a partially captured old turn.
- Reset clears active work/history/metrics according to existing session semantics.
  Old request/cycle/config versions must never affect the replacement state.

## 5. Continuous Steering Mode

- Start enters silent PREPARING, not USER/WAIT. The same explicit playable-prefix
  policy starts a fresh downbeat immediately after client readiness. Record
  startup latency separately from steady-state steering latency.
- Continue musical playback and incremental generation until stopped.
- Reuse existing topic/prompt and flow semantics rather than adding a new prompt
  rewriting service. Show both requested and actually playing input versions.
- Steering applies to the earliest mutable future musical span supported by the
  active backend/commit policy. Do not promise tick-level edits with a bar-only adapter.
- Preserve committed playback. If a new version cannot replace old prepared
  output in time, retain compatible old output and mark it as stale context.
- If neither updated nor compatible old output is ready, use the explicit
  drums-only missing-span policy. Do not hide this as a successful response.
- New-context playback proves which input was used, not semantic compliance.
- Stop while PREPARING cancels immediately; Stop while playing finishes the current
  musical bar and cancels/fences future work. Start after Stop begins a fresh
  playback epoch, preserving latest inputs and already performed lyric history
  without reusing invalidated queued audio or requiring Reset. Reset alone clears
  the session according to normal semantics.

### Commitment And Replacement Rules

- A span is committed when the playback layer accepts it into an immutable output
  queue, or the active policy deliberately freezes it earlier. Emit that exact
  event, covered frame range, reason, and deadline. Never report a fictional
  mutable span after audio has actually been enqueued irreversibly.
- Each adapter/policy exposes its commit quantum, guard time, and maximum immutable
  lookahead. The legacy adapter initially uses its verified existing values;
  unknown values cannot be advertised as finer-grained editable coverage.
- Replacement wins only if its session/epoch and target span match, its requested
  input version is still current, and it is ready before commitment. Resolve races
  atomically at commitment; keep an auditable selected output/revision ID.
- Compatible retained output has the same playback epoch, musical/frame coverage,
  tempo/meter mapping, and supported audio format/render policy. Its old topic or
  flow is allowed only as an explicitly stale-context fallback, not relabeled as
  satisfying the new instruction. An old flow must not be retimed implicitly.
- Retention fills an already planned slot; it does not loop that old output
  forever. Continue preparing the latest requested version for later mutable
  slots, exposing stale duration, pending age, errors, and any lack of progress.

## 6. Shared Session And Code Boundaries

- Add `/inference` to the existing FastAPI application. No second session is
  created by opening it, refreshing it, or opening another browser tab.
- Only one active playback mode owns the shared session/audio sink. Change the
  active implementation only while stopped; never start two competing controllers.
- Original controls and new controls act on the same session. Display its active
  mode everywhere necessary; switching pages must not reset or change mode.
- Global Stop/Reset remain meaningful from either page. Reject incompatible
  mode-specific writes clearly rather than silently mutating another mode's state.
- Preserve legacy defaults, endpoints, and controller behavior when legacy modes
  are active. Add only minimal navigation/status integration to the original page.

### Browser Lifecycle And Capture Ownership

- The Mac Python runtime owns audio, not a browser tab. Continuous playback
  survives navigation, refresh, or browser disconnect; reopening attaches to
  current state without replaying audio or issuing Start automatically.
- Cyclic keyboard capture has one explicit browser owner. Other tabs can monitor
  and use global Stop/Reset but cannot submit taps concurrently. Bind taps to
  owner, cycle, event ID, and calibrated capture time; deduplicate retries.
- Proposed implementation default: owner heartbeat every 2 s, expiring after 6 s.
  A refresh can reclaim its tab token within that lease and receive current state;
  it must not replay old taps. A second tab takes ownership only when stopped or
  after the runtime has handled an expired lease, not by merely opening the page.
- On owner loss, continuous mode is unaffected. Cyclic mode stops immediately in
  WAIT, or completes its current USER/MODEL phase and then stops. It does not
  silently generate new empty-input cycles indefinitely. Show the reason and
  allow a new owner to Start a fresh cycle without Reset.

Separate responsibilities:

1. Mode controller: USER/WAIT/MODEL or continuous-steering policy.
2. Optimization/backend adapter: requests, candidate budget, concurrency,
   cancellation/revision, and generation/delivery capabilities.
3. Playback: readiness, immutable spans, drums/vocals, callback scheduling.
4. Telemetry/projector: correlate events and produce one UI/export representation.
5. UI: display state and submit commands; no independent scheduling clock authority.

The inference engineer may replace optimization policy/adapter internals without
rewriting the page or audio sink. Changes to shared components still require legacy
regression tests: UI separation alone cannot guarantee code-level isolation.

This is an extension point, not a restriction on editing. Everything serving the
two inference-page modes is open to modification: UI, state machines, prompts,
request strategy, models/serving integration, candidate generation and evaluation,
audio rendering/alignment, transport, buffering/playback, configuration, and metrics.
There are no protected implementation black boxes for these modes. Changes to
shared code must either preserve legacy behavior or be scoped to the new modes;
the handoff must identify those shared dependencies explicitly. Changes to public
contracts or metric definitions need corresponding tests/documentation updates.

Suggested implementation locations, to refine after review:

- New domain contracts in `src/streammuse/domain/rap/inference.py`.
- Focused modules under `src/streammuse/application/rap/inference/` for modes,
  policy/adapter boundaries, and metrics; avoid another giant controller file.
- New `presentation/rap_demo/inference_api.py`, `static/inference.html`,
  `static/js/rap-inference.js`, and `static/css/rap-inference.css`.
- Minimal composition changes in `application/rap/runtime.py` and
  `presentation/rap_demo/server.py`; reuse existing writer/rendering/audio services.

## 7. Optimization Contract

Versioned request context must include session epoch, cycle/steering version,
request ID, musical resolution, target span, tempo, flow, text history, and
generation-policy identity/configuration. Do not mix user intent versions with
transport retry IDs.

Incremental results must declare:

- Request/output ID and revision, input version, and target musical start/duration.
- Payload type: candidate, selected lyric, audio, or control/status event.
- For audio: reference/bytes, sample rate, frame count, corresponding musical span,
  declared silence, safe join/split positions, and final/end-of-response status.
- Available stage timing and clock domain, not fabricated timestamps.

Duplicate results are idempotent. Out-of-order future spans may be buffered but
must not fill a missing prefix by accident. Superseded versions cannot replace
committed audio. Late canceled work is accounted for but not performed. Backend
disconnects are visible; cancellation acknowledgment must distinguish client-side
abort from confirmed server/GPU cancellation.
Validate positive durations, finite timing values, frame/sample-rate consistency,
payload bounds, and nonconflicting revision coverage before scheduling. A zero-frame
payload is not a playable prefix. All response-readiness metrics refer to the
intended current input/cycle, not a cached fallback or an old response version.

Keep candidate/decision identity separate from audio chunks: one validated lyric
may produce many packets, and one packet may contain several decisions.

## 8. Focused Screen Requirements

Use the existing visual language but a fresh layout and dedicated state projector.
No inherited dashboard sections unless they serve this workflow.

### Always Visible

- Mode selector, Start/Stop/Reset, BPM, connection/error state, active renderer.
- Mode inputs: USER/MODEL duration and capture surface for cyclic mode;
  topic/prompt input and compact flow selection/preview for continuous steering.
- Current lyrics with syllable/flow positions, current beat/tick, and scheduled
  versus observed playback where available. Separate intended timing from measured
  callbacks; do not claim software timestamps measure acoustic output.
- Phase indicator: USER/MODEL remaining musical duration; WAIT elapsed seconds.
- A compact time-span preparation view with selected, ready, committed, playing,
  late/missing, or stale-context states. Do not require one rectangular item per bar.
- A compact metrics strip and recent-response table; expandable request details.

### Secondary Controls / Details

- Output device and essential level controls in a compact audio popover.
- Active backend capabilities, output/commit granularity, startup buffer policy,
  candidate budget and concurrency in an optimization settings panel. Unsupported
  controls are unavailable with a reason, not decorative toggles.
- Inspect request context/candidates, rejection reasons, and stage traces only
  on demand. Bounded history and downloadable traces avoid an endless raw log wall.

Exclude from this milestone: microphone UI, characters/counting tabs, story branch
suggestions, chord editing, alignment-policy experiments, Chinese support, generated
fillers, GPU administration, and automatic semantic-quality scoring. Existing
audio settings remain usable elsewhere in the shared application.

## 9. Metrics And Measurement Boundaries

Never call total request latency TTFT, and never equate a valid candidate with a
meaningful answer. Use milliseconds internally, readable seconds/ms in the UI.

| Metric | Start -> end | Scope / caveat |
| --- | --- | --- |
| LLM TTFT | Request dispatch -> first content token observed | Per request; client-observed or server-reported must be labeled |
| First valid candidate | Response work released -> first complete candidate passing checks | Per response/decision; validity rules belong to the backend |
| Playable-prefix ready | Response work released -> contiguous prefix decoded/prepared on client | Includes delivery and local preparation, not just TTS completion |
| Response start latency | USER capture closes -> MODEL playback starts | Primary cyclic metric; no first-bar deadline miss |
| Steering playback latency | Edit accepted -> first performed span using its version | Not proof of semantic compliance |
| First vocal onset | Same interaction origin -> first intended vocal onset | Separate from drums/MODEL start and intentional leading rests |
| Ready-to-start overhead | Client startup readiness -> actual MODEL callback start | Reveals buffering/scheduler delay, not model compute |
| Deadline margin | Playback commitment deadline minus ready time | Only for spans with an established deadline; negative means late |
| Candidate yield | Valid/selectable candidates out of evaluated candidates | Per decision/round, not per packet; retain requested/parsed counts |
| Missing/stale/fallback coverage | Duration affected / eligible scheduled duration | Duration-weighted so packet-size changes cannot improve the metric artificially |

For pending output display time remaining; for missing completion retain missing
status, not a made-up ready time. Late discarded results may still provide a
separate measured lateness. Do not count deliberate USER/WAIT silence as an underrun.

Expandable serving breakdown: queue, prefill, decode, per-choice inter-token
latency/output throughput, input/output token counts, validation/ranking, TTS,
alignment, fetch/decode/preparation, and scheduler wait, where measured.
For `n` choices distinguish request-level first token from individual choice
completion; total tokens divided by whole request time is not per-choice TPOT.

Current code evidence: `infrastructure/inference/local_chat_client.py` awaits a
complete JSON response and records `latency_ms` plus token counts. True TTFT/ITL
requires streaming or correlated server traces. Add that instrumentation behind
the experimental adapter; legacy non-streaming behavior stays supported. Missing
metrics display unavailable with a reason, never zero. Server-wide histograms
must not be presented as the current request's timing.

Use local monotonic clocks for local durations and request IDs for correlation.
Do not subtract H200 monotonic timestamps from Mac/browser timestamps. Exact
cross-host one-way timing requires calibrated clocks/uncertainty; otherwise show
client round-trip and server durations without falsely naming their residual
"network time". Overlapping stages are spans, not necessarily additive totals.

Recent-run summaries: median/p95 response latency, mean valid candidates per
decision, missing/stale/fallback duration, and valid on-time musical seconds per
wall-clock second. Show sample counts and canceled/timed-out/no-valid-result
counts, not just latency for successful requests. Group by mode/config/backend
and record warm/cold state where known. Clear metrics must not silently Reset rap.

Export JSON event traces, CSV per-response/per-span metrics, and config/model/
policy versions where known. Preserve source timings and units for recomputation.

## 10. Delivery Sequence And Review Gates

Each implementation task includes failing tests, a minimal implementation, and
passing targeted tests before integration. Do not restart H200 during planning.

1. **Contract and deterministic timing tests:** introduce versioned musical spans,
   readiness/commit events, and a controllable producer/clock. Prove bar, beat, tick,
   and uneven chunks are interchangeable at the controller boundary.
2. **Cyclic controller and real adapter:** implement USER/WAIT/MODEL with the existing
   capture/render/audio path. Demonstrate repeated audible cycles, variable WAIT,
   leading rests, and incremental MODEL deadlines. Review behavior before UI polish.
3. **Continuous controller and shared ownership:** expose experimental steering
   behind the same span contract; preserve legacy operation and one active owner.
   Test cross-page commands and mode switches while stopped.
4. **Instrumentation and export:** add request/choice timing where supported,
   candidate/renderer timelines, latency/coverage summaries, and correlation.
   Validate metrics against deterministic traces before drawing UI numbers.
5. **Clean inference page:** implement only the specified controls/performance/
   timing view. Test desktop/mobile, keyboard input, refresh/reconnection, and
   duplicate tabs against the shared session.
6. **Mac-H200 acceptance and handoff:** after GPU availability/authorization is
   checked, run real lyric/vocal playback in both modes and original-page regressions.
   Deliver a short user guide, trace examples, and the implementation-verified
   inference-engineer handoff specified in section 13. Adapter replacement without
   UI changes is supported, but the engineer is also free to edit the UI and all
   other components serving these two modes.

## 11. Acceptance Matrix

- At least five consecutive real cyclic exchanges finish without manual rearming.
- Fake-clock first prefixes arriving at non-beat times (e.g. 137 ms and 1.37 s)
  start MODEL at the first schedulable audio boundary, not the next beat/bar.
- A one-tick playable prefix can start MODEL in the test producer; later chunks
  arrive progressively. No full-bar readiness condition is hidden in the UI/controller.
- Unordered chunks, duplicate IDs, intentional leading rests, and revision races
  preserve placement. A later chunk cannot masquerade as a ready prefix.
- Delayed MODEL spans produce explicit missed duration/drums-only output without
  moving phase boundaries or leaking response audio into the next USER phase.
- No configured WAIT exists. Timeout/disconnect is an error, not a fake deadline
  failure; Stop during WAIT remains responsive and stale completions are ignored.
- Identical musical output split into different packet counts produces identical
  coverage metrics. Candidate yield remains tied to decisions, not packets.
- Known traces reproduce all metric formulas; missing instrumentation stays null.
  TTFT is verified on an instrumented real request, or explicitly documented as
  unavailable for the deployed backend rather than claimed as measured.
- Steering visibly distinguishes latest requested from currently performed input
  and preserves immutable audio with the active adapter's advertised granularity.
- Continuous PREPARING starts from the ready prefix without beat quantization;
  Stop/Start retains intent/history without replaying canceled output. Test a
  steering replacement racing exact commitment and incompatible fallback coverage.
- Opening either page does not create a session or reset playback. Original modes
  remain usable; shared Stop/Reset works and competing mode ownership is prevented.
- Refresh never restarts audio. Duplicate tabs cannot duplicate keyboard taps.
  Owner loss follows the phase-specific stop rules; lease recovery/expiry tests
  use a fake clock and verify no old-cycle input is accepted.
- Real Mac-H200 traces include generation, validation, rendered vocals, and client
  playback. Deterministic fine-grained fixtures are labeled as fixtures, not live
  beat/tick inference benchmarks. Physical audibility requires a listening check.
- The final engineering handoff matches the delivered source and tested setup,
  identifies mode-specific/shared files, and explicitly permits end-to-end changes
  to both inference-page modes. A proposal copied from this plan is not sufficient.

## 12. Review Decisions Still Needed

The confirmed mode semantics above are settled. Review the proposed defaults:
keyboard-flow USER input, one USER/two MODEL bars at 60 BPM, empty
capture repeating USER, bar-boundary Stop, and stopped-only tempo changes on this
page, plus repeating captured flow when USER and MODEL lengths differ. Backend
optimization remains free to change granularity and buffering,
provided those choices are advertised, measured, and preserve the mode contract.

## 13. Required Final Inference-Engineer Handoff

Added November 11, 2026 at the user's request. Produce at the end of implementation:
`docs/developer-guide/inference-optimization-handoff.md`.

Write it from inspected final code and observed execution, not the intended
architecture alone. Clearly distinguish implemented, tested, unsupported, and
planned capabilities. It must contain:

1. **Orientation:** the two modes, shared session, page/entry points, what runs on
   the Mac versus H200, and how the experimental modes can become official.
2. **Source map:** actual directories/files, important classes/functions, callers,
   and responsibilities from UI/API through controllers, prompts/generation,
   validation/ranking, TTS/alignment, transfer, scheduling, audio, and telemetry.
   Mark each entry mode-specific, shared by the two modes, or also used by legacy
   modes. Include relevant tests, scripts, configs, model assets, and output paths.
3. **Actual execution flows:** a startup and one complete exchange/steering update,
   with concrete request/result examples, buffering/commit points, threading or
   async ownership, cancellation, and failure/late-result behavior.
4. **Units and interfaces:** distinguish turn, generation, delivery, and commitment
   granularity; document active backend limitations, version/epoch handling,
   musical versus wall-clock time, and supported stream/result schemas.
5. **Optimization freedom:** explicitly state that every component serving both
   modes can be edited, including the UI and audio path, not just LLM serving.
   Identify safe mode-scoped changes and how shared changes can affect legacy
   behavior. This is a staging implementation, not a permanent isolated fork.
6. **Metrics:** exact definitions, timestamp origins and clock domains, per-request
   versus aggregate measurements, formulas, unavailable fields, and export schemas.
   Explain how to compare changes without gaming chunk counts or excluding failures.
7. **Run and test instructions:** tested Mac/H200 commands, SSH forwarding, ports,
   dependencies, backend selection, deterministic fixtures, real-model smoke runs,
   and legacy regressions. Give repo-relative source paths for portability and
   separately label verified machine-specific deployment paths. No credentials.
8. **Evidence and caveats:** baseline configuration/revision where available, test
   results and trace locations, known limitations, and unverified claims. The
   document must say whether services were left running; do not assume old health.

Verification gate: resolve every listed source path against the final tree, check
the cited symbols and call flow, and verify commands against the tested environment.
Keep the new document distinct from the user guide and from this requirements plan.
