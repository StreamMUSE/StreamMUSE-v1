# Real-Time Generative Pipeline: Scheduling Model Design

Status: research-design draft for review. No scheduler superiority, performance prediction, or empirical distribution is assumed. This document does not gate delivery of the interaction/UI features in the companion design.

Code baseline: `feature/rap_optimization`, `f8194f87`. The current architecture SVG is unchanged.

## 1. Intent and Boundaries

Study scheduling for a causal, externally timed, interactive generative pipeline. Rap is the fixed application used to instantiate the model; replacing rap with an easier task is not a scheduling improvement. The scientific objects are release rules, dependent computation, resources, revision validity, deadlines, commitment, and observable output.

The previous [StreamMUSE paper, Section IV](https://arxiv.org/html/2606.11886v1) relates inference interval, generation length, tick duration, and round-trip latency to configuration feasibility. Retain the distinction between physical time and application time. Do not import its fitted inference-cost or network-noise distributions into this pipeline without measurements.

Deliverables for this theory work:

1. A typed parameter and state registry, with units and implementation status.
2. A dependency/resource model with explicit causal and timing relationships.
3. An observation contract sufficient to estimate and test those relationships.
4. A mapping from the existing implementation to the model.
5. A coverage checklist for later experiments, without predictions or a chosen optimal policy.

Non-goals today: claiming novelty or schedulability guarantees; selecting a winning scheduler; assuming independent candidate validity or latency; implementing every modeled alternative; changing the existing system diagram before the model is reviewed.

The separately requested keyboard interaction remains an engineering deliverable. Its first version captures one complete bar, starts generation, echoes the captured rhythm while waiting, and starts a progressive response when its first playable bar is ready. Other input-release/turn policies may be modeled without being implemented in that feature.

## 2. Variable Classes and Registry Contract

Every entry has one primary class:

- **Control:** deliberately configurable mechanism, policy, or budget.
- **Workload/environment:** external input trace or deployment condition. Controlled in an experiment only when explicitly injected or provisioned.
- **State:** changes during execution, such as queue contents, available memory, and available audio coverage.
- **Derived/observed:** computed or measured result, such as latency, miss rate, cache-hit rate, or candidate-validity rate. Do not present these as directly settable controls.

Each registry entry records: stable name; definition; units; type/domain; scope (run, session, job, revision, stage, resource); class; when it can change; dependencies; current value/source or `unknown`; observation method; clock/uncertainty; and status (`implemented-configurable`, `implemented-fixed`, `requires-engineering`, `model-only`).

Separate configuration versions from content revisions and timeline versions. A run manifest stores requested and resolved configuration, code/model/tool versions, seeds, workload trace identity, hardware, and parameter changes with their effective boundaries.

Parameterizing a quantity does not make it independent, cheap to change, or appropriate for a UI control. Invalid combinations need explicit constraints.

## 3. Granularity Is a Vector

Use physical units for costs and application units for semantic spans. Do not silently equate tokens, beats, ticks, PCM samples, or callbacks.

| Quantity | Meaning | Example domains |
|---|---|---|
| Input observation resolution | Smallest observable input timestamp/event distinction | device samples, milliseconds, application ticks |
| Input aggregation span | Amount of input accumulated into one update | event, tick, beat, bar, turn |
| Input release rule | When observed input becomes eligible for generation | immediate, periodic, boundary, full-span, explicit commit |
| Submission interval | Spacing of requests, separate from their coverage | milliseconds or ticks; event-triggered alternative |
| Replanning cadence | When priorities, budgets, and revisions are reconsidered | event, tick, beat, bar, timer |
| Context span | Observed history available to a request | tokens, lines, beats, seconds |
| Generation horizon | Future output span requested from one generation job | part-bar, bar, multiple bars |
| Candidate/selection unit | Span scored and chosen atomically | partial line, line, multi-line phrase |
| Text commitment unit | Amount of selected text made immutable | token, word, phrase, bar |
| TTS input/output units | Text accepted and audio yielded per synthesis step | word, phrase, bar; audio chunks |
| Alignment unit | Span required before usable onset estimates exist | chunk, phrase, bar |
| Warp/mix unit | Span transformed before playback preparation completes | audio block, phrase, bar |
| Transfer unit | Application payload available before consumption | record, audio packet, bar package |
| Artifact/cache unit | Smallest independently reusable intermediate result | prefix, selected lyric, source audio, rendered span |
| Commit unit | Output made immutable at one scheduling decision | tick, beat, bar, chunk |
| Playback scheduling resolution | Smallest location where output can be placed | sample, callback block, tick |
| Cancellation/preemption unit | Earliest point where computation can actually stop | iteration, token, kernel boundary, complete request |
| Measurement/display resolution | Smallest event observable or displayed | timestamp precision, tracing period, UI refresh |

These quantities have phase as well as size. A beat-sized commit unit does not imply immediate response to an edit that arrived just after a beat boundary.

Current v3 is bar-granular for chosen lyrics, remote synthesis packages, and commitment. Its template grid has 16 ticks per bar and its audio is sample-indexed. It is therefore not tick-responsive merely because it displays ticks or places audio at sample indices.

Beat/tick-granular generation and commitment are model dimensions, not existing flags. They require compatible partial-output contracts, semantic dependencies, and render/selection boundaries. One can retain bar-sized generation while consuming smaller units, or use finer requests while retaining bar-sized commitment; neither arrangement should be mislabeled as the other.

## 4. Parameter Coverage Catalogue

### 4.1 External Clock and Output Obligations

- Tempo trajectory, meter, tick resolution, phase origin, application-time/physical-time mapping.
- Tempo-change arrival time, effective boundary, step/ramp shape, allowed slew, and notice.
- Clock source, precision, offset, drift, synchronization uncertainty, timer jitter, audio-device latency, and calibration.
- Required output intervals and their release/commit/playback boundaries; scheduled silence versus an unintended gap.
- Deadline class: playback availability, incorporation of an input revision, acoustic alignment tolerance, or completion of a whole response.
- Hard requirement versus probabilistic service target; target percentile/probability is specified, not inferred from a mean.

### 4.2 Input Workload and Information Availability

- Event/update trace; arrival frequency, burstiness, clustering, correlated edits, and idle periods.
- Input completeness, aggregation, release phase, quantization, debounce, boundary handling, late/out-of-order events, and batching.
- Input history/context size, flow density, output length, and other task-preserving workload descriptors.
- Revision scope and frequency; which already-planned spans an edit invalidates.
- Input finalization/confirmation rules and whether speculative work may use incomplete but observed input.
- Availability watermark for each dependency, not merely a single last-input timestamp.
- Turn/phase lengths and overlap, including replay/wait intervals, modeled independently of the first feature implementation.

### 4.3 Job Graph and Causality

- Pipeline stages, fork/join barriers, cross-unit dependencies, streaming edges, and artifact validity requirements.
- Dependence on earlier selected text, actually committed text, rendered audio, or actually played audio.
- Whether a downstream stage can start from partial output; minimum usable partial span.
- Per-session ordering, cross-session independence, continuation/prefix dependency depth.
- Model state/KV cache ownership and whether jobs mutate shared state.
- Recompute scope after an update and whether completed intermediates can be reused.

### 4.4 Generative Work and Serving

- Model identity/size, precision, inference backend, context length, output-token limits, stopping rules, and prompt format.
- Candidate count per wave, number of waves, rescue sizes, retry layers, and stopping thresholds.
- API `n`, requests in flight, sequences concurrently decoded, and server batch size as separate quantities.
- Sampling configuration, diversity policy, seeds, duplicate removal, and cross-candidate/wave dependence.
- Prefill/decode work, prefix-cache policy, KV-cache capacity, batching delay, batch admission policy, maximum sequences/tokens, and warmup.
- Structured-output/parsing policy and failure handling; partial output availability versus completion of an entire response.
- Work budget expressed separately in candidates, tokens, elapsed time, and compute demand.
- Multiple retry layers must be composed: an 80-candidate writer attempt is not necessarily an 80-candidate lifetime limit for a bar.

### 4.5 Validation and Selection

- Validation stages, their order, short-circuiting, parallelism, lookup caches, and per-candidate work.
- Validity rules, minimum number of acceptable choices, score weights/thresholds, tie-breaking, and selection deadline.
- Whether selection waits for the full batch, an acceptable count, or a budget boundary.
- Scoring/analysis algorithms and any model-assisted evaluation cost, if present.
- Validity counts, selectable counts, score distributions, duplicate rate, and useful-output yield are outcomes conditional on workload/configuration.
- Holding task requirements fixed is distinct from relaxing acceptance criteria; log either choice explicitly.

### 4.6 Synthesis, Alignment, and Audio Preparation

- Renderer/profile, voice/reference, synthesis span, conditioning prefix length, continuation policy, and source-audio duration.
- Whether synthesis exposes incremental usable audio; model/vocoder buffering and lookahead if applicable.
- Alignment span, algorithm/backend, device placement, fallback policy, confidence threshold, and alignment work.
- Warp policy, anchor selection, duration constraints, timing tolerance, stretch bounds, retries, and trim/pad policy.
- Audio format, sample rate, channels, resampling, mixing granularity, gain/limiter cost, and encoding/decoding cost.
- CPU/GPU transfers and intermediate storage; subprocess launch and file I/O overhead.
- Quality and fidelity constraints remain explicit. Changing them is not an unreported scheduling improvement.

### 4.7 Admission, Queues, and Scheduling Policies

- Admission control, maximum in-flight jobs by stage/session/resource, queue capacities, and backpressure.
- Dispatch priority: order, deadline, slack, stage, revision freshness, user/session priority, or a specified combination.
- Priority propagation across dependencies, shared locks, and non-preemptible work.
- Coalescing/debouncing updates, dropping obsolete queued work, and retaining useful older artifacts.
- Stage overlap, pipeline depth, speculative horizon, and maximum speculative work.
- Preemption/checkpoint availability, cancellation request latency, actual cancellation latency, and resume/recompute cost.
- Resource reservation, fairness, budget replenishment, overload behavior, and starvation protection.
- Adaptive policy decision period, estimator window, information delay, hysteresis, and decision overhead. No adaptive policy is presumed better.

### 4.8 Hardware and Shared Resources

- Number/type of CPUs and GPUs; placement of each stage; resource shares and affinity.
- CPU/GPU memory capacity, bandwidth, transfer paths, KV residency, fragmentation, and memory pressure.
- Worker/thread/process counts, shared locks, runtime/event-loop contention, garbage collection, and allocator behavior.
- Thermal/power state, clock scaling, kernel/graph warmup, model loading, and cold/warm execution.
- Competing sessions or unrelated load, contention traces, and isolation/reservation mechanism.
- Exclusive versus concurrent execution compatibility between stages; CPU preprocessing or synchronization that prevents apparent GPU overlap.

### 4.9 Transport and Serialization

- Client/server placement, connection topology, SSH forwarding, connection reuse, setup/reconnection, and pool limits.
- Separate control/data paths, head-of-line blocking, concurrent streams, priorities, and flow control.
- Payload sizes, representation, codec/bitrate/compression level, serialization, checksums, and copies.
- Link bandwidth, propagation/RTT, jitter, loss, retransmission, outages, and time-varying/correlated conditions.
- Polling/long-poll limits, push notifications, HTTP timeouts, retries, and cancellation propagation.
- Server waiting time inside an HTTP request must not be labeled network transfer time.

### 4.10 Buffers, Caches, and Versions

- Separate target horizons for lyrics, source audio, prepared audio, and fetching; ready coverage versus intended coverage.
- Cache capacity/TTL/eviction, key definition, validation cost, and storage location.
- Semantic, source-audio, timing, voice, and renderer dependencies of each cache key.
- Whether obsolete outputs remain eligible as fallback; maximum age/version lag and compatible revisions.
- Artifact identity, generation epoch, idempotency, deduplication, late-result rejection, and reordering tolerance.
- Hits, misses, occupancy, invalidations, and useful reuse are measured state/outcomes, not independent knobs.

### 4.11 Commitment and Playback

- Commit granularity/phase, early-freeze margin, preparation reserve, and downstream device buffering.
- Immutable committed region, replacement policy before commitment, and handling of deadline changes.
- Sample clock, callback block size, ring-buffer capacity, device format, and underrun behavior.
- Startup priming depth/timeout, stop boundary, resume policy, retained queue/state, and reset epoch.
- Fallback order and eligibility, replay policy/limit, silence/drums-only behavior, and whether fallback is prepared proactively.
- Completion requirements for first response unit versus entire response; no hidden waiting by renaming replay as successful fresh output.

### 4.12 Instrumentation and Experiment Conditions

- Trace detail, sampling frequency, log serialization, flush policy, retention, UI refresh, and observer overhead.
- Timestamp source, cross-host conversion/uncertainty, audio callback observation delay, and measurement perturbation.
- Workload replay, random seeds, initial state, cache warmness, experiment duration, ordering, and resource load controls.
- Definition of a successful unit, observation denominator, timeout/cancellation censoring, and failure accounting.
- Distribution estimation windows, correlations, nonstationarity, sample size, confidence intervals, and validation split.
- Measured acoustic onset versus waveform-derived estimate versus playback-cursor observation; never interchange these labels.

## 5. Structural Relationships

These define the model, not empirical performance laws.

### 5.1 Musical Time and Physical Deadlines

Let `b` denote musical position in beats and `B(b) > 0` the configured tempo at that position. The planned time mapping is:

`T(b) = t0 + integral[b0,b] 60 / B(u) du`.

For constant tempo and a span of `g` ticks with `r` ticks per beat, the span duration is `60*g/(r*B)` seconds. With tempo changes, use the actual piecewise mapping rather than multiplying the whole history by the latest BPM. Device sample positions use `round(fs * (T(b)-t0))`, with a specified clock origin and rounding rule.

An output unit `j` has planned audible onset `P_j`, duration/span, and final preparation deadline `D_j = P_j - M_j`. Define what the margin `M_j` includes (freeze guard, enqueue/device buffering, uncertainty) to avoid double-counting. Upstream stage deadlines may reserve time for remaining stages. Tempo changes create a new timeline version; committed history is not retroactively moved.

### 5.2 Jobs, Release, and Dependencies

A job is `(session, output_span, content_revision, timeline_revision, stage, attempt)`, not merely a bar number. Its graph includes required input snapshots and resource demands.

For a non-preempted stage `v` of job `j`:

- `R_jv = max(policy_release, required_input_available, required_predecessor_finishes)`.
- `S_jv >= R_jv`, subject to the chosen resource schedule.
- `Q_jv = S_jv - R_jv` is queue/resource wait.
- `F_jv = S_jv + C_jv` is finish time for measured service time `C_jv`.

For preempted work, record service segments plus suspension/resume overhead rather than pretending wall duration equals uninterrupted service. Streaming edges release downstream work after the required chunk, not necessarily the entire upstream job. A resource model must enforce capacity and concurrency compatibility, not just assign each stage an independent mean time.

`C_jv` is a conditional quantity depending on workload, configuration, resource state, batch membership, cache state, and concurrent work. Its functional form and stochastic dependence require data. Pipeline latency follows the dependent schedule/critical path; it is not the sum of independently measured stage percentiles.

### 5.3 Causality, Validity, and Commitment

Every generation/selection uses a recorded input watermark and artifact dependency set. No job may claim to have incorporated unobserved future input. Speculation may use incomplete observed input, but must be identified as such.

At `D_j`, the scheduler chooses an artifact that is both prepared and eligible under the revision/fidelity policy. A ready but incompatible artifact is not a successful latest-revision response. Record fallback class separately from continuity.

An edit arrival, server acceptance, replacement lyric, rendered audio, client preparation, commitment, and audible influence are distinct times. Maintain their provenance chain. Tempo, topic, flow, renderer, and device changes need not invalidate the same stages.

### 5.4 Readiness, Slack, and Coverage

- Realized final-stage lateness: `F_j - D_j`; report misses for the relevant obligation.
- Predicted remaining slack at time `t`: `D_j - t - estimated_remaining_latency(j,t)`; log estimator/version and error separately.
- Ready-buffer coverage is the union of eligible prepared output intervals. A far-ahead ready bar does not fill a hole immediately ahead of playback.
- A lead of `L` ticks corresponds to `T(b + L/r) - T(b)`, not a fixed number of milliseconds when tempo varies.
- Quantized release/commit boundaries add phase-dependent waiting. Finer audio samples do not remove coarser semantic release or commitment delays.

### 5.5 Resource Demand and Useful Work

For each resource, measure admitted work, executed service, queue wait, cancellation overhead, and capacity over the same interval. Include retries, backups, speculation, failed candidates, and invalidated jobs. Utilization alone is not a sufficiency test for meeting dependent deadlines.

Distinguish executed work that contributed to played output, reusable intermediates, and work ultimately discarded. Candidate count is not equivalent to useful yield; do not calculate `P(no valid) = (1-p)^N` without an independence justification.

## 6. Observation Contract

Record event identity, session/epoch, job/stage/attempt, content/timeline/configuration revision, dependency IDs, input watermark, planned output span, timestamps and clock identity, resource, policy decision/reason, queue depth, and payload/work size.

Required lifecycle timestamps: input observed; update submitted/accepted; job released; queued; started; partial output usable; finished; cancellation requested/acknowledged/completed; transfer started/completed; client prepared; selected; committed; sink enqueued; playback observed/completed. Missing events are missing data, not zero cost.

Outcome families, without a chosen scalar objective:

- First-input-influenced-output delay and full-response completion delay.
- Deadline miss rate and lateness, classified by stage and final obligation.
- Latest-revision service, age/version lag, and time until edits become audible.
- Valid/selectable candidate counts and selected score components.
- Audio continuity, fallback/replay/silence duration, and consecutive failure episodes.
- Intended/estimated/measured onset errors, with measurement confidence and clock uncertainty.
- Resource usage, transfer bytes, ready coverage, discarded/reused work, and instrumentation cost.

Report count-weighted and duration-weighted outcomes when output units have different lengths. Separate startup, steady state, edit bursts, tempo transitions, stop/resume, overload, and recovery. Changing granularity changes the denominator; per-unit misses alone are not comparable.

Cross-host monotonic timestamps cannot be subtracted directly. Use per-host spans plus a documented synchronization/offset-estimation method and uncertainty, or restrict conclusions to local round trips and correlated spans. GPU asynchronous work and callback observation delay require equally explicit measurement definitions.

## 7. Existing Code Mapping

| Mechanism | Current source | Model status |
|---|---|---|
| Fixed tempo mapping | `domain/timing/tempo.py`; `application/rap/audio_rendering.py` | Fixed per session; trajectory requires engineering |
| One-bar lyric/flow identity | `domain/rap/lyric_stream.py`; `domain/rap/flow.py` | One bar, 4/4, four ticks/beat |
| Writer budget and ranking | `application/rap/lyric_writer.py` | 32 initial + up to three 16-candidate rescue waves per attempt |
| Server ordering/horizons | `application/rap/lyric_session.py` | Writer/render workers; default lyric lead 3, audio lead 2; bounded retries |
| Client fetch/backup/commit | `application/rap/bar_stream.py` | Fetch window 2; one-tick freeze guard; separate prepared alternatives |
| Source audio and local preparation | `application/rap/bar_audio.py` | Per-bar source package; local warp/mix before commitment |
| Transport wait/transfer | `infrastructure/rap/lyric_stream_client.py` | WebSocket + long-poll HTTP; current request timing includes server wait |
| Audio-clock observations | `application/rap/playback.py` | Sample-indexed sink observations drive tick events |
| Settings/lifecycle | `presentation/rap_demo/cli.py`; `application/rap/runtime.py` | Some CLI controls; some fixed values; resume path already exists |

Paths above are relative to `src/streammuse/`. Values describe the inspected implementation, not guarantees or every deployment.

## 8. Theory Work Plan and Completion Gate

1. Review this inventory for missing scheduling dimensions; keep product requirements in the companion document.
2. Turn the registry contract into a machine-readable schema; populate current configuration and fixed assumptions with source references.
3. Specify the minimal event/job graph and timeline model, including units, invariants, and revision compatibility.
4. Map existing traces to the observation contract; list unavailable measurements and the instrumentation needed to obtain them.
5. Build a trace-driven or synthetic-cost scheduling simulator only after model review. Use measured service traces where available and explicitly synthetic inputs elsewhere.
6. Validate conservation of time/work, dependency order, causality, deadline accounting, and behavior at boundary cases before using simulations to make claims.
7. Design experiments from independent axes and fixed workload/task constraints. Do not propose outcomes until the measurement/model foundations are reviewed.

Theory acceptance: a reviewer can distinguish each granularity, calculate job eligibility and deadlines on a supplied trace, identify which costs are measured versus assumed, and explain why a reported miss/fallback/fresh response occurred. No claim of exhaustive physical modeling is made: unmodeled effects are named, measured as residuals where possible, and promoted to explicit variables if consequential.
