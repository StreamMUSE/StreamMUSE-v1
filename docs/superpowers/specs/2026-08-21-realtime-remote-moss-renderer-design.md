# Realtime Remote MOSS Renderer Design

Date: 2026-08-21

## Status

Approved architecture for adding an offline-quality MOSS forced-alignment
renderer to the realtime rap demonstration while preserving the existing local
eSpeak renderer as a selectable mode.

This design supersedes only the H200 text-only and isolated-syllable-only
decisions in `2026-08-14-realtime-rap-audio-design.md`. The Mac remains the
authoritative playback clock, and the existing eSpeak path remains supported.

## Context

The current realtime system sends lyric-generation requests to an H200-hosted
vLLM server, evaluates and selects candidates on the Mac, synthesizes every
syllable independently with eSpeak, and places those syllables directly at
their flow-slot sample offsets. This provides deterministic timing and robust
fallback behavior, but independent syllables sound robotic and disconnected.

The successful offline experiments use a higher-quality phrase pipeline:

1. MOSS-TTS synthesizes a coherent two-bar utterance.
2. Montreal Forced Aligner (MFA) measures word and phoneme locations.
3. Syllable onsets are associated with the required MCFlow targets.
4. Rubber Band R3 applies a pitch-preserving continuous time warp.
5. Exact-duration vocal chunks are assembled and mixed with drums.

Running candidate generation, selection, and phrase rendering as separate
Mac-to-H200 operations would insert an unnecessary synchronization boundary.
The approved design moves candidate evaluation and selection to an H200 chunk
orchestrator. One request generates, evaluates, selects, synthesizes, aligns,
and warps a future chunk before returning its selected lyrics, diagnostics, and
audio to the Mac.

## Goal

Allow the realtime demonstration to switch between:

- `espeak`: the current local, isolated-syllable renderer;
- `moss_aligned_remote`: an H200-hosted, two-bar MOSS + forced alignment +
  Rubber Band renderer that approximates the quality and timing method of the
  current offline system.

Both modes must produce immutable, exact-duration audio scheduled by the same
Mac sample clock. Missing or late remote audio must never stop the clock or
produce an avoidable gap.

## Non-Goals

This change does not add:

- live drum or microphone input;
- browser-owned audio playback;
- interactive renderer or tuning controls in the website;
- arbitrary cloud deployment or public authentication;
- audio streaming syllable by syllable from the H200;
- automatic claims that a result is perceptually good based only on alignment
  metrics;
- replacement of the existing eSpeak implementation.

## Approved Decisions

1. The Mac remains the authoritative musical and sample clock.
2. Remote MOSS audio is prepared as an entire two-bar vocal chunk, not emitted
   tick by tick.
3. The H200 performs candidate generation, prosody analysis, evaluation,
   selection, MOSS synthesis, forced alignment, and Rubber Band warping within one
   externally visible chunk operation.
4. Internal H200 components remain independently testable and configurable.
5. The request includes both bars' complete flow schedules, topic context,
   lyric history, timing, and candidate-budget policy.
6. The response includes the selected lyrics, scoring and rejection
   diagnostics, stage timings, alignment diagnostics, and exact-duration mono
   vocal audio.
7. Drums remain local so the Mac can use the same drum renderer and mix policy
   in both modes.
8. The Mac prepares an eSpeak fallback independently of the remote request.
9. MOSS audio is eligible only if it is valid and arrives before the immutable
   playback commitment deadline.
10. Initial deployment uses an SSH-forwarded private HTTP service. The H200
    binds the orchestrator to loopback unless explicitly configured otherwise.
11. Candidate and rendering budgets are H200 configuration profiles with
    optional per-request overrides. Moving selection to the H200 must make
    adaptive candidate generation easier, not hide its behavior.

## System Architecture

```text
Mac realtime process
  Scenario + history + two future flow schedules
                 |
                 | one render-chunk request over SSH-forwarded HTTP
                 v
H200 RapChunkOrchestrator
  CandidateGenerator --------> local vLLM process / GPU
          |
  ProsodyAnalyzer
          |
  CandidateEvaluator + Selector --------> CPU
          |
  MossPhraseSynthesizer ----------------> MOSS worker / GPU
          |
  ResidentMmsForcedAligner -------------> GPU (MFA remains offline reference)
          |
  RubberBandTimeWarper -----------------> CPU
          |
  ResultPackager
                 |
                 | one result: manifest + mono vocal WAV
                 v
Mac realtime process
  validate duration and identity
          |
  split into exact bar buffers
          |
  mix local drums
          |
  commit immutable bars
          |
  sample-clock playback + WAV recording + monitoring
```

The vLLM and MOSS workers may use different GPUs. The orchestrator is a thin
composition layer and does not require both models to share a process or GPU.

## Client Strategy Boundary

The application must model the true unit of work: preparation of playable
future audio, rather than only synthesis of one syllable.

```text
RapChunkPreparationStrategy
  prepare(request, deadline) -> PreparedRapChunk

LocalEspeakChunkStrategy
  existing remote text generation
  + existing local candidate evaluation
  + existing per-syllable eSpeak rendering

RemoteMossChunkStrategy
  one remote orchestrator request
  + response validation
  + local drum mix
```

`PreparedRapChunk` contains two `PreparedBarAudio` values with exact frame
counts, committed lyric and flow metadata, renderer provenance, warnings, and
research diagnostics. The playback service consumes this common contract and
does not know which strategy produced it.

Renderer selection is startup configuration, initially exposed as:

```text
--rap-audio-renderer espeak
--rap-audio-renderer moss_aligned_remote
```

The website continues to expose only Start, Stop, and Reset. It displays the
configured renderer and its diagnostics but cannot change it during a run.

## Remote Request Contract

One request describes exactly one two-bar chunk and includes:

- protocol and schema version;
- session ID, chunk index, and deterministic idempotency key;
- BPM, meter, ticks per beat, sample rate, and exact expected frame count;
- topic and optional topic-transition context;
- previously committed lyrics and rhyme/continuity context;
- both materialized flow schedules, including ticks, target stresses,
  boundaries, rhyme groups, and template provenance;
- candidate policy profile and optional safe tuning overrides;
- remaining client render budget in milliseconds;
- model aliases and reproducibility seed policy.

The Mac sends a remaining-duration budget rather than relying on synchronized
wall clocks. The H200 uses its own monotonic clock from request receipt. The Mac
still enforces the final deadline because network return time is outside the
server's direct control.

Malformed, unsupported, or internally inconsistent schedules are rejected
before model work begins. A request with a reused idempotency key and identical
body returns the cached completed result. Reusing the key with a different body
is an error.

## Candidate Generation And Selection

The orchestrator calls the existing OpenAI-compatible vLLM endpoint locally,
so candidate payloads do not travel through the SSH tunnel. Existing prosody,
syllable validity, stress, boundary, rhyme, topic, continuity, and novelty
logic is reused from the StreamMUSE package rather than copied into server-only
scripts.

Both flow schedules are included in the generation context. The first version
generates candidate pools for each bar independently and applies a final
pair-level continuity/rhyme score before synthesis. This preserves the proven
single-bar validity gate while allowing the selected pair to form one coherent
MOSS utterance.

Candidate generation is budget-aware:

1. Reserve a configurable render budget for MOSS, alignment, warping, packaging, and
   return transfer.
2. Spend only the remaining budget on candidate generation.
3. Generate candidates in configurable independent waves.
4. Evaluate each completed wave immediately.
5. Stop when the policy's valid-count and quality thresholds are met or when
   the generation budget expires.
6. If at least one valid candidate exists, select the best available result.
7. If no valid result exists, render a prevalidated server-side lyric only when
   the policy explicitly permits it; otherwise return a structured failure and
   let the Mac use its eSpeak fallback.

Profiles hold defaults such as model, candidate wave sizes, sampling
parameters, score weights, minimum valid candidates, quality threshold, and
stage budget reservations. Every resolved value is returned in the manifest.

## Remote Audio Pipeline

For the selected lyric pair, the H200:

1. Builds the same two-bar render contract used by the offline experiment.
2. Synthesizes the complete text as one connected MOSS utterance.
3. Validates that the waveform is finite, non-silent, and within configured
   source-duration limits.
4. Runs the resident torchaudio MMS forced aligner against the exact selected
   transcript. MFA remains available as the offline/reference aligner because
   a measured warmed MFA CLI run took 49.63 seconds for two phrases, while a
   warmed resident MMS pass took 0.037 seconds for the same two phrases.
5. Maps MMS character/word spans to planned syllable onsets using the existing
   CMU word/syllable analysis. Unmatched words use an explicit
   transcript-proportional fallback and produce warnings.
6. Exposes the aligner behind one injected anchor contract so MFA-derived phone
   anchors can still be used for offline A/B comparison.
7. Constructs a strictly monotonic source-to-target onset map.
8. Applies the current continuous Rubber Band R3 pitch-preserving warp.
9. Trims or pads only at the outer chunk boundary to the exact expected frame
   count.
10. Returns a mono PCM16 WAV plus diagnostics. It does not mix drums.

The initial mode intentionally matches the accepted non-stress-augmented
offline configuration. Stress metadata remains available for later controlled
experiments but does not introduce a new gain envelope in this implementation.

## Response And Transport

The successful response is one binary package containing:

```text
manifest.json
vocals.wav
```

The first implementation uses an in-memory ZIP response with media type
`application/vnd.streammuse.rap-chunk+zip`. This avoids base64's size increase
and keeps the operation to one request and one response. At 90 BPM, two bars
last 5.333 seconds; 24 kHz mono PCM16 requires approximately 256 KiB before
container overhead. The Mac resamples to its 48 kHz stereo float mix format
after validation.

`manifest.json` includes:

- request identity and resolved configuration;
- selected text and complete syllable/flow assignments;
- requested, parseable, valid, and selectable candidate counts;
- selected score and component scores;
- bounded top-candidate and rejection summaries;
- generation, evaluation, MOSS, aligner, warp, packaging, and total server timing;
- aligner identity/confidence/fallback counts, source and target anchors, local warp ratios, and
  warnings;
- WAV format, frame count, duration, peak, and SHA-256 hash;
- model and tool versions required for reproduction.

Large full-candidate ledgers remain server-side artifacts keyed by request ID.
The response carries enough summary data for live monitoring without making
audio transfer depend on an unbounded diagnostic payload.

## Realtime Scheduling

At 90 BPM in 4/4, a two-bar chunk lasts approximately 5.333 seconds. MOSS mode
uses chunk-level rolling preparation:

1. Before Start, prepare the first remote chunk and its local fallback.
2. Begin playback only after the first chunk is ready or the user-visible
   startup attempt has failed and fallback is ready.
3. While chunk `N` plays, request chunk `N+1`.
4. Prepare local fallback bars concurrently with the remote request.
5. Validate and split returned vocals at deterministic bar frame boundaries.
6. Mix each bar with its own active flow template and local drums.
7. Commit the next chunk before the existing guard interval.
8. Discard and record any remote result that arrives after commitment.

The audio queue remains at most one two-bar chunk ahead. Server-side candidate
generation and audio rendering share the available chunk deadline and can tune
their internal budget without additional Mac round trips.

The first H200 benchmark determines whether the complete remote operation fits
reliably inside the 5.333-second rolling budget. If it does not, the runtime
must remain correct through eSpeak fallback while the experiment evaluates
larger pre-roll, a longer planning horizon, smaller candidate waves, persistent
MOSS workers, and parallel GPU allocation. It must not silently extend the
musical clock.

## Commitment And Fallback

The Mac owns the final decision:

```text
REMOTE_PENDING + LOCAL_FALLBACK_READY
               |
       remote valid and on time?
          /                 \
        yes                  no
         |                    |
COMMIT_MOSS_CHUNK      COMMIT_ESPEAK_FALLBACK
```

A remote result is accepted only when:

- request identity and schema match;
- selected lyrics satisfy the returned flow assignments;
- WAV hash, format, and exact frame count validate;
- samples are finite and not unexpectedly silent;
- alignment diagnostics pass configured hard safety limits;
- both bars can be mixed before the immutable deadline.

HTTP errors, timeouts, model failures, missing alignments, invalid warp maps,
malformed packages, late responses, and validation failures all retain the
local fallback. Remote failures do not return a successful silent WAV. The
terminal, website, and session logs display the fallback reason.

Retries use the same idempotency key and may retrieve a completed cached result
without repeating inference. The Mac never retries beyond the useful musical
deadline.

## Monitoring And Artifacts

Live monitoring exposes:

- renderer mode and H200 endpoint status;
- chunk state: requested, generating, rendering, returned, accepted, committed,
  late, failed, or fallback;
- request budget, elapsed time, estimated slack, and final deadline slack;
- candidate counts and selected score components;
- selected two-bar lyric and complete visible flow schedules;
- per-stage server timings;
- alignment fallback and warp warnings;
- transfer bytes and transfer duration;
- committed renderer for each bar;
- playback underruns independently from renderer misses.

The Mac session log stores every returned manifest, acceptance decision, and
audio hash. Optional debug configuration stores returned vocal chunks. The
H200 stores complete request manifests, candidate ledgers, source MOSS WAVs,
TextGrids, aligned WAVs, and failure records in a request-ID-addressable run
directory with retention configurable outside the realtime protocol.

## Deployment

The H200 deployment contains:

- the existing vLLM OpenAI-compatible server;
- a persistent MOSS worker with its model and reference voice loaded;
- resident torchaudio MMS alignment and Rubber Band runtime environments, plus
  MFA for offline/reference experiments;
- the StreamMUSE package at the same protocol revision as the Mac;
- the new FastAPI chunk orchestrator bound to `127.0.0.1`;
- explicit GPU assignments so vLLM and MOSS do not unexpectedly contend.

The Mac opens an SSH local forward to the orchestrator. Startup performs a
health and compatibility check reporting schema version, code revision, model
availability, worker warm status, and supported profiles. A mismatch prevents
MOSS mode from starting but does not damage the eSpeak mode.

## Testing Strategy

### Unit Tests

- Request and response schema validation.
- Idempotency and cache identity behavior.
- Candidate-budget allocation and wave stopping.
- Pair selection and score diagnostics.
- Binary package creation, parsing, and hash validation.
- Exact chunk-to-bar frame splitting at representative tempos.
- Remote acceptance and every fallback branch.
- Late-result rejection and immutable commitment.
- Renderer configuration parsing.

### Integration Tests

- Fake vLLM, MOSS, aligner, and warper components exercise the full orchestrator.
- A fake remote endpoint exercises client request, package validation, drum
  mixing, and playback commitment without an H200.
- Existing eSpeak CLI, WAV, live, composite, terminal, and website workflows
  remain covered and unchanged.

### H200 Verification

- Cold and warm two-bar smoke renders using a known lyric and flow schedule.
- Repeated end-to-end measurements separating generation, evaluation, MOSS,
  alignment, warp, packaging, transfer, and Mac mix latency.
- Candidate-profile sweeps under the complete chunk deadline, not only LLM
  latency in isolation.
- At least one continuous 20-bar 90 BPM run recording MOSS acceptance rate,
  fallback rate, deadline slack, and playback underruns.
- Listening comparison using the same selected lyrics rendered through remote
  MOSS and local eSpeak.

## Acceptance Criteria

The implementation is complete when:

1. The CLI selects `espeak` or `moss_aligned_remote` without code changes.
2. Existing eSpeak tests and demonstrated workflows still pass.
3. One external remote request performs candidate generation through aligned
   vocal rendering and returns one auditable result package.
4. The H200 internally uses the shared StreamMUSE evaluator and returns its
   resolved candidate and budget parameters.
5. A valid remote chunk produces exact-duration bars on the Mac sample clock
   with the intended flow-slot alignment method.
6. Any remote error or missed deadline commits a prepared eSpeak fallback
   without changing tempo or inserting an application-level gap.
7. Terminal, website, and session artifacts identify the selected renderer and
   expose generation, scoring, alignment, transfer, and deadline diagnostics.
8. H200 measurements report warm latency distributions, MOSS acceptance rate,
   fallback rate, and playback underruns for a continuous 90 BPM run.
9. Setup and SSH-forwarding commands are documented and verified from a clean
   Mac client shell and the designated H200 environment.

## Assumptions

- The demonstration may perform a visible startup pre-roll before the musical
  clock starts.
- Prescheduled topics remain sufficient for this milestone.
- A dedicated or otherwise non-contending H200 GPU is available for MOSS during
  performance tests.
- Returning approximately 256 KiB per two-bar chunk is not the dominant
  latency; measurements will verify this rather than relying on the estimate.
- The accepted offline `continuous_onset_r3` implementation is the reference
  behavior for the first remote renderer.
