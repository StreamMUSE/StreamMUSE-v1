# Six-Track Real-Time LLM Research Day Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or
> superpowers:executing-plans when implementation is approved. This document is
> currently a proposed plan; unchecked items are not completed work.

**Goal:** Produce a defensible November 5 presentation with bounded progress on
all six ideas, measured examples, and a clearer research direction.

**Architecture:** Preserve the Mac client, H200 model services, v3 lyric/audio
protocol, and existing playback behavior. Add small optional modules for stronger
alignment rules, spoken input, story state, and logical character state; keep
concurrent collaboration in a separate experimental harness initially. Share
event/provenance conventions so results can be compared without a framework rewrite.

**Tech stack:** Existing Python rap modules, current web UI, sounddevice, existing
Qwen/MOSS services, MMS alignment and Rubber Band R3. Optional faster-whisper
for local utterance transcription. No model training or serving-engine migration.

**Spec:** [Captured requirements](../../research/2026-11-04-research-direction-notes.md).
**Research basis:** [Close reading](../../research/2026-11-04-realtime-llm-close-reading.md).
**Date:** November 4, 2026, America/Los_Angeles. Estimates below are elapsed time
from approved execution, not a promise about an unspecified presentation time.

## Delivery Agreement Proposed

| Idea | Minimum we aim to implement/test | What we show tomorrow |
| --- | --- | --- |
| 1. Stress alignment | New optional strong-anchor policy with explicit collision outcomes | Matched audio, timing overlays, collision table |
| 2. Microphone interaction | Push-to-talk, final transcript, spoken story direction; separate own-voice feasibility check | One live or recorded spoken-direction exchange and latency trace |
| 3. Voice consistency | Isolate source versus warp variation and tempo/continuation effects; fix only supported cause | Controlled listening comparison and diagnosis, even if unresolved |
| 4. Directed story | Persistent ordered plot events, distinct from replaceable topic/history | Short incremental story with 3 visible interventions and response delays |
| 5. Two characters | Shared model, isolated personas/histories, explicit shared observations | 8-12 turn dialogue/battle transcript; audio when integration is ready |
| 6. Concurrent collaboration | Small asynchronous counting harness with naive and coordinated baselines | Attempt/commit timeline, success/collision/latency results |

This is an approximately 8-hour elapsed pilot with parallel work, not six polished
product releases. All tracks receive an artifact. If a minimum prototype fails,
deliver the reproducible failure and its cause; do not relabel feasibility as a
working feature. Full-duplex voice, new model training, and custom inference
kernels are explicitly beyond this day.

## Global Constraints

- Preserve the old selectable alignment policies, ordinary continuous generation,
  keyboard exchange mode, drums/chords, output device, and live settings behavior.
- No newly added behavior becomes default until it passes regression and listening.
- Current audible/committed bars remain unchanged by future edits.
- Check H200 GPU ownership and availability before using previously authorized
  GPUs 0 and 1. Never evict another process. Serialize timed GPU experiments;
  concurrent implementation workers must not contaminate one another's benchmarks.
- Do not change model checkpoint, candidate policy, prompt, tempo, and alignment
  simultaneously in a comparison. Record every intentional difference.
- Use fresh words for listening tests, hide transcripts initially, randomize order,
  and reuse identical source PCM for alignment comparisons.
- Client-recorded schedule/device timestamps are not proof of acoustic arrival.
- Keep all timeouts, fallbacks, invalid candidates, and failed runs in the ledger.
- No automatic microphone recording or voice cloning. Recording requires the
  user's explicit action; use only their volunteered voice or an authorized reference.
- Git worktree metadata still references the old UGRIP path after the rename.
  Repair/verify linkage before implementation; do not reset, reclone, or discard
  existing edits. No runtime maintenance or Git repair was performed during review.

## Review Focus

1. Tempo/edit races must not play stale replacements or mutate committed audio.
2. Dense stress anchors must not silently delete weak syllables or claim exact
   timing when the policy relaxed an anchor.
3. Silence, microphone denial, ASR timeout, or speaker feedback must not generate
   repeated story edits or freeze playback.
4. Rescue generations and stale results must not erase plot facts or leak another
   character's private history.
5. Counting output must distinguish model proposals from coordinator acceptance;
   filtered success is not evidence that models coordinated unaided.

## Shared Evidence Contract

Prefer extending existing ledgers over introducing a second telemetry framework.
Each run gets a manifest and JSONL events with:

    run_id, condition, event_id, request_id, actor_id, bar,
    context_version, parent_event_ids, client_monotonic_time,
    stage, outcome, reason, artifact_path

Fields may be absent when inapplicable. Define IDs once in
`scripts/experiments/research_day/common.py` (new); do not require that production
code import experiment modules. Server spans remain server-local durations;
never subtract independent Mac/H200 monotonic timestamps.

Separate stages: input received, accepted instruction, request released, generation
started/completed, selection, source audio ready, transfer, local preparation,
commit, playback callback. Log cancellation requested and cancellation confirmed
separately; an abandoned HTTP result does not prove GPU work stopped.

For semantic edits, each fixture specifies what must become true and what must
remain unchanged. A context version is provenance, not a compliance score. Human
audit small pilots; any optional LLM judge is advisory and uses a fixed rubric.

Artifacts go in `output/research_day_20261104/<track>/<run_id>/`, with small
summaries/figures in `docs/research/`. Every WAV filename includes the excerpt or
song name, policy, tempo, and stem. Keep raw source, vocal-only, and mixed audio.

## Task 0: Baseline And Reproducibility

**Budget:** 30-45 minutes. **Owner:** integration worker.

**Reuse:** `scripts/experiments/rap_live_edit_acceptance.py`,
`rap_tempo_acceptance.py`, `rap_alignment_comparison.py`,
`rap_scheduling_evidence.py`; current session manifests and quickstart.

- [ ] Verify worktree linkage and preserve the dirty-worktree inventory.
- [ ] Record actual loaded models, reference hashes, decoding policy, renderer,
  service endpoints, GPU use, client devices, and software versions.
- [ ] Run one short unchanged 60-BPM Mac-to-H200 baseline, plus a live edit and
  tempo switch; record stages, fallbacks, and source/mix audio.
- [ ] Add tests for missing telemetry fields and clock-domain separation before
  extending the existing ledger/experiment helpers.
- [ ] Capture the exact launch command and a known-working recovery command.
- [ ] Estimate the full experiment cost from observed source generation, render,
  request, and session times. Choose the sample tier below before collecting
  results; reserve 25% timing contingency and the final 90 minutes for delivery.

**Gate:** distinguish working code from reachable model services and audible
playback. If baseline fails, spend at most 45 minutes diagnosing before switching
research comparisons to retained source audio and a labeled model-only harness.
Do not claim live success from offline output.

## Task 1: Strong Anchors Without Crushing Weak Syllables

**Budget:** 2 hours development, then the shared comparison block.
**Owner:** audio worker, shared with Task 3.

**Modify:** `src/streammuse/infrastructure/rap/phrase_warp.py` and policy wiring.
**Reuse:** `src/streammuse/experiments/rap_audio_protocols/warp.py`,
`src/streammuse/application/rap/bar_audio.py`,
`scripts/experiments/rap_alignment_comparison.py`.
**Tests:** existing `tests/unit/infrastructure/rap/test_phrase_warp.py`, plus a
focused `tests/unit/experiments/test_strong_anchor_policy.py` if extraction helps.

**Proposed policy:** `strong_word_anchors_r3`. Lexical primary stress (dictionary
stress 1) identifies preferred word anchors; template stress sets their musical
importance. Secondary stress is not automatically a primary anchor. Function
words with no lexical stress travel with the surrounding phrase. Repeated word
occurrences are distinguished by syllable index/word occurrence, not string alone.

The source-to-target map remains monotone and renders the whole phrase. Weak
syllables remain in source audio but do not each impose a hard time-map anchor.
Start with bounded local duration ratios of 0.75-1.33 as experimental defaults,
not perceptually validated limits. On a conflict, relax lower-priority anchors
before permitting an extreme ratio; retain the first planned rest and bar endpoint.
Report each requested/realized anchor and why it moved. If no bounded map fits,
return an explicit infeasible outcome. Existing live fallback may keep playback
alive, but must be labeled and cannot count as success for the new policy.

- [ ] Add failing tests for lexical/target-stress disagreement, repeated words,
  weak tails, two too-close strong anchors, leading rests, and rounding collisions.
- [ ] Implement the optional policy in the existing warper interface, exposing
  selected anchors, relaxed anchors, interval ratios, and infeasibility reasons.
- [ ] Run unit tests; confirm exact output length, monotone mapping, no negative
  positions, no syllable removal, and preservation of old policy behavior.
- [ ] Main listening comparison: 8 novel lines at 90 BPM x 3 policies (current
  relaxed, strict all-onset, new strong-anchor), with duration-appropriate source
  speech shared exactly across policies: 24 renders. Record infeasibility rather
  than selecting only lines where the new policy succeeds.
- [ ] Separate feasibility stress test: reuse 2 of those fixed sources at
  60/90/120 BPM across policies. This tests stretch limits, not an expectation
  that bounded warping works across a 2x tempo range. Cross-tempo listening
  comparisons, if added, use duration-matched sources per tempo and the same
  source across policies within each tempo.
- [ ] Ask Andrew to transcribe/rate 6 blinded short clips, then reveal text and
  timing overlays. Manually inspect a few acoustic onsets; map-derived times alone
  cannot validate pronunciation or physical timing accuracy.

**Show:** WAV comparisons, selected-versus-relaxed anchors, intelligibility notes,
render-time and timing-error distributions. All attempted renders remain in results;
the listening subset is not a statistical user study.

## Task 2: Spoken Direction And Own-Voice Feasibility

**Budget:** 60-90 minutes plus a maximum 30-minute voice-reference check.
**Owner:** input worker.

**New:** `src/streammuse/infrastructure/rap/utterance_input.py`,
`scripts/experiments/research_day/spoken_direction.py`, focused unit tests.
**Reuse:** `build_faster_whisper_transcriber` in
`src/streammuse/experiments/rap_audio_protocols/evaluation.py` for the spike;
do not make production code depend on this experiment module.

**Interface:** `TranscriptEvent(event_id, text, capture_start, capture_end,
ready_time, source, is_final)`; conversion to story input is handled by Task 4.
Start with one explicit push-to-talk utterance, maximum 15 seconds, displayed
transcript, and explicit submission. Use a persistent ASR worker so loading is
not charged as warm per-utterance time. Initially use faster-whisper CPU/int8
through its existing adapter; dependencies go in an optional environment.

[faster-whisper](https://github.com/SYSTRAN/faster-whisper) is the primary package
reference. [whisper.cpp](https://github.com/ggml-org/whisper.cpp) is a Mac-accelerated
alternative only if the first path is unusable; avoid building both today.

- [ ] Test silence, empty transcript, denied device, timeout, cancellation, clipping,
  sample-rate conversion, deduplicated submission, and capture stopping cleanly.
- [ ] Capture on the Mac with visible recording status. Prefer built-in microphone
  plus headphones to avoid Bluetooth microphone-mode changes and feedback.
- [ ] Run 6 prepared utterances; report transcript errors and capture-end-to-final
  latency. Route 3 accepted transcripts through the story adapter when ready.
- [ ] If Andrew supplies a clean reference clip, use it in an isolated MOSS request
  for 2 fixed lines; compare reference conditioning and latency without changing
  the shared production voice. Otherwise deliver the consented-reference protocol
  and a test with the already authorized reference, not a claim of cloning Andrew.

**Show:** one spoken plot edit, visible transcript, generated continuation, and
its latency breakdown. A captured WAV with actual ASR output is the fallback
artifact if microphone permissions block live capture. No claim of full duplex,
streaming ASR, or echo cancellation. Partial-transcript speculation stays a
recorded-trace follow-up unless all minimum deliveries are complete.

## Task 3: Locate Voice Variation Before Fixing It

**Budget:** 60-90 minutes, sharing source capture and worker with Task 1.

**Inspect/modify only if supported:** `application/rap/bar_audio.py`,
`infrastructure/rap/sglang_moss_tts.py`, `moss_generation.py`,
`experiments/rap_audio_protocols/timing.py`.
**New experiment:** `scripts/experiments/research_day/voice_consistency.py`.

The code supplies tempo-dependent duration/token targets and looks up continuation
audio by `(previous.lyric_id, tempo_bpm)`. A tempo change can lose the previous
prefix. This is a causal candidate, not proof of the reported sound change.

- [ ] Save one fixed source per line and warp it to 60/90/120 BPM: this isolates
  postprocessing from resynthesis.
- [ ] For 2 lines, generate at 3 tempos with continuation enabled/disabled and
  2 repeats: 24 source generations, holding reference and policy fixed. Record
  actual seed support, target duration/tokens, prefix use, and fallback provenance.
- [ ] Add a short 60 -> 90 -> 60 -> 120 BPM transition sequence with a controlled
  predecessor. Record prefix lookup at each transition; the factorial comparison
  alone does not exercise the live cache-miss path. Do not mix eSpeak fallback
  clips into a MOSS voice-consistency condition.
- [ ] Compare source-only and warped clips before drums are added; use randomized
  perceptual voice-consistency ratings. Pitch estimates are diagnostic, not a
  standalone measure of speaker identity.
- [ ] If prefix loss is implicated, test a voice-conditioning cache independent
  of target tempo while retaining tempo-specific rendered-audio cache keys.
  Never reuse mismatched final-duration audio to fix the conditioning cache.
- [ ] Add regression tests including `test_server_continuation_and_workspace_are_keyed_by_source_tempo`
  and cross-tempo conditioning isolation; verify no cross-session/character leak.

**Show:** source/warp matrix and evidence supporting or ruling out each cause.
When no cause is established, keep the diagnosis open and do not change defaults.

## Task 4: Persistent Directed Story

**Budget:** 90 minutes bounded story prototype; coordinator owns protocol/UI
integration. A new live replacement scheduler is a separate optional 60-minute
extension, not hidden inside this estimate.

**New:** `src/streammuse/domain/rap/story.py`, story tests and a scripted fixture.
**Modify:** `application/rap/lyric_session.py`, `lyric_writer.py`,
`infrastructure/rap/generators.py`, relevant request serialization and the existing
topic-control UI. Inspect both producer and consumer when extending the schema.

**Interface:** `StoryEvent(event_id, sequence, text, received_time, source)` and
`StoryContext(version, events, played_lines)`. Use an ordered event buffer capped
at 32 events and a verbatim ledger capped at 64 played bars, with explicit
rejection/stop at their limits; no silent history truncation or summary model today.
Typed and transcribed submissions use the same event path. Keep immutable plot
directions separate from recent generated lines and preserve them in rescue waves.
Only a confirmed playback-start notification promotes a generated line into the
played ledger, with its lyric ID/version. Discarded drafts never become established
story history. Preserve this ledger in rescues. Any future speculative predecessor
text is a separately labeled field with dependency IDs, not an established fact.
Playback-start is an operational proxy, not proof of acoustic arrival at the ear.

Story fixtures use additive future events, not retroactive changes to heard facts:
"a courier loses the map", "a stranger offers help", "the stranger wants the
package". State whether a direction was merely received, used for generation,
or audibly fulfilled. Do not mark events fulfilled from prompt inclusion alone.

- [ ] Test order/deduplication, rescue preservation, stale results after edits,
  bursts, and unchanged committed/current bars. Include a model-invented fact
  surviving in the played ledger after rescue/replacement; a discarded alternative
  must not survive there. Require duplicate playback callbacks to be idempotent.
- [ ] Implement the minimal prompt/context extension and a compact event queue
  using the existing ASAP edit path; do not replace ordinary topic mode.
- [ ] Produce a 12-bar progressive demo with 3 interventions. No pre-generated
  entire song substituted for the live run.
- [ ] Compare two live policies on matched traces: keep already queued bars and
  replace uncommitted bars using the existing ASAP path. Record the actual old/new
  publication behavior rather than assuming old audio vanishes immediately.
- [ ] Evaluate a third, deadline-fit replacement policy first in labeled trace
  replay. Estimate the remaining dependent path, not each bar independently.
  Publish a replacement only if its predecessor IDs match the output branch
  that will actually play. If a predecessor loses, invalidate/recompute descendants.
  Test this invariant before any optional live scheduler experiment. An isolated
  replay has no measured semantic or live-queue benefit; do not report one.
- [ ] Use 6 fixtures per live policy: early/middle/late edit arrival crossed with
  low/high queue pressure selected from Task 0. Hold the prompt/input trace and
  generation settings matched across policies; request counts can legitimately
  differ. Record compliance, first compliant playback, stale output, gaps, and
  wasted work. In a fixed recording horizon, unresolved edits remain failures or
  censored delays, never disappear from a successful-response average.

**Show:** one edit-to-audio timeline, an annotated story excerpt, 12 live pilot
runs, and a separately labeled third-policy replay. Extend to 18 live runs only
if the optional scheduler passes dependency/publication tests and time permits.
No population reliability claim from this small sample; retain failed edits.

## Task 5: Shared Model, Different Characters

**Budget:** 60-90 minutes. **Owner:** agent-interaction worker.

**New:** `scripts/experiments/research_day/characters.py`, its unit tests and
two fixed persona fixtures. **Reuse:** `LocalChatModelClient` and existing lyric
validation/rendering where appropriate; no new model runtime.

**Interface:** `CharacterState(actor_id, persona, private_history, shared_event_ids)`.
One client instance per active character points at the same H200 endpoint: the
existing client explicitly rejects overlapping requests on one instance. Sampling
`n=2` from one prompt does not give two independent character histories.

- [ ] Test isolation with private markers, explicit publication, out-of-order
  replies, cancellation isolation, and correct persona on retries.
- [ ] Generate an 8-12 turn SFW exchange between contrasting fictional characters.
  Fix speaking turns initially so content/history behavior is not confounded with
  turn selection. Publish only declared shared observations to the other history.
- [ ] For a scheduling-only comparison, freeze/replay identical per-character
  context snapshots and assigned speaking slots under serial and concurrent
  execution. Serial B must not receive A's just-completed reply when concurrent B
  did not. Test actual conversational reaction separately with live histories and
  explicitly report the information/readiness tradeoff.
- [ ] Keep text/timeline as the required deliverable. Add scheduled vocals through
  the existing renderer if ready; use the same fixed voice initially, with actor
  labels, to avoid conflating persona behavior and voice-stability experiments.

**Show:** separate histories, shared transcript, request timeline, responses that
reference the other character. One shared model does not imply independent model
weights or shared hidden memory. Do not claim context updates affect an in-flight
ordinary request; they affect its next request or a deliberate restart.

## Task 6: Concurrent Counting As A Coordination Probe

**Budget:** 60-90 minutes, reuse Task 5 worker/harness.
**New:** `scripts/experiments/research_day/concurrent_counting.py` and tests.
**Reuse:** task domain types where useful; existing alternating task runtime is
not itself concurrent and should not be relabeled.

**Rules:** 2 agents initially, count 1 through 10, no skipped or repeated accepted
number, both agents participate, and no consecutive accepted contribution by the
same agent. Agents see published events and may choose propose/wait. Enforce a
60-second run limit and record every proposal, including malformed ones.

Compare three explicitly different policies:

1. Deterministic alternating turns with real model calls: competent simple baseline.
2. Concurrent uncoordinated proposals: expose collisions/stale observations; log
   attempted publication without repairing it into apparent success.
3. Concurrent preparation with version-checked arbitration: one authoritative
   state owner accepts only a legal proposal for the current state version.

- [ ] Test simultaneous proposals, stale version, illegal number, missing actor,
  cancellation, timeout, fairness rule, and no eleventh accepted event.
- [ ] Run 10 trials per policy with matched initial prompts and logged seeds.
  Initial speaker/arrival jitter are recorded; seeds do not guarantee exact replay.
- [ ] Measure valid-prefix length, completion rate/time, duplicate/stale proposals,
  waiting, and model calls. Keep arbitration rejections visible alongside success.
- [ ] Optional: 3 agents or a second observation-delay level only after all six
  tracks have their minimum artifact. Audio is a replay of the logged timeline
  unless a live audio path is actually exercised.

**Show:** a compact attempt-versus-accepted timeline and a 15- or 30-trial table,
according to the preselected budget tier below.
This tests scheduling/coordination scaffolding, not arithmetic intelligence or
proof of human-like group collaboration. If deterministic turns win, report that.
Agents receive complete published events in this pilot. Partial-message listening,
simultaneous acoustic speech, and continuous shared-KV decoding remain explicit
follow-ups; this harness does not claim to implement those parts of the broader idea.

## Experiment Budget Tiers

Choose the tier using Task 0's runtime estimate, not after inspecting favorable
results. Publish the selected counts and all attempted runs. The full tier is
optional, not necessary to mark every track as addressed.

| Track | Minimum tier | Standard tier |
| --- | --- | --- |
| Alignment | 4 fixed-source lines x 3 policies at 90 BPM; 2-source feasibility sweep | 8 lines plus the same sweep |
| Voice | 1 line x 3 tempos x on/off x 2 repeats, plus transition probe | 2 lines: 24 sources plus transition probe |
| Spoken input | 3 utterances, one routed edit | 6 utterances, three routed edits |
| Story | 3 arrival phases x 2 policies at one load; separate replay | 6 traces x 2 policies, separate replay |
| Characters | 8-turn demo and one paired snapshot experiment | 12 turns and three paired experiments |
| Counting | 5 trials x 3 policies, 60-second cap | 10 trials x 3 policies |

Both tiers are exploratory. Counting's minimum worst-case budget is 15 minutes;
its standard budget is 30 minutes. If the estimated total exceeds 105 minutes
including contingency, start the minimum tier earlier as workers finish or shorten
the prototype scope explicitly. Do not silently consume the presentation reserve.

## Execution Order And Gates

| Elapsed window | Parallel work | Gate |
| --- | --- | --- |
| 0:00-0:45 | Baseline, manifests, dependency/device check | Known-good baseline or labeled failure |
| 0:45-2:45 | Audio policy/voice; microphone; story/event contract | Tests plus first tiny artifact per active track |
| 2:45-4:45 | Story demo; character/counting harness; audio comparisons | All six have a bounded prototype or explicit blocker |
| 4:45-6:30 | Timed model experiments run without cross-worker interference; blinded listening | Raw evidence, no result selection |
| 6:30-8:00 | Regression, recordings, summaries, presentation | Reproducible demo and qualified findings |

Use at most three implementation workers plus one integration owner. Assign audio
and voice to one worker because they share files; only the integration owner edits
session/protocol/UI files. Delegate an independent review before merging behavior.

If time shrinks, cut optional voice cloning, partial-ASR speculation, audio battle,
and 3-agent expansion first. Reduce repetitions with explicit counts next. Do not
cut all work on a track, fabricate measurements, or sacrifice the working demo for
an inference-engine migration. Reserve the final 90 minutes for verification and
presentation rather than adding features.

## Final Verification And Presentation

- [ ] Run focused unit suites for changed modules and existing browser/state
  reducers, then integration tests for live edits, tempo, controls, and playback.
- [ ] Exercise Mac client plus H200 services with the actual selected renderer;
  record fallback provenance rather than accepting silent eSpeak substitution.
- [ ] Exercise start, stop-at-bar, restart/reset, topic/flow/tempo edits, original
  exchange mode, leading rests, alignment switching, device selection, and mix.
- [ ] Capture a 2-3 minute progressive demonstration plus separate short clips for
  the research comparisons. Mark offline/replayed artifacts clearly.
- [ ] Produce a results index with commands, configurations, counts, raw paths,
  limitations, failures, and which conclusions are observation versus hypothesis.
- [ ] Prepare approximately 7 slides: question and prior work; current pipeline;
  six-track status; audio evidence; steering evidence; character/coordination
  evidence; contribution candidates and next discriminating experiment.

The decision at day's end is not "which idea looked most impressive?" It is:
which measured failure remains after a competent simple baseline, which variable
causes it, and what general scheduling decision could improve it?
