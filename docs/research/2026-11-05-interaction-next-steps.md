# Consolidated Interaction Roadmap

Updated November 5, 2026 after user review. Planning only: no further feature
implementation has started. The previous nine TODOs are grouped into five
workstreams, with shared work owned once. Detailed requirements remain in
[the reference appendix](2026-11-05-interaction-task-details.md).

The active next milestone is specified in
[Inference page requirements](2026-11-10-inference-page-milestone.md). It selects
shared timing/diagnostics, cyclic exchange, and continuous steering from this
backlog. Its newer requirements override older fixed-WAIT/bar-sized assumptions:
WAIT has no configured duration, starts MODEL on client playback readiness, and
response/commit units are not necessarily bars. Other workstreams remain backlog.

## Scope And Decisions

- Preserve the working rap pipeline and existing continuous steering mode.
- Keep directed story as-is; a verified story-state engine is not required now.
- Microphone is an input method for the existing rap system, not a separate demo.
- Two user-chosen personas should continuously roast each other in audible rap.
- Counting remains a separate coordination task with time and logical constraints.
- Stress-alignment upgrades remain deferred. Voice variability is reopened for
  diagnosis. Chinese support is a feasibility task, not a current capability.
- These groups organize ownership and reuse, not a requirement to combine every
  mode into one experience or implement everything in one session.

## 1. Shared Timing, Playback, And Late-Response Policy

Combines timing requirements from fixed cycles, rap battle, and counting with
generated fillers. Extend existing scheduling/playback modules rather than
building a generic agent framework in advance.

- [ ] Define work release, input/context version, output slot, preparation
  deadline, audio commitment, and actual performance times consistently.
- [ ] Express segment lengths and output cadence in bars/beats as appropriate;
  keep those choices explicit rather than assuming every task uses one bar.
- [ ] Reuse rendering, transfer, and scheduled output where appropriate. Count
  all these stages against end-to-end deadlines, not only LLM generation.
- [ ] Define late policies: generated filler, captured-rhythm replay, drums-only
  wait, or skip/miss. Specify which policies are legal in each mode.
- [ ] Prepare a bounded reserve of model-generated fillers and their audio before
  need. Refill during slack without starving substantive generation.
- [ ] Track compatibility with persona, language, flow, duration/tempo, and voice.
  Invalidate stale reserve items. Filler is not a fulfilled request or valid count.
- [ ] Provide shared diagnostics: proposed -> validated -> selected/committed ->
  performed, with rejection reasons, deadlines, stale context, and wasted work.
- [ ] Distinguish time to any audio from time to a meaningful response. Test
  early, late, empty, stale, canceled, and Reset-crossing results.

Acceptance: a result may be valid but late, timely but invalid, or based on stale
input, and the UI distinguishes each. Continuity alone is not responsiveness.

## 2. Live Steering And Story Input

Combines spoken topics, existing story directions, and suggested narrative
branches into alternative input paths using existing APIs and the same rap system.

```text
typed topic / spoken topic ----------> topic-change API ---+
typed event / spoken event / option -> story-event API ----+-> existing rap
```

- [ ] Put microphone capture beside the normal topic control. Transcribe locally
  and send the recognized utterance through the normal live topic-change path.
- [ ] Retain typed input, transcript display, optional Story destination, session
  fencing, and feedback distinguishing recognized, accepted, and playing states.
- [ ] Suggest distinct next events using retained directions and recent performed
  lyrics. Three options is a proposed initial count, not a fixed requirement.
- [ ] Clicking an option uses the same story-event submission/scheduling path as
  typing. Preserve free typing, refresh, and dismissal.
- [ ] Generate suggestions asynchronously; bind them to a session/story version
  and invalidate stale sets. Unchosen branches must not become narrative facts.
- [ ] Test a physical microphone during playback, including permissions and
  playback leakage, not only prerecorded browser input.

Acceptance: spoken topics change future rap without stopping or copying text.
Options reduce authoring burden without changing current Story semantics. A full
persistent branch tree is out of scope.

## 3. Timed Turn-Based Performance

Two distinct modes reuse workstream 1's clock, readiness rules, and playback.

### User / Model Exchange

```text
USER (musical duration) -> WAIT (until playable) -> MODEL (musical duration) -> repeat
```

- [ ] Parameterize USER/MODEL lengths; show their remaining musical duration
  and WAIT elapsed wall time. There is no configured W.
- [ ] Capture only the current USER window, freeze/submit at its boundary, and
  isolate successive cycles. Keyboard flow is the proposed first input; the exact
  modality needs confirmation. Reuse workstream 2 if spoken input is added.
- [ ] Pause music during WAIT and begin MODEL on a fresh downbeat as soon as a
  current response prefix is playable on the client, without beat quantization.
  Initial waiting is latency, not a missed musical deadline. Late policies apply
  after MODEL playback has established its timeline.
- [ ] Decide whether all M bars must be ready initially or whether each has a
  progressive deadline. Test repeated cycles and Stop/resume/Reset.

### Two-Persona Rap Battle

- [ ] Let the user choose two personas with separate instructions/histories and
  shared model weights. Alternate audible roasts continuously on the musical clock.
- [ ] Generate/rank/render through the rap pipeline. Require meaningful responses
  to the opponent, not merely nonempty or interchangeable repeated text.
- [ ] Define turn length and when opponent output becomes observable. Separate
  performed history from speculative future lyrics in prompts and logs.
- [ ] Label the active persona and performed lyrics. Distinct voices are not yet
  required; consistent usable audio is sufficient for initial integration.
- [ ] Test progressive exchanges, persona distinction, relevance, repetition,
  readiness, and late handling, not just request completion.

Acceptance: both modes are live audible experiences. They remain separate user
experiences even though they reuse scheduling and audio infrastructure.

## 4. Cooperation Use Cases And Controlled Experiments

Combines use-case selection with counting. Rap battle is multi-actor interaction,
not automatically a cooperation task.

| Candidate | Complementary work / logical constraint | External deadline |
| --- | --- | --- |
| Co-written bar | Setup + responsive ending; combined flow/meaning constraints | Complete audio before assigned bar |
| Writer + critic | Generate, check, repair; selected line must be valid | Critique before selection/audio commitment |
| Shared story | Different contributions without contradiction/repeated progress | Next scheduled scene/turn |
| Instruction revision | Update prepared content to reflect new user intent | Next editable output slot |
| Counting diagnostic | Agree next contributor/number; no duplicates/skips | One legal contribution per scheduled interval |

- [ ] Choose one main cooperation use case before another standalone harness.
  Retain counting as a simple diagnostic, not the sole evidence of collaboration.
- [ ] Specify roles, observations, communication timing, revision/cancellation,
  and which decisions the host versus agents enforce.
- [ ] Give counting recurring deadlines and scheduled audible output using
  workstream 1; do not force it into nine-syllable lyrics. Cadence remains open.
- [ ] Compare explicit turn allocation and communication/coordination policies;
  generating duplicate numbers and filtering one is only a baseline.
- [ ] Separate proposals, rejections, accepted contributions, and spoken outputs.
  Do not display one request twice as if two actors contributed it.
- [ ] Compare serial/concurrent policies at stated resource budgets/load; vary
  observation delay and available time. Report logical errors, deadline misses,
  stale work, and discarded compute separately.

Acceptance: observers can explain what cooperation accomplishes, what changes
under a tighter deadline, and which correctness comes from the host. Host-filtered
success is not evidence of independently learned agent coordination.

## 5. Voice Reliability And Language Feasibility

Related through speech/phonetic processing and controlled listening tests, but
independent subtracks. Chinese feasibility need not wait for every voice fix;
a new language must not silently inherit English validation assumptions.

### Voice / Pitch Diagnosis

- [ ] Retain raw TTS/aligned audio with renderer/fallback identity, reference
  hashes, generation settings, sample rates, tempo, and stretch ratios.
- [ ] Repeat a fixed line with fixed conditioning to isolate synthesis variation;
  reuse one raw waveform across tempos/policies to isolate processing effects.
- [ ] Check conditioning, source switching, sample-rate handling, and time warp.
  These are hypotheses, not established causes.
- [ ] Combine listening with voiced-region pitch and appropriate speaker/timbre
  diagnostics. Implement the smallest evidence-supported fix.

### Mandarin Rap Feasibility

- [ ] Audit generation, segmentation, pronunciation, counting, rhyme, alignment,
  TTS, UI, and optional ASR. Confirm deployed model/aligner support experimentally.
- [ ] Replace relevant English assumptions: ASR currently permits only English
  models/language; pronunciation/stress analysis uses English CMUdict.
- [ ] Define language-appropriate syllable/prominence/rhyme rules rather than
  translating English lexical stress. Test short phrases before full bars.
- [ ] Use Chinese-capable ASR for spoken input, reusing workstream 2's UI/routing.
  Include native listening review and matched timing comparisons.

Acceptance: explicitly document supported paths and limitations. Multilingual
LLM/TTS alone is not end-to-end Chinese rap support. Other varieties and
code-switching are separate extensions.

## Dependencies And Delivery

```text
Shared timing/playback (1) ----> exchange + battle (3)
                         +----> cooperation/counting (4)
Live input (2) ----------------> optional spoken input for (3)
Audio/language findings (5) ---> rendering, fillers, and input adaptations
```

Build thin changes to the working system, not all infrastructure upfront. Spoken
topics are an independent small integration. Develop shared timing while delivering
one selected audible mode, then add fillers as its late policy. Choose the
cooperation use case before the research comparison. Voice diagnosis and language
feasibility can proceed in parallel as bounded experiments.

Reconfirm priority and time budget before implementation. Preserve continuous
steering and Story mode; do not assume all workstreams fit one session. Update
demo/presentation claims only after end-to-end testing.

## Mapping From Original TODOs

| Original TODO | Canonical owner |
| --- | --- |
| 1 Spoken topics + 7 Story branches | 2 Live steering and story input |
| 2 Rap battle + 9 Fixed cycles | 3 Turn-based performance; shared timing in 1 |
| 3 Counting + 4 Cooperation use cases | 4 Cooperation experiments; timing/audio in 1 |
| 5 Generated backup lyrics/audio | 1 Shared late-response policy |
| 6 Chinese + 8 Voice variability | 5 Language feasibility and voice reliability |

The appendix preserves earlier detailed requirements, not additional duplicate
tasks. Grouping and dependencies in this document take precedence.
