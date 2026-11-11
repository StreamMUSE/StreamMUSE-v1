# Interaction Task Details: Reference Appendix

The canonical grouped plan is `2026-11-05-interaction-next-steps.md`. This appendix
preserves detailed requirements and original TODO numbers for traceability.
These are not nine additional workstreams. Shared responsibilities, dependencies,
and execution order follow the consolidated plan, not the older delivery order
at the end of this appendix.

Recorded November 5, 2026. This supersedes the earlier interpretation of
"separate features" for the next development cycle. No implementation is
authorized in these notes: the user explicitly deferred further work for time.
The later request to add detail, including the fixed-cycle implementation task,
is recorded below as future work rather than a request to start development now.

## Decisions

- Keep the current directed-story feature. Its cumulative directions and prompt
  steering are acceptable for now. A verified story-state engine is not a new
  requirement for the next iteration.
- Microphone input must extend the existing rap workflow, not require a separate
  transcript demonstration or route only to Story.
- Replace the standalone dialogue demonstration with an audible, continuous rap
  battle between two user-chosen personas.
- Keep counting as a separate coordination task. Make recurring deadlines and
  logical correctness explicit, rather than using only an overall run timeout.
- Stress-alignment changes remain deferred. A subsequent user review explicitly
  reopened pitch/voice variability as an investigation task; see TODO 8.

## TODO 1: Spoken Topic Input In The Rap UI

- [ ] Add microphone recording beside the existing topic input.
- [ ] Reuse local transcription and the normal live topic-change API. Initially
  treat the recognized utterance as the topic; no extra interpretation LLM needed.
- [ ] Display recognized text and distinguish recognition, submission, and the
  change taking effect in playback. Preserve typed input and current-bar safety.
- [ ] Preserve opt-in Story routing without making it the only rap destination.
- [ ] Test with the user's physical microphone while rap plays, not only a
  prerecorded browser fixture. Check microphone permissions and playback leakage.

Acceptance: while rap is running, the user speaks a new topic and subsequent rap
uses it through the same scheduling path as a typed topic change. The user does
not need to stop playback or copy a transcript between pages.

## TODO 2: Two-Persona Continuous Rap Battle

- [ ] Let the user choose or enter two personas before starting.
- [ ] Give each persona its own instructions/history while allowing shared model
  weights. Both participate in one continuing battle, roasting each other.
- [ ] Connect candidate generation, validity checks/ranking, existing vocal
  rendering, drums, and scheduled playback. Text-only dialogue is not acceptance.
- [ ] Ensure responses address the opponent's recent performed lines and persona,
  rather than producing interchangeable generic lines. Repeated nonempty text is
  not sufficient evidence of a successful interaction.
- [ ] Clearly label the active performer and the line being heard. Distinct voice
  identities are not yet an agreed requirement and should not block integration.
- [ ] Define turn length, when the opponent's line becomes observable, generation
  lookahead, per-turn deadline, and late-result behavior before implementation.
- [ ] Log the opponent context used for each response, generated candidates,
  selection, audio readiness, and actual playback. Distinguish heard history from
  speculative future lines if early generation uses them.
- [ ] Demo multiple successive exchanges with Start/Stop and no full pregeneration.

Acceptance: two selected personas continuously perform audible rap responses to
each other on the existing musical timeline. Evaluate responsiveness and persona
distinction, not merely API completion or number of generated turns.

## TODO 3: Counting With Time And Logical Constraints

- [ ] Retain counting as an independent task, not a forced nine-syllable rap lyric.
- [ ] Introduce an explicit recurring output schedule and configurable per-output
  deadlines. A 60-second whole-run timeout alone is insufficient.
- [ ] Retain explicit logical rules: increasing count, no duplicates/skips, and
  the chosen participation/turn-taking rule. State who enforces each rule.
- [ ] Address both actors proposing the same number: compare explicit turn
  allocation and communication/coordination policies, not just hidden deduplication.
- [ ] Separate generated proposals, rejected proposals, accepted outputs, and
  performed outputs in the UI. Avoid logging the same accepted request twice as
  though it were two contributions. Keep request/actor identity visible.
- [ ] Show time remaining, result readiness, publication time, and deadline misses.
  Make the role of observation delay and concurrent work demonstrable.
- [ ] Use scheduled audible counting to expose timing, reusing playback machinery
  where practical; the exact beat/bar cadence needs a design decision.
- [ ] Compare serial and concurrent policies under stated budgets and load.
  Report wasted proposals and logical failures separately from deadline misses.

Acceptance: observers can see which actor proposed what, which result became
public, whether it was legal, and whether it arrived on time. Host validation
must not be presented as evidence that the agents independently coordinated.

## TODO 4: Choose Concrete Cooperation Use Cases With Deadlines

Select tasks in which useful contributions from multiple agents must combine
before an external deadline. Merely issuing two identical requests and rejecting
one is a baseline, not a compelling collaboration demonstration.

Candidate tasks below are proposed experiments, not implemented capabilities:

| Use case | Different contributions | Logical requirement | Deadline |
| --- | --- | --- | --- |
| Co-written rap bar | One agent proposes a setup; another supplies a responsive ending | Combined line fits the flow and preserves meaning | Entire bar and audio ready before its performance slot |
| Writer plus critic | Writer generates lines; critic checks/repairs a candidate | Selected output passes explicit validity checks | Critique must finish before selection/audio commitment, not after playback |
| Shared story performance | Characters contribute different facts/actions | No contradictions or repeated scene progress | Next scheduled scene/turn starts on time |
| Cooperative counting | Agents propose who contributes the next number | Correct sequence, participation, no duplicates/skips | One published/spoken contribution per scheduled interval |
| Live instruction revision | One agent prepares content; another incorporates a new user direction | Response reflects the new instruction while retaining required context | Replacement must reach the next editable performance slot |

- [ ] Choose one main cooperation task and retain counting as a simple diagnostic.
- [ ] Define what each agent knows, when it learns other agents' results, and
  whether it can revise or cancel work. Make the need for cooperation explicit.
- [ ] Start with a no-cooperation/serial baseline and compare meaningful policies
  at stated resource budgets. More model calls must not be mistaken for smarter
  cooperation.
- [ ] Vary the available time and observation delay; report logical correctness,
  deadline success, stale-context responses, and discarded work separately.

Example only: at 90 BPM a four-beat bar lasts 2.667 s. If an instruction arrives
with 0.6 s until the next boundary, preparation has less than 0.6 s after allowing
for playback commitment. Increasing the deadline is not equivalent to improving
the algorithm. End-to-end audio work must count against the deadline where used.

## TODO 5: Model-Generated Backup Lyrics And Audio

Goal: use short, freestyle-like fillers while substantive generation is still
in progress, rather than always falling back to unrelated fixed lyrics or silence.
Enable this only in appropriate modes/settings, with an explicit policy.

- [ ] Let the model prepare its own small reserve of short fillers, ad-libs, or
  backup lines, with rendered audio available before they are needed.
- [ ] Make reserve items compatible with the active persona, language, flow,
  tempo/duration, and voice as required. Invalidate or adapt them after changes.
- [ ] Prepare/refill the reserve during slack or initialization; do not make the
  last-resort path depend on a fresh slow model request at the missed deadline.
- [ ] Avoid unsupported plot claims or pretending that a filler answered the
  user's newest input. Mark it visibly as filler, not as a successful response.
- [ ] Define insertion granularity and switching boundaries so filler does not
  collide with the response. Track reserve depth, reuse, age, and audible duration.
- [ ] Compare filler, drums-only wait, and the existing fallback. Retain a bounded
  final safety path if even the generated reserve is unavailable.

Research distinction: filler can preserve continuity and reduce perceived
awkwardness, but does not reduce time to a meaningful answer. Report both time to
any audio and time to the actual response. Filler generation also consumes model
capacity; it should not starve the response it is intended to cover.

## TODO 6: Audit And Prototype Chinese Rap

Current answer: the existing end-to-end system should not be advertised as
supporting Chinese rap. The code has explicit English assumptions:

- `infrastructure/rap/utterance_input.py` allows only `base.en`/`small.en` and
  transcribes with `language="en"`.
- `infrastructure/rap/prosody.py` uses `pronouncing`/CMUdict for pronunciation and
  stress analysis. Passing Chinese text to a multilingual LLM or TTS alone does
  not replace this validation and alignment path.

- [ ] Scope the first experiment to Mandarin; assess other varieties and
  code-switching separately rather than treating all Chinese as one setting.
- [ ] Audit lyric generation, segmentation, pronunciation, syllable counting,
  rhyme evaluation, forced alignment, rendering, UI display, and optional ASR.
- [ ] Define suitable timing units and prominence constraints. Do not transplant
  English lexical-stress rules unchanged; consider Mandarin tones and prosody.
- [ ] Confirm the actual deployed TTS/aligner language support, then test native
  short phrases before full bars. Language support remains to be verified.
- [ ] Use a Chinese-capable ASR configuration if spoken topic input is included.
- [ ] Run a small matched Mandarin/English timing and listening comparison,
  including native-listener review, and document which components were replaced.

## TODO 7: Story Suggestions And Branching Options

Keep the current typed story-event path. Add suggestions as an optional aid,
not a replacement and not an automatic decision on the user's behalf.

- [ ] Generate a small set of distinct possible next events from the existing
  directions and recent performed lyrics. Three options is an initial UI proposal.
- [ ] Present options as clickable narrative branches alongside the typed input.
  Clicking one must submit through the existing story-event API and scheduling
  path, with normal accepted/playing version feedback.
- [ ] Generate suggestions ahead of need; do not block ongoing rap or typed input
  while waiting for suggestions.
- [ ] Bind each suggestion set to its story session/version. Refresh or reject
  outdated options after Reset or intervening directions; do not silently apply
  a branch from an unrelated story state.
- [ ] Allow refresh/dismissal and preserve free typing. Unchosen options must not
  enter the performed narrative or be treated as established facts.

Initial scope: alternative next-step options for a single ongoing story, not a
full editable narrative tree with rollback and replay. A persistent branch graph
would be a separate extension.

## TODO 8: Diagnose Pitch And Voice Variability

The observed variability is not yet attributed to a confirmed cause. Investigate
speaker identity, pitch, pronunciation/prosody, and stretch artifacts separately.

- [ ] Capture the raw per-bar TTS waveform and final aligned waveform, with the
  selected renderer/fallback, reference voice hash, reference text, generation
  settings/seed where available, sample rates, tempo, and stretch ratios.
- [ ] Repeat one fixed line with fixed conditioning. Check whether the perceived
  voice changes before any alignment. This isolates synthesis variability.
- [ ] Reuse one raw waveform across several tempos/policies. Differences then
  isolate time-warping/playback processing rather than a newly sampled voice.
- [ ] Check reference-conditioning consistency across bars and source changes
  between MOSS and robotic fallback. Check sample-rate interpretation and any
  remaining rate-changing resampling paths instead of assuming all warps preserve
  perceived pitch and timbre.
- [ ] Use controlled listening comparisons plus voiced-region pitch summaries
  and, if appropriate, speaker-similarity measurements. Pitch alone does not
  establish speaker identity or prove a sample-rate bug.
- [ ] Choose the smallest fix supported by the results; do not present cloning,
  new voices, or a renderer replacement as necessary before diagnosis.

Working hypotheses, not findings: independent per-bar TTS variation, inconsistent
voice conditioning, renderer/fallback switching, or timing/postprocessing effects.

## TODO 9: Implement Fixed USER -> WAIT -> MODEL Cycles

This is distinct from continuous model generation with live topic/story steering.

```text
| USER: U bars | WAIT: W bars | MODEL: M bars | USER: U bars | ...
  capture input   prepare /     perform the     begin a fresh
                  replay       response        capture window
```

- [ ] Make U, W, and M explicit session parameters. For fixed four-beat bars at
  tempo B, segment durations are U*240/B, W*240/B, and M*240/B seconds.
- [ ] In USER, play drums and collect the designated user input. The existing
  keyboard-flow capture is the natural first input; microphone/topic integration
  can be connected separately. The exact input modality needs confirmation.
- [ ] Close capture deterministically at the USER boundary and submit the frozen
  input. Keep input belonging to one cycle out of the next cycle's buffer.
- [ ] In WAIT, continue drums and optionally replay the captured rhythm so the
  user hears what was entered. Model preparation runs during this interval.
- [ ] In MODEL, play the scheduled response on time, then return to USER without
  requiring another Start or Reset. Label the current phase and show a countdown.
- [ ] Define the response-ready deadline before MODEL commitment, including
  generation, checks, synthesis, transfer, and client preparation.
- [ ] Decide the late policy explicitly. A fixed cycle must not silently become
  an unbounded WAIT: use a declared filler/replay/skip behavior or report a miss.
  If early, hold the ready response until MODEL begins.
- [ ] Specify whether the full M-bar response must be ready at the first MODEL
  boundary or whether later bars can be produced progressively with their own
  deadlines. This changes the required preparation budget.
- [ ] Test several automatic cycles, empty/partial user input, early/late results,
  Stop/resume/reset, and preserve the existing continuous steering mode.

These segment mechanics and parameter names are a proposed concrete design;
defaults, capture modality, and missed-deadline behavior are not yet approved.

## Shared Research Framing

The common problem is agents producing useful outputs under **both time and
logical constraints**, not merely producing different personalities or more text.

| Aspect | Rap | Counting |
| --- | --- | --- |
| Work toward an output | Generate lyric candidates | Generate next-action proposals |
| Logical constraints | Candidate validity, e.g. syllable/flow requirements | Correct next number and participation rules |
| Selection/publication | Choose a candidate for performance | Decide which legal proposal becomes public |
| Time constraint | Audio ready before musical commitment/playback | Output ready before its scheduled deadline |
| Changing observations | User steering or opponent's performed line | Other actors' newly published contributions |

For a work item released at time r with deadline d, the available budget is d-r.
Record generation, validation, audio (when relevant), transfer, and scheduling
time separately. A result can be legal but late, timely but illegal, or based on
outdated observations. These must remain separate outcomes.

Open design choices: turn/output granularity, observation-sharing policy,
centralized versus agent-level coordination, amount of lookahead, and what to
perform when no valid result is ready. Do not silently fix these choices or claim
that the current host-refereed harness solves the broader research problem.

## Delivery Order

Original recommendation was spoken topics, audible battle, then timed counting.
The expanded backlog now also includes fixed cycles, fillers, story branches,
Chinese feasibility, and voice diagnosis. Reconfirm priorities and time budget
before the next implementation session; do not assume all fit one session.
Choose the cooperation use case before investing in another standalone harness.
Shared timing instrumentation can support the battle, counting, and fixed cycles.
Keep the working rap demo and Story mode available throughout. Update the
presentation only after revised behaviors have been tested, distinguishing new
evidence from today's diagnostic pilots.
