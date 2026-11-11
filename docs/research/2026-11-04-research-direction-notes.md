# Research Direction Notes: From Live Rap To Real-Time Interaction

Date: 2026-11-04
Status: idea capture and preliminary framing, not an approved implementation plan.

## Subsequent Clarification From Andrew

- Confirmed focus: general real-time LLM systems, with rap as the example.
- Research existing scheduling, live steering, and multi-agent work before
  selecting the main experiment.
- Keep every proposed idea. They are not equally important, equally concrete,
  or all intended to become research contributions.
- Andrew wants all areas addressed by tomorrow, with experiments and engineering
  progress helping reveal promising questions. Do not silently reduce this to
  implementing just one favorite idea.
- The exact demonstrable outcome for each area still needs definition; a working
  feature, a controlled comparison, and a feasibility investigation are different
  deliverables and must not be reported interchangeably.
- Initial literature findings: [research landscape](2026-11-04-realtime-llm-literature-map.md).

## Purpose Of This Session

Andrew wants to regain ownership of the research direction, not simply continue
adding demo features. The eventual session deliverable is a presentation for
tomorrow, November 5, grounded in whatever research, experiments, and coding we
actually complete today. This document preserves the ideas before prioritizing
them. No new experiments or implementation are claimed here.

## 1. Stress-Anchored, Word-Preserving Alignment

User intent:
- Align the strongly stressed syllable of a word to its intended rhythmic slot.
- Let weaker syllables travel with the word instead of independently forcing
  every syllable onto a grid.
- Avoid aggressively squeezing weak syllables or trailing speech when the next
  strong syllable begins too soon.
- Prefer a smarter, explicit rule-based policy that remains real-time capable.

Important distinction: lexical stress describes pronunciation within a word;
flow-template stress describes emphasis at a musical position. They are related
but not interchangeable. We need to specify which creates an anchor when they
disagree. Sentence emphasis and unstressed one-syllable words also need rules.

Illustration: anchor WHIS in WHIS-per while preserving the natural transition
into per. If the next anchor is too close, do not compress per into an artifact.
The unresolved decision is what gives way: the next anchor's exact timing,
the trailing syllable's duration, or the scheduling of the phrase.

Assistant proposal, not a settled design: use strong anchors plus bounded
duration changes and explicit collision handling. Exact timing, preservation of
all speech, and natural word duration may be mutually incompatible for a given
source recording and template. Expose that conflict instead of hiding it.

Potential measurements: strong-anchor timing error, weak-syllable distortion,
intelligibility, stretch-ratio extremes, collision frequency, and render time.

## 2. Microphone Input And Transcription

User intent:
- Let the system listen to spoken or rapped input, not only typed topics or taps.
- Explore interaction modes where the model repeats, responds to, or transforms
  what the user says.
- Investigate using the user's voice as a reference for synthesis.

Keep three independent functions separate:
1. Transcription: what words did the user say?
2. Interaction policy: what should the system do with those words?
3. Voice conditioning: whose vocal identity should the output resemble?

Voice conditioning is a proposed capability to investigate, not a newly verified
claim about our exact deployed MOSS configuration. Use the user's own consenting
voice for any such experiment.

Assistant proposal: first consider one bounded example, such as a user speaking
a sentence and the model turning its meaning into a beat-aligned response. This
would make response correctness and response time observable without requiring
voice imitation or full-duplex conversation at the same time.

Open decisions: turn-end detection, partial versus final transcripts, handling
transcript corrections, whether listening continues during playback, and keeping
the system's own loudspeaker output out of the user's transcript.

## 3. Unstable Vocal Identity

User observation: the rap sometimes sounds like different speakers, particularly
after changing tempo. A resemblance to a recognizable speaker is a perceptual
description, not evidence that a particular voice is being selected.

Possible causes to test, not conclusions:
- Stochastic TTS output or changes in voice/reference conditioning.
- Different continuation histories or phrase boundaries across generated bars.
- A switch between MOSS and an eSpeak fallback.
- Time warping that alters perceived timbre or prosody even if pitch is preserved.
- A tempo-dependent change in source synthesis, rather than only in alignment.

Useful isolation protocol: hold lyrics and voice reference fixed; listen to raw
TTS before alignment; then render that same source at several tempos. Separately
repeat TTS with the same input to estimate synthesis variability. Log the actual
renderer, fallback status, reference identity, synthesis settings, and warp map.
This separates source-voice variation from playback/alignment artifacts.

## 4. User-Directed Improvised Storytelling

User intent: replace isolated topic selection with incremental plot direction.
The user supplies events, constraints, or twists while the model continuously
raps a story, inventing details and connecting the gaps. The model continues
until the next instruction, like performers responding to an improvisation
director.

Unlike a topic switch, a plot instruction changes a continuing shared situation.
We need to distinguish established story facts, the newest direction, generated
but unheard material, and material the audience has already heard.

Research opportunity: determine how quickly a new direction can become audible
without contradictions or broken narrative continuity. Prepared output may need
to be retained, revised, or discarded; more preparation can protect delivery
deadlines while making the story slower to redirect.

Potential measurements: instruction-to-audible-effect latency, instruction
adherence, contradictions, continuity judgments, and discarded generation work.

## 5. Multiple Characters / Freestyle Rap Battle

User intent: present two or more distinct characters that respond to one another.
They can share one underlying model while maintaining separate chat histories,
personas, and possibly voices. Extend this into a freestyle battle or an ensemble.

Distinguish logical agents from physical resources. Two character histories do
not require two model copies or two GPUs. Conversely, sharing a model does not
mean their requests must always be serialized; the serving policy determines
how requests share compute.

Candidate parameters: character count, private versus shared history, persona
instructions, access to others' partial outputs, response length, turn policy,
voice assignment, and compute allocation. These are possibilities, not a mandate
to vary every parameter in the paper.

Research opportunity: measure how a shared inference resource supports several
time-sensitive, mutually dependent speakers without losing character identity,
relevance, or fairness of speaking opportunities.

## 6. Concurrent Real-Time Agent Collaboration

User intent: explore beyond strict request-response turns. Agents should be able
to hear one another as events unfold and potentially prepare or speak while
others are active. A rapid counting game is one possible simple testbed.

The referenced video of three ChatGPT instances counting is an anecdotal
motivation, not yet a verified baseline or research result.

Important distinction: concurrent listening and preparation do not require
unrestricted overlapping speech. Simultaneous speaking may be a permitted
behavior or a failure to coordinate, depending on the task.

Specify the game before claiming improvement: shared counting sequence, who may
speak next, what agents can observe, whether partial speech is visible, and what
counts as a collision, duplicate, skipped number, interruption, or success.

Essential baseline: an explicit deterministic turn scheduler can solve a
centrally assigned counting sequence. A learned or decentralized method must
demonstrate a benefit under stated constraints, rather than merely making this
simple task harder. Candidate constraints include delayed or partial perception,
interruptions, no central turn assignment, and shared compute limits.

Potential measurements: completion time, sequence accuracy, speech overlap,
missed observations, recovery time, and resource use. Choosing an informative
task is an open research question, not yet an agreed project expansion.

## Proposed Unifying Question

Assistant recommendation, subject to Andrew's approval:

> How should a real-time language-model system schedule and revise its output
> when new information arrives while previously planned output is still being
> generated, prepared, or delivered?

Rap supplies explicit rhythmic deadlines and an audible consequence when timing
fails. Story direction tests changing intent and stale plans. Multiple characters
add mutually dependent outputs and contention for compute. Microphone input adds
uncertainty about when information is complete and safe to act on.

This framing does not assert novelty. Literature review is needed to locate the
contribution relative to streaming dialogue, incremental generation, scheduling,
and multi-agent coordination. An entertaining demonstration is not itself proof
of a research contribution.

## Suggested Organization, Not Yet A Game Plan

- System validity: intelligible rhythm and a sufficiently stable voice so that
  audio failures do not obscure interaction results.
- Main research candidate: response to changing instructions during continuous
  output, with explicit deadline and revision policies.
- Extension candidates: microphone interaction, multiple characters, and
  concurrent agent collaboration. Keep their ideas without promising all of them
  for tomorrow.

A useful presentation tomorrow would distinguish: the central question, the
working testbed, one controlled result, what that result does and does not show,
and the next experiment. It should not present untested hypotheses as findings.

## Decisions Still Needed

1. What principal claim should tomorrow's presentation argue toward?
2. Is the primary contribution intended to be general real-time scheduling,
   interactive rap generation, or real-time multi-agent coordination?
3. Which single interaction creates the clearest measurable scheduling conflict?
4. Which audio defects must be resolved before participants can fairly judge it?
5. What baseline, metrics, and controlled comparison could challenge our claim?

All six idea areas remain captured. No implementation order, final paper claim,
or experiment outcome has been approved by this note.
