# Real-Time LLM Systems: Initial Literature Map

Date: November 4, 2026. Target presentation: November 5.
Status: source-checked first pass, not an exhaustive survey or novelty claim.
No new benchmark results or implementations are reported here.

Follow-up: [closest-paper review](2026-11-04-realtime-llm-close-reading.md) and
[six-track day plan](../superpowers/plans/2026-11-04-six-track-research-day.md).
These add method/assumption checks and incremental-dialogue precedents missing
from this initial map.

## Agreed Direction And Scope

Study general real-time LLM systems; use rap as an observable, clock-driven
example. Preserve all proposed feature and exploration areas. Use their results
to identify useful questions rather than prematurely declare a single feature
the paper's contribution. All areas are intended to receive attention by
tomorrow, but the concrete completion criterion for each remains to be agreed.

## Terms

- Deadline: latest time an output must be ready for its intended use.
- SLO (service-level objective): a performance target, not necessarily a hard
  real-time guarantee; for example, a response-latency target.
- Goodput: useful service delivered while meeting the specified targets. Papers
  use different units, so their goodput numbers are not automatically comparable.
- Live steering: a user changes instructions while the system is working.
- Commit boundary: the point after which this implementation will no longer
  replace a prepared output. This is distinct from physical irreversibility once
  the sound has actually been heard.
- Asynchronous interaction: participants need not wait for fixed complete turns.
- Full duplex: listening and speaking can operate concurrently.
- Freshness: how current the information reflected by an output is. A new
  timestamp alone does not establish that a response follows the new instruction.

## A. Serving And Scheduling

### Clockwork (OSDI 2020)

[Serving DNNs like Clockwork](https://www.usenix.org/conference/osdi20/presentation/gujarati)
builds predictable inference serving around predictable DNN execution and tight
request-level latency targets. Read: official proceedings abstract.

Relevance: deadline-oriented inference is not new. Borrow its emphasis on
predictability and measured tails; do not assume variable-length autoregressive
generation has the same runtime properties as the DNN workloads studied there.

### Sarathi-Serve (OSDI 2024)

[Taming Throughput-Latency Tradeoff](https://www.usenix.org/conference/osdi24/presentation/agrawal)
uses chunked prompt processing and schedules it alongside ongoing token decoding
to reduce stalls while retaining batching efficiency. Read: proceedings abstract.

Relevance: supplies an execution-layer explanation for contention between new
requests and existing streams. It does not by itself define whether a generated
rap bar remains semantically useful after a user edit.

### JITServe (NSDI 2026)

[SLO-aware LLM Serving with Imprecise Request Information](https://www.usenix.org/conference/nsdi26/presentation/zhang-wei)
handles per-token, whole-response, and compound request requirements. It revises
runtime estimates during generation and allocates enough service to meet targets
while preserving capacity for other work. Read: proceedings abstract and metadata.

Relevance: a close scheduling baseline, not merely generic infrastructure.
We cannot claim that deadline-aware scheduling or uncertain request duration is
our novelty. Full algorithm and artifact inspection are needed before deciding
whether to integrate it or build a carefully labeled application-level analogue.

### Agentix (NSDI 2026; earlier preprint titled Autellix, 2025)

[Agentix](https://www.usenix.org/conference/nsdi26/presentation/luo) schedules LLM
calls using information about the programs they belong to, rather than treating
all calls as unrelated requests. Its policies use accumulated program service
to prioritize and preempt work. Read: proceedings abstract; earlier
[Autellix preprint](https://arxiv.org/abs/2502.13965) metadata/abstract.

Relevance: multiple characters sharing inference resources need program-level
accounting, not just independent chat histories. A shared-model character demo
alone is not a new serving contribution. Use the final venue title in slides.

## B. Live Steering And Revision

### Revisable by Design (2026 arXiv preprint)

[A Theory of Streaming LLM Agent Execution](https://arxiv.org/html/2604.23283v1)
explicitly studies mid-execution user changes, classifies action reversibility,
and proposes earliest-conflict rollback. Its evaluation includes simulated tools
with real LLM agents, as well as larger mock-model ablations, measuring quality,
wasted actions, compensation, steps, and tokens. Read: abstract, framework
outline, experiment design, and metrics sections.

This is especially close prior work. Neither live revision nor irreversibility
is a novel concept for us. A possible distinction to investigate is recurring
wall-clock delivery obligations while repair is occurring, rather than only the
eventual quality and work cost of a repaired tool plan. Its broader claims about
other frameworks are the authors' claims, not independently verified facts here.
No peer-reviewed venue was verified in this pass.

### Answering While You Type (October 2026 arXiv preprint)

[Optimizing Time-to-Answer with Streaming Generation](https://arxiv.org/abs/2610.26296)
starts computing from an incomplete request and revises as input arrives. It
examines autoregressive and diffusion models. Read: abstract and metadata only;
no peer-reviewed venue verified here.

Relevance: early generation from partial microphone transcripts is an existing
research direction. Our experiments should account for incorrect speculation
and discarded work, not report only faster response after final input.

### Age Of Information (IEEE JSAC 2021; preprint 2020)

[Age of Information: An Introduction and Survey](https://arxiv.org/abs/2007.08564)
formalizes the timeliness of timestamped updates under limited resources.
[MIT's publication record](https://dspace.mit.edu/entities/publication/ca039de7-e561-4a0f-bd3a-b92d0c436020)
verifies journal and year. Read: abstract and publication metadata.

Relevance: freshness and update replacement have a substantial prior foundation.
Timestamp age could be a useful trace measure, but it is not equivalent to
semantic compliance: a freshly generated line can still ignore a new plot event.
Follow the survey to original results before borrowing a theorem or claiming a
new freshness metric.

## C. Concurrent Interaction And Multiple Agents

### Moshi (2024 technical paper / arXiv)

[A speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
models user and system speech as parallel streams, supporting full-duplex spoken
interaction rather than requiring explicit complete turns. Read: abstract and
metadata; no later conference venue verified here.

Relevance: simultaneous listening and speaking is already demonstrated. This is
Moshi, not our deployed MOSS-TTS. A microphone-plus-transcription pipeline is not
automatically equivalent to a full-duplex dialogue model, and we should avoid
equating model frame latency with end-to-end instruction-to-audible-response time.

### Time To Talk (Findings of EMNLP 2025)

[LLM Agents for Asynchronous Group Communication in Mafia Games](https://aclanthology.org/2025.findings-emnlp.608/)
separates an LLM's decision about when to speak from its generation of what to
say. It adapts participation and evaluates an asynchronous group-chat setting;
the implementation includes simulated typing time. Read: official publication
record and paper's introduction/method sections.

Relevance: a concrete reference for timing and character behavior. Message-level
participation in a group chat is not the same experimental target as overlapping
audio streams, audible deadlines, or syllable timing. Those differences need
measurement, not merely a different demo interface.

### HANDRAISER (CoLM 2026)

[Learning to Interrupt in Language-based Multi-agent Communication](https://arxiv.org/abs/2604.06452)
lets listeners interrupt generation and learns when enough information has been
received using predicted reward and communication cost. The paper reports that
uninformed interruption can happen too early. Acceptance is stated in the
authors' arXiv metadata. Read: abstract and metadata.

Relevance: "agents can interrupt each other" is already studied. We could test
how interruption decisions change with delivery deadlines, but must compare
quality against saved time rather than assuming interruption is beneficial.

### StreamMA (2026 arXiv; version 3 dated October 27)

[Streaming Communication in Multi-Agent Reasoning](https://arxiv.org/abs/2606.05158)
streams reasoning steps to downstream agents before complete messages finish,
pipelining work across agent topologies. Read: current abstract and metadata;
no peer-reviewed venue verified in this pass.

Relevance: starting downstream work on partial agent output is existing work.
Its abstract concerns reasoning-task latency and effectiveness; the handling of
continuous external steering, sound playback, and shared-server deadlines needs
full-text examination before we assert a difference.

## D. Our Own Previous Contribution

[Real-Time Language Model Jamming](https://arxiv.org/abs/2606.11886), June 2026,
already defines frame-synchronous streaming inference, uses a client-server
architecture tied to an external clock, and models relationships between
configuration and round-trip latency. Read: abstract and paper structure.

Therefore neither "LLMs need timing, not just speed" nor "music supplies an
external clock" is sufficient as the new claim. We must explain what additional
decision, guarantee, objective, or empirical finding goes beyond that work.

## What Might Be Distinctive: Hypotheses, Not Established Gaps

Candidate umbrella: scheduling useful output under changing intent and ongoing
delivery obligations. All three ingredients have prior work individually.
Combining them is not automatically a contribution either.

Potentially consequential constraints in our testbed:
- New instructions can invalidate work that is already generated or synthesized.
- Speech has downstream preparation and network costs, not just LLM decode time.
- The application keeps delivering output while new plans are being computed.
- The software commit point and the point where a listener hears audio differ.
- Multiple characters can share compute while depending on one another's output.

We need to show that one of these changes the useful policy, creates a measurable
failure of a strong baseline, or supports a general model that predicts behavior.
We have not yet demonstrated that.

## Candidate Controlled Comparisons

These are experiment sketches, not a selected main experiment or results.

### 1. Revision Versus Continuity

Compare keeping all queued output, replacing every uncommitted output, and
replacing only incompatible uncommitted output. Keep model, hardware, scripted
edits, and flow schedule matched. Include an appropriate deadline-aware baseline;
do not compare only against an intentionally weak FIFO implementation.

Sweep edit arrival phase/rate and available preparation time. Measure deadline
misses, stale-but-played content, first audibly compliant response, discarded
compute, and narrative continuity. A new-context version number is evidence of
provenance, not proof of semantic compliance. Score compliance separately.

Disconfirming result: a simple replacement rule matches the proposed policy
across relevant loads. That would narrow the contribution rather than be hidden.

### 2. Multiple Characters On Shared Compute

Use separate character histories with a shared model server. Compare fixed turn
assignment, concurrent preparation with explicit speaking slots, and a simple
deadline-aware allocation. Distinguish simulated timing from measured wall-clock
execution and distinguish application scheduling from the server's own batching.

Sweep participant count, message length, and reaction deadlines. Measure useful
on-time responses, content relevance, overlap, fairness, and resource use. Start
with two characters; a counting task needs an explicit coordinator baseline and
clear rules before its failures can support a research claim.

### 3. Acting On Partial User Speech

Compare waiting for final transcription against speculative preparation from
partial transcripts. Use the same recorded speech and transcript updates, then
validate separately with live capture if implemented.

Measure first correct response time, corrections, discarded work, and wrong
commitments. A lower time-to-first-token is insufficient if the spoken response
is incorrect or the user cannot redirect it.

## Preserve The Full Work Program

| Area | Candidate Deliverable By Tomorrow | Role For Now |
|---|---|---|
| Smarter stress alignment | Rule specification and matched audio comparison; implement the bounded policy if feasible | Engineering plus perceptual evidence |
| Voice variation | Fixed-source tempo comparison versus repeated raw TTS | Diagnosis; potentially informs timing-quality tradeoffs |
| Microphone/transcription | Minimal input path and latency trace; separate voice-conditioning feasibility | Interaction capability and uncertainty measurement |
| Story direction | Scripted plot edits and a trace showing when they become audible | Candidate revision workload |
| Multiple characters/battle | Small shared-model, separate-history prototype and timing trace | Candidate resource-sharing workload |
| Concurrent collaboration | Explicit task rules, baseline, and bounded feasibility experiment | Exploratory research |
| Presentation | Related-work map, actual results, negative findings, and next decisions | Research communication |

This table proposes levels of delivery; it does not override Andrew's request
that all areas receive attention, nor promise that every area will be a polished
feature. Agree the minimum observable outcome per item before implementation.

For each run record: question, source/baseline, configuration, input trace,
measurement definition, artifact path, result, limitations, and next decision.
Separate planned, implemented, measured, and interpreted status. Never fill a
results slide with a hypothesis or a feature screenshot presented as evidence.

## Next Research Steps

1. Read JITServe, Revisable by Design, StreamMA, and our prior StreamMUSE paper
   in full for the exact closest overlap; inspect public artifacts and assumptions.
2. Check adjacent incremental dialogue, speculative speech/translation, online
   plan repair, and real-time/anytime scheduling literature. This first pass does
   not establish coverage of those fields.
3. Convert the work-program rows into explicit, bounded tasks and acceptance
   evidence; only then choose experiment order and allocate GPU time.
4. Let measured results update the contribution hypothesis. Negative or simple
   results are useful, but a collection of results still needs a coherent question.
