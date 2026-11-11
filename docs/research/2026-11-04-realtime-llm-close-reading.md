# Real-Time LLM Research: Closest Work And Experimental Implications

Date: November 4, 2026. Presentation target: November 5, 2026.
Status: literature review and proposed experiments, not new experimental results.

## Main Finding

There is no defensible novelty claim in merely making an LLM asynchronous,
interruptible, deadline-aware, or capable of multiple characters. Our previous
paper also already studies synchronization with an external clock. A plausible
next investigation is **how to revise useful output under continuing delivery
deadlines and limited shared computation**. Whether that supports a contribution
depends on experiments against competent baselines, not the rap application alone.

Keep all six ideas. Treat them as a connected experimental program, with different
levels of delivery, rather than six claims that must all become a paper.

## Reading Scope

Read the relevant method, assumptions, and evaluation sections of the papers below,
not only abstracts. This is a targeted close reading, not a replication, exhaustive
survey, or verification of every proof. Versions matter: dates and venues below
refer to the versions inspected. Repository inspection establishes availability,
not that an artifact installs or reproduces its results.

## Closest Papers

### 1. JITServe

**Venue/year:** NSDI 2026; earlier arXiv version in 2025.
[Paper](https://www.usenix.org/system/files/nsdi26-zhang-wei.pdf),
[artifact](https://github.com/UIUC-MLSys/JITServe).

- **Method:** estimate conservative output-length bounds with quantile regression,
  refine during decoding, infer compound-request dependencies, and allocate token
  service using grouped margin-goodput scheduling. It considers both priorities
  and batch composition, not just earliest deadline first.
- **Evidence:** Sections 4-6 evaluate mixed request requirements, load changes,
  models, and ablations. Core comparisons include vLLM, Sarathi-Serve, Autellix,
  and a length-ranking policy.
- **Boundary:** its central objective concerns requests/tokens satisfying serving
  targets. That does not automatically establish whether audible content still
  obeys a newly injected instruction.
- **Use here:** record stage costs and available time, and measure useful on-time
  output, not only average response time. A simple application-level deadline
  policy must not be called a JITServe reproduction.
- **Deployment:** the artifact requires its modified, vendored vLLM and prediction
  assets. Do not replace the working demo environment for today's pilot.

### 2. Revisable By Design

**Venue/year:** arXiv preprint, April 2026; no peer-reviewed venue verified.
[Paper](https://arxiv.org/html/2604.23283v1),
[artifact](https://github.com/zhiyuanZhai20/stream-agent).

- **Method:** classify actions by reversibility, find the earliest incompatible
  action, preserve the valid prefix, compensate where possible, and replan.
- **Assumptions:** the rollback result depends on compatibility separability and
  monotone compensation/waste costs. It is not a general optimality guarantee for
  arbitrary semantic objectives, real-time deadlines, or concurrent audio jobs.
- **Evidence:** Sections 5-7 compare rollback, full restart, appending a revision,
  ignoring it, and an informed oracle. Tool scenarios include bookings/planning
  and reporting; metrics include judged quality, discarded work, compensation,
  steps, and tokens. Large mock-model ablations are distinct from real-LLM runs.
- **Use here:** compare preserving versus replacing pending bars; treat already
  heard speech as history that cannot be deleted. Evaluate instruction compliance
  separately from whether a new instruction was included in the prompt.
- **Caution:** inspect assumptions rather than importing the authors' broad claims
  about all existing agent frameworks. We have not replicated their artifact.

### 3. Incremental Dialogue Processing

**Venue/year:** Schlangen and Skantze, EACL 2009; expanded in Dialogue & Discourse,
2011. [2011 paper](https://aclanthology.org/2011.dnd-2.11.pdf).

- **Method:** connect processing modules with small, linked information units;
  permit additions, revisions/revocations, and commitments.
- **Important detail:** Section 3.4 discusses commitment flowing backward from a
  publicly observable action, and conflicts when supporting information is later
  revoked. Section 5 describes concrete messaging patterns.
- **Boundary:** this is an architectural framework, not an LLM GPU scheduling
  benchmark or an empirical guarantee of rhythmic speech timing.
- **Use here:** distinguish tentative transcript, proposed lyrics, prepared audio,
  committed audio, and heard output. Track dependencies between them. Both the
  commitment concept and revisable speech precede modern LLM agents.

**Related:** Skantze and Hjalmarsson, SIGDIAL 2010,
[Towards Incremental Speech Generation](https://aclanthology.org/W10-4301/),
studies simultaneous interpretation, planning, realization, and self-monitoring
in a Wizard-of-Oz comparison. This is a direct precedent for speech pipelines
that begin before the entire interaction is resolved, not a neural-model baseline.

### 4. Agentix

**Venue/year:** NSDI 2026; earlier preprint named Autellix.
[Paper](https://www.usenix.org/system/files/nsdi26-luo.pdf).

- **Method:** associate requests with programs and track program service, waiting,
  and execution dependencies. Program-level attained service handles sequential
  calls; the multi-threaded case uses critical-path information. Scheduling and
  locality-aware placement operate beyond individual requests.
- **Assumptions:** future arrivals and program structure are not known in advance;
  structure is observed as the program executes.
- **Evidence:** Sections 4-6 develop the policies and evaluate agentic workloads,
  end-to-end latency, throughput, and waiting behavior.
- **Use here:** a character is a logical history, not a dedicated GPU. Record the
  waiting and service of each character as well as overall throughput. Avoid
  presenting independent HTTP calls as a new multi-agent serving system.
- **Boundary:** shorter program completion is not identical to meeting a particular
  speaking slot or reacting to the latest utterance.

### 5. Staircase Streaming

**Venue/year:** 2025 arXiv preprint; no later venue verified in this review.
[Paper](https://arxiv.org/html/2510.05059v1).

- **Method:** forward partial chunks between proposer and aggregator agents;
  generate another chunk after incorporating new input. Smaller initial chunks
  reduce initial waiting; appending updates can preserve prompt-cache reuse.
- **Evidence:** Sections 2-3 test mixture-of-agents and debate using response
  quality, time to first token, and output throughput; chunk-size ablations expose
  a quality/latency/processing-cost tradeoff.
- **Use here:** a practical baseline for partial-message versus complete-message
  delivery. Count repeated prompt work and additional calls as costs.
- **Boundary:** chunked continuation does not mean arbitrary new input can be
  inserted into an already running ordinary chat-completion request. Today's
  prototype must expose when it actually takes a new context snapshot.

### 6. StreamMA

**Venue/year:** 2026 arXiv preprint, v3 dated October 27; no venue verified.
[Paper](https://arxiv.org/pdf/2606.05158).

- **Method:** use explicit reasoning-step boundaries and queues to pipeline agents
  in chains and other directed acyclic graphs; downstream agents make successive
  calls with growing context rather than waiting for the whole predecessor.
- **Evidence:** method, theoretical analysis, and evaluation compare streaming,
  serial, and single-agent protocols on reasoning benchmarks. The effectiveness
  analysis has cases favoring different protocols, not unconditional superiority.
- **Assumptions:** step decomposition, cache behavior, context costs, and dependency
  structure matter. The latency expressions do not substitute for measuring
  contention on our one shared model server.
- **Use here:** measure both partial-information errors and any readiness benefit.
  A count-to-ten utterance has little content to pipeline; use it to study turn
  arbitration, not as the sole test of step-streaming quality or speedup.

### 7. Time To Talk

**Venue/year:** Findings of EMNLP 2025.
[Paper](https://aclanthology.org/2025.findings-emnlp.608.pdf).

- **Method:** one LLM decision determines whether to speak; another generates the
  message. Participation prompts adapt to speaking frequency.
- **Evidence:** human-agent Mafia games, timing/content analysis, game outcomes,
  and participant judgments. Generated messages receive a length-dependent delay
  simulating human typing; each game includes one asynchronous agent.
- **Boundary:** social willingness to speak, permission to use the audio channel,
  and compute scheduling are three different decisions. This paper mainly informs
  the first, not sample-level overlapping audio or hard playback deadlines.
- **Use here:** preserve an explicit send/wait decision in the collaboration test,
  and include a deterministic turn coordinator. Do not require an extra LLM call
  to arbitrate every beat unless its cost is itself the experiment.

### 8. AsyncLLM

**Title/year:** LLMs are General Asynchronous Agents, September 2026 arXiv preprint.
[Paper](https://arxiv.org/html/2609.35427v1),
[artifact](https://github.com/dvmazur/async_llm).

- **Method:** inference coroutines use overlapping model-memory views. A modified
  mini-SGLang engine balances concurrent prefill/decode work, including chunked
  prefill, so reaction work is not blocked by larger background work.
- **Evidence:** Sections 3-4 cover architecture, mid-reasoning updates, streaming
  video, games, and monitoring; Appendix E contains controlled throughput tests.
- **Implication:** continuously listening/reacting through shared model state is
  already a direct research target, not an untouched extension of our project.
- **Use here:** distinguish an application event loop with separate requests from
  this lower-level inference capability. Implement the former first; do not call
  it shared-KV asynchronous decoding. Adopting this engine is outside today's
  bounded demo work.

### 9. Async-RBench

**Venue/year:** October 2026 arXiv preprint; no venue verified.
[Paper](https://arxiv.org/html/2610.32429v1).

- **Method:** concurrent subagents return potentially stale, duplicate, partial,
  or conflicting results. A controller releases results under defined conditions;
  event-specific contracts evaluate the correct response separately from final
  task success.
- **Evidence:** Sections 3-4 compare asynchronous/batched delivery and replanning
  behavior. Release timing is controlled, not a natural wall-clock arrival trace.
  Limitations explicitly include isolated target events rather than overlapping
  target events.
- **Use here:** give each scripted plot edit an expected observable consequence.
  Evaluate whether the right information was retained, revised, or ignored. Log
  event delivery, not merely API receipt or final song success.
- **Boundary:** this is a useful evaluation precedent; our tiny pilot is neither a
  reproduction nor evidence of superiority over its agents.

### 10. Our Previous StreamMUSE Paper

**Year:** 2026 arXiv version.
[Paper](https://arxiv.org/html/2606.11886v1).

Sections IV-V already model request interval, generated horizon, tempo, inference
and network latency, feasible configurations, and quality. A new experiment that
only varies lookahead and tempo risks repeating that contribution. Its fitted
latency functions belong to that model/deployment; do not assume the same
coefficients or distributions for Qwen plus MOSS plus local audio preparation.

Our proposed extension must add an evidenced decision about revision, resource
sharing, incomplete input, or perceptual/timing constraints, not simply a longer
list of tunable parameters.

## Practical Synthesis

The following are our proposed modeling choices, not borrowed theorems.

For output unit j, define:

- a_j: local time the required information becomes available.
- r_j: time its generation request is released.
- D_j: planned audible start; g_j: commit/device guard time.
- d_j = D_j - g_j: latest allowed preparation completion.
- F_j: actual preparation completion, including required downstream work.
- v_j: instruction/history version used; U_j: independently checked compliance.

On-time readiness is F_j <= d_j. A useful scheduled unit must also be valid and
compatible with the applicable instruction, not merely generated from a recent
version. Report deadline misses and semantic failures separately before combining.

For an edit arriving at e, define L_edit as the time of the first audibly compliant
response minus e. Report API acknowledgement, generation readiness, playback
scheduling, and observed audio separately. A callback timestamp estimates device
delivery, not necessarily the instant sound reaches a Bluetooth listener.

Remaining slack at time t can be estimated as:

    slack_j(t) = d_j - t - estimated_remaining_pipeline_time_j(t)

Estimate the whole remaining path from traces; adding individual stage P95s does
not generally produce an end-to-end P95. This is a heuristic diagnostic, not a
deadline guarantee. Compare a fixed policy against a slack-aware policy only
after costs and cancellation semantics are actually measured.

In 4/4, a bar lasts 240/BPM seconds. At 90 BPM this is 2.667 s. Two bars of
generation lead can give an initial preparation window near 5.333 s, but not a
guaranteed 5.333 s response window for an edit arriving midway through playback.
Existing commitments and the edit's arrival phase determine the actual window.

## Experiments That Could Change Our Mind

| Candidate question | Competent comparison | Evidence against making it the main contribution |
| --- | --- | --- |
| Which pending output should a live edit invalidate? | Keep queued output; replace all uncommitted; selective replacement with explicit dependencies | A simple rule meets both semantic and timing objectives across relevant loads |
| When is concurrent preparation useful for characters? | Sequential request execution versus bounded concurrent requests, with identical speaking slots and history visibility | Speed benefit disappears on shared compute or responses become less relevant |
| When should partial input trigger work? | Final transcript only versus tentative preparation with delayed commitment | Extra work/errors dominate any correct audible-response improvement |
| How much timing precision can intelligible speech tolerate? | Strict syllable anchors versus bounded strong anchors on identical source speech | No reliable listening benefit, or benefit requires unacceptable timing drift |

Do not select a main experiment only because its demo is entertaining. Prefer a
repeatable baseline failure, a controllable causal variable, and a result that can
be explained beyond rap. Negative findings can simplify the system and narrow the
paper scope; they must remain in the evidence ledger.

## Limits And Next Literature Work

This review establishes substantial overlap, not a proven gap. Before a paper
claim, extend coverage to control/anytime scheduling, cancellation under batched
inference, simultaneous translation, and incremental speech prosody. Inspect and
reproduce the most relevant artifact in an isolated environment after the pilot
identifies the actual bottleneck. Do not claim beat/tick generation experiments
when only bar-based production or simulated finer deadlines have been tested.
