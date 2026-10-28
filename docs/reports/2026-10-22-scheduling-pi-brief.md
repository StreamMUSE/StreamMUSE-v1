# Parameterizing and Modeling a Real-Time Generative System

**PI technical companion | StreamMUSE rap case study | Original: 22 October 2026 | Revised: 23 October 2026**

Read in order: **foundations -> group guides -> overview -> architecture -> model guides -> compact models -> evidence**.
The 24-page [illustrated brief](../../output/pdf/2026-10-22-scheduling-pi-brief.pdf)
is the presentation version. This document defines its notation and preserves
the qualifications behind each figure. The [original evidence report](2026-10-22-scheduling-research.md)
remains the reproducibility and service-handoff record.

## Reading Guide

| PDF pages | What to read there |
|---|---|
| 1 | Foundations: the work, musical units and how to read symbols |
| 2-8 | One guide per group, G1-G7: each variable, its units and what it means |
| 9 | Overview: "What exactly can change in this system?" |
| 10 | Architecture: where the work runs on the Mac and H200 |
| 11-20 | Model guides M1-M10: every equation in plain English with an illustrative rap example |
| 21-22 | Compact models: timing, resources, search, revisions and allowed output |
| 23-24 | Evidence: observations and fixed-trace deadline sensitivity |

Start with the guides before using the overview as a reference. Guide examples
are illustrations unless marked measured; the added teaching pages do not add
experiments or change the recorded results.
See the [validation record](#validation-of-this-revision) for build and review checks.

### Plain-Language Glossary

| Term | Meaning in this brief |
|---|---|
| Job / stage | Work for one output bar and version / one step in that work |
| Beat / bar / tick / sample frame | Musical pulse / group of beats / subdivision of a beat / one digital-audio time step; the current setup has 4 beats/bar and 4 ticks/beat |
| Flow / candidate / wave | Target syllable times and stresses / one proposed lyric line / one request for several proposed lines; rescue means an additional request |
| Lead / guard / reserve | How far ahead a stage plans / how early the choice freezes / time left for later steps and safety; ready-audio reserve in G7 instead means usable output ahead |
| Dependency / DAG | An earlier input or result a step needs / a directed dependency graph with no cycles |
| Artifact / revision / epoch | Saved lyric or audio result / version of content, timing or settings / identifier separating a restarted session from an older one |
| Eligible / fresh / fallback | Allowed to use / matches the output's required input versions / permitted alternative when fresh model audio is unavailable |
| Source-ready / received / prepared | Speech exists on the server / its package has reached the Mac / local timing, decoding and mixing are complete |
| Commit / enqueue / audible | Freeze the selected result / hand audio to playback / sound actually reaches the listener; these are different events |
| Temporal contract / granularity / phase | Rules for input availability and allowed output boundaries / size of a piece handled by a stage / offset of recurring boundaries |
| Contiguous coverage / critical path / slack | Ready output ahead with no hole / longest chain of required work / time left after subtracting estimated remaining work |
| Fixed-trace counterfactual | Ask which saved results would miss a different deadline without rerunning or changing their completion times |

**Names in the system diagram:** Qwen is the lyric-generating language model
(LLM); MOSS generates speech from text (TTS); MMS estimates speech-to-text
alignment; R3 is Rubber Band pitch-preserving time stretching; eSpeak is local
fallback speech. PCM stores audio samples directly; Opus compresses audio.
The manifest describes the audio and its timing. SSH carries the network
connection; WebSocket carries session events, HTTP fetches audio, and ACK means
acknowledgment. The sink is the audio output device; RTT is round-trip time.

**Reading symbols:** `j` identifies an output job, `v` a stage, `q` a resource,
`w` a candidate wave, `c` a candidate, `a` an artifact, and `class` an eligibility
rule set. `t` is clock time and `beta` is musical position in beats. Subscripts
identify the relevant job, stage or version. A timestamp says when; a duration
says how long. Times use milliseconds unless marked seconds. `S_jv` is an
execution start, not the candidate score `s(c)`; `A_jv` is input-ready time,
not the evaluated candidate set `X_jw`. `N_sessions` counts admitted sessions,
`N_wait` counts whole-bar waiting intervals, and `N_obs` denotes observations.
`1[condition]` is 1 when true and 0 otherwise; summing it counts passing tests.
`max` takes the largest value, union combines intervals, `sup` finds the longest
covered lead, and the ceiling brackets round upward to a whole number.

## 1. Parameterization Before Modeling

The system must turn a lyric into usable audio before its scheduled musical
start, using inputs that are already available. Each bar and version is a job;
its flow sets the target rhythm, and topic edits can require new work. Some
steps must wait for earlier results, and several jobs may compete for the same
worker. The seven groups describe these timing, work and playback rules.

### 1.1 Four Different Kinds of Quantity

| Kind | Definition | Examples |
|---|---|---|
| Control | Chosen by an experiment or scheduling policy | Lead, candidate counts, admission limit, commit unit |
| Workload/environment | Arrives from the user or execution environment | Edit times, topic/flow/history, network conditions |
| State | Changes as the system executes | Queue contents, revisions, ready artifacts, playback frontier |
| Derived/observed | Computed or measured from the other quantities | Deadline budget, valid yield, queue delay, fresh-success rate |

These distinctions prevent errors such as treating valid-candidate count as an
independent knob, or treating an observed latency as a fixed service guarantee.
A UI-configurable input can still be supplied by the workload rather than
controlled by the experiment.

![Primitive-to-group parameter map](../research/2026-10-22-parameter-map.png)

### 1.2 Merge Rule and Seven Groups

Group variables when they jointly answer one scheduling question, but keep
each variable's value and units. A group can contain counts, times, rules,
dependencies and saved results; it is not one number. The groups are not
claimed to be independent, statistically sufficient or the smallest possible
set. They organize the problem rather than forming a fitted statistical
reduction or a prediction of performance from seven inputs.

| Group | Primitive definitions, units and domains | Retained representation | Why these belong together |
|---|---|---|---|
| **G1 Time budget** | `B(beta)>0`: beats/min; `m`: beats/bar; `r`: ticks/beat; `h_v`: stage lead in beats; `gamma`: guard ticks; `M_v>=0`: downstream reserve in ms | Timeline `T`, stage deadlines `D_jv`, available budget `b_jv=D_jv-R_jv` | Tempo, lead and guard jointly create physical time. A lead count alone is not a deadline. The measured sweeps use `m=r=4`. |
| **G2 Work and capacity** | `C_jv>=0`: stage execution duration in ms, excluding separately modeled waiting; `p(v)`: execution resource `q`; `k_q`: effective execution slots; `N_sessions`: admitted sessions; `pi_q`: queue rule | Stage costs, placement and capacity `(C,p,k,N_sessions,pi)` | The same work has different completion times under different contention. In-flight requests, decoder sequences and source-render slots are distinct counts. |
| **G3 Useful search** | `n_w`: requested candidates in wave `w`; `W_max`: maximum additional rescue waves; `k_min`: desired valid count; `tau_s`: selectable score threshold; `V(c),s(c)`: validity test and score | Search policy `theta_s=(n_0,...,n_{W_max},W_max,k_min,tau_s,V,s)`; evaluated set `X_jw`, valid count `Y_jw`, selectable count `U_jw` and best score `s*_jw` | Budget and acceptance jointly determine usable output. More samples can still yield no selectable line. Valid/selectable counts and best score remain outcomes, not controls. |
| **G4 Revision work** | `E={(e_l,scope_l,delta_l)}`: accepted edit times in ms, affected spans and changed inputs; `chi_v`: cancellation policy; dependency graph `D_dep`; `kappa_j`: required input versions and revision cutoff for output `j` | Edit trace plus invalidation/cancellation and cutoff rules `(E,D_dep,chi,kappa)` | An edit matters through what it invalidates and whether obsolete work can stop. Average edit rate alone loses burst timing and scope. The input cutoff states which version must be satisfied; it is not a completion deadline. |
| **G5 Delivery** | `z_j`: package bytes; route and codec; `C_x,C_p`: transport and preparation durations in ms; `M_sink`: enqueue/device reserve in ms; `f_s`: sample frames/second | Source-to-ready path `(z,route,codec,C_x,C_p,M_sink,f_s)` | Server completion does not mean local readiness. Preserve network, preparation and sink boundaries rather than calling everything RTT. |
| **G6 Temporal contract** | `g_v`: stage chunk size in beats or a named unit; `phi_v`: periodic boundary offset in the same unit; `A_jv`: latest required input availability in ms; `r_pol_jv`: earliest start permitted by policy, in ms; `pred(j,v)`: required earlier job/stage pairs; `L_user,L_model`: turn spans in bars | Chunk sizes, boundaries and dependencies `(g,phi,A,r_pol,pred,L_user,L_model)`, with capture grace, transition and timeout rules | Allowed release and commit times depend on both unit size and earlier inputs. Sample-accurate playback cannot substitute for beat-level lyric generation. |
| **G7 Eligible reserve** | `K_a`: actual producer dependency key; `I_a`: artifact's beat interval; `eta_a`: readiness/integrity/ownership/policy tests; `beta_play`: current beat; `pi_f`: fallback order; `class`: allowed-output rule set, using `kappa_j` from G4 | Eligible interval union `C_class(t)` and contiguous lead `H_class(t)`, separately for each class | A far-future ready bar does not fill a nearer hole. Stale or fallback output can preserve continuity while failing freshness. |

**Inputs and settings still matter in G2/G3.** Topic, target flow and selected-text
history describe what the writer must produce. Model and configuration versions,
context/input length, token limit, sampling temperature, hardware and cache state
can change execution time, the proposed lines and how many pass the checks.
Keep these values with the workload and configuration; the groups do not assume
them away or treat their effects as measured causal relationships.

**G6 includes waiting rules.** Capture grace is extra time to receive input for
the bar that just ended; a transition bar separates capture from the response;
a response timeout limits how long policy waits for remote output. These rules
can delay when a response is allowed even if fallback audio is already prepared.
They affect input availability and eligibility, not just execution duration.

**Cross-group coupling is intentional.** G3 controls affect G2's observed work;
G4 changes G7 eligibility; G6 determines G1 release/budget; G5 is a named suffix
of G2's complete stage graph. Do not add its costs a second time. `h_v` is stage
lead in beats and `L_v` is the same lead in bars (`h_v=m*L_v`), not a second
independent horizon.

**Two concrete merges.** At fixed tempo, with lead measured from the relevant
legal release and the guard as the only downstream reserve, lead and guard
reduce to `b=(m*L-gamma/r)*60000/B` ms. Otherwise use `b=P-M-R`, count the guard
once, and include all downstream reserve. But `L=1,B=60` and `L=2,B=120` need not be
equivalent systems: without guard both yield 4 seconds, yet their edit exposure,
semantic horizon and workload can differ. Similarly, `n=64` is not 64 independent
usable choices: the acceptance gates, duplication and shared context remain in G3.

### 1.3 Coverage of the Existing Registry

The registry contains **21 typed entries**, including vectors, observations and
states. It does not contain 21 independent scalar controls. This crosswalk is
complete; the formulas above expand shorthand where needed.

| Group | Existing registry entries | Expanded or formalized here |
|---|---|---|
| G1 | `tempo`, `lead_vector`, `counterfactual_deadline` | Beat clock, meter, guard, named-stage deadline and budget |
| G2 | `resource_placement`, `request_concurrency`, `source_render_capacity`, `stage_cost_profile` | Explicit resource occupancy and queue policy |
| G3 | `candidate_policy`, `usable_yield`, `writer_latency` | Gates vs score vs desired stopping yield; no candidate-independence assumption |
| G4 | `edit_trace`, `latest_source_delay`, `cancellation_and_reuse_policy` | Edit scope, dependency descendants, release/freshness cutoffs |
| G5 | `transport_profile`, `mac_source_ready`, `mac_warp_mix_cost` | Source receipt vs prepare vs enqueue boundaries |
| G6 | `measured_output_units`, `finer_output_granularity` | Granularity/phase vectors, input availability and causal edges |
| G7 | `eligible_ready_coverage`, `cache_dependency_key`, `fresh_vs_continuous_output` | Policy-class coverage, holes and contiguous ready lead |

The [full registry](../research/2026-10-22-reduced-scheduling-registry.json)
retains implementation status, source, change boundary, units and evidence
references. The [coverage catalogue](../superpowers/specs/2026-10-22-realtime-scheduling-model-design.md#4-parameter-coverage-catalogue)
also includes model/context/precision, resource memory, batching, retries,
preemption, clocks, transport failure, cache expiry, fairness and measurement
overhead. Those remain conditioning variables inside the appropriate groups;
this report does not claim that they are irrelevant or experimentally eliminated.

## 2. Comprehensive System Mapping

![Parameterized Mac/H200 system](../research/2026-10-22-comprehensive-system.png)

This diagram follows the visual organization of `rap-v1.1.svg` without overwriting
that historical version. Commands/events and audio bytes use separate paths:

1. **Mac:** owns musical time, input capture, plan edits and playback obligations.
2. **H200:** generates candidates, evaluates/selects a lyric, synthesizes source
   speech and estimates syllable onsets. Selected text conditions later writing.
3. **Transport:** WebSocket session/control/events plus per-bar HTTP audio fetch.
   Audio is a source package, not a remotely timed speaker stream.
4. **Mac:** decodes, maps source onsets to flow targets, warps and mixes audio,
   then selects an eligible prepared bar at its commit cutoff and enqueues it.
5. **Commit:** freezes the chosen whole bar one tick before its start, except
   startup. Continuous edits target eligible uncommitted work and use an ordered
   fallback path. Alternating mode echoes captured rhythm while each logical
   response waits; topic edits apply to the next capture. Sink progress feeds
   the Mac controller, while commit messages advance H200 planning.

**Version separation:** architecture includes the newly implemented tempo and
alternating-turn features. Numerical experiments below used the isolated
`bde2f62c` baseline, Qwen2.5-7B-Instruct and MOSS-TTS-v1.5 with MMS alignment;
they are not acceptance measurements of the new tempo/alternating features or
benchmarks of the model variants shown in historical v1.1. Baseline transfer was
PCM16; Opus is an optional implementation path. The formal job identity and
complete observation chain are model/instrumentation contracts, not a single
existing wire key or a fully instrumented acoustic trace.

## Equation Walkthroughs (Illustrative)

M1-M10 correspond to PDF pages 11-20. Every number and scenario in this section
is **illustrative, not a measurement**. These worked examples explain the
technical equations in Sections 3, 4 and 6; they do not fit a model,
calibrate a score or add observations to the evidence tables. Unless stated
otherwise, clock times and durations are in milliseconds on one common clock.
Decimal displays are rounded; calculations use the unrounded values.

### M1. Musical Position to Clock Time and Samples

**In plain English:** `T(beta)` adds up the seconds taken by every beat up to
the chosen position. `P_j` subtracts the reference origin and converts seconds
to milliseconds. The sample-position expression converts elapsed seconds from
the session origin to audio sample frames, then rounds to an integer.

**Illustrative rap example:** the first four beats run at 90 BPM and the next
four at 120 BPM. At beat 8, with `T_0=t_ref=0`, the integral gives
`4*60/90 + 4*60/120 = 14/3 = 4.666667` seconds. Thus `P_j=4666.667` ms and,
at `f_s=48000` sample frames/second, the position is
`round(48000*(14/3))=224000` frames. Sample frames are time steps, not a count
multiplied by the channel count; this schedule is not measured audible onset.

### M2. Deadline and Available Budget

**In plain English:** `D_jv` is the planned playback time minus the reserve
needed after the named stage. `b_jv` is the time between legal readiness and
that deadline, not necessarily the time available from the original request.

**Illustrative rap example:** a prepared rap bar is scheduled for `P=8000` ms.
Reserve `M=200` ms for commit/sink handling, including the guard once, so
`D=8000-200=7800` ms. If the stage becomes legally ready at `R=4000` ms,
its available budget is `b=7800-4000=3800` ms. Do not subtract the guard again.

### M3. Legal Readiness, Queueing and Finish

**In plain English:** `R_jv` is the latest of the policy release, required-input
availability and all predecessor finishes. `S_jv>=R_jv` prevents starting
before those requirements are met. `Q_jv` is the subsequent wait for execution;
`F_jv` adds execution duration to the actual start, not to legal readiness.

**Illustrative rap example:** speech rendering for the next bar is policy-allowed
at 4000 ms, has the required topic/flow at 4300 ms, and receives that bar's
selected lyric from the writer at 4500 ms. Then `R=max(4000,4300,4500)=4500` ms.
The renderer starts it at `S=5000` ms: `Q=5000-4500=500` ms. With illustrative
execution duration `C=1800` ms, it finishes at `F=5000+1800=6800` ms.

### M4. One Shared Worker Is One Execution Slot

**In plain English:** the capacity sum counts stages executing on resource `q`
at time `t`; that count cannot exceed `k_q`. The intervals include their start
but exclude their finish, so one stage can start exactly when another ends.

**Illustrative rap example:** a shared source-render worker has `k_q=1`.
Session A renders during `[5000,6800)` ms and session B during `[6800,8600)` ms.
At 6800 ms only B occupies the slot. Starting both at 5000 ms would make
occupancy 2, violating capacity 1, even though both sessions may be admitted.

### M5. Critical Path and Remaining Slack

**In plain English:** `CP_j` sums execution costs along each required dependency
path and takes the longest path, not the sum of parallel branches. Slack
`sigma_j(t)` subtracts both the current time and estimated remaining work/wait
from the deadline. A negative value means the estimate no longer fits.

**Illustrative rap example:** writer 200 + speech 1800 + transfer 100 + mix 50
gives a 2150-ms path; parallel drums 100 + the same final mix 50 gives 150 ms.
The critical-path lower bound is `max(2150,150)=2150` ms, not 2300 ms.
For another illustrative snapshot, `D=7800`, `t=6000` and remaining-work/wait
estimate 1500 give slack `7800-6000-1500=300` ms. Adding 600 ms of waiting to
that estimate gives `7800-6000-2100=-300` ms. This remaining-time estimate is
illustrative and unfitted, not a measured or calibrated predictor.

### M6. On Time Does Not Necessarily Mean Fresh

**In plain English:** lateness `ell_j` is prepared finish minus deadline;
nonpositive is on time. `Z_j^fresh` is 1 only when the prepared bar is both
on time and eligible for the obligation's required input/revision cutoff.

**Illustrative rap example:** for `D=7800` ms, assume all other eligibility
checks pass. A fresh-topic bar finishing at 7500 has lateness -300 ms and
`Z=1`; an old-topic bar finishing at 7500 also has lateness -300 ms but `Z=0`;
a fresh-topic bar finishing at 7900 has lateness +100 ms and `Z=0`.
Here old-topic means incompatible with the declared cutoff, not merely older
than a later edit. Prepared-stage success is not proof of audible success.

### M7. Evaluated, Valid and Selectable Candidates

**In plain English:** `Y_jw` counts distinct evaluated candidates that pass the
validity gates. `U_jw` counts those valid candidates whose score also meets the
threshold. `s*_jw` is the largest score in that selectable set; it is undefined
when the set is empty, not zero.

**Illustrative rap example:** four distinct evaluated lyric candidates target
nine syllables; assume all other validity gates pass for the nine-syllable lines.

| Candidate | Syllables | Valid? | Toy score (0-100) | Selectable at toy threshold 75? |
|---|---:|---|---:|---|
| A | 9 | Yes | 82 | Yes |
| B | 9 | Yes | 65 | No |
| C | 10 | No | Not scored | No |
| D | 9 | Yes | 78 | Yes |

Thus `Y=3`, `U=2`, and `s*=max(82,78)=82`. These 0-100 scores and threshold 75
are teaching inventions, **not production thresholds or score calibration**.
The actual measured threshold was 0. Invalid C needs no score to be rejected;
none of these internal scores is a human rap-quality measurement.

### M8. Coverage Is Limited by the First Hole

**In plain English:** `C_class(t)` combines the beat intervals of artifacts
eligible under one output rule set. `H_class(t)` measures how far playback can
advance from the current beat without leaving that union, not total stored audio.

**Illustrative rap example:** at play beat 0, fresh bars cover `[0,4)` and
`[8,12)`. Their union has a hole at `[4,8)`, so `H_fresh=4` beats, not 8 or 12.
If an allowed fallback fills `[4,8)`, the allowed-output union is `[0,12)` and
`H_allowed=12` beats. The fallback fills the continuity hole without becoming
fresh model output; eligible coverage alone is not a device no-gap measurement.

### M9. Round Up to a Legal Response Boundary

**In plain English:** `b_next` rounds a position up to the next allowed boundary
with spacing `g` and phase `phi`, keeping an exact boundary unchanged.
`N_wait` counts whole-bar intervals needed after capture to fit the eligible
prepared response plus sink reserve; `P_response` adds those intervals to
capture end. The outer maximum prevents a negative waiting count.

**Illustrative rap example:** with four beats/bar at 90 BPM,
`tau=8000/3` ms and `t_cap=8000/3` ms. A first policy-eligible prepared response
at `F_elig=6200` ms and sink reserve `M_sink=200` ms require a boundary at or
after 6400 ms. Taking `g=tau` and `phi=t_cap`, the ceiling argument is
`(6200+200-8000/3)/(8000/3)=1.4`. Therefore `N_wait=ceil(1.4)=2`, and both
`b_next(6400;tau,t_cap)` and `P_response=t_cap+2*tau` give 8000 ms.
The earlier boundary at `16000/3=5333.333` ms was missed. These two elapsed
intervals include transition/echo waiting; they are **not an exact echo count**.
Capture grace and policy restrictions must already be included in `F_elig`.
If no finite eligible `F_elig` exists, record nonresponse rather than inventing
a response onset or dropping the observation.

### M10. Shift a Deadline, Not the Saved Work

**In plain English:** `D_j^cf` converts the bar ordinal plus lead, minus the
one-tick guard, into a hypothetical source-stage cutoff. `miss_j^cf` is 1 if
the unchanged finish is too late, the job failed, or the result is not latest
for the obligation. Finishing exactly at the cutoff is not a timing miss.

**Illustrative rap example:** hold a hypothetical successful, current-version
source finish at `f=5000` ms, with 90 BPM, `m=r=4` and first-bar ordinal `j=0`.
Leads `L=1/2/3` give cutoffs 2500 / 5166.667 / 7833.333 ms: the same finish is
late / on time / on time, so misses are 1 / 0 / 0. An old-version result still
misses even if it finishes early. This hypothetical trace is not a measured
observation or a live lead/tempo rerun, and it omits preparation and the sink.

## 3. Model: Timing, Causality and Resources

### 3.1 Common Clock and Output Obligation

Let beat position be `beta` and tempo be `B(beta)` beats/min. Let `T_0` be the
session origin in seconds and `t_ref` a common reference origin on that clock:

$$T(\beta)=T_0+\int_0^\beta\frac{60}{B(u)}\,du,\qquad
P_j=1000\,[T(\beta_j)-t_{ref}].$$

`P_j` is the nominal start of output `j`, in common-reference milliseconds.
Within one measured session `t_ref` is its initial request time; across sessions
use one shared origin before comparing resource occupancy. Cross-host absolute
timestamps require calibration; this study uses local spans/receipts instead.
Its sample position is `round(f_s*(T(beta_j)-T_0))`, with `f_s` sample frames/second.
This is an intended schedule, not a measurement of acoustic onset.

Define each obligation as an output interval, an input/revision cutoff `kappa_j`,
an eligibility class (fresh model, older remote, fallback, etc.), a named
completion boundary, and a deadline. Its job identity is
`(session, epoch, span, content_revision, timeline_revision, config_revision, stage, attempt)`.
Plan edit sequence is not interchangeable with per-bar content revision.

$$D_{jv}=P_j-M_{jv},\qquad b_{jv}=D_{jv}-R_{jv}.$$

Use positive finite tempo with finite timeline integrals, positive cadence,
and finite nonnegative reserves `M_jv,M_sink`. Deadlines and available budgets
can be negative. `M_jv` reserves work/margin downstream of stage `v`. For a prepared-bar deadline,
it can be sink reserve; for a source-ready deadline, it must additionally
account for remaining delivery/preparation. A deadline after request release
can still have negative budget if required input becomes available too late.

### 3.2 Stage DAG and Resource Occupancy

For job `j` and stage `v`, `r_pol` is the earliest time policy permits work,
`A` is when all required input versions are available, `S` is execution start,
and `F` is finish. These are timestamps in ms; `C` is execution duration in ms,
excluding the waiting duration `Q` modeled separately:

$$R_{jv}=\max\left(r^{pol}_{jv},A_{jv},\max_{(i,u)\in pred(j,v)}F_{iu}\right),$$
$$S_{jv}\ge R_{jv},\qquad Q_{jv}=S_{jv}-R_{jv},\qquad F_{jv}=S_{jv}+C_{jv}.$$

Empty predecessor maxima are zero on the chosen common-reference clock. `R` is logical
readiness; `S` may be later because a resource is occupied or the policy waits.
For unit-slot stages assigned to resource `q` (distinct from ticks/beat `r`):

$$\sum_{j,v:p(v)=q}\mathbf{1}[S_{jv}\le t<F_{jv}]\le k_q.$$

For heterogeneous occupancy, multiply each occupancy indicator by resource
demand `d_jvq` and use capacity `K_q` in matching units; unused resources have
zero demand. Stages here are nonpreemptive; preemption requires execution
segments instead of one start-finish interval. Decoder batching needs its own capacity/cost
model; API `n`, active sequences and render-lock capacity are not substitutes.
No constant or iid service distribution is assumed. Input length, hardware,
warmness, search policy and concurrent work may all condition `C_jv`.

The important cross-bar edge is selected text of bar `j` -> writer of bar `j+1`.
Rendering `j` may overlap writing `j+1`; independent sessions may still queue
behind a shared renderer. The measured renderer lock permits one source
synthesis/alignment call at a time.

### 3.3 Slack and Deadline Success

$$CP_j=\max_{\rho\in paths\_to(j)}\sum_{(i,v)\in\rho}C_{iv},\qquad
\sigma_j(t)=D_j-t-\widehat{W}^{rem}_j(t).$$

`rho` denotes one dependency path. `CP` is the longest chain of required execution work, a lower bound rather than
a completion predictor. `sigma` is slack: time until the deadline minus the
estimate of work and waiting still needed.
`W_hat_rem` must include remaining dependencies, queues, delivery and preparation
for the named boundary. It is a placeholder: this study did not fit or calibrate
a remaining-time predictor. Ignoring queues makes a service-only slack optimistic.

$$\ell_j=F_{j,prep}-D_{j,prep},\qquad
Z^{fresh}_j=\mathbf{1}[\ell_j\le0]\,\mathbf{1}[a_j\text{ satisfies }\kappa_j].$$

Here `prep` denotes preparation, `ell_j` is lateness (positive means late), and
`a_j` is the artifact covering obligation `j`, with its preparation finish
being tested. The second indicator includes readiness, integrity, timing and required input
identity. This is prepared-stage fresh success; audible success additionally
needs sink/acoustic evidence. Do not retroactively invalidate an already frozen
bar merely because a later edit arrives. Do not add independent stage medians
or percentiles and call the result an end-to-end percentile.

## 4. Model: Search, Revision, Granularity and Coverage

### 4.1 Useful Search Is an Outcome

Let `X_jw` be the de-duplicated evaluated candidate set after waves `0..w`;
`V(c)` is validity under the applicable gates and `s(c)` is the internal score.
Wave 0 is the initial request; rescue waves are additional requests. The slides
omit `w` when referring to the final evaluated set `X_j`:

$$Y_{jw}=\sum_{c\in\mathcal{X}_{jw}}\mathbf{1}[V(c)=1],\qquad
U_{jw}=\sum_{c\in\mathcal{X}_{jw}:V(c)=1}\mathbf{1}[s(c)\ge\tau_s],$$
$$
s^*_{jw}=\max_{c\in\mathcal{X}_{jw}:V(c)=1,\,s(c)\ge\tau_s}s(c).$$

`Y` counts valid candidates; `U` counts selectable candidates, evaluating scores
only for valid candidates. Invalid candidates need not have scores. `s*` is undefined
when the selectable set is empty, not zero. Rescue can stop when the desired
valid count is met, when the wave budget is exhausted,
or under a future deadline-aware policy. The current writer stops on valid count
alone and checks selectability afterward; a positive score threshold can make
those different tests. With the measured threshold zero, retain both definitions
rather than assuming equivalence for other policies.
The deployed desired minimum is three; one selectable result can still succeed
after exhaustion. The study varied one-wave `n=16/32/64`, not rescue timing.
Candidate independence is not established: do not infer no-valid probability
as `(1-p)^n`. Internal score and structural validity do not measure intelligibility
or human-rated rap quality.

### 4.2 Edits Change Eligibility, Not Just Latency

An edit `e_l` changes a declared dependency set. Invalidate its affected
descendants that remain uncommitted. Completed source is reusable only when
all actual synthesis inputs and producer/config versions remain compatible;
prepared audio additionally depends on target flow, timeline, output format
and mix. Tempo may also condition synthesis: source reuse is not automatically
tempo-independent. Cancellation support determines whether obsolete work stops
or still occupies a resource. Executed obsolete work and cancellation latency
were not measured in this study.

Define `eligible_class(a,t)` as ready at the class's required stage by `t`,
integrity-valid, ownership/epoch/policy-eligible, and compatible with all actual
producer dependencies at the class's declared obligation cutoff. Missing
required dependency keys prohibit reuse. For a committed artifact `a`, eligibility is evaluated against the obligation's
cutoff, not against every future edit. Separate classes permit continuity
fallback without labeling it a fresh-model success.

$$\mathcal{C}_{class}(t)=\bigcup_{a:\,eligible_{class}(a,t)} I_a,\qquad
H_{class}(t)=\sup\{h\ge0:[\beta_{play},\beta_{play}+h)\subseteq\mathcal{C}_{class}(t)\}.$$

`C_class` is the set of eligible beat intervals; `H_class` is the number of
contiguous ready beats ahead of the playhead. The label `class` names the
allowed-output rules, such as fresh-only or fallback-permitted. Ready bars 1
and 3 with bar 2 missing do **not** provide three bars of
continuous reserve. Availability of a fallback is distinct from actual no-gap
device output. Full class-specific coverage telemetry remains model-only.

### 4.3 Realtime Has Multiple Granularities

The complete vector has 18 components: observation, aggregation, release/phase,
submission, replanning, context, horizon, selection, text commit, TTS, alignment,
warp/mix, transfer, cache, commit/phase, playback, cancellation and measurement.
Keep phases beside periods. A positive cadence `g` with phase `phi` has its
boundary at or after `x` at

$$b_{next}(x;g,\phi)=\phi+g\left\lceil\frac{x-\phi}{g}\right\rceil.$$

Here all three arguments use the same time unit; at varying tempo, apply this
to musical position and then use `T`. Current semantic generation, selection,
source, preparation and commit are bar-level. Flow slots use ticks; PCM placement
uses samples. Finer placement does not imply finer semantic response. Beat/tick
generation requires partial-output dependencies and compatible synthesis/
commit contracts, not simply increasing tick frequency.

**Two modes, not two parameter settings of the same input contract:**

- **Steering:** topics/flow/tempo modify eligible future obligations while the
  model keeps playing. ASAP means the earliest eligible uncommitted output;
  it does not mean zero-latency acoustic change.
- **Alternating:** collect one complete user bar, then replay its rhythm while
  waiting for a model response. Replay can also occur between logical response
  bars; later responses may use eSpeak of the same generated text, and an
  incomplete turn may terminate on timeout. Topic edits affect the next capture,
  not the active response. `L_user=1` is currently fixed; `L_model=1..8` is
  configurable while stopped, default 2. Generalized user-span control is future
  work. These turn lengths belong to this mode's causal contract.

For a simple constant-tempo, whole-bar echo policy, let the nominal capture end be `t_cap`,
`tau=60000*m/B` ms and the first policy-eligible, timing-compatible prepared
response become available at finite `F_elig`. `M_sink` includes the commit guard:

$$N_{wait}=\max\left(0,\left\lceil\frac{F_{elig}+M_{sink}-t_{cap}}{\tau}\right\rceil\right),
\qquad P_{response}=t_{cap}+N_{wait}\tau.$$

Capture grace and all remaining input availability must be included upstream of
`F_elig`; the current implementation has a 100-ms grace and a transition bar.
`N_wait` counts elapsed bar intervals including transition, not the number of echoes.
First-response eSpeak may already be prepared but policy-ineligible until the
remote wait timeout, so preparation time alone must not be used as `F_elig`.
This is a structural first-response model with boundaries anchored at `t_cap`, not
a fitted response forecast or measured acoustic latency. Echo maintains rhythm
but does not make the model response faster. Response-timeout/nonresponse
observations must not disappear from statistics. If no finite eligible `F_elig`
exists, this equation assigns no response onset; retain the nonresponse case.

## 5. What the Measurements Support

![Observed patterns](../research/2026-10-22-observed-patterns.png)

| Probe | Observation | Supported interpretation | Not established |
|---|---|---|---|
| Search budget | H200 mean valid: 4.17 / 6.50 / 11.17; selectable attempts: 10/12, 10/12, 12/12 at n=16/32/64 | Search budget and useful yield must be modeled together | 100% reliability, independent samples, optimal budget or higher perceptual quality |
| Shared renderer | Median response total 1.806 -> 3.583 s with two admitted sessions; MOSS spans 1.782 -> 1.780 s | Admission count is not execution capacity | Pure queue delay, generalized throughput scaling or an isolated GPU kernel cost |
| Revision trace | Latest-topic first source: 1.917 / 3.799 / 3.826 s medians for no/one/burst edits | Freshness can require replacement after old work executes | Causal cost per edit: final topics differ and only three matched seeds were tested |
| Local preparation | 12 calls: median 49.99 ms, max 59.17 ms | Preparation is a distinct measured downstream stage | Acoustic readiness, a latency upper bound, or negligible cost for all flows |
| Mac receipt | Two source bars received at 5.786 / 7.794 s from session request | Client-observed readiness can differ from server stage spans | Pure SSH/network time: upstream work and connection setup are included |

Budget measurements: 12 matched topic/template/seed workloads per budget/route;
six topics used twice, three nine-slot templates, empty history, 32-token cap,
temperature 1, no rescue. All observations, including cold CMU initialization,
remain in the primary analysis. `n=32` always occupied the middle order position;
seed reproducibility is best-effort and the server sequence limit was 48.
The Mac and H200 route runs were separate in time, not a paired network experiment.

Concurrency: three paired groups, each two sessions x two bars, 12 bars per
condition. Bar observations are clustered, not 12 independent concurrency trials.
Residual `render_total - MOSS - MMS` mixes waiting and uninstrumented overhead.
Numerical bars in the plots are separate medians, never additive components.

Short-flow diagnostics are not pooled into these measurements: four-slot and
six-slot fixtures showed concentration/novelty-gate failures despite larger
budgets. They motivate retaining gates and workload in G3, not a general rate.
No causal importance ranking or statistical significance test is claimed.

## 6. Deadline Sensitivity, Not a New Live Benchmark

For measured source-ready time `f_j`, zero-based bar ordinal `j` (first bar `j=0`),
lead `L` bars, `m=r=4`,
and one-tick guard, set `t_ref=T_0=initial request time`, hold the trace fixed and
change only the hypothetical cutoff:

$$D^{cf}_j=\left[(L+j)m-\frac{1}{r}\right]\frac{60000}{B},\qquad
miss^{cf}_j=\mathbf{1}[f_j>D^{cf}_j\lor\neg success_j\lor\neg latest_j].$$

![Fixed-trace deadline sensitivity](../research/2026-10-22-deadline-sensitivity.png)

At 90 BPM, first-bar cutoffs for lead 1/2/3 are **2.500 / 5.167 / 7.833 s**.
Observed source-ready misses:

| Trace | Lead 1 | Lead 2 | Lead 3 |
|---|---:|---:|---:|
| Sequential sessions | 0/12 | 0/12 | 0/12 |
| Concurrent sessions | 9/12 | 0/12 | 0/12 |
| No edits | 0/6 | 0/6 | 0/6 |
| One edit | 6/6 | 0/6 | 0/6 |
| Burst edits | 6/6 | 0/6 | 0/6 |
| Mac two-bar session | 2/2 | 1/2 | 0/2 |

These values are **source-stage fixed-trace counterfactuals**. They omit local
preparation/enqueue/acoustics and hold service/queueing fixed, although changing
lead or tempo in a live system can change admission, exposure to edits and work.
One guard tick is an analysis convention, not a measured sufficient delivery
reserve. Original output obligations survive edits. Zero observed misses is
not a bound on tail risk.

### 6.1 Concise Research Position

The defensible model is a **deadline-constrained, revision-aware DAG**, not one
RTT threshold. Its operating point can be summarized as available budget,
resource-constrained remaining work, useful search, revision work, downstream
delivery, causal granularity and eligible reserve. Preserve those seven
mechanisms; vary fewer controls in any one experiment.

For a first controlled study, vary **lead/tempo (G1), admitted load (G2), search
policy (G3), and edit timing/scope (G4)**. Fix and record delivery configuration
(G5), semantic contract (G6) and fallback policy (G7), while observing their
state and outcomes. This is a tractable experiment design, not evidence that
the fixed groups are unimportant. No weighted optimization objective is chosen yet.

First instrumentation priorities: request/release/start/finish timestamps with
job and revision identity; separate queue wait from service; source receipt,
prepare, commit, enqueue and device timestamps; invalidation/cancellation work;
class-specific contiguous coverage; every timeout and nonresponse. Report
continuity, fresh on-time output, validity, response gap and perceptual quality
separately. Quality and acoustic timing are presently unmeasured in this study.

### 6.2 Relation to the Prior StreamMUSE Paper

The [2026 StreamMUSE paper, Section IV](https://arxiv.org/html/2606.11886v1)
models inference interval, generated span and latency relative to musical time.
This study retains the physical/musical clock distinction, then adds a
multi-stage DAG, useful-output gates, revisions and local preparation. Candidate
count is not generated musical span; input availability is not inference
interval. We do not import that paper's fitted latency distribution or
configuration guarantees into this uncalibrated pipeline.

## Provenance and Reproduction

- Numerical source: [analysis JSON](../research/2026-10-22-scheduling-evidence.json), including raw input paths and SHA-256 hashes.
- Model status: [typed registry](../research/2026-10-22-reduced-scheduling-registry.json); theoretical quantities are not represented as measurements.
- Code anchors: `application/rap/lyric_session.py`, `lyric_writer.py`, `bar_stream.py`, `interaction.py`, `bar_audio.py`; `domain/rap/timeline.py`; `infrastructure/rap/moss_aligned_phrase.py`, `audio_output.py`; `presentation/rap_lyric_stream_server.py`.
- Regenerate figures/PDF with `python scripts/experiments/build_scheduling_pi_brief.py` in an environment containing Matplotlib, NumPy, ReportLab and pypdf. Optionally pass `--pdf-python /path/to/python-with-reportlab-and-pypdf` to use a separate PDF runtime. The builder checks recorded input hashes and data invariants and performs no model requests or production changes.
- Editable diagrams: [parameter map](../research/2026-10-22-parameter-map.svg), [comprehensive system](../research/2026-10-22-comprehensive-system.svg).

Feature acceptance remains in the [separate delivery report](2026-10-22-feature-delivery.md).

### Validation of This Revision

**23 October 2026: verified.**

- Rebuilt the 24-page vector PDF and rendered all pages with PDFium. Visually
  inspected all ten new model guides and the revised compact-model reference.
- Checked page numbering, M1-M10 order, extractable text and teaching labels.
  All 24 pages passed text-boundary checks; the final ten model guides also
  passed pairwise text-overlap checks.
- Worked-example arithmetic assertions passed. An independent equation review
  identified a score-domain ambiguity; the selectable-count sum now evaluates
  scores only for valid candidates. The explanatory example and companion agree.
- Rechecked all six primary evidence hashes and all 90 deadline cells.
  Scheduling-evidence unit tests: **51 passed**. `git diff --check` passed.
- No model requests, live feature changes or changes to recorded measurements
  were made for this teaching revision.

### Historical Fourteen-Page Validation

The following recorded checks belong to the prior 14-page teaching revision;
they do not validate the new 24-page expansion:

- The 14-page PDF was rebuilt; all pages were rendered and visually inspected,
  including the new group guides and updated model pages.
- Text and page-order assertions passed. No out-of-bounds text or overlap
  between variable labels and the meaning column was found.
- All six primary evidence hashes and all 90 deadline cells were checked.
- Scheduling-evidence tests: **51 passed**. No new model requests or live
  changes were made.

### Historical Six-Page Validation

The following checks belong to the previous six-page revision, not validation
of the expanded 24-page teaching brief:

- All six primary evidence files matched their recorded SHA-256 hashes; all 90
  deadline cells were recalculated and checked by the artifact builder.
- Scheduling-evidence unit suite: **51 passed**. No new model requests or live
  feature changes were made for that report revision.
- Independent formal-model and current-code architecture reviews approved the
  corrected companion and diagram content.
- All six pages of that vector PDF were rendered with PDFium and visually
  inspected. Text extraction and six-page count were checked after assembly.
