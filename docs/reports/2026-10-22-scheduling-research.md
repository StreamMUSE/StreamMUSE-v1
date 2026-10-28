# Scheduling Research: 2026-10-22

**Original report: 22 October 2026. Teaching revision: 23 October 2026.**

**PI-facing revision:** start with the [illustrated technical brief](2026-10-22-scheduling-pi-brief.md)
and its [24-page PDF](../../output/pdf/2026-10-22-scheduling-pi-brief.pdf).
Reader map: page 1 foundations; pages 2-8 G1-G7 guides; page 9 overview;
page 10 architecture; pages 11-20 M1-M10 model guides; pages 21-22 compact models;
pages 23-24 evidence. Start with the companion's
[reading guide and plain-language glossary](2026-10-22-scheduling-pi-brief.md#reading-guide)
before the technical material. This original report remains the detailed evidence and
service-handoff record; its measurements have not been replaced or rerun.
The [equation walkthroughs](2026-10-22-scheduling-pi-brief.md#equation-walkthroughs-illustrative)
give every model equation a plain-English reading and a concrete rap example;
all new examples are explicitly illustrative, not measured evidence.
The 24-page build, render, equation and evidence checks are documented in the
[companion's current validation record](2026-10-22-scheduling-pi-brief.md#validation-of-this-revision).
Earlier validation records remain separately labeled historical.

**Status, finalized 07:12 PDT:** research implementation and evidence are
complete: seven-group registry, paired figure, structural SVG/PNG diagram,
raw archives and verified analysis; independent rigor review is complete.
Feature release acceptance remains separate.
GPU1/GPU2 were released at 05:38 and receive no further research load.
GPU0/18601 was safely stopped at 06:16:47; GPU1/GPU2 and the existing local
18620 tunnel were verified healthy afterward and left running.
Exact deployment/restart instructions are in [Service Handoff](#service-handoff).
These baseline results are not new-feature or device-playback acceptance.

**Separate feature-debug handoff:** the requested four-slot `space` pair used
all 80 candidates per line and found two valid each. Bar 1's initial 32 choices
collapsed to three unique five-syllable lines, all rejected for a four-slot
flow. No all-attempt failure was reproduced. [Diagnostic and ledger summary](../research/2026-10-22-feature-four-slot-diagnostic.md)
are separate from every research number below. Parent owns prompt/generator
fixes. The newer feature ACK timeout and keyboard-turn acceptance are likewise
independent evidence and are not pooled into this baseline study.

A later [six-slot feature diagnostic](../research/2026-10-22-feature-six-slot-diagnostic.md)
reproduced rejection of syllable-valid lines by the existing novelty gate.
Larger one-wave budgets did not repair those particular short-flow fixtures.
This is additional evidence for separating requested search from useful yield
in G3, not a new population-level failure estimate, and is not pooled with the
paired nine-slot results below. Original failed ledgers were unavailable; the
diagnostic clearly separates reconstruction from original observations.

## Findings and Reduced Model

Retain **five empirically informed groups plus two structural constraints**.
This is an exploratory organization, not a validated minimal model or causal
importance ranking. Observed source-render spans were much longer than
host-local writer and local-preparation spans. Larger budgets increased mean
valid count on these paired workloads at a host-local cost; the two route runs
do not isolate network causation. Edits can obsolete executing work, so ready
source audio is not the same as fresh on-time playback.

| Group | Compact quantity | Reason for merge/retention | Evidence |
|---|---|---|---|
| G1 Deadline budget | Tempo + stage lead + guard, in ms | Jointly define time available | 90-BPM traces; tempo/lead sweeps counterfactual |
| G2 Service/capacity | Stage costs + placement + effective slots | Request concurrency is not execution capacity | 24 real bars, shared renderer lock |
| G3 Useful search | Candidate/rescue policy + acceptance/yield | Count is a control; validity is an outcome | 12 paired workloads at 16/32/64 on two routes |
| G4 Revision pressure | Edit trace + invalidation/cancellation scope | Finished work can fail the latest-input obligation | 18 real bars; exact cancelled work unmeasured |
| G5 Delivery reserve | Transport + preparation + enqueue margin | Remote completion is not playback readiness | Two Mac receipts, 12 real preparations |
| G6 Granularity/causality | Stage units + input/predecessor watermarks | Determines legal schedules | Bar-level contract; finer outputs theoretical |
| G7 Eligible coverage | Ready intervals + cache dependencies + fallback | Continuity and freshness are separate outcomes | Structural model/invariants, not hit-rate evidence |

The [JSON registry](../research/2026-10-22-reduced-scheduling-registry.json)
records units, class, scope, implemented/configurable/fixed/model-only status,
current value/source, change boundary, dependencies and evidence references.
Values describe the measured baseline, not today's evolving product defaults.
The [exhaustive inventory](../superpowers/specs/2026-10-22-realtime-scheduling-model-design.md#4-parameter-coverage-catalogue)
is retained only as a coverage appendix.

### Quantitative Evidence

Real Qwen2.5-7B-Instruct, 12 matched topic/template/seed workloads per budget,
alternating ascending/descending order, no rescue, 32-token cap, temperature 1.0.
Six topics repeat twice across three existing nine-slot templates, empty history.

| Budget | H200 selectable | H200 mean valid | H200 median/max ms | Mac selectable | Mac median/max ms |
|---|---:|---:|---:|---:|---:|
| 16 | 10/12 | 4.17 | 91 / 368 | 10/12 | 1288 / 1605 |
| 32 | 10/12 | 6.50 | 95 / 115 | 10/12 | 1278 / 1635 |
| 64 | 12/12 | 11.17 | 156 / 165 | 11/12 | 1138 / 3296 |

Selectable means at least one selectable candidate after budget exhaustion,
not the desired minimum of three valid. That minimum was met in 10/12, 10/12,
and 11/12 H200 attempts. First n16 includes first-use CMU initialization
(277 ms evaluation versus 91 ms generation; initialization itself was not
separately timed). Excluding
the whole first triple yields n=11 medians 90.6/95.3/159.4 ms. Primary data retain
all attempts. API seeds do not guarantee identical outputs; server
`max_num_seqs=48` constrains a 64-choice request. This supports a budget/yield
trade-off, not optimality, quality superiority, or a claim that 64 is faster over
SSH. The deployed 32 + 16x3 rescue policy is not this one-wave experiment.

![Paired budget evidence](../research/2026-10-22-scheduling-evidence.png)

Each thin line links a workload/seed pair; points can overlap. Black diamonds
are medians, n=12 per budget, failures retained, no confidence bands or fitted
candidate-independence law. Validity is a structural gate, not listening quality.
The 12 topic/template/seed pairs are related workloads, not 12 independent
human-quality samples. Budget 32 always occupied the middle order position.
Mac and H200 runs occurred separately and differ in route/client conditions;
subtracting their medians would not identify network cost.

**Source/concurrency:** three paired groups, each two independent sessions x
two bars, sequential versus simultaneous: 12 bars per condition, 24/24 returned.
Same-session text causality remains sequential. This was deliberate research
contention before 05:38, with no known overlapping feature load.

| Median stage ms | Sequential | Two concurrent sessions |
|---|---:|---:|
| MOSS service span | 1782 | 1780 |
| MMS alignment | 23.6 | 23.6 |
| Render response total | 1806 | 3583 |
| Total minus MOSS/alignment | 2.9 | 1786 |

The renderer lock encloses synthesis/alignment; total timing begins outside
it while stage spans begin inside. The residual includes lock wait **and
uninstrumented overhead**, not a separately measured queue span. Four-bar
makespans were 8.15/7.42/7.47 s sequential versus 7.52/7.45/7.35 s concurrent. Three
pairs cannot establish throughput superiority, but in-flight requests clearly
cannot be treated as independent render slots.
Bars are clustered within sessions and paired groups: 12 bars are not 12
independent concurrency trials. The paired-group count is three.

**Edits:** three matched seeds for no edit, one at 250 ms, and edits at 250/500/750 ms,
six bars each, 18/18 returned. Median latest-topic bar 0 receipt was 1.917/3.799/3.826 s
from session request. Old bar 0 source could finish before replacement. Final
topics also differ, so this is **not an isolated causal estimate of edit-count
cost**. Exact cancelled GPU work, audible response and global equivalence of
one versus three edits were not measured.

**Delivery:** one Mac/H200 session received source bars at 5.786/7.794 s, including
connection, writer/render/wait and transfer, not pure network time. Twelve
real Mac `BarAudioPreparer` calls with Rubber Band 4.0.0 and drum mixing took
median 49.99 ms, max 59.17 ms. All outputs were finite/non-silent, exactly 128000 frames,
48 kHz stereo at 90 BPM. No device playback was performed by this worker.

### Deadline Counterfactual

Hold observed source-ready elapsed times fixed and test
`D_j=((L+j)*16-1)*60000/(4*BPM)` ms from session request: bar ordinal `j`, lead `L`
bars, one-tick guard. Original obligations survive edits; stale output/failure
cannot count as fresh success.

| 90-BPM source-ready misses | Lead 1 | Lead 2 | Lead 3 |
|---|---:|---:|---:|
| Sequential sessions | 0/12 | 0/12 | 0/12 |
| Concurrent sessions | 9/12 | 0/12 | 0/12 |
| No edits | 0/6 | 0/6 | 0/6 |
| One edit | 6/6 | 0/6 | 0/6 |
| Burst edits | 6/6 | 0/6 | 0/6 |
| Mac two-bar session | 2/2 | 1/2 | 0/2 |

[Analysis JSON](../research/2026-10-22-scheduling-evidence.json) keeps all 90 cells
(five tempos 60/90/120/150/180, three leads, six trace conditions), including
per-bar outcomes. These are **trace counterfactuals**, not live playback miss
rates or tempo reruns. They omit local preparation/enqueue/acoustics and assume
source cost does not change with tempo or lead. No fitted distributions,
synthetic performance samples, or delay-injection results are presented.
This is deadline-shift sensitivity, not resimulation of a scheduler with a
different lookahead: changed lead could also change admission, queueing and
revision work. Zero observed source misses cannot establish a tail guarantee.

## Structural Relationships

![Parameterized stage DAG and timeline](../research/2026-10-22-scheduling-stage-model.png)

[Editable vector SVG](../research/2026-10-22-scheduling-stage-model.svg).
This is an unscaled dependency illustration, not a measured execution trace.
Existing v1.1 diagrams are unchanged.

For job `j`, stage `v`, input availability `A`, predecessor set `pred(j,v)`, and
service cost `C`, use a common request-relative clock. `R,S,F,A,Q,D,P,M,C` are
in milliseconds. `T` below is in seconds, `B` in beats/min, `beta` in beats and
`fs` in samples/second; conversion to a request-relative deadline is explicit.

```text
R_jv = max(policy_release_jv, A_jv, max_{(i,u) in pred(j,v)} F_iu)
S_jv >= R_jv subject to actual resource capacity; F_jv = S_jv + C_jv
Q_jv = S_jv - R_jv
critical_path_lower_bound = max_dependent_path sum(C_jv)
slack_j(t) = D_j - t - estimated_remaining_critical_path_j(t)
D_j = P_j_ms - required_delivery_margin_j
lateness_j = F_j - D_j; fresh_success requires lateness <= 0 AND eligibility
T(beta) = t0 + integral_0^beta 60/B(u) du; sample(beta) = round(fs*(T(beta)-t0))
P_j_ms = 1000*(T(beta_j) - request_time_seconds)
dependency_compatible(a,current) = all actual producer dependencies are present and equal
reuse_eligible = ready AND integrity_valid AND epoch/policy_eligible AND dependency_compatible
eligible_ready_coverage = union(interval(a) for each reuse_eligible artifact a)
```

The service sum includes transfer when that stage is modeled; queueing and
unmodeled downstream work make it only a lower bound. Independently measured
stage percentiles cannot be added to obtain an end-to-end percentile. The
remaining-path estimator is a model placeholder, not a fitted/validated predictor.
`F_j` is completion of the stage under the stated obligation; the source-ready
counterfactual does not substitute for prepared, committed or audible completion.
`beta` is beats, not bars; varying tempo requires the integral.
Input watermarks prohibit future-input use. Plan edit sequence, per-bar content
revision and timeline revision are distinct. Corrected collection checks latest
lyric identity, final topic and edit acknowledgment separately. Source cache
keys must include actual synthesis conditioning, including tempo if used;
prepared keys additionally depend on flow/timeline/format/mix. These are
structural requirements, not a fitted law or a complete implemented cache.
Both keys must also identify the producer implementation/configuration. Key
equality is necessary, not sufficient: integrity, session/epoch ownership,
fidelity/expiry policy and readiness still apply. The tested `cache_compatible`
helper checks dependency equality only.

The job identity is `(session, epoch, span, content_revision, timeline_revision,
configuration_revision, stage, attempt)`. An explicit cross-bar edge runs from
selected text for bar `j` to the writer for bar `j+1`; source rendering may
overlap the later writer. Partial streaming, suspension/resume and cross-session
fairness are not exercised by the two-stage FIFO invariant helper. Its invariant
tests do not validate a complete production scheduler or an acoustic timeline.

The registry preserves the 18-part granularity vector: observation, aggregation,
release/phase, submission, replanning, context, horizon, selection, text commit,
TTS, alignment, warp/mix, transfer, cache, commit/phase, playback, cancellation,
and measurement. Measured generation/selection/source/preparation are bar-level.
A 16-tick template and sample placement do not imply beat/tick semantic output.

### Relation to the Prior Paper

Section IV of the [StreamMUSE paper](https://arxiv.org/html/2606.11886v1)
connects inference interval `I`, generated span `GL`, tick duration and
round-trip latency, including an empirical latency model and percentile-based
configuration constraints. We retain its separation of musical time from
physical time and of fresh service from continuity. Here, request/release cadence
belongs to G6, horizon/lead to G1/G6, and compute/communication to G2/G5.
Candidate count is not `GL`, and input availability `A` is not its interval `I`.

The rap pipeline adds selected-text dependencies, synthesis/preparation stages,
revision eligibility and stage-specific resource contention. Its bar-level
contracts do not inherit tick-level responsiveness or the paper's feasibility
inequalities. No quadratic inference law, Gaussian perturbation or Pareto
network model is imported. These short traces cannot identify a reliable tail
distribution or guarantee a feasible operating region; the contribution here
is measured conditional evidence and a structural model, not a fitted predictor.

### Outcome Vector, Separate from Controls

Record `y_j = (continuity, freshness, content_validity, quality, response_gap)`
for a stated output obligation `j`. These are descriptive outcomes, not controls,
expected values, predictions or a weighted objective. G1-G7 organize conditions
and mechanisms; they do not imply a fitted map to `y_j`.

| Component | Operational meaning | Evidence here |
|---|---|---|
| Continuity | Intended interval played without an unintended gap; permitted fallback can count | Unmeasured at the sink; non-silent source WAVs are not proof |
| Freshness | On-time output satisfies the obligation's declared input/revision cutoff and eligibility | Latest-source checks and source-deadline counterfactuals only; audible freshness unmeasured |
| Content validity | Writer output passes the declared structural/selectability gates | Candidate ledgers and selection outcomes; not perceptual quality |
| Quality | Listener-assessed musical/linguistic merit under a declared rubric | Unmeasured; no invented rating or score substitution |
| Response gap | Milliseconds from accepted affected input/edit to first audible eligible response reflecting it | Unmeasured; request-to-source receipt is a different boundary |

Missing observations remain null, not zero or failure. A future response-gap
study must retain timeout/nonresponse cases with their observation windows;
successful-only latencies would conceal failures. Continuity can hold while
freshness fails, and structural validity does not establish quality. The registry
stores this vector separately from the seven groups and their 21 parameters.

## Measured and Unmeasured

| Quantity | Evidence | Not established |
|---|---|---|
| Candidate work | 72 real attempts; prompts/text/ledgers/seeds | Broad workloads, human quality, candidate independence |
| Remote source | 24 concurrency + 18 corrected edit + 2 Mac bars; 44 WAV audits | GPU kernel durations, exact cancellation waste |
| Preparation | 12 real warp/mix calls; exact finite output checks | Device latency, underruns, acoustic onsets |
| Deadline/lead | 90 deterministic trace-counterfactual cells | Tempo-dependent synthesis, actual playback misses |
| Replay helper | FIFO writer/render, capacities 1/2/3, invariant tests | Production scheduler fit or broad simulation superiority |
| Cache/revision | Dependency and boundary invariants | Cache-hit/eviction/fallback coverage trace |

Durations are same-host monotonic spans or server-reported local spans;
cross-host timestamps are never subtracted. GPU utilization is not used as an
isolated performance metric. Shared CPU/host/network effects remain possible.
Other voices/models, dense flows, long contexts and finer output contracts
remain unmeasured, not controls demonstrated to be unimportant.

## Reproduction and Verification

Owned: experiment script, research tests, this report, dated research registry,
analysis JSON, figure/diagram assets and ignored raw artifacts. No product
source, commits, branches or other workers' edits were changed.

From `real_rap` with its environment and `PYTHONPATH=src`:

```bash
.venv/bin/python -m pytest tests/unit/experiments/test_rap_scheduling_evidence.py -q
.venv/bin/python scripts/experiments/rap_scheduling_evidence.py candidates --output NEW_RUN_DIR --pairs 12 --budgets 16 32 64 --rescue-waves 0
.venv/bin/python scripts/experiments/rap_scheduling_evidence.py analyze --raw-root logs/scheduling_evidence_20261022 --output-prefix docs/research/2026-10-22-scheduling-evidence
```

At 06:07, 51 focused research tests pass within 235 experiment tests; 88 related
writer/generator/flow-prompt/client tests also pass (323 combined). Two client
tests require permission to bind their controlled localhost HTTP server; the
permission-enabled rerun passes. These are not GPU or feature live tests.
Coverage includes failure denominators/pair identity,
missing observations, deadline equality, stale eligibility, revision identity,
input release watermarks, invalid timestamps, serial writer causality, work
conservation and render capacity over 100 deterministic jobs at 1/2/3 slots. These
synthetic integer test cases validate invariants, not performance distributions.
Audio and exact process-ownership checks are covered. Full feature acceptance
belongs to the coordinator and is not inferred from these tests.

Both Python files compile. All six primary input hashes, 44 package ZIP CRCs,
44 non-silent source WAVs, 12 finite/non-silent preparation records, and SVG
syntax were checked. Both figures were visually inspected. The repository
ignores `*.png`; the parent must explicitly include the two dated PNGs when
packaging/committing artifacts. No ignore rule was changed here.

Independent review against the approved research scope and prior paper checked
units, dependency directions, inference strength and figure presentation.
It tightened workload clustering/order caveats, source-only deadline sensitivity,
cache eligibility and the distinction between structural bounds and fitted laws.
The 51 focused tests were rerun successfully; no new model measurements or tests
were added for this review. Markdown-to-HTML conversion resolved all nine
link/image targets; both PNGs decoded and were visually inspected, and SVG XML
parsed. A separate 06:17 review independently recomputed the primary summaries
and all 90 deadline cells with no actionable findings in that snapshot; its
audit and snapshot hashes are in
`.superpowers/sdd/2026-10-22-rap-interactions-and-evidence/research-review.md`.
The later cleanup/status and explicit coverage-eligibility wording do not
change those measurements.
At 06:23, the same reviewer checked the added five-component outcome vector
and registry entries with no dimensional or inference finding. This addition
defines descriptive outcomes only; it does not add measurements or parameters
to the seven-group model.

Raw root `logs/scheduling_evidence_20261022/`; canonical remote snapshot
`h200-evidence.tar.gz`, extracted into `h200-snapshot/`, SHA-256:
`1852d47991401ee97a7e87299916fa6bcec1f25e59eb03a2b1fc1fd0151701c8`.
Contains measured packages/audio, prompts/ledgers, config/source hashes,
stage events and isolated launch logs/records. Remote originals remain in
isolated-root `evidence/` and `smoke/`. Mac runs are `budget-16-32-64-mac/`,
`mac-v3-two-bars/`, and `mac-preparation-measured/`. Partial `h200/` is not canonical.
Mac runs and excluded Mac harness attempts are also preserved in
`mac-research-evidence.tar.gz`, SHA-256
`2eb3032deb0d8e473c333615729f5aba8e804fa9008df4c23b7cae3f55d8a47b`.
The feature diagnostic is a separate folder, not included in that archive.
Run manifests preserve code hashes; the current reproducer incorporates the
documented harness repairs, so it is not byte-identical to every earlier run.

Preserved exclusions: `budget-no-rescue/` failed frozen-mapping serialization;
`stages-edit-churn/` wrongly censored a latest-topic bar by equating unrelated
revision counters; `mac-preparation/` failed a Tempo constructor before any
measurement. These are harness failures, not model failures. Older
`http_after_ready_ms` included decode/archive overhead; no finding uses it as
isolated network time. Feature diagnostics have their own folder and are never
included by the analysis loader.

### Interfaces and Owned Files

No product API changes. `scripts/experiments/rap_scheduling_evidence.py` exposes:

| CLI | Interface and output |
|---|---|
| `candidates` | `--output --llm-url --pairs --budgets --rescue-waves --seed`; run config, raw waves/ledgers, JSONL and summary |
| `stream` | `--output --render-url --bars --bpm --topic --template`; v3 start/updates/stop and source HTTP fetch, never playback/commit |
| `stages` | `--study` selects `concurrency` or `edits`; `--output --pairs --bars --bpm`; paired events/packages/bar/group summaries |
| `prepare` | `--input --output --limit`; real local preparation records and PCM WAVs |
| `analyze` | `--raw-root --output-prefix`; measured summary, fixed-trace counterfactuals, paired PNG, structural SVG/PNG |
| `launch` / `stop` | Service `vllm`, `vllm-live`, or `render`; `--root`, launch also `--old-deployment`; isolated process records/logs |

Do not rerun `stream` or `stages` against GPU1/GPU2 while feature acceptance
owns those resources. Analysis and preparation use local saved files.

Changed/created owned files, relative to `real_rap`:

- `scripts/experiments/rap_scheduling_evidence.py`
- `tests/unit/experiments/test_rap_scheduling_evidence.py`
- `docs/reports/2026-10-22-scheduling-research.md`
- `docs/research/2026-10-22-reduced-scheduling-registry.json`
- `docs/research/2026-10-22-scheduling-evidence.json` and `.png`
- `docs/research/2026-10-22-scheduling-stage-model.svg` and `.png`
- `docs/research/2026-10-22-feature-four-slot-diagnostic.md`
- Raw records, archives and standalone feature diagnostic under
  `logs/scheduling_evidence_20261022/` (ignored).

## Service Handoff

Research stage window is DONE. No research requests on GPU1/GPU2 after 05:38.
Services are handed to the parent; subsequent product sync/restarts are its
responsibility. Remote research source was baseline `bde2f62c`; sync new product
source before feature acceptance to:

`/data/home/Andrew.Yang/StreamMUSE/deploy/rap_research_20261022/repo/src/`

Isolated root `R=/data/home/Andrew.Yang/StreamMUSE/deploy/rap_research_20261022`.
Preserve other files, old-deployment `deps`, and model environments.

| Resource | Service/endpoint | Research-start identity |
|---|---|---|
| GPU0 | Research Qwen, STOPPED 06:16:47 PDT | PID 3908763 exited; record archived |
| GPU1 | Live Qwen, remote 18611/v1 | PID 3911563, `vllm-live.pid.json` |
| GPU2 | MOSS/MMS renderer, remote/local 18620 | PID 3912294 at handoff, parent may replace |

LLM env `/data/home/Andrew.Yang/qwen-extract-venv`; model
`Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`,
served name `qwen-rap`, vLLM 0.31.0, max model length 2048, max sequences 48,
memory utilization 0.25. MOSS env
`/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/moss`; model
`OpenMOSS-Team/MOSS-TTS-v1.5`, revision `cdd3b911b1585e3f2dbc7775ef10f9926f58850a`.
In-process synthesis, no SGLang continuation. Voice:
`/data/home/Andrew.Yang/StreamMUSE/audio-protocol-downloads/MOSS-TTS/assets/audio/reference_02_s1.wav`.
Renderer uses GPU1/18611, 32 lyric candidates, PCM packages. Complete argv/env
and exact process identity are in isolated `*.pid.json` records.

Safe renderer restart after source sync (on H200 only):

```bash
R=/data/home/Andrew.Yang/StreamMUSE/deploy/rap_research_20261022
python3 "$R/evidence.py" stop render --root "$R"
python3 "$R/evidence.py" launch render --root "$R" \
  --old-deployment /data/home/Andrew.Yang/StreamMUSE/deploy/rap_v3_gui_efd621ed_gpu1_20261014
curl -fsS http://127.0.0.1:18620/health
```

Wait for `ready:true`; MOSS warmup took about 6 s after load. Stop verifies exact
PID/start time/UID/process group, sends TERM only and archives its own record.
Launch refuses existing pidfile, occupied port or nonempty GPU. Logs append.
Do not invoke old deployment start/stop or `model_upgrade_stack.sh`; do not
restart either LLM unnecessarily.

Existing Mac tunnel control socket `/tmp/rap-research-20261022-ssh.sock` forwards
local 18601 and 18620 to the same remote ports. Parent uses local 18620; do not
open duplicates. The unused 18601 forward remains, but its research service is
stopped. Only if absent, recreate with:
`ssh -N -L 18601:127.0.0.1:18601 -L 18620:127.0.0.1:18620 H200`.
Live LLM reproduction, already launched, do not duplicate:
`python3 "$R/evidence.py" launch vllm-live --root "$R" --old-deployment /data/home/Andrew.Yang/StreamMUSE/deploy/rap_v3_gui_efd621ed_gpu1_20261014`.

### Operational Chronology

- 05:18: rechecked empty GPUs/free ports immediately before isolated launches.
- 05:21: GPU0 Qwen ready. Earlier renderer readiness probe had failed before
  MOSS load; failed record/log preserved.
- 05:23: real baseline v3 smoke: lyric 0.625 s, aligned source receipt 2.460 s,
  32 requested/four valid, non-silent mono 24k source, 65280 frames, 132208-byte
  package, 1835.6-ms render. Source/transport only, not device playback.
- 05:26-05:28: separate same-model GPU1 Qwen launched; only isolated GPU2
  renderer safely replaced to use it. Old/shared pidfiles untouched.
- 05:31-05:38: controlled concurrency and corrected churn completed; GPU1/GPU2
  released with exact sync/restart instructions.
- 05:46-05:57: consolidated raw evidence, visually checked paired figure,
  added registry and causality checks, and performed separate GPU0 feature
  diagnostic. No integration contention introduced.
- 06:08: structural diagram, typed registry metadata and raw integrity checks
  completed; final research handoff ready. Services/tunnel remain available;
  no product code, branch, commit or shared process was changed by this worker.
- 06:09-06:18: independent rigor review tightened inference/units/cache wording
  and the prior-paper comparison; plots regenerated only from preserved data.
  The old v1.1 diagram is unchanged. No further model measurements were run.
- 06:16:47: authorized GPU0-only cleanup used
  `python3 "$R/evidence.py" stop vllm --root "$R"`. Exact recorded
  PID/start-time/UID/process-group identity matched; TERM stopped PID 3908763.
  Record preserved as `vllm.stopped-20261022T131647Z.pid.json` under `R`.
  Post-stop checks: process absent, port 18601 refused connection, GPU0 0 MiB.
  GPU1 `/v1/models` still serves `qwen-rap`; GPU2 `/health` is ready; local
  18620 `/health` is ready through tunnel master PID 53223. GPU1/GPU2, tunnel,
  model cache, raw evidence and logs were not stopped or deleted. Cleanup
  observations are recorded in `logs/scheduling_evidence_20261022/cleanup.json`.
