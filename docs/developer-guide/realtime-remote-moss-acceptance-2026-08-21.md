# Real-Time Remote MOSS Acceptance Record

## Status And Evidence Boundary

Task 7 implements and locally verifies the bounded monitoring, terminal, Mac
website, and operations surfaces. It does not claim a real H200 run. Task 8
must replace every `TODO (Task 8)` entry below with retained evidence from the
designated H200 and Mac environments. Synthetic fixtures and local fake-service
results are not performance measurements.

- Accepted Task 6 base: `0af9f57c8e3046326b6cd7aa47e65310fcd3d60c`.
- Task 7 revision: the commit containing this file; record `git rev-parse HEAD`
  before Task 8 deployment.
- Runtime artifact root: `logs/rap/remote_moss_acceptance_20260821/`.
- Exact startup, health, Mac control, artifact, and shutdown commands:
  `docs/developer-guide/rap-demo-quickstart.md`.

## Task 7 Local Acceptance

| Check | Result |
| --- | --- |
| Chunk events and projector state are bounded | PASS: two lines, two flows, bounded maps/lists, and fixed timing keys |
| Event and browser state exclude raw WAV, full candidate ledgers, and character spans | PASS |
| Terminal reports commitment, lifecycle, lines, schedules, counts, scores, prompt/context, timings, slack, alignment, warnings, hashes, artifacts, and failure | PASS |
| Website reports the same evidence without adding runtime controls | PASS |
| Existing dense desktop layout remains two columns | PASS: browser inspection at 1440 x 900 |
| Responsive audit panel has no horizontal overflow | PASS: browser inspection at 390 x 844 |
| Existing eSpeak snapshots and regressions remain unchanged | PASS |
| Normal Mac runtime forwards only H200 port 8020; website remains Mac-local | PASS: documented |
| Direct vLLM forwarding is optional diagnostics only | PASS: documented |

Local verification commands and exact counts are retained in
`.superpowers/sdd/2026-08-21-realtime-remote-moss-renderer/task-7-report.md`.

## Task 8 Deployment Record

| Item | Evidence |
| --- | --- |
| Mac revision | TODO (Task 8) |
| H200 revision | TODO (Task 8) |
| vLLM physical GPU / UUID / logical mapping | TODO (Task 8) |
| MOSS and MMS physical GPU / UUID / logical mapping | TODO (Task 8) |
| vLLM PID and retained log | TODO (Task 8) |
| Chunk-service PID and retained log | TODO (Task 8) |
| `/v1/models` response artifact | TODO (Task 8) |
| `/health` response artifact and schema revision | TODO (Task 8) |
| Cold two-bar request ID and artifact directory | TODO (Task 8) |
| Warm two-bar request ID and artifact directory | TODO (Task 8) |
| Exact 24 kHz mono PCM16 frame validation | TODO (Task 8) |
| MFA reference comparison | TODO (Task 8) |

Do not replace GPU entries until `nvidia-smi` is captured immediately before
service launch. Do not terminate or repurpose unrelated H200 processes.

## Warm Latency Distribution

Task 8 must run at least ten warm requests across the three existing flow
templates. Record milliseconds from retained manifests and Mac logs.

| Stage | Samples | p50 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Candidate generation | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| Candidate evaluation | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| MOSS synthesis | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| MMS alignment | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| Rubber Band R3 | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| Package creation | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| SSH transfer | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| Mac validation and mix | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |
| End to end | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) | TODO (Task 8) |

## Candidate And Alignment Results

| Measure | Result |
| --- | --- |
| Requested candidates | TODO (Task 8) |
| Parseable candidates | TODO (Task 8) |
| Valid candidates | TODO (Task 8) |
| Selectable candidates | TODO (Task 8) |
| Failure counts by stage and reason | TODO (Task 8) |
| Alignment method distribution | TODO (Task 8) |
| Alignment confidence distribution | TODO (Task 8) |
| Alignment fallback counts by method | TODO (Task 8) |
| Stretch-ratio warning counts | TODO (Task 8) |

## Continuous Mac Run

Run 20 bars at 90 BPM through the normal port-8020 tunnel. Retain the mixed
WAV, event stream, session manifest, per-chunk manifests, and service logs.

| Measure | Result |
| --- | --- |
| Session ID and artifact paths | TODO (Task 8) |
| Expected frames and duration | TODO (Task 8) |
| Observed frames and duration | TODO (Task 8) |
| Application-level gaps | TODO (Task 8) |
| Remote MOSS committed chunks / bars | TODO (Task 8) |
| Local eSpeak fallback chunks / bars | TODO (Task 8) |
| Primary MOSS acceptance rate | TODO (Task 8) |
| Fallback rate and reason distribution | TODO (Task 8) |
| Deadline slack p50 / p95 / min | TODO (Task 8) |
| Playback underruns | TODO (Task 8) |
| Hash verification failures | TODO (Task 8) |

## Listening Comparison

Render at least three identical accepted lyric pairs through remote MOSS and
local eSpeak. Retain both outputs and document human listening observations;
alignment confidence is not a substitute for perceptual evaluation.

| Item | Result |
| --- | --- |
| Compared request IDs and artifact pairs | TODO (Task 8) |
| Review conditions and listener | TODO (Task 8) |
| Word coherence and intelligibility | TODO (Task 8) |
| Rhythmic placement and bar transitions | TODO (Task 8) |
| Recommendation | TODO (Task 8) |

## Known Limitations And Next Decision

- Real H200 availability, GPU isolation, persistent-worker warm behavior,
  transfer cost, Mac mix cost, and perceptual quality are unmeasured in Task 7.
- Task 8 must tune only from retained evidence and must not relabel service
  fallback output as primary MOSS acceptance.
- Final acceptance remains pending until the real request, warm distribution,
  exact-duration 20-bar run, zero-gap check, and listening comparison are
  complete.
