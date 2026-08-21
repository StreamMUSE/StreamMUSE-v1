# SDD ledger — plan: docs/superpowers/plans/2026-08-21-realtime-remote-moss-renderer.md

## Execution Context

- Spec: `docs/superpowers/specs/2026-08-21-realtime-remote-moss-renderer-design.md`
- Branch: `feature/real_rap_audio`
- Worktree: `.worktrees/real_rap`
- Plan commit: `be5420e0`
- Design commit: `8ac8bc56`
- Baseline suite: `1405 passed, 4 skipped, 1 pre-existing pretty_midi warning in 41.25s`.

## Pre-Flight Interface Scan

| Producer | Consumer | Shared file/interface | Finding or ruling |
|---|---|---|---|
| Task 1 | Task 2 | `RemoteRapChunkRequest`, `RemoteSelectedBar`, manifest values | Clean: Task 2 consumes the exact Task 1 names. |
| Task 1 | Task 4 | `encode_chunk_package`, remote manifest | Clean: Task 4 packages only the two specified members. |
| Task 1 | Task 5 | `decode_chunk_package`, `PreparedRapChunk` | Clean: Task 5 validates before converting to playback bars. |
| Task 1 | Task 6 | `PreparedRapChunk` | Clean: Task 6 consumes complete immutable chunks. |
| Task 2 | Task 3 | `PhraseVocalRenderer`, `PhraseRenderResult` | Ruling: define the protocol/result in Task 2's application module; Task 3 implements/imports those types rather than redefining them. Cost if wrong: one import-boundary refactor. |
| Task 2 | Task 4 | `RapChunkOrchestrator`, `RemoteChunkRenderArtifact` | Clean: server remains a thin transport adapter. |
| Task 2 | Task 8 | candidate profile and resolved budget diagnostics | Clean: tuning changes configuration, not candidate validity semantics. |
| Task 3 | Task 4 | persistent real renderer | Clean: worker is injected into the server composition root. |
| Task 3 | Task 8 | real MOSS/MFA/R3 environment | Clean: Task 3 performs an early worker smoke; Task 8 validates the integrated service. |
| Task 4 | Task 5 | HTTP request and ZIP response | Clean: one render request and one binary response per chunk. |
| Task 4 | Task 7 | bounded manifest diagnostics | Clean: full ledgers remain server-side; UI receives bounded summaries. |
| Task 4 | Task 8 | loopback server CLI and artifact cache | Clean: H200 deployment uses the production entry point. |
| Task 5 | Task 6 | `RapChunkPreparationStrategy` lifecycle | Clean: Task 6 owns scheduling; Task 5 owns transport and audio conversion. |
| Task 5 | Task 8 | transfer and Mac conversion timings | Clean: integration reports both without changing timing authority. |
| Task 6 | Task 7 | new `RapEventType` values and payloads | Clean: Task 6 emits; Task 7 projects and records. |
| Task 6 | Task 8 | CLI and rolling controller | Ruling: remote mode rejects odd nonzero `--max-bars` in the first implementation because every request and result must contain exactly two bars. Cost if wrong: users must round finite demos to an even bar count. |
| Task 7 | Task 8 | acceptance report and commands | Clean: Task 8 fills measured evidence into the Task 7 document. |

## Per-Task Consistency Scan

| Task | Internal consistency | Finding or ruling |
|---|---|---|
| Task 1 | Tests match two-bar/domain/ZIP implementation | Ruling: `request_id` excludes remaining budget, but every retry using that key must resend the original canonical body including its original budget. A changed budget is a new request. Cost if wrong: retries could receive HTTP 409. |
| Task 2 | Tests match wave planning, ranking, and orchestration boundaries | Clean. Independent-choice support is conditional on the current client's proven limitation. |
| Task 3 | Tests match persistent runtime and plain onset-R3 output | Ruling: inspect all existing unstaged changes in both backend scripts and preserve them; extraction may delegate but cannot overwrite current experiment work. Cost if wrong: offline experiment regression. |
| Task 4 | Tests match FastAPI, cache, package, and CLI behavior | Clean. `/health` is not counted as a per-chunk request loop. |
| Task 5 | Tests match persistent HTTP client and exact local mix | Clean. `abort()` is reusable; `close()` is final. |
| Task 6 | Tests match chunk controller lifecycle and mode switch | Ruling: eSpeak mode remains on `RollingRapController`; remote mode alone uses `RollingRapChunkController`. Cost if wrong: duplicated controller maintenance, accepted to isolate regression risk. |
| Task 7 | Tests match bounded diagnostics and no-new-controls requirement | Clean. HTML/CSS changes occur only if existing containers cannot represent the new fields. |
| Task 8 | Commands and evidence match real H200 acceptance | Clean. P0 ends only after a real MOSS/MFA/R3 chunk and a continuous 90 BPM WAV run. |

## Rulings

- Ruling: Keep two controllers behind one lifecycle protocol — this avoids destabilizing the proven eSpeak controller — cost if wrong: shared lifecycle fixes may need duplication.
- Ruling: Use server-side adaptive profiles but return every resolved parameter — this enables H200 tuning without losing research reproducibility — cost if wrong: manifests become slightly larger.
- Ruling: P0 monitoring is event/log visibility; website polish is P1 — this protects the audible real path under the execution budget — cost if wrong: the first web view may show generic rather than specialized chunk presentation.
- Ruling: Do not deploy by pulling into the existing H200 `real_rap` checkout because it contains extensive prior uncommitted experiment work. Use a separate request-specific deployment directory assembled from the Mac worktree. Cost if wrong: a small additional disk footprint, but no risk to the user's H200 state.
- Task 1 review ruling: preserve the specified idempotency identity, which intentionally excludes `remaining_budget_ms`. Add an immutable serialized transport-attempt value so retries can only reuse the original canonical body; Task 4 must reject the same ID with different bytes and Task 5 must retry the retained attempt rather than rebuild a request. Cost if wrong: a reconstructed request could collide with a cached ID and receive HTTP 409.
- Task 1 review ruling: make the manifest diagnostic envelope structurally mandatory. It must carry the accepted request budget, resolved policy, candidate counts/summaries, stage timings, alignment diagnostics, audio diagnostics, tool/model versions, and warnings, while selected bars retain explicit total score plus component-score diagnostics. Cost if wrong: monitoring and experiment reports would have to guess whether missing data means zero, unavailable, or an implementation bug.
- Task 1 review ruling: reject truncated or malformed WAV members as `ValueError` and test compressed ZIP expansion limits, because the Mac fallback boundary must handle every corrupt remote package deterministically. Cost if wrong: malformed remote data could escape the fallback path during playback preparation.

## H200 Reconnaissance

- Host reachable as `Andrew.Yang@masdar`.
- Free at inspection: physical GPUs 0, 2, and 3 (0 MiB, 0%); GPU 1 hosts vLLM, GPUs 4-7 are occupied.
- Existing reusable vLLM: `127.0.0.1:8010`, model `Qwen/Qwen3.6-27B`, GPU 1.
- Ports 8001 and 8020 were free at inspection.
- MOSS Python: `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/moss/bin/python`.
- MFA: `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/align/bin/mfa`.
- Rubber Band root from setup: `/data/home/Andrew.Yang/StreamMUSE/assets/rap-audio-protocols/rubberband/rootfs/usr/bin/rubberband` with its documented library path.
- Verified versions: MOSS environment torch `2.9.1+cu128` with CUDA available, MFA `3.4.1`, Rubber Band `3.3.0`.
- Verified reference voice: `/data/home/Andrew.Yang/StreamMUSE/audio-protocol-downloads/TED-TTS/datasets/Ref/0011_000001.wav`.
- Existing H200 checkout is dirty and remains untouched during deployment.

## Task Status

- Task 1: complete (initial contracts plus review-fix follow-up)
- Task 2: pending
- Task 3: pending
- Task 4: pending
- Task 5: pending
- Task 6: pending
- Task 7: pending
- Task 8: pending
