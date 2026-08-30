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
- Task 1 re-review ruling: accept all five findings. This is an untrusted HTTP boundary, so JSON booleans must not pass as numeric fields, successful selected bars must be structurally non-empty, response summaries must be finite JSON-safe records rather than arbitrary mappings, and malformed archive/parser failures must normalize to `ValueError`. Cost if wrong: bad remote data would escape local fallback or poison research diagnostics.
- Task 1 re-review ruling: add only basic cross-field invariants at the wire boundary: equal non-empty source/target anchor arrays, positive finite warp ratios, audio duration equal to frames/rate within one sample, normalized peak in `[0, 1]`, required non-empty `moss`/`aligner`/`rubberband` versions, and `total` timing no smaller than any component stage. Task 5 still owns comparison with the original requested flow and stricter renderability thresholds. Cost if wrong: Task 1 could either accept internally contradictory success manifests or overconstrain later alignment experiments.

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
- Early persistent-runtime evidence on physical GPU 0: the existing MOSS backend rendered two two-bar requests successfully in one process in 26.33 s total at 24 kHz, with about 6.0 GiB peak host RSS. The first process includes model load; model generation progress was approximately 2.75 s for chunk 0 and 2.54 s for chunk 1, confirming that a resident MOSS worker is viable whereas per-request process startup is not.
- H200 environment root cause recorded: TorchCodec requires `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/ffmpeg7/lib`, the Torch library directory, and every packaged NVIDIA `lib` directory. The failed probe used a nonexistent `assets/.../ffmpeg7` path; an isolated `import torchcodec` succeeded after correcting the path. Evidence: `/data/home/Andrew.Yang/StreamMUSE/experiments/realtime_remote_moss_warm2b_20260821`.
- H200 candidate probe against the warmed `Qwen/Qwen3.6-27B` vLLM on GPU 1: API-level `n=64`, 32-token independent choices, temperature 1.0, and top-p 0.95 completed in 1.166 s. It returned 64 non-empty choices, 63 unique lines, and 15 exact nine-syllable lines under `CmuProsodyAnalyzer`. The first preceding `n=4` probe paid a roughly 30 s cold/idle penalty; its immediate repeat took 0.208 s. Ruling: both vLLM and MOSS require explicit warmup/readiness before a session.
- H200 MFA blocker: MFA 3.4.1 took 49.63 s wall time to align a warmed two-utterance corpus (two 5.44 s MOSS phrases), even with downloaded models and local files. This cannot fit a 5.333 s two-bar lookahead. The existing dirty H200 checkout also exposes an older `render_aligned_chunk` signature without continuous-onset R3, confirming that integrated tests must use the isolated deployment. A fast-alignment investigation is now required before Task 3 can claim realtime viability.
- Isolated code validation: synced the current package and Task 1 tests to `/data/home/Andrew.Yang/StreamMUSE/deploy/real_rap_audio_task1_eaeb1b06` without touching the dirty H200 checkout. Using the existing `streammuse-isochron` Python runtime, the remote contract/package suite passed `89 passed in 0.96s`.
- H200 fast-alignment result on physical GPU 0: torchaudio 2.9.1 `MMS_FA` loaded successfully (1.18 GiB cached model; 6.97 s cold load including download/model setup) and aligned both real 5.44 s MOSS phrases. The first CUDA pass took 0.392 s for two files; the second resident pass took 0.0371 s total, with individual align calls of 0.0174 s and 0.0174 s. All 14 transcript words in each phrase received character/word spans. Evidence script: `.superpowers/sdd/2026-08-21-realtime-remote-moss-renderer/mms-fa-probe.py` and remote inputs under `realtime_remote_moss_warm2b_20260821`.
- Architecture amendment from measured evidence: use resident torchaudio MMS forced alignment as the realtime primary aligner, retain MFA 3.4.1 as the offline/reference implementation, and feed either aligner's source syllable anchors into the unchanged non-stress continuous-onset Rubber Band R3 warp. Derive MMS syllable onsets from character spans plus the existing CMU word/syllable analysis, with confidence warnings and a transcript-proportional fallback when a word cannot be aligned. Cost if wrong: MMS character-to-syllable anchor quality may be below MFA; preserve aligner identity/confidence diagnostics and offline A/B artifacts so this remains measurable and replaceable.
- Task 1 final-review ruling: require contiguous schedule slot indices, bound recursive diagnostic depth and normalize recursion errors over the complete decode/retained-body paths, and reject any BPM/sample-rate pair whose exact two-bar frame count is nonpositive, non-finite, or too large for the transport's bounded PCM payload. These are untrusted-boundary invariants, not Task 5 flow matching. Cost if wrong: malformed requests could still trigger ambiguous schedules, uncaught exceptions, or impossible allocations.
- Contract amendment for modular alignment: require version keys `moss`, `aligner`, and `rubberband`, not a hard-coded `mfa` key. An MFA implementation reports `aligner=mfa-3.4.1`; realtime reports the torchaudio/MMS versions. Cost if wrong: monitoring would falsely imply MFA ran when the resident CTC path actually produced the anchors.
- H200 R3 evidence from the isolated deployment: current non-stress `continuous_onset_r3` rendered the two MFA-reference chunks successfully in 0.180 s total, each exactly 128,000 frames at 24 kHz. The Rubber Band runtime requires `align/lib` in `LD_LIBRARY_PATH` in addition to the bundled Rubber Band and FFmpeg paths; omitting it reproduced a missing `libFLAC.so.12` failure. Observed unconstrained local stretch ratios remain wide and must be exposed as warnings rather than hidden.
- Warm H200 candidate wave spot checks on `Qwen/Qwen3.6-27B`: `n=16` took 0.395 s and produced 16 unique / 4 exact-nine-syllable lines; `n=32` took 0.735 s and produced 32 unique / 3 exact-nine-syllable lines; the prior `n=64` took 1.166 s and produced 63 unique / 15 exact-nine-syllable lines. These are single-context feasibility samples, not confidence estimates. Initial tuning hypothesis: start near 16 choices per bar with small rescues under a hard roughly 2.3 s candidate cutoff, then validate across templates and contexts in Task 8.
- Independent fast-alignment investigation: resident torchaudio `MMS_FA` averaged 19.58 ms per real phrase (22.12 ms maximum over 12 runs); resident direct Kalpy averaged 20.69 ms; the existing `mfa align --clean` path remained 49.63 s. On one retained phrase, MMS word starts differed from the accepted MFA reference by 49.46 ms mean absolute and 102.44 ms maximum. Full evidence and reproducible drivers are in `fast-alignment-investigation.md` and `/data/home/Andrew.Yang/StreamMUSE/experiments/realtime_fast_alignment_20260821/`.
- Task 3 mapping ruling: implement deterministic orthographic/CMU mapping with phoneme-weighted and word-proportional fallbacks for the first working prototype, emitting visible per-anchor confidence/method warnings. Do not add the investigation's proposed acoustic-onset dynamic program to P0 and do not hard-reject solely because a lower-confidence complete map used fallback; the user explicitly prioritized timing and requested best-effort pronunciation with warnings. Impossible, incomplete, out-of-bounds, or non-monotonic maps still fail the remote phrase and activate the prepared local fallback. Cost if wrong: some accepted words may have less accurate internal-syllable onsets; retained artifacts permit direct comparison and a later acoustic-DP or resident-Kalpy upgrade.
- Coordinator H200 Task 2 reproduction: synced the current source/tests to `/data/home/Andrew.Yang/StreamMUSE/deploy/real_rap_audio_task2_a1c14f15` and forced that directory first via `PYTHONPATH`. The focused suite passed `86 passed in 2.38s`. The retained live driver `task2_h200_live_smoke.py` then exercised `Qwen/Qwen3.6-27B` at `127.0.0.1:8010`: 8 independent choices, 8 parseable, 5 exact-valid/selectable, 585.72 ms generation, 1.03 ms evaluation, 855.57 ms planner total. Both selected lines had exactly 9 syllables and the combined request contained 18 strictly scheduled targets spanning syncopated and staggered flow templates with an exact 128,000-frame 90 BPM target.
- Task 2 review round 1: core algorithm and H200 behavior passed, but acceptance is pending fixes for pre-first-wave cutoff classification, analysis-time attribution, partial API-choice index/warning propagation, and typed workspace failures. Ruling: defer splitting the 842-line planner/orchestrator module until after the working vertical slice; address every behavioral finding and named branch test now. Cost if wrong: maintainability remains worse during later server integration, but the public interface and test surface stay stable while P0 is built.
- Task 5 review round 1: acceptance is pending fixes for pre-header cancellation/deadline races, exact frame conversion at non-90 tempos, target-anchor equality to the Mac schedule, bounded health reads, missing edge tests, and visible warning/latency propagation. Ruling: the Mac's summed per-bar frame count is authoritative after one full-chunk resample; endpoint-rounding correction may trim/pad only at the outer chunk boundary. Cost if wrong: a non-90 BPM chunk could otherwise be rejected or split with a one-frame discontinuity.
- Accepted Task 2 H200 rerun after `29f88c8d`: `97 passed in 2.34s`. A fresh live 27B planner request returned 8 parseable / 3 valid / 3 selectable candidates, selected two exact nine-syllable lines, and completed in 837.07 ms: 563.56 ms generation, 271.40 ms CMU analysis/ranking, and 2.11 ms explicit overhead. This confirms the repaired timing attribution rather than changing planner latency.

## Task Status

- Task 1: complete (`955b3bfe`; acceptance review PASS; local 99 focused/384 adjacent; H200 99 focused)
- Task 2: complete (`a1c14f15` + `29f88c8d`; re-review PASS; local 97 focused/190 adjacent; isolated H200 97 focused plus accepted live-Qwen smoke)
- Task 3: complete (persistent MOSS + resident MMS + continuous-onset R3; real H200 smoke PASS)
- Task 4: complete (`0a8b90fd`; final review PASS; 195 adjacent/30 server tests and clean wheel)
- Task 5: complete (`f2e40438` + `56c20bc1`; re-review PASS; local 56 focused/114 adjacent)
- Task 6: complete (`0af9f57c`; review PASS; six-bar exact WAV and zero-underrun lifecycle smoke)
- Task 7: complete (`9c285aa2`; fresh review PASS; 343 focused plus 66 eSpeak regression tests)
- Task 8: complete with a qualified deployment result (`9c285aa2`; final suite 1753 passed/4 skipped)

## Final Acceptance, 2026-08-21

- Exact final revision deployed to isolated H200 path
  `/data/home/Andrew.Yang/StreamMUSE/deploy/real_rap_audio_9c285aa2`.
- Final direct H200 request: 3.473 s, 36 requested / 28 parseable / 6
  valid-selectable candidates, exact 128,000-frame two-bar vocal, MMS
  confidence 0.8405, and no alignment fallback.
- Robust warm profile: 9/10 success before digit normalization, 8.11 mean
  selectable candidates, 4 minimum, and 3.442 s p95 server total. The one
  failure was the digit-bearing transcript fixed in `d67360ba`.
- Final Mac run `rap-20260821T122702Z-782db4bd`: 20 bars at 90 BPM,
  2,560,000 exact frames, 53.333333 s, zero underruns, and every bar audible.
- Qualified result: 2 remote MOSS bars and 18 prepared local fallback bars.
  The measured SSH path took 5.245 s for a cached 196 KB response before any
  H200 rendering, so two-bar lookahead at 90 BPM cannot carry full PCM on this
  topology. The fallback was visible and deadline-safe.
- Real website smoke passed Start, bar-quantized Stop, Reset, remote chunk
  monitoring, exact flow/context display, and zero document-level horizontal
  overflow.
- Full report and retained artifact paths:
  `docs/developer-guide/realtime-remote-moss-acceptance-2026-08-21.md`.
