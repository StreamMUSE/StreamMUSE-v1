# Inference Milestone Execution Ledger

Spec: `docs/research/2026-11-10-inference-page-milestone.md`.
User approved implementation with subagents, Mac/H200 testing, GPUs 1 and 2 only.

## Baseline And Rulings

- Existing `feature/rap_optimization` worktree contains substantial uncommitted
  earlier work. Preserve it; no branch change, reset, commit, or push requested.
- Ruling: use this existing worktree, not another branch/session, per user scope.
- Ruling: GPU1 hosts Qwen and GPU2 hosts MOSS/aligner with limited allocations.
  Both have existing unrelated low-occupancy jobs; leave those untouched and label
  measurements shared-load. Cost: timings are not isolated hardware benchmarks.
- Ruling: approved proposed defaults are keyboard USER capture, 60 BPM, USER one
  bar / MODEL two bars, silent readiness-driven WAIT, empty capture repeats USER.
  Cost: adjustable defaults may need user refinement after listening.

## Task Gates

- [x] Granularity-independent contracts and experimental controllers/backends.
- [x] Shared-runtime/API integration with exclusive playback ownership.
- [x] Clean `/inference` UI and browser tests, including live keyboard capture.
- [x] Timing metrics, traces, and exports.
- [x] H200 GPU1/2 model deployment and real Mac playback acceptance.
- [x] Independent review and original-feature regressions.
- [x] Source-verified optimization handoff and final user guide.

## Verification

- New HTTP control boundary: 21 tests passed before page integration. Covers
  dynamically resolving the shared service, bounded/strict JSON, lifecycle fields,
  error status mapping, and JSON/CSV export. Page route test added subsequently.
- Initial legacy regression sweep: 1,853 passed; one localhost HTTP test was
  blocked by sandbox socket permissions. Rerunning its full module with localhost
  permission passed all 16 tests. A final post-integration sweep remains required.
- No Mac StreamMUSE client or SSH forwarding processes were running at startup.
  Existing launch records/logs are retained; the updated client will be started
  after remote model health and composition are verified.
- Original GUI reducer/control regressions: 88 tests passed. Existing research
  page Node tests: 42 passed. Shared page/API route tests: 23 passed.
- H200 Qwen and warmed MOSS/aligner health verified through localhost SSH forwards
  18711/18721. Mac original-mode smoke: six completed bars (0 through 5),
  1,152,000 frames at 48 kHz, zero reported underruns and zero fallbacks, clean
  bar-boundary Stop. Software/device evidence, not a human listening judgment.
  Recording: `output/inference_20261111/client/sessions/rap-20261111T090133Z-08995a5f/mixed.wav`.

## Integrated Acceptance In Progress

- Final reviewed code regression: 2,039 scoped rap tests passed, plus 88 legacy
  GUI tests and 81 inference/research Node tests. Full unrelated repository
  collection has five existing Lekai/Transformers import failures; these results
  are explicitly scoped to rap, not a claim that the entire repository passes.
- Real five-cycle run completed ten MODEL bars with zero missing/stale/fallback
  duration, callback warnings/errors, or recording drops. Response-start p50
  5.280 s, p95 5.715 s, maximum 5.806 s (n=5). Ten decisions averaged ten valid
  candidates; nine requested 32 and one requested 48 including rescue.
  Evidence: `output/inference_20261111/final-cycle/`.
- Desktop 1440x1000 and mobile 390x844 read-only browser checks passed with no
  overflow/page errors; refresh and duplicate pages retained session/epoch.
- Live steering exposed a genuine integration issue: a retryable server lyric
  failure was treated as a terminal adapter error. Failure trace retained in
  `output/inference_20261111/steering-failure.json`. Fix and rerun are pending;
  steering is not yet accepted. No silent fallback was substituted.
- Source-verified handoff and user guide exist; their verification section will
  be finalized after the remaining live acceptance and independent review.

## Final Verification

- Final source regression: **2,055 scoped rap tests passed** (35.28 s), plus
  **88 legacy GUI checks** and **81 inference/research Node tests**. The 35 WAV
  metadata warnings are known test-fixture warnings, not failing tests.
- Independent review resolved all six initial findings. A follow-up caught an
  older listener-signature compatibility issue in retry telemetry. The final
  server uses optional `writer_error` metadata while retaining legacy
  `error(message, *, bar=None)`. Retry limits/timeouts and fatal errors remain
  enforced; no hidden fallback was introduced.
- Real browser keyboard acceptance passed: seven Space taps, leading onset tick
  2 retained, visible real syllables, refresh preserved epoch/owner/playback,
  Stop completed, and original-page Stop enabled/disabled correctly.
  Evidence: `output/inference_20261111/browser-cycle/`.
- Live topic plus flow change reached new-version playback after **7.480 s**.
  Old prepared output covered the wait as **7.478 s stale context**; zero global
  missing vocals, callback errors/warnings, or recording drops. Stop/Start kept
  history without Reset. Both directions of cross-page Stop passed, competing
  starts and legacy mode-specific edits were rejected, and original playback
  reached bar 1 with zero fallbacks/underruns.
  Evidence: `output/inference_20261111/steering/`.
- Source-synchronized final renderer is PID 2948248 on physical GPU 2; Mac client
  PID 93588 remains running. Model processes remain on GPUs 1/2.
- The **final-source repeat five-cycle run passed**: ten MODEL bars (40 s), all
  five requests `played`, zero missing/stale/fallback coverage, zero errors or
  canceled requests, and no callback/recording warnings, errors or drops.
  Response-start p50 **5.337 s**, p95 **5.973 s**, maximum **6.132 s** (n=5).
  Ten selected decisions averaged **6.2 valid candidates**. Their per-response
  requested totals (two bars each) were 64, 64, 96, 64, 64; these exclude any
  entirely unsuccessful writer attempts not represented by selected provenance.
  Evidence: `output/inference_20261111/final-source-cycle/`.
  These are shared-load observations, not a general latency or reliability guarantee.
- Final-source steering also passed: new-version playback at **7.480 s**, stale
  duration **7.465 s**, zero global missing audio and callback warnings/errors.
  Restart/history and cross-page ownership tests passed again; original playback
  had zero fallbacks/underruns. Evidence:
  `output/inference_20261111/final-source-steering/`.
- Final H200 verification at 2026-11-11T10:08:02Z passed owned-service health,
  real Qwen completion and model endpoints, with our workloads on physical GPUs
  1 and 2 only. Source hashes and restart identities are in the deployment report.
- Traces were exported before cleanup. The app is left stopped on `/inference`,
  cyclic mode, 60 BPM, USER one bar / MODEL two bars, topic `space`. Mac client,
  SSH forwarding, and H200 services remain running for demonstration.
- Subsequent human listening feedback: after two requested two-cycle replays,
  the user reported that playback sounded "pretty good", but the rapper's
  tone/voice changed completely across the two parts. This confirms subjective
  audibility, not measured acoustic timing or voice consistency. Whether the
  change was between bars or between response turns remains to be clarified.
  Replay evidence: `output/inference_20261111/listening-replay/` and
  `output/inference_20261111/listening-replay-2/`. No renderer changes were made.
- The final session recording is readable stereo PCM at 48 kHz, 143.915 s at
  verification, peak 0.7342 and zero samples at or above 0.999 full scale. This
  cumulative recording includes active tests and silent WAIT, not stopped idle:
  `output/inference_20261111/client/sessions/rap-20261111T095919Z-6e9d91ae/inference.wav`.

## Delivered Documents

- [Inference optimization handoff](../developer-guide/inference-optimization-handoff.md):
  source map, editable scope, actual execution, granularity/version contracts,
  measurement boundaries, deployment, and evidence.
- [Inference page guide](../developer-guide/inference-page-guide.md): controls,
  launch/SSH commands, limitations, and demonstration results.
- Final documentation validation: 83 links/anchors and five shell snippets
  checked. No code was changed after final-source acceptance. No commit/push or
  branch switch was requested or performed; earlier dirty work was preserved.
