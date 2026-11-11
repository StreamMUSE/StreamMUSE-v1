# Three-Hour Interaction Feature Delivery

Started November 5, 2026 at 05:58 Pacific (13:58 UTC).
Hard end: 08:58 Pacific (16:58 UTC), including verification and presentation.

## Approved Scope

Separate features: directed story, microphone transcription, two characters and
a small concurrent-collaboration experiment. Stress alignment and voice
consistency are deferred to next week. Preserve the existing audio pipeline.
Microphone capture is explicit push-to-talk; release transcribes automatically and
displays the result. A selected Story destination submits it without confirmation.

## Execution

- Integration owner: environment, service connectivity, API mounting, deployment,
  end-to-end checks, demo artifacts and presentation.
- Story worker: Kierkegaard / 01a1ac02-a841-7210-89a3-5a803acd47f4.
- Microphone worker: Rawls / 01a1ac02-a8d1-7c33-9403-76299aca8e42.
- Character worker: Arendt / 01a1ac02-aa28-7d52-825a-e1fb1be1456a.
- UI worker: Epicurus / 01a1ac05-4884-7d31-870b-dc3476ce7c28.
  Initial GUI dispatch failed model capacity before implementation.

## Baseline Findings

- Git worktree links repaired after UGRIP -> MBZUAI parent-directory rename.
  Existing branch feature/rap_optimization and all preexisting changes preserved.
- Local demo PID 57381 on port 8018 is stopped and responds correctly.
- Existing H200 Qwen port 18711 and MOSS-backed rap port 18720 are healthy.
- SSH tunnel had expired; restored at /tmp/rap-research-20261105.sock with
  forwards 18720 and 18711. No model services restarted at baseline.
- Other users share GPUs 0 and 1. User explicitly permits continuing; no other
  processes are touched. Latency results are shared-load observations.
- Baseline tests: 85 passed (21 existing WAV chunk warnings), covering lyric
  session, phrase warping, and lyric-stream presentation. PYTHONPATH=src is needed
  because the existing editable installation references the old directory.
- Live playback/edit baseline passed: two edits at 60 BPM reached callback-scheduled
  vocal playback in 4.123 and 5.947 seconds. These are not acoustic measurements.
- Isolated renderer 18721 is healthy and uses the existing Qwen 18711 and MOSS
  18730 services. Mac research client runs on 8019; original 8018/18720 retained.
- Eight real-model character turns completed in 7.240 seconds. Counting pilot:
  fixed turns 3/3 complete (10 calls), naive concurrency 0/3 (duplicate number),
  coordinated publication 3/3 complete (20 calls). Small shared-load pilot only.
- Browser prerecorded-microphone capture -> local ASR -> visible transcript passed
  in 1.262 seconds after release. This is not a human microphone/accuracy test.
- First live story run accepted all three directions but could not verify played
  story provenance. Investigating instrumentation and weak semantic adherence;
  not yet an accepted story demonstration.
- Broad rap unit suite: 1829 passed, one cancellation-test failure under review.

## Integration Results

- The cancellation failure was a test race (cancellation before its handler was
  entered); fixed and repeated 10 times. Broad rerun: 1848 passed, 35 preexisting
  WAV chunk warnings. Subsequent small API/telemetry additions have focused tests.
- Story prompt diagnosis: the old topic dominated initial generation. A bounded
  prompt probe made the newest direction the lyric subject. That story-only change
  left ordinary prompts unchanged. Keyword counts are diagnostics, not semantic
  correctness scores.
- Updated 16-bar live story run: all three context versions reached callback
  playback after 3.850, 3.886, and 3.953 seconds at 60 BPM. All bars used MOSS.
  The actual courier/map/stranger/package lyrics are retained. Story coherence
  remains uneven; expressing the requested action can take longer than carrying
  its context version into playback.
- Full Chromium prerecorded-microphone -> local ASR -> opt-in Story -> H200 rap
  test passed. Source was correctly marked microphone. Start, Stop, and resume
  without Reset passed. This does not verify a physical microphone or headphones.
- Real browser character tests passed dialogue, coordinated counting, expected
  naive failure, cancellation, and restart; desktop/mobile screenshots retained.
- Existing live controls regression passed: three alignment policies, tempo,
  gains/chords, topic/flow changes, Stop/Start, and alternating keyboard-flow
  capture with two model response bars. Zero reported audio underruns. Original
  alignment/mix restored afterward.

## Decisions

- Ruling: use a separate /research page instead of reworking the main demo page;
  this reduces regression risk. A compact lyric monitor and Start/Stop are included
  there; detailed audio/session settings remain in Rap Lab.
- Ruling: local CPU transcription first; reuse already running H200 models for
  lyrics and characters. Cost: ASR latency is measured, not assumed low.
- Ruling: no new serving engine or scheduler in this three-hour delivery.
  Cost: concurrency results characterize the current engine, not an optimal one.

## Final Verification

- Independent backend review reproduced three issues: playback-lock contention
  during story delivery, obsolete retry sequence after a later edit, and late
  submissions crossing Reset. All were fixed and independently rechecked. The
  review report preserves the original findings and resolution evidence.
- Final broad Python rerun: 1881 passed, 35 existing WAV metadata warnings,
  31.56 seconds. A restricted-sandbox run could not bind the one HTTP fixture;
  rerunning with localhost access passed the full suite.
- Real-browser delayed-ASR/Reset test passed: transcript visible, explicit warning,
  new story ledger empty. Warm normal browser-to-Story routing took 493 ms.
- Five 16-bar live story runs retained 80 MOSS-played bars. Early edits reached
  the next bar at 60/90/120 BPM. Three late 90 BPM edits instead reached the
  following bar. These small shared-load pilots are not reliability estimates.
- Eight-slide editable PPTX and image-based PDF preview are built and verified.
  Source-matched numbers, chart data, page count, and all slide previews passed.
- Final UI context-version check found and fixed a session-identity mismatch:
  playback snapshots use a base ID, while the story controller appends its epoch.
  Live desktop/mobile screenshots now show the correct frozen playing version.
  Cross-session and unknown-epoch guards remain covered. Research UI: 42 Node
  tests passed. Original GUI reducers and interaction controls: 88 tests passed.

## Delivery Gates

- [x] Directed story implementation and focused tests.
- [x] Microphone implementation and real ASR test.
- [x] Characters/counting implementation and H200 runs.
- [x] Final research UI desktop/mobile/context-readout check.
- [x] Independent code review and required fixes.
- [x] Live Mac-H200 story and regression acceptance.
- [x] Demo guide, recordings, results, and presentation.

## Handoff Artifacts

- User guide: `docs/developer-guide/interaction-research-demo.md`.
- Research report: `docs/reports/2026-11-05-interaction-research-results.md`.
- Coordination details: `docs/reports/2026-11-05-coordination-pilot.md`.
- Presentation: `docs/research/presentation-20261105/output/final/`.
- Audio, chosen lyrics, and timing summary: `output/research_day_20261105/demo/`.

The original 8018 client and 18720 renderer remain available. The new Mac client
is on 8019, with its isolated H200 renderer on 18721, reusing existing Qwen 18711
and MOSS 18730 services. No other GPU processes were stopped. No changes were
committed or pushed during this task; existing unrelated work remains intact.

Remaining scope limits: physical microphone and acoustic playback are unverified;
characters/counting are text-only; concurrent counting uses synchronized proposal
rounds with host-checked publication, not free-running listening/speaking agents.
The ASR dependency is installed locally and documented as optional setup rather
than made mandatory for every rap installation. Stress/voice work stays deferred.

Final handoff check: research client stopped at 60 BPM, empty story ledger,
continuous mode, existing all_onsets_r3 alignment and mix retained. ZoomAudioDevice
is still selected. Local ASR is warmed and idle; character runner idle. The SSH
master and forwarded Qwen/isolated-renderer health endpoints responded correctly.
Original and research services intentionally remain running for the demo.

## User Review: Next-Cycle Scope

The delivery gates above describe tested plumbing, not acceptance of the overall
meeting demo. User review found microphone integration incomplete and standalone
text characters/counting insufficient. Directed story is accepted as-is for now.
Further implementation was explicitly deferred for lack of time.

The agreed next steps are recorded in
`docs/research/2026-11-05-interaction-next-steps.md`: spoken topics in the rap UI,
two user-chosen personas in a continuous audible rap battle, and a separate
counting coordination task with visible time and logical constraints. Preserve
the distinction between generated, accepted, and performed outputs, and do not
equate host filtering with successful agent collaboration.
