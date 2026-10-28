# Independent Rap Settings Implementation Plan

> **For agentic workers:** Use test-driven development and scoped subagents. Preserve the existing dirty worktree; do not commit unrelated work.

**Goal:** Separate local audio, server-coordinated generation, and stopped session setup, without restarting playback for local changes.

**Architecture:** Independently synchronized settings groups feed existing controllers. Volume changes reach the Mac audio callback with smoothing; alignment changes reuse retained MOSS source speech and affect only uncommitted bars. Generation updates keep their existing server acknowledgements and scheduling behavior.

**Tech Stack:** Python/FastAPI, existing v3 stream controllers, SoundDevice/NumPy, vanilla JavaScript, pytest.

**Spec:** User-approved design in this conversation: current bar remains unchanged by alignment, earliest uncommitted bar changes; local controls remain available while server updates wait.

## Constraints
- Use the existing feature/rap_optimization worktree and preserve previous edits.
- No new H200 model calls for local audio changes.
- Preserve continuous/exchange modes, topic/flow ASAP, devices, and Stop/Start.
- Do not restart H200 models unnecessarily; Mac client may restart to load code.
- Keep legacy settings endpoint compatible; add group-specific paths.
- Volume is an explicit exception to current-bar immutability: it changes smoothly during playback; alignment and chord structure do not.

## Review Focus
- Tempo acknowledgement blocks while a local volume/alignment update completes.
- Old render finishes after a newer alignment policy and cannot replace it.
- Policy changes at a commit boundary never alter an already enqueued bar.
- Rapid independent updates preserve accepted values and truthful per-group status.
- Stop/Start, reset, device changes, and exchange playback retain correct settings.

## Task 1: Local render policy updates
Files: application/rap/{bar_audio,bar_stream,interaction}.py, infrastructure/rap/phrase_warp.py, matching tests.
Interface: controllers expose update_alignment_policy(policy: str) -> dict with effective_from_bar, alignment_policy, revision, coordinator_epoch. Existing source audio is retained only for bounded uncommitted windows; stale jobs are rejected.
- [x] Write failing tests for cached-source reuse, committed-bar immutability, stale rendering, continuous/exchange modes.
- [x] Add per-policy preparation and controller updates without remote calls or session reset.
- [x] Run focused tests and review.

## Task 2: Immediate local gain control
Files: infrastructure/rap/audio_output.py and focused helper/test files.
Interface: SoundDevice/Composite sinks expose set_live_gains(*, drum_gain, vocal_gain, chord_gain) -> dict. output_device_status includes applied live-gain evidence.
- [x] Test same-bar gain changes, short smoothing ramp, zero-to-nonzero gain, unchanged timing and stop behavior.
- [x] Implement callback-level mixing from retained stems; preserve clipping protection and truthful recording/evidence.
- [x] Run focused tests and review.

## Task 3: Independent settings control plane
Files: application/rap/{settings,audio_settings,runtime}.py, presentation/rap_demo/server.py, matching tests.
Interfaces: GET/POST /api/control/settings/{local,generation,session}; existing endpoint retained. Full snapshots contain values, capabilities, application, and groups with fields and independent acceptance metadata. Local = drum_gain/vocal_gain/chord_gain/chord_loop/alignment_policy; generation = tempo_bpm; session = mode/response_bars/output_device.
- [x] Test local updates bypass blocked remote discovery/ACK, values merge without loss, independent acknowledgements, mixed-group validation.
- [x] Split update locks, avoid remote discovery on local path, bind live gain and alignment APIs.
- [x] Preserve lifecycle fencing for reset/start/session rebuild; permit local edits during generation waits.
- [x] Run focused tests and review.

## Task 4: Grouped controls
Files: presentation/rap_demo/static/{index.html,css/rap-demo.css,js/rap-settings.js}, UI integration tests.
- [x] Test independent requests, local auto-apply/coalescing, pending generation not disabling local fields, per-group statuses.
- [x] Group local controls, generation tempo/topic/flow, and stopped setup without dropping existing diagnostics.
- [x] Show honest activation timing and alignment applied evidence from played bars.
- [x] Run JavaScript-backed tests and browser checks.

## Task 5: End-to-end acceptance
- [x] Run full pytest suite and git diff --check.
- [x] Test Mac audio and H200 models: live gains, alignment changes, unchanged current bar, queued source reuse, both modes, topic/flow and tempo.
- [x] Verify client/server health and leave a stopped ready demo.
- [x] Update user guide and write measured acceptance report with limitations.

## Execution Log
- Plan created after explicit user approval; execution proceeds without another review gate.
- No commits are planned because the worktree contains accumulated work from earlier tasks.
- First live acceptance: 10 continuous MOSS bars, two exchange MOSS response bars, zero underruns. Three policy edits applied to the exact next uncommitted bar with the prior bar preserved. Local gain POST 84 ms while tempo POST remained pending (1064 ms).
- Review fixes: stopped tempo rebuild must rebase accepted local settings; untouched groups must not inherit another group's acknowledgement; exchange echo percussion must not be muted by the vocal slider.
- Ruling: mix fields coalesce together, alignment uses a separate queued local operation. This avoids reporting partial success from a mixed render/mix operation; the UI handles separation automatically.
- Latest user request adds explicit availability labels and reasons for disabled controls. Played confirmation must remain confirmed after a following drums-only bar, without mislabeling stopped playback as playing.
- Repeated live acceptance passed: 10 continuous plus two exchange MOSS bars, zero underruns; local gain response 130 ms while tempo response was pending (510 ms). Final three-bar smoke test confirmed retained sink gains after a stopped bulk rebuild.
- Final review fixes: polls started during a POST cannot overwrite its accepted group; stopped bulk gain changes also update the retained sink. Both regressions were observed failing before their fixes.
- Mac client left stopped at 60 BPM, continuous mode, relaxed alignment, drums 0.78, vocals 0.8, chords off. H200 model health remains ready. No commits made.
- Final full suite: 2863 passed, 5 skipped, 36 warnings in 114.19 seconds. JavaScript syntax check and git diff --check passed.
