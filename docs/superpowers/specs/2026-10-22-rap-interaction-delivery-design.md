# Rap Interaction and GUI Delivery Design

Status: approved for implementation on 2026-10-22. Progress and live acceptance evidence are recorded separately in `docs/reports/2026-10-22-feature-delivery.md`.

Companion research track: [scheduling model](2026-10-22-realtime-scheduling-model-design.md). That model is not a dependency or acceptance gate for shipping these features. The features are required deliverables, not theory-only examples.

## 1. Goal and Existing System

Keep the existing Mac GUI/client and H200 model-server workflow working while adding live tempo control, a keyboard-flow interaction mode, and the requested usability/diagnostic improvements. Use the current Python audio/runtime and existing browser UI, not a replacement application or browser-only audio engine.

Primary integration target: protocol v3 with the selected remote renderer and explicit eSpeak fallback. Existing standalone eSpeak and v1/v2 workflows remain available. New v3-only capabilities must be visibly unavailable for unsupported backends rather than silently changing their behavior.

Confirmed user requirements:

- Continuous model generation with steering and alternating user/model exchange are two separate interaction modes.
- Drums continue while the user captures a rhythm with keyboard taps.
- Generate after one complete input bar, not from partial input in this version.
- Replay the captured rhythm while waiting.
- Start the model response when its first bar is ready; later bars are generated progressively.
- Make model response length configurable.
- Add live BPM changes, retaining existing topic and template controls.
- Expose audio device selection, more prominent drums and gain controls, readable stress templates, additional `UNTESTED` templates, and better timing diagnostics.
- Start must work after Stop without requiring a manual Reset.
- Preserve existing monitoring, terminal logs, session artifacts, and fallback visibility.

## 2. Practical Defaults Chosen for Review

These engineering defaults were approved for this delivery. They are not research conclusions:

1. Remain in 4/4 with a 16-tick grid. This delivery does not implement beat/tick-sized lyric generation.
2. BPM changes are steps at an acknowledged future bar boundary, never a mid-bar ramp. UI range is 30-180 BPM; validation outside that demo range rejects clearly rather than clamping silently. The range is not a promise that primary-model audio meets deadlines at every tempo.
3. Space records onsets with neutral stress `0.5`. Stress inference and multiple stress keys are deferred; current chosen templates retain their existing stress information.
4. The first Space press in the focused capture surface arms the next full bar. The UI shows the countdown and capture state. The arming press is not a captured onset. This avoids labeling a partial current bar as a complete input bar.
5. Quantize recorded taps to the nearest valid tick in that bar, with ties toward the earlier tick. Clip a last-half-tick rounding result to tick 15; merge duplicate ticks. Retain raw timestamps and quantization/collision diagnostics.
6. Capture length is one bar in this feature. Model response length is selectable while stopped, default two bars, range 1-8. Other capture lengths remain in the theoretical model, not this delivery.
7. Repeat the one-bar flow for each model-response bar. Use distinct model lyrics per response bar; do not pre-generate an entire song or require the entire response before starting.
8. Wait for the selected primary renderer for the first response bar, with a 30-second startup budget. At budget expiry, a ready eSpeak rendering of the matching generated lyric may start with a visible fallback warning. Never substitute a different flow silently. If there is no compatible generated line/audio, keep drums/replay available, stop the generation attempt, and show failure; permit recapture or Stop.
9. Output-device changes and interaction-mode/response-length changes are stopped-only. BPM, topic/template, and mix changes have explicit future effective boundaries.
10. Increase the default drum contribution by approximately 3 dB relative to the existing mix, preserving the peak limiter. Expose drum/vocal levels independently; listening verification may reduce the default if it obscures speech. Do not change sound output on an existing running session without an explicit settings action/restart.
11. Response length counts delivered generated-lyric bars, including a matching eSpeak rendering, not echo/drums-only slots. After the response starts, prefer ready primary audio, then matching generated-lyric eSpeak at each boundary. If neither is ready, replay the input while waiting for that same logical response bar. Give each blocked response bar at most 30 seconds of additional wait, then terminate the turn as incomplete with a visible error and return to drums idle. No unbounded retry loop or silent counting of echoes as generated output.

## 3. Interaction State Machine

`DRUMS_IDLE -> ARMED -> CAPTURING -> WAIT_REPLAY -> MODEL_PLAY -> DRUMS_IDLE`. A later unavailable response bar can transition `MODEL_PLAY -> WAIT_REPLAY -> MODEL_PLAY` without advancing the logical response cursor.

- `DRUMS_IDLE`: continuous drums, no model vocal. Space can arm capture only when the capture surface is focused and no editable form control has focus.
- `ARMED`: countdown to the next bar; no partial bar is sent. Stop cancels the arm.
- `CAPTURING`: record one full bar. Ignore key auto-repeat; do not interpret typing in topic/settings fields as rhythm. A tempo edit does not change the timing basis halfway through this capture.
- `WAIT_REPLAY`: freeze the captured template, send it once, and render its percussion echo over the continuous drums. Show the exact captured grid, requested renderer, elapsed wait, and replay count. Delayed/failed generation is visible.
- `MODEL_PLAY`: begin at the next legal bar boundary after first-bar preparation; use progressive generation for the remaining configured bars. Show generated lyrics and their flow together. Later-bar fallback/misses retain explicit provenance; do not count them as fresh primary output. Advance the logical response cursor only when a compatible generated-lyric bar is delivered. Echo occupies physical time but does not satisfy that response obligation. Record the originally intended boundary, missed deadline, and actual delayed delivery rather than redefining the miss away.
- Return to `DRUMS_IDLE` after the requested response length. No automatic new capture is assumed.
- Stop is bar-quantized in every phase. Reset clears capture, turn, queued work, and epoch. Reconnect cannot duplicate a submitted turn.
- An empty captured bar does not create an empty `FlowTemplate` or trigger generation. Continue drums and return to idle with a visible empty-input state.

Echo uses a distinct simple percussive sound at the captured onsets. Immediate audible key monitoring during capture is not required for this first version; visual feedback is immediate and the recorded rhythm is audibly replayed in the next phase.

### Logical Response Bars Versus Playback Bars

Replay consumes physical musical bars but does not advance the server's logical generated-response cursor. Do not send an echo bar as a committed model lyric or repeatedly move the target while generation runs.

Use explicit `turn_id`, logical response ordinal, and physical playback-bar mapping. A turn-scoped server session can start at a monotonically increasing response ordinal and carry the previous generated-text context. It produces only the requested response span. At delivery, validate source identity first, then materialize the selected flow and audio at the chosen physical bar/timeline. Preserve both identities in events.

The existing continuous `BarStreamController` remains the default. Add a dedicated alternating coordinator/adapter instead of filling that class with phase-specific conditionals. Reuse the existing lyric writer, source-audio service, preparation adapters, playback sink, and monitoring.

## 4. Tempo and Artifact Contracts

Do not mutate the shared piano-system `Tempo` globally. Introduce a rap-specific immutable, versioned timeline with piecewise-constant tempo segments.

A single per-bar timing snapshot contains: timeline revision, physical bar, BPM, sample rate, absolute start/end frames, and 17 boundary/tick offsets. Rendering, drums, commit guards, playback, and diagnostics all use that snapshot. Static-tempo behavior must match existing frame rounding; changing BPM must not recompute earlier bars with the new tempo.

Tempo updates apply no earlier than the first uncommitted bar with a feasible preparation boundary. Return requested and effective boundaries. Capturing a flow locks its bar's timing; changes can affect later phases/bars, not distort already captured timestamps.

Preparation deadlines and lookahead durations are recalculated from the timeline. Never enqueue audio whose frame count/timeline revision is incompatible with its destination. If re-preparation is not ready, use a correctly timed local alternative or drums-only output with an explicit warning, not an old-duration waveform.

Keep content identity distinct from timing/render identity. Extend v3 through explicit capability negotiation and optional versioned fields, with legacy behavior unchanged when the capability is absent. The server needs per-bar tempo updates and bounded turn output; unsupported servers must reject these operations before the client changes its plan.

Cache reuse must follow documented dependencies. Retain source audio/onsets where safely reusable; rewarp/remix for the new destination timing. Do not assume all source audio is tempo-independent merely because its text is unchanged. Include all relevant configuration and timing dependencies in render keys. Late old-version work cannot overwrite new-version preparation.

## 5. Settings and Audio Output

Add a settings endpoint and compact GUI panel, with a typed schema rather than arbitrary runtime attribute mutation. The response includes current/requested values, supported capabilities, application mode (`stopped-only` or `bar-boundary`), and effective bar/revision for accepted changes.

First-version controls:

- BPM and its pending/effective boundary.
- Continuous or alternating interaction mode.
- Response length, 1-8 bars, for alternating mode.
- Output-device menu with host API/name, availability, and selected device.
- Drum and vocal levels with a bounded master output and existing limiter.

Device enumeration uses the existing Python/sounddevice backend. Device indices are not stable identifiers; resolve and validate the selected host/name/device descriptor at application time. Unsupported formats, missing devices, and hot unplug are visible errors. Do not show audible playback success when a timed/no-device fallback is active. Reopening a device while stopped must preserve or explicitly reinitialize playback state without losing the resume position silently.

Keep vocal/drum stems or equivalent reusable preparation inputs long enough to remix uncommitted bars without repeating LLM/TTS work solely for a level change. Mix updates are versioned and applied at acknowledged boundaries.

## 6. Stop, Resume, and Reset

Reproduce the reported failure first. Existing code already contains a resume path and only intentionally requires Reset after a completed finite scenario; do not replace it blindly.

Required behavior:

- Stop completes the current bar and quiesces playback.
- Start after a manual Stop resumes at the next appropriate bar/phase, without a separate Reset.
- Start after natural finite completion begins a new epoch/session automatically, with clear start-over labeling.
- Reset remains an explicit discard-and-return-to-start action while stopped.
- Stop while arming/capturing/waiting cannot later start model audio from an obsolete asynchronous result.
- Repeated clicks, stop during priming, connection failures, and device errors cannot strand the UI in a false running/starting state.

## 7. Flow Visibility and Timing Diagnostics

Keep the existing dense monitoring layout. Place a persistent readable template strip directly beside/above the currently played lyrics. Give weak/medium/strong stress visibly distinct heights and redundant symbols/fills; do not rely on color or tiny height differences alone.

Add six handcrafted stress variants using existing onset grids, preserving provenance and marking every new variant `UNTESTED` in the selector, preview, metadata, and logs. Structural validation does not remove that label; listening/live validation is a separate event. Never label handcrafted variants as MCFlow-extracted.

For each played bar, the grid distinguishes:

1. **Requested** onset positions from the selected flow/revision.
2. **Rendered estimate** from source alignment and the actual applied warp time map, including sparse-anchor drift and confidence.
3. **Playback cursor/observation** from the local audio clock.

MOSS preparation currently discards useful warp diagnostics; retain and publish them. Do not display scheduled positions twice and call the second trace actual audio. Device observation is not a measured acoustic syllable onset. Independent loopback/acoustic verification can be a later diagnostic mode; unavailable measurements remain explicitly unavailable.

Show played versus requested topic/template/tempo revisions and fallback identity. Older audio must not be depicted as if it realized a newer edit. Preserve terminal events and existing candidate/latency views.

## 8. Execution Milestones

These are independently testable engineering deliveries; theory work can proceed separately.

### E1. Baseline, Lifecycle, and Settings Contract

Primary files: `application/rap/runtime.py`, `presentation/rap_demo/server.py`, `presentation/rap_demo/cli.py`, existing lifecycle tests; new `application/rap/settings.py` and unit tests.

- Reproduce manual Stop/Start and finite-completion behavior with regression tests before fixes.
- Add the typed settings/capability contract and explicit errors/effective-state responses.
- Add device enumeration and stopped-only device selection through audio factories/sinks.
- Verify manual resume, finite replay, repeated commands, priming interruption, stale epoch, and missing device behavior.

### E2. Versioned Rap Timeline and Dynamic BPM

Primary files: new `domain/rap/timeline.py`; `application/rap/audio_rendering.py`, `playback.py`, `bar_stream.py`, `bar_audio.py`, `bar_renderer.py`; `domain/rap/lyric_stream.py`, `application/rap/lyric_session.py`, `presentation/rap_lyric_stream_server.py`, `infrastructure/rap/lyric_stream_client.py`; timing/protocol tests.

- Introduce the immutable per-bar timing snapshot and backwards-compatible static adapter.
- Add capability-negotiated tempo edits and revision-correct preparation/invalidation.
- Preserve past sample positions, exact current-bar length, and uncommitted-only changes.
- Test faster/slower transitions, repeated edits, late results, rewarp failure, fallback timing, and old-server rejection. No piano timing refactor.

### E3. Keyboard Capture, Echo, and Alternating Coordinator

Primary files: new `application/rap/interaction.py`, `domain/rap/interaction.py`, `infrastructure/rap/flow_echo.py`, `presentation/rap_demo/static/js/rap-interaction.js`; existing runtime, server, template transport and audio preparation integration; new deterministic interaction tests.

- Implement the state machine and explicit logical/physical bar mapping.
- Record raw browser event times and server receipt, with a calibrated local clock mapping and uncertainty; do not equate HTTP receipt time to the tap time.
- Build a valid ephemeral flow from quantized taps; submit once after capture completes.
- Keep drums/echo running without treating replay as model progress.
- Start from the first compatible ready response bar, then generate progressively to the configured response length.
- Test empty/dense/duplicate taps, held Space, focus in a text field, mid-bar arming, initial and mid-response timeouts, delayed old turn, Stop/Reset, and a tempo edit during capture/waiting. A two-bar response with three intervening echoes must deliver exactly two generated-lyric bars, log all three echoes and the original missed boundaries, and never advance the server's committed model cursor for an echo.

### E4. GUI Controls, Stress Visibility, and Mix Levels

Primary files: `presentation/rap_demo/static/index.html`, `css/rap-demo.css`, `js/rap-demo.js`, `js/rap-plan.js`; new `js/rap-settings.js`; `infrastructure/rap/templates.py`, common mixing/preparation paths, browser integration tests.

- Add settings, mode/response controls, capture/echo state, and effective-change indicators.
- Make the current flow readable, retain existing diagnostics, and add the six `UNTESTED` variants.
- Wire drum/vocal levels to real audio preparation with peak protection, not merely UI state.
- Verify desktop/mobile layouts, keyboard focus, no accidental Space scrolling/capture in inputs, and disabled controls for unsupported capability/state.

### E5. Truthful Timing Diagnostics

Primary files: `application/rap/bar_audio.py`, `infrastructure/rap/phrase_warp.py`, `domain/rap/audio.py`, `application/rap/monitoring_payloads.py`, playback/projector code, `js/rap-demo.js`, `js/rap-demo-state.js`; diagnostics tests.

- Preserve requested/effective anchors and estimated drift/confidence through preparation and events.
- Render distinct requested, estimated, and observed-playhead layers.
- Test intentional anchor drift, missing diagnostics, older revisions, tempo changes, and device-clock observation delay.
- Keep heavy diagnostics outside the playback callback and bound event/UI payloads.

### E6. Mac/H200 Acceptance and Handoff

- Run the relevant full test suite and browser integration tests; keep fakes for deterministic edge cases, not as the final model demonstration.
- Host client/audio on the Mac and real models on H200, using GPU1 only after checking occupancy. Reuse the authorized service when appropriate; do not stop another person's work or start conflicting GPU processes.
- Exercise continuous mode, live topic/flow/BPM changes, manual Stop/Start, finite replay, device selection, gains, and alternating capture/echo/progressive responses.
- Run at least ten alternating turns, including empty input, timeout/failure injection, recapture, and a tempo change. Record primary/fallback/replay outcomes rather than hiding poor model performance.
- Compare submitted templates, generated/played revisions, waveform duration, and timing markers against logs; verify drum continuity except explicit Stop/device failure.
- Update the quick guide with exact local/SSH commands, capabilities, settings behavior, limitations, and rollback instructions. Preserve the prior working entry point/configuration.

## 9. Delivery Gates and Autonomy

After this design is approved, produce the detailed task-by-task implementation plan and execute milestone-by-milestone with narrowly scoped subagents/reviews. Routine internal naming, layout, and test-fixture decisions are delegated to the implementer. Deviations from the confirmed interaction cycle, truthful diagnostics, or backwards compatibility require explicit review.

Do not wait for theory results to complete E1-E6. Do not expand E1-E6 to implement every parameter in the research catalogue. A feature is complete only after its real Mac/H200 workflow and failure path are demonstrated, or an external blocker is precisely reported; unit tests alone do not establish live readiness.

The final handoff must distinguish implemented/tested behavior, remaining limitations, untested templates, theory-only dimensions, and any defaults chosen without a user preference.
