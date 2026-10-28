# Feature Delivery Progress

Status: delivered and extensively exercised, with the model-quality limitations below. Finalized 2026-10-22 07:12 America/Los_Angeles.

**Ready to use:** open `http://127.0.0.1:8018/` and press Start. The Mac client
is stopped in Continuous mode at 60 BPM, MacBook Pro Speakers selected, drums
0.776896 and vocals 0.80. H200 GPU1/GPU2 services and the SSH tunnel are left
running. For alternating exchange, change mode while stopped and apply it;
the default response length is two bars. Exact commands are in the
[current-session guide](../developer-guide/rap-demo-current-session-2026-10-22.md).

## Scope

Two separate modes: continuous model generation with user steering, and alternating one-bar user-flow capture followed by progressive model responses. This report is independent of the scheduling research report.

## What Is Implemented

- Continuous mode retains progressive model rap and ASAP topic/flow steering. Tempo and mix edits have explicit future-bar acknowledgments and immutable playback snapshots.
- Alternating mode captures one full bar of calibrated keyboard taps, quantizes to the sixteen-tick grid, submits a bounded response request, repeats the captured rhythm while waiting, and delivers model bars progressively. Echoes do not advance the logical response cursor.
- The GUI exposes separate modes, BPM, response length, stable audio-device selection, drum/vocal gains, readable stress, captured rhythm, requested/estimated onset layers, and played revision evidence. Six additional stress patterns are explicitly listening-UNTESTED.
- Lifecycle changes allow Start after Stop and automatic restart after a finite run. Reset fences old asynchronous work with a new epoch and retains accepted tempo/mix settings. Hardware restart explicitly completes PortAudio's prior callback-stop lifecycle.
- Existing robotic and remote renderer paths remain; custom dynamic controls are capability-gated. Source synthesis and alignment stay on H200; warp, mix, scheduling, and audio output run on the Mac.

## Design Choices

1. The first version captures exactly one full bar, then permits 100 ms for late HTTP receipt of an in-window tap. A drums-only transition bar follows capture because the current whole-bar queue must commit before the next downbeat. This is labeled `capture_transition`, not a successful model response or replay. It avoids silently dropping the last fraction of the input bar.
2. Capture represents onset rhythm. All newly captured stress targets initially equal 0.5; key velocity or multiple stress levels are not inferred from a spacebar.
3. The model starts when its first usable response bar can be committed; it does not wait for all response bars. The primary wait has a 30-second bound, then matching generated-text eSpeak may be used. Failures and fallback remain visible.
4. MOSS uses the existing verified in-process backend for this delivery. SGLang continuation is not claimed tested.
5. Timing graphics distinguish requested targets, source-alignment/warp estimates, and the audio clock cursor. There is no acoustic loopback measurement or new intelligibility claim.

## Live Environment

- Mac test client: `http://127.0.0.1:8018`, actual Core Audio / MacBook Pro Speakers, composite recording.
- H200 isolated deployment: `/data/home/Andrew.Yang/StreamMUSE/deploy/rap_research_20261022`.
- GPU1: Qwen2.5-7B-Instruct on remote 18611. GPU2: MOSS plus MMS and v3 service on remote 18620, forwarded to Mac 18620. GPU0 research service was stopped at 06:16:47.
- Each allocated GPU was checked empty before its service was launched. Existing deployment files and the old Mac client on 8012 are preserved.
- Raw Mac evidence: `logs/rap_feature_acceptance_20261022/`; real-model response artifacts and server logs stay in the isolated H200 root.

## Acceptance Evidence

### Final-Build Rerun

Parent full-suite run on frozen final sources: **2,563 passed, five
skipped**, 86.49 seconds. All bounded review findings were independently
reverified fixed. The audio repair received a fresh independent review with
125 passing focused tests; final GUI mode/status changes received 58 scoped
tests plus independent review. Neither review had remaining actionable findings.

| Final device-backed workflow | Result and evidence directory |
|---|---|
| Ten captured patterns plus empty input | Nine complete turns, 19 MOSS response bars; one four-slot turn ended after one response because the next lyric exhausted candidates. All 77 physical bars had zero underruns/device failures. `device_matrix_final/` |
| Actual GUI keyboard capture | Nine native Space events became nine slots; two MOSS responses followed two replays. Presets were correctly disabled and topic copy said next capture. `gui_keyboard_nine_final/` |
| Longer responses | Four- and eight-response turns completed, all 12 response bars MOSS, one replay each, zero underruns. `long_responses_device_final/` |
| Cancel then resume | Stop during capture and waiting, then Start without Reset: drums idle, zero stale responses after a five-second observation, active hardware, zero underruns. `cancel_device_final/` |
| Capture-time tempo edit | 60-to-90 preserved the four-second input window, delivered two responses, played 128,000-frame bars and retained 90 BPM after Reset. `capture_tempo_device/` |
| Finite replay | Two two-bar runs using Start alone, each exactly 384,000 stereo frames, active hardware and zero underruns. `finite_device_replay/` |
| Device changes | GUI selected ASUS output, played, stopped, restored Mac speakers and resumed. Both stream identity and monitor device label matched. `final_handoff_controls/` |
| Final steering | Continuous preset control re-enabled; city-gardens / UNTESTED staggered rising reached `remote_latest` MOSS playback at API bar 16. `final_handoff_controls/final_steering.json` |

The final handoff health audit at 07:11:25 confirmed ready H200 models,
the active tunnel and a stopped Mac hardware backend without errors. The last
continuous epoch played 19 MOSS bars and recorded exactly 3,648,000 frames,
peak 0.9172, with zero underruns. Its queued twentieth bar is not counted as
played or included in the WAV. See `final_handoff_controls/handoff_readiness.json`.

The final matrix WAV contains exactly **14,784,000 frames** at 48 kHz,
all finite, peak 0.8410. Its committed-bar frame sum matches exactly. It includes
21 drums-idle bars, 11 transition bars, 26 flow echoes and 19 MOSS response bars;
these categories are not conflated into a model-success count.

Final GUI screenshots are in `logs/rap_feature_acceptance_20261022/screenshots/`:
`alternating-captured-flow.png`, `continuous-stress-lyrics.png`, and
`requested-rendered-timing.png`. The parent visually checked the live 546-pixel
wide view and found no overlapping controls/text. Browser warning/error log was
empty. Earlier worker visual checks do not establish final desktop-wide or
mobile-device acceptance; this final evidence is the actual narrow in-app view.

Full-app hardware lifecycle evidence in `device_lifecycle_final/summary.json`
passed three manual Stop/Start cycles (24-to-25, 26-to-27, 28-to-29) and Reset
recovery. Every resumed run reported active `sounddevice`, no device error and
zero underruns. The component owner separately exercised eight real speaker
starts across four Reset epochs. These are device-execution checks, not
independent acoustic measurements. A first parent observer used the wrong
state label (`playing` instead of `running`); that harness timeout is retained
and is not classified as a product failure.

- Final continuous source session `rap-20261022T132904Z-d739c78b`: 52 bars through
  the physical device before manual Stop. Of these, 51 used MOSS (including two
  older revisions during an ASAP edit) and one used matching generated-text
  eSpeak. Zero underruns or rejected remote fetches in this interval.
- GUI 60-to-50 ACK: 504.9 ms, effective API bar 6, 230,400 frames at 48 kHz.
  GUI 50-to-90 ACK: 938.6 ms, effective API bar 11, 128,000 frames. Server control
  processing was about 1.9/1.5 ms, respectively. Both reached actual playback;
  the header displayed played rather than requested BPM.
- GUI gains 0.90 drums / 0.60 vocals reached the played mix revision. The
  city-gardens / syncopated-rising edit showed older audio first, one new-text
  eSpeak bar, then new-topic MOSS. This is best-effort ASAP, not a promise that
  the requested bar can always contain freshly rendered audio.
- **Fixed device defect:** manual Stop/Start previously resumed scheduling but
  switched the composite sink to timed/no-speaker fallback. PortAudio was
  inactive but not stopped after `CallbackStop`; host-side `stop()` is now
  performed before restart. Composite output preserves the failure notice and
  error, keeps device selection available, and retries hardware after Reset.
- The saved `final_continuous_before_reset.wav` contains 53 started bars,
  including that last timed resumed bar: exactly 7,680,000 stereo frames at
  48 kHz, all finite, peak 0.8829. Its expected frame sum matches exactly. The
  audit explicitly distinguishes the 52 device bars from the timed resumed bar.
- Reset retained accepted 90 BPM and 0.90/0.60 mix at new bar zero, with the
  correct 128,000 frames. This particular Reset check used the detected timed
  fallback and is **not** counted as speaker acceptance.
- Capturing while changing 60-to-90 BPM preserved the full original four-second
  input window and delivered two responses with zero underruns. This separate
  `capture_tempo_logic` run also used timed output, explicitly labeled.

The live harness now requires `audio_output.backend == sounddevice`,
`audible == true`, and no device error by default, in addition to zero underruns.
Five regressions prevent a timed/null/failed sink from passing that gate; a
sixth prevents arming while the controller has started but audio is still priming.
`--allow-silent` is an explicit logic-only override.

### Earlier Tests And Retained Failures

| Workflow | Current evidence |
|---|---|
| Continuous real-model start | Four consecutive MOSS-selected lyric bars played at 60 BPM before the tempo test; exact 192,000-frame stereo bars at 48 kHz. |
| Live tempo edit | Initial one-second ACK failure caused an underrun. A later five-second ACK failure safely continued the confirmed tempo. Bounded ten-second ACK/retry repair then passed actual 60-to-50-to-90 playback, detailed above. |
| Mode switch while stopped | GUI continuous-to-alternating succeeded with a new epoch. |
| Empty captured bar | Returned to drums idle with visible `empty_capture`; drums continued. |
| Actual GUI keyboard capture | Six Space events became five distinct slots, one collision explicitly recorded. Two rhythm-replay bars preceded two MOSS response bars. No underruns in that alternating epoch. |
| Repeated varied turns | `alternating_matrix_v3`: ten rhythms plus empty capture, including 2-16 distinct slots and late tick-15.8 taps. All ten delivered two response bars, zero underruns. Synthetic timed input is separate from the actual GUI keyboard test. |
| Longer progressive responses | Reviewed controller: four- and eight-bar turns both completed, zero underruns. Combined delivery was seven MOSS bars and five generated-text eSpeak bars. This proves continuity/progression, not all-MOSS reliability. |
| Cancel capture / waiting | Stop during `CAPTURING` and `WAIT_REPLAY`, followed by Start without Reset, returned to drums idle with zero delivered stale responses. |
| Finite-session replay | Two consecutive three-bar runs using Start alone; six of six bars used MOSS, with bar indices 0/1/2 each run. WAV/timed sink, not physical speakers. |
| ASAP topic/flow edit | An early run had multiple HTTP disconnects and eSpeak fallbacks before recovery. The final transport build had no rejected fetches in 52 device bars; one new-text eSpeak bar bridged the edit before MOSS recovered. Exact tunnel cost was not isolated. |
| Output device | GUI switched from MacBook speakers to ASUS monitor output; Core Audio stream opened with zero underruns. The stale displayed device name was fixed and rechecked in both directions. No acoustic loopback claim. |

The first repeated-turn attempt is retained: a four-slot response generated one
bar, then exhausted candidate attempts and hit the bounded response timeout.
Drums/replay continued with zero underruns. Later matrix success does not erase
that failure or establish a zero-failure probability for arbitrary tapped flows.

Evidence directories under the raw Mac root include `alternating_matrix_v1`
(failed attempt), `alternating_matrix_v3`, `finite_replay`, `response_lengths`,
and `cancel_turns`. The latter two use the reviewed interaction implementation.
Canonical `capture_finalized` events now preserve raw calibrated taps, collisions,
rejection counts, quantized flow and capture timing in the session recorder.

The stored HTTP evidence confirms `sounddevice` during all ten matrix turns and
both four/eight-response runs. The cancellation suite later encountered the same
manual-resume timed fallback; its state-machine assertions passed, but its later
iterations are not physical-device acceptance. WAV files reset per epoch, whereas
events remain cumulative; archived copies must be taken before a reset/rebuild.

Generated content is currently unfiltered. One test line contained a slur; no
classroom-safe content guarantee is made. A separate clean-demo preference was
sent to the user and has not been answered. Generation was left unchanged.

### Remaining Limits

- Custom rhythms can fail lyric generation. A separate seven-key/six-slot GUI
  turn delivered its first MOSS response but no second line. A bounded replay
  reproduced a prompt/novelty-gate mismatch: candidates repeated a forbidden
  three-word opening. Both 32 and 64 initial choices failed the short-flow
  diagnostic fixtures; a budget-only increase was not justified. See the
  [separate diagnostic](../research/2026-10-22-feature-six-slot-diagnostic.md).
  Original failed candidate ledgers were not persisted by the exception path,
  so the replay establishes a mechanism, not the exact original sampled strings.
- Use roughly eight or nine distinct taps for an initial showcase, while
  retaining visible errors and replay. This is practical guidance from these
  trials, not a guaranteed operating region or a validated optimal density.
- Gentle sparse R3 preserves speech quality at the expense of exact internal
  syllable anchors. The GUI exposes estimated drift, sometimes hundreds of
  milliseconds; it does not claim every syllable hits its requested tick.
- Timing diagnostics are estimates and stream-clock observations, not acoustic
  loopback measurements. Hardware execution does not prove intelligibility.
- Live BPM changes reserve the ten-second ACK budget plus preparation margin;
  they are future-boundary updates, not immediate tempo ramps. Topic/flow uses
  a separate best-effort ASAP policy.
- Device/mode/response changes require stopped playback. Apply device changes
  separately, and live tempo separately from mix changes. The guide explains
  these operation groups and how to recover from unknown ACK/device outcomes.

Earlier full-suite failures included finite-run expectations, huge-integer
validation and a stale enum-list assertion. All were repaired before the final
green suite. Focused counts overlap and are not summed.

Independent reviews covered interaction, timing/audio, runtime, GUI/assembly
and research. Repairs include ordered commit replay, finite response bounds,
delivered-only history, capture evidence, immutable settings reads, truthful
ACK/timeout handling, preserved scripted edits after rebuild, and device
restart/fallback recovery. All reported findings are closed with regressions.

## Delivery Gates

1. Passed focused unit tests for each owned component.
2. Passed the integrated full suite and GUI contract tests.
3. Exercised actual Mac client/H200 models, delayed responses, edits, keyboard input, cancellation, replay and device resume.
4. Completed independent review and documented retained failures/limitations.

Work remains in the isolated `.worktrees/real_rap` checkout on
`feature/rap_optimization`, based on `bde2f62c`. No final commit or push was made;
the unrelated primary piano checkout was not modified. Temporary test clients
were closed, the old 8012 client was left untouched, and only the current 8018
client plus required H200 services remain from this feature trial.

## Deadline

Target 07:30, hard cutoff 08:00 Los Angeles wall-clock time (PDT on this date). The user permits earlier completion only after both tracks and reasonable extensive verification are finished. No usage-limit reset has been used.
