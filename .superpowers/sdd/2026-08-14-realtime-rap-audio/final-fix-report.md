# Final Realtime Rap Audio Fix Report

Date: 2026-08-15

Branch: `feature/real_rap`

Fix-wave base: `7648c5e1d4119cd9ba97ddf37b99dfd9c18243a9`

Commit: the containing `fix(rap): close final realtime audio review gaps` commit

## Outcome

All 12 Important findings and all four Minor findings from
`final-review-findings.md` are resolved. Text-only startup and behavior remain
unchanged. No acceptance assertion was weakened. There are no unresolved
review items.

## Important Findings

1. Lifecycle Stop now calls the restartable local-chat `abort`; permanent
   client close is reserved for runtime Close.
2. Audio selection uses real monotonic absolute commitment deadlines. A
   primary completed after its deadline cannot replace the ready fallback,
   and commit slack is measured from the actual commit time.
3. Stop/Start uses `discard_pending` rather than Reset. Composite WAV output
   keeps all completed audio and appends continuation bars.
4. The WAV sink patches and flushes RIFF/data sizes after completed bars and
   Stop, so the artifact is readable before permanent Close.
5. Timed playback counts one underrun per starvation episode, preserves its
   absolute sample clock, and causes the composite WAV to materialize silence
   for absolute gaps before late bars.
6. Reset quiesces the playback observer and fences old-epoch poll effects
   before coordinator/controller state advances.
7. Sink Start has an explicit success/failure result. Pure sound-device
   failure leaves playback stopped and publishes the failure; composite mode
   switches coherently to the timed fallback while retaining WAV recording.
8. Monitoring and terminal projections keep authoritative playback lifecycle
   state separate from render pipeline state, so future rendering cannot
   disable valid controls.
9. Browser state reduction rebases the bounded event cache on
   `SESSION_RESET`; old bars and ticks cannot repopulate the new epoch.
10. Non-final vocal tails are explicitly cropped at the bar boundary and emit
    truthful rendered-frame, cropped-frame, and forced-fit diagnostics.
11. Stop while priming preserves bar zero as the restart successor, making an
    immediate Start/Stop/Start sequence valid before playback begins.
12. A finite non-looping scenario reaches a terminal Stop without reserving an
    out-of-range successor. Restart is rejected until Reset establishes a new
    scenario epoch.

## Minor Findings

- eSpeak execution has a timeout and stdout bound; synthesized PCM uses a
  thread-safe bounded LRU cache.
- Syllable telemetry and consumers use canonical `stress`, `subdivision`, and
  `observation_delay_ms` fields without describing observation delay as
  physical jitter.
- Audio CLI configurations reject sample rates other than 48 kHz before audio
  dependencies are constructed; text-only mode is not restricted.
- The stale acceptance-report follow-up now reflects implemented scheduled
  playback.

## TDD Regressions

The fix wave began with deterministic RED coverage. After two test-import
mistakes were corrected, the selected regression run had 19 expected behavior
failures before production changes. The corresponding GREEN coverage includes:

- `test_real_local_chat_client_abort_is_restartable_across_audio_stop`
- `test_primary_audio_completed_after_absolute_deadline_cannot_beat_ready_fallback`
- `test_audio_commit_uses_absolute_deadline_and_reports_measured_negative_slack`
- `test_composite_stop_start_continuation_preserves_completed_wav_prefix`
- `test_float32_wav_is_header_patched_and_flushed_after_each_completed_bar`
- `test_timed_starvation_counts_once_and_wav_records_absolute_silence_gap`
- `test_runtime_reset_quiesces_old_tick_delivery_before_controller_epoch_reset`
- `test_composite_device_start_failure_falls_back_to_timed_recording_coherently`
- `test_failed_sink_start_reports_failure_without_entering_running_state`
- `test_successful_start_preserves_an_immediate_bar_started_notice`
- `test_future_render_events_do_not_overwrite_authoritative_playback_lifecycle`
- `test_browser_event_cache_rebases_on_session_reset`
- `test_nonfinal_tail_cropped_at_bar_boundary_has_truthful_warning_and_lengths`
- `test_immediate_start_stop_start_before_bar_zero_replays_bar_zero`
- `test_finite_non_looping_scenario_stops_terminally_and_requires_reset_to_restart`

Minor regressions cover bounded eSpeak execution/LRU eviction, canonical
syllable telemetry and recorder projections, and pre-construction 48 kHz CLI
validation.

## Verification

```text
Focused changed regression modules plus server lifecycle:
323 passed in 4.05s

Playback lifecycle module after final Start contract review:
30 passed in 1.04s

Complete rap suites:
534 passed, 1 warning in 4.28s

Full repository:
1169 passed, 4 skipped, 1 warning in 30.47s

Ruff over all 30 changed Python files:
All checks passed

git diff --check:
passed
```

The warning is the existing third-party `pretty_midi` import of deprecated
`pkg_resources`. The four full-suite skips are environment-dependent and were
not introduced by this wave.

Bounded device-free artifact smoke:

```text
uv run streammuse-rap-demo --audio-output wav --generator phrase_bank \
  --max-bars 1 --no-web --terminal-layout stream --terminal-detail summary \
  --tempo 240 --log-dir /tmp/streammuse-rap-final-smoke.gkRdoD \
  --audio-file /tmp/streammuse-rap-final-smoke.gkRdoD/smoke.wav

exit: 0
format: RIFF/WAVE IEEE Float, stereo, 48000 Hz
file bytes: 384044
patched data bytes: 384000
frames: 48000
```

The H200 was not restarted, as remote acceptance was already bounded and
documented. No physical sound device was needed for this fix wave.

## Files Changed

```text
docs/developer-guide/rap-acceptance-report-2026-08-09.md
src/streammuse/application/rap/audio_coordination.py
src/streammuse/application/rap/audio_service.py
src/streammuse/application/rap/bar_renderer.py
src/streammuse/application/rap/monitoring.py
src/streammuse/application/rap/playback.py
src/streammuse/application/rap/realtime.py
src/streammuse/application/rap/runtime.py
src/streammuse/domain/rap/audio.py
src/streammuse/infrastructure/rap/audio_output.py
src/streammuse/infrastructure/rap/recorder.py
src/streammuse/infrastructure/rap/speech.py
src/streammuse/presentation/rap_demo/cli.py
src/streammuse/presentation/rap_demo/server.py
src/streammuse/presentation/rap_demo/static/index.html
src/streammuse/presentation/rap_demo/static/js/rap-demo-state.js
src/streammuse/presentation/rap_demo/static/js/rap-demo.js
src/streammuse/presentation/rap_demo/terminal_state.py
src/streammuse/presentation/rap_demo/terminal_stream.py
tests/integration/test_rap_demo_browser_reducer.py
tests/integration/test_realtime_rap_audio.py
tests/unit/application/rap/test_audio_coordination.py
tests/unit/application/rap/test_bar_renderer.py
tests/unit/application/rap/test_monitoring.py
tests/unit/application/rap/test_playback.py
tests/unit/application/rap/test_realtime.py
tests/unit/application/rap/test_runtime.py
tests/unit/infrastructure/rap/test_audio_output.py
tests/unit/infrastructure/rap/test_recorder.py
tests/unit/infrastructure/rap/test_speech.py
tests/unit/presentation/rap_demo/test_cli.py
tests/unit/presentation/rap_demo/test_terminal_dashboard.py
tests/unit/presentation/rap_demo/test_terminal_state.py
tests/unit/presentation/rap_demo/test_terminal_stream.py
.superpowers/sdd/2026-08-14-realtime-rap-audio/final-fix-report.md
```
