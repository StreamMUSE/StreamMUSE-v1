# Default Late Note-On Drop Policy

## Scope

Branch: `codex/unify-late-note-on-drop`.
Parent: `660f363b` (`codex/fix-pc-beat-tail-delivery`), the production
P+C beat-tail and Stop-boundary fixes used for the current formal results.

This commit changes only the ordinary `RealTimeMusicService` playback planner.
It does not change model inputs, raw generation, checkpoints, seed handling,
quantization, client request timing, the scheduler implementation, offline
inference or Legacy M2A. No experiment outputs are replaced or relabeled.

## Policy

`_plan_model_events_for_playback()` now drops a note-on when
`note_on.tick < current_tick`, whether or not its note-off is already known.
An unplayed paired note's note-off is dropped with it. Open late note-ons
are not moved to the current tick. Their later isolated note-offs are ignored
when no corresponding note is active.

Current-tick and future note-ons retain their original ticks. An isolated
late note-off for an already sounding note is still emitted at the current
tick to close it. Current/future note-offs and same-tick off-before-on order
are unchanged. This cleanup is not onset recovery.

The existing P+C default already drops late note-ons with recovery disabled;
its production implementation is unchanged. The new P+C regression test
checks that default and verifies that on-time note-on/off ticks are preserved.
Future orphan note-offs can still pass through P+C's internal event queue;
they do not constitute a recovered note-on or an audible note by themselves.
No new UI, CLI switch or environment variable is added. Existing explicit
P+C recovery overrides are outside this change; formal runs keep them off.

## Trace Compatibility

- Dropped partial/open notes use `dropped_late_note_on` in the schedule trace.
- Both events of a dropped partial note carry that policy to identify the
  reason for dropping the pair; their original logical ticks remain recorded.
- Fully past paired notes retain `dropped_past_note`.
- The historical `clamped_onsets` counter remains readable and is now zero
  for ordinary playback; dropped-note counts include rejected late onsets.
- The opt-in realtime/offline consistency test recognizes the new drop policy
  as a delivery issue rather than silently overlooking it.

## History

The prior implementation is retained in the parent commit. Git traces its
partial/open-note clamping to Stanley's `1085e1b0f` (partial-note recovery).
If finite-duration partial recovery is wanted later, review and restore that
specific logic on a separate branch instead of reverting unrelated fixes.
No backup implementation or second runtime mode is introduced here.

## Validation

Before the production edit, the six selected new regression cases failed
against the old clamping behavior. After the edit:

```text
python -m pytest tests/unit/application tests/unit/infrastructure/output -q
239 passed
```

CPU-only local tests, with `CUDA_VISIBLE_DEVICES` empty. Coverage includes:
- Late paired notes whose end is still in the future; fully expired notes.
- Late open notes and their subsequent orphan note-offs.
- Onsets exactly at the current tick and in the future.
- Active late note-off cleanup and current-tick same-pitch retriggering.
- The observed test40 tick 40-to-41 and 88-to-89 onset cases.
- No mutation of the supplied raw events.
- Ordinary/P+C service, session cleanup and MIDI-output regressions.

The test40 examples are planner fixtures based on the audited events, not
fresh GPU inference or a replay of all native traces. Full event-vs-MIDI
acceptance and formal music/system metrics have not been rerun for this
commit. Existing result folders still describe their original source versions.
