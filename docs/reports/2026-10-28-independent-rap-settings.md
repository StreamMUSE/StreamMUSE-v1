# Independent Rap Settings

## Behavior

The Mac client owns playback gains, chord loops, and MOSS timing/stress rendering.
H200 still owns lyric selection, source speech, and source alignment. A local
control request crosses browser-to-Mac HTTP only; it does not request new model
output or wait for remote capability discovery.

| Controls | Activation | Server interaction |
| --- | --- | --- |
| Drum/vocal/chord gain | Audio callback, with 10 ms smoothing | None |
| MOSS alignment policy | Earliest uncommitted bar after local re-warp | None; retained source speech is reused |
| Chord loop | Eligible bar boundary | None |
| Tempo | Existing acknowledged future-bar schedule | Required |
| Topic/flow | Existing ASAP revision behavior | Required |
| Mode/response length/output device | While stopped | Mode/response capability checks; device is local |

Gain changes deliberately affect the current sound. Alignment and chord
structure never replace an already committed bar. Three independent settings
groups have their own pending/acceptance evidence, so a slow generation change
cannot disable live audio controls.

## Technical Changes

- Group-specific settings routes supplement the legacy endpoint. Separate group
  locks protect writes; only changed fields are merged into shared settings.
- Capability discovery is cached and refreshed off the request path. Generation
  validation can wait for H200, but local validation and responses do not.
- Alignment controllers retain bounded source packages and version local warp
  work by session, policy, and timing. Obsolete work cannot replace newer audio.
- Live mixing uses raw stems, bounded 1024-frame blocks, smoothing, and peak
  protection. Completed composite WAV bars contain exact callback PCM, not the
  original premixed buffers.
- Stop/reset/controller-rebuild fencing prevents in-flight independent settings
  from being lost. Stopped rebuilds rebase on the latest accepted values.
- The GUI auto-applies local edits, coalesces gain updates, and serializes
  alignment separately. Tempo and session setup have independent submissions.
- Per-control availability labels distinguish immediate gains, next-eligible-bar
  rendering, server coordination, and stopped-only setup. A poll started during
  a settings POST cannot overwrite that POST's accepted values or schedule.
- Rebuilt controllers synchronize retained device gain targets. Exchange echo
  percussion is not treated as speech when the vocal slider changes.

## Live Evidence

Initial run against the Mac SoundDevice client and existing H200 GPU0/GPU1
services: `output/rap_settings_20261028/acceptance/summary.json` and `http.jsonl`.

The repeated run is in `output/rap_settings_20261028/final_acceptance/`.

| Test | Initial run | Repeated run |
| --- | --- | --- |
| Continuous output | 10/10 committed lyric bars MOSS, zero underruns | 10/10 MOSS, zero underruns |
| Strict timing edit | 2.26 ms acceptance | 3.31 ms acceptance |
| Strict + stress edit | 3.24 ms acceptance | 12.86 ms acceptance |
| Relaxed edit | 3.89 ms acceptance | 3.18 ms acceptance |
| Concurrent gain / tempo response | 84.03 / 1063.60 ms | 129.82 / 510.01 ms |
| Exchange response | 2/2 MOSS; relaxed then strict + stress | 2/2 MOSS; relaxed then strict + stress |
| Stop/Start without Reset | Device output confirmed | Device output confirmed |

In both runs, all three alignment edits changed exactly the next uncommitted
bar (bars 1, 2, 3); previous committed text and warp evidence remained unchanged.
The local gain request finished while the tempo request was still pending.
Chord-loop and tempo changes reached playback. Topic/flow requests were accepted,
but the scripted ten-second window did not confirm matching playback; see the
trace correction below.
A final post-review smoke test confirmed the stopped bulk-update gain fix on
the real sink, three further MOSS commits, matching callback gain targets, and
zero underruns.

### Topic/Flow Trace Correction

The subsequent response-latency audit found a gap in this acceptance script:
after the topic/flow POST it sleeps ten seconds and checks only underruns, not
whether the requested revision played. Its overall `passed` status must not be
interpreted as validation of timely topic/flow application.

| Time from edit POST start | Initial run | Repeated run |
| --- | --- | --- |
| HTTP acceptance | 4.13 ms | 4.40 ms |
| Changed bar-8 lyric received | 0.788 s | 0.438 s |
| Changed bar-8 eSpeak render ready | 1.514 s | 1.236 s |
| Bar 8 starts with old topic/flow | 4.326 s | 4.340 s |
| Bar 9 starts with old topic/flow | 8.693 s | 8.691 s |

The commit preference is latest MOSS, older MOSS, then latest streamed eSpeak.
Consequently, fresh edited lyrics/audio can be ready while old MOSS still wins
the next playback slot. At the tested 55 BPM, each missed bar adds 4.364 seconds.
These event times are software scheduling observations, not acoustic recordings.
They were obtained by deduplicating `recent_events` by sequence and comparing
their monotonic timestamps with the recorded HTTP request start. No matching
played revision was present in either ten-second snapshot. Later matching
playback occurred after the scripted Stop/Start, so it does not establish an
uninterrupted live-edit latency.

The next acceptance revision should explicitly await a played bar whose topic,
flow, and revision match the edit. No responsiveness policy was changed in this
audit; this correction supersedes the earlier claim that all edits reached
playback within the observed continuous run.

Bar numbers in this report are zero-based; GUI bar numbers are one-based.
Alignment response times acknowledge scheduling, not completion of the warp or
the time until the next bar is heard. This is a short integration run, not a
latency percentile benchmark.

## Verification Status

Implementation, two live acceptance runs, final review fixes, and browser
availability checks are complete.

- Full suite: **2863 passed, 5 skipped**, 36 warnings, 114.19 seconds.
  Command: `.venv/bin/python -m pytest tests/ -q --tb=short`.
- JavaScript syntax check and `git diff --check`: passed.
- Browser interaction verified local auto-apply during playback, committed-bar
  alignment protection, independent pending states, Stop/Start, and stopped-only
  controls. Reload preserved played confirmation. Screenshots are under
  `output/rap_settings_20261028/`.
- Mac client remains at `http://127.0.0.1:8018/`, stopped, continuous, 60 BPM,
  relaxed alignment, drums 0.78, vocals 0.8, chords off. H200 Qwen/MOSS/MMS health
  is ready through the existing SSH forward. No commit or push was performed.

## Limits

- Alignment uses forced-alignment estimates; strict timing is not an acoustic
  measurement or a guarantee of natural pronunciation.
- An edit too close to commitment can still encounter the existing deadline
  fallback behavior. Already committed audio is never rewritten.
- Live gain evidence is callback PCM, not microphone verification. Unsupported
  or missing stems do not fabricate applied-gain evidence.
- Completed-bar WAV recording omits incomplete bars and trailing silence.
- No H200 model/server restart was needed for these client-only changes.
- A longer manual playback run reported a lyric-writer `no_valid_candidates`
  error at bar 8 and transient retried audio transport failures; later MOSS bars
  continued. These tests validate settings behavior, not failure-free lyric
  generation or higher-BPM capacity.
