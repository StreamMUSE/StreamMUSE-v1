# Live Tempo Updates: Earliest Feasible Uncommitted Bar

## Problem And Fix

Continuous-mode tempo updates previously reserved the entire **10-second
acknowledgement failure timeout** before choosing a destination bar. That timeout
was a failure threshold, not measured request latency. One previous edit asked
for internal bar 6 but was scheduled at bar 9, despite a 686 ms acknowledgement
and 1.8 ms server processing time.

The client now searches from the first uncommitted bar and chooses the earliest
boundary with enough normal coordination time. The current bar and queued,
committed audio remain immutable. No frontend controls or synthesis policies
were removed or changed.

Define:

- `A_last`: elapsed time of the preceding successful tempo negotiation.
- `L`: normal coordination budget, `min(10 s, max(2 s, 2 * A_last))`.
- `C_b`: commit deadline of bar `b`, one tick before its playback boundary.
- `M`: existing 250 ms Mac preparation margin.

The selected bar is the first uncommitted `b` for which
`C_b >= request_time + L + M`. The full 10-second request timeout remains.
This is the earliest **feasible under the coordination budget** bar, not a promise
that every edit can reach the immediately following bar.

## Late-Reply Safety

Simply shortening the lead exposed a real regression in delayed-ACK tests:
an acknowledgement could arrive after its destination audio was committed.
The new negotiated `tempo_reschedule_v1` capability handles that case:

1. Keep playing the confirmed old timeline while waiting for acknowledgement.
2. If the first reply misses its safe boundary, send one later reschedule using
   the conservative timeout-sized budget.
3. The server replaces the speculative switch against the saved pre-edit
   timeline. It does not leave the missed switch active in between.
4. Install only the safe acknowledged revision on the client.

Lost replies or an unsafe second reply retain the existing local-fallback and
Reset-required recovery. Older servers without this capability retain the
conservative ten-second guard. The capability is additive to protocol v3.

## Real Mac / H200 Results

Mac client: 8018, native sounddevice output to `ZoomAudioDevice`, Opus transport,
`all_onsets_r3`, drums/vocals/chords retained. Actual Qwen and optimized
MOSS-Local/SGLang ran on H200 through the SSH forward, not mocks.

Bar numbers in the artifacts are zero-based.

| Change | Request Phase | Bars Ahead | ACK | Changed-Bar Playback After Request |
|---|---|---:|---:|---:|
| 90 -> 75 BPM | Early | 1 | 459 ms | 2.625 s |
| 75 -> 90 BPM | Late | 2 | 241 ms | 3.745 s |
| 90 -> 60 BPM | Early | 2 | 368 ms | 5.250 s |
| 60 -> 90 BPM | Early | 1 | 285 ms | 3.866 s |

All four destination bars actually started and completed with MOSS vocals.
Current frozen-bar payloads stayed unchanged, destination callback frame counts
matched the requested tempos exactly, and no underruns were reported. All 12
frozen bars in this run used `moss_lyric_stream`. The early two-bar case illustrates
why a short coordination guard can still miss the immediately following bar;
the next bar is not unconditional.

An additional alternating-mode test changed 75 -> 90 BPM at the next bar,
captured an offbeat user flow, replayed it once, and delivered both model-response
bars with zero underruns or generation errors. Alternating mode retains its
existing logical-response-destination timing behavior.

These are callback/playback-event measurements, not microphone or listener
verification. Browser automation could not attach in this run; live controls
were exercised through the same Mac HTTP endpoints used by the GUI.

## Verification And Artifacts

- Full suite: **3,027 passed, 5 skipped**, 105.42 seconds, with loopback networking
  enabled. The sandbox-only run blocked four local HTTP-server tests; the full
  permitted rerun passed them.
- Real WebSocket regression coverage includes fast, delayed (1.15 s and 5.25 s),
  lost, and obsolete acknowledgements; committed-bar protection; old-server
  compatibility; server reschedule restoration; and timeline revision jumps.
- `git diff --check`: clean.
- Repeatable live collector: `scripts/experiments/rap_tempo_acceptance.py`.
- Evidence root: `output/rap_tempo_asap_20261031/`.
- `live_acceptance_final/summary.json`: four-edit results and timing snapshots.
- `live_acceptance_final/http.jsonl`: request/response evidence.
- `live_acceptance_final/tempo_changes_mix.wav`: completed callback PCM.
- `alternating_regression/`: exchange results and separate callback recording.
- `pytest-network.xml`: complete regression results.
- `before_fix_user_recording.wav`: preserved user recording from before restart.

Two earlier collector attempts stopped because the collector initially read the
wrong response/payload keys. Their directories remain for transparency; use
`live_acceptance_final` for the completed four-edit test.

## Deployment And Final State

Only the H200 render service was restarted; Qwen and MOSS stayed running. Source
backups are in
`/data/home/Andrew.Yang/StreamMUSE/deploy/rap_mosslocal_20261028/tempo_asap_20261031_backup`.
An unrelated process appeared on GPU1, so the identity-checked render/alignment
service was placed on GPU0 alongside our owned Qwen process, with ample memory.
MOSS remained on GPU1. No unrelated process was stopped. `render.pid.json`
records the actual placement and preflight identities; the standard launcher's
new-start default remains GPU1 and its occupancy checks must not be bypassed.

The Mac client, SSH forward, and H200 services are left available. Playback is
stopped, with the user's **90 BPM, continuous mode, ZoomAudioDevice**, gain values,
minor-cycle chords, and strict alignment restored. No commit or push was made.
