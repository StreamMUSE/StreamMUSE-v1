# Optimized MOSS-Local Restoration

## Scope

Restored MOSS-Local Transformer v1.5, SGLang-Omni 0.1.8, SGLang 0.5.21,
and the existing continuation patch on H200 GPU1. Qwen2.5-7B remains on
GPU0 with 32 candidates. The current feature source, local alignment policies,
curated flows, chords, independent settings groups, and v3 playback selection
policy were not changed. The Mac still owns its clock, final warping, mixing,
and audio device. See the [operating guide](../developer-guide/rap-mosslocal-live-2026-10-28.md).

The collaborator's historical environment was not accessible. An isolated
environment was built under Andrew's own deployment directory instead; this is
not a claim that the entire old environment was reproduced byte for byte.
The current reference voice was retained, with its published matching transcript.
Model and codec checkpoints were materialized because this Transformers version
resolved Python symlinks into cache blobs and lost sibling imports. MPS is
explicitly disabled to keep GPU process ownership and shutdown scoped; CPU
workers use eight threads. The older in-process deployment is retained.

## Measurement Method

Each run starts the real Mac audio device and waits for two completed latest-MOSS
bars. It makes six topic-and-flow edits, one per curated pattern, including the
two leading-rest flows. It waits for the exact edited lyric ID/revision to be
committed, started, and fully played before advancing. Subsequent edits follow
completion by two seconds. Timeouts, fallback choices, and underruns are retained.

"Vocal onset" below means the Mac playback-callback timestamp plus the first
requested onset offset, validated against completed callback frames. It is
**not** microphone evidence or a hardware DAC timestamp. WAV files preserve the
completed device callback PCM. No subjective intelligibility judgment is claimed.
Six edits per condition are integration evidence, not a reliable p99 estimate.
The traffic and generation content were not controlled enough for causal codec
or BPM comparisons.

## PCM Trial

| Tempo | Edits played | Edit-to-vocal onset, mean | Median | Range | Underruns |
|---|---:|---:|---:|---:|---:|
| 60 BPM | 6/6 | 7.01 s | 6.92 s | 1.91-13.92 s | 0 |
| 90 BPM | 6/6 | 5.38 s | 5.86 s | 3.15-8.06 s | 0 |

First matching edited voices: 11 MOSS, one generated-lyric eSpeak. Across all
played bars, including warmup and waiting: 30 latest MOSS, four older MOSS,
nine generated-lyric eSpeak, and four local-lyric eSpeak. These are not
all-MOSS or all-model-lyric demonstrations, despite all edits eventually passing.

| Stage | 60 BPM median / maximum | 90 BPM median / maximum |
|---|---:|---:|
| MOSS synthesis | 456 / 867 ms (25 results) | 339 / 401 ms (20 results) |
| Forced alignment | 16 / 31 ms | 15 / 29 ms |
| Blocking audio GET (`transfer`) | 2,689 / 5,527 ms | 2,077 / 7,260 ms |
| Mac decode/warp | 89 / 151 ms | 84 / 120 ms |

MOSS synthesis means were 478 ms and 332 ms respectively. Faster synthesis at
90 BPM here is not evidence that tempo itself speeds up the model: text, source
duration, warmup, caching, and request mix also differ. The earlier in-process
trace showed roughly 2.3 seconds of MOSS synthesis, but was not a paired trial.

## Opus Trial And Current Configuration

The existing Opus transport was then enabled on both ends, with encoder
complexity five. Models, source, candidate count, and playback policy stayed
unchanged. This is the configuration left running.

| Tempo | Edits played | Edit-to-vocal onset, mean | Median | Range | Underruns |
|---|---:|---:|---:|---:|---:|
| 60 BPM | 6/6 | 6.33 s | 6.12 s | 1.82-9.89 s | 0 |
| 90 BPM | 6/6 | 4.94 s | 4.51 s | 2.71-8.81 s | 0 |

Again, 11 edited topics/flows first arrived in MOSS and one in generated-lyric
eSpeak. All 24 edits across both transports completed without an underrun.
Opus's full playback contained 29 latest-MOSS, nine older-MOSS, two
generated-lyric eSpeak, and five local-lyric eSpeak bars. Thus seven of 45 bars
used eSpeak, and five were local-lyric fallbacks. Zero underruns does **not**
mean zero fallback or immediate responsiveness.

| Stage | 60 BPM median / maximum | 90 BPM median / maximum |
|---|---:|---:|
| MOSS synthesis | 460 / 476 ms (26 results) | 347 / 367 ms (23 results) |
| Forced alignment | 17 / 31 ms | 15 / 26 ms |
| Blocking audio GET (`transfer`) | 1,096 / 8,339 ms | 1,010 / 5,191 ms |
| Mac warp | 84 / 165 ms | 79 / 161 ms |
| End-to-end audio request | 1,357 / 8,476 ms | 1,174 / 5,318 ms |

Opus had lower medians, but it did not eliminate the long tail. The runs were
sequential on a variable network with different generated text, so the numerical
improvement is observational, not a controlled estimate of codec benefit.
Across the 49 Opus MOSS results, synthesis averaged 399 ms; the maximum was
476 ms. This measures synthesis, not edit-to-playback response time.

## Remaining Delay

Five tiny capabilities requests over the Mac-to-H200 tunnel took
1,719, 1,020, 1,575, 908, and 4,806 ms (median 1,575 ms). The same route called
on H200 took 20.8 ms initially, then 0.50-1.65 ms. Eight ICMP probes had no loss
but 276-1,279 ms RTT, averaging 706 ms. This establishes substantial network
delay during this run; it does not isolate VPN, Wi-Fi, congestion, or SSH as its
root cause. The application `transfer` timer also includes server wait and
synthesis, so it must not be added to MOSS as an independent stage.

The preserved selection policy prefers latest MOSS, then older MOSS, then
latest generated-lyric eSpeak. An edit can therefore be acknowledged and have
new text available while older audio still plays. Protected committed bars and
bar-boundary playback add discrete waiting time: four seconds per bar at
60 BPM, 2.667 seconds at 90 BPM. Restoring fast synthesis alone does not remove
those delays. The longest PCM case also waited 11.03 seconds for the first
matching lyric; that was not 11 seconds of MOSS synthesis.

In that longest PCM case, the client acknowledged the edit at 4 ms, but new
lyrics arrived at 11.031 s. Matching MOSS audio was prepared at 13.250 s
(449 ms synthesis, 2,144 ms blocking fetch, 69 ms Mac processing), then reached
its requested onset at 13.921 s. Three new lyric reservations arrived within
1.98 ms of one another. That suggests queued delivery, but the client trace
cannot distinguish server writing, transport, or reconnection delays. An edit
ACK confirms the client sent the update, not that the server accepted it.

Other traces explicitly show old MOSS selected despite a newer lyric revision
being known: edit 1 selected revision 0 with latest revision 1, and edit 4
selected revision 1 with latest revision 2. This is the preserved policy in
`src/streammuse/application/rap/bar_stream.py`, not a synthesis regression.

## Retained Feature Acceptance

The real Mac audio client and H200 services passed the settings acceptance run:

- All three alignment policies reached their requested eligible bars; each
  currently playing bar stayed unchanged. Local acceptance took 1.5-2.9 ms.
- Gain changes remained available while a server-coordinated 55 BPM change was
  pending. Both minor and major chord loops reached playback.
- A further topic/flow edit reached matching playback at 9.06 s.
- Stop/Start worked without Reset.
- Alternating mode captured ticks `[2, 4, 5, 7, 8, 10, 12, 14]`, replayed while
  waiting, and delivered both requested response bars. Relaxed and strict-stress
  alignment both reached responses. The complete exchange took 20.07 s, with
  two waiting replays and no underruns; this is not a 20-second synthesis time.

These checks preserve functionality, not a claim of ideal responsiveness.
Detailed evidence is in `settings_acceptance_opus/summary.json` and its raw logs.

A final browser-driven edit changed the topic to `ocean observatory` and flow
to Basketball m20 - Delayed 8. The GUI confirmed the requested settings played
in bar 6, and subsequent displayed bars used `moss_lyric_stream`. Screenshots
`browser_live_edit.png` and `browser_played_result.png` preserve the UI evidence.
This extra manual check is not included in the 24-edit timing aggregate.

The demo is left stopped at `http://127.0.0.1:8018/`, continuous mode, 60 BPM,
relaxed alignment, chords off, drum gain 0.78, vocal gain 0.8, and MacBook Pro
Speakers. Start is enabled without requiring Reset. Final remote health reports
the SGLang-Omni MOSS backend, MMS aligner, and Qwen service ready.

## Regression Verification

Final local suite: **2,966 passed, five skipped, 36 warnings** in 96.86 s with
`.venv/bin/python -m pytest tests/ -q --tb=short`. Warnings concern deprecated
`pkg_resources` and ignored non-data WAV chunks. Deployment tests cover scoped
GPU/process ownership, pinned runtime evidence, and explicit Opus configuration.
Measurement tests reject ACK-only and incomplete or mismatched playback evidence.
No production application source was changed for this restoration.

## Artifacts

Local root: `output/rap_mosslocal_20261028/` in the `real_rap` worktree.

- `deployment/`: health snapshots, exact package versions, model/codec pins,
  and launch records.
- `live_edits/01_60bpm/` and `live_edits/02_90bpm/`: PCM summaries, raw HTTP,
  canonical events, per-edit sample offsets, and `session.wav`.
- `live_edits_opus/`: corresponding trials of the existing Opus transport.
- `settings_acceptance_opus/`: live settings and alternating-mode acceptance.
- `live_topic_flow_60bpm_mosslocal_mix.wav`: 20-second listening excerpt from
  the 60 BPM Opus callback recording.
- `client.log`, `client.pid.json`, and `sessions/`: the restored Mac client.

The acceptance script now rejects ACK-only, commit-only, wrong-revision,
unfinished/muted playback, and timeout cases. The older settings acceptance
script's ten-second sleep was replaced with the same exact played-revision gate.
