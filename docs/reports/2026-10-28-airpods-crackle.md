# AirPods And Crackle Investigation

## Device Recovery

The long-running Mac client rejected Andrew's AirPods Pro with PortAudio
`-9986`, while a fresh process accepted the same device at 44.1/48 kHz, mono
and stereo. A fresh output stream played a two-second 440 Hz tone with a
0.025 peak and no write underflow. On replay, the user confirmed the tone was
audible without crackling.

Restarting only the stopped Mac client restored the AirPods in the GUI and
opened them successfully. The H200 services and SSH forward were untouched.
The previous selected output was the virtual `ZoomAudioDevice`; output now goes
directly to the AirPods. The user's 75 BPM, alternating mode, two response bars,
strict syllable timing, minor chords, and 0.5/0.8/0.2 drum/vocal/chord gains were
preserved. A 22.4-second real-client idle playback completed with zero reported
underruns and no device failure.

This strongly supports stale PortAudio/Core Audio state after a Bluetooth
reconnect, not an unsupported sample rate. Automatic hot-plug recovery is not
implemented; reconnect headphones before starting the Mac Python process, or
restart that client while stopped. Website reload and rap Reset do not
reinitialize PortAudio. See the
[PortAudio lifecycle documentation](https://portaudio.com/docs/v19-doxydocs/portaudio_8h.html).

The direct tone stream reported 512 ms output latency. That is a host-reported
buffer/transport estimate, not an acoustic measurement; Bluetooth can add a
substantial delay even with clean playback.

## Waveform Evidence

| Callback recording | Duration | Absolute peak | Samples at or above 0.95 |
| --- | ---: | ---: | ---: |
| Previous Opus, 60 BPM | 84 s | 0.7519 | 0 |
| Previous Opus, 90 BPM | 64 s | 0.9419 | 0 |
| AirPods idle, 75 BPM | 22.4 s | 0.3407 | 0 |

All samples were finite. These captures do not support output-level clipping
or gaps between adjacent bars as the explanation. They do not rule out every
upstream distortion or every possible Bluetooth delivery problem.

The 60 BPM recording has a concrete vocal-edge discontinuity at 32.500000 s
(frame 1,560,000): silence jumps to +0.2385 in one sample. An estimate removing
the coincident hat attack gives about +0.2856 of instantaneous vocal contribution.
Other vocal stems show the same zero-to-nonzero edge under both relaxed and
strict policies. Source cropping and removal of resampling pre-ringing create
a hard edge when the aligned onset lands on a nonzero waveform sample.

Procedural snare/hat noise also has deliberately sharp attacks. The largest
adjacent-sample changes recur at the same snare-relative offset in both rap
recordings and the no-vocal idle control. That is evidence of source percussion
harshness, not random callback corruption. Percussion was left unchanged to
avoid confounding the vocal fix with a different drum sound.

The live mixer can hard-clip under sufficiently high gain. That is a separate
conditional risk; it was not observed in these captures.

## Narrow Fix

MOSS per-bar preparation now applies a two-millisecond, squared-sine amplitude
ramp at the first vocal onset and at the end of the bar. It runs after output
resampling. Leading rests stay exactly silent; bar duration, slot timestamps,
interior syllables, pitch, alignment policy, drums, and chords stay unchanged.
Only at most 192 stereo frames are scaled at 48 kHz, outside the audio callback.
This removes the identified boundary discontinuity; it does not claim to remove
all perceptual artifacts from aggressive internal time stretching.

Regression tests first reproduced nonzero vocal edge samples with first-tick
and delayed-onset flows, then passed with the ramp. Listening comparisons must
still establish whether it addresses the user's perceived crackle.

## Listening Isolation And Verification

The user described the crackle as continuous, not just at word/bar starts. They
then reported the no-vocal drum/chord recording clean. Both raw MOSS speech and
strictly aligned vocal-only audio also sounded clean to them. Therefore the
boundary discontinuity is a confirmed separate defect, but is **not established
as the cause of the reported continuous crackle**. Live-only behavior, the old
virtual routing, fallback vocals, and mix-dependent effects require checking;
do not label alignment or loudness the root cause based only on waveform tests.

The controlled two-bar H200 comparison at 75 BPM reused exactly the same source
speech, forced alignment, strict policy, drums, minor chords, and gains. First
vocal samples were +0.01890 and -0.07014 before the fix and zero afterward.
Samples outside the two edge windows were bit-identical, and onset estimates
and 153,600-frame bar lengths were identical. Assets and per-bar evidence are
under `alignment_comparison_opus/declick/`; source generation succeeded against
the real H200 Opus server. The comparison helper was also fixed to decode Opus
while preserving PCM support.

Regression suite after the edge and comparison fixes: **2,970 passed, five skipped, 36
warnings**, 94.43 s. The two edge regression cases failed before the fix; the
comparison regression reproduced the missing-Opus-decoder error before its fix.

## Live-Only Native Failure

A subsequent real-client vocal test did not reach audible playback: the client
remained priming and Core Audio logged `Audio Hardware: Illegal Operation`.
Native sampling of only that process showed three threads concurrently inside
`Pa_IsFormatSupported -> OpenAndSetupOneAudioUnit -> AudioComponentInstanceNew`,
blocked on Core Audio mutexes, plus another in `FinishStoppingStream`. The GUI
settings endpoint format-probed every listed output on each poll (normally every
five seconds), including during playback; the acceptance poll added more probes.
Thus capability checks are not harmless metadata reads and can interfere with
native stream lifecycle operations. This trace is stronger evidence for a
live-only device-control problem than the clean offline vocal comparisons.

The failed live test is not counted as passing or as zero-underrun playback:
no audio callbacks started. Its diagnostics and pre-test settings were retained
under `live_verse/`. Graceful client shutdown released the GUI port but stalled
in native cleanup; only the identity-verified client process was terminated.

The native adapter now caches format availability during RUNNING and
STOP_REQUESTED playback. It serializes device discovery/format probes with
native stream open/start/stop/close and checks formats freshly while stopped.
Callbacks never take the native-control lock, and blocking native start/stop
operations do not hold the callback state lock. Startup primes the cache even
without a preceding GUI query. Stable host/name resolution and injected-factory
test behavior are preserved; this is not a general Bluetooth hot-plug refresh.

The corrected real-client test played **nine complete MOSS bars at 75 BPM**
through the AirPods, with no reported underruns or device failures. Two threads
made **548 settings requests** during playback, with no request errors, 3.49 ms
median and 410 ms maximum response time. The user reported **a very occasional
crackle, otherwise clean**, so the continuous issue improved substantially but
the residual crackle is not claimed fixed. The test then restored stopped
alternating mode and all original settings. Events and settings are retained
under `live_verse_fixed/`; restoring mode reset the shared recorder before its
WAV was copied, so that run's empty WAV is not used as waveform evidence.

A final independent capture saved PCM **before** restoring mode: four complete
MOSS bars (12.8 s), peak **0.68835**, zero samples at the 0.95 limiter, and zero
reported underruns. It is `live_capture/airpods_live_moss_mix.wav`, with raw events
and summary beside it. This provides the reference for deciding whether any
remaining occasional crackle exists in callback PCM or only after device output.

Final full regression suite including the native-device fix: **2,991 passed,
five skipped, 36 warnings**, 103.07 s. All live tests ended stopped, with AirPods
selected and the user's 75 BPM alternating/strict-timing/mix settings restored.
H200 services and SSH forwarding were not restarted during this investigation.

## Evidence Location

Local artifacts: `output/rap_audio_device_20261028/` in the `real_rap` worktree.
Includes before/after settings, identity-scoped client launch records,
`airpods_selected.png`, and completed callback PCM under `sessions/`.
