# Rap Opus Transport Experiment (2026-08-21)

## Objective

Test whether compressing the H200-to-Mac vocal package reduces enough network
time for remote MOSS audio to meet the 90 BPM, two-bar realtime deadline.

Revision `09eef6f2` adds an opt-in 48 kbps Opus wire package while retaining
the canonical PCM `response.zip` and `vocal.wav`. The Mac decodes Opus back to
24 kHz mono PCM and verifies the exact expected frame count before scheduling.
PCM remains the compatibility default.

## Protocol And Safety

- H200: `--wire-audio-codec opus` enables negotiated Opus responses while
  continuing to serve PCM clients.
- Mac: `--rap-audio-transport opus` requests Opus and accepts PCM fallback.
- The derived ZIP contains `manifest.json`, `transport.json`, and
  `vocals.opus`; its media type is
  `application/vnd.streammuse.rap-chunk-opus+zip`.
- FFmpeg must expose the `libopus` encoder. Both startup paths probe it.
- Encoding runs outside the FastAPI event loop and store locks. Derivatives
  are keyed to the canonical package hash, so stale work cannot overwrite a
  regenerated package.
- Client cancellation is deadline bounded. A worker that does not stop during
  the 50 ms grace period retains request ownership until cleanup finishes,
  preventing overlapping replacement requests.

## Test Setup

- Mac client, H200 render service, and persistent SSH local forwarding.
- H200 GPU 1: Qwen2.5-7B vLLM; H200 GPU 0: MOSS-TTS, MMS alignment, and R3.
- Tempo: 90 BPM; lookahead: two bars; rolling timeout: 5.0 seconds.
- Opus bitrate: 48 kbps.
- PCM baseline: retained run at revision `9c285aa2`.
- Opus run: revision `09eef6f2`.

The live-run comparison is not a controlled server-compute benchmark: model
outputs and evaluation cost differed. Payload size and client-observed latency
are direct measurements. Residual transport time is an inference obtained by
subtracting reported server stages and therefore also includes uninstrumented
queuing, serialization, and Opus work.

## Results

| Metric | PCM baseline | Opus | Change |
|---|---:|---:|---:|
| First live response bytes | 198,193 | 36,188 | -81.74% |
| First live end-to-end | 8,227.908 ms | 5,974.139 ms | -2,253.769 ms (-27.39%) |
| Reported server stages | 4,018.171 ms | 3,410.895 ms | -607.276 ms |
| Inferred residual | 4,197.303 ms | 2,554.626 ms | -1,642.677 ms |
| Successful remote bars | 2 / 20 | 2 / 20 | unchanged |
| Fallback bars | 18 / 20 | 18 / 20 | unchanged |
| Audio underruns | 0 | 0 | unchanged |

Eight paired cached requests over one reused HTTP connection measured:

| Cached request | Mean | Median | Maximum |
|---|---:|---:|---:|
| Opus, 35,978 bytes | 1,585.4 ms | 1,541.6 ms | 2,680.9 ms |
| PCM, 192,108 bytes | 1,760.8 ms | 1,667.6 ms | 2,801.1 ms |

The same cached packages requested directly on H200 took 5.665 ms for Opus
and 12.646 ms for PCM. This establishes that the large remaining cached delay
is in the Mac-to-H200 route rather than artifact lookup or packaging.

Standalone codec checks on the retained two-bar vocal measured:

- 48 kbps Opus package: approximately 36 KB versus approximately 194 KB PCM.
- Exact decoded length: 128,000 frames.
- Decoded quality: 21.37 dB SNR and 0.9964 waveform correlation.
- H200 encode: 94.02 ms mean, 99.95 ms maximum over ten samples.
- Mac decode: 38.35 ms mean, 40.88 ms maximum.

## Conclusion

The method works as a transport optimization: it removes more than 80% of the
payload and reduced the comparable first response by 2.25 seconds. It does not
make the present SSH route reliable at a five-second rolling deadline. The
first Opus chunk arrived in 5.97 seconds, and subsequent uncached chunks timed
out, so the scheduler correctly used its local eSpeak fallback without audio
underruns.

Keep Opus as the recommended remote transport. To preserve two-bar lookahead,
the next bottleneck is the route's fixed delay plus roughly 3.4 seconds of
server generation/render work. The practical next experiments are a
lower-latency network path and server pipeline overlap; additional lookahead is
the available scheduling fallback when those cannot be changed.

## Retained Evidence

- PCM events:
  `logs/rap/remote_moss_acceptance_20260821/mac_final_20bar_9c285aa2/rap-20260821T122702Z-782db4bd/events.jsonl`
- Opus events and summary:
  `logs/rap/opus_transport_acceptance_20260821/opus_live_20bars/rap-20260821T140635Z-d92755a9/`
- Opus mixed WAV:
  `logs/rap/opus_transport_acceptance_20260821/opus_live_20bars.wav`
- H200 canonical and Opus packages:
  `/data/home/Andrew.Yang/StreamMUSE/deploy/opus_transport_38a932a3/logs/rap/opus_transport_server/<request-id>/`

