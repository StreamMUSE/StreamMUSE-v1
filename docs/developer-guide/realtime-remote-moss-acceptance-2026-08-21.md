# Real-Time Remote MOSS Acceptance Record

## Result

The complete Mac-client/H200-server vertical slice is implemented and verified
at revision `9c285aa2ba75f85c26cf14cd83b7394664da225c`. Candidate generation,
selection, MOSS synthesis, MMS alignment, R3 warping, package validation, local
drum mixing, continuous playback, recording, terminal monitoring, and the Mac
website all ran together.

Acceptance is qualified by one measured deployment constraint: **90 BPM with
only two bars of lookahead does not meet its deadline over the tested Mac-H200
SSH path**. The H200 pipeline is fast enough in isolation, but the approximately
194-198 KB PCM response takes too long to return on this link. The prepared
local eSpeak fallback kept playback continuous, with no underruns or missing
frames. This is a transport/lookahead result, not a hidden generation failure.

Local verification at the final revision: `1753 passed, 4 skipped` in 44.94 s,
plus fresh independent reviews of the digit-normalization and monitoring fixes.

## Deployment Evidence

| Item | Evidence |
| --- | --- |
| Mac revision | `9c285aa2ba75f85c26cf14cd83b7394664da225c` |
| H200 revision | Same revision, isolated deployment `/data/home/Andrew.Yang/StreamMUSE/deploy/real_rap_audio_9c285aa2` |
| vLLM | Physical GPU 1, UUID `GPU-74ee34a6-23bd-7e60-e012-6bf3d30b4c3a`, PID `1787400`, `Qwen/Qwen2.5-7B-Instruct` served as `qwen-rap` |
| MOSS and MMS | Physical GPU 0, UUID `GPU-4f4e1a5f-dfd1-e747-40d2-993bf2548918`, PID `2271239` |
| Chunk service | Loopback `127.0.0.1:8020`, schema `streammuse.rap_chunk.v1`, all warmup checks ready |
| Final direct request | `b5cc1cc92e0bde3881db24caf3cf56d8e2f344de5ef339c9485d84cce7fd920a`, H200 wall time 3.473 s, 194,347-byte response |
| Final direct audio | 128,000 frames, 24 kHz, mono PCM16, exactly 5.333333 s |
| Final direct candidates | 36 requested, 28 parseable, 6 valid/selectable |
| Final direct alignment | `torchaudio.pipelines.MMS_FA`, confidence 0.8405, no alignment fallback |
| MFA reference comparison | Earlier retained probe: MMS word starts differed from the accepted MFA reference by 49.46 ms mean absolute and 102.44 ms maximum |

Local retained evidence is under
`logs/rap/remote_moss_acceptance_20260821/h200_evidence/`. It includes health,
model, GPU, process, profile-sweep, headers, ZIP, manifest, and vocal WAV files.
The H200 server-side artifact root is
`/data/home/Andrew.Yang/StreamMUSE/deploy/acceptance_20260821_remote_moss_live/server_9c285aa2/`.

## Warm H200 Profile

The robust policy used 16 initial choices per bar, 4 rescue choices, 20 maximum,
3 minimum valid choices, and a 3000 ms render reserve. Nine of ten requests
completed; the sole failure was a digit-bearing transcript rejected by MMS.
Revision `d67360ba` fixed that preflight mismatch by verbalizing ASCII digits
before prosody, scoring, selection, and rendering. The final direct request
then completed with six selectable candidates.

| Stage | Samples | p50 ms | p95 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Candidate generation | 9 | 461.274 | 580.230 | 580.230 |
| Candidate evaluation | 9 | 7.057 | 8.357 | 8.357 |
| MOSS synthesis | 9 | 2700.570 | 2767.975 | 2767.975 |
| MMS alignment | 9 | 20.902 | 25.139 | 25.139 |
| Rubber Band R3 | 9 | 89.576 | 101.917 | 101.917 |
| Package creation | 9 | 45.673 | 47.600 | 47.600 |
| H200 server total | 9 | 3336.136 | 3441.980 | 3441.980 |
| Direct HTTP wall time | 9 | 3361.011 | 3466.499 | 3466.499 |

Across successful profile requests, valid/selectable candidates averaged 8.11
per two-bar request, with a minimum of 4. The mean selected score was 0.5085.
Candidate ranking was not a latency bottleneck: evaluation stayed below 9 ms
while MOSS synthesis dominated the server budget.

## Mac 20-Bar Run

Session `rap-20260821T122702Z-782db4bd` ran at 90 BPM with exactly two bars of
lookahead. Retained outputs:

- `logs/rap/remote_moss_acceptance_20260821/mac_final_20bar_9c285aa2.wav`
- `logs/rap/remote_moss_acceptance_20260821/mac_final_20bar_9c285aa2/rap-20260821T122702Z-782db4bd/`

| Measure | Result |
| --- | --- |
| Expected / observed frames | 2,560,000 / 2,560,000 |
| Expected / observed duration | 53.333333 s / 53.333333 s |
| Format | 48 kHz, stereo float32 |
| Finite samples / peak / RMS | yes / 0.6864 / 0.0721 |
| Per-bar RMS | 0.0296 minimum, 0.0772 maximum; every bar contains audio |
| Application-level gaps | 0 |
| Playback underruns | 0 |
| Hash or package validation failures | 0 |
| Remote MOSS bars | 2 of 20 |
| Local fallback bars | 18 of 20 |
| Primary acceptance / fallback rate | 10% / 90% |
| Fallback reason | 9 rolling chunk requests exceeded the useful transport deadline |
| Startup pair end to end | 8.228 s including 8.215 s Mac-observed request/transfer time |

The first pair is allowed the 120 s startup budget and therefore used real MOSS.
Every rolling pair had only the musical two-bar deadline. A cached 196,136-byte
response through the same SSH tunnel took 5.245 s even with no rendering. The
same class of request took 3.473 s to generate and package on H200. These costs
cannot fit inside the approximately 5.333 s two-bar window when added together.

## Website And Controls

The final Mac website was exercised against the real H200 service at
`http://127.0.0.1:8012/`:

- Start produced and displayed real MOSS bars.
- The UI showed the selected lyrics, exact flow slots/stresses, deterministic
  generation input summary, committed context, chunk state, and audio evidence.
- The dense two-column desktop layout had zero document-level horizontal
  overflow.
- Stop completed exactly the bar active when clicked, then disabled Stop and
  enabled Start/Reset.
- Reset cleared the live bar and monitoring state.

## Listening Result

Earlier listening in this project found MOSS plus forced alignment substantially
more word-like and rap-like than the local robotic syllable/eSpeak path. The
local renderer remains the correct timing-safe fallback, not the preferred
voice. This was an informal engineering comparison, not a blinded perceptual
study. Alignment confidence and exact timing should not be treated as a proxy
for intelligibility.

## Decision And Next Work

The implementation is accepted as a complete research prototype with an
explicit transport limitation. Server-side candidate evaluation is no longer a
concern, and the audio scheduler demonstrably preserves timing under fallback.
For real MOSS on every rolling pair at 90 BPM, the next experiment must change
one of the constraints:

1. Return a low-bitrate speech codec instead of full PCM-in-ZIP.
2. Use a persistent streaming/push transport and begin transfer earlier.
3. Permit three or four bars of buffering on this network path.
4. Run the client on a lower-latency network path to H200.

The current code intentionally does not pretend that increasing a timeout can
solve a musical deadline. It falls back visibly and records the reason.
