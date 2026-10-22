# Live Rap GUI: ASAP Topic and Flow Editing

Date: 2026-10-14

Branch: `feature/rap_optimization`, based on `efd621ed`.

## Behavior

- Added a compact topic input, flow selector, rhythm/stress preview, and Apply command to the existing monitor.
- Apply sends only changed fields to `POST /api/control/plan`, without `from_bar`. The existing v3 coordinator chooses the next uncommitted bar; there is no additional two-bar delay in the GUI.
- Acknowledgment means the edit was accepted, not that new audio has played. The GUI separately reports the target bar, queued changes, older-revision playback, eSpeak fallback, and actual application.
- UI bar numbers are one-based; API/log bar indices are zero-based.
- Selecting a flow previews it locally. No change is submitted until Apply.
- Start, Stop-at-bar-end, Reset, existing diagnostics, and older runtime support remain. Editing is disabled while stopped or when the runtime does not support plan editing.
- Page reload/reconnect restores the last edit. Reset clears it and allows edit sequences to restart.

The backend now exposes plan capabilities, full flow templates, and the latest edit through GET on the same endpoint. Committed v3 bars carry the flow and syllable schedule that actually played, including when an older revision wins the deadline race. This avoids displaying a newly requested flow over older audio.

## Verification

Automated project suite: **2,187 passed, 5 skipped**, 7 warnings, 79.04 seconds:

```bash
.venv/bin/python -m pytest tests/ -q --tb=short
```

Tests cover capability discovery, validation, ASAP payloads, effective targets, actual-played feedback, fallback reporting, discovery/reset races, reconnect recovery, and compatibility with unsupported runtimes. A pre-existing shared-adapter regression found by the suite was repaired by adding the default `resample_output=False` field to `MlxMossConfig`.

Browser checks used 1440 x 1000 desktop and 390 x 844 mobile viewports. Neither had horizontal overflow. Browser console checks reported no errors. Existing monitor sections remain present. Existing v3 limitations remain: it does not populate the older raw-prompt/candidate-ledger panels with full server diagnostics.

## Real Mac / H200 Acceptance Test

Client: Mac, actual `MacBook Pro Speakers`, local warp/drums/playback, 90 BPM, v3 PCM transport.

Server: H200 **GPU1 only**, real Qwen2.5-7B-Instruct and MOSS-TTS v1.5 with MMS alignment. Continuation was disabled for this test. Other GPUs were not used.

| Action | Earliest target | First updated playback | Outcome |
| --- | --- | --- | --- |
| Topic + syncopated flow | Bar 5 | Bar 7 | New lyrics first used eSpeak; later MOSS |
| Staggered flow only | Bar 16 | Bar 18 | MOSS; topic retained; reload preserved pending status |
| Mountain-trails topic only | Bar 30 | Bar 32 | New lyrics first used eSpeak; flow retained; later MOSS |
| Reset, restart, city-gardens topic | Bar 2 | Bar 3 | Edit sequence restarted at 1; later MOSS |

Changing the dropdown without Apply left the server edit sequence unchanged. Both Stop actions finished the current bar; Reset cleared the edit. The mobile layout was also checked during live playback.

**47 completed audible bars across two runs, zero audio underruns.** Of these, 34 played latest-revision MOSS, 6 played older-revision MOSS, and 7 used eSpeak with generated lyrics. This was a functional acceptance test, not a statistical reliability benchmark. In particular, ASAP does not guarantee that freshly synthesized MOSS audio will meet the first eligible deadline.

The log contains 49 committed bars but only 47 completed bars because each Stop left a successor committed but unplayed. Counts above use completed playback, not commits.

Acceptance artifacts:

```text
logs/rap_gui_v3_20261014/rap-20261014T233449Z-7ec1715a/
  events.jsonl
  bars.csv
  mixed.wav
  session.json
  summary.json
output/rap-gui-v3-live-desktop.png
output/rap-gui-v3-live-mobile.png
```

Reset starts a new metrics epoch, so the final `summary.json` describes the second, seven-bar run. `events.jsonl` retains both runs.

## Running Test Deployment

The Mac GUI was left stopped and ready for Start at `http://127.0.0.1:8012/`. Its detached process log is `logs/rap_gui_v3_20261014/client-ready.log` (initial PID 66268).

H200 services live in an isolated committed-source deployment, leaving the existing dirty checkout and shared environments intact:

```text
/data/home/Andrew.Yang/StreamMUSE/deploy/rap_v3_gui_efd621ed_gpu1_20261014/
  deploy.py
  repo/
  vllm.log
  render.log
  smoke/summary.json
```

The deployment uses ports 8601 (Qwen) and 8620 (render/v3). A private FastAPI 0.116.1 / Starlette 0.47.2 overlay avoids an incompatibility in the shared environment. Initial service PIDs were 680831 and 693069; use the helper for current status:

```bash
ssh H200 'python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_v3_gui_efd621ed_gpu1_20261014/deploy.py status all'
```

The active tunnel is equivalent to:

```bash
ssh -N -L 18420:127.0.0.1:8620 \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 H200
```

Mac command, from this worktree (do not start a second copy while port 8012 is occupied):

```bash
.venv/bin/python -u -m streammuse.presentation.rap_demo.cli \
  --rap-audio-renderer moss_aligned_remote \
  --rap-protocol v3 --rap-audio-transport pcm \
  --rap-render-url http://127.0.0.1:18420 \
  --tempo 90 --audio-output composite \
  --audio-device 'MacBook Pro Speakers' \
  --max-bars 0 --terminal-detail summary --terminal-layout stream \
  --port 8012 --log-dir logs/rap_gui_v3_20261014
```

No commit or push was made for this task.
