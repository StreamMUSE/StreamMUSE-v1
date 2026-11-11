# MOSS-Local Live Deployment, 2026-10-28

This deployment restores MOSS-Local with SGLang-Omni and speech continuation
while retaining the current protocol-v3 client, controls, and playback policy.
The website and audio device run on the Mac; models run on H200.

## Endpoints

| Location | Endpoint | Component |
|---|---|---|
| Mac | `http://127.0.0.1:8018/` | Existing Rap Lab GUI and audio playback |
| Mac -> H200 | `18720` -> `127.0.0.1:18720` | Render/v3 service |
| H200 GPU0 | `127.0.0.1:18711` | Existing Qwen2.5-7B service, unchanged |
| H200 GPU1 | `127.0.0.1:18730` | MOSS-Local / SGLang-Omni |
| H200 GPU0 (current placement) | `127.0.0.1:18720` | Render server and MMS alignment |

Deployment root:
`/data/home/Andrew.Yang/StreamMUSE/deploy/rap_mosslocal_20261028`.
Do not launch duplicate services or stop unrelated GPU jobs.

**2026-10-31:** the renderer was moved from GPU1 to GPU0 alongside the owned
Qwen process after another user's job appeared on GPU1. MOSS stayed on GPU1;
no unrelated process was stopped. `render.pid.json` records the actual device
and the checked process identities. The launcher below still defaults to GPU1
for a new start and will refuse unsafe occupancy; do not force past that check.

## H200: Inspect Or Launch

The following commands assume the isolated environment, pinned models, and
source copy have already been provisioned. They do not install dependencies.

```bash
R=/data/home/Andrew.Yang/StreamMUSE/deploy/rap_mosslocal_20261028
L="$R/rap_mosslocal_20261028_deploy.py"
python3 "$L" status tts
python3 "$L" status render
```

Reuse healthy services. For a clean, stopped deployment, start TTS first:

```bash
python3 "$L" start tts
tail -n 40 "$R/tts.log"
curl --fail --silent --show-error http://127.0.0.1:18730/health
```

Wait for TTS health, then start the renderer once:

```bash
python3 "$L" start render --wire-audio-codec opus
tail -n 40 "$R/render.log"
curl --fail --silent --show-error http://127.0.0.1:18720/health
curl --fail --silent --show-error http://127.0.0.1:18720/v1/rap/capabilities
```

Render health must say `ready: true`, `backend: sglang-omni`, and identify
`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`. Capabilities must include
`tempo_updates_v1`, `bounded_turns_v1`, and `tempo_reschedule_v1` for the
fast live-tempo path. A listening port is not sufficient.
Launches refuse occupied GPUs/ports, mismatched process identities, or stale
launch records. Inspect a refusal; do not delete records or kill broad groups.

## Mac: Forward And Client

Check for a working forward first:

```bash
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18720/health
```

If local port 18720 is unused, keep this running in **Mac Terminal A**:

```bash
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -N \
  -L 18720:127.0.0.1:18720 H200
```

Reuse the existing 8018 client. To launch a new client on an unused 8018 port,
run in **Mac Terminal B**:

```bash
cd "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE-NewSystem/.worktrees/real_rap"
PYTHONPATH=src .venv/bin/python -m streammuse.presentation.rap_demo.cli \
  --rap-audio-renderer moss_aligned_remote --rap-protocol v3 \
  --rap-render-url http://127.0.0.1:18720 --rap-audio-transport opus \
  --rap-warp-policy gentle_sparse_r3 --chord-loop off \
  --tempo 60 --audio-output composite --audio-device "MacBook Pro Speakers" \
  --max-bars 0 --port 8018 --log-dir logs/rap/mosslocal_20261028
```

After a server replacement, use Reset while stopped to create a fresh session.
Use the existing Live audio, Generation, and Session setup controls as described
in the [flow/audio guide](rap-flow-audio-controls.md).

Opus is the compressed transport option, not a different voice or alignment
policy. To reproduce the PCM comparison, start the renderer with
`--wire-audio-codec pcm` and the client with `--rap-audio-transport pcm`.
The Opus-enabled client can also decode PCM responses; a PCM-only client cannot
decode Opus responses. The deployment controller defaults to PCM when the
render flag is omitted, so include the explicit option above.

## Reproduce Playback-Latency Measurements

This takes control of the running demo, makes audible edits, and stops at a bar
boundary afterward. Do not interact with its controls during the run.

```bash
.venv/bin/python scripts/experiments/rap_live_edit_acceptance.py \
  --output output/rap_mosslocal_live_retest --bpm 60 90 --edits 6
```

The test waits for two completed MOSS bars, then changes topic and curated flow
together. Each edit must reach playback with the matching topic, template,
lyric ID, and revision. ACKs and committed-but-unplayed bars do not pass.
JSON includes timeouts, source choices, stage timings, and underruns. WAV files
contain completed Mac callback PCM, with sample offsets for auditioning edits.

The vocal-onset figure is a **playback callback plus scheduled onset proxy**,
not a microphone measurement or hardware DAC timestamp. Leading rests are
included. The `transfer` stage includes blocking server-side wait/generation;
it is not a pure SSH network-delay measurement.

## Roll Back The Renderer

Stop playback on the Mac first. In the H200 terminal:

```bash
R=/data/home/Andrew.Yang/StreamMUSE/deploy/rap_mosslocal_20261028
L="$R/rap_mosslocal_20261028_deploy.py"
python3 "$L" stop render
python3 "$L" stop tts
python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py start render
```

The old renderer installation is retained. Its launcher requires GPU1 to be
empty and reuses Qwen on GPU0. Check health before resetting the Mac session.
Never use historical PID numbers or broad process-name kills.
