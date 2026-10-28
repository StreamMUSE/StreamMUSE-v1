# Rap Demo: Isolated Session, 2026-10-22

Companion to the [feature quickstart](rap-demo-quickstart.md). These commands
describe the current integration setup. Live acceptance and its limitations
are recorded separately, not implied by successful launch commands.
Consult the [feature report](../reports/2026-10-22-feature-delivery.md) before
claiming acceptance and the
[research handoff](../reports/2026-10-22-scheduling-research.md#service-handoff)
before operating shared services. Live services and the tunnel are already
running; recheck rather than launching duplicates.

| Location | Endpoint | Purpose |
|---|---|---|
| Mac | `127.0.0.1:8018` | Current test GUI, local clock and audio |
| Mac | `127.0.0.1:8012` | Preserved client only; old 18420 remote is dead |
| Mac -> H200 | `18620` -> `127.0.0.1:18620` | Render/v3 service tunnel |
| H200 GPU1 | `127.0.0.1:18611` | Live Qwen, served as `qwen-rap` |
| H200 GPU2 | `127.0.0.1:18620` | In-process MOSS-TTS-v1.5 and MMS |
| H200 GPU0 | Research service stopped | Released at 06:16:47 PDT; no demo dependency |

The current test defaults are **in-process MOSS**, **PCM transport**, and no
SGLang continuation. The Mac needs no direct Qwen forward. Old 8020/18420
examples are different deployments. No H200 `uv sync`, model download, or
environment reinstall is needed for this provisioned setup.
The old 18420 endpoint is currently dead: keeping its 8012 client alive does not
make that old remote-audio stack a working rollback. GPU0 research has finished;
preserve the shared tunnel used by live 18620.

## H200 Terminal: Inspect And Reuse

From a Mac terminal, connect with the configured SSH alias, then run these
read-only checks on H200:

```bash
ssh H200
R=/data/home/Andrew.Yang/StreamMUSE/deploy/rap_research_20261022
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18611/v1/models
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18620/health
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18620/v1/rap/capabilities
```

Reuse healthy services. Wait for render health `ready: true`; a successful HTTP
response alone is insufficient. An older service may pass health but omit the
new capability route/flags: tempo/mode can then remain disabled on the Mac.
The integration owner must deploy the matching product source before claiming
new-feature support. The baseline research smoke is not that deployment gate.

The isolated source destination is `$R/repo/src/`. Preserve the rest of that
remote checkout, the old deployment's `deps`, model environments, and logs.
Launch records are `$R/vllm-live.pid.json` and `$R/render.pid.json`; corresponding
logs are `vllm-live.log` and `render.log`. Do not act on historical PID numbers.

### Optional Launch Or Renderer Restart

**Skip this section when the services are healthy.** These are operator commands
for an authorized maintenance window, not ordinary Mac startup steps. They
depend on the already-installed `$R/evidence.py` and verified old deployment.
If either is missing, stop and consult the owner; do not invent a replacement
launcher or mutate the old deployment.

In the same H200 terminal, preflight:

```bash
R=/data/home/Andrew.Yang/StreamMUSE/deploy/rap_research_20261022
OLD=/data/home/Andrew.Yang/StreamMUSE/deploy/rap_v3_gui_efd621ed_gpu1_20261014
test -f "$R/evidence.py" || { printf '%s\n' 'Missing isolated evidence.py; do not launch.' >&2; exit 1; }
test -f "$OLD/deploy.py" || { printf '%s\n' 'Missing verified deployment reference; do not launch.' >&2; exit 1; }
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
ss -ltnp 'sport = :18611 or sport = :18620'
```

Only if the live Qwen service is absent, GPU1 is free, and its isolated pidfile
has been reconciled by the owner, launch it once and wait for model discovery:

```bash
python3 "$R/evidence.py" launch vllm-live --root "$R" --old-deployment "$OLD"
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18611/v1/models
```

For an absent renderer with GPU2 free and reconciled pidfile:

```bash
python3 "$R/evidence.py" launch render --root "$R" --old-deployment "$OLD"
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18620/health
```

After the integration owner has synced new source, an authorized restart of
**only the isolated renderer** uses these commands instead of a second launch:

```bash
python3 "$R/evidence.py" stop render --root "$R"
python3 "$R/evidence.py" launch render --root "$R" --old-deployment "$OLD"
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18620/health
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18620/v1/rap/capabilities
```

Model loading/warmup is asynchronous; retry health after startup, not launch.
The script verifies PID/start-time/UID/process-group identity before TERM,
archives its own pid record, and refuses occupied ports/nonempty GPUs. A refusal
is an inspection point, not permission to delete pidfiles or kill other jobs.
Do not restart either Qwen service for a renderer code update. Existing model
environments are `/data/home/Andrew.Yang/qwen-extract-venv` and
`/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/moss`; the launcher
reuses their verified argv/environment and appends logs.

## Mac Terminal A: Reuse Or Open The Tunnel

```bash
lsof -nP -iTCP:18620 -sTCP:LISTEN
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18620/health
```

If health is ready through an existing listener, **do not open another tunnel**.
The research handoff's tunnel uses control socket
`/tmp/rap-research-20261022-ssh.sock` and also forwards research port 18601;
neither restart nor close that shared connection from the feature trial.

Only when local 18620 is unused and the H200 service is ready, keep this
dedicated tunnel in the foreground:

```bash
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N \
  -L 18620:127.0.0.1:18620 H200
```

## Mac Terminal B: Existing GUI Or New Client

The actual test GUI uses **8018**. Check first:

```bash
lsof -nP -iTCP:8018 -sTCP:LISTEN
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8018/api/session
```

If it is the existing test process, reuse `http://127.0.0.1:8018/` and coordinate
controls with its operator. Do not run the launch block below while 8018 is
occupied. A fresh client starts only the local runtime/website, not H200 models.
Use the existing isolated worktree and interpreter; dependencies must already
include local eSpeak, PortAudio, FFmpeg, and Rubber Band R3:

```bash
cd "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE-NewSystem/.worktrees/real_rap"
command -v espeak-ng
command -v ffmpeg
command -v rubberband
PYTHONPATH="$PWD/src" .venv/bin/python -m streammuse.presentation.rap_demo.cli \
  --rap-audio-renderer moss_aligned_remote \
  --rap-protocol v3 \
  --rap-audio-transport pcm \
  --rap-render-url http://127.0.0.1:18620 \
  --rap-render-startup-timeout 30 \
  --interaction-mode continuous \
  --response-bars 2 \
  --tempo 60 \
  --audio-output composite \
  --audio-device "MacBook Pro Speakers" \
  --max-bars 0 \
  --log-dir logs/rap/current_20261022 \
  --terminal-layout split \
  --terminal-detail full \
  --host 127.0.0.1 \
  --port 8018
```

Open `http://127.0.0.1:8018/`, then Start explicitly. Use the stopped-state mode
selector to test Alternating; for an initially alternating process, replace
`--interaction-mode continuous` with `--interaction-mode alternating` in the
same launch, not a second process. Leave gains omitted to use the v3 defaults
(drums about 0.776896, vocals 0.80), or set `--drum-gain` / `--vocal-gain`
explicitly. Omit `--audio-device` for the system default on another Mac.

In the GUI/API, apply live BPM separately from gains, and device selection
separately from every other change. Both gains may be submitted together;
stopped coordinator-rebuild settings may combine mode/response/tempo/gains.
Mixed operation groups are rejected before effects, and the current GUI does
not split them automatically. See [settings and playback evidence](rap-demo-quickstart.md#settings-and-playback-evidence)
for how to clear conflicting unsaved edits and retry.

Continuous-mode live tempo changes reserve a **10-second ACK window plus
preparation margin**, then align to a future bar boundary. They are not
next-bar ASAP, even if the ACK returns quickly; inspect `effective_from_bar`
and later played timing evidence. Topic/flow edits remain ASAP. Alternating
capture keeps its current bar timing, with remote response timing negotiated
separately. The current GUI POST timeout is **35 seconds**, covering possible
10-second capability probes both before and after the 10-second ACK wait,
plus margin. One identical backend retry halfway through the ACK window does
not extend it; the browser does not automatically retry the settings POST.

## Mac Terminal C: Read-Only Diagnostics

```bash
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18620/v1/rap/capabilities
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8018/api/control/settings
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8018/api/control/interaction
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:8018/api/state
```

Settings capability values, not the presence of HTML controls, determine what
is supported. An ACK/effective boundary is not proof of played tempo/gain;
inspect the current bar's `audio.timing_evidence`. A device stream is not an
acoustic onset measurement. Runtime errors and `audio_output` warnings must be
resolved before claiming audible acceptance. Preserve each Mac session under
`logs/rap/current_20261022/<session-id>/` and remote artifacts under
`$R/render_artifacts/`; the launch/health checks alone do not establish success.

**WAV retention differs from event retention:** `events.jsonl` is cumulative
across restarts, but `mixed.wav` is reset/truncated when an epoch is reset or
rebuilt. After playback is stopped, archive the WAV under a unique run name
before Reset, mode/response-length changes, stopped-state tempo changes, or
automatic finite replay. Otherwise a later capture/cancel run can replace the
audio from earlier completed turns even though their events remain in the log.
Copy the final showcase WAV before any subsequent Reset; do not infer WAV
coverage from cumulative event/bar totals.

## Preserved 8012 And Rollback Limits

**8012 is a preserved client, not a verified working rollback. Its old 18420
remote endpoint is dead.** Returning to that page does not restore remote
vocals; a complete rollback needs its owner to restore and verify the paired
old remote service/tunnel first.

1. In the **8018** GUI, request Stop and wait for `stopped`. Preserve its session
   logs/WAV. Use Ctrl-C only in the terminal that owns this new Mac client if
   it should exit; do not kill by port or process-name wildcard.
2. Inspect the preserved endpoint with
   `curl --fail --max-time 10 http://127.0.0.1:8012/api/session`, then open
   `http://127.0.0.1:8012/` for inspection. Do not Start it as a remote-audio
   rollback until its operator restores and verifies its old remote endpoint.
   Do not Reset it, replace its process, or bind new code onto 8012.
3. Leave its old checkout, remote services, dependencies, tunnels, pidfiles,
   and logs unchanged. No branch switch, Git reset, or package sync is part of
   rollback. If 8012 is absent, ask its owner to restore the recorded old
   command/environment; do not guess a launch from this new worktree.
4. Close only a dedicated tunnel you personally started in Terminal A, and
   only when no session uses it. Preserve the shared research tunnel and live
   H200 GPU1/18611 and GPU2/18620 services. GPU0/18601 is already stopped; the
   unused research forward does not require closing that shared tunnel.

These instructions do not change any service themselves and do not certify
continuous/alternating live acceptance. That evidence belongs in the feature
delivery report after the integration owner runs it.
