# Real-Time Rap Audio Demo Quick Start

## Split Architecture

The normal remote-MOSS runtime runs a loopback-bound chunk orchestrator on the
H200. That one service calls the H200-local vLLM process, selects two lines,
renders one connected MOSS phrase, aligns it, applies Rubber Band R3, and
returns an exact two-bar vocal package. The Mac runs the authoritative rap
clock, local eSpeak fallback, drum rendering, playback, WAV recording, session
logging, and the web monitor. The browser never owns audio; closing it does not
stop the Mac process.

The local website is always `http://127.0.0.1:8012/` for the commands below.
For the normal runtime, forward only H200 port `8020`. The H200-local vLLM port
is not a Mac runtime dependency, and the website is never forwarded from the
H200.

The existing local eSpeak renderer remains available as a startup-selected
legacy mode. It does not change the clock, website controls, or artifact
layout.

## Normal Runtime: Start H200 Services

Connect to the H200, enter the accepted checkout, and record the revision. Pick
two unused physical GPUs immediately before launch. Do not stop or reuse GPUs
with existing compute processes.

```bash
ssh Andrew.Yang@masdar
cd /data/home/Andrew.Yang/StreamMUSE/real_rap
git rev-parse HEAD
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

In H200 terminal A, start the private vLLM dependency on one unused GPU. It
remains behind the chunk orchestrator and is not forwarded to the Mac.

```bash
PATH=/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin:$PATH \
CUDA_VISIBLE_DEVICES=<UNUSED_VLLM_GPU_ID> \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen-rap \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.25
```

In H200 terminal B, declare the installed MOSS/Rubber Band assets and start the
private render service on the second unused GPU. `CUDA_VISIBLE_DEVICES` maps
that physical GPU to logical `cuda:0` for MOSS and the resident MMS aligner.

```bash
cd /data/home/Andrew.Yang/StreamMUSE/real_rap
export REPO_ROOT=$PWD
export ENV_ROOT=/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols
export ASSET_ROOT=/data/home/Andrew.Yang/StreamMUSE/assets/rap-audio-protocols
export MOSS_ENV="$ENV_ROOT/moss"
export FFMPEG7_ENV="$ENV_ROOT/ffmpeg7"
export MANIFEST_PATH="$ENV_ROOT/environment_manifest.json"
export RUBBERBAND_ROOT="$ASSET_ROOT/rubberband/rootfs"
export MOSS_REFERENCE="$ASSET_ROOT/ted/0011_000001.wav"
export MOSS_SNAPSHOT="$("$MOSS_ENV/bin/python" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["models"]["moss_tts"]["snapshot_path"])' \
  "$MANIFEST_PATH")"
export RAP_ARTIFACT_ROOT="$REPO_ROOT/logs/rap/remote_moss_server"
mkdir -p "$RAP_ARTIFACT_ROOT"

PATH="$RUBBERBAND_ROOT/usr/bin:$FFMPEG7_ENV/bin:$PATH" \
LD_LIBRARY_PATH="$RUBBERBAND_ROOT/usr/lib:$FFMPEG7_ENV/lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" \
CUDA_VISIBLE_DEVICES=<UNUSED_MOSS_GPU_ID> \
"$MOSS_ENV/bin/python" -m streammuse.presentation.rap_render_server \
  --host 127.0.0.1 \
  --port 8020 \
  --artifact-root "$RAP_ARTIFACT_ROOT" \
  --vllm-url http://127.0.0.1:8001/v1 \
  --vllm-model qwen-rap \
  --moss-model "$MOSS_SNAPSHOT" \
  --moss-device cuda:0 \
  --moss-reference-wav "$MOSS_REFERENCE" \
  --aligner-device cuda:0 \
  --aligner-cache "$ASSET_ROOT/mms-cache" \
  --candidate-profile realtime
```

Wait for model loading and warmup to complete. From the H200, verify that the
orchestrator reports `ready: true` and the expected schema before opening a
tunnel:

```bash
curl --fail --silent --show-error http://127.0.0.1:8020/health \
  | python -m json.tool
```

## Normal Runtime: Mac Tunnel, Demo, And Website

Install the local fallback and audio dependencies from a Mac terminal:

```bash
brew install espeak-ng portaudio
uv sync
command -v espeak-ng
```

In Mac terminal A, forward only the chunk service:

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L 8020:127.0.0.1:8020 \
  Andrew.Yang@masdar
```

Verify the same private health endpoint through the tunnel:

```bash
curl --fail --silent --show-error http://127.0.0.1:8020/health \
  | python3 -m json.tool
```

In Mac terminal B, start playback, recording, terminal monitoring, and the Mac
website. The command does not construct a Mac-side vLLM client.

```bash
uv run streammuse-rap-demo \
  --rap-audio-renderer moss_aligned_remote \
  --rap-render-url http://127.0.0.1:8020 \
  --rap-render-profile realtime \
  --rap-render-startup-timeout 120 \
  --rap-render-rolling-timeout 5.0 \
  --audio-output composite \
  --tempo 90 \
  --lookahead-bars 2 \
  --max-bars 0 \
  --log-dir logs/rap \
  --terminal-layout split \
  --terminal-detail full \
  --host 127.0.0.1 \
  --port 8012
```

Open `http://127.0.0.1:8012/` in the Mac browser. Start, Stop, and Reset are the
only runtime actions. They can also be invoked from another Mac terminal:

```bash
curl --fail --silent --show-error -X POST http://127.0.0.1:8012/api/control/start
curl --fail --silent --show-error -X POST http://127.0.0.1:8012/api/control/stop
curl --fail --silent --show-error -X POST http://127.0.0.1:8012/api/control/reset
```

Stop is bar-quantized. Wait until the state is stopped before Reset. Inspect
health and the bounded browser projection at any time with:

```bash
curl --fail --silent --show-error http://127.0.0.1:8012/api/session \
  | python3 -m json.tool
curl --fail --silent --show-error http://127.0.0.1:8012/api/state \
  | python3 -m json.tool
```

Mac session artifacts are under `logs/rap/<session-id>/`. The H200 retains
`request.json`, `manifest.json`, `candidate_ledger.json`, `alignment.json`,
`mms_alignment.json`, `source.wav`, `aligned.wav`, `server_timing.json`,
`response.zip`, or `failure.json` under
`$RAP_ARTIFACT_ROOT/<request-id>/`. Live events contain bounded summaries,
hashes, and artifact references; they never contain WAV bytes, full ledgers, or
full character spans.

## Normal Runtime Shutdown

1. Select Stop in the Mac website or call the stop endpoint and wait for the
   current bar to complete.
2. Stop the Mac demo process with `Ctrl-C`.
3. Stop the Mac SSH tunnel with `Ctrl-C`.
4. Stop only the render-service process in H200 terminal B with `Ctrl-C`.
5. Stop only the vLLM process in H200 terminal A with `Ctrl-C`.

Confirm the Mac listeners and tunnel are gone. These commands inspect state;
they do not terminate unrelated processes:

```bash
lsof -nP -iTCP:8012 -sTCP:LISTEN
lsof -nP -iTCP:8020 -sTCP:LISTEN
curl --fail --silent --show-error http://127.0.0.1:8020/health
```

The `lsof` commands should return no matching listener, and the final curl
should fail after shutdown. On H200, rerun the `nvidia-smi` process query and
verify only the two PIDs started for this session disappeared.

## Return To The Local eSpeak Renderer

Stop the remote-mode Mac process, then use this exact device-free legacy
command. It requires no H200 service or tunnel and preserves the prior isolated
eSpeak renderer for fallback and regression comparison.

```bash
uv run streammuse-rap-demo \
  --generator phrase_bank \
  --rap-audio-renderer espeak \
  --audio-output wav \
  --tempo 90 \
  --lookahead-bars 2 \
  --max-bars 12 \
  --terminal-layout stream \
  --terminal-detail full \
  --no-web
```

## Optional Diagnostics: Start A Direct H200 vLLM Endpoint

The remaining direct-vLLM workflow is optional diagnostics and legacy eSpeak
experimentation. It is not a prerequisite for the normal remote-MOSS runtime.
Connect to the H200 and choose a GPU with zero used memory. Do not select a GPU
with an existing compute process.

```bash
ssh Andrew.Yang@masdar
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

On the H200, start one server on the selected unused GPU. The environment `bin`
directory is intentionally first in `PATH`: vLLM/FlashInfer may need its
existing `ninja` executable during startup.

```bash
PATH=/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin:$PATH \
CUDA_VISIBLE_DEVICES=<UNUSED_GPU_ID> \
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen-rap \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.25
```

Keep this terminal visible. Stop only this server with `Ctrl-C` when the Mac
session is finished. It is not an audio server and must not host the website.

## Optional Diagnostics: Mac Direct-vLLM Tunnel

This optional path checks vLLM directly or supports the legacy direct-vLLM
eSpeak experiment below. It is not used by `moss_aligned_remote`. Install the
local rendering and device dependencies, then resolve the project environment
from the repository root:

```bash
brew install espeak-ng portaudio
uv sync
command -v espeak-ng
```

In another Mac terminal, optionally forward the H200 model endpoint for this
diagnostic path:

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L 8001:127.0.0.1:8001 \
  Andrew.Yang@masdar
```

Keep that tunnel open only while running the direct diagnostic or legacy
experiment. It maps Mac `127.0.0.1:8001` to the H200 model endpoint; it does not
forward the web monitor.

Before a session or candidate sweep, verify both the model discovery and one
minimal chat request through this exact tunnel:

```bash
curl --fail --silent --show-error http://127.0.0.1:8001/v1/models
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-rap","messages":[{"role":"user","content":"Reply exactly: ok"}],"max_tokens":4,"temperature":0}' \
  http://127.0.0.1:8001/v1/chat/completions
```

The second command must return a chat completion before starting a sweep. If it
fails, retain its curl output and do not report a candidate sweep as remote
evidence.

## Optional Legacy Direct-vLLM eSpeak Demo

Run this optional comparison in a third Mac terminal after opening the direct
vLLM diagnostic tunnel. It records an IEEE-float WAV while playing through the
selected CoreAudio default device and starts the local web monitor at
`http://127.0.0.1:8012/`. The accepted device-free evidence confirms nonempty
eSpeak vocal PCM alongside drums; physical speaker audibility and latency
remain unmeasured.

```bash
uv run streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 \
  --model qwen-rap \
  --audio-output composite \
  --tempo 60 \
  --candidate-count 36 \
  --lookahead-bars 3 \
  --minimum-score 0.30 \
  --max-bars 0 \
  --terminal-layout split \
  --terminal-detail full \
  --port 8012
```

`--max-bars 0` runs until stopped. Candidate count, lookahead, tempo, drums,
and voice are startup configuration; the website intentionally exposes no
controls for them.

### Start, Stop, And Reset

The website has exactly three runtime controls:

- **Start** begins a stopped audio session. It is the only way to begin an
  audio-web session after the server is opened.
- **Stop** is bar-quantized: the active bar finishes, then no successor starts.
- **Reset** is available only after the runtime is stopped. It clears the
  current epoch's bars, warnings, metrics, WAV contents, and audio clock so the
  next Start begins again at bar zero.

There are no browser controls for topic, model, tempo, candidate count,
lookahead, drums, voice, or mixer settings.

## Device-Free WAV Fallback

Use this deterministic, no-web command when speakers, PortAudio, or a web
browser are unavailable. `wav` uses the timed device-free primary sink and
writes only completed bars to `mixed.wav`.

```bash
uv run streammuse-rap-demo \
  --generator phrase_bank \
  --audio-output wav \
  --tempo 60 \
  --candidate-count 12 \
  --lookahead-bars 3 \
  --max-bars 12 \
  --terminal-layout stream \
  --terminal-detail summary \
  --no-web
```

Each invocation creates `logs/rap/<session-id>/` with `session.json`, ordered
`events.jsonl`, `bars.csv`, `summary.json`, and, for `wav` or `composite`,
`mixed.wav`. Regenerate derived text artifacts with:

```bash
uv run python scripts/summarize_rap_session.py logs/rap/<session-id>
```

The canonical log exposes `software_error_samples`, audio commit slack,
completed frame counts, and underruns. It does not measure physical speaker
latency.

## Legacy H200-Hosted Website (Text Only)

This is the previous symbolic/text workflow, retained for comparison and for
H200-only experiments. It is not the audio-demo architecture: the H200 runs
both its text process and website, while the Mac only views the forwarded page.

On the H200:

```bash
cd /data/home/Andrew.Yang/StreamMUSE/real_rap
conda activate /data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron
streammuse-rap-demo \
  --generator phrase_bank \
  --audio-output none \
  --max-bars 0 \
  --terminal-layout split \
  --terminal-detail full \
  --port 8012
```

On the Mac, forward the old H200 website with any unused local port, then open
the matching local URL:

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L <UNUSED_LOCAL_PORT>:127.0.0.1:8012 \
  Andrew.Yang@masdar
```

Visit `http://127.0.0.1:<UNUSED_LOCAL_PORT>/`. Stop the tunnel with `Ctrl-C`;
it does not stop the H200 process.

To inspect the pre-audio website baseline without changing this branch, create
a detached worktree pinned to the implementation baseline:

```bash
git worktree add --detach <BASELINE_WORKTREE_PATH> 884cd616
```

## Troubleshooting

### `uv` Or Virtual Environment

Run commands as `uv run ...` from the repository root. If a shell resolves a
different executable, compare it with the project environment instead of
installing packages globally:

```bash
which streammuse-rap-demo
.venv/bin/streammuse-rap-demo --help
uv run streammuse-rap-demo --help
```

If `uv` reports a cache permission error, rerun from an authorized terminal or
use the existing `.venv` executable only for diagnosis. `uv sync` is the normal
repair after dependency changes.

### Ports And Tunnels

Check a local port before binding it:

```bash
lsof -nP -iTCP:8020 -sTCP:LISTEN
lsof -nP -iTCP:8012 -sTCP:LISTEN
```

For the normal split demo, Mac port `8020` maps only to the H200 chunk service.
If it is occupied, choose one unused local port consistently in the SSH `-L`
left-hand side and `--rap-render-url`; the H200 destination remains `8020`.
Direct port `8001` forwarding is only for the optional vLLM diagnostic above.
If the website port is occupied, select any unused `--port` and open the
corresponding `127.0.0.1` URL. Do not forward the website for the split
Mac/H200 flow.

### eSpeak And Audio Warnings

`espeak-ng` must be present before audio mode starts. A `synthesis_failed`
warning means the renderer substituted empty vocal PCM for that syllable; drum
PCM and the timing/WAV path can still run. Homebrew eSpeak 1.52 may advertise a
streaming RIFF/data length that is larger than stdout's actual PCM payload. The
adapter validates format and frame alignment, then derives the frame count from
the complete payload rather than trusting that streaming length. Reproduce the
environment check with:

```bash
espeak-ng -D -z -v en-us -s 175 -p 50 --stdout "[['mu:v]]" > /tmp/espeak-check.wav
file /tmp/espeak-check.wav
```

Do not claim vocal PCM evidence until the session has no `synthesis_failed`
warnings and its `audio_render_completed` events report nonzero
`vocal_source_frames`. Transport failures retain a sanitized
`target_url`, exception class, and exception repr in the generator warning and
session artifact; request bodies, credentials, and query strings are excluded.
The complete Task 10 evidence is recorded in
`rap-acceptance-report-2026-08-09.md`.
