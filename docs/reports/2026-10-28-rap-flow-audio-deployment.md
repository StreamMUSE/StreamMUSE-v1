# RAP Flow/Audio H200 Deployment, 2026-10-28

## Scope

- Local worktree: `/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE-NewSystem/.worktrees/real_rap`.
- Remote root: `/data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028`.
- Controller: `scripts/experiments/rap_flow_audio_20261028_deploy.py`, uploaded as remote `deploy.py`.
- Source copied from the current, concurrently edited worktree: `src/` and the required `scripts/rap_audio_backends/`. No rsync deletion, package installation, environment synchronization, or model download.
- The existing `rap_v3_gui_efd621ed_gpu1_20261014/deploy.py` is imported only for `command()`, following the verified research launcher's approach. Its start/stop functions are never invoked.
- No old services or unrelated GPU workloads are stopped. No local tunnel is created. Product files are not edited by this deployment task.

## Preflight

The authoritative recorded preflight was at `2026-10-28T20:29:02.395509+00:00`:

| Resource | Observation |
| --- | --- |
| GPU0 | 0 MiB, no compute process; `GPU-4f4e1a5f-dfd1-e747-40d2-993bf2548918` |
| GPU1 | 0 MiB, no compute process; `GPU-74ee34a6-23bd-7e60-e012-6bf3d30b4c3a` |
| Ports 18711/18720 | Both bindable on loopback |
| GPUs4/5 | Existing VLLM workers 278215/278214, left untouched |
| Ports 18611/18620 | No listener in initial inspection; no process was stopped |

Each start repeats the relevant port/GPU check immediately before launching. A GPU UUID/index mismatch, any compute process on the assigned GPU, or memory use above 16 MiB refuses launch. The small memory allowance permits driver overhead; both assigned GPUs actually reported zero before their starts.

## Runtime Configuration

| Service | Physical GPU | Endpoint | Existing environment |
| --- | --- | --- | --- |
| Qwen/vLLM | 0 | `http://127.0.0.1:18711` | `/data/home/Andrew.Yang/qwen-extract-venv` |
| In-process MOSS/MMS renderer | 1 | `http://127.0.0.1:18720` | `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/moss` |

`CUDA_VISIBLE_DEVICES` uses the exact physical UUID, so `cuda:0` inside the renderer means physical GPU1. Both services bind only to `127.0.0.1`.

- Qwen: `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`, served as `qwen-rap`; maximum context 2048, maximum sequences 48, GPU memory utilization 0.25.
- MOSS: `OpenMOSS-Team/MOSS-TTS-v1.5`, revision `cdd3b911b1585e3f2dbc7775ef10f9926f58850a`; in-process, `cuda:0`, reference audio `/data/home/Andrew.Yang/StreamMUSE/audio-protocol-downloads/MOSS-TTS/assets/audio/reference_02_s1.wav`.
- MMS: `cuda:0`; pre-existing cache `/data/home/Andrew.Yang/StreamMUSE/assets/rap-audio-protocols/mms-cache`, containing `checkpoints/model.pt` (1,262,047,414 bytes).
- Renderer: concurrent bar generation, realtime candidate profile, 32 lyric candidates, `gentle_sparse_r3` warp policy, PCM wire audio, lyric streaming enabled.
- Shared installed dependencies are read from the old deployment's `deps/`; current application/backend source comes from the new `repo/`.
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `PYTHONDONTWRITEBYTECODE=1`. Existing Hugging Face model cache is reused. Generated HF modules, vLLM compilation cache, and render artifacts are under the new root.

## Identity And Evidence

Both services are healthy and intentionally left running. The parent confirmed health and started a real two-bar comparison request. No further restart or resync is authorized without coordination; subsequent UI-only review changes do not require a server restart.

| Process | PID | PGID / SID | Linux start ticks | GPU evidence |
| --- | --- | --- | --- | --- |
| Qwen API leader | 2125763 | 2125763 / 2125763 | 77713409 | Owns worker below |
| Qwen engine worker | 2126249 | 2125763 / 2125763 | 77715727 | Physical GPU0, 35,172 MiB |
| Qwen resource tracker | 2126248 | 2125763 / 2125763 | 77715727 | No GPU allocation |
| Final MOSS/MMS renderer | 2134130 | 2134130 / 2134130 | 77752992 | Physical GPU1, 25,736 MiB |

All have UID `1010` and boot ID `0c8cdfa4-e042-4b56-a127-f6ce974485dd`. Qwen started at `2026-10-28T20:29:11.845847+00:00`; the final renderer started at `2026-10-28T20:35:47.677571+00:00`. Each GPU compute PID was explicitly matched to the verified service's live process membership, not inferred from the configured GPU flag. GPUs4/5 retain the unrelated workers recorded above.

Verified endpoints (Qwen evidence at `2026-10-28T20:36:37.719137+00:00`, renderer evidence at `2026-10-28T20:36:39.182157+00:00`):

| Endpoint | Result |
| --- | --- |
| `http://127.0.0.1:18711/health` | HTTP 200, empty body as expected for vLLM |
| `http://127.0.0.1:18711/v1/models` | HTTP 200; `qwen-rap`, root `Qwen/Qwen2.5-7B-Instruct`, context 2048 |
| `http://127.0.0.1:18720/health` | HTTP 200; `ready=true`, `state=ready`, `backend=inprocess`; Qwen serving, MOSS and MMS warmed |
| `http://127.0.0.1:18720/v1/rap/capabilities` | HTTP 200; schema `streammuse.rap_lyric_stream.v3`; `tempo_updates_v1`, `bounded_turns_v1` |

Final MOSS/MMS startup synthesis/alignment warmup took 6,307.66 ms. MOSS health confirms the exact model revision above and `PersistentMossSynthesizer`; MMS confirms `torchaudio.pipelines.MMS_FA`, `torchaudio 2.9.1+cu128 / MMS_FA`. Rubber Band 3.3.0 is available. Producer fingerprint: `b7e39229fd039b535ca5a8ea5e82e64b0b68153bacb919fb66373b2d8738cb5c`.

The initial copied source manifest is `78869316f0143fd289cfa367a30e7bcb5db06636785f27a38d8a44880355d268`. Every start writes `source-<sha256>.json`, listing the content hash of every copied source file. This identifies a point-in-time snapshot, not a clean Git commit or a promise that parallel agents have finished.

After the parent reported all product agents finished, source was recopied without deletion and only the owned renderer was restarted. Final renderer source manifest: `dc617bc3ecd30406d7a8f07f1c198e4f75ceffb6ba6c4955ed0df71fb0eba937`. This includes the parent's post-resample leading-rest zeroing in `src/streammuse/application/rap/bar_audio.py`, SHA-256 `13e30db58c1fa77e3910a5ca87364b7c1f17d36f506a0de4f93c70b7a7866a4e`. Qwen retains its original launch-time source manifest because it does not load the application source. Use the full source manifest, not the producer fingerprint alone, to identify this source snapshot.

A subsequent checksum-only rsync dry run found later local changes in `src/streammuse/application/rap/settings.py` and `src/streammuse/presentation/rap_demo/static/js/rap-settings.js`. It made no writes. The parent explicitly classified the review changes as not requiring a server refresh and requested no further restart/resync while the real comparison runs; that instruction was honored. The manifest above remains the authoritative deployed snapshot.

Final operational controller SHA-256: `01545d7c6a81e40a668dbb0063da562a64621a436feeef9d946d6eb0d0a3ec53`. PID records retain the controller hash that existed at their launch, before the final regression-test additions.

Remote evidence files, relative to the new root:

- `preflight.json`, `vllm.preflight.json`, `render.preflight.json`.
- `vllm.pid.json`, `render.pid.json`: PID, PGID, SID, UID, Linux start ticks, boot ID, ownership token, exact argv/environment, cwd, port, physical GPU UUID, source and launcher hashes.
- `vllm.command.txt`, `render.command.txt`: shell-reproducible argv and environment. Use the controller for launching, since running these commands directly bypasses safety checks.
- `vllm.log`, `render.log`: startup and request logs.
- `vllm.status.json`, `render.status.json`: health/model/capability responses and GPU process snapshots.
- `verification-final.json`: combined healthy evidence, including all process identities and GPU allocation records.
- `offline-module-cache-proof.json`: byte-identical SHA-256 comparisons for the three MOSS dynamic Python modules against the pinned pre-existing Hugging Face snapshot.

Transformers logged its generic "new version ... downloaded" warning while populating the new dynamic-module directory. Both offline flags remained set; all three mentioned modules were verified byte-identical to the existing pinned cache. No package manager, model download, or reinstall command was run.

### Startup Control Issues Resolved

The first Qwen readiness monitor found HTTP 200 but rejected the engine worker because `/proc/<worker>/environ` did not expose inherited markers. Only that exact monitor process was stopped using a PID-fd; Qwen was not stopped. The controller now accepts an environmentless worker only with verified live ancestry to the token-verified leader plus matching UID, session, group, boot ID, and valid start time. Conflicting visible ownership markers still fail closed.

The first renderer shutdown sent SIGTERM to the correct recorded processes, then encountered a `/proc/environ` exit race while observing completion. All recorded processes were independently verified gone, with no surviving process group, no listener, and empty GPU1, before archiving `render.stopped-1793219746222188905.pid.json`. No extra signals were used. Completion now checks pinned identities without rereading dying processes' environment. Port preflight uses `SO_REUSEADDR` to permit closed TIME_WAIT sockets, while the live-listener refusal test still passes.

## Parent Commands

Run these from a machine with the existing `H200` SSH alias. There is no local tunnel managed by this task.

```bash
ssh H200 'python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py status all'
ssh H200 'curl -fsS http://127.0.0.1:18711/health'
ssh H200 'curl -fsS http://127.0.0.1:18711/v1/models'
ssh H200 'curl -fsS http://127.0.0.1:18720/health'
ssh H200 'curl -fsS http://127.0.0.1:18720/v1/rap/capabilities'
```

The native capabilities route is `/v1/rap/capabilities`, not `/capabilities`. Qwen exposes `/health` and `/v1/models`; no unsupported alias was added to product code.

### Resync After Product Agents Finish

Coordinate with the parent to avoid interrupting tests. Stop only the owned renderer, copy source without deletion, then restart and wait. Qwen does not need a restart for application source changes.

```bash
ssh H200 'python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py stop render'
cd '/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE-NewSystem/.worktrees/real_rap'
rsync -azR --exclude=__pycache__ --exclude='*.pyc' src scripts/rap_audio_backends H200:/data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/repo/
ssh H200 'python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py start render'
ssh H200 'python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py wait render --timeout 600'
```

Rsync intentionally retains remote files removed locally. If agents delete or rename source files, compare the manifest before treating a resync as an exact snapshot; do not silently add `--delete`.

### Owned Shutdown

```bash
ssh H200 'python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py stop render'
ssh H200 'python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py stop vllm'
```

Shutdown requires the recorded leader's exact boot/PID/start-time/UID/PGID/SID identity and matching per-service token/root/GPU environment. Every group member needs live ancestry to that leader and matching boot/UID/PGID/SID; visible ownership markers must agree. The controller opens Linux PID-fds, rechecks identity, and sends only SIGTERM to those pinned processes. It never uses `pkill`, `killall`, numeric process-group broadcasts, or SIGKILL. A missing/reused leader, unexpected group member, surviving new group member, or 30-second timeout fails closed and retains the record for inspection. Successful shutdown archives the PID record and saves the signaled identities. Start refuses any existing active record, including stale records.

## Verification

- Initial safety tests were observed failing before implementation, then passing.
- H200: all seven final embedded tests passed, including real port-collision refusal, real PID-fd shutdown of a disposable CPU-only process and its environmentless child, ancestry enforcement, and proof that a tampered identity sends no signal. Command: `python3 /data/home/Andrew.Yang/StreamMUSE/deploy/rap_flow_audio_20261028/deploy.py --self-test`.
- Mac sandbox: the socket test was blocked by bind permissions; Linux-only shutdown was skipped. The complete H200 run covers both.
- Product tests and feature acceptance remain parent-owned; this task verifies deployment readiness and model residency, not speaker output or the final concurrently edited feature.
