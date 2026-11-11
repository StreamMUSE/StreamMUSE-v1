# Inference Page Guide

Mac/H200 services were running and healthy at **2026-11-11 10:08 UTC** (point-in-time verification).
The demo was reset after exports and left stopped: cycle, 60 BPM, USER one bar / MODEL two bars, topic `space`.
Open [Inference](http://127.0.0.1:8018/inference).
[Rap Lab](http://127.0.0.1:8018/) is the original page in the **same app/session**; opening/refreshing does not start/reset.
Audio comes from Mac Python and its selected output device, not the browser.

## Cyclic exchange

1. Stop any active performance. Select **Cyclic exchange**, set BPM and USER/MODEL durations,
   then **Apply settings**. Default: 60 BPM, USER one bar, MODEL two bars. Real backend: 4/4,
   four ticks/beat, whole-bar turns only; BPM 30-240. Choosing Beats does not enable beat generation.
2. Click **Start**. The USER capture surface receives focus; click it again if focus moves.
   Tap **Space** during USER over the drums. Holding Space, modified keys and typing elsewhere do not capture.
3. At USER end, **WAIT is silent** and shows elapsed time, not a countdown. MODEL starts on
   the first playable decoded prefix, with drums at its own downbeat; leading vocal rests are preserved.
4. MODEL finishes and USER repeats automatically. No captured rhythm repeats USER with
   `No rhythm captured`; it does not generate an invented response.
5. **Stop** during WAIT is immediate; during USER/MODEL it finishes the current bar/phase.
   Wait for stopped, then **Start** for a fresh cycle. **Reset is not required**; it clears history/metrics.

## Continuous steering

While stopped, choose **Continuous steering** and **Apply settings**. Set Topic / prompt and
Flow, then **Apply steering** and **Start**. Silent PREPARING ends when the prefix is playable.
While playing, edit Topic / prompt or Flow and click **Apply steering** again. Compare Requested
input with Playing input: committed audio stays; late replacements may retain old prepared audio
as **stale context**. If neither is ready, missing intervals play drums only without stretching time.
Earliest eligible replacement is not immediate: the verified edit took about 7.48 seconds while old audio continued.
Stop/Start keeps latest intent and performed lyric history, but discards invalidated queued audio.

Mode, BPM and turn settings require stopped playback. In the audio gear popover, device and
drum/vocal gains are **stopped-only**, capability-dependent, and apply at next Start, not live.
Candidate budget/concurrency are currently backend-fixed. Another tab is monitor-only for cyclic
capture; it may use global Stop/Reset. Owner heartbeat expires after six seconds: WAIT stops
immediately, USER/MODEL completes its phase. Continuous mode survives navigation/disconnection.

## Reading results

Response start measures WAIT to MODEL callback start; missing counts MODEL only, not WAIT. TTFT `--` means unavailable.
Request `completed_missing` / `missing_frames` means that request's own output did not cover its whole range;
old/stale audio may have covered it, so this is not the global Missing vocals metric.
Candidate counts cover selected-lyric decisions, not all LLM work/cost: earlier failed writer attempts are omitted.
Retry warnings remain in traces and canceled requests in bounded history; neither makes those counts complete.
Callback/intended-onset timing is not acoustic measurement. Expand details; export JSON/CSV before Reset.
On errors, Stop, check connection/device, and Start without Reset when recovered.

## Launch only when needed

Do not relaunch healthy services. Commands reuse the environment/SSH alias `H200`, without install/config/source sync.
First inspect current state:

```sh
cd "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/MBZUAI/StreamMUSE-NewSystem/.worktrees/real_rap"
curl --fail --silent --show-error http://127.0.0.1:8018/api/inference/state
ssh -S /tmp/streammuse-inference-20261111.sock -O check H200
ssh H200 'cd /data/home/Andrew.Yang/StreamMUSE/deploy/rap_inference_20261111 && python3 deploy_h200.py status render'
```

Only if the tunnel is absent and local ports 18711/18721 are free, run on the Mac:

```sh
ssh -M -S /tmp/streammuse-inference-20261111.sock -fNT -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18711:127.0.0.1:18711 -L 127.0.0.1:18721:127.0.0.1:18721 H200
curl --fail --silent --show-error http://127.0.0.1:18721/health
.venv/bin/python scripts/experiments/inference_milestone/launch_mac.py --port 8018 --renderer http://127.0.0.1:18721 --device "MacBook Pro Speakers" --output output/inference_20261111/client
```

Launch only after the old client exits and 8018 is free; never bypass live-PID/port checks by deleting its record.
Logs: `output/inference_20261111/client/client.log`. For an absent H200 stack in the **already prepared** deployment:

```sh
ssh H200
cd /data/home/Andrew.Yang/StreamMUSE/deploy/rap_inference_20261111
python3 deploy_h200.py start vllm
python3 deploy_h200.py start tts
python3 deploy_h200.py wait vllm --timeout 2400
python3 deploy_h200.py wait tts --timeout 2400
python3 deploy_h200.py start render
python3 deploy_h200.py wait render --timeout 2400
python3 deploy_h200.py verify
exit
```

This is the deployed copy of [deploy_h200.py](../../scripts/experiments/inference_milestone/deploy_h200.py).
Qwen: 18711/GPU 1; renderer: 18721/GPU 2; MOSS: 18730/GPU 2. Do not rerun `prepare`, launch
duplicates, or kill by port/name. For a partial stack, inspect each service and start only absent ones.
See [deployment ownership/restart details](../reports/2026-11-11-inference-h200.md#restart-and-source-sync-handoff).

## Verification status

[Final code checks](../reports/2026-11-11-inference-milestone-progress.md#final-verification): 2055 scoped rap,
88 legacy GUI and 81 Node tests passed; final review's 10 targeted tests passed with no remaining findings.
[Final-source five-cycle evidence](../../output/inference_20261111/final-source-cycle/summary.json): ten MODEL bars / 40 s,
all requests played; zero missing/stale/fallback coverage, errors/cancellations, callback warnings/errors or recording drops.
Response-start p50 **5.337 s**, p95 **5.973 s**, maximum **6.132 s** (n=5).
[Final desktop/mobile checks](../../output/inference_20261111/browser/summary.json) passed with no overflow/page errors, epoch 7 stopped.
[Real browser capture](../../output/inference_20261111/browser-cycle/evidence.json) passed: seven Space taps,
leading tick 2, populated MODEL syllable grid, refresh preserving epoch/owner, and original-page Stop
enable/disable updates. See the [MODEL screenshot](../../output/inference_20261111/browser-cycle/model-playing.png).
[Final-source steering](../../output/inference_20261111/final-source-steering/evidence.json) passed: new input played after
**7.480467 s**, with **7.465292 s retained stale audio** and zero global missing vocals at transition.
Restart/history, both directions of cross-page Stop, exclusive ownership and session-preserving Reset passed.
No retry warning occurred in that run; retry recovery is covered by deterministic tests, not live evidence.
Listener compatibility verification passed. Subsequent user listening confirmed good playback,
but a marked tone/voice change between parts remains unresolved; acoustic timing was not measured.
See the [progress ledger](../reports/2026-11-11-inference-milestone-progress.md) and [engineering handoff](inference-optimization-handoff.md).
