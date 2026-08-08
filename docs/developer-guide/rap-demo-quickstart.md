# Real-Time Rap Demo Quick Start

## Open the H200 Checkout

```bash
cd /data/home/Andrew.Yang/StreamMUSE/real_rap
conda activate /data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron
```

The checkout should be on `feature/real_rap`. Confirm the commands resolve from
the StreamMUSE environment:

```bash
git branch --show-current
which streammuse-rap-demo
```

## Run Without a Model

Use the deterministic phrase bank to check the real-time clock, candidate gate,
ranking, fallback behavior, terminal monitor, and browser monitor:

```bash
streammuse-rap-demo \
  --generator phrase_bank \
  --max-bars 0 \
  --terminal-layout split \
  --terminal-detail full \
  --port 8012
```

`--max-bars 0` runs continuously until interrupted. Use a positive number for a
finite run.

## Run With a Real LLM

First start one OpenAI-compatible model server on an unused GPU:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen-rap \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.25
```

In a second terminal, activate the same Conda environment and run:

```bash
streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 \
  --model qwen-rap \
  --max-bars 0 \
  --terminal-layout split \
  --terminal-detail full \
  --port 8012
```

The model proposes candidates; it does not control the clock. Candidates with
the wrong syllable count, low score, errors, or late responses are rejected and
the prevalidated fallback remains available.

## Read the Monitor

- **Live delivery** shows the frozen lyric, sounding syllable, beat/subdivision,
  stress, jitter, generation latency, deadline slack, and fallback state.
- **Beat-aligned flow** shows the moving tick playhead plus the exact tick,
  duration, stress, boundary, rhyme, and template-provenance arrays.
- **Generation audit** shows frozen context, the role-labelled LLM prompt, raw
  response, token/deadline/error diagnostics, and the active request identity.
- **Candidate gate and ranking** shows observed versus required syllables,
  validity, rejection reasons, OOV words, and every score value, weight, and
  contribution. The winning-score strip isolates the selected breakdown.
- **Queue and health** shows the next reserved lyric, lifecycle state, and last
  retained error. History and the canonical event console remain available
  below the cumulative metrics.
- Clear **Follow live** to inspect older events without automatic scrolling;
  generation and all other page updates continue normally.
- **Replaced** means a candidate passed validation, ranking, threshold, and the
  planning deadline.
- **Frozen** is the authoritative line being emitted; later responses cannot
  change it.
- **Fallback** means no generated candidate was usable in time. The first bar
  intentionally uses a fallback so playback can start immediately.
- `*` marks a stressed syllable in stream output.

Unless `--no-web` is supplied, open `http://127.0.0.1:8012`. When connecting to
the H200 remotely, use the corresponding SSH-forwarded local URL.

## Stop and Inspect Results

Press `Ctrl-C` for a clean shutdown. Each run writes a directory under
`logs/rap/<session-id>/` containing:

- `session.json`: scenario, model, template, scoring, and environment settings.
- `events.jsonl`: complete ordered runtime evidence.
- `bars.csv`: one row per frozen bar.
- `summary.json`: candidate, fallback, latency, deadline, and jitter metrics.

The current prototype schedules and visualizes symbolic syllables. It does not
yet produce speech or accept live drum input.
