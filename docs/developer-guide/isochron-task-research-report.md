# StreamMUSE Isochron Task Runtime Research Report

## Summary

This branch adds a general realtime LLM task runtime to StreamMUSE and validates it with two ProjectIsochron-style tasks:

- **Zip-Zap-Zop**: deterministic symbolic rule-following.
- **Animal Naming**: open-ended semantic generation with deterministic referee validation.

Both tasks run through the same StreamMUSE task contract, local OpenAI-compatible model client, trace recorder, and runner modes:

- `offline_benchmark`: runs as fast as the local model server responds.
- `realtime_loop`: runs on a fixed tick/deadline schedule.

The main research result is that the runtime path works end to end with a local open-source LLM server on H200. It records task state, prompts, responses, referee results, token counts, latency, and deadline misses. The traces let us separate **runtime/timing failures** from **model/task failures**.

## Experimental Setup

Remote machine:

- Host: `H200`
- GPU: NVIDIA H200 NVL
- Repo: `/data/home/Andrew.Yang/StreamMUSE/StreamMUSE-v1`
- Env: `/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron`

Local model server:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8012 \
  --served-model-name local-model \
  --gpu-memory-utilization 0.35 \
  --max-model-len 4096
```

Client endpoint:

```text
http://127.0.0.1:8012/v1/chat/completions
```

Task command shape:

```bash
streammuse-task run \
  --task <task_name> \
  --runner <offline_benchmark|realtime_loop> \
  --max-turns <N> \
  --model-url http://127.0.0.1:8012/v1 \
  --model local-model \
  --temperature 0 \
  --max-tokens 8 \
  --output-dir task_runs/<experiment_name>
```

For live demos, add `--live-output` to print each completed turn directly in the terminal:

```bash
streammuse-task run \
  --task animal_naming \
  --runner realtime_loop \
  --max-turns 20 \
  --model-url http://127.0.0.1:8012/v1 \
  --model local-model \
  --temperature 0 \
  --max-tokens 8 \
  --tick-rate-hz 1 \
  --deadline-ms 1000 \
  --live-output \
  --output-dir task_runs/live_demo
```

Example terminal output:

```text
[turn 000] OK response='Dog' reason=NONE latency_ms=24.4
[turn 001] OK response='cat' reason=NONE latency_ms=23.3
[turn 002] OK response='elephant' reason=NONE latency_ms=32.2
```

For Zip-Zap-Zop, deterministic expected outputs are also shown:

```text
[turn 000] OK response='1' expected='1' reason=NONE latency_ms=24.2
[turn 001] OK response='2' expected='2' reason=NONE latency_ms=22.7
[turn 002] OK response='Zip' expected='Zip' reason=NONE latency_ms=23.0
```

## Implementation Details

The implementation uses the new generic task abstraction:

```text
RealtimeTask
TaskState
TaskTurn
TaskRefereeResult
TaskRuntime
LocalChatModelClient
JsonlDebugTraceRecorder
```

Each task implements:

```text
initial_state()
build_turn(state)
validate_response(state, response_text)
advance_state(state, referee_result, response_text)
```

The runtime owns:

- model request orchestration
- offline vs realtime loop scheduling
- deadline measurement
- validation/referee calls
- state progression
- trace artifact writing

This is important because adding a new task does not require another client/server path. New tasks plug into the same runtime and produce the same trace format.

## Task 1: Zip-Zap-Zop

### Task Definition

The task asks the model to transform each number according to deterministic rules:

```text
multiples of 3 -> Zip
multiples of 4 -> Zap
multiples of 5 -> Zop
combined multiples concatenate
otherwise output the number
```

Examples:

```text
1 -> 1
3 -> Zip
4 -> Zap
5 -> Zop
12 -> ZipZap
15 -> ZipZop
20 -> ZapZop
```

### Prompt

System prompt:

```text
You are playing Zip-Zap-Zop. Respond with only the next value. Multiples of 3 are Zip, multiples of 4 are Zap, multiples of 5 are Zop, and combined multiples concatenate those words. Otherwise respond with the number.
```

User prompt example:

```text
12:
```

Expected response:

```text
ZipZap
```

### Results

Qwen 7B, 20 turns:

| Runner | Turns | Valid | Invalid | Deadline Misses | Latency min/avg/max |
|---|---:|---:|---:|---:|---:|
| `offline_benchmark` | 20 | 16 | 4 | 0 | 19.5 / 45.6 / 402.7 ms |
| `realtime_loop` | 20 | 16 | 4 | 0 | 23.9 / 31.5 / 45.8 ms |

Offline and realtime produced identical responses:

```text
response_diffs: 0
```

Invalid examples:

| Number | Expected | Model Response |
|---:|---|---|
| 10 | `Zop` | `ZapZop` |
| 14 | `14` | `Zap` |
| 15 | `ZipZop` | `ZipZapZop` |
| 18 | `Zip` | `Zip-Zap` |

### Interpretation

The runtime and local server path were stable. There were no deadline misses, and offline/realtime responses matched exactly. The remaining failures were model reasoning or prompt-following errors, not scheduling failures.

## Task 2: Animal Naming

### Task Definition

Animal Naming asks the model to name one real animal per turn without repeating earlier answers.

The task state tracks:

```text
used_animals
attempted_animals
history
```

The referee checks:

- response is non-empty
- normalized animal is in a deterministic whitelist
- animal has not already been used

Invalid attempts are also added to `attempted_animals`, so future prompts can forbid both valid previous answers and invalid repeated attempts.

### Prompt

First turn:

System prompt:

```text
You are playing Animal Naming. Respond with exactly one common real animal name. Never output an animal listed as forbidden. Do not explain your answer.
```

User prompt:

```text
Name an animal:
```

Later turn example:

```text
Forbidden animal names: dog, cat, elephant, lion
Choose one animal name that is not in the forbidden list. Output only the new animal name:
```

### Results: 20-Turn Run

Qwen 7B, 20 turns:

| Runner | Turns | Valid | Invalid | Deadline Misses | Latency min/avg/max |
|---|---:|---:|---:|---:|---:|
| `offline_benchmark` | 20 | 20 | 0 | 0 | 17.8 / 21.8 / 31.2 ms |
| `realtime_loop` | 20 | 20 | 0 | 0 | 22.4 / 27.9 / 38.0 ms |

Offline and realtime matched exactly:

```text
response_diffs: 0
```

Outputs:

```text
Dog, cat, elephant, lion, giraffe, zebra, cow, deer, monkey, rhino,
fox, bear, camel, bird, snake, tiger, horse, panda, chimpanzee, wolf
```

### Results: 100-Turn Stress Test

The stress test was run up to 100 turns to find the first invalid response.

| Runner | Turns | Valid | Invalid | First Invalid Turn | Deadline Misses | Latency min/avg/max |
|---|---:|---:|---:|---:|---:|---:|
| `offline_benchmark` | 100 | 29 | 71 | 27 | 0 | 18.0 / 24.1 / 31.0 ms |
| `realtime_loop` at 10 Hz | 100 | 29 | 71 | 27 | 0 | 21.3 / 33.2 / 38.0 ms |

First invalid response:

```text
turn 27
response: foxhound
reason: UNKNOWN_ANIMAL
```

Sequence through the first failure:

```text
00 Dog
01 cat
02 elephant
03 lion
04 giraffe
05 zebra
06 cow
07 deer
08 monkey
09 rhino
10 fox
11 bear
12 camel
13 bird
14 snake
15 tiger
16 horse
17 panda
18 chimpanzee
19 wolf
20 koala
21 penguin
22 chicken
23 duck
24 kangaroo / gorilla depending on runner
25 crocodile / kangaroo depending on runner
26 gorilla / crocodile depending on runner
27 foxhound -> invalid
```

Observed failure modes after turn 27:

- Offline mostly generated invented `fox...` compounds such as `foxsnake`, `foxbat`, and `foxquirrel`.
- Realtime eventually repeated `kangaroo`, causing `REPEATED_ANIMAL`.
- Both modes had zero deadline misses.

### Interpretation

Animal Naming shows a different class of model failure than Zip-Zap-Zop. The runtime can keep stepping and tracing reliably, but the model begins to struggle as the forbidden list grows. This makes the task useful for studying context-following, repetition avoidance, and semantic validity under realtime constraints.

## Comparison

| Task | Task Type | Best 20-Turn Result | Stress Result | Main Failure Mode |
|---|---|---:|---:|---|
| Zip-Zap-Zop | deterministic symbolic rules | 16/20 | not run to 100 | arithmetic/rule mistakes |
| Animal Naming | open-ended semantic generation | 20/20 | first invalid at turn 27 | invented names or repeats |

Both tasks met realtime deadlines comfortably with Qwen 7B on H200.

## What This Demonstrates

This work demonstrates that StreamMUSE can now run local open-source LLM tasks through a common realtime benchmarking framework:

```text
Task -> Prompt -> Local LLM Server -> Response -> Referee -> State Update -> Trace
```

The key research value is that we can compare:

- offline vs realtime behavior
- deterministic vs semantic tasks
- latency and deadline behavior
- response reproducibility
- first failure point
- model-specific failure modes

For the mentor presentation, the strongest evidence is:

```text
Zip-Zap-Zop:
offline and realtime responses matched exactly; 0 deadline misses.

Animal Naming:
20-turn offline and realtime runs were 20/20 valid; 0 response diffs; 0 deadline misses.

Animal Naming stress:
first invalid at turn 27; no deadline misses; trace identifies model failure mode.
```

## Current Limitations

- Animal Naming uses a fixed whitelist, so some real animals must be added manually.
- The runtime currently completes all requested turns; it does not yet have a built-in `--stop-on-invalid` flag.
- The trace viewer can read task traces indirectly through artifacts, but the web UI is not yet optimized for task-specific prompt/response/referee views.
- Zip-Zap-Zop currently sends only the current number plus rules, not a full prior sequence.
- Results are from Qwen 2.5 7B Instruct at `temperature=0`; other local models may behave differently.

## Recommended Next Steps

1. Add `--stop-on-invalid` to `streammuse-task run`.
2. Add task-trace views to the replay debugger: prompt, response, referee, state, latency.
3. Add a richer animal ontology or synonym table.
4. Add more ProjectIsochron tasks to test different cognitive patterns.
5. Run the same tasks against multiple local models and compare first-failure turns, latency, and determinism.
