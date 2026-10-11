# StreamMUSE / Offline Pre-Model Input Audit on H200

Date: 2026-08-14

## Question

For the same melody, continuation checkpoint, BPM conditioning, and greedy decoding,
does StreamMUSE pass the same token sequence to the continuation model as the native
offline inference path?

## Observation Point

The hook records the complete `input_ids` immediately before generation. It does not
call the model again and does not modify the scheduler, tokenizer, model, or decoded
events.

```text
StreamMUSE public MIDI client
  -> HTTP backend
  -> MIDI events to beat pianoroll/tokens
  -> [pre-model input_ids hook]
  -> continuation model
  -> [raw output-token hook]

Native offline inference
  -> NPZ generation schedule
  -> [pre-model input_ids hook]
  -> continuation model
  -> [raw output-token hook]
```

## H200 Conditions

- Host: `MBZ-Martina` (`masdar`).
- GPU: GPU 0, NVIDIA H200 NVL.
- StreamMUSE commit: `3a264d56ee574c68b3b23f49c65a39535c192bc3`.
- Test piece: song 4.
- StreamMUSE melody: `prompts/inputs_lekai/mel/4.mid`.
- Offline data: reconstructed `4.npz` used by the existing consistency test.
- Checkpoint: `RT-accompanimentV2/checkpoints-resume/epoch_15_0307_1858/model.safetensors`.
- Checkpoint SHA-256: `d93139044a8614aeb66c58b5696371a575389199fbe64653b8994d3f6b056271`.
- Conditioning and playback BPM: 120 on both paths.
- Device and dtype: CUDA, float16.
- Decoding: deterministic greedy on both paths.
- Test entry: the existing public MIDI client consistency test, not a direct backend shortcut.

The checkpoint, NPZ, and melody MIDI hashes match the files used in the local setup.

## H200 Result

| Comparison | Result |
|---|---:|
| StreamMUSE requests | 80 (includes 24-beat tail) |
| Missing or duplicate realtime requests | 0 |
| Comparable target beats | 59 (beats 1-59) |
| Exact pre-model `input_ids` matches | 59 / 59 |
| Input digest matches | 59 / 59 |
| Exact raw model-output matches | 59 / 59 |
| Realtime round trip, mean | 58.77 ms |
| Realtime round trip, median | 55.00 ms |
| Realtime round trip, p95 | 68.52 ms |
| Realtime round trip, max | 334.72 ms |

The existing consistency test passed in 63.68 seconds. Offline contains one additional
target, beat 0. StreamMUSE starts requesting at beat 1, so the aligned comparison scope
is beats 1-59. StreamMUSE continues through beat 80 to preserve the existing 24-beat
playback tail; those tail beats have no offline counterpart and are not compared.

## Interpretation

For this controlled H200 case, StreamMUSE's MIDI/event-to-continuation-token path is
exactly equivalent to the native offline path at the model boundary. The continuation
model also returns exactly the same tokens at every aligned beat. This run provides no
evidence of a tokenizer or pre-model encoding defect in the standard continuation path
at commit `3a264d56`.

The earlier RTX 3050 run was a preliminary local control and is not the requested primary
result. Its first 120 BPM attempt also had confounds: mismatched BPM conditioning and
merged requests under local latency. The H200 result above has neither confound.

This is a one-piece controlled audit. It does not prove equivalence for every MIDI file,
time signature, tempo bucket, prompt-continuation mode, or quantization edge case. If an
audible difference remains for this exact standard-continuation case, localization should
continue after raw model output: token decoding, event conversion, scheduler, or playback.

## Reconciliation With The Earlier Audit

The 2026-07-31 audit reported a two-block exchange on beats 36-51. Rechecking the raw
artifacts shows that the observed difference was real, but its attribution to StreamMUSE
was wrong.

| Artifact comparison | Exact matches | Meaning |
|---|---:|---|
| Old StreamMUSE vs H200 StreamMUSE | 80 / 80 | StreamMUSE output did not change |
| Old offline vs current offline without BPM override | 60 / 60 | Old oracle used NPZ BPM conditioning |
| Old offline vs corrected BPM-120 offline | 44 / 60 | Only beats 36-51 change |
| H200 StreamMUSE vs corrected BPM-120 offline | 59 / 59 | Correct controlled comparison passes |

The NPZ metadata contains BPM 57. Without an explicit override, offline starts with BPM
token 264. The public client was run at 120 BPM and StreamMUSE starts with BPM token 265.
The earlier report claimed both sides used 120, but its offline runner did not actually
apply that override. This one-token conditioning difference caused the structured
36-51 output difference.

This was verified with a same-H200 A/B run. With all other variables fixed, BPM 57
reproduces the old offline output on 60/60 beats. BPM 57 versus BPM 120 differs only on
beats 36-51. The corrected BPM-120 offline output then matches StreamMUSE on all 59
aligned beats.

The earlier slow measurements also came from different environments or paths. The RTX
3050 could merge requests at 120 BPM, and the historical stable backend took about one
second per request. Neither timing describes the current H200 path, whose median round
trip is 55.00 ms at 120 BPM with no skipped requests.

The statement that later calls contain more than 100 tokens refers to accumulated input
context, not to one beat generating more than 100 tokens. In this H200 run, pre-model
input grows from 10 tokens at beat 1 to 132 at beat 16, 483 at beat 59, and 594 at beat
80. Each beat actually generates only 2-3 output tokens. Input length and end-to-end
round-trip latency have a weak Pearson correlation of 0.194 in this run; the growing
context is real architectural work, but it does not make the H200 path miss 120 BPM.

## Artifacts

Local copy:

- `output/consistency/20260814-h200-input-audit-bpm120/`
- `input_comparison.json`
- `round_model_output_comparison_song4.json`
- `streammuse_model_inputs.jsonl`
- `offline_model_inputs_song4.jsonl`
- `streammuse_model_outputs.jsonl`
- `offline_model_outputs_song4.jsonl`
- `inferences.json`
- `server.log`

Remote source:

- Worktree: `/data/home/yuanxin/StreamMUSE-v1-h200-input-audit-20260814`
- Run: `output/consistency/20260813-210718/`
