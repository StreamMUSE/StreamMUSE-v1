# StreamMUSE / Offline Per-Beat Model Output Audit

Date: 2026-07-31

## Correction (2026-08-14)

The original attribution below is not valid. Although the report stated that BPM was
pinned to 120 on both paths, the offline runner still used the NPZ metadata BPM of 57.
Consequently, StreamMUSE used BPM token 265 while offline used BPM token 264.

A direct artifact comparison established:

- the 2026-07-31 StreamMUSE trace and the 2026-08-14 H200 StreamMUSE trace match on
  80/80 beats;
- the 2026-07-31 offline trace and a 2026-08-14 offline rerun without BPM override
  match on 60/60 beats;
- overriding offline BPM to 120 removes the reported beat 36-51 mismatch, producing
  59/59 aligned input-token matches and 59/59 output-token matches on H200.

A same-H200 A/B rerun confirms the cause without changing hardware, code, checkpoint,
NPZ, or decoding: BPM 57 reproduces the old offline trace on 60/60 beats, while changing
only the BPM override to 120 changes exactly beats 36-51 and matches StreamMUSE on 59/59
aligned beats.

Therefore, the two exchanged eight-beat blocks were caused by unequal BPM conditioning
in the audit, not by a StreamMUSE context-order or encoding defect. The historical data
below is retained, but its conclusion should not be used.

## Question

When the same melody and continuation checkpoint are used with greedy decoding, does
StreamMUSE produce the same raw accompaniment tokens as the correct offline inference
path at every generated beat?

Input-token equality is diagnostic only. The pass condition is exact equality of the
model's output-token sequence for each aligned target beat.

## Observation Points

```text
StreamMUSE public MIDI client
  -> HTTP backend
  -> model generates one accompaniment beat
  -> [raw output-token hook]
  -> decode to pianoroll/events
  -> scheduler/playback
  -> combined MIDI

Correct offline continuation
  -> native generation schedule
  -> _generate_one_beat()
  -> [raw output-token hook]
  -> decode to pianoroll
  -> output MIDI
```

The hooks observe existing return values and do not call the model a second time or
change the client/scheduler behavior.

## Fixed Conditions

- Test piece: song 4, 15 measures, BPM metadata 57.
- StreamMUSE input: `prompts/inputs_lekai/mel/4.mid` through the public MIDI client.
- Offline input: `4.npz`, reconstructed using the tracked melody and accompaniment MIDI.
- The reconstructed NPZ melody pianoroll is exactly equal to the public client's melody
  history: zero differing cells and zero differing beats.
- Checkpoint SHA-256:
  `d93139044a8614aeb66c58b5696371a575389199fbe64653b8994d3f6b056271`.
- The local checkpoint and remote epoch-15 checkpoint have the same SHA-256.
- Correct offline continuation implementation matches the remote
  `RT-accompanimentV2/model.py` SHA-256:
  `c92d9fd9bc4a2d0be5daff5298ed43996286a8a6ab808602e7969839f611b79a`.
- Device/dtype: CUDA, float16.
- Decoding: deterministic greedy. StreamMUSE uses `temperature=0`; offline uses
  `top_k=1`, which selects the same argmax policy.

## Current Branch Result

Branch/commit: `integrate-prompt-continuation-switch` at `3a264d56`.

| Item | Result |
|---|---:|
| Comparable target beats | 59 (beats 1-59) |
| Exact raw-token matches | 43 / 59 |
| Raw-token mismatches | 16 / 59 |
| Mismatch range | beats 36-51 |
| Token mismatches also changing decoded pianoroll | 16 / 16 |
| Dropped realtime requests | 0 |
| Final active-cell intersection / union | 37 / 69 (53.62%) |

The mismatch is structured rather than numerical noise:

```text
StreamMUSE beats 36-43 == offline beats 44-51
StreamMUSE beats 44-51 == offline beats 36-43
```

Two consecutive eight-beat blocks are exchanged. Beats 1-35 and 52-59 are exact raw
token matches.

Therefore the observed difference already exists in the model output before event
decoding and before the scheduler. Playback timing or late-event handling cannot cause
this particular mismatch.

All 59 aligned prompt digests differ, but prompt equality is not part of the pass
condition. It only indicates that the next localization target is context construction
and beat/measure mapping around beats 36-51.

## Stable Branch Control

Control commit: `origin/stable` at `acab43d9`.

At 120 BPM, the old backend took about 1.03 seconds per request while one beat lasted
0.5 seconds. The public client consequently requested only 38 target beats and skipped
intermediate beats. This run is not valid for an every-beat equivalence comparison.

It was repeated at 15 BPM, giving every beat a four-second window:

| Item | Result |
|---|---:|
| Requests / traced beats | 59 / 59 |
| Missing target beats | 0 |
| Mean backend latency | 1033.0 ms |
| Exact matches to correct offline | 0 / 59 |
| Generated accompaniment notes | 0 |

The stable backend uses the older continuation encoding/runtime and is not a valid
drop-in baseline for the current epoch-15 checkpoint. It does not establish offline
equivalence for this checkpoint.

## Secondary Finding

`scripts/run_lekai_offline.py` imports the legacy
`streammuse.infrastructure.inference.lekai_model` implementation. With the reconstructed
song-4 NPZ it generates no accompaniment beats and eventually fails while decoding an
empty result. It must not be used as the offline oracle for this audit.

The diagnostic runner instead imports
`streammuse.infrastructure.inference.lekai_continuation_model` and hooks its native
`_generate_one_beat()` method.

## Conclusion

The current StreamMUSE path and the correct offline path are **not fully output
equivalent**. For this controlled case, 16 beats differ before scheduler/playback, in a
clear two-block exchange. The next focused diagnosis is the context and beat/measure
mapping used to build model calls around beats 36-51; scheduler changes are not relevant
to this mismatch.

## Artifacts

- Current-branch comparison:
  `output/consistency/20260731-220045/round_model_output_audit/comparison.json`
- Decoded mismatch summary:
  `output/consistency/20260731-220045/round_model_output_audit/decoded_mismatch_summary.json`
- Final MIDI pianoroll comparison:
  `output/consistency/20260731-220045/round_model_output_audit/midi_pianoroll_comparison.json`
- Current StreamMUSE trace:
  `output/consistency/20260731-220045/streammuse_model_outputs.jsonl`
- Correct offline trace and MIDI:
  `output/consistency/20260731-220045/round_model_output_audit/`
- Stable controls:
  `output/consistency/20260731-220045/stable_branch_audit/`
