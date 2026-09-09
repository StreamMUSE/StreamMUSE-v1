# Closed quantized input windows

This fix continues `codex/fix-same-tick-melody-encoding`. It is separate from
the earlier same-tick event-order fix. No model weights, sampling parameters,
quantization formula, playback deadlines, or late-event playback policy change.

## Client contract

With snap-forward fraction `f`, quantized tick `k` covers arrival positions
`[k-f, k+1-f)`. A request advertising the exclusive end `E` can seal only
after `E-f`. The existing eight-beat Prompt observation period is retained:
the first request waits until `E`, not `E-f`.

An independent, interruptible input-window worker seals requests. It does not
extend `_tick_loop`'s 0.1-tick wait or move metronome/playback times. Timestamp
assignment and buffer insertion share a lock with sealing. Only events with
`tick < E` are released; future events remain buffered. Same-tick input order
and empty REST windows are preserved. Stop joins the new worker before closing
outputs. Finite runs do not submit a new input boundary at their exclusive
`max_ticks` endpoint.

The established input policies remain: devices use configured snap-forward;
MIDI files use floor quantization. Each path closes according to its effective
quantizer. They share the same exclusive tick partition, not necessarily the
same wall-clock release time. This is not a replay of original arrival times.

## Backend contract

Closing client windows alone is insufficient: previously Prompt completion
injected all melody received during inference, and continuation consumed all
new events visible when its background worker happened to run.

Injection now sees only the fixed Prompt interval `[0,P)`. For each generated
beat beginning at `G`, continuation receives new melody only through the first
closed append boundary that could trigger this beat under the existing
ceil-to-beat counter and append interval `I`:

```
E(G) = P + max(0, floor((G - 4 - P) / I) + 1) * I
```

For `P=32, I=2`, generations at `32,36,40,...,84` use exclusive melody
boundaries `32,34,38,...,82`. For `I=4` they use `32,36,40,...,84`.
This deliberately preserves existing generation trigger positions. It does
NOT change the client interval to one beat, or relabel half-beat progress as
a newly introduced full-beat policy. Later buffered melody cannot leak into
an earlier generation solely because its worker runs slowly.

Exact-window tests use the deployed `max_continuation_chunk_beats=1` setting.
For multi-beat chunks the cutoff is the final beat's boundary; changing chunk
grouping is outside the exact-replay claim. This change affects the shared
Prompt-continuation scheduler, including other callers of its start/append
API; direct offline inference and model decoding code are not edited.

## Acceptance

- Include A's off at arrival tick 81.244 and B's off at 81.022 in the same
  exclusive window ending at 82; retain a future event at 82 for the next window.
- Preserve same-tick on/off ordering, Prompt endpoint exclusion, and RESTs.
- Exercise an admission/sealing race using thread barriers.
- Verify Stop interrupts a long count-in/window wait and playback timing stays
  unchanged in the fake-clock test.
- Delay Prompt or continuation while submitting later windows; compare every
  injected/incremental Melody input, not only final output counts.
- Run real-model A virtual device, B exported-MIDI replay, C repeated replay
  with frozen seed/config on the same H200. Report model inputs/outputs and
  emitted playback independently. Passing a software case is not a universal
  guarantee for physical devices, different GPUs, MIDI quantization loss, or
  missed playback deadlines.

Do not repair old recordings, overwrite their mismatch reports, or claim this
revision reproduces a session recorded with the previous collection policy.
