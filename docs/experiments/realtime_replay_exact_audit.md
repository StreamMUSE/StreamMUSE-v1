# Realtime Replay Exact Audit

## Exact Claim

This audit tests one narrow claim: after replaying the quantized Melody from an
ORIGINAL human-live RuntimeSession through the realtime MIDI-file path, the
ordered protocol requests and Prompt+Continuation model inputs and outputs are
exactly identical to the ORIGINAL run.

The claim is valid only when both runs contain complete audit evidence. A
matching MIDI note count, similar audio, or matching digests from only part of
the pipeline is not sufficient.

## Required Reset

Before both ORIGINAL and REPLAY, explicitly call the Prompt+Continuation debug
reset endpoint with identical `prompt_seed` and `continuation_seed` values.
Advancing an existing server RNG is not reproducible, even if its initial seed
is known.

Both traces must report:

- `runtime_info.trace_capture_complete = true`
- complete seed provenance
- an active seeded server session

## Session Artifacts

- `prompt_continuation_replay_requests.jsonl`: ordered start/append requests
- `prompt_continuation_model_trace.json`: Prompt tokens, selection evidence,
  continuation generation digests, runtime seeds, and capture completeness
- `replay_audit_manifest.json`: capture status and artifact inventory
- `prompt_continuation_replay_melody.json`: captured quantized Melody events
- `prompt_continuation_replay_melody.mid`: MIDI-file replay input

The comparator requires the request and model trace files. The manifest is
optional input to the comparator and does not replace missing model evidence.

## Comparison

```bash
python scripts/compare_realtime_replay_exact.py \
  ORIGINAL_SESSION_DIR REPLAY_SESSION_DIR \
  --output replay_exact_comparison.json
```

Exit codes:

- `0`: evidence is complete and `model_exact=true`
- `1`: evidence is complete, but an exact comparison mismatched
- `2`: evidence is missing, incomplete, or invalid

The report separates protocol, Prompt input/output, and continuation
input/output exactness and identifies the first mismatch. Request
acknowledgement contents, timing measurements, request IDs, and session
IDs/epochs are not compared.

## Scope Boundary

`model_exact` certifies the captured protocol semantics and model computation
only. It does not certify scheduler delivery, realtime playback, active-note
handling, or `combined.mid` export exactness. Those require a separate
playback/export comparison.

Sessions created before these audit artifacts and completeness fields existed
cannot be retroactively certified as exact. Regenerate both ORIGINAL and
REPLAY under the audited session contract.
