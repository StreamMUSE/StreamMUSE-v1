# Realtime Replay Exact Audit

## Exact Claim

This audit tests one narrow claim: after replaying the quantized Melody from an
ORIGINAL human-live RuntimeSession through the realtime MIDI-file path, the
ordered protocol requests and Prompt+Continuation model inputs and outputs are
exactly identical to the ORIGINAL run.

The claim is valid only when both runs contain complete audit evidence. A
matching MIDI note count, similar audio, or matching digests from only part of
the pipeline is not sufficient.

## Session Seeds

An ordinary Prompt+Continuation Web Start requires no seed input. Before the
clock, MIDI input, or model protocol starts, the RuntimeSession calls the normal
`/prompt_continuation/session/initialize` lifecycle endpoint. The server creates
fresh system-random Prompt and Continuation seeds and returns their requested
and effective values together with the server session ID and epoch.

The ORIGINAL session stores those values in
`prompt_continuation_session_seed.json`. A replay harness must read the two
effective seeds and pass them as `prompt_seed` and `continuation_seed` to the
same lifecycle endpoint before starting the realtime MIDI-file RuntimeSession.
This endpoint is not debug-gated and does not require
`LEKAI_ENABLE_DEBUG_RESET`.

`scripts/run_matched_system_eval.py` remains externally seeded: its existing
trial reset runs first, then the runner threads that reset acknowledgement into
the CLI environment. The CLI adopts and records the already-active server
session after a read-only provenance check; it does not initialize or reset the
server a second time.

Both traces must report:

- `runtime_info.trace_capture_complete = true`
- complete seed provenance
- an active seeded server session

## Session Artifacts

- `prompt_continuation_replay_requests.jsonl`: ordered start/append requests
- `prompt_continuation_model_trace.json`: Prompt tokens, selection evidence,
  continuation generation digests, runtime seeds, and capture completeness
- `prompt_continuation_session_seed.json`: requested/effective Prompt and
  Continuation seeds plus local and server session provenance
- `replay_audit_manifest.json`: capture status and artifact inventory
- `prompt_continuation_replay_melody.json`: captured quantized Melody events
- `prompt_continuation_replay_melody.mid`: MIDI-file replay input

The comparator requires the request, model trace, and session seed files. It
also verifies that the saved effective seeds and server session ID/epoch match
the model trace. The manifest does not replace missing model evidence.

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
