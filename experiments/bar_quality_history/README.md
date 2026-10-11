# Single-bar history experiment (CPU only)

This page describes the original `harness.py` CPU transaction tool.
The separately gated model snapshot probe is documented in [MODEL_PROBE.md](MODEL_PROBE.md).
Its model-facing code is prepared but not GPU-validated; it is not wired into Web playback.

This standard-library scaffold records one explicit completed-bar transaction.
It combines an externally supplied accept/reject decision with optional offline
history replacement. It does not determine musical quality. No thresholds are
approved or implemented; the descriptive statistics are not calibrated scores.
This is an offline `HISTORY_ONLY` transaction prototype, not a realtime
simulation or a playback quality gate. The user's playback-drop versus
history-drop semantics remain unsettled; this prototype does not resolve them.
No torch, GPU, model loading, scheduler, MIDI writer, cache, or remote commands
are involved. Models are not integrated. There are no performance, inference,
quality-improvement, or latency claims.

## Run locally

From PowerShell, these exact commands run independently of repository pytest
and its conftest. Python 3.9+ is sufficient; no dependencies need installation.

```powershell
Set-Location C:/Users/User/.codex/worktrees/bar-quality-history-experiment/StreamMUSE-v1
python -B -m unittest discover -s experiments/bar_quality_history/tests -p "test_*.py" -v
python -B experiments/bar_quality_history/harness.py experiments/bar_quality_history/example.synthetic.json
```

`example.synthetic.json` is a hand-authored synthetic dry run, NOT measured
data, model output, or evidence of musical quality. Its reject decision is an
explicit demonstration input, not the result of a score. The CLI writes the
provenance/result JSON to stdout only; invalid input exits 2 without a result.
It does not modify the input or create MIDI. The module API is
`run_transaction(document)`, raising `InputError` on invalid transactions.

## Input schema

JSON objects reject unknown fields, missing required fields, duplicate keys,
and non-finite numbers. All time and pitch fields must be JSON integers:
booleans, strings, floats (including `16.0`), and fractional ticks are invalid.
Times are nonnegative absolute system steps, never bar-relative offsets.

Required top-level fields:

| Field | Meaning |
| --- | --- |
| `steps_per_beat` | Exactly integer `4`. |
| `bar_start_step` | Explicit nonnegative bar start; no inferred alignment or meter. |
| `bar_length_steps` | Explicit positive integer bar length; no default threshold or meter. |
| `observed_until` | Observation boundary, exactly `bar_start_step + bar_length_steps`. |
| `melody_notes` | Observed reference notes, not future conditioning. |
| `history_notes` | Original experiment accompaniment context, not necessarily the audible record. |
| `played_notes` | Separate immutable record of already played notes. |
| `policy` | `baseline`, `drop`, or `repair_history`. |
| `decisions` | List of exactly one external decision object described below. |

Every note has exactly `{"onset":16,"end":20,"pitch":60}` with
`0 <= onset < end <= observed_until` and MIDI pitch `0..127`. Note intervals
are half-open `[onset,end)`. Polyphony and overlapping notes are allowed.
Input ordering is preserved, not silently sorted or deduplicated. All supplied
note arrays are validated, even candidates unused by baseline or acceptance.
Future onsets and unfinished notes extending beyond the cutoff are rejected.
This narrow schema cannot represent active sustained notes at the cutoff;
it deliberately fails rather than guessing their eventual end.

The sole decision object requires `bar_start_step`, `boundary_step`, `decision`
(`accept` or `reject`), and a nonblank `source` string for provenance. Its start
must match the transaction start and its boundary must equal the cutoff.
Multiple decisions, including duplicate or overlapping boundaries, are rejected;
this is not a full timeline simulator.

Optional `repair_candidate` is null/absent (missing) or an object with exactly
`source` (nonblank provenance string for the supplied candidate and its external
evaluation), `decision` (explicit external `accept` or `reject`), and `notes`
(same note schema). Candidate
notes must be wholly inside the selected bar. An explicit empty note list is
a valid supplied silent candidate, distinct from missing. Candidates must be
supplied in JSON; the harness never fabricates or generates them. Source strings
are caller assertions, not independently verified lineage or causal availability.

`repair_fallback` is required for `repair_history`, even if a candidate
is present: `keep`, `drop`, or `error`. It applies when a candidate is missing
OR externally rejected. If provided for another policy it must still be valid.
No implicit fallback is chosen. Presence alone never approves a candidate.

## Transaction semantics

`baseline` preserves the entire history regardless of the recorded decision.
All policies preserve history on acceptance. On rejection, `drop` with explicit
output scope `HISTORY_ONLY` removes notes
overlapping only this completed bar from experiment history. It does NOT mute,
erase, or alter previously played MIDI. `repair_history` replaces those notes
with an externally accepted explicit candidate, or applies the explicit
missing/rejected-candidate fallback. Even a rejected candidate must satisfy the
schema and cutoff constraints; fallback does not bypass validation.
Replacement appends the supplied notes after the retained history, preserving
relative input order. All notes outside the selected bar remain unchanged.

Before drop or replacement, any original history note overlapping the bar but
crossing a boundary causes an error. Candidates crossing a boundary also fail.
No sustain is truncated or split. A note ending exactly at bar start is outside
the bar; one starting at bar start or ending exactly at bar end is inside.
A note starting at bar end is future data and rejected. Baseline/keep may retain
a history note crossing the start because they do not mutate it.

The result records the full input, original and resulting history, decision
source, action, reason, scope `HISTORY_ONLY`, and `conceptual_boundary_step`.
This boundary is not a measured repair-ready time. Future repair timing is not
simulated (`repair_timing_simulated` is false); candidate availability, compute
latency, deadlines, and future generation are not tested. `future_context_notes` is merely
an independent copy of resulting history for a possible future consumer. It is
not a model state or KV cache. `playback_before` and `playback_after` are equal
independent copies of `played_notes`, which is never inferred from replacement
history. Generated historical content, if a caller later supplies it, would be
counterfactual context, not audible history. This harness generates no content.

## Descriptive statistics

For each original-history, resulting-history, and melody bar: onsets count
`start <= onset < end`; coverage is the union of note intersections with the
bar, in steps and as a fraction of bar length (never double-counted polyphony).
Duration median uses the positive lengths of those intersections, one per
overlapping note. Pitch min/max/range use the same overlapping notes.
Clipping here is only for measurement, never for editing stored notes.
The original-history/melody median-duration ratio is reported only when both
have overlapping notes; otherwise it is null. Empty pitch/duration statistics
are null, empty coverage/onset counts zero. None imply musical quality or
automatically select a decision.

## Integration boundary

Parent source inspection is recorded in
`docs/experiments/bar-quality-history-preparation-20261006.md`: base `a657fe81` on
`codex/fix-web-session-display-reset` is a known local web-fix descendant,
NOT claimed to be the latest remote revision. `continuation_engine.py` forwards
`inject_history` to the backend. Backend `_generate_part1_tokens_from_prompt` starts each
generation with `past_key_values=None` (approximately line 870 at that base).
The `inject_history` path via `_inject_history_locked` (approximately line 1747)
resets melody/accompaniment and token
histories, `tonal_tracker`, `empty_token_guard`, and active pitches.

Do NOT hook `inject_history` live from this experiment. These findings are
integration cautions from the parent's source inspection, not a runtime audit
or integration implemented here. Prefill is not generation; resetting history is not proof of
cached incremental inference or a measured speedup. All production scheduler,
training, offline, and model paths remain untouched.
