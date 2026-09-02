# Matched System Trace v2

`system_trace.jsonl` schema version 2 separates model-decision availability
from audible note emission. Every record has `schema_version: 2` and one of
the following `record_type` values.

## Frame deadlines

The realtime service writes one `frame_deadline` record for every playback
tick. Required timing and diagnostic fields include:

```json
{
  "schema_version": 2,
  "record_type": "frame_deadline",
  "condition": "standard",
  "mode": "realtime",
  "tick": 12,
  "nominal_tick_time_s": 101.5,
  "deadline_time_s": 101.5125,
  "observed_emit_time_s": 101.513,
  "emitted_model_note_on_count": 0,
  "decision": "missing"
}
```

The legacy `decision`, `arrival_time_s`, `arrived_by_deadline`,
`explicit_rest`, and event-provenance fields remain diagnostic only. They
describe scheduled or emitted events and must not be used to decide whether a
model frame was available.

## Availability spans

An `availability_span` records the half-open frame range `[start_tick,
end_tick_exclusive)` covered by a model response and the time at which that
response became usable by the client:

```json
{
  "schema_version": 2,
  "record_type": "availability_span",
  "condition": "prompt_continuation",
  "mode": "realtime",
  "clock_domain": "service_now",
  "start_tick": 32,
  "end_tick_exclusive": 40,
  "availability_time_s": 104.2,
  "generation_start_tick": 32,
  "request_id": "playable-0001",
  "source_stage": "continuation"
}
```

All spans satisfy `start_tick < end_tick_exclusive`. A span may cover ticks
whose playback deadlines have already passed; this is how the trace represents
a valid but late decision. `availability_time_s` and frame deadlines share the
`service_now` clock domain.

For the standard realtime service, every successful inference response creates
a span from its generation start tick over the requested generation length.
This is independent of the number of accompaniment events returned, so a
successful empty-accompaniment response still covers its frames.

For Prompt+Continuation, a completed playable HTTP response reports cumulative
coverage as `accompaniment_history_beats * 4`. Only coverage beyond the previous
playable response is recorded. The first response starts at tick 0. A newly
available range crossing `prompt_length_ticks` is split into `prompt` and
`continuation` spans. `availability_time_s` is sampled immediately after the
playable HTTP call returns. If the server does not expose a request ID, the
client assigns a session-local `playable-NNNN` correlation ID. A new service
instance resets both the cumulative coverage and this counter.

## ISR join

For each `frame_deadline` row, match `availability_span` rows with the same
condition and:

```text
start_tick <= frame_deadline.tick < end_tick_exclusive
```

The frame availability time is the earliest `availability_time_s` among its
matching spans. The frame is on time exactly when that value is less than or
equal to `deadline_time_s`. A covered frame containing REST or no `note_on` is
still a valid model decision. Therefore:

```text
no_note_on != missing decision
```

An `availability_span` does not classify a frame as on time by itself. Formal
evaluation must perform the span-to-deadline join above, including when the
availability timestamp is later than the matched deadline.

Schema version 1 inferred availability from emitted note/rest diagnostics and
cannot support the formal ISR calculation. It must not be mixed with version 2
records in matched evaluation.
