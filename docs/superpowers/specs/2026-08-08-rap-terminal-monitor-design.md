# Dense Rap Terminal Monitor and Flow-Aware Generation

## Purpose

Make the live rap prototype readable as both a performance monitor and a research instrument without removing any existing diagnostics. The interactive terminal will use a dense two-column layout. The same event stream must remain readable when redirected to a file, captured by an experiment runner, or displayed in a narrow terminal.

This change also closes a generation-design gap: the LLM currently receives only a flow-template ID, syllable count, and meter. It must instead receive the actual immutable flow template that the deterministic alignment and ranking stages use.

## Success Criteria

- The user can see the current lyric, its bar position, the template slots, target stresses, emitted syllables, timing jitter, queued lyric, and armed fallback at a glance.
- The user can inspect the exact system and user messages provided to the LLM.
- The request display distinguishes model input from post-generation analysis.
- The LLM receives the template's slot ticks, target stresses, phrase boundary, and rhyme-role metadata in a compact human-readable form.
- Candidate validity, syllable-count rejection, OOV words, weighted component scores, selection, latency, errors, deadlines, and fallback decisions remain visible.
- Rendering remains off the musical tick path and does not alter planning or freeze behavior.
- Interactive terminals receive the split view; non-interactive and narrow terminals receive a structured append-only representation of the same information.

## Information Architecture

### Header

The header shows stable session identity and current clock state:

- session state
- scenario and topic
- model and generator
- tempo and meter
- current bar and absolute tick

### Left Column: Performance

The left column answers "what is happening now?":

1. **Live delivery**: committed lyric, source, score, bar, and topic.
2. **Flow strip**: 16 tick positions grouped into four beats, target-stress strength, scheduled syllables, and current tick.
3. **Queue**: next generated line and its score.
4. **Safety**: prevalidated fallback currently armed for the next bar.
5. **Clock and health**: tick period, current/aggregate jitter, pending request count, model state, and fallback count.
6. **Recent commits**: compact source and lyric history for frozen bars.

### Right Column: Research Trace

The right column answers "why did the system choose this?":

1. **Structured request**: request ID, target bar, topic, seed, model parameters, context lines, and complete flow snapshot.
2. **Exact prompt**: the exact system and user message content sent to the model.
3. **Model response**: candidate count, raw response in full mode, token counts, latency, late flag, warning, and error.
4. **Candidate gate and ranking**: syllable count, validity, rejection reasons, OOV words, score, lyric, and selection state.
5. **Selected score**: every weighted component with raw value, weight, and contribution.
6. **Event trace**: recent ordered lifecycle events and timing.

## Flow As Model Context

### Single Source of Truth

`CandidateRequest` will carry the validated immutable `FlowTemplate`, rather than carrying only a template ID and independently supplied syllable count. Compatibility properties may expose `template_id` and `required_syllables`, but both derive from the template.

This prevents the prompt, terminal display, aligner, and scorer from describing different rhythms.

### Prompt Representation

The local-chat generator will serialize the template into both exact arrays and compact beat notation. For `baseline_syncopated_9`, the request includes:

```text
Flow template: baseline_syncopated_9
Syllable ticks: [0, 2, 3, 5, 7, 8, 10, 13, 15]
Target stress: [1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9]
Pattern: S . w M | . w . M | S . w . | . M . S
Final slot: phrase boundary strength 3, rhyme group A
```

`S`, `M`, and `w` are prompt-friendly summaries of continuous target-stress values. Exact numeric values remain authoritative and are always included. The model is told to place naturally stressed syllables near stronger slots, close the phrase at the final slot, and return plain lyric lines without syllable markup.

The LLM is still a candidate generator, not the authority on alignment. CMU prosody analysis, exact syllable gating, deterministic alignment, thresholding, and fallback remain unchanged and occur after generation.

## Components

### Flow Prompt Formatter

A pure infrastructure formatter converts `FlowTemplate` into:

- compact beat notation
- exact tick array
- exact stress array
- phrase-boundary description
- rhyme-role description

The formatter has no terminal dependency and is unit tested independently.

### Structured Monitoring Payloads

`BAR_PLANNING_STARTED` will include a JSON-ready snapshot of:

- request ID and target bar
- topic and context lines
- seed and candidate count
- flow-template identity, provenance, meter, slots, stresses, boundaries, and rhyme groups

`CANDIDATE_BATCH_RECEIVED` remains the source for the exact prompt and raw response. The UI consumes structured event fields and never parses human-readable prompt text to reconstruct state.

Session-start metadata will include scenario, generator, model, terminal mode, tempo, meter, and finite/unbounded bar target.

### Terminal State Projector

A pure `TerminalRapViewState` projector consumes ordered `RapEvent` values and retains only the bounded state needed for rendering:

- current and next bars
- active flow snapshot
- latest request and batch
- latest candidate evaluations and selected score
- bounded recent event trace
- health and timing aggregates

This separates event interpretation from Rich rendering and permits deterministic tests without a terminal.

### Interactive Split Renderer

The interactive renderer uses Rich `Layout`, `Panel`, `Table`, and `Live` primitives. It refreshes only when an event changes state. The renderer is called by the existing event-dispatch thread, so no terminal work is added to `RollingRapController.on_tick()`.

The target layout is optimized for terminals at least 120 columns wide. Below that width it stacks performance above research while preserving section order.

### Append-Only Renderer

The current append-only behavior becomes an explicitly structured stream renderer. It uses the same phase vocabulary and state semantics as the split view:

```text
[BAR 02][PLAN]   ...
[BAR 02][MODEL]  ...
[BAR 02][GATE]   ...
[BAR 02][SELECT] ...
[BAR 02][PLAY]   ...
```

This renderer is selected automatically when stdout is not a TTY. It remains available explicitly for reproducible captured logs.

## CLI and Compatibility

Add:

```text
--terminal-layout auto|split|stream
```

- `auto` is the default: split for a sufficiently wide TTY, stream otherwise.
- `split` requests the Rich dashboard and stacks sections if width is constrained.
- `stream` forces append-only output.

Existing `--terminal-detail summary|candidates|full` remains an orthogonal density control:

- `summary`: performance, selected line, request summary, health, and fallback state.
- `candidates`: summary plus candidate gate/ranking rows.
- `full`: candidates plus exact prompts, raw response, score components, provenance, and event trace.

The existing injected `write` callback remains supported by the stream renderer for unit tests and embedding. Rich is added as a direct project dependency. Rich honors `NO_COLOR`; stream output remains free of ANSI codes when redirected.

## Failure Behavior

- Generation failures visibly retain and freeze the armed fallback.
- Late responses are shown but cannot replace a frozen bar.
- Invalid candidates remain visible with stable rejection reasons.
- A Rich rendering failure switches once to the stream renderer and prints one concise presentation warning; it does not terminate music planning.
- A malformed or missing flow template is rejected when constructing the domain request, before model work starts.
- Model secrets and authorization values remain subject to existing sanitization before display.

## Testing

### Domain and Prompt Tests

- `CandidateRequest` derives template ID and required syllables from one `FlowTemplate`.
- Prompt output contains the exact slot ticks, stress values, boundary, rhyme role, and beat notation.
- Two nine-slot templates with different timing produce different prompts.
- Prompt remains immutable in `CandidateBatch` diagnostics.

### Monitoring Tests

- Planning events contain a complete JSON-ready flow snapshot and request context.
- The projector reconstructs current/next bars, fallbacks, candidate selection, prompts, and timing from ordered events.
- State history remains bounded during long sessions.

### Rendering Tests

- Fixed-width recorded Rich output contains both columns and all required sections.
- Narrow-width output stacks sections without truncating lyrics or prompt text.
- Candidate rows distinguish selected, valid, rejected, and OOV states.
- The flow strip shows beat boundaries, all 16 ticks, nine slots, and the current tick.
- Non-TTY `auto` output selects the stream renderer and contains no ANSI escape codes.
- `summary`, `candidates`, and `full` preserve their documented content boundaries.
- A simulated Rich failure switches to stream output without propagating into the dispatcher.

### Integration Verification

- Run the affected rap and CLI suites.
- Run phrase-bank and scripted-failure sessions to verify fallback and rejection presentation.
- Run a finite Qwen session on H200 and confirm that the displayed flow matches the exact prompt and the template used by alignment/ranking.
- Record generation latency, valid-candidate rate, selected score, tick jitter, and fallback activation. Prompt-aware flow quality is observed but not claimed from a single demonstration.

## Scope Boundaries

Included:

- dense terminal information design
- actual flow template in LLM context
- structured request observability
- interactive and captured-log rendering
- tests and H200 verification

Deferred:

- speech synthesis or audible pronunciation
- live keyboard/drum input
- web UI
- recorder, CSV, and aggregate experiment reports
- changes to score weights or ranking formulas
- automatic MCFlow corpus expansion
- claims that prompting alone solves rap-flow quality
