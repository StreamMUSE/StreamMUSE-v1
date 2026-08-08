# Dense Rap Terminal Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Supply the real flow template to lyric generation and present the complete realtime rap state in a dense split terminal with a structured stream fallback.

**Architecture:** `CandidateRequest` owns one immutable `FlowTemplate`, from which prompt constraints, monitoring payloads, alignment, and ranking derive. Ordered `RapEvent` values feed a bounded presentation projector; a Rich dashboard and an append-only stream renderer consume the same projected state without entering the musical tick path.

**Tech Stack:** Python 3.10, immutable dataclasses, Rich 14+, argparse, pytest, existing StreamMUSE rap events and one-worker realtime planner.

## Global Constraints

- Preserve the current clean architecture boundaries and existing `streammuse-rap` and `streammuse-rap-demo` entry points.
- Keep LLM generation, prosody analysis, scoring, and terminal rendering off `RollingRapController.on_tick()`.
- Keep prevalidated fallback, freeze-before-emission, late-result rejection, and bounded lookahead behavior unchanged.
- Use one validated immutable `FlowTemplate` as the source for prompting, monitoring, alignment, and scoring.
- Show exact model prompts only from `CandidateBatch.prompt`; never reconstruct exact prompts by parsing display text.
- Sanitize model errors using the existing generator sanitization before presentation.
- Default `--terminal-layout auto`: split only for a sufficiently wide TTY, stream otherwise.
- Keep `--terminal-detail summary|candidates|full` independent of layout.
- Interactive target width is 120 columns or wider; explicit split mode stacks below 120 columns.
- Redirected stream output contains no ANSI escapes and remains useful as a captured experiment log.
- Do not add speech synthesis, live input, web UI, recorders, aggregate metrics, score-formula changes, or MCFlow corpus expansion.

---

## File Structure

### Domain and Generation

- `src/streammuse/domain/rap/generation.py`: own `CandidateRequest`/`CandidateBatch`, make `flow_template` authoritative, and expose compatibility properties without creating a `models.py`/`flow.py` import cycle.
- `src/streammuse/domain/rap/models.py`: retain syllable, alignment, and plan values after generation request/batch values move out.
- `src/streammuse/domain/rap/__init__.py`: preserve public request/batch imports from `streammuse.domain.rap`.
- `src/streammuse/application/rap/rhythm.py`: adapt legacy 16-slot named presets into validated flow templates.
- `src/streammuse/infrastructure/rap/flow_prompt.py`: pure flow notation and prompt serialization.
- `src/streammuse/infrastructure/rap/generators.py`: include exact flow structure in local-chat messages.
- `src/streammuse/application/rap/service.py`: construct requests with the actual template.
- `src/streammuse/application/rap/realtime.py`: construct realtime requests with the actual template and emit structured request/alignment payloads.

### Monitoring and Presentation

- `src/streammuse/application/rap/monitoring_payloads.py`: JSON-ready flow and scheduled-syllable snapshots.
- `src/streammuse/application/rap/runtime.py`: publish resolved session metadata at session start.
- `src/streammuse/presentation/rap_demo/terminal_state.py`: bounded event-to-view-state projector.
- `src/streammuse/presentation/rap_demo/terminal_stream.py`: structured append-only renderer.
- `src/streammuse/presentation/rap_demo/terminal_dashboard.py`: pure Rich layout builder and Live adapter.
- `src/streammuse/presentation/rap_demo/terminal.py`: backward-compatible sink facade, layout selection, and renderer fallback.
- `src/streammuse/presentation/rap_demo/cli.py`: `--terminal-layout` and resolved metadata assembly.
- `pyproject.toml`: direct Rich dependency.

---

### Task 1: Flow-Aware Candidate Request and Prompt

**Files:**
- Create: `src/streammuse/infrastructure/rap/flow_prompt.py`
- Create: `src/streammuse/domain/rap/generation.py`
- Modify: `src/streammuse/domain/rap/models.py:88-161`
- Modify: `src/streammuse/domain/rap/__init__.py`
- Modify: `src/streammuse/application/rap/rhythm.py:67-93`
- Modify: `src/streammuse/application/rap/service.py:30-60`
- Modify: `src/streammuse/application/rap/realtime.py:437-451`
- Modify: `src/streammuse/infrastructure/rap/generators.py:115-132`
- Modify: `tests/unit/infrastructure/rap/test_generators.py`
- Modify: `tests/unit/infrastructure/rap/test_fallback.py`
- Modify: `tests/unit/application/rap/test_service.py`
- Modify: `tests/unit/application/rap/test_rhythm.py`
- Modify: `tests/unit/application/rap/test_realtime.py`
- Test: `tests/unit/infrastructure/rap/test_flow_prompt.py`

**Interfaces:**
- Consumes: validated `FlowTemplate` and `FlowSlot` domain values.
- Produces: `CandidateRequest.flow_template: FlowTemplate`, compatibility properties `template_id: str` and `required_syllables: int`, `flow_template_for_pattern(tempo, pattern)`, `FlowPromptDescription`, `describe_flow(template)`, and `format_flow_for_prompt(template)`.

- [ ] **Step 1: Write failing domain tests for one flow source of truth**

Add tests that construct a `CandidateRequest` with `flow_template=BUILTIN_TEMPLATES.get("baseline_syncopated_9")` and no independent template ID/count:

```python
def test_candidate_request_derives_template_identity_and_syllable_count() -> None:
    template = BUILTIN_TEMPLATES.get("baseline_syncopated_9")
    request = CandidateRequest(
        request_id="request-1",
        target_bar=1,
        topic="space",
        flow_template=template,
        count=4,
        context_lines=("stars cross night",),
        seed=7,
    )

    assert request.template_id == "baseline_syncopated_9"
    assert request.required_syllables == 9
    assert request.flow_template is template
```

- [ ] **Step 2: Run the domain/request tests and verify RED**

Run:

```bash
uv run pytest tests/unit/infrastructure/rap/test_generators.py tests/unit/infrastructure/rap/test_fallback.py tests/unit/application/rap/test_service.py tests/unit/application/rap/test_realtime.py -q --tb=short
```

Expected: failures because `CandidateRequest` does not accept `flow_template` and existing call sites still supply `template_id` and `required_syllables`.

- [ ] **Step 3: Make the template authoritative**

Move `CandidateRequest` and `CandidateBatch` from `models.py` into `generation.py` so the request can import `FlowTemplate` without cycling through `flow.py -> models.py`. Preserve public imports through `domain/rap/__init__.py`. Change the request fields to:

```python
@dataclass(frozen=True)
class CandidateRequest:
    request_id: str
    target_bar: int
    topic: str
    flow_template: FlowTemplate
    count: int
    context_lines: tuple[str, ...]
    seed: int

    @property
    def template_id(self) -> str:
        return self.flow_template.template_id

    @property
    def required_syllables(self) -> int:
        return len(self.flow_template.slots)
```

Validate `flow_template` with `isinstance(value, FlowTemplate)` and update every constructor in production and tests. Require `FlowTemplate.slots` to be a tuple containing only `FlowSlot` values and require `FlowTemplate.provenance` to be a `FlowProvenance`, so the validated template remains immutable and fails deterministically on malformed runtime inputs. In `RollingRapController`, resolve the catalog template once and pass that same object into the request and later alignment/ranking calls.

Preserve the older `RapPrototypeService` by adapting each named 16-slot preset into a validated flow template:

```python
def flow_template_for_pattern(tempo: Tempo, pattern: str) -> FlowTemplate:
    slots = build_bar_slots(tempo, pattern, bar=0)
    final = len(slots) - 1
    return FlowTemplate(
        template_id=f"legacy_{pattern}",
        name=f"Legacy {pattern} rhythm preset",
        ticks_per_beat=tempo.ticks_per_beat,
        beats_per_bar=tempo.beats_per_bar,
        slots=tuple(
            FlowSlot(
                tick_in_bar=slot.tick,
                duration_ticks=1,
                target_stress=slot.accent,
                boundary_strength=3 if index == final else 0,
                rhyme_group="A" if index == final else None,
            )
            for index, slot in enumerate(slots)
        ),
        provenance=FlowProvenance(
            kind="legacy_rhythm_preset",
            source=pattern,
        ),
    )
```

`RapPrototypeService` passes this template into `CandidateRequest` while retaining its existing `pattern` field and `build_bar_slots()` alignment behavior. Parameterize adapter coverage across `boom_bap`, `straight_8`, and `trap_sparse`, and add a service regression asserting `boom_bap` yields 16 prompt slots and keeps `plan.pattern == "boom_bap"`.

- [ ] **Step 4: Run request and existing rap tests and verify GREEN**

Run:

```bash
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap -q --tb=short
```

Expected: all selected tests pass with no independent request template ID/count remaining.

- [ ] **Step 5: Write failing flow-format tests**

Create `tests/unit/infrastructure/rap/test_flow_prompt.py`:

```python
def test_syncopated_template_formats_exact_ticks_stress_boundary_and_rhyme() -> None:
    template = BUILTIN_TEMPLATES.get("baseline_syncopated_9")

    description = describe_flow(template)
    rendered = format_flow_for_prompt(template)

    assert description.ticks == (0, 2, 3, 5, 7, 8, 10, 13, 15)
    assert description.stresses == (1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9)
    assert description.boundary_strengths == (0, 0, 0, 0, 0, 0, 0, 0, 3)
    assert description.rhyme_groups == (None, None, None, None, None, None, None, None, "A")
    assert description.notation == "S . w M | . w . M | S . w . | . M . S"
    assert "Boundary strengths: [0, 0, 0, 0, 0, 0, 0, 0, 3]" in rendered
    assert 'Rhyme groups: [null, null, null, null, null, null, null, null, "A"]' in rendered
    assert "phrase boundary strength 3" in rendered
    assert "rhyme group A" in rendered


def test_equal_length_templates_with_different_timing_have_different_prompts() -> None:
    straight = BUILTIN_TEMPLATES.get("baseline_straight_9")
    syncopated = BUILTIN_TEMPLATES.get("baseline_syncopated_9")

    assert format_flow_for_prompt(straight) != format_flow_for_prompt(syncopated)
```

- [ ] **Step 6: Run flow-format tests and verify RED**

Run:

```bash
uv run pytest tests/unit/infrastructure/rap/test_flow_prompt.py -q --tb=short
```

Expected: import failure because `flow_prompt.py` does not exist.

- [ ] **Step 7: Implement the pure flow formatter**

Create:

```python
import json


@dataclass(frozen=True)
class FlowPromptDescription:
    ticks: tuple[int, ...]
    stresses: tuple[float, ...]
    boundary_strengths: tuple[int, ...]
    rhyme_groups: tuple[str | None, ...]
    notation: str
    final_boundary_strength: int
    final_rhyme_group: str | None


def describe_flow(template: FlowTemplate) -> FlowPromptDescription:
    ticks = tuple(slot.tick_in_bar for slot in template.slots)
    stresses = tuple(slot.target_stress for slot in template.slots)
    boundary_strengths = tuple(slot.boundary_strength for slot in template.slots)
    rhyme_groups = tuple(slot.rhyme_group for slot in template.slots)
    grid = ["."] * (template.ticks_per_beat * template.beats_per_bar)
    for slot in template.slots:
        grid[slot.tick_in_bar] = "S" if slot.target_stress >= 0.85 else "M" if slot.target_stress >= 0.5 else "w"
    beats = (
        " ".join(grid[start : start + template.ticks_per_beat])
        for start in range(0, len(grid), template.ticks_per_beat)
    )
    final = template.slots[-1]
    return FlowPromptDescription(
        ticks=ticks,
        stresses=stresses,
        boundary_strengths=boundary_strengths,
        rhyme_groups=rhyme_groups,
        notation=" | ".join(beats),
        final_boundary_strength=final.boundary_strength,
        final_rhyme_group=final.rhyme_group,
    )


def format_flow_for_prompt(template: FlowTemplate) -> str:
    flow = describe_flow(template)
    ticks = ", ".join(str(value) for value in flow.ticks)
    stresses = ", ".join(repr(float(value)) for value in flow.stresses)
    boundary_strengths = ", ".join(str(value) for value in flow.boundary_strengths)
    rhyme_groups = json.dumps(flow.rhyme_groups)
    rhyme = flow.final_rhyme_group or "none"
    return (
        f"Flow template: {template.template_id}\n"
        f"Syllable ticks: [{ticks}]\n"
        f"Target stress: [{stresses}]\n"
        f"Boundary strengths: [{boundary_strengths}]\n"
        f"Rhyme groups: {rhyme_groups}\n"
        f"Pattern: {flow.notation}\n"
        f"Final slot: phrase boundary strength {flow.final_boundary_strength}, rhyme group {rhyme}"
    )
```

Build a 16-position notation grid. Use `S` for stress `>= 0.85`, `M` for `>= 0.5`, `w` for lower occupied slots, and `.` for rests. Insert ` | ` between beats. Serialize every slot's boundary strength and rhyme group in aligned arrays. Format stress values deterministically with `repr(float(value))` so values such as `0.15`, `0.35`, and `0.75` round-trip without precision loss.

- [ ] **Step 8: Write failing generator tests for actual flow context**

Extend `test_generators.py`:

```python
def test_local_chat_prompt_contains_actual_flow_not_only_template_id() -> None:
    request = _request(flow_template=BUILTIN_TEMPLATES.get("baseline_syncopated_9"))
    client = RecordingChatClient("one line")

    LocalChatCandidateGenerator(client).generate(request)

    user = client.messages[1]["content"]
    assert "Syllable ticks: [0, 2, 3, 5, 7, 8, 10, 13, 15]" in user
    assert "Target stress: [1.0, 0.2, 0.7, 0.2, 0.6, 1.0, 0.2, 0.7, 0.9]" in user
    assert "Boundary strengths: [0, 0, 0, 0, 0, 0, 0, 0, 3]" in user
    assert 'Rhyme groups: [null, null, null, null, null, null, null, null, "A"]' in user
    assert "S . w M | . w . M | S . w . | . M . S" in user
    assert "plain lyric lines without syllable markup" in user
```

- [ ] **Step 9: Run the generator test and verify RED**

Run:

```bash
uv run pytest tests/unit/infrastructure/rap/test_generators.py::test_local_chat_prompt_contains_actual_flow_not_only_template_id -q
```

Expected: assertion failure because the current prompt contains only the ID/count/meter.

- [ ] **Step 10: Add the flow block to local-chat messages**

Call `format_flow_for_prompt(request.flow_template)` inside `_build_messages()`. Preserve recent frozen lines, topic, count, seed, no-numbering instruction, and immutable prompt diagnostics.

- [ ] **Step 11: Run Task 1 tests and commit**

Run:

```bash
uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap -q --tb=short
git diff --check
```

Commit:

```bash
git add src/streammuse/domain/rap/generation.py src/streammuse/domain/rap/models.py src/streammuse/domain/rap/__init__.py src/streammuse/application/rap/rhythm.py src/streammuse/application/rap/service.py src/streammuse/application/rap/realtime.py src/streammuse/infrastructure/rap/flow_prompt.py src/streammuse/infrastructure/rap/generators.py tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap
git commit -m "feat(rap): provide flow structure to lyric generation"
```

---

### Task 2: Structured Monitoring State

**Files:**
- Create: `src/streammuse/application/rap/monitoring_payloads.py`
- Create: `src/streammuse/presentation/rap_demo/terminal_state.py`
- Modify: `src/streammuse/application/rap/realtime.py:208-451`
- Modify: `src/streammuse/application/rap/runtime.py:49-83`
- Modify: `src/streammuse/presentation/rap_demo/cli.py:69-114`
- Test: `tests/unit/application/rap/test_monitoring_payloads.py`
- Modify: `tests/unit/application/rap/test_realtime.py`
- Modify: `tests/unit/application/rap/test_runtime.py`
- Test: `tests/unit/presentation/rap_demo/test_terminal_state.py`

**Interfaces:**
- Consumes: Task 1 `CandidateRequest.flow_template`, existing `RapEvent` stream, `ScheduledSyllable`, and session manifest values.
- Produces: `flow_template_payload(template)`, `scheduled_syllables_payload(items, bar)`, `TerminalRapStateProjector.apply(event)`, and read-only `TerminalRapViewState` snapshots.

- [ ] **Step 1: Write failing payload tests**

Create tests asserting complete JSON-ready values:

```python
def test_flow_template_payload_preserves_every_alignment_field() -> None:
    payload = flow_template_payload(BUILTIN_TEMPLATES.get("baseline_syncopated_9"))

    assert payload["template_id"] == "baseline_syncopated_9"
    assert payload["ticks_per_beat"] == 4
    assert payload["beats_per_bar"] == 4
    assert [slot["tick_in_bar"] for slot in payload["slots"]] == [0, 2, 3, 5, 7, 8, 10, 13, 15]
    assert payload["slots"][-1]["boundary_strength"] == 3
    assert payload["slots"][-1]["rhyme_group"] == "A"
    json.dumps(payload)
```

Also test that scheduled payloads use relative `tick_in_bar`, label, word, stress, slot index, and target stress.

- [ ] **Step 2: Run payload tests and verify RED**

Run:

```bash
uv run pytest tests/unit/application/rap/test_monitoring_payloads.py -q --tb=short
```

Expected: module import failure.

- [ ] **Step 3: Implement JSON-ready payload builders**

Use explicit dictionaries rather than `asdict()` so the monitoring contract is stable:

```python
def flow_template_payload(template: FlowTemplate) -> dict[str, Any]:
    return {
        "template_id": template.template_id,
        "name": template.name,
        "ticks_per_beat": template.ticks_per_beat,
        "beats_per_bar": template.beats_per_bar,
        "provenance": {
            "kind": template.provenance.kind,
            "source": template.provenance.source,
            "source_hash": template.provenance.source_hash,
            "quantization_error_ticks": template.provenance.quantization_error_ticks,
        },
        "slots": [
            {
                "slot_index": index,
                "tick_in_bar": slot.tick_in_bar,
                "duration_ticks": slot.duration_ticks,
                "target_stress": slot.target_stress,
                "boundary_strength": slot.boundary_strength,
                "rhyme_group": slot.rhyme_group,
            }
            for index, slot in enumerate(template.slots)
        ],
    }

def scheduled_syllables_payload(
    scheduled: tuple[ScheduledSyllable, ...], *, bar: int
) -> list[dict[str, Any]]:
    return [
        {
            "slot_index": item.slot.slot_index,
            "tick_in_bar": item.slot.tick - (bar * 16),
            "target_stress": item.slot.accent,
            "label": item.syllable.label,
            "word": item.syllable.word,
            "stress": item.syllable.stress,
            "stressed": item.syllable.stressed,
        }
        for item in scheduled
    ]
```

Include provenance kind/source/hash/quantization error and every slot's tick, duration, stress, boundary, and rhyme group.

- [ ] **Step 4: Write failing controller-event tests**

Extend realtime tests to assert:

```python
planning = payload_for_bar(events, "bar_planning_started", 1)
assert planning["context_lines"] == []
assert planning["seed"] == 20260808
assert planning["flow"]["slots"][1]["tick_in_bar"] == 2

frozen = payload_for_bar(events, "bar_frozen", 0)
assert len(frozen["scheduled_syllables"]) == 9
assert frozen["flow"]["template_id"] == "baseline_syncopated_9"
```

Assert `BAR_RESERVED` includes `flow`, and `BAR_REPLACED` includes the selected scheduled syllables for the queued preview.

- [ ] **Step 5: Run controller tests and verify RED**

Run:

```bash
uv run pytest tests/unit/application/rap/test_realtime.py -q --tb=short
```

Expected: missing structured request/flow/alignment keys.

- [ ] **Step 6: Emit structured request, flow, and alignment data**

Use Task 2 payload builders in `_reserve_through()`, `_submit_next_primary()`, `_drain_primary_result()`, and `_freeze()`. Do not parse prompt strings or duplicate scoring formulas.

- [ ] **Step 7: Write failing session-metadata tests**

Add a `RapDemoDependencies` test that supplies:

```python
session_metadata = {
    "scenario_id": "default_research_demo",
    "generator": "local_chat",
    "model": "qwen-rap",
    "terminal_layout": "split",
}
```

Assert `SESSION_STARTED` merges these values with `tempo_bpm`, `ticks_per_beat`, `beats_per_bar`, and `max_bars`.

- [ ] **Step 8: Run runtime tests and verify RED**

Run:

```bash
uv run pytest tests/unit/application/rap/test_runtime.py -q --tb=short
```

Expected: constructor/signature failure because `session_metadata` does not exist.

- [ ] **Step 9: Add immutable session metadata to runtime assembly**

Add `session_metadata: Mapping[str, Any]` to `RapDemoDependencies`, copy it during construction, and merge it into the start event without allowing it to overwrite canonical tempo/meter/max-bar values. Pass the sanitized manifest subset from `build_demo()`.

- [ ] **Step 10: Write failing projector tests**

Create `test_terminal_state.py` with ordered fixtures covering session start, reserve, plan, batch, candidate evaluation, replacement, freeze, tick, syllable, error, fallback, and stop. Assert:

```python
projector = TerminalRapStateProjector(history_limit=5)
for event in events:
    projector.apply(event)

state = projector.state
assert state.current_bar == 1
assert state.bars[1].text == "Galaxies dance in a cosmic fight"
assert state.bars[1].scheduled_syllables[0]["tick_in_bar"] == 0
assert state.latest_request.flow["template_id"] == "baseline_syncopated_9"
assert state.latest_batch.prompt[0]["role"] == "system"
assert state.candidates[0].rejection_reasons == ("syllable_count:10!=9",)
assert len(state.recent_events) == 5
```

- [ ] **Step 11: Run projector tests and verify RED**

Run:

```bash
uv run pytest tests/unit/presentation/rap_demo/test_terminal_state.py -q --tb=short
```

Expected: module import failure.

- [ ] **Step 12: Implement bounded terminal projection**

Define focused view dataclasses for bar, request, batch, candidate, and session state. `TerminalRapStateProjector.apply()` updates them from structured event payloads and stores `deque(maxlen=history_limit)` for trace events. Copy mutable event payload values into immutable tuples/dicts so later producer mutation cannot affect presentation state.

- [ ] **Step 13: Run Task 2 tests and commit**

Run:

```bash
uv run pytest tests/unit/application/rap tests/unit/presentation/rap_demo/test_terminal_state.py -q --tb=short
git diff --check
```

Commit:

```bash
git add src/streammuse/application/rap/monitoring_payloads.py src/streammuse/application/rap/realtime.py src/streammuse/application/rap/runtime.py src/streammuse/presentation/rap_demo/cli.py src/streammuse/presentation/rap_demo/terminal_state.py tests/unit/application/rap tests/unit/presentation/rap_demo/test_terminal_state.py
git commit -m "feat(rap): project structured realtime research state"
```

---

### Task 3: Dense Split and Structured Stream Renderers

**Files:**
- Create: `src/streammuse/presentation/rap_demo/terminal_stream.py`
- Create: `src/streammuse/presentation/rap_demo/terminal_dashboard.py`
- Rewrite: `src/streammuse/presentation/rap_demo/terminal.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Rewrite: `tests/unit/presentation/rap_demo/test_terminal.py`
- Test: `tests/unit/presentation/rap_demo/test_terminal_dashboard.py`
- Test: `tests/unit/presentation/rap_demo/test_terminal_stream.py`

**Interfaces:**
- Consumes: Task 2 `TerminalRapStateProjector` and `TerminalRapViewState`.
- Produces: `StructuredStreamRenderer`, `build_dashboard(state, detail, width) -> RenderableType`, `RichLiveRenderer`, and `TerminalRapSink(detail="full", layout="auto", write=None, console=None, is_tty=None, terminal_width=None, dashboard_factory=None)`.

- [ ] **Step 1: Add Rich as a direct dependency**

Add `"rich>=14.0.0"` to project dependencies and run:

```bash
uv lock
uv sync --frozen
```

Expected: Rich remains version-compatible with the existing lock and is importable from the project environment.

- [ ] **Step 2: Write failing structured-stream tests**

Create tests with an injected `lines.append` writer:

```python
def test_stream_groups_events_by_bar_and_phase_without_ansi() -> None:
    lines: list[str] = []
    renderer = StructuredStreamRenderer(detail="full", write=lines.append)

    renderer.render(projected_state, candidate_event)

    output = "\n".join(lines)
    assert "[BAR 02][GATE]" in output
    assert "9/9" in output
    assert "stress_alignment" in output
    assert "\x1b[" not in output
```

Cover PLAN, MODEL, GATE, SELECT, PLAY, FALLBACK, ERROR, exact prompts in full mode, and detail boundaries.

- [ ] **Step 3: Run stream tests and verify RED**

Run:

```bash
uv run pytest tests/unit/presentation/rap_demo/test_terminal_stream.py -q --tb=short
```

Expected: module import failure.

- [ ] **Step 4: Implement the structured append-only renderer**

Render each incoming event once with aligned phase prefixes and compact continuation rows. Use only plain strings. Read candidate and flow values from projector state/event payloads; do not maintain a second independent interpretation of request lifecycle.

- [ ] **Step 5: Write failing pure dashboard tests**

Use `Console(record=True, color_system=None, width=160)`:

```python
def test_wide_dashboard_contains_performance_flow_context_and_ranking() -> None:
    console = Console(record=True, color_system=None, width=160)
    console.print(build_dashboard(projected_state, detail="full", width=160))
    output = console.export_text()

    for label in (
        "LIVE DELIVERY", "QUEUE", "CLOCK + HEALTH", "LLM REQUEST",
        "EXACT CONTEXT", "CANDIDATE GATE + RANKING", "SELECTED SCORE",
    ):
        assert label in output
    assert "0  1  2  3" in output
    assert "S . w M" in output
    assert "Syllable ticks: [0, 2, 3, 5, 7, 8, 10, 13, 15]" in output
```

Add tests for selected/rejected/OOV markers, full prompt bodies, score components, current tick, and scheduled labels.

- [ ] **Step 6: Run dashboard tests and verify RED**

Run:

```bash
uv run pytest tests/unit/presentation/rap_demo/test_terminal_dashboard.py -q --tb=short
```

Expected: module import failure.

- [ ] **Step 7: Implement the Rich dashboard builder**

Build two columns at width `>= 120` and a vertical `Group` below 120. Use compact `Table.grid()` layouts rather than nested decorative panels. Apply functional colors only: cyan headings/current flow, green selected/ready, amber fallback/pending, red rejected/error, dim metadata. Let Rich wrap lyrics and prompt bodies; never truncate them.

The flow strip must render all ticks and slots from the structured flow snapshot, not from a built-in template lookup.

- [ ] **Step 8: Write failing facade and fallback tests**

Rewrite sink tests around:

```python
class RecordingDashboard:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.states = []
        self.closed = False

    def render(self, state) -> None:
        if self.fail:
            raise RuntimeError("render failed")
        self.states.append(state)

    def close(self) -> None:
        self.closed = True


def test_auto_uses_stream_for_non_tty() -> None:
    lines = []
    sink = TerminalRapSink(layout="auto", write=lines.append, is_tty=False)
    sink(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0, "max_bars": 3}))
    assert any("[SESSION][START]" in line for line in lines)


def test_auto_uses_split_for_wide_tty() -> None:
    dashboard = RecordingDashboard()
    sink = TerminalRapSink(
        layout="auto",
        is_tty=True,
        terminal_width=160,
        dashboard_factory=lambda **_: dashboard,
    )
    sink(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0, "max_bars": 3}))
    assert len(dashboard.states) == 1


def test_explicit_split_stacks_at_narrow_width() -> None:
    console = Console(record=True, color_system=None, width=90)
    console.print(build_dashboard(projected_state, detail="full", width=90))
    output = console.export_text()
    assert output.index("LIVE DELIVERY") < output.index("LLM REQUEST")


def test_dashboard_failure_switches_once_to_stream() -> None:
    lines = []
    dashboard = RecordingDashboard(fail=True)
    sink = TerminalRapSink(
        layout="split",
        write=lines.append,
        is_tty=True,
        terminal_width=160,
        dashboard_factory=lambda **_: dashboard,
    )
    sink(_event(RapEventType.SESSION_STARTED, {"tempo_bpm": 92.0, "max_bars": 3}))
    sink(_event(RapEventType.TICK, {"beat": 0, "tick_in_beat": 0}, tick=0))
    assert sum("[PRESENTATION][WARN]" in line for line in lines) == 1
    assert any("[TICK]" in line for line in lines)


def test_session_stop_closes_live_renderer() -> None:
    dashboard = RecordingDashboard()
    sink = TerminalRapSink(
        layout="split",
        is_tty=True,
        terminal_width=160,
        dashboard_factory=lambda **_: dashboard,
    )
    sink(_event(RapEventType.SESSION_STOPPED, {}))
    assert dashboard.closed is True
```

Inject renderer factories and terminal capabilities so tests do not depend on the host terminal.

- [ ] **Step 9: Run facade tests and verify RED**

Run:

```bash
uv run pytest tests/unit/presentation/rap_demo/test_terminal.py -q --tb=short
```

Expected: constructor/signature assertions fail because the current sink has no layout/projector/renderers.

- [ ] **Step 10: Implement the sink facade and Live adapter**

Required constructor surface:

```python
class TerminalRapSink:
    def __init__(
        self,
        detail: str = "full",
        *,
        layout: str = "auto",
        write: Callable[[str], None] | None = None,
        console: Console | None = None,
        is_tty: bool | None = None,
        terminal_width: int | None = None,
        dashboard_factory: Callable[..., DashboardRenderer] | None = None,
    ) -> None:
        if detail not in {"summary", "candidates", "full"}:
            raise ValueError("terminal detail must be summary, candidates, or full")
        if layout not in {"auto", "split", "stream"}:
            raise ValueError("terminal layout must be auto, split, or stream")
        self._projector = TerminalRapStateProjector()
        self._stream = StructuredStreamRenderer(detail=detail, write=write)
        tty = bool(is_tty) if is_tty is not None else self._stream.is_tty
        width = terminal_width or self._stream.terminal_width
        use_split = layout == "split" or (layout == "auto" and tty and width >= 120)
        factory = dashboard_factory or RichLiveRenderer
        self._dashboard = factory(detail=detail, console=console, width=width) if use_split else None
        self._dashboard_failed = False

    def __call__(self, event: RapEvent) -> None:
        state = self._projector.apply(event)
        if self._dashboard is None:
            self._stream.render(state, event)
            return
        try:
            self._dashboard.render(state)
            if event.event_type == RapEventType.SESSION_STOPPED:
                self._dashboard.close()
        except Exception as exc:
            self._dashboard.close()
            self._dashboard = None
            if not self._dashboard_failed:
                self._stream.warning(f"dashboard disabled: {type(exc).__name__}")
                self._dashboard_failed = True
            self._stream.render(state, event)
```

Preserve `TerminalRapSink(detail, write=...)` behavior by selecting stream in `auto` mode when `write` is supplied and `is_tty` is not explicitly true. On dashboard failure, stop Live, emit one `[PRESENTATION][WARN]` line through the stream writer, switch permanently to stream, and render the triggering event once.

- [ ] **Step 11: Run renderer tests and commit**

Run:

```bash
uv run pytest tests/unit/presentation/rap_demo/test_terminal.py tests/unit/presentation/rap_demo/test_terminal_state.py tests/unit/presentation/rap_demo/test_terminal_stream.py tests/unit/presentation/rap_demo/test_terminal_dashboard.py -q --tb=short
git diff --check
```

Commit:

```bash
git add pyproject.toml uv.lock src/streammuse/presentation/rap_demo/terminal.py src/streammuse/presentation/rap_demo/terminal_state.py src/streammuse/presentation/rap_demo/terminal_stream.py src/streammuse/presentation/rap_demo/terminal_dashboard.py tests/unit/presentation/rap_demo
git commit -m "feat(rap): add dense split terminal research monitor"
```

---

### Task 4: CLI Integration and Live Verification

**Files:**
- Modify: `src/streammuse/presentation/rap_demo/cli.py:32-114`
- Modify: `tests/unit/presentation/rap_demo/test_cli.py`
- Modify: `tests/unit/presentation/rap/test_rap_cli.py`
- Modify: `.superpowers/sdd/2026-08-07-research-realtime-rap/live-showcase-report.md`

**Interfaces:**
- Consumes: Task 3 `TerminalRapSink(detail="full", layout="auto")` and Task 2 session metadata.
- Produces: `--terminal-layout auto|split|stream`, resolved manifest metadata, local smoke evidence, and H200 Qwen evidence.

- [ ] **Step 1: Write failing parser and assembly tests**

Add:

```python
def test_parser_defaults_to_auto_terminal_layout() -> None:
    assert build_parser().parse_args([]).terminal_layout == "auto"


def test_build_demo_passes_terminal_layout_and_session_metadata(tmp_path, monkeypatch) -> None:
    args = build_parser().parse_args([
        "--terminal-layout", "stream",
        "--log-dir", str(tmp_path),
        "--max-bars", "2",
    ])
    demo = build_demo(args)
    assert demo.session_metadata["terminal_layout"] == "stream"
    assert demo.session_metadata["scenario_id"] == "default_research_demo"
```

Assert `session.json` records the resolved layout.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```bash
uv run pytest tests/unit/presentation/rap_demo/test_cli.py tests/unit/presentation/rap/test_rap_cli.py -q --tb=short
```

Expected: parser failure because `--terminal-layout` is unknown.

- [ ] **Step 3: Wire layout and metadata through dependency assembly**

Add:

```python
parser.add_argument(
    "--terminal-layout",
    choices=("auto", "split", "stream"),
    default="auto",
)
```

Pass it to `TerminalRapSink`, session metadata, and `session.json`. Do not change generator, lookahead, threshold, or default detail behavior.

- [ ] **Step 4: Run all affected tests**

Run:

```bash
uv run pytest \
  tests/unit/domain/rap \
  tests/unit/application/rap \
  tests/unit/infrastructure/rap \
  tests/unit/presentation/rap \
  tests/unit/presentation/rap_demo \
  tests/unit/presentation/test_cli_config_parser.py \
  tests/integration/test_cli_entry_point.py \
  -q --tb=short
```

Expected: all tests pass; only the existing `pretty_midi/pkg_resources` warning is acceptable.

- [ ] **Step 5: Run deterministic local smoke sessions**

Run stream capture:

```bash
uv run streammuse-rap-demo \
  --generator scripted_failure \
  --max-bars 2 \
  --terminal-layout stream \
  --terminal-detail full \
  --log-dir /tmp/streammuse-rap-terminal-failure \
  --no-web
```

Verify PLAN, exact flow, ERROR, FALLBACK, PLAY, and clean session stop are readable without ANSI escapes.

Run split mode in a PTY:

```bash
uv run streammuse-rap-demo \
  --generator phrase_bank \
  --max-bars 2 \
  --candidate-count 4 \
  --terminal-layout split \
  --terminal-detail full \
  --log-dir /tmp/streammuse-rap-terminal-split \
  --no-web
```

Verify both columns or narrow stacking, flow strip, candidate rejection rows, armed fallback, and tick progress.

- [ ] **Step 6: Deploy the committed tree to an isolated H200 directory**

Use a new directory under `/data/home/Andrew.Yang/StreamMUSE/rap_demo_runs/`; do not modify the existing dirty `StreamMUSE-v1` checkout. Reuse the existing isolated environment pattern, install the project editable with `--no-deps`, and confirm Rich and pronouncing imports.

- [ ] **Step 7: Run a finite real-Qwen H200 session**

With `qwen-rap` served on an unused H200 GPU:

```bash
streammuse-rap-demo \
  --generator local_chat \
  --model-url http://127.0.0.1:8001/v1 \
  --model qwen-rap \
  --candidate-count 8 \
  --lookahead-bars 2 \
  --minimum-score 0.3 \
  --seed 20260807 \
  --max-bars 3 \
  --terminal-layout split \
  --terminal-detail full \
  --no-web
```

Confirm the displayed template arrays exactly match the prompt and the template used by alignment/ranking. Record candidate count, valid count, selected scores, generation latency, tick jitter, and fallback activation. Do not claim flow-quality improvement from one run.

- [ ] **Step 8: Commit integration and verification notes**

Run `git diff --check`, ensure the worktree is clean after the commit, and commit:

```bash
git add src/streammuse/presentation/rap_demo/cli.py tests/unit/presentation/rap_demo/test_cli.py tests/unit/presentation/rap/test_rap_cli.py
git commit -m "feat(rap): expose terminal layout controls"
```

Append verification evidence to the ignored SDD report without staging it.

---

## Final Review Gate

- [ ] Dispatch one fresh whole-branch reviewer for the range `14de757e..HEAD`.
- [ ] Require findings first, ordered by severity with exact file/line references.
- [ ] Fix load-bearing findings through one scoped implementation/re-review loop.
- [ ] Re-run the affected suite and one finite H200 Qwen session after the final fix.
- [ ] Push `feature/real_rap` only after verification and report the exact commit, command, and residual risks.
