# Rap Alignment Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline-first `streammuse-rap` prototype that generates candidate lines, schedules syllables to StreamMUSE ticks, renders the schedule, exports JSON, and can play the schedule in terminal time.

**Architecture:** A separate rap vertical slice shares only the stable `Tempo` value object with the existing real-time accompaniment system. Pure domain models and prosody analysis feed application-level rhythm and alignment services. Infrastructure supplies candidate generators and presentation owns CLI, serialization, output, and timed terminal playback.

**Tech Stack:** Python 3.10+, `dataclasses`, `argparse`, `json`, `time`, existing `requests`-based `LocalChatModelClient`, pytest. No new dependency is required; optional future pronunciation packages are not part of this prototype.

## Global Constraints

- Keep `RealTimeMusicService`, `MusicalEvent`, and the accompaniment inference protocol unchanged.
- Use StreamMUSE's existing default timing: 4 ticks per beat and 4 beats per bar; one rap slot is one tick.
- `phrase_bank` is the default and must work without a model server or network.
- LLM text is candidate material only; deterministic local analysis and alignment choose the schedule.
- A scheduled line must have no more syllables than its 16 available bar slots.
- All production behaviour is introduced through red-green TDD and all new public functions have focused tests.

---

### Task 1: Add domain rap contracts and offline prosody analysis

**Files:**
- Create: `src/streammuse/domain/rap/__init__.py`
- Create: `src/streammuse/domain/rap/models.py`
- Create: `src/streammuse/domain/rap/prosody.py`
- Create: `tests/unit/domain/rap/test_prosody.py`

**Interfaces:**
- Produces `Syllable(word: str, index_in_word: int, syllable_count: int, stressed: bool)`.
- Produces `analyse_syllables(text: str) -> tuple[Syllable, ...]`.
- Produces immutable `BeatSlot`, `ScheduledSyllable`, `AlignedLine`, `CandidateBatch`, and `RapPlan` contracts consumed by application, infrastructure, and presentation code.

- [x] **Step 1: Write failing prosody tests**

```python
from streammuse.domain.rap import analyse_syllables

def test_analyse_syllables_marks_word_boundaries_and_irregular_counts() -> None:
    syllables = analyse_syllables("rhythm exploration")
    assert [(item.word, item.index_in_word, item.syllable_count, item.stressed) for item in syllables] == [
        ("rhythm", 0, 2, True), ("rhythm", 1, 2, False),
        ("exploration", 0, 4, True), ("exploration", 1, 4, False),
        ("exploration", 2, 4, False), ("exploration", 3, 4, False),
    ]
```

- [x] **Step 2: Run the focused test and verify it fails because `streammuse.domain.rap` does not exist**

Run: `uv run pytest tests/unit/domain/rap/test_prosody.py -q`

Expected: collection failure for missing module.

- [x] **Step 3: Add the smallest pure-domain implementation**

```python
@dataclass(frozen=True)
class Syllable:
    word: str
    index_in_word: int
    syllable_count: int
    stressed: bool

    @property
    def label(self) -> str:
        return self.word if self.index_in_word == 0 else "."

def analyse_syllables(text: str) -> tuple[Syllable, ...]:
    return tuple(
        syllable
        for word in _WORDS.findall(text.lower())
        for syllable in _syllables_for_word(word)
    )
```

Use a small documented irregular-count mapping and vowel-group heuristic with a
minimum count of one. Implement the immutable scheduling data contracts in the
same module and re-export their names from `domain/rap/__init__.py`.

- [x] **Step 4: Run the focused domain test and confirm it passes**

Run: `uv run pytest tests/unit/domain/rap/test_prosody.py -q`

Expected: PASS.

### Task 2: Plan beat slots and select a deterministic syllable alignment

**Files:**
- Create: `src/streammuse/application/rap/__init__.py`
- Create: `src/streammuse/application/rap/rhythm.py`
- Create: `src/streammuse/application/rap/alignment.py`
- Create: `tests/unit/application/rap/test_rhythm.py`
- Create: `tests/unit/application/rap/test_alignment.py`

**Interfaces:**
- Produces `available_patterns() -> tuple[str, ...]` and `build_bar_slots(tempo: Tempo, pattern: str, bar: int) -> tuple[BeatSlot, ...]`.
- Produces `align_text_to_slots(text: str, slots: Sequence[BeatSlot]) -> AlignedLine` and `choose_best_line(candidates: Sequence[str], slots: Sequence[BeatSlot]) -> AlignedLine`.
- Consumes `Tempo`, `BeatSlot`, and `analyse_syllables` from Task 1.

- [x] **Step 1: Write failing rhythm and overflow-selection tests**

```python
def test_boom_bap_slots_share_streammuse_tick_coordinates() -> None:
    slots = build_bar_slots(Tempo(92, 4, 4), "boom_bap", bar=2)
    assert len(slots) == 16
    assert slots[0].tick == 32
    assert slots[0].beat == 0
    assert slots[4].beat == 1
    assert slots[0].accent > slots[1].accent

def test_choose_best_line_rejects_overflow_when_a_fitting_candidate_exists() -> None:
    slots = build_bar_slots(Tempo(92, 4, 4), "boom_bap", bar=0)
    line = choose_best_line(("one two three four", "one " * 17), slots)
    assert line.text == "one two three four"
    assert line.overflow_count == 0
```

- [x] **Step 2: Run the focused tests and verify missing-module failures**

Run: `uv run pytest tests/unit/application/rap/test_rhythm.py tests/unit/application/rap/test_alignment.py -q`

Expected: collection failure for missing module.

- [x] **Step 3: Implement 16-slot 4/4 patterns and dynamic-programming alignment**

```python
def build_bar_slots(tempo: Tempo, pattern: str, bar: int) -> tuple[BeatSlot, ...]:
    accents = _PATTERN_ACCENTS[pattern]
    if tempo.ticks_per_bar != len(accents):
        raise ValueError("rap patterns require exactly 16 ticks per bar")
    return tuple(
        BeatSlot(bar=bar, tick=bar * tempo.ticks_per_bar + index,
                 beat=index // tempo.ticks_per_beat,
                 tick_in_beat=index % tempo.ticks_per_beat, accent=accent)
        for index, accent in enumerate(accents)
    )
```

For an in-range syllable count, compute the maximum-scoring monotonic slot path
with the first syllable assigned to the downbeat. Reward stressed syllables on
accented slots, add a small beat-onset preference, and penalize distance from
evenly spread ideal positions. Return no events and a severe negative score for
overflow. `choose_best_line` must preserve candidate order for equal scores.

- [x] **Step 4: Run focused application tests and confirm they pass**

Run: `uv run pytest tests/unit/application/rap/test_rhythm.py tests/unit/application/rap/test_alignment.py -q`

Expected: PASS.

### Task 3: Add candidate-generation adapters and a planning service

**Files:**
- Create: `src/streammuse/application/rap/service.py`
- Create: `src/streammuse/infrastructure/rap/__init__.py`
- Create: `src/streammuse/infrastructure/rap/generators.py`
- Create: `tests/unit/application/rap/test_service.py`
- Create: `tests/unit/infrastructure/rap/test_generators.py`

**Interfaces:**
- Produces `RapPrototypeService(tempo: Tempo, pattern: str, generator: CandidateGenerator).build_plan(topic: str, bars: int, candidate_count: int) -> RapPlan`.
- Defines a structural `CandidateGenerator` protocol returning `CandidateBatch`.
- Produces `PhraseBankGenerator.generate(topic: str, count: int) -> CandidateBatch` and `LocalChatCandidateGenerator.generate(topic: str, count: int) -> CandidateBatch`.

- [x] **Step 1: Write failing selection and fallback tests**

```python
def test_phrase_bank_plan_contains_a_unique_aligned_line_per_requested_bar() -> None:
    plan = RapPrototypeService(Tempo(92, 4, 4), "boom_bap", PhraseBankGenerator()).build_plan(
        "space travel", bars=2, candidate_count=8
    )
    assert len(plan.lines) == 2
    assert all(line.events for line in plan.lines)
    assert len({line.text for line in plan.lines}) == 2

def test_local_chat_falls_back_to_phrase_bank_after_client_error() -> None:
    batch = LocalChatCandidateGenerator(FailingClient(), PhraseBankGenerator()).generate("space travel", 4)
    assert batch.source == "phrase_bank"
    assert batch.warning is not None
```

- [x] **Step 2: Run the focused tests and verify missing-module failures**

Run: `uv run pytest tests/unit/application/rap/test_service.py tests/unit/infrastructure/rap/test_generators.py -q`

Expected: collection failure for missing modules.

- [x] **Step 3: Implement the offline phrase bank, optional chat adapter, and service**

```python
class CandidateGenerator(Protocol):
    def generate(self, topic: str, count: int) -> CandidateBatch: ...

class PhraseBankGenerator:
    def generate(self, topic: str, count: int) -> CandidateBatch:
        clean_topic = _normalise_topic(topic)
        return CandidateBatch(tuple(template.format(topic=clean_topic) for template in _TEMPLATES[:count]), "phrase_bank")
```

Prompt `LocalChatModelClient` for newline-separated, one-bar candidate lines;
strip bullets and blank lines from its response. Catch adapter failures, empty
responses, and malformed text, then return a phrase-bank batch with an explicit
warning. The service obtains a generous candidate pool, chooses one fitting
line per bar without repeating it until necessary, and propagates generator
metadata into `RapPlan`.

- [x] **Step 4: Run focused service and adapter tests**

Run: `uv run pytest tests/unit/application/rap/test_service.py tests/unit/infrastructure/rap/test_generators.py -q`

Expected: PASS.

### Task 4: Expose the prototype through a CLI, JSON artifact, and timed terminal playback

**Files:**
- Create: `src/streammuse/presentation/rap/__init__.py`
- Create: `src/streammuse/presentation/rap/cli.py`
- Modify: `pyproject.toml`
- Create: `tests/unit/presentation/rap/test_rap_cli.py`
- Create: `docs/developer-guide/rap-alignment-prototype.md`

**Interfaces:**
- Adds `streammuse-rap = "streammuse.presentation.rap.cli:main"`.
- Produces `build_parser()`, `render_plan(plan) -> str`, `plan_to_dict(plan) -> dict[str, object]`, and `play_plan(plan, write=..., clock=..., sleep=...)`.
- Supports `--topic`, `--tempo`, `--pattern`, `--bars`, `--candidate-count`, `--generator`, `--model-url`, `--model`, `--output-json`, and `--play`.

- [x] **Step 1: Write failing CLI and serialized-schedule tests**

```python
def test_default_cli_run_writes_an_inspectable_schedule_json(tmp_path: Path, capsys) -> None:
    output = tmp_path / "plan.json"
    assert main(["--topic", "space travel", "--bars", "1", "--output-json", str(output)]) == 0
    payload = json.loads(output.read_text())
    assert payload["pattern"] == "boom_bap"
    assert payload["lines"][0]["events"][0]["tick"] == 0
    assert "Bar 1" in capsys.readouterr().out
```

- [x] **Step 2: Run the focused CLI test and verify missing-module failure**

Run: `uv run pytest tests/unit/presentation/rap/test_rap_cli.py -q`

Expected: collection failure for missing module.

- [x] **Step 3: Implement the command surface and record prototype operation**

Create the parser with conservative defaults: 92 BPM, `boom_bap`, four bars,
and the phrase bank. Render one line per bar followed by tick, musical
coordinate, seconds offset, accent, and syllable label. Write JSON using the
same events. `play_plan` waits until each schedule offset from a shared
`perf_counter` origin and prints the exact syllable event.

Document the three supported workflows and the deliberate limits: no live drum
input, no TTS, heuristic syllable counting, and no claim that terminal output
is audio playback.

- [x] **Step 4: Run the focused CLI test and a one-bar manual smoke command**

Run: `uv run pytest tests/unit/presentation/rap/test_rap_cli.py -q`

Run: `uv run streammuse-rap --topic "space travel" --tempo 92 --pattern boom_bap --bars 1 --output-json /tmp/rap-plan.json`

Expected: PASS test; command prints an aligned schedule and writes JSON with
one `lines[0].events` array.

### Task 5: Verify the whole slice and capture prototype findings

**Files:**
- Modify: `docs/developer-guide/rap-alignment-prototype.md`

**Interfaces:**
- Verification command: `uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap -q`.
- Regression command: `uv run pytest tests/unit/domain/timing/test_tempo.py tests/unit/infrastructure/inference/test_local_chat_model_client.py -q`.

- [x] **Step 1: Run the complete rap suite**

Run: `uv run pytest tests/unit/domain/rap tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap -q`

Expected: PASS with no warnings.

- [x] **Step 2: Run surrounding regression tests**

Run: `uv run pytest tests/unit/domain/timing/test_tempo.py tests/unit/infrastructure/inference/test_local_chat_model_client.py -q`

Expected: PASS.

- [x] **Step 3: Exercise an alternative pattern and timed path**

Run: `uv run streammuse-rap --topic "ocean research" --tempo 120 --pattern trap_sparse --bars 1 --play`

Expected: printed event timestamps follow tick offsets from 0 through the last scheduled syllable; the command exits normally.

- [x] **Step 4: Update the prototype record with observed results**

Record the exact commands, passed test counts, the offline prosody-library
check, what the outputs prove, and the next integration increment: a
timestamped keyboard-drum event buffer feeding the same beat-slot planner.

## Execution Record

All five tasks were completed inline on 2026-07-17. Development followed
red-green cycles for domain prosody, rhythm/alignment, candidate generation,
and CLI behaviour. Two manually observed prosody defects from smoke runs were
captured as failing tests before correction. The final verification command
reported `43 passed in 0.28s`, followed by the full repository result of `292
passed, 1 skipped`; exact commands and smoke observations are in
`docs/developer-guide/rap-alignment-prototype.md`.
