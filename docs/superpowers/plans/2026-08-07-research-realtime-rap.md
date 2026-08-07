# Research-Grade Real-Time Rap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a continuously running, beat-aligned rap-text experiment whose flow templates, generated candidates, pronunciation analysis, ranking, fallback decisions, timing, and outputs are observable live and reproducible from session artifacts.

**Architecture:** Extend the existing rap modules rather than replacing StreamMUSE's timing system. A scenario chooses a topic and `FlowTemplate` for each bar; an asynchronous generator proposes text; a pronunciation analyzer and deterministic ranker produce inspectable `CandidateEvaluation` records; a rolling controller freezes a generated or prevalidated fallback bar before playback; one canonical event stream drives terminal output, JSONL artifacts, metrics, and a read-only FastAPI/WebSocket monitor. The standalone demo owns a tick loop for controlled experiments, while the existing `streammuse-cli` continues to inject the same controller through its current `TickObserver` seam.

**Tech Stack:** Python 3.10+, dataclasses, `pronouncing`/CMU Pronouncing Dictionary, existing `Tempo` and `TickObserver`, `ThreadPoolExecutor`, FastAPI, Uvicorn, WebSockets, vanilla HTML/CSS/JavaScript, pytest, JSONL, CSV.

## Global Constraints

- Keep the active musical timeline at four ticks per beat and four beats per bar for this increment.
- Do not add speech synthesis, microphone input, live drum extraction, model training, or a learned quality ranker.
- Never block the musical tick path on HTTP generation, disk logging, WebSocket delivery, or UI rendering.
- Freeze a bar no later than its first tick; never replace or mutate it after freeze.
- Require exactly one analyzed syllable per flow slot. Do not silently stretch, compress, drop, or invent syllables.
- Preserve the current vowel-group analyzer as an explicitly named baseline and OOV fallback.
- Record every candidate, analysis source, component score, rejection reason, timing measurement, selection, and fallback reason.
- Record prompts and raw model responses, but never API keys, authorization headers, or environment secrets.
- Treat built-in templates as `hand_authored_mcflow_inspired`; do not represent them as extracted MCFlow data.
- Do not commit MCFlow lyrics, artist identifiers, or source transcriptions. The optional extractor emits anonymous structural templates from a user-supplied checkout.
- The browser is read-only and projects the canonical event stream; it is never a second runtime authority.
- All scoring weights and thresholds are session configuration, not hidden constants.
- State the limitation that deterministic scores measure chosen proxies, not human-perceived rap quality.
- Work with the current dirty tree. Do not revert or reformat unrelated input, WebSocket, piano, or runtime changes.

---

## Design Rationale

### Why template-first

MCFlow contains symbolic transcriptions of performed rap flow, while the current StreamMUSE clock already provides stable musical ticks. A flow template is therefore the narrowest useful bridge: it describes syllable-bearing positions, target stress, phrase breaks, and rhyme locations independently of the lyric generator and independently of future voice synthesis. This follows the intermediate-representation strategy seen in controllable music systems and lets each component be replaced without changing the runtime contract.

### Why generate many and rank locally

Language models are unreliable at exact syllable counting, and a single response gives no recovery path. DeepBeat demonstrated candidate selection using rhyme, structural, and semantic features; Shimon the Rapper used CMU-derived phonemes and explicit rhyme scoring. The first StreamMUSE ranker will be simpler and deterministic: the model proposes, while local code verifies and chooses. Every feature remains visible so a later learned ranker can be compared against the baseline.

### Why reserve fallback before generation

The musical clock cannot wait. Every future bar first receives a validated provisional fallback. A generated line may replace that reservation only before the bar freezes. This changes generation failure from a playback failure into an observable experimental outcome: fallback rate, failure reason, and deadline slack become measurable.

### Why one canonical event stream

Printing, browser state, and research files must not disagree. The controller emits ordered typed events; terminal, recorder, state projector, and WebSocket queue are independent sinks. `summary.json` and `bars.csv` are derived from those events, making sessions auditable and enabling replay-based tests.

## File Map

### Domain

- Create `src/streammuse/domain/rap/flow.py`: relative flow slots, templates, provenance, and absolute slot materialization.
- Create `src/streammuse/domain/rap/evaluation.py`: analyzer results, score components, candidate evaluations, selection, and frozen bars.
- Create `src/streammuse/domain/rap/events.py`: typed research event envelope and event names.
- Create `src/streammuse/domain/rap/scenario.py`: immutable topic/template schedule and lookup rules.
- Modify `src/streammuse/domain/rap/models.py`: pronunciation-aware syllables and enriched candidate request/batch records.
- Modify `src/streammuse/domain/rap/__init__.py`: public exports.

### Application

- Create `src/streammuse/application/rap/scoring.py`: validation, feature extraction, deterministic scoring, and ranking.
- Create `src/streammuse/application/rap/monitoring.py`: ordered publisher, state projection, and summary derivation.
- Create `src/streammuse/application/rap/runtime.py`: standalone monotonic tick loop.
- Modify `src/streammuse/application/rap/realtime.py`: scenario-aware rolling planning, deadlines, freeze, and fallback.
- Modify `src/streammuse/application/rap/service.py`: new generator/analyzer protocols and offline compatibility.
- Modify `src/streammuse/application/rap/alignment.py`: exact sequential mapping path plus retained named legacy baseline.
- Modify `src/streammuse/application/rap/rhythm.py`: compatibility mapping from old pattern names to built-in flow templates.

### Infrastructure

- Create `src/streammuse/infrastructure/rap/prosody.py`: CMU analyzer with explicit heuristic fallback.
- Create `src/streammuse/infrastructure/rap/templates.py`: built-in template catalog and JSON loading.
- Create `src/streammuse/infrastructure/rap/scenarios.py`: default scenario and external JSON loader.
- Create `src/streammuse/infrastructure/rap/recorder.py`: session manifest, canonical JSONL, derived CSV/summary.
- Create `src/streammuse/infrastructure/rap/mcflow.py`: optional Humdrum structure extractor.
- Modify `src/streammuse/infrastructure/rap/generators.py`: request-aware phrase-bank and local-chat generators with raw diagnostics.
- Modify `src/streammuse/infrastructure/rap/__init__.py`: public exports.

### Presentation

- Create `src/streammuse/presentation/rap_demo/__init__.py`.
- Create `src/streammuse/presentation/rap_demo/cli.py`: demo configuration and dependency assembly.
- Create `src/streammuse/presentation/rap_demo/server.py`: FastAPI lifecycle, snapshot API, and WebSocket broadcast.
- Create `src/streammuse/presentation/rap_demo/terminal.py`: human-readable event sink with detail levels.
- Create `src/streammuse/presentation/rap_demo/static/index.html`.
- Create `src/streammuse/presentation/rap_demo/static/css/rap-demo.css`.
- Create `src/streammuse/presentation/rap_demo/static/js/rap-demo.js`.
- Modify `src/streammuse/presentation/cli/cli.py`: adapt existing fixed-topic rap mode to the new controller.
- Modify `pyproject.toml`: pronunciation dependency and `streammuse-rap-demo` entry point.

### Research utilities and documentation

- Create `scripts/extract_mcflow_templates.py`: opt-in anonymous template extraction.
- Create `scripts/summarize_rap_session.py`: regenerate metrics and bar CSV from the canonical event stream.
- Create `docs/developer-guide/research-realtime-rap.md`: commands, artifacts, metric definitions, limitations, and experiment protocol.

---

### Task 1: Flow Templates and Prescheduled Scenarios

**Files:**
- Create: `src/streammuse/domain/rap/flow.py`
- Create: `src/streammuse/domain/rap/scenario.py`
- Create: `src/streammuse/infrastructure/rap/templates.py`
- Create: `src/streammuse/infrastructure/rap/scenarios.py`
- Modify: `src/streammuse/domain/rap/models.py`
- Modify: `src/streammuse/domain/rap/__init__.py`
- Test: `tests/unit/domain/rap/test_flow.py`
- Test: `tests/unit/domain/rap/test_scenario.py`
- Test: `tests/unit/infrastructure/rap/test_templates.py`

**Interfaces:**
- Produces: `FlowSlot`, `FlowTemplate`, `FlowProvenance`, `materialize_flow(template, bar)`, `ScenarioSegment`, `RapScenario`, `TemplateCatalog`, and `load_scenario(path)`.
- Consumes: existing `BeatSlot`, `Tempo`, and four-ticks-per-beat timing convention.

- [ ] **Step 1: Write failing template and scenario tests**

```python
def test_materialize_flow_preserves_relative_structure_at_absolute_bar() -> None:
    template = FlowTemplate(
        template_id="test_syncopated",
        name="Test syncopated",
        ticks_per_beat=4,
        beats_per_bar=4,
        provenance=FlowProvenance(kind="hand_authored_test", source="unit-test"),
        slots=(
            FlowSlot(tick_in_bar=0, duration_ticks=2, target_stress=1.0),
            FlowSlot(tick_in_bar=3, duration_ticks=1, target_stress=0.2),
            FlowSlot(tick_in_bar=8, duration_ticks=2, target_stress=0.9, rhyme_group="A"),
        ),
    )

    slots = materialize_flow(template, bar=2)

    assert [slot.tick for slot in slots] == [32, 35, 40]
    assert [slot.slot_index for slot in slots] == [0, 1, 2]
    assert slots[-1].rhyme_group == "A"


def test_scenario_changes_only_at_bar_boundaries_and_loops() -> None:
    scenario = RapScenario(
        scenario_id="test",
        tempo_bpm=96.0,
        loop=True,
        segments=(
            ScenarioSegment(start_bar=0, bars=2, topic="space", template_id="a", fallback_lines=("one two",)),
            ScenarioSegment(start_bar=2, bars=1, topic="deep sea", template_id="b", fallback_lines=("three four",)),
        ),
    )

    assert scenario.segment_for_bar(0).topic == "space"
    assert scenario.segment_for_bar(2).topic == "deep sea"
    assert scenario.segment_for_bar(3).topic == "space"
```

- [ ] **Step 2: Run tests and confirm missing contracts**

Run: `uv run pytest tests/unit/domain/rap/test_flow.py tests/unit/domain/rap/test_scenario.py -v`

Expected: collection fails because `FlowTemplate` and `RapScenario` do not exist.

- [ ] **Step 3: Implement validated flow and scenario models**

```python
@dataclass(frozen=True)
class FlowSlot:
    tick_in_bar: int
    duration_ticks: int
    target_stress: float
    boundary_strength: int = 0
    rhyme_group: str | None = None


@dataclass(frozen=True)
class FlowProvenance:
    kind: str
    source: str
    source_hash: str | None = None
    quantization_error_ticks: float = 0.0


@dataclass(frozen=True)
class FlowTemplate:
    template_id: str
    name: str
    ticks_per_beat: int
    beats_per_bar: int
    slots: tuple[FlowSlot, ...]
    provenance: FlowProvenance

    def __post_init__(self) -> None:
        ticks_per_bar = self.ticks_per_beat * self.beats_per_bar
        ticks = [slot.tick_in_bar for slot in self.slots]
        if not self.template_id or not self.slots:
            raise ValueError("flow template requires an id and at least one slot")
        if ticks != sorted(set(ticks)):
            raise ValueError("flow slots must have unique increasing onsets")
        if any(tick < 0 or tick >= ticks_per_bar for tick in ticks):
            raise ValueError("flow slot onset lies outside the bar")
        if any(slot.duration_ticks <= 0 for slot in self.slots):
            raise ValueError("flow slot duration must be positive")
        if any(not 0.0 <= slot.target_stress <= 1.0 for slot in self.slots):
            raise ValueError("target stress must be between zero and one")
        if any(not 0 <= slot.boundary_strength <= 5 for slot in self.slots):
            raise ValueError("boundary strength must be between zero and five")
```

Define the scenario and catalog contracts in the same task:

```python
@dataclass(frozen=True)
class ScenarioSegment:
    start_bar: int
    bars: int
    topic: str
    template_id: str
    fallback_lines: tuple[str, ...]


@dataclass(frozen=True)
class RapScenario:
    scenario_id: str
    tempo_bpm: float
    segments: tuple[ScenarioSegment, ...]
    loop: bool = True

    @property
    def total_bars(self) -> int:
        return sum(segment.bars for segment in self.segments)

    def segment_for_bar(self, bar: int) -> ScenarioSegment:
        if bar < 0:
            raise ValueError("bar must not be negative")
        effective = bar % self.total_bars if self.loop else bar
        for segment in self.segments:
            if segment.start_bar <= effective < segment.start_bar + segment.bars:
                return segment
        raise IndexError(f"bar {bar} lies outside scenario {self.scenario_id}")


class TemplateCatalog:
    def __init__(self, templates: Iterable[FlowTemplate]) -> None:
        by_id = {template.template_id: template for template in templates}
        if not by_id:
            raise ValueError("template catalog must not be empty")
        self._by_id = by_id

    @classmethod
    def from_templates(cls, templates: Iterable[FlowTemplate]) -> "TemplateCatalog":
        return cls(templates)

    def get(self, template_id: str) -> FlowTemplate:
        try:
            return self._by_id[template_id]
        except KeyError as exc:
            raise ValueError(f"unknown flow template: {template_id}") from exc
```

Extend `BeatSlot` with defaulted research fields so current callers remain valid:

```python
@dataclass(frozen=True)
class BeatSlot:
    bar: int
    tick: int
    beat: int
    tick_in_beat: int
    accent: float
    duration_ticks: int = 1
    boundary_strength: int = 0
    rhyme_group: str | None = None
    template_id: str = "legacy"
    slot_index: int = 0
```

- [ ] **Step 4: Add three built-in templates with honest provenance**

Use nine slots in each initial template so the default fallback corpus can be validated once while timing patterns vary:

```python
def _slots(ticks: tuple[int, ...], *, stresses: tuple[float, ...]) -> tuple[FlowSlot, ...]:
    if len(ticks) != len(stresses):
        raise ValueError("tick and stress arrays must have equal length")
    last = len(ticks) - 1
    return tuple(
        FlowSlot(
            tick_in_bar=tick,
            duration_ticks=max(1, (ticks[index + 1] - tick) if index < last else 16 - tick),
            target_stress=stresses[index],
            boundary_strength=3 if index == last else 0,
            rhyme_group="A" if index == last else None,
        )
        for index, tick in enumerate(ticks)
    )


BUILTIN_TEMPLATES = TemplateCatalog.from_templates(
    (
        FlowTemplate(
            template_id="baseline_straight_9",
            name="Straight nine-slot baseline",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=_slots((0, 2, 4, 6, 8, 10, 12, 14, 15), stresses=(1, 0, .7, 0, 1, 0, .7, 0, .9)),
            provenance=FlowProvenance(kind="hand_authored_mcflow_inspired", source="StreamMUSE baseline"),
        ),
        FlowTemplate(
            template_id="baseline_syncopated_9",
            name="Syncopated nine-slot baseline",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=_slots((0, 2, 3, 5, 7, 8, 10, 13, 15), stresses=(1, .2, .7, .2, .6, 1, .2, .7, .9)),
            provenance=FlowProvenance(kind="hand_authored_mcflow_inspired", source="StreamMUSE baseline"),
        ),
        FlowTemplate(
            template_id="baseline_staggered_9",
            name="Staggered nine-slot baseline",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=_slots((0, 1, 4, 6, 7, 9, 11, 12, 14), stresses=(1, .2, .8, .2, .6, .9, .2, .7, .9)),
            provenance=FlowProvenance(kind="hand_authored_mcflow_inspired", source="StreamMUSE baseline"),
        ),
    )
)
```

- [ ] **Step 5: Add and validate the default looping scenario**

The default scenario uses prescheduled single-syllable topic anchors and nine-syllable fallback lines:

```python
def default_scenario() -> RapScenario:
    return RapScenario(
        scenario_id="default_research_demo",
        tempo_bpm=92.0,
        loop=True,
        segments=(
            ScenarioSegment(0, 4, "space", "baseline_syncopated_9", ("space dreams rise while bright stars cross dark night",)),
            ScenarioSegment(4, 4, "deep sea", "baseline_straight_9", ("deep sea winds move while moon lights guide ships",)),
            ScenarioSegment(8, 4, "code", "baseline_staggered_9", ("code sparks grow as quick hands shape new sound",)),
        ),
    )
```

Validate all segment ranges, template references, nonempty fallback lines, and loop length during dependency assembly rather than failing during playback.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/unit/domain/rap/test_flow.py tests/unit/domain/rap/test_scenario.py tests/unit/infrastructure/rap/test_templates.py -v`

Expected: all pass.

```bash
git add src/streammuse/domain/rap src/streammuse/infrastructure/rap/templates.py src/streammuse/infrastructure/rap/scenarios.py tests/unit/domain/rap tests/unit/infrastructure/rap/test_templates.py
git commit -m "feat(rap): add flow templates and scenarios"
```

### Task 2: Pronunciation-Backed Prosody With Measured Fallback

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/streammuse/domain/rap/models.py`
- Create: `src/streammuse/infrastructure/rap/prosody.py`
- Modify: `src/streammuse/domain/rap/prosody.py`
- Modify: `src/streammuse/domain/rap/__init__.py`
- Test: `tests/unit/domain/rap/test_prosody.py`
- Test: `tests/unit/infrastructure/rap/test_prosody.py`

**Interfaces:**
- Produces: `ProsodyAnalyzer.analyze(text) -> ProsodyAnalysis`, `CmuProsodyAnalyzer`, and `HeuristicProsodyAnalyzer`.
- Consumes: `Syllable` and normalized English text.

- [ ] **Step 1: Write failing dictionary, stress, rhyme-tail, and OOV tests**

```python
def test_cmu_analyzer_exposes_stress_and_rhyme_tail() -> None:
    result = CmuProsodyAnalyzer().analyze("moving night")

    assert [syllable.word for syllable in result.syllables] == ["moving", "moving", "night"]
    assert [syllable.stress for syllable in result.syllables] == [1, 0, 1]
    assert result.end_rhyme_tail[-1].startswith("T")
    assert result.heuristic_words == ()


def test_oov_word_is_retained_and_identified_as_heuristic() -> None:
    result = CmuProsodyAnalyzer().analyze("xyzzy beat")

    assert "xyzzy" in result.oov_words
    assert "xyzzy" in result.heuristic_words
    assert any(syllable.word == "xyzzy" for syllable in result.syllables)
```

- [ ] **Step 2: Add the declared CMU dependency**

Add `"pronouncing>=0.2.0"` to `[project].dependencies`, then run `uv sync` during execution. Do not download a neural phonemizer model for this baseline.

- [ ] **Step 3: Expand the immutable analysis model**

```python
@dataclass(frozen=True)
class Syllable:
    word: str
    index_in_word: int
    syllable_count: int
    stress: int
    phonemes: tuple[str, ...] = ()
    analysis_source: str = "heuristic"

    @property
    def stressed(self) -> bool:
        return self.stress > 0

    @property
    def is_word_end(self) -> bool:
        return self.index_in_word == self.syllable_count - 1


@dataclass(frozen=True)
class ProsodyAnalysis:
    text: str
    normalized_text: str
    syllables: tuple[Syllable, ...]
    end_rhyme_tail: tuple[str, ...]
    oov_words: tuple[str, ...]
    heuristic_words: tuple[str, ...]
    punctuation_boundary_after: tuple[int, ...]
```

Declare the replaceable analyzer boundary in `application/rap/service.py`:

```python
class ProsodyAnalyzer(Protocol):
    def analyze(self, text: str) -> ProsodyAnalysis:
        """Return syllable, stress, rhyme-tail, boundary, and fallback diagnostics."""
```

- [ ] **Step 4: Implement deterministic CMU selection and fallback**

```python
class CmuProsodyAnalyzer:
    def __init__(self, fallback: ProsodyAnalyzer | None = None) -> None:
        self._fallback = fallback or HeuristicProsodyAnalyzer()

    def analyze(self, text: str) -> ProsodyAnalysis:
        words = extract_words(text)
        syllables: list[Syllable] = []
        oov: list[str] = []
        heuristic: list[str] = []
        final_phones: tuple[str, ...] = ()
        for word in words:
            pronunciations = pronouncing.phones_for_word(word)
            if not pronunciations:
                fallback = self._fallback.analyze(word)
                syllables.extend(fallback.syllables)
                oov.append(word)
                heuristic.append(word)
                final_phones = ()
                continue
            phones = tuple(pronunciations[0].split())
            word_syllables = split_arpabet_syllables(word, phones)
            syllables.extend(word_syllables)
            final_phones = phones
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalize_text(text),
            syllables=tuple(syllables),
            end_rhyme_tail=rhyme_tail_from_last_stressed_vowel(final_phones),
            oov_words=tuple(oov),
            heuristic_words=tuple(heuristic),
            punctuation_boundary_after=punctuation_boundaries(text, tuple(syllables)),
        )
```

Use one vowel nucleus per CMU syllable and retain ARPABET stress digits for auditability:

```python
def split_arpabet_syllables(word: str, phones: tuple[str, ...]) -> tuple[Syllable, ...]:
    nuclei = [index for index, phone in enumerate(phones) if phone[-1:].isdigit()]
    if not nuclei:
        return HeuristicProsodyAnalyzer().analyze(word).syllables
    items: list[Syllable] = []
    start = 0
    for syllable_index, nucleus in enumerate(nuclei):
        end = nuclei[syllable_index + 1] if syllable_index + 1 < len(nuclei) else len(phones)
        chunk = phones[start:end]
        items.append(
            Syllable(
                word=word,
                index_in_word=syllable_index,
                syllable_count=len(nuclei),
                stress=int(phones[nucleus][-1]),
                phonemes=tuple(chunk),
                analysis_source="cmudict_first_pronunciation",
            )
        )
        start = end
    return tuple(items)


def rhyme_tail_from_last_stressed_vowel(phones: tuple[str, ...]) -> tuple[str, ...]:
    stressed = [index for index, phone in enumerate(phones) if phone.endswith(("1", "2"))]
    vowels = [index for index, phone in enumerate(phones) if phone[-1:].isdigit()]
    start = stressed[-1] if stressed else vowels[-1] if vowels else len(phones)
    return phones[start:]
```

`HeuristicProsodyAnalyzer` wraps the current vowel-group algorithm, sets stress `1` only on the first estimated syllable of each word, leaves phonemes empty, and marks `analysis_source="vowel_group_heuristic"`. `punctuation_boundaries` scans comma, semicolon, colon, dash, question mark, exclamation mark, and period tokens and maps each to the preceding syllable index.

The first CMU pronunciation is chosen deterministically. Document and log that alternate pronunciations, dialect, elision, and pronunciation bending are not modeled.

- [ ] **Step 5: Run analyzer tests and preserve the baseline**

Run: `uv run pytest tests/unit/domain/rap/test_prosody.py tests/unit/infrastructure/rap/test_prosody.py -v`

Expected: dictionary words expose ARPABET stress, OOV words remain schedulable, and all fallback words are listed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/streammuse/domain/rap src/streammuse/infrastructure/rap/prosody.py tests/unit/domain/rap/test_prosody.py tests/unit/infrastructure/rap/test_prosody.py
git commit -m "feat(rap): add observable CMU prosody analysis"
```

### Task 3: Exact Alignment and Transparent Candidate Ranking

**Files:**
- Create: `src/streammuse/domain/rap/evaluation.py`
- Create: `src/streammuse/application/rap/scoring.py`
- Modify: `src/streammuse/application/rap/alignment.py`
- Modify: `src/streammuse/application/rap/__init__.py`
- Modify: `src/streammuse/domain/rap/__init__.py`
- Test: `tests/unit/application/rap/test_scoring.py`
- Modify: `tests/unit/application/rap/test_alignment.py`

**Interfaces:**
- Produces: `ScoreWeights`, `ScoreComponent`, `CandidateEvaluation`, `SelectionResult`, `evaluate_candidate(...)`, and `rank_candidates(...)`.
- Consumes: `FlowTemplate`, `ProsodyAnalysis`, topic, recent frozen bars, and rhyme anchors.

- [ ] **Step 1: Write failing hard-gate and component-score tests**

```python
def test_exact_syllable_count_is_a_hard_gate() -> None:
    result = evaluate_candidate(
        candidate_id="c1",
        text="one two",
        analysis=analysis_with_syllables("one", "two"),
        template=template_with_slots(3),
        topic="space",
        history=(),
        rhyme_anchors={},
        weights=ScoreWeights(),
    )

    assert result.valid is False
    assert result.rejection_reasons == ("syllable_count:2!=3",)
    assert result.total_score is None


def test_score_components_are_reconstructable_from_logged_values() -> None:
    result = evaluate_candidate(
        candidate_id="c1",
        text="space lights move",
        analysis=analysis_with_stress((1, 0, 1), rhyme_tail=("AY1", "T")),
        template=template_with_stress((1.0, 0.0, 1.0), rhyme_group="A"),
        topic="space",
        history=(),
        rhyme_anchors={},
        weights=ScoreWeights(),
    )

    assert result.valid is True
    assert result.component("stress_alignment").value == 1.0
    assert result.component("topic_coverage").value == 1.0
    assert result.total_score == pytest.approx(sum(item.contribution for item in result.components))
```

- [ ] **Step 2: Run tests and confirm the evaluator is absent**

Run: `uv run pytest tests/unit/application/rap/test_scoring.py -v`

Expected: import failure for `streammuse.application.rap.scoring`.

- [ ] **Step 3: Define explicit score records and default weights**

```python
@dataclass(frozen=True)
class ScoreWeights:
    stress_alignment: float = 0.30
    boundary_fit: float = 0.10
    rhyme_quality: float = 0.20
    topic_coverage: float = 0.20
    lexical_continuity: float = 0.15
    novelty: float = 0.05

    def __post_init__(self) -> None:
        if abs(sum(asdict(self).values()) - 1.0) > 1e-9:
            raise ValueError("score weights must sum to one")


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    value: float
    weight: float
    contribution: float
    method: str


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    text: str
    analysis: ProsodyAnalysis
    valid: bool
    rejection_reasons: tuple[str, ...]
    components: tuple[ScoreComponent, ...]
    total_score: float | None
    scheduled: tuple[ScheduledSyllable, ...]

    def component(self, name: str) -> ScoreComponent:
        return next(item for item in self.components if item.name == name)


@dataclass(frozen=True)
class SelectionResult:
    evaluations: tuple[CandidateEvaluation, ...]
    selected: CandidateEvaluation | None
    threshold: float
    fallback_reason: str | None
```

- [ ] **Step 4: Implement exact sequential alignment**

```python
def align_exact(analysis: ProsodyAnalysis, slots: Sequence[BeatSlot]) -> tuple[ScheduledSyllable, ...]:
    if len(analysis.syllables) != len(slots):
        raise ValueError(f"exact alignment requires {len(slots)} syllables, got {len(analysis.syllables)}")
    return tuple(
        ScheduledSyllable(slot=slot, syllable=syllable)
        for syllable, slot in zip(analysis.syllables, slots, strict=True)
    )
```

Retain the existing dynamic-programming function under the explicit name `align_legacy_flexible`. Tests must distinguish it from the research default so old results remain reproducible.

- [ ] **Step 5: Implement documented feature formulas**

Use these exact baseline definitions:

```python
def stress_alignment(analysis: ProsodyAnalysis, template: FlowTemplate) -> float:
    lexical = [1.0 if item.stress == 1 else 0.5 if item.stress == 2 else 0.0 for item in analysis.syllables]
    errors = [abs(actual - slot.target_stress) * (1.0 + slot.target_stress) for actual, slot in zip(lexical, template.slots, strict=True)]
    denominator = sum(1.0 + slot.target_stress for slot in template.slots)
    return 1.0 - sum(errors) / denominator


def boundary_fit(analysis: ProsodyAnalysis, template: FlowTemplate) -> float:
    targets = [index for index, slot in enumerate(template.slots) if slot.boundary_strength > 0]
    if not targets:
        return 1.0
    word_ends = {index for index, syllable in enumerate(analysis.syllables) if syllable.is_word_end}
    punctuation = set(analysis.punctuation_boundary_after)
    return sum(1.0 if index in punctuation else 0.6 if index in word_ends else 0.0 for index in targets) / len(targets)


def rhyme_quality(tail: tuple[str, ...], anchor: tuple[str, ...] | None) -> float:
    if anchor is None:
        return 0.5
    if tail == anchor and tail:
        return 1.0
    tail_vowels = tuple(re.sub(r"\d$", "", phone) for phone in tail if phone[-1:].isdigit())
    anchor_vowels = tuple(re.sub(r"\d$", "", phone) for phone in anchor if phone[-1:].isdigit())
    return 0.6 if tail_vowels and tail_vowels == anchor_vowels else 0.0
```

`topic_coverage` is the fraction of normalized non-stopword topic tokens present in the candidate. `lexical_continuity` is `min(1.0, shared_non_topic_content_tokens / 2)` against the previous two frozen bars, or `0.5` when there is no history. `novelty` is `1 - max_bigram_jaccard` against the previous four bars. Exact normalized duplicates are hard-rejected. Rhyme anchors are keyed by `(segment.start_bar, rhyme_group)`, so a topic/segment transition does not inherit an unrelated sound. The first occurrence of a rhyme group receives the documented neutral score `0.5` and establishes its anchor only when the bar freezes.

- [ ] **Step 6: Rank valid candidates with deterministic ties**

Sort by descending total score, then ascending source candidate index. Return every evaluation, not only the winner:

```python
valid = [item for item in evaluations if item.valid and item.total_score is not None]
selected = max(valid, key=lambda item: (item.total_score, -candidate_order[item.candidate_id])) if valid else None
return SelectionResult(evaluations=tuple(evaluations), selected=selected, threshold=minimum_score)
```

- [ ] **Step 7: Run focused tests and commit**

Run: `uv run pytest tests/unit/application/rap/test_alignment.py tests/unit/application/rap/test_scoring.py -v`

Expected: exact alignment, all score components, rejection reasons, thresholds, and deterministic ties pass.

```bash
git add src/streammuse/domain/rap/evaluation.py src/streammuse/application/rap/scoring.py src/streammuse/application/rap/alignment.py src/streammuse/application/rap/__init__.py src/streammuse/domain/rap/__init__.py tests/unit/application/rap
git commit -m "feat(rap): add transparent candidate ranking"
```

### Task 4: Request-Aware Generation and Prevalidated Fallback

**Files:**
- Modify: `src/streammuse/domain/rap/models.py`
- Modify: `src/streammuse/application/rap/service.py`
- Modify: `src/streammuse/infrastructure/rap/generators.py`
- Create: `src/streammuse/infrastructure/rap/fallback.py`
- Modify: `tests/unit/infrastructure/rap/test_generators.py`
- Create: `tests/unit/infrastructure/rap/test_fallback.py`

**Interfaces:**
- Produces: `CandidateRequest`, enriched `CandidateBatch`, `CandidateGenerator.generate(request)`, and `PrevalidatedFallbackCatalog.line_for(bar_context)`.
- Consumes: scenario segment, flow template, recent frozen lines, analyzer, and local chat client.

- [ ] **Step 1: Write failing prompt, raw-response, error, and fallback tests**

```python
def test_local_chat_request_contains_exact_structure_and_history() -> None:
    request = CandidateRequest(
        request_id="bar-4-request-1",
        target_bar=4,
        topic="space",
        template_id="baseline_syncopated_9",
        required_syllables=9,
        count=8,
        context_lines=("stars cross the night",),
        seed=20260807,
    )
    batch = LocalChatCandidateGenerator(FakeChatClient("first line\nsecond line")).generate(request)

    assert "exactly 9 spoken syllables" in batch.prompt[-1]["content"]
    assert "stars cross the night" in batch.prompt[-1]["content"]
    assert batch.raw_response == "first line\nsecond line"
    assert batch.request_id == request.request_id


def test_generator_failure_is_not_hidden_by_internal_fallback() -> None:
    batch = LocalChatCandidateGenerator(FailingChatClient()).generate(request_for_bar(2))

    assert batch.candidates == ()
    assert batch.error_type == "generation_error"
    assert "connection refused" in batch.error_message
```

- [ ] **Step 2: Replace the positional generator protocol**

```python
class CandidateGenerator(Protocol):
    def generate(self, request: CandidateRequest) -> CandidateBatch:
        """Return raw parsed candidates and complete non-secret request diagnostics."""
```

```python
@dataclass(frozen=True)
class CandidateRequest:
    request_id: str
    target_bar: int
    topic: str
    template_id: str
    required_syllables: int
    count: int
    context_lines: tuple[str, ...]
    seed: int


@dataclass(frozen=True)
class CandidateBatch:
    request_id: str
    candidates: tuple[str, ...]
    source: str
    prompt: tuple[dict[str, str], ...]
    raw_response: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    warning: str | None = None
    error_type: str | None = None
    error_message: str | None = None
```

- [ ] **Step 3: Make generation diagnostics lossless**

Do not syllable-filter in `_parse_candidate_lines`; only normalize list labels, remove empty lines, and deduplicate while preserving order. Copy `ChatModelResponse.text`, latency, token counts, and serialized raw response into `CandidateBatch`. Convert exceptions into an empty explicit error batch so the controller can record the failure and apply fallback.

- [ ] **Step 4: Build and validate fallback catalog at startup**

```python
class PrevalidatedFallbackCatalog:
    @classmethod
    def build(cls, scenario: RapScenario, templates: TemplateCatalog, analyzer: ProsodyAnalyzer) -> "PrevalidatedFallbackCatalog":
        lines: dict[tuple[str, str], tuple[ProsodyAnalysis, ...]] = {}
        for segment in scenario.segments:
            template = templates.get(segment.template_id)
            analyses = tuple(analyzer.analyze(line) for line in segment.fallback_lines)
            invalid = [item.text for item in analyses if len(item.syllables) != len(template.slots)]
            if invalid:
                raise ValueError(f"fallback lines do not match {template.template_id}: {invalid}")
            lines[(normalize_topic(segment.topic), template.template_id)] = analyses
        return cls(lines)
```

Select fallback lines round-robin by absolute bar so repeated scenarios are deterministic. Record the fallback as source `prevalidated_fallback`, never as a generated candidate.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/unit/infrastructure/rap/test_generators.py tests/unit/infrastructure/rap/test_fallback.py -v`

Expected: prompt constraints, raw diagnostics, explicit errors, and startup fallback validation pass.

```bash
git add src/streammuse/domain/rap/models.py src/streammuse/application/rap/service.py src/streammuse/infrastructure/rap/generators.py src/streammuse/infrastructure/rap/fallback.py tests/unit/infrastructure/rap
git commit -m "feat(rap): make generation and fallback observable"
```

### Task 5: Canonical Research Events, Recorder, and Metrics

**Files:**
- Create: `src/streammuse/domain/rap/events.py`
- Create: `src/streammuse/application/rap/monitoring.py`
- Create: `src/streammuse/infrastructure/rap/recorder.py`
- Test: `tests/unit/application/rap/test_monitoring.py`
- Test: `tests/unit/infrastructure/rap/test_recorder.py`

**Interfaces:**
- Produces: `RapEventType`, `RapEvent`, `RapEventPublisher`, `RapEventDispatcher`, `RapStateProjector`, `RapSessionRecorder`, `derive_summary(events)`, and `derive_bar_rows(events)`.
- Consumes: serializable domain records, injected wall/monotonic clocks, session manifest, and event sinks.

- [ ] **Step 1: Write failing sequence, snapshot, artifact, and derivation tests**

```python
def test_publisher_assigns_one_monotonic_sequence_to_all_sinks() -> None:
    left: list[RapEvent] = []
    right: list[RapEvent] = []
    publisher = RapEventPublisher("session-1", utc_now=fixed_utc, monotonic_ns=fake_ns)
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(left.append, right.append))
    dispatcher.start()

    publisher.emit(RapEventType.BAR_PLANNING_STARTED, bar=1, tick=0, payload={"request_id": "r1"})
    publisher.emit(RapEventType.CANDIDATE_BATCH_RECEIVED, bar=1, tick=1, payload={"request_id": "r1"})
    dispatcher.flush_and_close()

    assert [event.sequence for event in left] == [1, 2]
    assert left == right


def test_summary_is_derived_from_canonical_events(tmp_path: Path) -> None:
    recorder = RapSessionRecorder(tmp_path, manifest={"scenario_id": "test"})
    for event in scripted_session_events():
        recorder(event)
    recorder.close()

    assert (tmp_path / "session.json").exists()
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == len(scripted_session_events())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["bars"]["frozen"] == 2
    assert summary["bars"]["fallback_rate"] == 0.5
```

- [ ] **Step 2: Define the complete event vocabulary**

```python
class RapEventType(str, Enum):
    SESSION_STARTED = "session_started"
    SESSION_STOPPED = "session_stopped"
    BAR_RESERVED = "bar_reserved"
    BAR_PLANNING_STARTED = "bar_planning_started"
    CANDIDATE_BATCH_RECEIVED = "candidate_batch_received"
    CANDIDATE_EVALUATED = "candidate_evaluated"
    GENERATION_FAILED = "generation_failed"
    BAR_REPLACED = "bar_replaced"
    BAR_FROZEN = "bar_frozen"
    FALLBACK_ACTIVATED = "fallback_activated"
    TICK = "tick"
    SYLLABLE_EMITTED = "syllable_emitted"
    PRESENTATION_ERROR = "presentation_error"


@dataclass(frozen=True)
class RapEvent:
    session_id: str
    sequence: int
    event_type: RapEventType
    utc_time: str
    monotonic_ns: int
    bar: int | None
    tick: int | None
    request_id: str | None
    payload: dict[str, Any]
```

- [ ] **Step 3: Implement off-clock ordered dispatch and state projection**

`RapEventPublisher.emit` assigns sequence/time once under a lock and performs only `SimpleQueue.put(event)` afterward. `RapEventDispatcher` owns one worker thread that drains this unbounded queue and invokes recorder, state projector, terminal, and WebSocket queue sinks in sequence order. This keeps disk and presentation I/O off the musical clock. `flush_and_close()` places a sentinel, waits for all earlier events to dispatch, and then returns. A sink that raises is disabled for the remainder of the session; the dispatcher publishes one `presentation_error` for the remaining sinks without invoking the failed sink recursively.

`RapStateProjector.apply` stores current tick, current segment, pending request, candidate table, frozen bars, emitted syllables, latency aggregates, and fallback counters. `snapshot()` returns a deep serializable copy under its own lock. The browser may lag the clock by one dispatcher interval, but it can never alter runtime state.

- [ ] **Step 4: Implement crash-tolerant session artifacts**

```python
class RapSessionRecorder:
    def __init__(self, session_dir: Path, manifest: dict[str, Any]) -> None:
        session_dir.mkdir(parents=True, exist_ok=False)
        self._events_path = session_dir / "events.jsonl"
        (session_dir / "session.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        self._stream = self._events_path.open("a", encoding="utf-8", buffering=1)

    def __call__(self, event: RapEvent) -> None:
        self._stream.write(json.dumps(event_to_dict(event), sort_keys=True) + "\n")

    def close(self) -> None:
        self._stream.close()
        events = read_events(self._events_path)
        write_json(self._events_path.parent / "summary.json", derive_summary(events))
        write_bar_csv(self._events_path.parent / "bars.csv", derive_bar_rows(events))
```

The manifest must include scenario, seed, tempo, template definitions and provenance, generator/model config, score weights, minimum score, timeout, lookahead, Python/platform versions, package version, git revision, and dirty-tree boolean. Redact keys whose normalized names contain `key`, `token`, `secret`, or `authorization`.

- [ ] **Step 5: Implement exact metric definitions**

Derive candidate validity, fallback, deadline miss, generator error, pronunciation fallback, and repetition rates as ratios with explicit numerator and denominator fields. Derive p50/p95/max for generation latency, deadline slack, and emission jitter. Store `null` when a metric has no observations; do not report zero for missing data.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/unit/application/rap/test_monitoring.py tests/unit/infrastructure/rap/test_recorder.py -v`

Expected: ordered fan-out, snapshot projection, redaction, JSONL recovery, summary, and CSV tests pass.

```bash
git add src/streammuse/domain/rap/events.py src/streammuse/application/rap/monitoring.py src/streammuse/infrastructure/rap/recorder.py tests/unit/application/rap/test_monitoring.py tests/unit/infrastructure/rap/test_recorder.py
git commit -m "feat(rap): record canonical research event stream"
```

### Task 6: Rolling Planner, Deadlines, Freeze, and No-Gap Fallback

**Files:**
- Modify: `src/streammuse/application/rap/realtime.py`
- Modify: `tests/unit/application/rap/test_realtime.py`
- Modify: `tests/unit/application/rap/test_service.py`

**Interfaces:**
- Produces: scenario-aware `RollingRapController.start()`, `on_tick(tick)`, `close()`, `bar_for(index)`, `bar_state(index)`, and read-only `scenario` property.
- Consumes: `Tempo`, `RapScenario`, `TemplateCatalog`, `CandidateGenerator`, `PrevalidatedFallbackCatalog`, `ProsodyAnalyzer`, `ScoreWeights`, and `RapEventPublisher`.

- [ ] **Step 1: Rewrite tests around observable bar state**

Add tests for these independent invariants:

```python
def test_start_reserves_fallback_through_lookahead() -> None:
    controller, events = build_controller(primary=None, lookahead_bars=2)
    controller.start()
    assert controller.bar_state(0) == "reserved"
    assert controller.bar_state(1) == "reserved"
    assert event_types(events).count("bar_reserved") == 2


def test_valid_primary_replaces_only_unfrozen_reservation() -> None:
    controller, events, executor = build_manual_controller(valid_primary_batch())
    controller.start()
    executor.complete()
    controller.on_tick(0)
    assert controller.bar_for(1).source == "local_chat"
    assert "bar_replaced" in event_types(events)


def test_late_result_is_logged_and_cannot_replace_frozen_bar() -> None:
    controller, events, executor = build_manual_controller(valid_primary_batch(target_bar=1))
    controller.start()
    controller.on_tick(16)
    executor.complete()
    controller.on_tick(17)
    assert controller.bar_for(1).source == "prevalidated_fallback"
    assert payload_for(events, "candidate_batch_received")["late"] is True


def test_invalid_batch_freezes_fallback_with_reason() -> None:
    controller, events, executor = build_manual_controller(all_invalid_batch())
    controller.start()
    executor.complete()
    controller.on_tick(0)
    controller.on_tick(16)
    frozen = payload_for_bar(events, "bar_frozen", 1)
    assert frozen["fallback"] is True
    assert frozen["fallback_reason"] == "no_valid_candidate"
```

- [ ] **Step 2: Replace hidden line dictionaries with explicit bar records**

```python
@dataclass
class _BarState:
    bar: int
    segment: ScenarioSegment
    template: FlowTemplate
    scheduled: tuple[ScheduledSyllable, ...]
    text: str
    source: str
    fallback_reason: str | None
    request_id: str | None = None
    frozen: bool = False
```

- [ ] **Step 3: Reserve validated fallback synchronously**

At `start`, reserve bars `[0, lookahead_bars - 1]`, emit `bar_reserved`, submit generation only for bar 1, and leave bar 0 as guaranteed fallback. On every tick, ensure reservations through `current_bar + lookahead_bars - 1` before processing emission.

- [ ] **Step 4: Keep all model work off the tick path**

Submit at most one request to the one-worker executor. `on_tick` may only call `future.done()` and `future.result()` after completion. Measure `on_tick` in the slow-generator test and require less than 100 ms under local test conditions, preserving the current nonblocking contract.

- [ ] **Step 5: Evaluate completed batches and record every candidate**

For each parsed candidate, emit one `candidate_evaluated` event containing normalized text, syllables, per-word analysis source, OOV words, rejection reasons, component values/weights/contributions, total, and selected boolean. Emit the full prompt/raw batch once in `candidate_batch_received` rather than duplicating it per candidate.

- [ ] **Step 6: Freeze before first emission and calculate jitter**

When `tick % ticks_per_bar == 0`, freeze the current bar before finding scheduled syllables. Emit one `bar_frozen`; if source is fallback, emit one `fallback_activated` with the final reason. For each syllable emit planned monotonic time, actual monotonic time, and `jitter_ms`. Enforce with an assertion that no `syllable_emitted` event references an unfrozen bar.

- [ ] **Step 7: Run controller tests and commit**

Run: `uv run pytest tests/unit/application/rap/test_realtime.py tests/unit/application/rap/test_service.py -v`

Expected: reserve, replacement, threshold, late result, failure, freeze, no-blocking, topic transition, and no-gap tests pass.

```bash
git add src/streammuse/application/rap/realtime.py tests/unit/application/rap/test_realtime.py tests/unit/application/rap/test_service.py
git commit -m "feat(rap): add observable rolling bar planner"
```

### Task 7: Standalone Research Runtime and Terminal Monitor

**Files:**
- Create: `src/streammuse/application/rap/runtime.py`
- Create: `src/streammuse/presentation/rap_demo/__init__.py`
- Create: `src/streammuse/presentation/rap_demo/cli.py`
- Create: `src/streammuse/presentation/rap_demo/terminal.py`
- Modify: `pyproject.toml`
- Test: `tests/unit/application/rap/test_runtime.py`
- Test: `tests/unit/presentation/rap_demo/test_cli.py`
- Test: `tests/unit/presentation/rap_demo/test_terminal.py`

**Interfaces:**
- Produces: `RapTickLoop`, `RapDemoDependencies`, `build_demo(args)`, `TerminalRapSink`, and `streammuse-rap-demo`.
- Consumes: controller, publisher, recorder, scenario, templates, generator, analyzer, and `Tempo`.

- [ ] **Step 1: Write fake-clock runtime tests**

```python
def test_tick_loop_uses_absolute_monotonic_deadlines_without_drift() -> None:
    clock = FakeClock()
    ticks: list[int] = []
    loop = RapTickLoop(Tempo(120.0, 4, 4), on_tick=ticks.append, clock=clock, sleep=clock.sleep)

    loop.run(max_ticks=5)

    assert ticks == [0, 1, 2, 3, 4]
    assert clock.sleeps == pytest.approx([0.125, 0.125, 0.125, 0.125])


def test_terminal_detail_full_prints_candidate_components(capsys) -> None:
    sink = TerminalRapSink(detail="full")
    sink(candidate_evaluated_event())
    output = capsys.readouterr().out
    assert "stress_alignment" in output
    assert "rejection_reasons" in output
```

- [ ] **Step 2: Implement drift-resistant tick loop**

```python
def run(self, max_ticks: int | None = None) -> None:
    start = self._clock()
    tick = 0
    while not self._stop.is_set() and (max_ticks is None or tick < max_ticks):
        target = start + self._tempo.tick_to_seconds(tick)
        remaining = target - self._clock()
        if remaining > 0:
            self._sleep(remaining)
        self._on_tick(tick)
        tick += 1
```

- [ ] **Step 3: Add explicit CLI configuration**

```text
streammuse-rap-demo
  --scenario PATH
  --generator phrase_bank|local_chat|scripted_failure
  --model-url http://localhost:8000/v1
  --model local-model
  --candidate-count 8
  --lookahead-bars 2
  --minimum-score 0.55
  --seed 20260807
  --max-bars 12
  --log-dir logs/rap
  --terminal-detail summary|candidates|full
  --host 127.0.0.1
  --port 8012
  --no-web
```

`--max-bars 0` means continue until interrupted. CLI arguments override scenario defaults and the resolved configuration is written to `session.json`.

- [ ] **Step 4: Assemble sinks and lifecycle once**

`build_demo` creates one session ID and directory, then constructs recorder, state projector, terminal sink, WebSocket queue sink, publisher, dispatcher, controller, and tick loop. Shutdown stops the tick loop and controller first, publishes `session_stopped`, drains and closes the dispatcher, then finalizes the recorder. A generator/client close failure is an event and does not prevent recorder finalization.

- [ ] **Step 5: Register the entry point**

```toml
[project.scripts]
streammuse-rap-demo = "streammuse.presentation.rap_demo.cli:main"
```

- [ ] **Step 6: Run tests and a terminal-only smoke session**

Run: `uv run pytest tests/unit/application/rap/test_runtime.py tests/unit/presentation/rap_demo -v`

Run: `uv run streammuse-rap-demo --generator scripted_failure --max-bars 3 --terminal-detail full --no-web`

Expected: three bars emit at tempo, every bar uses fallback with `generation_error`, and the session directory contains valid `session.json`, `events.jsonl`, `summary.json`, and `bars.csv`.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/streammuse/application/rap/runtime.py src/streammuse/presentation/rap_demo tests/unit/application/rap/test_runtime.py tests/unit/presentation/rap_demo
git commit -m "feat(rap): add standalone research runtime"
```

### Task 8: Live Read-Only Web Monitor

**Files:**
- Create: `src/streammuse/presentation/rap_demo/server.py`
- Create: `src/streammuse/presentation/rap_demo/static/index.html`
- Create: `src/streammuse/presentation/rap_demo/static/css/rap-demo.css`
- Create: `src/streammuse/presentation/rap_demo/static/js/rap-demo.js`
- Test: `tests/unit/presentation/rap_demo/test_server.py`

**Interfaces:**
- Produces: `create_app(runtime, projector, websocket_queue)`, `GET /api/state`, `GET /api/session`, and `WS /ws`.
- Consumes: the state projector snapshot and serialized canonical events.

- [ ] **Step 1: Write HTTP and WebSocket contract tests**

```python
def test_state_endpoint_returns_complete_monitor_snapshot() -> None:
    app = create_app(runtime=FakeRuntime(), projector=projector_with_one_bar(), websocket_queue=Queue())
    response = TestClient(app).get("/api/state")
    assert response.status_code == 200
    assert response.json()["current"]["bar"] == 1
    assert response.json()["candidates"][0]["components"][0]["name"] == "stress_alignment"


def test_websocket_sends_snapshot_before_live_events() -> None:
    client = TestClient(create_app(runtime=FakeRuntime(), projector=projector_with_one_bar(), websocket_queue=Queue()))
    with client.websocket_connect("/ws") as websocket:
        assert websocket.receive_json()["type"] == "snapshot"
```

- [ ] **Step 2: Implement process lifecycle and broadcast isolation**

Start the tick-loop thread in FastAPI lifespan after all routes and sinks exist. The broadcaster drains a thread-safe queue every 20 ms and removes disconnected clients. Runtime, recorder, and generator shutdown happen once in lifespan finalization. WebSocket send errors never propagate to the controller.

- [ ] **Step 3: Build a dense research-oriented observer**

The first viewport contains:

- session state, tempo, tick, bar, topic, template, and generator status;
- one horizontal 16-subdivision bar with flow slots, target stress, active tick, and emitted syllables;
- selected line, source, fallback reason, total score, generation latency, and deadline slack;
- total/valid candidates, fallback rate, deadline miss rate, pronunciation fallback rate, and jitter p95.

Below it, show a sortable candidate table with raw text, syllable count, validity, rejection reasons, each component, total, OOV words, and selected marker; frozen-bar history; and a chronological event console. Use compact tables and unframed sections, not nested cards. Use the existing frontend icon library only if already available; do not add a JavaScript framework.

- [ ] **Step 4: Implement event-driven browser state**

```javascript
const state = { snapshot: null, events: [] };

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.type === "snapshot") state.snapshot = message.payload;
    else state.events.push(message.payload);
    render(state);
  };
  socket.onclose = () => window.setTimeout(connect, 1000);
}
```

Render text with `textContent`, never `innerHTML`, because model output is untrusted.

- [ ] **Step 5: Verify desktop and mobile behavior**

Start: `uv run streammuse-rap-demo --generator phrase_bank --max-bars 0 --port 8012`

Inspect at 1440×900 and 390×844. Verify the flow timeline is nonblank, the active tick moves, no text overlaps, long candidates wrap inside table cells, horizontal table overflow remains scrollable, reconnect restores a snapshot, and forced fallback is visibly distinguished without relying on color alone.

- [ ] **Step 6: Run tests and commit**

Run: `uv run pytest tests/unit/presentation/rap_demo/test_server.py -v`

Expected: static routes, snapshot, ordered WebSocket delivery, disconnect handling, and lifespan shutdown pass.

```bash
git add src/streammuse/presentation/rap_demo tests/unit/presentation/rap_demo/test_server.py
git commit -m "feat(rap): add live research monitor"
```

### Task 9: Preserve Active StreamMUSE Integration

**Files:**
- Modify: `src/streammuse/presentation/cli/cli.py`
- Modify: `src/streammuse/application/rap/rhythm.py`
- Modify: `tests/integration/test_cli_entry_point.py`
- Modify: `tests/unit/presentation/test_cli_config_parser.py`
- Modify: `tests/unit/application/rap/test_rhythm.py`

**Interfaces:**
- Produces: a compatibility assembly path from current `RapConfig(topic, pattern, generator, ...)` to the new controller.
- Consumes: the existing `RuntimeSessionBuilder.tick_observer_factory` and `RealTimeMusicService.TickObserver` lifecycle.

- [ ] **Step 1: Add a regression test for the existing CLI contract**

```python
def test_existing_streammuse_cli_builds_new_rap_controller_without_changing_tick_service() -> None:
    config = ApplicationConfig(
        tempo=TempoConfig(bpm=92.0),
        rap=RapConfig(topic="space", pattern="boom_bap", generator="phrase_bank", lookahead_bars=2),
    )
    controller = _build_rap_controller(config, Tempo(92.0, 4, 4))

    assert isinstance(controller, RollingRapController)
    assert controller.scenario.segment_for_bar(0).topic == "space"
```

- [ ] **Step 2: Map old pattern names explicitly**

```python
LEGACY_PATTERN_TEMPLATE_IDS = {
    "boom_bap": "baseline_syncopated_9",
    "straight_8": "baseline_straight_9",
    "trap_sparse": "baseline_staggered_9",
}
```

Record `compatibility_mapping` in event metadata so a result never appears to come from an extracted MCFlow template.

- [ ] **Step 3: Assemble a fixed-topic looping scenario in `_build_rap_controller`**

Use current `RapConfig` values for topic, generator, model, timeout, candidate count, and lookahead. Keep terminal syllable output compatible with `[RAP Bn beat.tick]`. Do not change `RealTimeMusicService`, `TickObserver`, output sink protocols, or active WebSocket message formats.

- [ ] **Step 4: Run focused and full rap integration tests**

Run: `uv run pytest tests/unit/application/rap tests/unit/infrastructure/rap tests/unit/presentation/rap tests/unit/presentation/test_cli_config_parser.py tests/integration/test_cli_entry_point.py -q`

Expected: current `streammuse-rap`, optional `streammuse-cli --rap-topic`, and the new controller all pass.

- [ ] **Step 5: Commit**

```bash
git add src/streammuse/presentation/cli/cli.py src/streammuse/application/rap/rhythm.py tests/integration/test_cli_entry_point.py tests/unit/presentation/test_cli_config_parser.py tests/unit/application/rap/test_rhythm.py
git commit -m "refactor(rap): preserve realtime CLI integration"
```

### Task 10: Optional MCFlow Extraction and Reproducible Session Analysis

**Files:**
- Create: `src/streammuse/infrastructure/rap/mcflow.py`
- Create: `scripts/extract_mcflow_templates.py`
- Create: `scripts/summarize_rap_session.py`
- Create: `tests/fixtures/rap/mcflow_minimal.rap`
- Create: `tests/unit/infrastructure/rap/test_mcflow.py`
- Create: `tests/unit/scripts/test_summarize_rap_session.py`
- Create: `docs/developer-guide/research-realtime-rap.md`
- Modify: `docs/developer-guide/rap-alignment-prototype.md`

**Interfaces:**
- Produces: `parse_mcflow_file(path)`, `extract_anonymous_templates(...)`, and deterministic summary regeneration.
- Consumes: user-supplied Humdrum files with `**recip`, `**stress`, `**break`, `**rhyme`, and `**lyrics` spines; canonical session artifacts.

- [ ] **Step 1: Create a synthetic Humdrum fixture and failing parser tests**

The fixture must contain invented syllables and no copied lyrics:

```text
**recip	**stress	**break	**rhyme	**lyrics
=1	=1	=1	=1	=1
8	1	.	.	ta
16	0	.	.	ka
16	1	3	A	boom
=2	=2	=2	=2	=2
*-	*-	*-	*-	*-
```

```python
def test_mcflow_extractor_discards_lyrics_and_preserves_structure(tmp_path: Path) -> None:
    templates = extract_anonymous_templates(FIXTURE, ticks_per_beat=4, max_quantization_error_ticks=0.25)
    assert len(templates) == 1
    assert [slot.tick_in_bar for slot in templates[0].slots] == [0, 2, 3]
    assert templates[0].slots[-1].rhyme_group == "A"
    assert "ta" not in json.dumps(flow_template_to_dict(templates[0]))
```

- [ ] **Step 2: Parse Humdrum spines structurally**

Read tab-separated records, locate required exclusive interpretations by name, skip global/local comments and tandem interpretations, reset measure accumulation at `=` records, and stop at `*-`. Parse reciprocal duration with `fractions.Fraction`; convert whole-note fractions to ticks using `duration * 4 * ticks_per_beat`.

- [ ] **Step 3: Refuse silent timing distortion**

Quantize rational onsets to integer StreamMUSE ticks only when every slot and duration is within `max_quantization_error_ticks`. Store maximum error in `FlowProvenance.quantization_error_ticks`. Reject the measure with an explicit reason when triplets or other values exceed tolerance. This keeps the first runtime honest about its sixteenth-note grid limitation.

- [ ] **Step 4: Emit anonymous hashes and provenance**

The output JSON contains structural slots, source-file SHA-256, measure index, extractor version, and quantization error. It omits lyrics, IPA, artist, song title, source filename, and absolute source path. The command requires `--mcflow-dir` and `--output`; no MCFlow data is downloaded automatically.

- [ ] **Step 5: Implement deterministic session regeneration**

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    events = read_events(Path(args.session_dir) / "events.jsonl")
    write_json(Path(args.session_dir) / "summary.regenerated.json", derive_summary(events))
    write_bar_csv(Path(args.session_dir) / "bars.regenerated.csv", derive_bar_rows(events))
    return 0
```

Test byte-equivalent JSON data and equivalent CSV rows against recorder output, ignoring final newline differences only.

- [ ] **Step 6: Document the experiment protocol and limitations**

The guide must include:

1. How to run phrase-bank, real local-chat, forced-failure, and continuous sessions.
2. How to interpret every score component and metric numerator/denominator.
3. How to inspect live terminal and browser state.
4. How to regenerate summaries and compare sessions with the same scenario/seed.
5. Why fallback rate and latency are system metrics, not lyric-quality metrics.
6. Current limitations: dictionary pronunciation, OOV heuristic, one pronunciation choice, lexical topic/continuity proxies, hand-authored templates, sixteenth-note quantization, no expressive audio, and no human validation.
7. The next research step: collect pairwise human preferences and fit/validate a learned ranker against the logged candidate features.

- [ ] **Step 7: Run parser, analysis, and full regression tests**

Run: `uv run pytest tests/unit/infrastructure/rap/test_mcflow.py tests/unit/scripts/test_summarize_rap_session.py -v`

Run: `uv run pytest tests/ -q --tb=short`

Expected: all tests pass; GPU/model-dependent tests remain skipped under their existing markers.

- [ ] **Step 8: Run the final observable smoke experiment**

Start a local OpenAI-compatible model server, then run:

```bash
uv run streammuse-rap-demo \
  --generator local_chat \
  --candidate-count 8 \
  --lookahead-bars 2 \
  --minimum-score 0.55 \
  --seed 20260807 \
  --max-bars 12 \
  --terminal-detail full \
  --log-dir logs/rap \
  --port 8012
```

Verify twelve frozen bars, at least three topic/template segments, no missing bar, complete candidate diagnostics, moving browser timeline, and summary consistency. It is acceptable if fallback is frequent; report the observed rate and reasons rather than hiding them.

- [ ] **Step 9: Commit**

```bash
git add src/streammuse/infrastructure/rap/mcflow.py scripts/extract_mcflow_templates.py scripts/summarize_rap_session.py tests/fixtures/rap tests/unit/infrastructure/rap/test_mcflow.py tests/unit/scripts/test_summarize_rap_session.py docs/developer-guide/research-realtime-rap.md docs/developer-guide/rap-alignment-prototype.md
git commit -m "feat(rap): add MCFlow extraction and research analysis"
```

---

## Milestones and Review Gates

### Milestone A: Defensible offline selection after Task 4

Given a topic and flow template, the system can show every raw candidate, dictionary/heuristic syllable analysis, hard rejection, component score, and deterministic winner or fallback. This is the first point at which the scoring design can be inspected independently of real-time behavior.

### Milestone B: Continuous terminal experiment after Task 7

The system produces uninterrupted bars on a monotonic clock, records all decisions, and survives forced generator failure. This is the minimum end-to-end research prototype and should be preserved even if browser work is delayed.

### Milestone C: Live monitor after Task 8

The browser shows the same event stream and metrics as the terminal and artifacts. No musical or planning logic lives in JavaScript.

### Milestone D: Active-system and corpus bridge after Tasks 9–10

The improved controller remains usable through `streammuse-cli`, and researchers can derive anonymous templates from a local MCFlow checkout without committing copyrighted lyrics or silently quantizing unsupported rhythm.

## Acceptance Criteria

1. A 12-bar scripted-failure run emits and freezes all 12 bars with no gap.
2. A slow generator never delays `on_tick` by 100 ms or more in the focused local test.
3. Every parsed candidate has either a complete evaluation or an explicit processing error.
4. Every frozen bar references one template, one source, one selection path, and one fallback reason when applicable.
5. No bar changes after `bar_frozen`; no syllable emits before its bar freezes.
6. `summary.json` and `bars.csv` regenerate from `session.json` plus `events.jsonl`.
7. Terminal, browser snapshot, and artifacts agree on current bar, selected line, source, and fallback state.
8. Session artifacts include model identity, prompt, raw response, seed, weights, threshold, template provenance, code revision, and timing metrics without secrets.
9. Built-in templates are labeled as hand-authored; extracted templates include anonymous source hashes and quantization error.
10. Documentation explicitly separates demonstrated engineering behavior from unvalidated lyrical-quality claims.

## Deferred Work

- Audio or TTS realization with duration control and expressive prosody.
- Live drum/MIDI rhythm extraction and variable-tempo tracking.
- Triplet-capable or higher-resolution timing beyond the current 16 subdivisions per bar.
- Dialect-aware grapheme-to-phoneme conversion and controllable pronunciation bending.
- Multi-bar joint generation rather than one-bar generation with recent context.
- Learned ranking weights, LLM-as-judge scoring, or reinforcement from listener choices.
- Formal listener study for flow naturalness, intelligibility, relevance, and overall rap quality.
- Training directly on MCFlow or copyrighted lyric text.

These are deferred because the first experiment must establish a reliable, inspectable symbolic pipeline before adding audio quality, learned judgments, or live-input uncertainty.
