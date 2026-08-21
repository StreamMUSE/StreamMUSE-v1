from __future__ import annotations

import io
import struct
import wave
from collections.abc import Mapping
from pathlib import Path

import pytest

from streammuse.application.rap import chunk_orchestration as orchestration_module
from streammuse.application.rap.chunk_orchestration import (
    ChunkCandidatePlanner,
    NoValidCandidates,
    PhraseRenderFailed,
    PhraseRenderResult,
    RapChunkOrchestrator,
    RenderBudgetExpired,
)
from streammuse.domain.rap import (
    CandidateBatch,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    ProsodyAnalysis,
    RemoteCandidatePolicy,
    RemoteRapBarRequest,
    RemoteRapChunkRequest,
    ScoreWeights,
    Syllable,
    normalize_text,
)


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class SequenceClock:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class FakeAnalyzer:
    def __init__(
        self,
        *,
        counts: Mapping[str, int] | None = None,
        rhyme_tails: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.counts = dict(counts or {})
        self.rhyme_tails = dict(rhyme_tails or {})
        self.calls: list[str] = []

    def analyze(self, text: str) -> ProsodyAnalysis:
        self.calls.append(text)
        if text == "ANALYZER_ERROR":
            raise ValueError("scripted analysis failure")
        normalized = normalize_text(text)
        count = self.counts.get(text, len(normalized.split()))
        words = normalized.split() or ("empty",)
        syllables = tuple(
            Syllable(
                word=words[index % len(words)],
                index_in_word=0,
                syllable_count=1,
                stress=1,
                phonemes=("AH1",),
                analysis_source="fake",
            )
            for index in range(count)
        )
        default_tail = ((words[-1].upper() + "1"),) if normalized else ()
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalized,
            syllables=syllables,
            end_rhyme_tail=self.rhyme_tails.get(text, default_tail),
            oov_words=(),
            heuristic_words=(),
            punctuation_boundary_after=(),
        )


class ScriptedGenerator:
    def __init__(
        self,
        script: list[tuple[int, tuple[str, ...] | BaseException]],
        *,
        clock: FakeClock | None = None,
        elapsed_per_call: tuple[float, ...] = (),
    ) -> None:
        self.script = list(script)
        self.clock = clock
        self.elapsed_per_call = elapsed_per_call
        self.requests = []

    def generate(self, request):
        call_index = len(self.requests)
        self.requests.append(request)
        expected_bar, result = self.script[call_index]
        assert request.target_bar == expected_bar
        if self.clock is not None and call_index < len(self.elapsed_per_call):
            self.clock.advance(self.elapsed_per_call[call_index])
        if isinstance(result, BaseException):
            raise result
        return CandidateBatch(
            request_id=request.request_id,
            candidates=result,
            source="scripted",
            prompt=(),
            raw_response="\n".join(result),
            latency_ms=0.0,
        )


def flow(template_id: str, ticks: tuple[int, ...] = (0,)) -> FlowTemplate:
    return FlowTemplate(
        template_id=template_id,
        name=template_id,
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(
            FlowSlot(
                tick_in_bar=tick,
                duration_ticks=1,
                target_stress=1.0,
                boundary_strength=3 if index == len(ticks) - 1 else 0,
                rhyme_group="A" if index == len(ticks) - 1 else None,
            )
            for index, tick in enumerate(ticks)
        ),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


def chunk_request(
    *,
    first_flow: FlowTemplate | None = None,
    second_flow: FlowTemplate | None = None,
    first_bar: int = 0,
    policy: RemoteCandidatePolicy | None = None,
    remaining_budget_ms: int = 5_000,
) -> RemoteRapChunkRequest:
    first_flow = first_flow or flow("one-a")
    second_flow = second_flow or flow("one-b")
    return RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=first_bar // 2,
        bars=(
            RemoteRapBarRequest(first_bar, "space", first_flow),
            RemoteRapBarRequest(first_bar + 1, "space", second_flow),
        ),
        tempo_bpm=90.0,
        remaining_budget_ms=remaining_budget_ms,
        policy=policy or RemoteCandidatePolicy("test", 2, 1, 4, 1, 0.0, 500),
        context_lines=("past line",),
        seed=41,
    )


def stress_only_weights() -> ScoreWeights:
    return ScoreWeights(
        stress_alignment=1.0,
        boundary_fit=0.0,
        rhyme_quality=0.0,
        topic_coverage=0.0,
        lexical_continuity=0.0,
        novelty=0.0,
    )


def planner(
    generator, analyzer: FakeAnalyzer | None = None, *, clock: FakeClock | None = None
):
    return ChunkCandidatePlanner(
        generator,
        analyzer or FakeAnalyzer(),
        stress_only_weights(),
        monotonic=clock or FakeClock(),
    )


def test_planner_generates_both_initial_waves_before_stopping() -> None:
    generator = ScriptedGenerator([(0, ("moon", "sun")), (1, ("light", "beat"))])
    request = chunk_request()

    plan = planner(generator).plan(request)

    assert [(item.target_bar, item.count) for item in generator.requests] == [
        (0, 2),
        (1, 2),
    ]
    assert plan.candidate_stats.requested_count == 4
    assert plan.candidate_stats.valid_count == 4
    assert plan.candidate_stats.selectable_count == 4


def test_planner_rescues_only_deficient_bar_and_clamps_final_wave() -> None:
    policy = RemoteCandidatePolicy("rescue", 2, 2, 5, 1, 0.0, 500)
    generator = ScriptedGenerator(
        [
            (0, ("too many", "still long")),
            (1, ("clean", "ready")),
            (0, ("more words", "also wrong")),
            (0, ("valid",)),
        ]
    )

    plan = planner(generator).plan(chunk_request(policy=policy))

    assert [(item.target_bar, item.count) for item in generator.requests] == [
        (0, 2),
        (1, 2),
        (0, 2),
        (0, 1),
    ]
    assert plan.candidate_stats.requested_count == 7
    assert tuple(item.text for item in plan.selected_bars) == ("valid", "clean")


def test_planner_rescues_only_bar_one_when_its_initial_wave_is_deficient() -> None:
    policy = RemoteCandidatePolicy("bar-one-rescue", 2, 1, 3, 1, 0.0, 500)
    generator = ScriptedGenerator(
        [
            (0, ("clean", "steady")),
            (1, ("too many", "still long")),
            (1, ("ready",)),
        ]
    )

    plan = planner(generator).plan(chunk_request(policy=policy))

    assert [(item.target_bar, item.count) for item in generator.requests] == [
        (0, 2),
        (1, 2),
        (1, 1),
    ]
    assert tuple(item.text for item in plan.selected_bars) == ("clean", "ready")


def test_planner_rescues_both_bars_after_both_initial_waves() -> None:
    policy = RemoteCandidatePolicy("both-rescue", 1, 1, 2, 1, 0.0, 500)
    generator = ScriptedGenerator(
        [
            (0, ("too many",)),
            (1, ("still long",)),
            (0, ("clean",)),
            (1, ("ready",)),
        ]
    )

    plan = planner(generator).plan(chunk_request(policy=policy))

    assert [(item.target_bar, item.count) for item in generator.requests] == [
        (0, 1),
        (1, 1),
        (0, 1),
        (1, 1),
    ]
    assert tuple(item.text for item in plan.selected_bars) == ("clean", "ready")


def test_planner_raises_render_budget_expired_before_any_wave() -> None:
    clock = FakeClock()
    generator = ScriptedGenerator([], clock=clock)
    request = chunk_request(
        remaining_budget_ms=500,
        policy=RemoteCandidatePolicy("expired", 1, 0, 1, 1, 0.0, 500),
    )

    with pytest.raises(RenderBudgetExpired):
        planner(generator, clock=clock).plan(request)

    assert generator.requests == []


def test_planner_raises_budget_expired_when_cutoff_hits_between_entry_checks() -> None:
    clock = SequenceClock((100.0, 100.0, 100.5))
    generator = ScriptedGenerator([])
    request = chunk_request(
        remaining_budget_ms=1_000,
        policy=RemoteCandidatePolicy("between-checks", 1, 0, 1, 1, 0.0, 500),
    )

    with pytest.raises(RenderBudgetExpired, match="before the first candidate wave"):
        planner(generator, clock=clock).plan(request)

    assert generator.requests == []


def test_completed_wave_is_evaluated_after_crossing_cutoff_but_no_new_wave_starts() -> (
    None
):
    clock = FakeClock()
    analyzer = FakeAnalyzer()
    generator = ScriptedGenerator(
        [(0, ("moon",))],
        clock=clock,
        elapsed_per_call=(1.1,),
    )
    request = chunk_request(
        remaining_budget_ms=1_500,
        policy=RemoteCandidatePolicy("cutoff", 1, 0, 1, 1, 0.0, 500),
    )

    with pytest.raises(NoValidCandidates) as error:
        planner(generator, analyzer, clock=clock).plan(request)

    assert [item.target_bar for item in generator.requests] == [0]
    assert analyzer.calls == ["moon"]
    assert any(row["status"] == "selectable" for row in error.value.candidate_ledger)


def test_planner_records_generator_failure_then_recovers_in_rescue() -> None:
    generator = ScriptedGenerator(
        [
            (0, RuntimeError("temporary model error")),
            (1, ("steady",)),
            (0, ("recovered",)),
        ]
    )
    request = chunk_request(policy=RemoteCandidatePolicy("retry", 1, 1, 2, 1, 0.0, 500))

    plan = planner(generator).plan(request)

    assert tuple(item.text for item in plan.selected_bars) == ("recovered", "steady")
    assert any(row["status"] == "generation_error" for row in plan.candidate_ledger)


def test_planner_records_explicit_error_batch_then_recovers_in_rescue() -> None:
    class ErrorBatchThenRecovery:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return CandidateBatch(
                    request_id=request.request_id,
                    candidates=(),
                    source="local_chat_independent",
                    prompt=(),
                    raw_response="",
                    latency_ms=12.0,
                    warning="upstream timeout",
                    error_type="generation_error",
                    error_message="upstream timeout",
                )
            text = "steady" if request.target_bar == 1 else "recovered"
            return CandidateBatch(
                request_id=request.request_id,
                candidates=(text,),
                source="local_chat_independent",
                prompt=(),
                raw_response=text,
                latency_ms=1.0,
            )

    generator = ErrorBatchThenRecovery()
    request = chunk_request(
        policy=RemoteCandidatePolicy("error-batch", 1, 1, 2, 1, 0.0, 500)
    )

    plan = planner(generator).plan(request)

    error_rows = [
        row for row in plan.candidate_ledger if row["status"] == "generation_error"
    ]
    assert len(error_rows) == 1
    assert error_rows[0]["reasons"] == ("generation_error", "upstream timeout")
    assert tuple(item.text for item in plan.selected_bars) == ("recovered", "steady")


def test_planner_deduplicates_normalized_lines_and_records_malformed_candidates() -> (
    None
):
    generator = ScriptedGenerator(
        [
            (0, ("", "moon", "MOON!", "ANALYZER_ERROR")),
            (1, ("sun", "beat", "light", "flow")),
        ]
    )
    request = chunk_request(policy=RemoteCandidatePolicy("messy", 4, 0, 4, 1, 0.0, 500))

    plan = planner(generator).plan(request)

    statuses = [row["status"] for row in plan.candidate_ledger if row["bar"] == 0]
    assert statuses == ["malformed", "selectable", "duplicate", "analysis_error"]
    assert plan.candidate_stats.requested_count == 8
    assert plan.candidate_stats.parseable_count == 5


def test_evaluation_timing_includes_analysis_and_accumulated_reranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()

    class TimedAnalyzer(FakeAnalyzer):
        def analyze(self, text: str) -> ProsodyAnalysis:
            analysis = super().analyze(text)
            clock.advance(0.010)
            return analysis

    real_rank_candidates = orchestration_module.rank_candidates

    def timed_rank_candidates(*args, **kwargs):
        result = real_rank_candidates(*args, **kwargs)
        clock.advance(0.020)
        return result

    monkeypatch.setattr(orchestration_module, "rank_candidates", timed_rank_candidates)
    generator = ScriptedGenerator([(0, ("moon",)), (1, ("sun",))])

    plan = planner(generator, TimedAnalyzer(), clock=clock).plan(
        chunk_request(policy=RemoteCandidatePolicy("timed", 1, 0, 1, 1, 0.0, 500))
    )

    assert plan.stage_timings_ms["generation"] == pytest.approx(0.0)
    assert plan.stage_timings_ms["evaluation"] == pytest.approx(60.0)
    assert plan.stage_timings_ms["overhead"] == pytest.approx(0.0)
    assert plan.stage_timings_ms["total"] == pytest.approx(
        plan.stage_timings_ms["generation"]
        + plan.stage_timings_ms["evaluation"]
        + plan.stage_timings_ms["overhead"]
    )


def test_planner_raises_when_hard_gate_or_minimum_score_leaves_a_bar_unselectable() -> (
    None
):
    invalid = ScriptedGenerator([(0, ("too many",)), (1, ("also long",))])
    with pytest.raises(NoValidCandidates, match="no selectable candidate"):
        planner(invalid).plan(
            chunk_request(policy=RemoteCandidatePolicy("invalid", 1, 0, 1, 1, 0.0, 500))
        )

    below_threshold = ScriptedGenerator([(0, ("moon",)), (1, ("sun",))])
    with pytest.raises(NoValidCandidates, match="minimum score"):
        planner(below_threshold).plan(
            chunk_request(
                policy=RemoteCandidatePolicy("threshold", 1, 0, 1, 1, 1.1, 500)
            )
        )


def test_selected_bars_retain_every_existing_ranker_component() -> None:
    generator = ScriptedGenerator([(0, ("moon",)), (1, ("sun",))])

    plan = ChunkCandidatePlanner(
        generator,
        FakeAnalyzer(),
        ScoreWeights(),
        monotonic=FakeClock(),
    ).plan(
        chunk_request(policy=RemoteCandidatePolicy("components", 1, 0, 1, 1, 0.0, 500))
    )

    expected = {
        "stress_alignment",
        "boundary_fit",
        "rhyme_quality",
        "topic_coverage",
        "lexical_continuity",
        "novelty",
    }
    assert set(plan.selected_bars[0].diagnostics["component_scores"]) == expected
    assert plan.selected_bars[0].scheduled


def test_pair_selection_uses_continuity_then_rhyme_then_source_order() -> None:
    two = flow("pair", (0, 8))
    request = chunk_request(
        first_flow=two,
        second_flow=two,
        policy=RemoteCandidatePolicy("pair", 2, 0, 2, 1, 0.0, 500),
    )
    generator = ScriptedGenerator(
        [
            (0, ("shared alpha", "plain beta")),
            (1, ("shared gamma", "other delta")),
        ]
    )

    plan = planner(generator).plan(request)

    assert tuple(item.text for item in plan.selected_bars) == (
        "shared alpha",
        "shared gamma",
    )
    pair = plan.selected_bars[0].diagnostics["pair_tiebreak"]
    assert pair["lexical_continuity"] == 0.5
    assert pair["source_order"] == (0, 0)

    rhyme_generator = ScriptedGenerator(
        [
            (0, ("red moon", "blue sun")),
            (1, ("cold rain", "warm star")),
        ]
    )
    rhyme_analyzer = FakeAnalyzer(
        rhyme_tails={
            "red moon": ("AA1",),
            "blue sun": ("IH1",),
            "cold rain": ("AA1",),
            "warm star": ("ER1",),
        }
    )
    rhyme_plan = planner(rhyme_generator, rhyme_analyzer).plan(request)
    assert tuple(item.text for item in rhyme_plan.selected_bars) == (
        "red moon",
        "cold rain",
    )
    assert (
        rhyme_plan.selected_bars[0].diagnostics["pair_tiebreak"]["rhyme_quality"] == 1.0
    )


def test_pair_selection_preserves_source_order_when_every_score_ties() -> None:
    two = flow("source-order", (0, 8))
    generator = ScriptedGenerator(
        [
            (0, ("red moon", "blue sun")),
            (1, ("cold star", "warm beat")),
        ]
    )

    plan = planner(generator).plan(
        chunk_request(
            first_flow=two,
            second_flow=two,
            policy=RemoteCandidatePolicy("source", 2, 0, 2, 1, 0.0, 500),
        )
    )

    assert tuple(item.text for item in plan.selected_bars) == ("red moon", "cold star")


def test_pair_selection_prioritizes_mean_score_over_continuity_and_rhyme() -> None:
    two = flow("mean-first", (0, 8))
    generator = ScriptedGenerator(
        [
            (0, ("space moon", "shared moon")),
            (1, ("space star", "shared moon")),
        ]
    )
    weights = ScoreWeights(
        stress_alignment=0.0,
        boundary_fit=0.0,
        rhyme_quality=0.0,
        topic_coverage=1.0,
        lexical_continuity=0.0,
        novelty=0.0,
    )

    plan = ChunkCandidatePlanner(
        generator,
        FakeAnalyzer(),
        weights,
        monotonic=FakeClock(),
    ).plan(
        chunk_request(
            first_flow=two,
            second_flow=two,
            policy=RemoteCandidatePolicy("mean-first", 2, 0, 2, 1, 0.0, 500),
        )
    )

    assert tuple(item.text for item in plan.selected_bars) == (
        "space moon",
        "space star",
    )
    pair = plan.selected_bars[0].diagnostics["pair_tiebreak"]
    assert pair["mean_total_score"] == 1.0
    assert pair["lexical_continuity"] == 0.0
    assert pair["rhyme_quality"] == 0.0


def test_over_returned_choices_keep_provider_index_separate_from_source_order() -> None:
    class OverReturningGenerator:
        def generate(self, request):
            candidates = ("moon", "extra") if request.target_bar == 0 else ("sun",)
            provider_indices = (40, 73) if request.target_bar == 0 else (11,)
            return CandidateBatch(
                request_id=request.request_id,
                candidates=candidates,
                source="independent",
                prompt=(),
                raw_response="\n".join(candidates),
                latency_ms=0.0,
                provider_choice_indices=provider_indices,
            )

    plan = planner(OverReturningGenerator()).plan(
        chunk_request(policy=RemoteCandidatePolicy("over-return", 1, 0, 1, 1, 0.0, 500))
    )

    bar_zero = [row for row in plan.candidate_ledger if row.get("bar") == 0]
    assert [
        (row["source_order"], row["provider_choice_index"]) for row in bar_zero
    ] == [
        (0, 40),
        (1, 73),
    ]
    assert [row["status"] for row in bar_zero] == ["selectable", "over_returned"]
    assert plan.candidate_stats.requested_count == 2
    assert plan.candidate_stats.parseable_count == 2


def test_render_request_contains_exact_targets_for_two_different_flows() -> None:
    first = flow("first", (0, 4))
    second = flow("second", (1, 15))
    generator = ScriptedGenerator([(4, ("red moon",)), (5, ("blue sky",))])
    request = chunk_request(
        first_flow=first,
        second_flow=second,
        first_bar=4,
        policy=RemoteCandidatePolicy("timing", 1, 0, 1, 1, 0.0, 500),
    )

    plan = planner(generator).plan(request)

    assert plan.render_request.start_bar == 4
    assert plan.render_request.end_bar == 6
    assert plan.render_request.text == "red moon\nblue sky"
    assert tuple(item.absolute_tick for item in plan.render_request.syllables) == (
        64,
        68,
        81,
        95,
    )
    assert tuple(item.tick_in_chunk for item in plan.render_request.syllables) == (
        0,
        4,
        17,
        31,
    )
    assert tuple(
        item.target_seconds for item in plan.render_request.syllables
    ) == pytest.approx((0.0, 4 / 6, 17 / 6, 31 / 6))


def test_candidate_summaries_are_bounded_while_ledger_remains_complete() -> None:
    left = (
        "red",
        "blue",
        "green",
        "black",
        "white",
        "gold",
        "bright",
        "dark",
        "fast",
        "slow",
    )
    right = (
        "kick",
        "snare",
        "bass",
        "drum",
        "beat",
        "flow",
        "verse",
        "rhyme",
        "sound",
        "stage",
    )
    generator = ScriptedGenerator([(0, left), (1, right)])
    request = chunk_request(
        policy=RemoteCandidatePolicy("bounded", 10, 0, 10, 1, 0.0, 500)
    )

    plan = planner(generator).plan(request)

    assert len(plan.candidate_ledger) == 20
    assert len(plan.candidate_stats.top_candidates) == 8
    assert len(plan.candidate_stats.rejections) <= 8
    for selected in plan.selected_bars:
        assert len(selected.diagnostics["top_candidates"]) == 8
        assert len(selected.diagnostics["representative_rejections"]) <= 8
    with pytest.raises(TypeError):
        plan.candidate_ledger[0]["status"] = "mutated"


def wav_bytes(*, frame_count: int = 128_000, sample: int = 1_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(struct.pack("<h", sample) * frame_count)
    return buffer.getvalue()


class FakeRenderer:
    def __init__(self, result: PhraseRenderResult | BaseException) -> None:
        self.result = result
        self.calls: list[tuple[object, Path]] = []

    def render(self, request, workspace: Path) -> PhraseRenderResult:
        self.calls.append((request, workspace))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def phrase_result(vocal_wav: bytes | None = None) -> PhraseRenderResult:
    return PhraseRenderResult(
        vocal_wav=vocal_wav or wav_bytes(),
        alignment_diagnostics={
            "fallback_counts": {"word": 0},
            "source_anchors": [0.0, 1.0],
            "target_anchors": [0.0, 1.0],
            "local_warp_ratios": [1.0],
        },
        audio_diagnostics={
            "sample_rate_hz": 24_000,
            "frame_count": 128_000,
            "duration_seconds": 128_000 / 24_000,
            "peak": 0.5,
        },
        model_tool_versions={
            "moss": "test",
            "aligner": "mms-test",
            "rubberband": "test",
        },
        warnings=("test warning",),
        stage_timings_ms={"moss": 100.0, "aligner": 20.0, "warp": 10.0},
        monitoring_summary={
            "schema_version": "streammuse.rap_chunk_monitor.v1",
            "alignment_method": "mms-test",
            "alignment_confidence": 0.87,
            "source_wav_sha256": "b" * 64,
        },
    )


def simple_planner_and_request():
    generator = ScriptedGenerator([(0, ("moon",)), (1, ("sun",))])
    request = chunk_request(
        policy=RemoteCandidatePolicy("render", 1, 0, 1, 1, 0.0, 500)
    )
    return planner(generator), request


def test_orchestrator_combines_plan_render_and_manifest_diagnostics(
    tmp_path: Path,
) -> None:
    lyric_planner, request = simple_planner_and_request()
    renderer = FakeRenderer(phrase_result())

    artifact = RapChunkOrchestrator(
        lyric_planner,
        renderer,
        workspace_root=tmp_path,
        monotonic=FakeClock(),
    ).render(request)

    assert artifact.manifest.request_id == request.request_id
    assert tuple(item.text for item in artifact.manifest.selected_bars) == (
        "moon",
        "sun",
    )
    assert artifact.vocal_wav == renderer.result.vocal_wav  # type: ignore[union-attr]
    assert artifact.workspace == tmp_path / request.request_id
    assert renderer.calls[0][0].text == "moon\nsun"
    diagnostics = artifact.manifest.diagnostics
    assert diagnostics.accepted_request_budget_ms == request.remaining_budget_ms
    assert diagnostics.resolved_policy == request.policy
    assert diagnostics.candidate_stats.requested_count == 2
    assert set(diagnostics.stage_timings_ms) == {
        "generation",
        "evaluation",
        "moss",
        "aligner",
        "warp",
        "packaging",
        "total",
    }
    assert diagnostics.stage_timings_ms["packaging"] == 0.0
    assert diagnostics.alignment_diagnostics["fallback_counts"] == {"word": 0}
    assert diagnostics.audio_diagnostics["frame_count"] == request.expected_frame_count
    assert diagnostics.model_tool_versions["aligner"] == "mms-test"
    assert diagnostics.monitoring_summary["alignment_method"] == "mms-test"
    assert diagnostics.monitoring_summary["alignment_confidence"] == 0.87
    assert diagnostics.monitoring_summary["source_wav_sha256"] == "b" * 64
    assert diagnostics.monitoring_summary["artifact_ids"]["manifest"] == "manifest.json"
    assert diagnostics.warnings == ("test warning", "packaging timing is provisional")
    assert "component_scores" in artifact.manifest.selected_bars[0].diagnostics


def test_partial_choice_warnings_survive_ledger_and_success_manifest(
    tmp_path: Path,
) -> None:
    class WarningGenerator:
        def generate(self, request):
            warning = f"choice[{request.target_bar + 7}] malformed"
            return CandidateBatch(
                request_id=request.request_id,
                candidates=("moon" if request.target_bar == 0 else "sun",),
                source="local_chat_independent",
                prompt=(),
                raw_response="raw",
                latency_ms=1.0,
                warning=warning,
                provider_choice_indices=(request.target_bar + 3,),
            )

    request = chunk_request(
        policy=RemoteCandidatePolicy("warnings", 1, 0, 1, 1, 0.0, 500)
    )
    artifact = RapChunkOrchestrator(
        planner(WarningGenerator()),
        FakeRenderer(phrase_result()),
        workspace_root=tmp_path,
    ).render(request)

    warning_rows = [
        row
        for row in artifact.candidate_ledger
        if row.get("status") == "generation_warning"
    ]
    assert [row["reasons"] for row in warning_rows] == [
        ("choice[7] malformed",),
        ("choice[8] malformed",),
    ]
    candidate_rows = [
        row for row in artifact.candidate_ledger if row.get("candidate_id")
    ]
    assert [row["provider_choice_index"] for row in candidate_rows] == [3, 4]
    assert artifact.manifest.diagnostics.warnings == (
        "choice[7] malformed",
        "choice[8] malformed",
        "test warning",
        "packaging timing is provisional",
    )


def test_orchestrator_wraps_renderer_failure(tmp_path: Path) -> None:
    lyric_planner, request = simple_planner_and_request()
    renderer = FakeRenderer(RuntimeError("MOSS failed"))

    with pytest.raises(PhraseRenderFailed, match="MOSS failed"):
        RapChunkOrchestrator(lyric_planner, renderer, workspace_root=tmp_path).render(
            request
        )


def test_orchestrator_wraps_workspace_failure_with_bounded_sanitized_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lyric_planner, request = simple_planner_and_request()
    renderer = FakeRenderer(phrase_result())
    secret = "workspace-secret-value"

    def fail_mkdir(_path, *args, **kwargs):
        del args, kwargs
        raise OSError(f"OPENAI_API_KEY={secret}; {'x' * 1_000}")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)

    with pytest.raises(PhraseRenderFailed) as error:
        RapChunkOrchestrator(
            lyric_planner,
            renderer,
            workspace_root=tmp_path,
        ).render(request)

    assert "workspace preparation failed" in str(error.value)
    assert secret not in str(error.value)
    assert "OPENAI_API_KEY=[REDACTED]" in str(error.value)
    assert len(str(error.value)) <= 320
    assert renderer.calls == []


def test_orchestrator_does_not_wrap_system_exit_from_renderer(tmp_path: Path) -> None:
    lyric_planner, request = simple_planner_and_request()

    with pytest.raises(SystemExit, match="stop now"):
        RapChunkOrchestrator(
            lyric_planner,
            FakeRenderer(SystemExit("stop now")),
            workspace_root=tmp_path,
        ).render(request)


@pytest.mark.parametrize(
    "vocal_wav",
    (
        b"not a wav",
        wav_bytes(sample=0),
        wav_bytes(frame_count=127_999),
    ),
    ids=("malformed", "silent", "wrong-duration"),
)
def test_orchestrator_rejects_malformed_silent_or_wrong_duration_wav(
    tmp_path: Path,
    vocal_wav: bytes,
) -> None:
    lyric_planner, request = simple_planner_and_request()

    with pytest.raises(PhraseRenderFailed):
        RapChunkOrchestrator(
            lyric_planner,
            FakeRenderer(phrase_result(vocal_wav)),
            workspace_root=tmp_path,
        ).render(request)
