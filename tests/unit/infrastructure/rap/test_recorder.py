"""Tests for deterministic rap research recording artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.infrastructure.rap.recorder import RapSessionRecorder, derive_bar_rows, derive_summary, read_events


def _event(
    sequence: int,
    event_type: RapEventType,
    *,
    bar: int | None = None,
    tick: int | None = None,
    request_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> RapEvent:
    return RapEvent(
        session_id="session-1",
        sequence=sequence,
        event_type=event_type,
        utc_time="2026-08-07T00:00:00+00:00",
        monotonic_ns=sequence * 100,
        bar=bar,
        tick=tick,
        request_id=request_id,
        payload=payload or {},
    )


def _scripted_session_events() -> tuple[RapEvent, ...]:
    return (
        _event(1, RapEventType.SESSION_STARTED),
        _event(2, RapEventType.BAR_PLANNING_STARTED, bar=0, request_id="r0"),
        _event(3, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=0, request_id="r0", payload={"latency_ms": 10.0, "deadline_slack_ms": 50.0}),
        _event(4, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "c0", "valid": True, "word_analysis_sources": [{"word": "clean", "source": "cmudict_first_pronunciation"}, {"word": "line", "source": "vowel_group_heuristic"}]}),
        _event(5, RapEventType.BAR_FROZEN, bar=0, request_id="r0", payload={"text": "clean line", "source": "local_chat", "fallback": False, "fallback_reason": None}),
        _event(6, RapEventType.BAR_PLANNING_STARTED, bar=1, request_id="r1"),
        _event(7, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=1, request_id="r1", payload={"latency_ms": 20.0, "deadline_slack_ms": 10.0, "error_type": "generation_error"}),
        _event(8, RapEventType.CANDIDATE_EVALUATED, bar=1, request_id="r1", payload={"candidate_id": "c1", "valid": False, "rejection_reasons": ["duplicate_normalized_text"], "word_analysis_sources": [{"word": "repeat", "source": "heuristic"}]}),
        _event(9, RapEventType.BAR_FROZEN, bar=1, request_id="r1", payload={"text": "fallback line", "source": "prevalidated_fallback", "fallback": True, "fallback_reason": "deadline_miss"}),
        _event(10, RapEventType.FALLBACK_ACTIVATED, bar=1, request_id="r1", payload={"fallback_reason": "deadline_miss"}),
        _event(11, RapEventType.SYLLABLE_EMITTED, bar=0, tick=1, payload={"label": "clean", "jitter_ms": 1.0}),
        _event(12, RapEventType.SYLLABLE_EMITTED, bar=1, tick=17, payload={"label": "fallback", "jitter_ms": -2.0}),
    )


def test_recorder_writes_redacted_manifest_recoverable_jsonl_and_summary(tmp_path: Path) -> None:
    recorder = RapSessionRecorder(
        tmp_path / "session",
        manifest={"scenario_id": "test", "api_key": "do-not-record", "nested": {"authorization": "Bearer no"}},
    )
    for event in _scripted_session_events():
        recorder(event)
    recorder._stream.flush()  # Simulate a process crash during its final JSONL write.
    with recorder._events_path.open("a", encoding="utf-8") as stream:
        stream.write('{"partial":')
    recorder.close()

    session_dir = tmp_path / "session"
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))

    assert manifest == {"api_key": "[REDACTED]", "nested": {"authorization": "[REDACTED]"}, "scenario_id": "test"}
    assert len(read_events(session_dir / "events.jsonl")) == len(_scripted_session_events())
    assert summary["bars"]["frozen"] == 2
    assert summary["bars"]["fallback_rate"] == 0.5


def test_summary_uses_explicit_ratio_denominators_and_null_empty_observations() -> None:
    summary = derive_summary(_scripted_session_events())

    assert summary["metrics"] == {
        "candidate_validity": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "fallback": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "deadline_miss": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "generator_error": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "pronunciation_fallback": {"numerator": 2, "denominator": 3, "rate": 2 / 3},
        "repetition": {"numerator": 1, "denominator": 2, "rate": 0.5},
    }
    assert summary["latencies"] == {
        "generation_latency_ms": {"count": 2, "p50": 15.0, "p95": 19.5, "max": 20.0},
        "deadline_slack_ms": {"count": 2, "p50": 30.0, "p95": 48.0, "max": 50.0},
        "emission_jitter_ms": {"count": 2, "p50": -0.5, "p95": 0.8499999999999996, "max": 1.0},
    }
    empty = derive_summary(())
    assert empty["latencies"]["generation_latency_ms"] == {"count": 0, "p50": None, "p95": None, "max": None}


def test_bar_rows_are_deterministic_and_written_as_csv(tmp_path: Path) -> None:
    rows = derive_bar_rows(_scripted_session_events())
    assert rows == [
        {
            "bar": 0,
            "request_id": "r0",
            "source": "local_chat",
            "text": "clean line",
            "frozen": True,
            "fallback": False,
            "fallback_reason": None,
            "candidate_count": 1,
            "valid_candidate_count": 1,
            "selected_candidate_id": None,
            "generation_latency_ms": 10.0,
            "deadline_slack_ms": 50.0,
            "generator_error": None,
            "emitted_syllables": 1,
            "mean_emission_jitter_ms": 1.0,
        },
        {
            "bar": 1,
            "request_id": "r1",
            "source": "prevalidated_fallback",
            "text": "fallback line",
            "frozen": True,
            "fallback": True,
            "fallback_reason": "deadline_miss",
            "candidate_count": 1,
            "valid_candidate_count": 0,
            "selected_candidate_id": None,
            "generation_latency_ms": 20.0,
            "deadline_slack_ms": 10.0,
            "generator_error": "generation_error",
            "emitted_syllables": 1,
            "mean_emission_jitter_ms": -2.0,
        },
    ]

    recorder = RapSessionRecorder(tmp_path / "session", manifest={"scenario_id": "test"})
    for event in _scripted_session_events():
        recorder(event)
    recorder.close()
    with (tmp_path / "session" / "bars.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))

    assert [row["bar"] for row in csv_rows] == ["0", "1"]
    assert csv_rows[1]["fallback_reason"] == "deadline_miss"


def test_bar_rows_do_not_double_count_declared_and_evaluated_candidates() -> None:
    events = (
        _event(1, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=0, request_id="r0", payload={"candidate_count": 1}),
        _event(2, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "c0", "valid": True}),
    )

    assert derive_bar_rows(events)[0]["candidate_count"] == 1
