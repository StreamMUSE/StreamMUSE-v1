"""Tests for deterministic rap research recording artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.infrastructure.rap.recorder import (
    DEFAULT_REPETITION_WINDOW_BARS,
    RapSessionRecorder,
    _is_json_prefix,
    build_session_manifest,
    derive_bar_rows,
    derive_summary,
    event_to_dict,
    read_events,
    validate_session_manifest,
)


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
        _event(1, RapEventType.SESSION_STARTED, payload={"repetition_window_bars": 2}),
        _event(2, RapEventType.BAR_PLANNING_STARTED, bar=0, request_id="r0"),
        _event(3, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=0, request_id="r0", payload={"candidate_count": 1, "latency_ms": 10.0, "deadline_slack_ms": 50.0, "late": False, "prompt_tokens": 21, "completion_tokens": 13, "warning": "short_batch"}),
        _event(4, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "c0", "valid": True, "word_analysis_sources": [{"word": "clean", "source": "cmudict_first_pronunciation"}, {"word": "line", "source": "vowel_group_heuristic"}]}),
        _event(5, RapEventType.BAR_FROZEN, bar=0, request_id="r0", payload={"text": "clean line now", "source": "local_chat", "fallback": False, "fallback_reason": None}),
        _event(6, RapEventType.BAR_PLANNING_STARTED, bar=1, request_id="r1"),
        _event(7, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=1, request_id="r1", payload={"latency_ms": 20.0, "deadline_slack_ms": 10.0, "error_type": "generation_error", "late": True, "prompt_tokens": 9}),
        _event(8, RapEventType.CANDIDATE_EVALUATED, bar=1, request_id="r1", payload={"candidate_id": "c1", "valid": False, "rejection_reasons": ["duplicate_normalized_text"], "word_analysis_sources": [{"word": "repeat", "source": "heuristic"}]}),
        _event(9, RapEventType.GENERATION_FAILED, bar=1, request_id="r1", payload={"error_type": "generation_error"}),
        _event(10, RapEventType.BAR_FROZEN, bar=1, request_id="r1", payload={"text": "line now again", "source": "local_chat", "fallback": True, "fallback_reason": "deadline_miss"}),
        _event(11, RapEventType.FALLBACK_ACTIVATED, bar=1, request_id="r1", payload={"fallback_reason": "deadline_miss"}),
        _event(12, RapEventType.SYLLABLE_EMITTED, bar=0, tick=1, payload={"label": "clean", "observation_delay_ms": 1.0}),
        _event(13, RapEventType.SYLLABLE_EMITTED, bar=1, tick=17, payload={"label": "fallback", "observation_delay_ms": -2.0}),
    )


def _manifest() -> dict[str, object]:
    return build_session_manifest(
        scenario_id="test",
        scenario={
            "scenario_id": "test",
            "loop": True,
            "tempo_bpm": 120.0,
            "segments": [{"start_bar": 0, "bars": 1, "topic": "space", "template_id": "one", "fallback_lines": ["space"]}],
        },
        seed=7,
        tempo={"bpm": 120.0, "ticks_per_beat": 4, "beats_per_bar": 4},
        templates=[{"template_id": "one", "name": "One slot", "definition": {"ticks_per_beat": 4, "beats_per_bar": 4, "slots": [{"tick_in_bar": 0, "duration_ticks": 1, "target_stress": 1.0, "boundary_strength": 0, "rhyme_group": "A"}]}, "provenance": {"kind": "test", "source": "fixture", "source_hash": None, "quantization_error_ticks": 0.0}}],
        generator_config={"name": "phrase_bank"},
        model_config={"name": "none"},
        score_weights={
            "stress_alignment": 0.30,
            "boundary_fit": 0.10,
            "rhyme_quality": 0.20,
            "topic_coverage": 0.20,
            "lexical_continuity": 0.15,
            "novelty": 0.05,
        },
        minimum_score=0.5,
        timeout_seconds=1.0,
        lookahead_bars=2,
        python_version="3.10",
        platform="test",
        package_version="0.1.0",
        git_revision="abc123",
        git_dirty=False,
        repetition_window_bars=2,
    )


def test_recorder_writes_redacted_manifest_recoverable_jsonl_and_summary(tmp_path: Path) -> None:
    recorder = RapSessionRecorder(
        tmp_path / "session",
        manifest={**_manifest(), "api_key": "do-not-record", "nested": {"authorization": "Bearer no"}},
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

    assert manifest["api_key"] == "[REDACTED]"
    assert manifest["nested"] == {"authorization": "[REDACTED]"}
    assert manifest["scenario_id"] == "test"
    assert len(read_events(session_dir / "events.jsonl")) == len(_scripted_session_events())
    assert summary["bars"]["frozen"] == 2
    assert summary["bars"]["fallback_rate"] == 0.5
    assert summary["generation_diagnostics"]["prompt_tokens"] == {"count": 2, "total": 30}


def test_summary_uses_explicit_ratio_denominators_and_null_empty_observations() -> None:
    summary = derive_summary(_scripted_session_events())

    assert summary["metrics"] == {
        "candidate_validity": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "fallback": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "deadline_miss": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "generator_error": {"numerator": 1, "denominator": 2, "rate": 0.5},
        "pronunciation_fallback": {"numerator": 2, "denominator": 3, "rate": 2 / 3},
        "repetition": {"numerator": 1, "denominator": 4, "rate": 0.25},
    }
    assert {key: summary["latencies"][key] for key in ("generation_latency_ms", "deadline_slack_ms", "syllable_observation_delay_ms")} == {
        "generation_latency_ms": {"count": 2, "p50": 15.0, "p95": 19.5, "max": 20.0},
        "deadline_slack_ms": {"count": 2, "p50": 30.0, "p95": 48.0, "max": 50.0},
        "syllable_observation_delay_ms": {"count": 2, "p50": -0.5, "p95": 0.8499999999999996, "max": 1.0},
    }
    for key in ("synthesis_latency_ms", "bar_render_latency_ms", "audio_commit_slack_ms"):
        assert summary["latencies"][key] == {"count": 0, "p50": None, "p95": None, "max": None}
    assert summary["generation_diagnostics"] == {
        "prompt_tokens": {"count": 2, "total": 30},
        "completion_tokens": {"count": 1, "total": 13},
        "warnings": {"count": 1},
    }
    empty = derive_summary(())
    assert empty["latencies"]["generation_latency_ms"] == {"count": 0, "p50": None, "p95": None, "max": None}


def test_summary_derives_additive_audio_metrics_from_canonical_events() -> None:
    summary = derive_summary(
        (
            _event(1, RapEventType.AUDIO_RENDER_COMPLETED, bar=0, payload={"render_latency_ms": 12.0}),
            _event(2, RapEventType.BAR_AUDIO_COMMITTED, bar=0, payload={"deadline_slack_ms": 250.0}),
            _event(3, RapEventType.PRONUNCIATION_FALLBACK, bar=0, payload={}),
            _event(4, RapEventType.AUDIO_UNDERRUN, payload={}),
            _event(5, RapEventType.BAR_PLAYBACK_COMPLETED, bar=0, payload={}),
        )
    )

    assert summary["audio"] == {
        "render_latency_ms": {"count": 1, "p50": 12.0, "p95": 12.0, "max": 12.0},
        "commit_slack_ms": {"count": 1, "p50": 250.0, "p95": 250.0, "max": 250.0},
            "warning_counts": {
                "pronunciation_fallback": 1,
                "timing_pressure": 0,
                "forced_bar_fit": 0,
                "synthesis_failed": 0,
            },
        "underruns": 1,
        "completed_bars": 1,
        "completed_frames": 0,
    }


def test_summary_uses_only_the_current_audio_epoch_after_reset() -> None:
    summary = derive_summary(
        (
            _event(1, RapEventType.BAR_FROZEN, bar=0, payload={"fallback": False}),
            _event(2, RapEventType.BAR_AUDIO_COMMITTED, bar=0, payload={"frame_count": 100}),
            _event(3, RapEventType.BAR_PLAYBACK_COMPLETED, bar=0, payload={}),
            _event(4, RapEventType.SESSION_RESET, payload={}),
            _event(5, RapEventType.BAR_FROZEN, bar=0, payload={"fallback": True}),
            _event(6, RapEventType.BAR_AUDIO_COMMITTED, bar=0, payload={"frame_count": 200}),
            _event(7, RapEventType.BAR_PLAYBACK_COMPLETED, bar=0, payload={}),
        )
    )

    assert summary["events"] == {"count": 3, "history_count": 7, "epoch": 1}
    assert summary["bars"]["frozen"] == 1
    assert summary["bars"]["fallback"] == 1
    assert summary["audio"]["completed_bars"] == 1
    assert summary["audio"]["completed_frames"] == 200


def test_summary_ignores_stale_coordinator_events_arriving_after_reset() -> None:
    summary = derive_summary(
        (
            _event(1, RapEventType.SESSION_RESET, payload={}),
            _event(
                2,
                RapEventType.AUDIO_RENDER_COMPLETED,
                bar=0,
                payload={"render_latency_ms": 99.0, "coordinator_epoch": 0, "accepted": False},
            ),
            _event(3, RapEventType.TIMING_PRESSURE, bar=0, payload={"coordinator_epoch": 0}),
            _event(4, RapEventType.BAR_AUDIO_COMMITTED, bar=0, payload={"frame_count": 99, "coordinator_epoch": 0}),
            _event(
                5,
                RapEventType.AUDIO_RENDER_COMPLETED,
                bar=0,
                payload={"render_latency_ms": 12.0, "coordinator_epoch": 1, "accepted": True},
            ),
            _event(6, RapEventType.TIMING_PRESSURE, bar=0, payload={"coordinator_epoch": 1}),
            _event(7, RapEventType.BAR_AUDIO_COMMITTED, bar=0, payload={"frame_count": 20, "coordinator_epoch": 1}),
            _event(8, RapEventType.BAR_PLAYBACK_COMPLETED, bar=0, payload={}),
        )
    )

    assert summary["events"] == {"count": 4, "history_count": 8, "epoch": 1}
    assert summary["audio"]["render_latency_ms"] == {"count": 1, "p50": 12.0, "p95": 12.0, "max": 12.0}
    assert summary["audio"]["warning_counts"]["timing_pressure"] == 1
    assert summary["audio"]["completed_frames"] == 20


def test_bar_rows_are_deterministic_and_written_as_csv(tmp_path: Path) -> None:
    rows = derive_bar_rows(_scripted_session_events())
    assert rows == [
        {
            "bar": 0,
            "request_id": "r0",
            "source": "local_chat",
            "text": "clean line now",
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
            "mean_syllable_observation_delay_ms": 1.0,
        },
        {
            "bar": 1,
            "request_id": "r1",
            "source": "local_chat",
            "text": "line now again",
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
            "mean_syllable_observation_delay_ms": -2.0,
        },
    ]

    recorder = RapSessionRecorder(tmp_path / "session", manifest=_manifest())
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
        _event(3, RapEventType.BAR_FROZEN, bar=0, request_id="r0", payload={"text": "one", "fallback": False}),
    )

    assert derive_bar_rows(events)[0]["candidate_count"] == 1


def test_manifest_builder_validator_and_recorder_require_reproducibility_contract(tmp_path: Path) -> None:
    manifest = _manifest()
    assert validate_session_manifest(manifest) == manifest
    incomplete = dict(manifest)
    incomplete.pop("git_revision")
    with pytest.raises(ValueError, match="git_revision"):
        validate_session_manifest(incomplete)
    with pytest.raises(ValueError, match="git_revision"):
        RapSessionRecorder(tmp_path / "incomplete", incomplete)
    assert not (tmp_path / "incomplete").exists()
    malformed = dict(manifest, git_dirty="false")
    with pytest.raises(ValueError, match="git_dirty"):
        validate_session_manifest(malformed)
    assert DEFAULT_REPETITION_WINDOW_BARS > 0


def test_recorder_redacts_secret_keys_in_event_payloads(tmp_path: Path) -> None:
    recorder = RapSessionRecorder(tmp_path / "session", manifest=_manifest())
    recorder(_event(1, RapEventType.SESSION_STARTED, payload={"repetition_window_bars": 2}))
    recorder(_event(2, RapEventType.CANDIDATE_BATCH_RECEIVED, payload={"api_key": "secret", "nested": {"token": "hidden"}, "safe": "kept"}))
    recorder.close()

    event = json.loads((tmp_path / "session" / "events.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert event["payload"] == {"api_key": "[REDACTED]", "nested": {"token": "[REDACTED]"}, "safe": "kept"}


@pytest.mark.parametrize(
    ("records", "message"),
    (
        ((1, 1), "sequence"),
        ((1, 3), "sequence"),
    ),
)
def test_read_events_rejects_duplicate_and_gapped_sequences(tmp_path: Path, records: tuple[int, int], message: str) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(event_to_dict(_event(sequence, RapEventType.TICK))) + "\n" for sequence in records), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_events(path)


def test_read_events_rejects_interior_corruption_complete_final_corruption_and_mixed_sessions(tmp_path: Path) -> None:
    interior = tmp_path / "interior.jsonl"
    interior.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + "\n{bad}\n" + json.dumps(event_to_dict(_event(2, RapEventType.TICK))) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_events(interior)
    blank = tmp_path / "blank.jsonl"
    blank.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + "\n\n" + json.dumps(event_to_dict(_event(2, RapEventType.TICK))) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_events(blank)
    final = tmp_path / "final.jsonl"
    final.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + "\n{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_events(final)
    final_without_newline = tmp_path / "final-without-newline.jsonl"
    final_without_newline.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + "\n{bad}", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_events(final_without_newline)
    invalid_prefix = tmp_path / "invalid-prefix.jsonl"
    invalid_prefix.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + "\n{\"payload\": NOT_JSON", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        read_events(invalid_prefix)
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + "\n" + json.dumps(event_to_dict(RapEvent("other", 2, RapEventType.TICK, "time", 2, None, None, None, {}))) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="session"):
        read_events(mixed)


def test_read_events_ignores_only_an_incomplete_final_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + "\n{\"partial\":", encoding="utf-8")
    assert [event.sequence for event in read_events(path)] == [1]


def test_every_strict_prefix_of_a_canonical_event_line_is_recoverable() -> None:
    event = _event(
        1,
        RapEventType.CANDIDATE_BATCH_RECEIVED,
        payload={
            "escaped": "quote \" backslash \\ snowman \u2603",
            "exponent": -1.25e-7,
            "flag": True,
        },
    )
    canonical_line = json.dumps(event_to_dict(event), sort_keys=True)

    prefixes = (canonical_line[:end] for end in range(1, len(canonical_line)))
    rejected = [prefix for prefix in prefixes if not _is_json_prefix(prefix)]

    assert rejected == []


@pytest.mark.parametrize(
    "value",
    (
        " ",
        "{",
        '{"key',
        '{"key"',
        '{"key":',
        '{"key": [',
        '{"key": [1,',
        '{"key": [1, {',
        ' [ true, false, null, -',
        '"unterminated',
        '"dangling\\',
        '"unicode\\u',
        '"unicode\\u12',
        *(f'"escape\\{escape}' for escape in '"\\/bfnrt'),
        "t",
        "fa",
        "nul",
        "-",
        "-0",
        "12.",
        "12.3e",
        "12.3e+",
        "12.3e-",
    ),
)
def test_json_prefix_recognizer_accepts_standard_json_prefix_states(value: str) -> None:
    assert _is_json_prefix(value)


@pytest.mark.parametrize(
    "value",
    (
        "]",
        "{]",
        '{"key" 1',
        '{"key": }',
        '{"key": 1,}',
        "[1,]",
        "[,",
        "[1 2",
        '{"key": 1]',
        '"bad\\x',
        '"bad\\u12g',
        '"bad\x1f',
        "trux",
        "falsehood",
        "nullx",
        "01",
        "-01",
        "1\u0662",
        ".1",
        "+1",
        "1..",
        "1ex",
        "1e++",
        "{} trailing",
    ),
)
def test_json_prefix_recognizer_rejects_invalid_json_near_matches(value: str) -> None:
    assert not _is_json_prefix(value)


@pytest.mark.parametrize("token", ("tru", "fals", "nul", "-", "1e", "1e+", "1e-", "1."))
def test_read_events_recovers_valid_scalar_token_crash_tails(tmp_path: Path, token: str) -> None:
    path = tmp_path / f"{token.replace('+', 'plus').replace('-', 'minus').replace('.', 'dot')}.jsonl"
    path.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + f"\n{{\"payload\": {token}", encoding="utf-8")

    assert [event.sequence for event in read_events(path)] == [1]


@pytest.mark.parametrize("token", ("trux", "tru ", "falsehood", "nullx", "1ex", "1e ", "1e+z", "1.."))
def test_read_events_rejects_invalid_scalar_token_crash_tails(tmp_path: Path, token: str) -> None:
    path = tmp_path / f"invalid-{token.replace('+', 'plus').replace('.', 'dot')}.jsonl"
    path.write_text(json.dumps(event_to_dict(_event(1, RapEventType.TICK))) + f"\n{{\"payload\": {token}", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        read_events(path)


def test_derivations_deduplicate_candidate_and_request_identities_and_only_emit_frozen_rows() -> None:
    events = (
        _event(1, RapEventType.SESSION_STARTED, payload={"repetition_window_bars": 2}),
        _event(2, RapEventType.BAR_PLANNING_STARTED, bar=0, request_id="r0"),
        _event(3, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=0, request_id="r0", payload={"candidate_count": 2}),
        _event(4, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=0, request_id="r0", payload={"candidate_count": 99}),
        _event(5, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "c0", "valid": True}),
        _event(6, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "c0", "valid": True}),
        _event(7, RapEventType.BAR_PLANNING_STARTED, bar=1, request_id="r1"),
        _event(8, RapEventType.GENERATION_FAILED, bar=1, request_id="r1", payload={"error_type": "generation_error"}),
        _event(9, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=1, request_id="r1", payload={"error_type": "generation_error", "late": True}),
        _event(10, RapEventType.BAR_FROZEN, bar=0, request_id="r0", payload={"text": "one two three", "fallback": False}),
    )

    summary = derive_summary(events)
    assert summary["metrics"]["candidate_validity"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert summary["metrics"]["deadline_miss"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert summary["metrics"]["generator_error"] == {"numerator": 1, "denominator": 2, "rate": 0.5}
    assert [row["bar"] for row in derive_bar_rows(events)] == [0]


def test_manifest_validation_is_deep_json_strict_and_does_not_create_artifacts(tmp_path: Path) -> None:
    manifest = _manifest()
    malformed = dict(manifest)
    malformed["tempo"] = {"bpm": 120.0, "ticks_per_beat": 0, "beats_per_bar": 4}
    with pytest.raises(ValueError, match="tempo"):
        RapSessionRecorder(tmp_path / "bad-tempo", malformed)
    malformed = _manifest()
    malformed["score_weights"] = {"score": float("nan")}
    with pytest.raises(ValueError, match="score_weights"):
        RapSessionRecorder(tmp_path / "bad-weight", malformed)
    malformed = _manifest()
    malformed["scenario"] = {"scenario_id": "test", "loop": True, "tempo_bpm": 120.0, "segments": []}
    with pytest.raises(ValueError, match="segments"):
        RapSessionRecorder(tmp_path / "bad-scenario", malformed)
    malformed = _manifest()
    malformed["model_config"] = {"name": "valid", "nested": object()}
    with pytest.raises(ValueError):
        RapSessionRecorder(tmp_path / "bad-json", malformed)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda manifest: manifest["scenario"]["segments"][0].__setitem__("template_id", "missing"), "template"),
        (lambda manifest: manifest["templates"][0]["definition"].__setitem__("ticks_per_beat", 8), "meter"),
        (lambda manifest: manifest["templates"][0]["definition"]["slots"][0].pop("boundary_strength"), "boundary_strength"),
        (lambda manifest: manifest["templates"][0]["definition"]["slots"][0].__setitem__("rhyme_group", ""), "rhyme_group"),
        (lambda manifest: manifest["templates"][0]["provenance"].__setitem__("source_hash", ""), "source_hash"),
        (lambda manifest: manifest["templates"][0]["provenance"].__setitem__("quantization_error_ticks", -1.0), "quantization_error_ticks"),
        (lambda manifest: manifest.__setitem__("score_weights", {"not_a_weight": 1.0}), "score_weights"),
        (lambda manifest: manifest["score_weights"].__setitem__("novelty", 0.06), "score_weights"),
    ),
)
def test_manifest_cross_validates_templates_slots_provenance_and_score_weights(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    manifest = _manifest()
    mutate(manifest)

    with pytest.raises(ValueError, match=message):
        RapSessionRecorder(tmp_path / "invalid", manifest)
    assert not (tmp_path / "invalid").exists()


def test_summary_rejects_manifest_repetition_window_disagreement() -> None:
    events = (_event(1, RapEventType.SESSION_STARTED, payload={"repetition_window_bars": 2}),)
    with pytest.raises(ValueError, match="repetition window"):
        derive_summary(events, expected_manifest_window=3)


def test_derivations_reject_invalid_or_contradictory_candidate_counts() -> None:
    invalid_count = (
        _event(1, RapEventType.CANDIDATE_BATCH_RECEIVED, request_id="r0", payload={"candidate_count": 1.5}),
    )
    with pytest.raises(ValueError, match="candidate_count"):
        derive_summary(invalid_count)
    contradiction = (
        _event(1, RapEventType.CANDIDATE_BATCH_RECEIVED, bar=0, request_id="r0", payload={"candidate_count": 1}),
        _event(2, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "one", "valid": True}),
        _event(3, RapEventType.CANDIDATE_EVALUATED, bar=0, request_id="r0", payload={"candidate_id": "two", "valid": True}),
        _event(4, RapEventType.BAR_FROZEN, bar=0, request_id="r0", payload={"text": "one", "fallback": False}),
    )
    with pytest.raises(ValueError, match="candidate count"):
        derive_summary(contradiction)
    with pytest.raises(ValueError, match="candidate count"):
        derive_bar_rows(contradiction)


def test_bar_rows_capture_generator_failure_without_a_batch() -> None:
    rows = derive_bar_rows((
        _event(1, RapEventType.GENERATION_FAILED, bar=0, request_id="r0", payload={"error_type": "generation_error"}),
        _event(2, RapEventType.BAR_FROZEN, bar=0, request_id="r0", payload={"text": "fallback", "fallback": True}),
    ))
    assert rows[0]["generator_error"] == "generation_error"
