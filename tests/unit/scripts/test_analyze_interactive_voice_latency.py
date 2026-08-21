from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


def _breakdown(
    *,
    actor: str,
    session_offset_ms: float,
    anchors: dict[str, float],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "clock": "monotonic",
        "origin": "interactive_turn_start",
        "origin_session_offset_ms": session_offset_ms,
        "actor": actor,
        "anchors_ms": {
            "turn_started": 0.0,
            "game_validation_completed": max(anchors.values()),
            **anchors,
        },
        "durations_ms": {
            "pipeline_to_decision": max(anchors.values()),
        },
        "components": {},
    }


def test_analyzer_derives_cross_turn_metrics_and_writes_outputs(
    load_script,
    tmp_path: Path,
) -> None:
    analyzer = load_script("analyze_interactive_voice_latency")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    human = {
        "turn_id": 0,
        "actor": "human",
        "number": 1,
        "is_valid": True,
        "deadline_missed": False,
        "latency_ms": 500.0,
        "metadata": {
            "timing_breakdown": _breakdown(
                actor="human",
                session_offset_ms=100.0,
                anchors={
                    "response_source_started": 5.0,
                    "microphone.stream_open_started": 10.0,
                    "microphone.first_callback": 20.0,
                    "microphone.first_voiced": 100.0,
                    "microphone.last_voiced": 300.0,
                    "microphone.endpoint_detected": 600.0,
                    "asr.completed": 750.0,
                },
            )
        },
    }
    llm = {
        "turn_id": 1,
        "actor": "llm",
        "number": 2,
        "is_valid": True,
        "deadline_missed": False,
        "latency_ms": 200.0,
        "metadata": {
            "timing_breakdown": _breakdown(
                actor="llm",
                session_offset_ms=900.0,
                anchors={
                    "llm_request_started": 10.0,
                    "llm_response_completed": 210.0,
                    "speech.first_dac_sample": 260.0,
                    "speech.playback_drained": 700.0,
                },
            )
        },
    }
    trace = run_dir / "response_trace.jsonl"
    trace.write_text(
        "\n".join(json.dumps(row) for row in (human, llm)) + "\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "test-run",
                "human_input": {
                    "recognizer": {
                        "model_resolution_ms": 10.0,
                        "model_load_ms": 20.0,
                        "warmup_ms": 30.0,
                    }
                },
                "speech_output": {
                    "prewarm_ms": 40.0,
                    "prewarm_entry_count": 7,
                },
            }
        ),
        encoding="utf-8",
    )

    result = analyzer.analyze_traces((trace,))
    output_dir = tmp_path / "analysis"
    analyzer.write_outputs(output_dir, result)

    assert result["turn_count"] == 2
    assert result["startup"][0]["stt_model_load_ms"] == 20.0
    metric = "conversation.human_last_voice_to_first_dac_ms"
    assert result["metrics"][metric]["p50"] == pytest.approx(760.0)
    with (output_dir / "breakdown_turns.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1][metric] == "760.0"
    assert (output_dir / "breakdown_summary.json").is_file()
    assert (output_dir / "breakdown_summary.md").is_file()


def test_analyzer_accepts_open_ended_turn_without_number_or_expected(
    load_script,
    tmp_path: Path,
) -> None:
    analyzer = load_script("analyze_interactive_voice_latency")
    trace = tmp_path / "response_trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "turn_id": 0,
                "actor": "human",
                "number": None,
                "expected": None,
                "is_valid": True,
                "deadline_missed": False,
                "latency_ms": 300.0,
                "metadata": {
                    "referee_metadata": {"normalized_animal": "lion"},
                    "timing_breakdown": _breakdown(
                        actor="human",
                        session_offset_ms=0.0,
                        anchors={
                            "response_source_started": 5.0,
                            "microphone.first_voiced": 50.0,
                            "microphone.last_voiced": 150.0,
                            "asr.completed": 250.0,
                        },
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyzer.analyze_traces((trace,))
    output_dir = tmp_path / "analysis"
    analyzer.write_outputs(output_dir, result)

    assert result["turn_count"] == 1
    assert result["rows"][0]["number"] is None
    with (output_dir / "breakdown_turns.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["number"] == ""


def test_analyzer_rejects_negative_stage_duration(
    load_script,
    tmp_path: Path,
) -> None:
    analyzer = load_script("analyze_interactive_voice_latency")
    trace = tmp_path / "response_trace.jsonl"
    breakdown = _breakdown(
        actor="human",
        session_offset_ms=0.0,
        anchors={"game_validation_completed": 1.0},
    )
    breakdown["durations_ms"] = {"bad": -1.0}
    trace.write_text(
        json.dumps(
            {
                "turn_id": 0,
                "actor": "human",
                "metadata": {"timing_breakdown": breakdown},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        analyzer.analyze_traces((trace,))


def test_analyzer_rejects_reversed_derived_anchors(
    load_script,
    tmp_path: Path,
) -> None:
    analyzer = load_script("analyze_interactive_voice_latency")
    trace = tmp_path / "response_trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "turn_id": 0,
                "actor": "human",
                "metadata": {
                    "timing_breakdown": _breakdown(
                        actor="human",
                        session_offset_ms=0.0,
                        anchors={
                            "response_source_started": 20.0,
                            "microphone.first_voiced": 10.0,
                        },
                    )
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reversed anchors"):
        analyzer.analyze_traces((trace,))
