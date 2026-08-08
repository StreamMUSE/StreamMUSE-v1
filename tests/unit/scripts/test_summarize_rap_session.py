"""Tests for deterministic regeneration of rap research session artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from streammuse.domain.rap import RapEvent, RapEventType
from streammuse.infrastructure.rap.recorder import RapSessionRecorder, build_session_manifest


def _manifest(*, repetition_window_bars: int) -> dict[str, object]:
    return build_session_manifest(
        scenario_id="analysis-test",
        scenario={
            "scenario_id": "analysis-test",
            "loop": True,
            "tempo_bpm": 92.0,
            "segments": [
                {
                    "start_bar": 0,
                    "bars": 3,
                    "topic": "space",
                    "template_id": "nine",
                    "fallback_lines": ["alpha beta gamma"],
                }
            ],
        },
        seed=20260807,
        tempo={"bpm": 92.0, "ticks_per_beat": 4, "beats_per_bar": 4},
        templates=[
            {
                "template_id": "nine",
                "name": "Nine",
                "definition": {
                    "ticks_per_beat": 4,
                    "beats_per_bar": 4,
                    "slots": [
                        {
                            "tick_in_bar": 0,
                            "duration_ticks": 1,
                            "target_stress": 1.0,
                            "boundary_strength": 3,
                            "rhyme_group": "A",
                        }
                    ],
                },
                "provenance": {
                    "kind": "test",
                    "source": "fixture",
                    "source_hash": None,
                    "quantization_error_ticks": 0.0,
                },
            }
        ],
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
        minimum_score=0.55,
        timeout_seconds=5.0,
        lookahead_bars=2,
        python_version="3.10",
        platform="test",
        package_version="0.1.0",
        git_revision="abc123",
        git_dirty=False,
        repetition_window_bars=repetition_window_bars,
    )


def _event(
    sequence: int,
    event_type: RapEventType,
    *,
    bar: int | None = None,
    payload: dict[str, object] | None = None,
) -> RapEvent:
    return RapEvent(
        session_id="rap-analysis-test",
        sequence=sequence,
        event_type=event_type,
        utc_time="2026-08-07T00:00:00+00:00",
        monotonic_ns=sequence * 1_000,
        bar=bar,
        tick=None,
        request_id=f"request-{bar}" if bar is not None else None,
        payload=payload or {},
    )


def _record_session(session_dir: Path, *, repetition_window_bars: int = 1) -> None:
    recorder = RapSessionRecorder(
        session_dir,
        manifest=_manifest(repetition_window_bars=repetition_window_bars),
    )
    events = (
        _event(
            1,
            RapEventType.SESSION_STARTED,
            payload={"repetition_window_bars": repetition_window_bars},
        ),
        _event(
            2,
            RapEventType.BAR_FROZEN,
            bar=0,
            payload={"text": "alpha beta gamma", "source": "phrase_bank", "fallback": False},
        ),
        _event(
            3,
            RapEventType.BAR_FROZEN,
            bar=1,
            payload={"text": "other words here", "source": "phrase_bank", "fallback": False},
        ),
        _event(
            4,
            RapEventType.BAR_FROZEN,
            bar=2,
            payload={"text": "alpha beta again", "source": "phrase_bank", "fallback": False},
        ),
    )
    for event in events:
        recorder(event)
    recorder.close()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_cli_regenerates_data_equivalent_recorder_artifacts(tmp_path: Path, load_script) -> None:
    """Catches regeneration that drifts from the recorder's canonical derivations."""
    session_dir = tmp_path / "session"
    _record_session(session_dir, repetition_window_bars=1)
    script = load_script("summarize_rap_session")

    assert script.main([str(session_dir)]) == 0

    canonical_json = (session_dir / "summary.json").read_text(encoding="utf-8")
    regenerated_json = (session_dir / "summary.regenerated.json").read_text(encoding="utf-8")
    assert regenerated_json.rstrip("\n") == canonical_json.rstrip("\n")
    assert _csv_rows(session_dir / "bars.regenerated.csv") == _csv_rows(session_dir / "bars.csv")
    assert json.loads(regenerated_json)["metrics"]["repetition"] == {
        "numerator": 0,
        "denominator": 6,
        "rate": 0.0,
    }


def test_cli_uses_manifest_repetition_window_and_rejects_event_disagreement(
    tmp_path: Path,
    load_script,
) -> None:
    """Catches silently deriving repetition with event/default state instead of the manifest."""
    session_dir = tmp_path / "session"
    _record_session(session_dir, repetition_window_bars=1)
    manifest_path = session_dir / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["repetition_window_bars"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    script = load_script("summarize_rap_session")

    with pytest.raises(ValueError, match="event and manifest repetition window disagree"):
        script.main([str(session_dir)])

    assert not (session_dir / "summary.regenerated.json").exists()
    assert not (session_dir / "bars.regenerated.csv").exists()
