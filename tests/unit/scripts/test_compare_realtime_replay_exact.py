from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import compare_realtime_replay_exact as comparator


def _request_rows(*, request_id: str, session_id: str, observed_at: float) -> list[dict]:
    protocol_context = {
        "prompt_length_ticks": 32,
        "generation_interval_ticks": 4,
        "bpm": 120,
    }
    return [
        {
            "schema_version": 1,
            "sequence": 1,
            "operation": "start",
            "request": {
                "melody_events": [
                    {"type": "note_on", "pitch": 60, "tick": 0, "velocity": 88},
                    {"type": "note_off", "pitch": 60, "tick": 2, "velocity": 0},
                ],
                "observed_until_tick": 4,
                **protocol_context,
            },
            "protocol_context": dict(protocol_context),
            "acknowledgement": {
                "request_id": request_id,
                "session_id": session_id,
                "session_epoch": 1,
                "phase": "prompt_running",
                "observed_at": observed_at,
            },
        },
        {
            "schema_version": 1,
            "sequence": 2,
            "operation": "append",
            "request": {
                "melody_events": [
                    {"type": "note_on", "pitch": 62, "tick": 4, "velocity": 91},
                    {"type": "note_off", "pitch": 62, "tick": 6, "velocity": 0},
                ],
                "observed_until_tick": 8,
            },
            "protocol_context": dict(protocol_context),
            "acknowledgement": {
                "request_id": f"{request_id}-append",
                "session_id": session_id,
                "session_epoch": 1,
                "phase": "catching_up",
                "observed_at": observed_at + 1.0,
            },
        },
    ]


def _continuation_generation(
    *, session_id: str, session_epoch: int, tick: int, empty: bool
) -> dict:
    return {
        "request_id": f"{session_id}-continuation-{tick}",
        "session_id": session_id,
        "session_epoch": session_epoch,
        "generation_start_tick": tick,
        "input_increment_digest": f"increment-{tick}",
        "input_cumulative_digest": f"cumulative-{tick}",
        "part0_roll_digest": f"roll-{tick}",
        "part0_roll_shape": [2, 88, tick + 4],
        "part0_roll_bytes_sha256": f"bytes-{tick}",
        "prompt_token_digest": f"prompt-{tick}",
        "part0_token_digest": f"part0-{tick}",
        "raw_token_digest": f"raw-{tick}",
        "token_decode_digest": f"decode-{tick}",
        "output_event_digest": f"events-{tick}",
        "empty_success": empty,
        "inference_time_ms": float(tick),
    }


def _model_trace(*, session_id: str, session_epoch: int) -> dict:
    return {
        "schema_version": 1,
            "runtime_info": {
                "trace_capture_complete": True,
                "seed_provenance_complete": True,
            "seeded_session_active": True,
            "prompt_sample_seed": 17,
            "continuation_sample_seed": 23,
            "session_id": session_id,
            "session_epoch": session_epoch,
        },
        "prompt_generation_log": {
            "prompt_tokens": [1, 10, 20],
            "generated_tokens": [1, 10, 20, 30, 31],
            "new_tokens": [30, 31],
            "selection_mode": "rule_s",
            "candidate_count": 2,
            "selected_candidate_number": 2,
            "rule_s_id": "rule-s-v1",
            "rule_s_recommended_candidate_number": 2,
            "eligible_candidate_count": 2,
            "selection_fallback_reason": None,
            "prompt_batch_generation_time_ms": 999.0,
            "prompt_batch_scoring_time_ms": 111.0,
            "prompt_candidates": [
                {
                    "candidate_number": 1,
                    "prompt_token_hash": "candidate-one",
                    "prompt_ppl": 2.5,
                    "acc_pitch_range": 12.0,
                    "rule_s_score": 0.2,
                },
                {
                    "candidate_number": 2,
                    "prompt_token_hash": "candidate-two",
                    "prompt_ppl": 2.1,
                    "acc_pitch_range": 18.0,
                    "rule_s_score": 0.8,
                },
            ],
        },
        "continuation_generations": [
            _continuation_generation(
                session_id=session_id,
                session_epoch=session_epoch,
                tick=32,
                empty=False,
            ),
            _continuation_generation(
                session_id=session_id,
                session_epoch=session_epoch,
                tick=36,
                empty=True,
            ),
        ],
    }


def _write_session(
    root: Path,
    *,
    request_rows: list[dict] | None = None,
    model_trace: dict | None = None,
) -> Path:
    root.mkdir()
    rows = request_rows if request_rows is not None else _request_rows(
        request_id=f"request-{root.name}",
        session_id=f"session-{root.name}",
        observed_at=1.0,
    )
    (root / "prompt_continuation_replay_requests.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    trace = model_trace if model_trace is not None else _model_trace(
        session_id=f"session-{root.name}", session_epoch=1
    )
    (root / "prompt_continuation_model_trace.json").write_text(
        json.dumps(trace), encoding="utf-8"
    )
    runtime_info = trace.get("runtime_info", {})
    (root / "prompt_continuation_session_seed.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "runtime_session_id": root.name,
                "success": True,
                "prompt_requested_seed": 17,
                "prompt_effective_seed": 17,
                "continuation_requested_seed": 23,
                "continuation_effective_seed": 23,
                "prompt_seed_source": "system",
                "continuation_seed_source": "system",
                "session_id": runtime_info.get(
                    "session_id", f"session-{root.name}"
                ),
                "session_epoch": runtime_info.get("session_epoch", 1),
            }
        ),
        encoding="utf-8",
    )
    return root


def _set_path(value: Any, path: tuple[Any, ...], replacement: Any) -> None:
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement


def test_exact_comparison_uses_direct_trace_and_ignores_ack_timing_and_ids(
    tmp_path, capsys
):
    original = _write_session(
        tmp_path / "original",
        request_rows=_request_rows(
            request_id="live-request",
            session_id="live-session",
            observed_at=10.0,
        ),
        model_trace=_model_trace(session_id="live-session", session_epoch=3),
    )
    replay_trace = _model_trace(session_id="replay-session", session_epoch=9)
    replay_trace["prompt_generation_log"][
        "prompt_batch_generation_time_ms"
    ] = 7.0
    replay_trace["prompt_generation_log"][
        "prompt_batch_scoring_time_ms"
    ] = 8.0
    replay_trace["continuation_generations"][0]["inference_time_ms"] = 987.0
    replay = _write_session(
        tmp_path / "replay",
        request_rows=_request_rows(
            request_id="replay-request",
            session_id="replay-session",
            observed_at=200.0,
        ),
        model_trace=replay_trace,
    )
    output = tmp_path / "comparison.json"

    assert comparator.main([str(original), str(replay), "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == printed
    assert saved["comparable"] is True
    assert saved["model_exact"] is True
    assert saved["protocol_request_exact"] is True
    assert saved["prompt_input_exact"] is True
    assert saved["prompt_output_exact"] is True
    assert saved["continuation_input_exact"] is True
    assert saved["continuation_output_exact"] is True
    assert saved["first_mismatch"] is None


def test_single_prompt_mode_does_not_require_rule_s_candidate_evidence(tmp_path):
    original_trace = _model_trace(session_id="original-session", session_epoch=1)
    replay_trace = _model_trace(session_id="replay-session", session_epoch=2)
    for trace in (original_trace, replay_trace):
        prompt = trace["prompt_generation_log"]
        prompt["selection_mode"] = "single"
        prompt["candidate_count"] = 1
        prompt["selected_candidate_number"] = 1
        for field in (
            "rule_s_id",
            "rule_s_recommended_candidate_number",
            "eligible_candidate_count",
            "selection_fallback_reason",
            "prompt_candidates",
        ):
            prompt.pop(field, None)
    original = _write_session(tmp_path / "original", model_trace=original_trace)
    replay = _write_session(tmp_path / "replay", model_trace=replay_trace)

    result = comparator.compare_session_directories(original, replay)
    assert result["comparable"] is True
    assert result["model_exact"] is True


@pytest.mark.parametrize(
    ("target_name", "path", "replacement", "component"),
    [
        (
            "requests",
            (1, "request", "melody_events", 0, "tick"),
            5,
            "protocol_request",
        ),
        (
            "trace",
            ("prompt_generation_log", "prompt_tokens", 1),
            99,
            "prompt_input",
        ),
        (
            "trace",
            ("prompt_generation_log", "new_tokens", 0),
            99,
            "prompt_output",
        ),
        (
            "trace",
            (
                "prompt_generation_log",
                "prompt_candidates",
                1,
                "rule_s_score",
            ),
            0.1,
            "prompt_output",
        ),
        (
            "trace",
            (
                "continuation_generations",
                0,
                "input_cumulative_digest",
            ),
            "different-input",
            "continuation_input",
        ),
        (
            "trace",
            (
                "continuation_generations",
                0,
                "output_event_digest",
            ),
            "different-output",
            "continuation_output",
        ),
    ],
)
def test_mismatch_is_comparable_and_returns_one(
    tmp_path, capsys, target_name, path, replacement, component
):
    requests = _request_rows(
        request_id="stable-request",
        session_id="stable-session",
        observed_at=1.0,
    )
    trace = _model_trace(session_id="stable-session", session_epoch=1)
    original = _write_session(
        tmp_path / "original",
        request_rows=copy.deepcopy(requests),
        model_trace=copy.deepcopy(trace),
    )
    replay_requests = copy.deepcopy(requests)
    replay_trace = copy.deepcopy(trace)
    _set_path(
        replay_requests if target_name == "requests" else replay_trace,
        path,
        replacement,
    )
    replay = _write_session(
        tmp_path / "replay",
        request_rows=replay_requests,
        model_trace=replay_trace,
    )

    assert comparator.main([str(original), str(replay)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["comparable"] is True
    assert result["model_exact"] is False
    assert result["first_mismatch"]["component"] == component


@pytest.mark.parametrize(
    "missing_kind",
    ["required_field", "seed_provenance", "missing_prompt_trace"],
)
def test_missing_evidence_never_claims_exact_and_returns_two(
    tmp_path, capsys, missing_kind
):
    original = _write_session(tmp_path / "original")
    replay_trace = _model_trace(session_id="replay-session", session_epoch=1)
    if missing_kind == "required_field":
        del replay_trace["continuation_generations"][0][
            "part0_roll_bytes_sha256"
        ]
    elif missing_kind == "seed_provenance":
        replay_trace["runtime_info"] = {
            "prompt_sample_seed": 17,
            "continuation_sample_seed": 23,
        }
    else:
        del replay_trace["prompt_generation_log"]
    replay = _write_session(tmp_path / "replay", model_trace=replay_trace)

    assert comparator.main([str(original), str(replay)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["comparable"] is False
    assert result["model_exact"] is False
    assert result["protocol_request_exact"] is False
    assert result["first_mismatch"]["component"] == "evidence"
    assert result["errors"]


@pytest.mark.parametrize(
    ("invalid_kind", "replacement"),
    [
        ("error_field", {"type": "RuntimeError", "message": "failed"}),
        ("missing_acknowledgement", None),
        ("non_object_acknowledgement", "ok"),
    ],
)
def test_failed_or_unacknowledged_protocol_request_is_missing_evidence(
    tmp_path, invalid_kind, replacement
):
    original = _write_session(tmp_path / "original")
    replay_requests = _request_rows(
        request_id="replay-request",
        session_id="replay-session",
        observed_at=1.0,
    )
    if invalid_kind == "error_field":
        replay_requests[1]["error"] = replacement
    elif invalid_kind == "missing_acknowledgement":
        replay_requests[1].pop("acknowledgement")
    else:
        replay_requests[1]["acknowledgement"] = replacement
    replay = _write_session(
        tmp_path / "replay",
        request_rows=replay_requests,
    )

    result = comparator.compare_session_directories(original, replay)
    assert comparator.exit_code(result) == 2
    assert result["comparable"] is False
    assert result["model_exact"] is False
    assert result["first_mismatch"]["component"] == "evidence"
    assert "protocol request 1" in result["errors"][0]


@pytest.mark.parametrize("capture_value", [None, False])
def test_incomplete_model_trace_capture_is_missing_evidence(
    tmp_path, capture_value
):
    original = _write_session(tmp_path / "original")
    replay_trace = _model_trace(session_id="replay-session", session_epoch=1)
    if capture_value is None:
        replay_trace["runtime_info"].pop("trace_capture_complete")
    else:
        replay_trace["runtime_info"]["trace_capture_complete"] = capture_value
    replay = _write_session(tmp_path / "replay", model_trace=replay_trace)

    result = comparator.compare_session_directories(original, replay)
    assert comparator.exit_code(result) == 2
    assert result["trace_capture_complete"] is False
    assert result["comparable"] is False
    assert result["model_exact"] is False
    assert "trace_capture_complete" in result["errors"][0]
