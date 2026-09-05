#!/usr/bin/env python3
"""Compare an original live RuntimeSession with its realtime MIDI replay.

The comparison is deliberately evidence-gated.  A result is comparable only
when both session directories contain complete protocol, prompt-model,
continuation-model, and seeded-session evidence.  Timing and per-run identity
fields are excluded; all model-semantic fields are compared exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REQUEST_TRACE_NAME = "prompt_continuation_replay_requests.jsonl"
MODEL_TRACE_NAME = "prompt_continuation_model_trace.json"
SESSION_SEED_NAME = "prompt_continuation_session_seed.json"
MANIFEST_NAME = "replay_audit_manifest.json"

PROMPT_FIELDS = ("prompt_tokens", "generated_tokens", "new_tokens")
PROMPT_SELECTION_FIELDS = (
    "selection_mode",
    "candidate_count",
    "selected_candidate_number",
    "rule_s_id",
    "rule_s_recommended_candidate_number",
    "eligible_candidate_count",
    "selection_fallback_reason",
    "prompt_candidates",
)
CONTINUATION_INPUT_FIELDS = (
    "generation_start_tick",
    "input_increment_digest",
    "input_cumulative_digest",
    "part0_roll_digest",
    "part0_roll_shape",
    "part0_roll_bytes_sha256",
    "prompt_token_digest",
    "part0_token_digest",
)
CONTINUATION_OUTPUT_FIELDS = (
    "raw_token_digest",
    "token_decode_digest",
    "output_event_digest",
    "empty_success",
)

PROMPT_OUTPUT_SELECTION_FIELDS = (
    "selection_mode",
    "candidate_count",
    "selected_candidate_number",
    "rule_s_id",
    "rule_s_recommended_candidate_number",
    "eligible_candidate_count",
    "selection_fallback_reason",
    "ranked_candidate_numbers",
)

class EvidenceError(ValueError):
    """Raised when a session lacks evidence required for exact comparison."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceError(f"missing required artifact: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON artifact {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise EvidenceError(f"missing required artifact: {path}") from exc
    except OSError as exc:
        raise EvidenceError(f"cannot read JSONL artifact {path}: {exc}") from exc

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(
                f"invalid JSON in {path} at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(record, dict):
            raise EvidenceError(
                f"{path} line {line_number} must contain a JSON object"
            )
        records.append(record)
    if not records:
        raise EvidenceError(f"required request trace is empty: {path}")
    return records


def _require_mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{description} must be a JSON object")
    return value


def _protocol_request_evidence(
    record: Mapping[str, Any], index: int
) -> dict[str, Any]:
    if "error" in record:
        raise EvidenceError(
            f"protocol request {index} contains an error field and is incomplete"
        )
    acknowledgement = record.get("acknowledgement")
    if not isinstance(acknowledgement, dict):
        raise EvidenceError(
            f"protocol request {index} lacks a successful acknowledgement object"
        )
    operation = record.get("operation")
    request = record.get("request")
    protocol_context = record.get("protocol_context")
    if not isinstance(operation, str) or not operation:
        raise EvidenceError(f"protocol request {index} lacks a valid operation")
    if not isinstance(request, dict):
        raise EvidenceError(f"protocol request {index} lacks a request object")
    if not isinstance(protocol_context, dict):
        raise EvidenceError(
            f"protocol request {index} lacks a stable protocol_context object"
        )
    return {
        "operation": operation,
        "request": request,
        "protocol_context": protocol_context,
    }


def _extract_prompt_record(trace: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in (
        "prompt_generation_log",
        "prompt_model",
        "prompt_generation",
        "prompt",
    ):
        candidate = trace.get(key)
        if isinstance(candidate, dict):
            return candidate
    if all(field in trace for field in PROMPT_FIELDS):
        return trace
    raise EvidenceError(
        "model trace lacks a prompt_generation_log/prompt_model/prompt_generation/"
        "prompt record"
    )


def _extract_continuation_records(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidate: Any = None
    for key in (
        "continuation_generations",
        "continuation_generation_logs",
        "continuation_model",
        "continuation",
    ):
        if key in trace:
            candidate = trace[key]
            break

    if isinstance(candidate, dict):
        for key in ("generations", "records", "trace"):
            if key in candidate:
                candidate = candidate[key]
                break
    if not isinstance(candidate, list):
        raise EvidenceError("model trace lacks an ordered continuation generation list")
    if not candidate:
        raise EvidenceError("continuation generation evidence is empty")
    if not all(isinstance(record, dict) for record in candidate):
        raise EvidenceError("every continuation generation must be a JSON object")
    return candidate


def _require_token_list(record: Mapping[str, Any], field: str) -> list[int]:
    if field not in record:
        raise EvidenceError(f"prompt model trace is missing {field}")
    value = record[field]
    if not isinstance(value, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in value
    ):
        raise EvidenceError(f"prompt model field {field} must be a list of integers")
    return list(value)


def _is_prompt_timing_field(field: str) -> bool:
    return field.startswith("prompt_batch_") and field.endswith("_time_ms")


def _without_prompt_timing_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_prompt_timing_fields(item)
            for key, item in value.items()
            if not _is_prompt_timing_field(key)
        }
    if isinstance(value, list):
        return [_without_prompt_timing_fields(item) for item in value]
    return value


def _candidate_has_score_or_feature(candidate: Mapping[str, Any]) -> bool:
    identity_fields = {
        "candidate_number",
        "prompt_token_hash",
        "generated_token_count",
    }
    return any(
        key not in identity_fields and not _is_prompt_timing_field(key)
        for key in candidate
    )


def _prompt_output_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "generated_tokens": _require_token_list(record, "generated_tokens"),
        "new_tokens": _require_token_list(record, "new_tokens"),
    }
    for field in PROMPT_SELECTION_FIELDS:
        if field in record:
            output[field] = _without_prompt_timing_fields(record[field])

    selection_mode = output.get("selection_mode")
    if selection_mode is not None and (
        not isinstance(selection_mode, str) or not selection_mode
    ):
        raise EvidenceError("prompt selection_mode must be a non-empty string")
    candidate_count = output.get("candidate_count")
    if candidate_count is not None and (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count <= 0
    ):
        raise EvidenceError("prompt candidate_count must be a positive integer")
    selected = output.get("selected_candidate_number")
    if selected is not None and (
        isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0
    ):
        raise EvidenceError(
            "prompt selected_candidate_number must be a positive integer"
        )
    if isinstance(candidate_count, int) and isinstance(selected, int):
        if selected > candidate_count:
            raise EvidenceError(
                "prompt selected_candidate_number exceeds candidate_count"
            )

    if selection_mode is not None and selection_mode != "single":
        required = (
            "candidate_count",
            "selected_candidate_number",
            "rule_s_id",
            "prompt_candidates",
        )
        missing = [field for field in required if field not in output]
        if missing:
            raise EvidenceError(
                "batch Prompt trace is missing stable selection evidence: "
                + ", ".join(missing)
            )

    candidates = output.get("prompt_candidates")
    if candidates is not None:
        if not isinstance(candidates, list):
            raise EvidenceError("prompt_candidates must be an ordered list")
        if isinstance(candidate_count, int) and len(candidates) != candidate_count:
            raise EvidenceError("prompt_candidates length differs from candidate_count")
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise EvidenceError(f"prompt candidate {index} must be a JSON object")
            candidate_number = candidate.get("candidate_number")
            token_hash = candidate.get("prompt_token_hash")
            if (
                isinstance(candidate_number, bool)
                or not isinstance(candidate_number, int)
                or candidate_number <= 0
            ):
                raise EvidenceError(
                    f"prompt candidate {index} lacks a valid candidate_number"
                )
            if not isinstance(token_hash, str) or not token_hash:
                raise EvidenceError(
                    f"prompt candidate {index} lacks a non-empty prompt_token_hash"
                )
            if not _candidate_has_score_or_feature(candidate):
                raise EvidenceError(
                    f"prompt candidate {index} lacks scores/features"
                )
    return output


def _prompt_generated_output_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract generated-token and discrete selection evidence only.

    The strict prompt comparison intentionally retains all scores and features.
    This narrower view answers whether candidate/final token sequences and the
    resulting discrete selection/ranking were identical, without treating
    timings or numerically insignificant PPL metadata as generated output.
    """

    output: dict[str, Any] = {
        "generated_tokens": _require_token_list(record, "generated_tokens"),
        "new_tokens": _require_token_list(record, "new_tokens"),
    }
    for field in PROMPT_OUTPUT_SELECTION_FIELDS:
        if field in record:
            output[field] = record[field]

    candidates = record.get("prompt_candidates")
    if candidates is not None:
        if not isinstance(candidates, list):
            raise EvidenceError("prompt_candidates must be an ordered list")
        candidate_tokens: list[dict[str, Any]] = []
        candidate_ranking: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                raise EvidenceError(f"prompt candidate {index} must be a JSON object")
            identity = {
                field: candidate[field]
                for field in (
                    "candidate_number",
                    "prompt_token_hash",
                    "generated_token_count",
                )
                if field in candidate
            }
            candidate_tokens.append(identity)

            ranking = {
                field: value
                for field, value in candidate.items()
                if "rank" in field and "score" not in field
            }
            for field in ("selected", "selection_status"):
                if field in candidate:
                    ranking[field] = candidate[field]
            candidate_ranking.append(
                {
                    "candidate_number": candidate.get("candidate_number"),
                    **ranking,
                }
            )
        output["candidate_tokens"] = candidate_tokens
        output["candidate_ranking"] = candidate_ranking
    return output


def _continuation_raw_generated_output_evidence(
    record: Mapping[str, Any], index: int
) -> dict[str, Any]:
    output = {
        "raw_token_digest": _require_continuation_field(
            record, "raw_token_digest", index
        )
    }
    raw_tokens = record.get("raw_tokens")
    if raw_tokens is not None:
        if not isinstance(raw_tokens, list) or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in raw_tokens
        ):
            raise EvidenceError(
                f"continuation generation {index} field raw_tokens must be an "
                "integer list"
            )
        output["raw_tokens"] = list(raw_tokens)
    return output


def _continuation_decoded_output_evidence(
    record: Mapping[str, Any], index: int
) -> dict[str, Any]:
    return {
        field: _require_continuation_field(record, field, index)
        for field in ("token_decode_digest", "output_event_digest", "empty_success")
    }


def _require_continuation_field(
    record: Mapping[str, Any], field: str, index: int
) -> Any:
    if field not in record or record[field] is None:
        raise EvidenceError(
            f"continuation generation {index} is missing non-null field {field}"
        )
    value = record[field]
    if field == "generation_start_tick":
        if isinstance(value, bool) or not isinstance(value, int):
            raise EvidenceError(
                f"continuation generation {index} field {field} must be an integer"
            )
    elif field == "part0_roll_shape":
        if not isinstance(value, list) or any(
            isinstance(dimension, bool) or not isinstance(dimension, int)
            for dimension in value
        ):
            raise EvidenceError(
                f"continuation generation {index} field {field} must be an integer list"
            )
    elif field == "empty_success":
        if not isinstance(value, bool):
            raise EvidenceError(
                f"continuation generation {index} field {field} must be boolean"
            )
    elif not isinstance(value, str) or not value:
        raise EvidenceError(
            f"continuation generation {index} field {field} must be a non-empty string"
        )
    return value


def _seed_locations(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    locations: list[Mapping[str, Any]] = [trace]
    for key in (
        "seed_provenance",
        "session",
        "runtime_info",
        "runtime",
        "reset_ack",
    ):
        value = trace.get(key)
        if isinstance(value, dict):
            locations.append(value)
    return locations


def _lookup(locations: Iterable[Mapping[str, Any]], names: Sequence[str]) -> Any:
    for location in locations:
        for name in names:
            if name in location:
                return location[name]
    return None


def _seed_provenance(trace: Mapping[str, Any]) -> dict[str, Any]:
    locations = _seed_locations(trace)
    explicit_values: list[tuple[str, bool]] = []
    for location in locations:
        for field in ("seed_provenance_complete", "seeded_session_active"):
            if field in location:
                value = location[field]
                if not isinstance(value, bool):
                    raise EvidenceError(f"seed provenance field {field} must be boolean")
                explicit_values.append((field, value))

    prompt_seed = _lookup(
        locations,
        ("prompt_seed", "prompt_effective_seed", "prompt_sample_seed"),
    )
    continuation_seed = _lookup(
        locations,
        (
            "continuation_seed",
            "continuation_effective_seed",
            "continuation_sample_seed",
        ),
    )
    session_id = _lookup(locations, ("active_session_id", "session_id"))
    session_epoch = _lookup(locations, ("active_session_epoch", "session_epoch"))

    if explicit_values:
        complete = all(value is True for _field, value in explicit_values)
        source = "explicit_boolean"
    else:
        epoch_valid = (
            isinstance(session_epoch, int)
            and not isinstance(session_epoch, bool)
            and session_epoch > 0
        )
        complete = (
            prompt_seed is not None
            and continuation_seed is not None
            and isinstance(session_id, str)
            and bool(session_id)
            and epoch_valid
        )
        source = "seed_and_active_session_fields"

    return {
        "complete": bool(complete),
        "source": source,
        "explicit": {field: value for field, value in explicit_values},
        "prompt_seed": prompt_seed,
        "continuation_seed": continuation_seed,
        "active_session_present": isinstance(session_id, str) and bool(session_id),
        "active_session_id": session_id,
        "active_session_epoch_valid": (
            isinstance(session_epoch, int)
            and not isinstance(session_epoch, bool)
            and session_epoch > 0
        ),
        "active_session_epoch": session_epoch,
    }


def _seed_identity(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return comparable seed values, excluding per-session identifiers."""

    return {
        key: provenance[key]
        for key in ("prompt_seed", "continuation_seed")
        if provenance.get(key) is not None
    }


def _first_difference(original: Any, replay: Any, path: str = "$") -> dict[str, Any] | None:
    if type(original) is not type(replay):
        return {
            "path": path,
            "reason": "type_mismatch",
            "original": original,
            "replay": replay,
        }
    if isinstance(original, dict):
        original_keys = set(original)
        replay_keys = set(replay)
        if original_keys != replay_keys:
            return {
                "path": path,
                "reason": "key_mismatch",
                "original_only": sorted(original_keys - replay_keys),
                "replay_only": sorted(replay_keys - original_keys),
            }
        for key in original:
            difference = _first_difference(
                original[key], replay[key], f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(original, list):
        if len(original) != len(replay):
            return {
                "path": path,
                "reason": "length_mismatch",
                "original": len(original),
                "replay": len(replay),
            }
        for index, (original_item, replay_item) in enumerate(zip(original, replay)):
            difference = _first_difference(
                original_item, replay_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if original != replay:
        return {
            "path": path,
            "reason": "value_mismatch",
            "original": original,
            "replay": replay,
        }
    return None


def _compare_component(
    component: str, original: Any, replay: Any
) -> tuple[bool, dict[str, Any] | None]:
    difference = _first_difference(original, replay)
    if difference is None:
        return True, None
    return False, {"component": component, **difference}


def _load_session(session_dir: Path) -> dict[str, Any]:
    if not session_dir.is_dir():
        raise EvidenceError(f"RuntimeSession directory does not exist: {session_dir}")

    requests = _read_jsonl(session_dir / REQUEST_TRACE_NAME)
    trace = _require_mapping(
        _read_json(session_dir / MODEL_TRACE_NAME),
        str(session_dir / MODEL_TRACE_NAME),
    )
    session_seed_document = _require_mapping(
        _read_json(session_dir / SESSION_SEED_NAME),
        str(session_dir / SESSION_SEED_NAME),
    )
    if session_seed_document.get("success") is not True:
        raise EvidenceError("session seed artifact is not a successful initialization")
    runtime_info = trace.get("runtime_info")
    if not isinstance(runtime_info, dict):
        raise EvidenceError("model trace lacks a runtime_info object")
    if runtime_info.get("trace_capture_complete") is not True:
        raise EvidenceError(
            "model trace runtime_info.trace_capture_complete must be explicitly true"
        )
    prompt = _extract_prompt_record(trace)
    prompt_evidence = {
        field: _require_token_list(prompt, field) for field in PROMPT_FIELDS
    }
    continuation = _extract_continuation_records(trace)
    continuation_input = [
        {
            field: _require_continuation_field(record, field, index)
            for field in CONTINUATION_INPUT_FIELDS
        }
        for index, record in enumerate(continuation)
    ]
    continuation_output = [
        {
            field: _require_continuation_field(record, field, index)
            for field in CONTINUATION_OUTPUT_FIELDS
        }
        for index, record in enumerate(continuation)
    ]
    continuation_raw_generated_output = [
        _continuation_raw_generated_output_evidence(record, index)
        for index, record in enumerate(continuation)
    ]
    continuation_decoded_output = [
        _continuation_decoded_output_evidence(record, index)
        for index, record in enumerate(continuation)
    ]
    trace_seed_provenance = _seed_provenance(trace)
    session_seed_provenance = _seed_provenance(session_seed_document)
    for field in (
        "prompt_requested_seed",
        "prompt_effective_seed",
        "continuation_requested_seed",
        "continuation_effective_seed",
        "session_epoch",
    ):
        value = session_seed_document.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EvidenceError(
                f"session seed artifact field {field} must be a non-negative integer"
            )
    for field in ("prompt_seed_source", "continuation_seed_source"):
        if session_seed_document.get(field) not in {"system", "requested"}:
            raise EvidenceError(
                f"session seed artifact field {field} must be system or requested"
            )
    if not trace_seed_provenance["complete"]:
        raise EvidenceError(
            "model trace lacks complete seeded-session provenance; require an "
            "explicit true seed_provenance_complete/seeded_session_active flag, "
            "or non-null prompt and continuation seeds plus an active session "
            "id and positive epoch"
        )
    if not session_seed_provenance["complete"]:
        raise EvidenceError(
            "session seed artifact lacks prompt/continuation effective seeds "
            "and active server session provenance"
        )
    if _seed_identity(trace_seed_provenance) != _seed_identity(
        session_seed_provenance
    ):
        raise EvidenceError("session seed artifact does not match model trace seeds")
    if (
        trace_seed_provenance["active_session_id"]
        != session_seed_provenance["active_session_id"]
        or trace_seed_provenance["active_session_epoch"]
        != session_seed_provenance["active_session_epoch"]
    ):
        raise EvidenceError(
            "session seed artifact does not match model trace session id/epoch"
        )
    session_seed_provenance.update(
        {
            field: session_seed_document[field]
            for field in (
                "prompt_requested_seed",
                "prompt_effective_seed",
                "continuation_requested_seed",
                "continuation_effective_seed",
                "prompt_seed_source",
                "continuation_seed_source",
            )
        }
    )

    return {
        "protocol_requests": [
            _protocol_request_evidence(request, index)
            for index, request in enumerate(requests)
        ],
        "prompt_input": {"prompt_tokens": prompt_evidence["prompt_tokens"]},
        "prompt_output": _prompt_output_evidence(prompt),
        "prompt_generated_output": _prompt_generated_output_evidence(prompt),
        "continuation_input": continuation_input,
        "continuation_output": continuation_output,
        "continuation_raw_generated_output": continuation_raw_generated_output,
        "continuation_decoded_output": continuation_decoded_output,
        "seed_provenance": session_seed_provenance,
        "trace_capture_complete": True,
        "manifest_present": (session_dir / MANIFEST_NAME).is_file(),
    }


def _invalid_result(
    original_dir: Path, replay_dir: Path, errors: list[str]
) -> dict[str, Any]:
    return {
        "schema_version": "streammuse.realtime_replay_exact_comparison.v1",
        "original": str(original_dir.resolve()),
        "replay": str(replay_dir.resolve()),
        "protocol_request_exact": False,
        "prompt_input_exact": False,
        "prompt_output_exact": False,
        "continuation_input_exact": False,
        "continuation_output_exact": False,
        "prompt_generated_token_sequence_exact": False,
        "continuation_raw_generated_tokens_exact": False,
        "continuation_decoded_output_events_exact": False,
        "inference_output_exact": False,
        "seed_provenance_complete": False,
        "trace_capture_complete": False,
        "seed_provenance_exact": False,
        "model_exact": False,
        "comparable": False,
        "first_mismatch": {
            "component": "evidence",
            "reason": "missing_or_invalid_evidence",
            "detail": errors[0],
        },
        "strict_differences": {
            "evidence": {
                "component": "evidence",
                "reason": "missing_or_invalid_evidence",
                "detail": errors[0],
            }
        },
        "errors": errors,
    }


def compare_session_directories(
    original: str | Path, replay: str | Path
) -> dict[str, Any]:
    original_dir = Path(original)
    replay_dir = Path(replay)
    loaded: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for label, session_dir in (("original", original_dir), ("replay", replay_dir)):
        try:
            loaded[label] = _load_session(session_dir)
        except EvidenceError as exc:
            errors.append(f"{label}: {exc}")
    if errors:
        return _invalid_result(original_dir, replay_dir, errors)

    original_evidence = loaded["original"]
    replay_evidence = loaded["replay"]
    components = (
        (
            "protocol_request",
            "protocol_request_exact",
            original_evidence["protocol_requests"],
            replay_evidence["protocol_requests"],
        ),
        (
            "prompt_input",
            "prompt_input_exact",
            original_evidence["prompt_input"],
            replay_evidence["prompt_input"],
        ),
        (
            "prompt_output",
            "prompt_output_exact",
            original_evidence["prompt_output"],
            replay_evidence["prompt_output"],
        ),
        (
            "continuation_input",
            "continuation_input_exact",
            original_evidence["continuation_input"],
            replay_evidence["continuation_input"],
        ),
        (
            "continuation_output",
            "continuation_output_exact",
            original_evidence["continuation_output"],
            replay_evidence["continuation_output"],
        ),
    )

    result: dict[str, Any] = {
        "schema_version": "streammuse.realtime_replay_exact_comparison.v1",
        "original": str(original_dir.resolve()),
        "replay": str(replay_dir.resolve()),
        "seed_provenance_complete": True,
        "trace_capture_complete": True,
        "comparable": True,
        "first_mismatch": None,
        "strict_differences": {},
        "errors": [],
        "evidence": {
            "original_protocol_request_count": len(
                original_evidence["protocol_requests"]
            ),
            "replay_protocol_request_count": len(replay_evidence["protocol_requests"]),
            "original_continuation_generation_count": len(
                original_evidence["continuation_input"]
            ),
            "replay_continuation_generation_count": len(
                replay_evidence["continuation_input"]
            ),
            "original_manifest_present": original_evidence["manifest_present"],
            "replay_manifest_present": replay_evidence["manifest_present"],
            "original_seed_provenance": original_evidence["seed_provenance"],
            "replay_seed_provenance": replay_evidence["seed_provenance"],
        },
    }

    seed_original = _seed_identity(original_evidence["seed_provenance"])
    seed_replay = _seed_identity(replay_evidence["seed_provenance"])
    if seed_original and seed_replay:
        seed_exact, seed_difference = _compare_component(
            "seed_provenance", seed_original, seed_replay
        )
    else:
        # Explicit server provenance booleans are authoritative when the schema
        # intentionally does not expose seed values.
        seed_exact, seed_difference = True, None
    result["seed_provenance_exact"] = seed_exact
    if seed_difference is not None:
        result["first_mismatch"] = seed_difference
        result["strict_differences"]["seed_provenance"] = seed_difference

    for component, result_field, original_value, replay_value in components:
        exact, difference = _compare_component(
            component, original_value, replay_value
        )
        result[result_field] = exact
        if result["first_mismatch"] is None and difference is not None:
            result["first_mismatch"] = difference
        if difference is not None:
            result["strict_differences"][component] = difference

    output_components = (
        (
            "prompt_generated_token_sequence_exact",
            original_evidence["prompt_generated_output"],
            replay_evidence["prompt_generated_output"],
        ),
        (
            "continuation_raw_generated_tokens_exact",
            original_evidence["continuation_raw_generated_output"],
            replay_evidence["continuation_raw_generated_output"],
        ),
        (
            "continuation_decoded_output_events_exact",
            original_evidence["continuation_decoded_output"],
            replay_evidence["continuation_decoded_output"],
        ),
    )
    for result_field, original_value, replay_value in output_components:
        exact, _difference = _compare_component(
            result_field, original_value, replay_value
        )
        result[result_field] = exact
    result["inference_output_exact"] = all(
        result[result_field]
        for result_field, _original_value, _replay_value in output_components
    )

    result["model_exact"] = bool(
        result["seed_provenance_exact"]
        and result["protocol_request_exact"]
        and result["prompt_input_exact"]
        and result["prompt_output_exact"]
        and result["continuation_input_exact"]
        and result["continuation_output_exact"]
    )
    return result


def exit_code(result: Mapping[str, Any]) -> int:
    if not result.get("comparable"):
        return 2
    return 0 if result.get("model_exact") is True else 1


def _write_output(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Strictly compare an original human-live RuntimeSession with a "
            "realtime MIDI-file replay"
        )
    )
    parser.add_argument("original", type=Path, help="original human-live session")
    parser.add_argument("replay", type=Path, help="realtime MIDI-file replay session")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args(argv)

    result = compare_session_directories(args.original, args.replay)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        try:
            _write_output(args.output, result)
        except OSError as exc:
            print(f"cannot write output report {args.output}: {exc}", file=sys.stderr)
            return 2
    return exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
