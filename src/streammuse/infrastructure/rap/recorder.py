"""Crash-tolerant recorder and deterministic derivations for rap research runs."""

from __future__ import annotations

import csv
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from streammuse.domain.rap import RapEvent, RapEventType, normalize_text


_SECRET_NAME = re.compile(r"[^a-z0-9]+")
_NON_SECRET_DIAGNOSTIC_KEYS = {"prompt_tokens", "completion_tokens"}
DEFAULT_REPETITION_WINDOW_BARS = 4
_REQUIRED_MANIFEST_FIELDS = (
    "scenario_id",
    "scenario",
    "seed",
    "tempo",
    "templates",
    "generator_config",
    "model_config",
    "score_weights",
    "minimum_score",
    "timeout_seconds",
    "lookahead_bars",
    "python_version",
    "platform",
    "package_version",
    "git_revision",
    "git_dirty",
    "repetition_window_bars",
)
_BAR_FIELDS = (
    "bar",
    "request_id",
    "source",
    "text",
    "frozen",
    "fallback",
    "fallback_reason",
    "candidate_count",
    "valid_candidate_count",
    "selected_candidate_id",
    "generation_latency_ms",
    "deadline_slack_ms",
    "generator_error",
    "emitted_syllables",
    "mean_syllable_observation_delay_ms",
)


class RapSessionRecorder:
    """Append canonical events first, then derive artifacts after a clean close."""

    def __init__(self, session_dir: Path, manifest: dict[str, Any]) -> None:
        validated_manifest = validate_session_manifest(manifest)
        session_dir.mkdir(parents=True, exist_ok=False)
        self._events_path = session_dir / "events.jsonl"
        self._repetition_window_bars = validated_manifest["repetition_window_bars"]
        write_json(session_dir / "session.json", redact_manifest(validated_manifest))
        self._stream = self._events_path.open("a", encoding="utf-8", buffering=1)
        self._closed = False

    def __call__(self, event: RapEvent) -> None:
        if self._closed:
            raise RuntimeError("rap session recorder is closed")
        self._stream.write(json.dumps(event_to_dict(event), sort_keys=True) + "\n")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stream.close()
        events = read_events(self._events_path)
        write_json(
            self._events_path.parent / "summary.json",
            derive_summary(events, expected_manifest_window=self._repetition_window_bars),
        )
        write_bar_csv(self._events_path.parent / "bars.csv", derive_bar_rows(events))


def event_to_dict(event: RapEvent) -> dict[str, Any]:
    """Convert a frozen domain event into JSON-safe primitive data."""

    return {
        "session_id": event.session_id,
        "sequence": event.sequence,
        "event_type": event.event_type.value,
        "utc_time": event.utc_time,
        "monotonic_ns": event.monotonic_ns,
        "bar": event.bar,
        "tick": event.tick,
        "request_id": event.request_id,
        "payload": redact_manifest(event.payload),
    }


def read_events(path: Path) -> list[RapEvent]:
    """Recover complete canonical records while rejecting evidence corruption."""

    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    events: list[RapEvent] = []
    expected_sequence = 1
    session_id: str | None = None
    for index, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            raise ValueError(f"invalid event JSONL at line {index}: empty record")
        try:
            record = json.loads(raw_line)
            event = _event_from_record(record)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            incomplete_final = (
                index == len(lines)
                and not raw_line.endswith(("\n", "\r"))
                and isinstance(error, json.JSONDecodeError)
                and _is_recoverable_eof_json_error(raw_line, error)
            )
            if incomplete_final:
                break
            raise ValueError(f"invalid event JSONL at line {index}: {error}") from error
        if event.sequence != expected_sequence:
            raise ValueError(f"invalid event sequence at line {index}: expected {expected_sequence}, got {event.sequence}")
        if session_id is None:
            session_id = event.session_id
        elif event.session_id != session_id:
            raise ValueError(f"mixed session IDs at line {index}")
        events.append(event)
        expected_sequence += 1
    return events


def redact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact values associated with secret-looking manifest keys."""

    def redact(value: Any, key: str | None = None) -> Any:
        if key in _NON_SECRET_DIAGNOSTIC_KEYS and (
            value is None or isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ):
            return value
        normalized = _SECRET_NAME.sub("", key.lower()) if key else ""
        if any(token in normalized for token in ("key", "token", "secret", "authorization")):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {str(child_key): redact(child_value, str(child_key)) for child_key, child_value in value.items()}
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, tuple):
            return [redact(item) for item in value]
        return deepcopy(value)

    return redact(manifest)


@dataclass(frozen=True)
class RapSessionManifest:
    """Complete reproducibility contract for a recorded rap session."""

    scenario_id: str
    scenario: dict[str, Any]
    seed: int
    tempo: dict[str, Any]
    templates: list[dict[str, Any]]
    generator_config: dict[str, Any]
    model_config: dict[str, Any]
    score_weights: dict[str, Any]
    minimum_score: float
    timeout_seconds: float
    lookahead_bars: int
    python_version: str
    platform: str
    package_version: str
    git_revision: str
    git_dirty: bool
    repetition_window_bars: int = DEFAULT_REPETITION_WINDOW_BARS

    def to_dict(self) -> dict[str, Any]:
        return {
            field: deepcopy(getattr(self, field))
            for field in _REQUIRED_MANIFEST_FIELDS
        }


def build_session_manifest(
    *,
    scenario_id: str,
    scenario: dict[str, Any],
    seed: int,
    tempo: dict[str, Any],
    templates: list[dict[str, Any]],
    generator_config: dict[str, Any],
    model_config: dict[str, Any],
    score_weights: dict[str, Any],
    minimum_score: float,
    timeout_seconds: float,
    lookahead_bars: int,
    python_version: str,
    platform: str,
    package_version: str,
    git_revision: str,
    git_dirty: bool,
    repetition_window_bars: int = DEFAULT_REPETITION_WINDOW_BARS,
) -> dict[str, Any]:
    """Build and validate a complete JSON-ready research manifest."""

    return validate_session_manifest(
        RapSessionManifest(
            scenario_id=scenario_id,
            scenario=scenario,
            seed=seed,
            tempo=tempo,
            templates=templates,
            generator_config=generator_config,
            model_config=model_config,
            score_weights=score_weights,
            minimum_score=minimum_score,
            timeout_seconds=timeout_seconds,
            lookahead_bars=lookahead_bars,
            python_version=python_version,
            platform=platform,
            package_version=package_version,
            git_revision=git_revision,
            git_dirty=git_dirty,
            repetition_window_bars=repetition_window_bars,
        ).to_dict()
    )


def validate_session_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete or non-reproducible session metadata before recording."""

    missing = [field for field in _REQUIRED_MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError(f"session manifest missing required field: {missing[0]}")
    _require_string(manifest, "scenario_id")
    _require_int(manifest, "seed")
    if manifest["seed"] < 0:
        raise ValueError("session manifest field seed must be nonnegative")
    _validate_tempo(manifest["tempo"])
    template_ids = _validate_templates(manifest["templates"], manifest["tempo"])
    _validate_scenario(manifest["scenario"], manifest["scenario_id"], manifest["tempo"]["bpm"], template_ids)
    _validate_identity_config(manifest["generator_config"], "generator_config")
    _validate_identity_config(manifest["model_config"], "model_config")
    _validate_score_weights(manifest["score_weights"])
    _require_finite_number(manifest, "minimum_score")
    if not 0.0 <= manifest["minimum_score"] <= 1.0:
        raise ValueError("session manifest field minimum_score must be between zero and one")
    _require_finite_number(manifest, "timeout_seconds")
    if manifest["timeout_seconds"] <= 0:
        raise ValueError("session manifest field timeout_seconds must be positive")
    for field in ("lookahead_bars", "repetition_window_bars"):
        _require_int(manifest, field)
        if manifest[field] <= 0:
            raise ValueError(f"session manifest field {field} must be positive")
    for field in ("python_version", "platform", "package_version", "git_revision"):
        _require_string(manifest, field)
    if not isinstance(manifest["git_dirty"], bool):
        raise ValueError("session manifest field git_dirty must be a bool")
    try:
        json.dumps(manifest, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"session manifest must be strict JSON serializable: {error}") from error
    return deepcopy(manifest)


def derive_summary(
    events: Iterable[RapEvent],
    *,
    expected_manifest_window: int | None = None,
) -> dict[str, Any]:
    """Derive research metrics from event evidence without hidden runtime state."""

    history = list(events)
    event_list, epoch = _current_epoch(history)
    event_list = _current_epoch_coordinator_events(event_list, epoch)
    frozen = _frozen_bars(event_list)
    plans = _requests(event_list, RapEventType.BAR_PLANNING_STARTED)
    batches = _requests(event_list, RapEventType.CANDIDATE_BATCH_RECEIVED)
    evaluations = _candidate_evaluations(event_list)

    candidate_validity, parsed_candidates = _candidate_counts(batches, evaluations)
    fallback_count = sum(event.payload.get("fallback") is True for event in frozen.values())
    deadline_misses = sum(request_id in plans and event.payload.get("late") is True for request_id, event in batches.items())
    generator_errors = _generator_error_requests(event_list, plans, batches)
    pronunciation_fallbacks, pronunciation_total = _pronunciation_counts(evaluations.values())
    repetitions, generated_bigrams = _repetition_counts(event_list, frozen, expected_manifest_window)
    audio_events = tuple(event for event in event_list if event.event_type == RapEventType.AUDIO_RENDER_COMPLETED)
    committed_audio = tuple(event for event in event_list if event.event_type == RapEventType.BAR_AUDIO_COMMITTED)
    completed_audio_bars = {
        event.bar for event in event_list if event.event_type == RapEventType.BAR_PLAYBACK_COMPLETED and event.bar is not None
    }
    committed_frames = {
        event.bar: int(event.payload["frame_count"])
        for event in committed_audio
        if event.bar is not None and isinstance(event.payload.get("frame_count"), int) and event.payload["frame_count"] >= 0
    }

    return {
        "events": {"count": len(event_list), "history_count": len(history), "epoch": epoch},
        "bars": {
            "frozen": len(frozen),
            "fallback": fallback_count,
            "fallback_rate": _ratio(fallback_count, len(frozen))["rate"],
        },
        "metrics": {
            "candidate_validity": _ratio(candidate_validity, parsed_candidates),
            "fallback": _ratio(fallback_count, len(frozen)),
            "deadline_miss": _ratio(deadline_misses, len(plans)),
            "generator_error": _ratio(len(generator_errors), len(plans)),
            "pronunciation_fallback": _ratio(pronunciation_fallbacks, pronunciation_total),
            "repetition": _ratio(repetitions, generated_bigrams),
        },
        "latencies": {
            "generation_latency_ms": _distribution(_payload_numbers(batches.values(), "latency_ms")),
            "deadline_slack_ms": _distribution(_payload_numbers(batches.values(), "deadline_slack_ms")),
            "syllable_observation_delay_ms": _distribution(
                _payload_numbers(
                    (event for event in event_list if event.event_type == RapEventType.SYLLABLE_EMITTED),
                    "observation_delay_ms",
                )
            ),
            "synthesis_latency_ms": _distribution(_payload_numbers(audio_events, "synthesis_latency_ms")),
            "bar_render_latency_ms": _distribution(_payload_numbers(audio_events, "render_latency_ms")),
            "audio_commit_slack_ms": _distribution(_payload_numbers(committed_audio, "deadline_slack_ms")),
        },
        "generation_diagnostics": {
            "prompt_tokens": _token_usage(batches.values(), "prompt_tokens"),
            "completion_tokens": _token_usage(batches.values(), "completion_tokens"),
            "warnings": {
                "count": sum(isinstance(event.payload.get("warning"), str) and bool(event.payload["warning"]) for event in batches.values())
            },
        },
        "audio": {
            "render_latency_ms": _distribution(_payload_numbers(audio_events, "render_latency_ms")),
            "commit_slack_ms": _distribution(_payload_numbers(committed_audio, "deadline_slack_ms")),
            "warning_counts": {
                "pronunciation_fallback": sum(
                    event.event_type == RapEventType.PRONUNCIATION_FALLBACK for event in event_list
                ),
                "timing_pressure": sum(event.event_type == RapEventType.TIMING_PRESSURE for event in event_list),
                "forced_bar_fit": sum(event.event_type == RapEventType.FORCED_BAR_FIT for event in event_list),
                "synthesis_failed": sum(event.event_type == RapEventType.SYNTHESIS_FAILED for event in event_list),
            },
            "underruns": sum(event.event_type == RapEventType.AUDIO_UNDERRUN for event in event_list),
            "completed_bars": len(completed_audio_bars),
            "completed_frames": sum(committed_frames.get(bar, 0) for bar in completed_audio_bars),
        },
    }


def _current_epoch(events: list[RapEvent]) -> tuple[list[RapEvent], int]:
    reset_index = -1
    epoch = 0
    for index, event in enumerate(events):
        if event.event_type == RapEventType.SESSION_RESET:
            reset_index = index
            epoch += 1
    return events[reset_index + 1 :], epoch


def _current_epoch_coordinator_events(events: list[RapEvent], epoch: int) -> list[RapEvent]:
    """Exclude late audio-worker events that belong to an earlier reset epoch."""

    active: list[RapEvent] = []
    for event in events:
        coordinator_epoch = event.payload.get("coordinator_epoch")
        if isinstance(coordinator_epoch, int) and not isinstance(coordinator_epoch, bool):
            if coordinator_epoch != epoch:
                continue
        active.append(event)
    return active


def derive_bar_rows(events: Iterable[RapEvent]) -> list[dict[str, Any]]:
    """Derive deterministic, one-row-per-bar CSV-ready research evidence."""

    history = list(events)
    event_list, epoch = _current_epoch(history)
    event_list = _current_epoch_coordinator_events(event_list, epoch)
    frozen = _frozen_bars(event_list)
    bars: dict[int, dict[str, Any]] = {}
    seen_batches: set[str] = set()
    seen_candidates: set[tuple[str, str]] = set()
    for event in event_list:
        if event.bar is None or event.bar not in frozen:
            continue
        row = bars.setdefault(event.bar, _empty_bar_row(event.bar))
        if event.request_id is not None:
            row["request_id"] = event.request_id
        payload = event.payload
        if event.event_type == RapEventType.CANDIDATE_BATCH_RECEIVED:
            if not isinstance(event.request_id, str) or event.request_id in seen_batches:
                continue
            seen_batches.add(event.request_id)
            row["generation_latency_ms"] = _number_or_none(payload.get("latency_ms"))
            row["deadline_slack_ms"] = _number_or_none(payload.get("deadline_slack_ms"))
            row["generator_error"] = payload.get("error_type") if isinstance(payload.get("error_type"), str) else None
            supplied_count = _candidate_count(payload)
            if supplied_count is not None:
                row["_declared_candidate_count"] = supplied_count
        elif event.event_type == RapEventType.CANDIDATE_EVALUATED:
            candidate_id = payload.get("candidate_id")
            if not isinstance(event.request_id, str) or not isinstance(candidate_id, str):
                continue
            identity = (event.request_id, candidate_id)
            if identity in seen_candidates:
                continue
            seen_candidates.add(identity)
            row["_evaluated_candidate_count"] += 1
            if payload.get("valid") is True:
                row["valid_candidate_count"] += 1
            if payload.get("selected") is True and isinstance(payload.get("candidate_id"), str):
                row["selected_candidate_id"] = payload["candidate_id"]
        elif event.event_type == RapEventType.BAR_FROZEN:
            row.update(
                {
                    "source": payload.get("source") if isinstance(payload.get("source"), str) else None,
                    "text": payload.get("text") if isinstance(payload.get("text"), str) else None,
                    "frozen": True,
                    "fallback": payload.get("fallback") is True,
                    "fallback_reason": payload.get("fallback_reason") if isinstance(payload.get("fallback_reason"), str) else None,
                }
            )
        elif event.event_type == RapEventType.GENERATION_FAILED:
            error_type = payload.get("error_type")
            if isinstance(error_type, str) and error_type:
                row["generator_error"] = error_type
        elif event.event_type == RapEventType.SYLLABLE_EMITTED:
            observation_delay = _number_or_none(payload.get("observation_delay_ms"))
            if observation_delay is not None:
                row["_observation_delays"].append(observation_delay)
            row["emitted_syllables"] += 1

    result: list[dict[str, Any]] = []
    for bar in sorted(bars):
        row = bars[bar]
        observation_delays = row.pop("_observation_delays")
        declared_count = row.pop("_declared_candidate_count")
        evaluated_count = row.pop("_evaluated_candidate_count")
        if declared_count is not None and evaluated_count > declared_count:
            raise ValueError(f"candidate count contradiction for bar {bar}")
        row["candidate_count"] = declared_count if declared_count is not None else evaluated_count
        row["mean_syllable_observation_delay_ms"] = (
            sum(observation_delays) / len(observation_delays)
            if observation_delays
            else None
        )
        result.append(row)
    return result


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_bar_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_BAR_FIELDS, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _ratio(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {"numerator": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else None}


def _event_from_record(record: object) -> RapEvent:
    if not isinstance(record, dict):
        raise ValueError("event record must be an object")
    session_id = record.get("session_id")
    sequence = record.get("sequence")
    utc_time = record.get("utc_time")
    monotonic = record.get("monotonic_ns")
    bar = record.get("bar")
    tick = record.get("tick")
    request_id = record.get("request_id")
    payload = record.get("payload")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("event session_id must be a non-empty string")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence <= 0:
        raise ValueError("event sequence must be a positive integer")
    if not isinstance(utc_time, str) or not isinstance(monotonic, int) or isinstance(monotonic, bool):
        raise ValueError("event timestamps are invalid")
    if bar is not None and (not isinstance(bar, int) or isinstance(bar, bool)):
        raise ValueError("event bar is invalid")
    if tick is not None and (not isinstance(tick, int) or isinstance(tick, bool)):
        raise ValueError("event tick is invalid")
    if request_id is not None and not isinstance(request_id, str):
        raise ValueError("event request_id is invalid")
    if not isinstance(payload, dict):
        raise ValueError("event payload must be an object")
    return RapEvent(
        session_id=session_id,
        sequence=sequence,
        event_type=RapEventType(record["event_type"]),
        utc_time=utc_time,
        monotonic_ns=monotonic,
        bar=bar,
        tick=tick,
        request_id=request_id,
        payload=payload,
    )


class _IncompleteJsonPrefix(Exception):
    """The parsed input ended in a JSON grammar state that can be extended."""


class _InvalidJsonPrefix(Exception):
    """The parsed input cannot be extended into a valid JSON document."""


class _JsonPrefixParser:
    _WHITESPACE = " \t\r\n"
    _DIGITS = frozenset("0123456789")
    _HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
    _SIMPLE_ESCAPES = frozenset('"\\/bfnrt')

    def __init__(self, value: str) -> None:
        self._value = value
        self._position = 0

    def parse(self) -> None:
        self._skip_whitespace()
        if self._at_end():
            raise _IncompleteJsonPrefix
        self._parse_value()
        self._skip_whitespace()
        if not self._at_end():
            raise _InvalidJsonPrefix

    def _parse_value(self) -> None:
        if self._at_end():
            raise _IncompleteJsonPrefix
        character = self._peek()
        if character == '"':
            self._parse_string()
        elif character == "{":
            self._parse_object()
        elif character == "[":
            self._parse_array()
        elif character == "t":
            self._parse_literal("true")
        elif character == "f":
            self._parse_literal("false")
        elif character == "n":
            self._parse_literal("null")
        elif character == "-" or character in self._DIGITS:
            self._parse_number()
        else:
            raise _InvalidJsonPrefix

    def _parse_object(self) -> None:
        self._position += 1
        self._skip_whitespace()
        if self._at_end():
            raise _IncompleteJsonPrefix
        if self._peek() == "}":
            self._position += 1
            return

        while True:
            if self._peek() != '"':
                raise _InvalidJsonPrefix
            self._parse_string()
            self._skip_whitespace()
            self._consume_required(":")
            self._skip_whitespace()
            self._parse_value()
            self._skip_whitespace()
            if self._at_end():
                raise _IncompleteJsonPrefix
            character = self._peek()
            if character == "}":
                self._position += 1
                return
            if character != ",":
                raise _InvalidJsonPrefix
            self._position += 1
            self._skip_whitespace()
            if self._at_end():
                raise _IncompleteJsonPrefix

    def _parse_array(self) -> None:
        self._position += 1
        self._skip_whitespace()
        if self._at_end():
            raise _IncompleteJsonPrefix
        if self._peek() == "]":
            self._position += 1
            return

        while True:
            self._parse_value()
            self._skip_whitespace()
            if self._at_end():
                raise _IncompleteJsonPrefix
            character = self._peek()
            if character == "]":
                self._position += 1
                return
            if character != ",":
                raise _InvalidJsonPrefix
            self._position += 1
            self._skip_whitespace()
            if self._at_end():
                raise _IncompleteJsonPrefix

    def _parse_string(self) -> None:
        self._position += 1
        while True:
            if self._at_end():
                raise _IncompleteJsonPrefix
            character = self._peek()
            if character == '"':
                self._position += 1
                return
            if character == "\\":
                self._position += 1
                self._parse_escape()
                continue
            if ord(character) < 0x20:
                raise _InvalidJsonPrefix
            self._position += 1

    def _parse_escape(self) -> None:
        if self._at_end():
            raise _IncompleteJsonPrefix
        character = self._peek()
        if character in self._SIMPLE_ESCAPES:
            self._position += 1
            return
        if character != "u":
            raise _InvalidJsonPrefix
        self._position += 1
        for _ in range(4):
            if self._at_end():
                raise _IncompleteJsonPrefix
            if self._peek() not in self._HEX_DIGITS:
                raise _InvalidJsonPrefix
            self._position += 1

    def _parse_literal(self, literal: str) -> None:
        for expected in literal:
            if self._at_end():
                raise _IncompleteJsonPrefix
            if self._peek() != expected:
                raise _InvalidJsonPrefix
            self._position += 1

    def _parse_number(self) -> None:
        if self._peek() == "-":
            self._position += 1
            if self._at_end():
                raise _IncompleteJsonPrefix

        if self._peek() == "0":
            self._position += 1
            if not self._at_end() and self._peek() in self._DIGITS:
                raise _InvalidJsonPrefix
        elif self._peek() in "123456789":
            self._consume_digits()
        else:
            raise _InvalidJsonPrefix

        if not self._at_end() and self._peek() == ".":
            self._position += 1
            self._require_digit()
            self._consume_digits()

        if not self._at_end() and self._peek() in "eE":
            self._position += 1
            if self._at_end():
                raise _IncompleteJsonPrefix
            if self._peek() in "+-":
                self._position += 1
                if self._at_end():
                    raise _IncompleteJsonPrefix
            self._require_digit()
            self._consume_digits()

    def _require_digit(self) -> None:
        if self._at_end():
            raise _IncompleteJsonPrefix
        if self._peek() not in self._DIGITS:
            raise _InvalidJsonPrefix

    def _consume_digits(self) -> None:
        while not self._at_end() and self._peek() in self._DIGITS:
            self._position += 1

    def _consume_required(self, character: str) -> None:
        if self._at_end():
            raise _IncompleteJsonPrefix
        if self._peek() != character:
            raise _InvalidJsonPrefix
        self._position += 1

    def _skip_whitespace(self) -> None:
        while not self._at_end() and self._peek() in self._WHITESPACE:
            self._position += 1

    def _peek(self) -> str:
        return self._value[self._position]

    def _at_end(self) -> bool:
        return self._position == len(self._value)


def _is_json_prefix(value: str) -> bool:
    """Return whether the entire input can be extended into valid JSON."""

    try:
        _JsonPrefixParser(value).parse()
    except _IncompleteJsonPrefix:
        return True
    except _InvalidJsonPrefix:
        return False
    return True


def _is_recoverable_eof_json_error(value: str, _error: json.JSONDecodeError) -> bool:
    """Recover a failed final record only when it is a valid JSON prefix."""

    return _is_json_prefix(value)


def _require_string(manifest: dict[str, Any], field: str) -> None:
    if not isinstance(manifest[field], str) or not manifest[field]:
        raise ValueError(f"session manifest field {field} must be a non-empty string")


def _require_int(manifest: dict[str, Any], field: str) -> None:
    if not isinstance(manifest[field], int) or isinstance(manifest[field], bool):
        raise ValueError(f"session manifest field {field} must be an integer")


def _require_finite_number(manifest: dict[str, Any], field: str) -> None:
    value = manifest[field]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"session manifest field {field} must be finite")


def _validate_tempo(tempo: object) -> None:
    if not isinstance(tempo, dict):
        raise ValueError("session manifest field tempo must be an object")
    for field in ("bpm", "ticks_per_beat", "beats_per_bar"):
        if field not in tempo:
            raise ValueError(f"session manifest tempo missing {field}")
    bpm = tempo["bpm"]
    if not isinstance(bpm, (int, float)) or isinstance(bpm, bool) or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("session manifest tempo bpm must be finite and positive")
    for field in ("ticks_per_beat", "beats_per_bar"):
        if not isinstance(tempo[field], int) or isinstance(tempo[field], bool) or tempo[field] <= 0:
            raise ValueError(f"session manifest tempo {field} must be a positive integer")


def _validate_scenario(scenario: object, scenario_id: str, tempo_bpm: object, template_ids: set[str]) -> None:
    if not isinstance(scenario, dict):
        raise ValueError("session manifest scenario must be an object")
    if scenario.get("scenario_id") != scenario_id or not isinstance(scenario.get("loop"), bool):
        raise ValueError("session manifest scenario ID or loop is invalid")
    bpm = scenario.get("tempo_bpm")
    if not isinstance(bpm, (int, float)) or isinstance(bpm, bool) or not math.isfinite(bpm) or bpm <= 0 or bpm != tempo_bpm:
        raise ValueError("session manifest scenario tempo_bpm is invalid")
    segments = scenario.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("session manifest scenario segments must be a non-empty list")
    expected_start = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("session manifest scenario segment must be an object")
        if segment.get("start_bar") != expected_start or not isinstance(segment.get("bars"), int) or isinstance(segment["bars"], bool) or segment["bars"] <= 0:
            raise ValueError("session manifest scenario segment start_bar/bars is invalid")
        for field in ("topic", "template_id"):
            if not isinstance(segment.get(field), str) or not segment[field]:
                raise ValueError(f"session manifest scenario segment {field} is invalid")
        if segment["template_id"] not in template_ids:
            raise ValueError("session manifest scenario segment template_id does not resolve to a recorded template")
        fallback_lines = segment.get("fallback_lines")
        if not isinstance(fallback_lines, list) or not fallback_lines or not all(isinstance(line, str) and line for line in fallback_lines):
            raise ValueError("session manifest scenario segment fallback_lines is invalid")
        expected_start += segment["bars"]


def _validate_templates(templates: object, tempo: dict[str, Any]) -> set[str]:
    if not isinstance(templates, list) or not templates:
        raise ValueError("session manifest field templates must be a non-empty list")
    seen_ids: set[str] = set()
    for template in templates:
        if not isinstance(template, dict):
            raise ValueError("session manifest template must be an object")
        template_id = template.get("template_id")
        if not isinstance(template_id, str) or not template_id or template_id in seen_ids:
            raise ValueError("session manifest template_id is invalid")
        seen_ids.add(template_id)
        if not isinstance(template.get("name"), str) or not template["name"]:
            raise ValueError("session manifest template name is invalid")
        definition = template.get("definition")
        provenance = template.get("provenance")
        if not isinstance(definition, dict) or not definition or not isinstance(provenance, dict) or not provenance:
            raise ValueError("session manifest template definition/provenance is invalid")
        for field in ("ticks_per_beat", "beats_per_bar"):
            if not isinstance(definition.get(field), int) or isinstance(definition[field], bool) or definition[field] <= 0:
                raise ValueError(f"session manifest template {field} is invalid")
            if definition[field] != tempo[field]:
                raise ValueError("session manifest template meter must match session tempo")
        slots = definition.get("slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError("session manifest template slots are invalid")
        previous_tick = -1
        ticks_per_bar = definition["ticks_per_beat"] * definition["beats_per_bar"]
        for slot in slots:
            if not isinstance(slot, dict):
                raise ValueError("session manifest template slot is invalid")
            required_slot_fields = {"tick_in_bar", "duration_ticks", "target_stress", "boundary_strength", "rhyme_group"}
            missing_fields = required_slot_fields - slot.keys()
            if missing_fields:
                missing_field = sorted(missing_fields)[0]
                raise ValueError(
                    f"session manifest template slot missing {missing_field}"
                )
            tick = slot.get("tick_in_bar")
            duration = slot.get("duration_ticks")
            stress = slot.get("target_stress")
            boundary_strength = slot.get("boundary_strength")
            rhyme_group = slot.get("rhyme_group")
            if not isinstance(tick, int) or isinstance(tick, bool) or tick <= previous_tick or tick < 0 or tick >= ticks_per_bar:
                raise ValueError("session manifest template slot tick is invalid")
            if not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0:
                raise ValueError("session manifest template slot duration is invalid")
            if not isinstance(stress, (int, float)) or isinstance(stress, bool) or not math.isfinite(stress) or not 0 <= stress <= 1:
                raise ValueError("session manifest template slot target_stress is invalid")
            if not isinstance(boundary_strength, int) or isinstance(boundary_strength, bool) or not 0 <= boundary_strength <= 5:
                raise ValueError("session manifest template slot boundary_strength is invalid")
            if rhyme_group is not None and (not isinstance(rhyme_group, str) or not rhyme_group):
                raise ValueError("session manifest template slot rhyme_group is invalid")
            previous_tick = tick
        for field in ("kind", "source"):
            if not isinstance(provenance.get(field), str) or not provenance[field]:
                raise ValueError(f"session manifest template provenance {field} is invalid")
        if "source_hash" not in provenance or "quantization_error_ticks" not in provenance:
            raise ValueError("session manifest template provenance fields are incomplete")
        source_hash = provenance["source_hash"]
        if source_hash is not None and (not isinstance(source_hash, str) or not source_hash):
            raise ValueError("session manifest template provenance source_hash is invalid")
        quantization_error = provenance["quantization_error_ticks"]
        if (
            not isinstance(quantization_error, (int, float))
            or isinstance(quantization_error, bool)
            or not math.isfinite(quantization_error)
            or quantization_error < 0
        ):
            raise ValueError("session manifest template provenance quantization_error_ticks is invalid")
    return seen_ids


def _validate_identity_config(config: object, field: str) -> None:
    if not isinstance(config, dict) or not config:
        raise ValueError(f"session manifest field {field} must be a non-empty object")
    identity = config.get("identity", config.get("name"))
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"session manifest field {field} requires a non-empty identity")


def _validate_score_weights(weights: object) -> None:
    if not isinstance(weights, dict) or not weights:
        raise ValueError("session manifest field score_weights must be a non-empty object")
    required_names = {
        "stress_alignment",
        "boundary_fit",
        "rhyme_quality",
        "topic_coverage",
        "lexical_continuity",
        "novelty",
    }
    if set(weights) != required_names:
        raise ValueError("session manifest field score_weights must contain exactly the ScoreWeights components")
    for name, value in weights.items():
        if not isinstance(name, str) or not name or not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("session manifest field score_weights must contain finite nonnegative named weights")
    if abs(math.fsum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("session manifest field score_weights must sum to one")


def _requests(events: Iterable[RapEvent], event_type: RapEventType) -> dict[str, RapEvent]:
    records: dict[str, RapEvent] = {}
    for event in events:
        if event.event_type == event_type and isinstance(event.request_id, str) and event.request_id:
            records.setdefault(event.request_id, event)
    return records


def _candidate_evaluations(events: Iterable[RapEvent]) -> dict[tuple[str, str], RapEvent]:
    evaluations: dict[tuple[str, str], RapEvent] = {}
    for event in events:
        candidate_id = event.payload.get("candidate_id")
        if (
            event.event_type == RapEventType.CANDIDATE_EVALUATED
            and isinstance(event.request_id, str)
            and event.request_id
            and isinstance(candidate_id, str)
            and candidate_id
        ):
            evaluations.setdefault((event.request_id, candidate_id), event)
    return evaluations


def _candidate_counts(
    batches: dict[str, RapEvent],
    evaluations: dict[tuple[str, str], RapEvent],
) -> tuple[int, int]:
    parsed = 0
    valid = 0
    for request_id, batch in batches.items():
        batch_evaluations = [
            event for (evaluation_request_id, _candidate_id), event in evaluations.items() if evaluation_request_id == request_id
        ]
        candidate_count = _candidate_count(batch.payload)
        if candidate_count is not None and len(batch_evaluations) > candidate_count:
            raise ValueError(f"candidate count contradiction for request {request_id}")
        parsed += candidate_count if candidate_count is not None else len(batch_evaluations)
        valid += sum(event.payload.get("valid") is True for event in batch_evaluations)
    return valid, parsed


def _generator_error_requests(
    events: Iterable[RapEvent],
    plans: dict[str, RapEvent],
    batches: dict[str, RapEvent],
) -> set[str]:
    errors = {request_id for request_id, batch in batches.items() if request_id in plans and bool(batch.payload.get("error_type"))}
    for event in events:
        if (
            event.event_type == RapEventType.GENERATION_FAILED
            and isinstance(event.request_id, str)
            and event.request_id in plans
        ):
            errors.add(event.request_id)
    return errors


def _repetition_counts(
    events: Iterable[RapEvent],
    frozen: dict[int, RapEvent],
    expected_manifest_window: int | None,
) -> tuple[int, int]:
    window = _repetition_window(events, expected_manifest_window)
    recent: list[set[tuple[str, str]]] = []
    repeated = 0
    generated = 0
    frozen_events = sorted(frozen.values(), key=lambda event: event.sequence)
    for event in frozen_events:
        text = event.payload.get("text")
        bigrams = _normalized_bigrams(text) if isinstance(text, str) else []
        prior_bigrams = set().union(*recent) if recent else set()
        generated += len(bigrams)
        repeated += sum(bigram in prior_bigrams for bigram in bigrams)
        recent.append(set(bigrams))
        del recent[:-window]
    return repeated, generated


def _repetition_window(events: Iterable[RapEvent], expected_manifest_window: int | None) -> int:
    if expected_manifest_window is not None and (
        not isinstance(expected_manifest_window, int)
        or isinstance(expected_manifest_window, bool)
        or expected_manifest_window <= 0
    ):
        raise ValueError("expected manifest repetition window must be a positive integer")
    for event in events:
        if event.event_type != RapEventType.SESSION_STARTED:
            continue
        if "repetition_window_bars" not in event.payload:
            window = DEFAULT_REPETITION_WINDOW_BARS
            if expected_manifest_window is not None and expected_manifest_window != window:
                raise ValueError("event and manifest repetition window disagree")
            return window
        value = event.payload["repetition_window_bars"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("session-start repetition_window_bars must be a positive integer")
        if expected_manifest_window is not None and expected_manifest_window != value:
            raise ValueError("event and manifest repetition window disagree")
        return value
    if expected_manifest_window is not None and expected_manifest_window != DEFAULT_REPETITION_WINDOW_BARS:
        raise ValueError("event and manifest repetition window disagree")
    return DEFAULT_REPETITION_WINDOW_BARS


def _candidate_count(payload: dict[str, Any]) -> int | None:
    if "candidate_count" not in payload:
        return None
    value = payload["candidate_count"]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("candidate_count must be a nonnegative integer")
    return value


def _normalized_bigrams(text: str) -> list[tuple[str, str]]:
    words = normalize_text(text).split()
    return list(zip(words, words[1:]))


def _distribution(values: Iterable[float]) -> dict[str, int | float | None]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "max": None}
    return {
        "count": len(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def _percentile(values: list[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _frozen_bars(events: Iterable[RapEvent]) -> dict[int, RapEvent]:
    frozen: dict[int, RapEvent] = {}
    for event in events:
        if event.event_type == RapEventType.BAR_FROZEN and event.bar is not None:
            frozen.setdefault(event.bar, event)
    return frozen


def _pronunciation_counts(events: Iterable[RapEvent]) -> tuple[int, int]:
    fallback = 0
    total = 0
    for event in events:
        sources = event.payload.get("word_analysis_sources")
        if not isinstance(sources, list):
            continue
        for item in sources:
            if not isinstance(item, dict) or not isinstance(item.get("source"), str):
                continue
            total += 1
            if item["source"] != "cmudict_first_pronunciation":
                fallback += 1
    return fallback, total


def _payload_numbers(events: Iterable[RapEvent], key: str) -> list[float]:
    return [number for event in events if (number := _number_or_none(event.payload.get(key))) is not None]


def _token_usage(events: Iterable[RapEvent], key: str) -> dict[str, int]:
    values = [
        value
        for event in events
        if isinstance((value := event.payload.get(key)), int) and not isinstance(value, bool) and value >= 0
    ]
    return {"count": len(values), "total": sum(values)}


def _number_or_none(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _string_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _empty_bar_row(bar: int) -> dict[str, Any]:
    return {
        "bar": bar,
        "request_id": None,
        "source": None,
        "text": None,
        "frozen": False,
        "fallback": False,
        "fallback_reason": None,
        "candidate_count": 0,
        "valid_candidate_count": 0,
        "selected_candidate_id": None,
        "generation_latency_ms": None,
        "deadline_slack_ms": None,
        "generator_error": None,
        "emitted_syllables": 0,
        "mean_syllable_observation_delay_ms": None,
        "_declared_candidate_count": None,
        "_evaluated_candidate_count": 0,
        "_observation_delays": [],
    }
