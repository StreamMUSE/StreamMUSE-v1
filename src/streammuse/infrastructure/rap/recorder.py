"""Crash-tolerant recorder and deterministic derivations for rap research runs."""

from __future__ import annotations

import csv
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from streammuse.domain.rap import RapEvent, RapEventType


_SECRET_NAME = re.compile(r"[^a-z0-9]+")
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
    "mean_emission_jitter_ms",
)


class RapSessionRecorder:
    """Append canonical events first, then derive artifacts after a clean close."""

    def __init__(self, session_dir: Path, manifest: dict[str, Any]) -> None:
        session_dir.mkdir(parents=True, exist_ok=False)
        self._events_path = session_dir / "events.jsonl"
        write_json(session_dir / "session.json", redact_manifest(manifest))
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
        write_json(self._events_path.parent / "summary.json", derive_summary(events))
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
        "payload": deepcopy(event.payload),
    }


def read_events(path: Path) -> list[RapEvent]:
    """Recover all complete records, ignoring incomplete/corrupt JSONL lines."""

    events: list[RapEvent] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            event = RapEvent(
                session_id=record["session_id"],
                sequence=record["sequence"],
                event_type=RapEventType(record["event_type"]),
                utc_time=record["utc_time"],
                monotonic_ns=record["monotonic_ns"],
                bar=record["bar"],
                tick=record["tick"],
                request_id=record["request_id"],
                payload=record["payload"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        events.append(event)
    return events


def redact_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact values associated with secret-looking manifest keys."""

    def redact(value: Any, key: str | None = None) -> Any:
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


def derive_summary(events: Iterable[RapEvent]) -> dict[str, Any]:
    """Derive research metrics from event evidence without hidden runtime state."""

    event_list = list(events)
    frozen = _frozen_bars(event_list)
    evaluations = [event for event in event_list if event.event_type == RapEventType.CANDIDATE_EVALUATED]
    batches = [event for event in event_list if event.event_type == RapEventType.CANDIDATE_BATCH_RECEIVED]

    candidate_validity = sum(event.payload.get("valid") is True for event in evaluations)
    fallback_count = sum(event.payload.get("fallback") is True for event in frozen.values())
    deadline_misses = sum(event.payload.get("fallback_reason") == "deadline_miss" for event in frozen.values())
    generator_errors = sum(bool(event.payload.get("error_type")) for event in batches)
    pronunciation_fallbacks, pronunciation_total = _pronunciation_counts(evaluations)
    repetitions = sum(
        "duplicate_normalized_text" in _string_list(event.payload.get("rejection_reasons")) for event in evaluations
    )

    return {
        "events": {"count": len(event_list)},
        "bars": {
            "frozen": len(frozen),
            "fallback": fallback_count,
            "fallback_rate": _ratio(fallback_count, len(frozen))["rate"],
        },
        "metrics": {
            "candidate_validity": _ratio(candidate_validity, len(evaluations)),
            "fallback": _ratio(fallback_count, len(frozen)),
            "deadline_miss": _ratio(deadline_misses, len(frozen)),
            "generator_error": _ratio(generator_errors, len(batches)),
            "pronunciation_fallback": _ratio(pronunciation_fallbacks, pronunciation_total),
            "repetition": _ratio(repetitions, len(evaluations)),
        },
        "latencies": {
            "generation_latency_ms": _distribution(_payload_numbers(batches, "latency_ms")),
            "deadline_slack_ms": _distribution(_payload_numbers(batches, "deadline_slack_ms")),
            "emission_jitter_ms": _distribution(
                _payload_numbers((event for event in event_list if event.event_type == RapEventType.SYLLABLE_EMITTED), "jitter_ms")
            ),
        },
    }


def derive_bar_rows(events: Iterable[RapEvent]) -> list[dict[str, Any]]:
    """Derive deterministic, one-row-per-bar CSV-ready research evidence."""

    bars: dict[int, dict[str, Any]] = {}
    for event in events:
        if event.bar is None:
            continue
        row = bars.setdefault(event.bar, _empty_bar_row(event.bar))
        if event.request_id is not None:
            row["request_id"] = event.request_id
        payload = event.payload
        if event.event_type == RapEventType.CANDIDATE_BATCH_RECEIVED:
            row["generation_latency_ms"] = _number_or_none(payload.get("latency_ms"))
            row["deadline_slack_ms"] = _number_or_none(payload.get("deadline_slack_ms"))
            row["generator_error"] = payload.get("error_type") if isinstance(payload.get("error_type"), str) else None
            supplied_count = _number_or_none(payload.get("candidate_count"))
            if supplied_count is not None:
                row["candidate_count"] = max(row["candidate_count"], int(supplied_count))
        elif event.event_type == RapEventType.CANDIDATE_EVALUATED:
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
        elif event.event_type == RapEventType.SYLLABLE_EMITTED:
            jitter = _number_or_none(payload.get("jitter_ms"))
            if jitter is not None:
                row["_jitters"].append(jitter)
            row["emitted_syllables"] += 1

    result: list[dict[str, Any]] = []
    for bar in sorted(bars):
        row = bars[bar]
        jitters = row.pop("_jitters")
        row["candidate_count"] = max(row["candidate_count"], row.pop("_evaluated_candidate_count"))
        row["mean_emission_jitter_ms"] = sum(jitters) / len(jitters) if jitters else None
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
    return {
        event.bar: event
        for event in events
        if event.event_type == RapEventType.BAR_FROZEN and event.bar is not None
    }


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
        "mean_emission_jitter_ms": None,
        "_evaluated_candidate_count": 0,
        "_jitters": [],
    }
