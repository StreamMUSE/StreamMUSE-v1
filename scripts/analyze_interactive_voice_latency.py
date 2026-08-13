"""Analyze per-stage latency from interactive voice response traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
FIXED_COLUMNS = (
    "source",
    "run_id",
    "turn_id",
    "actor",
    "number",
    "is_valid",
    "deadline_missed",
    "latency_ms",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Run directories, response_trace.jsonl files, or roots containing runs",
    )
    parser.add_argument(
        "--output-dir",
        default="voice_latency_analysis",
        help="Directory for CSV, JSON, and Markdown results",
    )
    args = parser.parse_args(argv)

    traces = discover_traces(Path(value) for value in args.inputs)
    if not traces:
        parser.error("no response_trace.jsonl files found")
    result = analyze_traces(traces)
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_outputs(output_dir, result)
    print(f"analyzed {result['turn_count']} timed turns from {len(traces)} trace(s)")
    print(f"wrote: {output_dir}")
    return 0


def discover_traces(inputs: Iterable[Path]) -> tuple[Path, ...]:
    traces: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            if path.name != "response_trace.jsonl":
                raise ValueError(f"expected response_trace.jsonl, received {path}")
            traces.add(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(path)
        direct = path / "response_trace.jsonl"
        if direct.is_file():
            traces.add(direct)
        else:
            traces.update(path.rglob("response_trace.jsonl"))
    return tuple(sorted(traces))


def analyze_traces(traces: Iterable[Path]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    startup: list[dict[str, Any]] = []
    skipped_turn_count = 0
    sources: list[str] = []
    for trace in traces:
        sources.append(str(trace))
        records = _read_jsonl(trace)
        timed_rows: list[dict[str, Any]] = []
        for record in records:
            breakdown = _record_breakdown(record)
            if breakdown is None:
                skipped_turn_count += 1
                continue
            _validate_breakdown(
                breakdown,
                label=f"{trace}:turn={record.get('turn_id')}",
            )
            row = _turn_row(trace, record, breakdown)
            timed_rows.append(row)
        _add_cross_turn_metrics(timed_rows)
        rows.extend(timed_rows)
        manifest = trace.parent / "manifest.json"
        if manifest.is_file():
            startup.append(_startup_row(manifest))

    metrics = _summarize_rows(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": sources,
        "turn_count": len(rows),
        "skipped_turn_count": skipped_turn_count,
        "startup": startup,
        "metrics": metrics,
        "rows": rows,
    }


def write_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    metric_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in FIXED_COLUMNS
        }
    )
    csv_path = output_dir / "breakdown_turns.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*FIXED_COLUMNS, *metric_columns],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {key: value for key, value in result.items() if key != "rows"}
    (output_dir / "breakdown_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "breakdown_summary.md").write_text(
        _render_markdown(summary),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: trace row must be an object")
        rows.append(value)
    return rows


def _record_breakdown(record: dict[str, Any]) -> dict[str, Any] | None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("timing_breakdown")
    return value if isinstance(value, dict) else None


def _validate_breakdown(value: dict[str, Any], *, label: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{label}: unsupported timing schema")
    _validate_numeric_mapping(value.get("anchors_ms"), label=f"{label}:anchors")
    _validate_numeric_mapping(
        value.get("durations_ms"),
        label=f"{label}:durations",
    )
    session_offset = value.get("origin_session_offset_ms")
    if session_offset is not None:
        _require_nonnegative_number(
            session_offset,
            label=f"{label}:origin_session_offset_ms",
        )
    components = value.get("components")
    if components is None:
        return
    if not isinstance(components, dict):
        raise ValueError(f"{label}:components must be an object")
    for name, component in components.items():
        if isinstance(component, dict):
            _validate_component(component, label=f"{label}:components.{name}")


def _validate_component(value: dict[str, Any], *, label: str) -> None:
    if "anchors_ms" in value:
        _validate_numeric_mapping(value.get("anchors_ms"), label=f"{label}:anchors")
    if "durations_ms" in value:
        _validate_numeric_mapping(
            value.get("durations_ms"),
            label=f"{label}:durations",
        )
    for name, child in value.items():
        if name in {"anchors_ms", "durations_ms"}:
            continue
        if isinstance(child, dict) and (
            "anchors_ms" in child or "durations_ms" in child
        ):
            _validate_component(child, label=f"{label}.{name}")


def _validate_numeric_mapping(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    for name, number in value.items():
        if number is not None:
            _require_nonnegative_number(number, label=f"{label}.{name}")


def _require_nonnegative_number(value: object, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _turn_row(
    trace: Path,
    record: dict[str, Any],
    breakdown: dict[str, Any],
) -> dict[str, Any]:
    run_id = None
    manifest = trace.parent / "manifest.json"
    if manifest.is_file():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            run_id = value.get("run_id")
    row: dict[str, Any] = {
        "source": str(trace),
        "run_id": run_id,
        "turn_id": record.get("turn_id"),
        "actor": record.get("actor"),
        "number": record.get("number"),
        "is_valid": record.get("is_valid"),
        "deadline_missed": record.get("deadline_missed"),
        "latency_ms": record.get("latency_ms"),
    }
    _flatten_durations(row, "turn", breakdown)
    components = breakdown.get("components")
    if isinstance(components, dict):
        for name, component in components.items():
            if isinstance(component, dict):
                _flatten_durations(row, f"component.{name}", component)
    _add_derived_turn_metrics(row, breakdown)
    row["_origin_session_offset_ms"] = breakdown.get(
        "origin_session_offset_ms"
    )
    row["_anchors_ms"] = breakdown.get("anchors_ms")
    return row


def _flatten_durations(
    row: dict[str, Any],
    prefix: str,
    breakdown: dict[str, Any],
) -> None:
    durations = breakdown.get("durations_ms")
    if isinstance(durations, dict):
        for name, value in durations.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[f"{prefix}.{name}_ms"] = float(value)
    for name, child in breakdown.items():
        if name in {
            "anchors_ms",
            "durations_ms",
            "components",
        }:
            continue
        if isinstance(child, dict) and (
            "anchors_ms" in child or "durations_ms" in child
        ):
            _flatten_durations(row, f"{prefix}.{name}", child)


def _add_derived_turn_metrics(
    row: dict[str, Any],
    breakdown: dict[str, Any],
) -> None:
    anchors = breakdown.get("anchors_ms")
    if not isinstance(anchors, dict):
        return
    actor = row.get("actor")
    if actor == "human":
        _derive(
            row,
            anchors,
            "human.mic_open_to_first_callback_ms",
            "microphone.stream_open_started",
            "microphone.first_callback",
        )
        _derive(
            row,
            anchors,
            "human.wait_for_speech_ms",
            "response_source_started",
            "microphone.first_voiced",
        )
        _derive(
            row,
            anchors,
            "human.voiced_utterance_ms",
            "microphone.first_voiced",
            "microphone.last_voiced",
        )
        _derive(
            row,
            anchors,
            "human.endpoint_delay_ms",
            "microphone.last_voiced",
            "microphone.endpoint_detected",
        )
        _derive(
            row,
            anchors,
            "human.last_voice_to_asr_text_ms",
            "microphone.last_voiced",
            "asr.completed",
        )
        _derive(
            row,
            anchors,
            "human.last_voice_to_decision_ms",
            "microphone.last_voiced",
            "game_validation_completed",
        )
    elif actor == "llm":
        _derive(
            row,
            anchors,
            "llm.request_to_text_ms",
            "llm_request_started",
            "llm_response_completed",
        )
        _derive(
            row,
            anchors,
            "llm.text_to_first_dac_ms",
            "llm_response_completed",
            "speech.first_dac_sample",
        )
        _derive(
            row,
            anchors,
            "llm.request_to_first_dac_ms",
            "llm_request_started",
            "speech.first_dac_sample",
        )
        _derive(
            row,
            anchors,
            "llm.request_to_audio_drained_ms",
            "llm_request_started",
            "speech.playback_drained",
        )


def _derive(
    row: dict[str, Any],
    anchors: dict[str, Any],
    metric: str,
    start: str,
    end: str,
) -> None:
    start_value = anchors.get(start)
    end_value = anchors.get(end)
    if isinstance(start_value, (int, float)) and isinstance(
        end_value,
        (int, float),
    ):
        row[metric] = _ordered_delta(
            float(start_value),
            float(end_value),
            metric=metric,
        )


def _add_cross_turn_metrics(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        actor = row.get("actor")
        if (
            actor == "llm"
            and index > 0
            and rows[index - 1].get("actor") == "human"
            and _are_consecutive_turns(rows[index - 1], row)
        ):
            previous = rows[index - 1]
            human_last_voice = _session_anchor(
                previous,
                "microphone.last_voiced",
            )
            llm_text = _session_anchor(row, "llm_response_completed")
            first_dac = _session_anchor(row, "speech.first_dac_sample")
            drained = _session_anchor(row, "speech.playback_drained")
            _cross_metric(
                row,
                "conversation.human_last_voice_to_llm_text_ms",
                human_last_voice,
                llm_text,
            )
            _cross_metric(
                row,
                "conversation.human_last_voice_to_first_dac_ms",
                human_last_voice,
                first_dac,
            )
            _cross_metric(
                row,
                "conversation.human_last_voice_to_audio_drained_ms",
                human_last_voice,
                drained,
            )
        if (
            actor == "human"
            and index > 0
            and rows[index - 1].get("actor") == "llm"
            and _are_consecutive_turns(rows[index - 1], row)
        ):
            previous = rows[index - 1]
            drained = _session_anchor(previous, "speech.playback_drained")
            first_callback = _session_anchor(row, "microphone.first_callback")
            _cross_metric(
                row,
                "conversation.audio_drained_to_mic_callback_ms",
                drained,
                first_callback,
            )

    for row in rows:
        row.pop("_origin_session_offset_ms", None)
        row.pop("_anchors_ms", None)


def _session_anchor(row: dict[str, Any], anchor: str) -> float | None:
    origin = row.get("_origin_session_offset_ms")
    anchors = row.get("_anchors_ms")
    if not isinstance(origin, (int, float)) or not isinstance(anchors, dict):
        return None
    value = anchors.get(anchor)
    if not isinstance(value, (int, float)):
        return None
    return float(origin) + float(value)


def _cross_metric(
    row: dict[str, Any],
    name: str,
    start_ms: float | None,
    end_ms: float | None,
) -> None:
    if start_ms is not None and end_ms is not None:
        row[name] = _ordered_delta(start_ms, end_ms, metric=name)


def _are_consecutive_turns(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    previous_id = previous.get("turn_id")
    current_id = current.get("turn_id")
    return (
        isinstance(previous_id, int)
        and not isinstance(previous_id, bool)
        and isinstance(current_id, int)
        and not isinstance(current_id, bool)
        and current_id == previous_id + 1
    )


def _ordered_delta(start_ms: float, end_ms: float, *, metric: str) -> float:
    delta = end_ms - start_ms
    if delta < -1e-6:
        raise ValueError(
            f"{metric} has reversed anchors: start={start_ms}, end={end_ms}"
        )
    return max(0.0, delta)


def _startup_row(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    human = manifest.get("human_input") if isinstance(manifest, dict) else None
    recognizer = human.get("recognizer") if isinstance(human, dict) else None
    speech = manifest.get("speech_output") if isinstance(manifest, dict) else None
    return {
        "manifest": str(path),
        "run_id": manifest.get("run_id") if isinstance(manifest, dict) else None,
        "stt_model_resolution_ms": (
            recognizer.get("model_resolution_ms")
            if isinstance(recognizer, dict)
            else None
        ),
        "stt_model_load_ms": (
            recognizer.get("model_load_ms")
            if isinstance(recognizer, dict)
            else None
        ),
        "stt_warmup_ms": (
            recognizer.get("warmup_ms")
            if isinstance(recognizer, dict)
            else None
        ),
        "tts_prewarm_ms": (
            speech.get("prewarm_ms") if isinstance(speech, dict) else None
        ),
        "tts_prewarm_entry_count": (
            speech.get("prewarm_entry_count")
            if isinstance(speech, dict)
            else None
        ),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if key in FIXED_COLUMNS or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.setdefault(key, []).append(float(value))
    return {
        name: _summary(numbers)
        for name, numbers in sorted(values.items())
    }


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p50": _percentile(ordered, 0.50),
        "p90": _percentile(ordered, 0.90),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Interactive Voice Latency Breakdown",
        "",
        f"- Timed turns: {summary['turn_count']}",
        f"- Skipped turns without timing: {summary['skipped_turn_count']}",
        f"- Trace files: {len(summary['sources'])}",
        "",
        "| Metric | n | mean | p50 | p90 | p95 | p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in summary["metrics"].items():
        lines.append(
            f"| `{name}` | {values['count']} | "
            f"{values['mean']:.3f} | {values['p50']:.3f} | "
            f"{values['p90']:.3f} | {values['p95']:.3f} | "
            f"{values['p99']:.3f} | {values['max']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
