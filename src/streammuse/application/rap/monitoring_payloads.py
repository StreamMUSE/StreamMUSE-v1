"""Stable JSON-ready monitoring payload builders for rap planning."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import islice
import json
from math import isfinite
from typing import Any

from streammuse.domain.rap import (
    FlowTemplate,
    REMOTE_CHUNK_ARTIFACT_IDS,
    RemoteRapChunkRequest,
    ScheduledSyllable,
)


_MAX_CHUNK_LINES = 2
_MAX_CHUNK_FLOWS = 2
_MAX_FLOW_SLOTS = 32
_MAX_CONTEXT_LINES = 4
_MAX_COMPONENT_SCORES = 16
_MAX_DIAGNOSTIC_ITEMS = 8
_MAX_ARTIFACT_REFS = len(REMOTE_CHUNK_ARTIFACT_IDS)
_MAX_LINE_BYTES = 512
_MAX_WARNING_BYTES = 256
_MAX_NAME_BYTES = 128
_MAX_SUMMARY_BYTES = 4_096
_MAX_REFERENCE_BYTES = 1_024
_MAX_NUMBER_MAGNITUDE = 1_000_000_000_000
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
MAX_BOUNDED_CHUNK_EVENT_BYTES = 24_000
_MISSING = object()
_CHUNK_TIMING_ALIASES = {
    "generation": ("generation", "generation_ms"),
    "evaluation": ("evaluation", "evaluation_ms"),
    "moss": ("moss", "moss_ms"),
    "aligner": ("aligner", "alignment", "aligner_ms", "alignment_ms"),
    "r3": ("r3", "warp", "rubberband", "r3_ms", "warp_ms"),
    "package": ("package", "packaging", "package_ms", "packaging_ms"),
    "transfer": ("transfer", "transfer_ms"),
    "mac": ("mac", "mac_ms", "mac_validation_mix", "mac_validation_mix_ms"),
    "total": ("total", "total_ms", "end_to_end", "end_to_end_ms"),
}


def flow_template_payload(template: FlowTemplate) -> dict[str, Any]:
    """Return every flow alignment field used to make a planning decision."""

    return {
        "template_id": template.template_id,
        "name": template.name,
        "ticks_per_beat": template.ticks_per_beat,
        "beats_per_bar": template.beats_per_bar,
        "provenance": {
            "kind": template.provenance.kind,
            "source": template.provenance.source,
            "source_hash": template.provenance.source_hash,
            "quantization_error_ticks": template.provenance.quantization_error_ticks,
        },
        "slots": [
            {
                "slot_index": index,
                "tick_in_bar": slot.tick_in_bar,
                "duration_ticks": slot.duration_ticks,
                "target_stress": slot.target_stress,
                "boundary_strength": slot.boundary_strength,
                "rhyme_group": slot.rhyme_group,
            }
            for index, slot in enumerate(template.slots)
        ],
    }


def scheduled_syllables_payload(
    scheduled: tuple[ScheduledSyllable, ...], *, bar: int
) -> list[dict[str, Any]]:
    """Return scheduled lyric syllables relative to their owning bar."""

    return [
        {
            "slot_index": item.slot.slot_index,
            "tick_in_bar": item.slot.tick - (bar * 16),
            "target_stress": item.slot.accent,
            "label": item.syllable.label,
            "word": item.syllable.word,
            "stress": item.syllable.stress,
            "stressed": item.syllable.stressed,
        }
        for item in scheduled
    ]


def remote_generation_input_summary(request: RemoteRapChunkRequest) -> str:
    """Summarize exact deterministic inputs without claiming to retain provider text."""

    bars = []
    for item in request.bars:
        schedule = ",".join(
            f"t{slot.tick_in_bar}@{slot.target_stress:.3f}"
            for slot in item.flow_template.slots
        )
        bars.append(
            f"bar={item.bar} topic={json.dumps(item.topic, ensure_ascii=True)} "
            f"template={item.flow_template.template_id} slots=[{schedule}]"
        )
    policy = json.dumps(
        request.policy.to_payload(), sort_keys=True, separators=(",", ":")
    )
    context = json.dumps(
        request.context_lines, ensure_ascii=True, separators=(",", ":")
    )
    summary = (
        "deterministic generation input summary; not a verbatim provider prompt: "
        f"{' ; '.join(bars)}; context_lines={context}; policy={policy}; seed={request.seed}"
    )
    return _bounded_text(summary, _MAX_SUMMARY_BYTES) or ""


def bounded_chunk_event_payload(value: object) -> dict[str, Any]:
    """Reduce remote chunk diagnostics to the fixed live-monitor contract."""

    payload = value if isinstance(value, Mapping) else {}
    manifest = _mapping(_safe_get(payload, "manifest"))
    manifest_diagnostics = _mapping(_safe_get(manifest, "diagnostics"))
    selected_bars = _mapping_items(_safe_get(manifest, "selected_bars"), _MAX_CHUNK_LINES)

    selected_lines = _bounded_strings(_safe_get(payload, "selected_lines"), _MAX_CHUNK_LINES)
    if not selected_lines:
        selected_lines = [
            text
            for item in selected_bars
            if (text := _bounded_text(_safe_get(item, "text"), _MAX_LINE_BYTES)) is not None
        ][:_MAX_CHUNK_LINES]

    raw_flows = _mapping_items(_safe_get(payload, "flows"), _MAX_CHUNK_FLOWS)
    flows = [_bounded_flow(item) for item in raw_flows]
    if not flows:
        flows = [
            _bounded_flow({"template_id": _safe_get(item, "flow_template_id"), "slots": []})
            for item in selected_bars
            if isinstance(_safe_get(item, "flow_template_id"), str)
        ][:_MAX_CHUNK_FLOWS]
    selected_schedules = _bounded_strings(
        _safe_get(payload, "selected_schedules"),
        _MAX_CHUNK_FLOWS,
        text_limit=_MAX_SUMMARY_BYTES,
    )
    for index, flow in enumerate(flows):
        schedule = (
            selected_schedules[index]
            if index < len(selected_schedules)
            else _selected_syllable_schedule(selected_bars[index])
            if index < len(selected_bars)
            else flow.get("selected_syllable_schedule")
        )
        flow["selected_syllable_schedule"] = schedule

    candidate_source = _mapping(_safe_get(payload, "candidate_counts")) or _mapping(
        _safe_get(manifest_diagnostics, "candidate_stats")
    )
    candidate_counts = {
        name: count
        for name, aliases in {
            "requested": ("requested", "requested_count"),
            "parseable": ("parseable", "parseable_count"),
            "valid": ("valid", "valid_count"),
            "selectable": ("selectable", "selectable_count"),
        }.items()
        if (count := _first_nonnegative_integer(candidate_source, aliases)) is not None
    }

    score_items = _mapping_items(_safe_get(payload, "selected_scores"), _MAX_CHUNK_LINES)
    if not score_items:
        score_items = selected_bars
    selected_scores = [_bounded_selected_score(item) for item in score_items[:_MAX_CHUNK_LINES]]

    timing_source = _mapping(_safe_get(payload, "stage_timings_ms")) or _mapping(
        _safe_get(manifest_diagnostics, "stage_timings_ms")
    )
    stage_timings = {
        name: timing
        for name, aliases in _CHUNK_TIMING_ALIASES.items()
        if (timing := _first_nonnegative_number(timing_source, aliases)) is not None
    }
    transfer = _mapping(_safe_get(payload, "transfer"))
    if "transfer" not in stage_timings:
        transfer_ms = _first_nonnegative_number(
            transfer, ("total_ms", "response_ms", "download_ms")
        )
        if transfer_ms is not None:
            stage_timings["transfer"] = transfer_ms
    if "mac" not in stage_timings:
        mac_ms = _first_nonnegative_number(
            payload,
            ("mac_validation_mix_ms", "mac_timing_ms", "mac_ms"),
        )
        if mac_ms is not None:
            stage_timings["mac"] = mac_ms

    alignment_source = _mapping(_safe_get(payload, "alignment")) or _mapping(
        _safe_get(payload, "alignment_diagnostics")
    ) or _mapping(_safe_get(manifest_diagnostics, "alignment_diagnostics"))
    monitoring_summary = _mapping(_safe_get(manifest_diagnostics, "monitoring_summary"))
    fallback_counts = _bounded_integer_mapping(_safe_get(alignment_source, "fallback_counts"))
    confidence_value = _first_value(
        alignment_source,
        ("confidence", "mean_confidence", "minimum_confidence"),
    )
    if confidence_value is None:
        confidence_value = _safe_get(monitoring_summary, "alignment_confidence")
    alignment = {
        "method": _bounded_text(
            _first_value(alignment_source, ("method", "alignment_method", "aligner"))
            or _safe_get(monitoring_summary, "alignment_method"),
            _MAX_LINE_BYTES,
        ),
        "confidence": _normalized_number(confidence_value),
        "fallback_counts": fallback_counts,
    }

    diagnostic_warnings = _bounded_strings(
        _safe_get(manifest_diagnostics, "warnings"),
        _MAX_DIAGNOSTIC_ITEMS,
        text_limit=_MAX_WARNING_BYTES,
    )
    warnings = _unique_bounded_strings(
        [
            *_bounded_strings(
                _safe_get(payload, "warnings"),
                _MAX_DIAGNOSTIC_ITEMS,
                text_limit=_MAX_WARNING_BYTES,
            ),
            *diagnostic_warnings,
        ],
        _MAX_DIAGNOSTIC_ITEMS,
    )
    stretch_warnings = _unique_bounded_strings(
        [
            *_bounded_strings(
                _safe_get(payload, "stretch_warnings"),
                _MAX_DIAGNOSTIC_ITEMS,
                text_limit=_MAX_WARNING_BYTES,
            ),
            *(item for item in warnings if any(token in item.lower() for token in ("stretch", "warp", "ratio"))),
        ],
        _MAX_DIAGNOSTIC_ITEMS,
    )

    hashes = _bounded_string_mapping(_safe_get(payload, "hashes"))
    vocal_hash = _bounded_text(_safe_get(manifest, "vocal_sha256"), _MAX_REFERENCE_BYTES)
    if vocal_hash is not None:
        hashes.setdefault("vocal_sha256", vocal_hash)

    transfer_bytes = _first_nonnegative_integer(payload, ("transfer_bytes",))
    if transfer_bytes is None:
        transfer_bytes = _first_nonnegative_integer(transfer, ("response_bytes", "bytes"))

    request_budget_ms = _first_nonnegative_number(
        payload,
        ("request_budget_ms", "accepted_request_budget_ms"),
    )
    if request_budget_ms is None:
        request_budget_ms = _first_nonnegative_number(
            manifest_diagnostics, ("accepted_request_budget_ms",)
        )

    result = {
        "state": _bounded_text(_safe_get(payload, "state"), _MAX_LINE_BYTES),
        "renderer_decision": _bounded_text(
            _first_value(payload, ("renderer_decision", "renderer_commitment", "renderer")),
            _MAX_LINE_BYTES,
        ),
        "coordinator_epoch": _nonnegative_integer(_safe_get(payload, "coordinator_epoch")),
        "chunk_index": _nonnegative_integer(_safe_get(payload, "chunk_index")),
        "bars": _bounded_integers(_safe_get(payload, "bars"), _MAX_CHUNK_LINES),
        "selected_lines": selected_lines,
        "flows": flows,
        "candidate_counts": candidate_counts,
        "selected_scores": selected_scores,
        "prompt_summary": _bounded_text(_safe_get(payload, "prompt_summary"), _MAX_SUMMARY_BYTES),
        "context_lines": _bounded_strings(_safe_get(payload, "context_lines"), _MAX_CONTEXT_LINES),
        "stage_timings_ms": stage_timings,
        "request_budget_ms": request_budget_ms,
        "elapsed_ms": _first_nonnegative_number(
            payload, ("elapsed_ms", "request_elapsed_ms")
        ),
        "deadline_slack_ms": _first_finite_number(
            payload,
            ("deadline_slack_ms", "final_deadline_slack_ms"),
        ),
        "alignment": alignment,
        "stretch_warnings": stretch_warnings,
        "warnings": warnings,
        "hashes": hashes,
        "artifact_refs": _bounded_string_mapping(
            _safe_get(payload, "artifact_refs"), limit=_MAX_ARTIFACT_REFS
        ),
        "transfer_bytes": transfer_bytes,
        "failure_reason": _bounded_text(
            _first_value(payload, ("failure_reason", "error_message")),
            _MAX_LINE_BYTES,
        ),
    }
    return _enforce_serialized_ceiling(result)


def _bounded_flow(value: Mapping[str, object]) -> dict[str, Any]:
    slots = []
    for item in _mapping_items(_safe_get(value, "slots"), _MAX_FLOW_SLOTS):
        tick = _nonnegative_integer(_safe_get(item, "tick_in_bar"))
        stress = _normalized_number(_safe_get(item, "target_stress"))
        slot: dict[str, object] = {"tick_in_bar": tick, "target_stress": stress}
        for key in ("duration_ticks", "boundary_strength"):
            number = _nonnegative_number(_safe_get(item, key))
            if number is not None:
                slot[key] = number
        rhyme = _bounded_text(_safe_get(item, "rhyme_group"), _MAX_NAME_BYTES)
        if rhyme is not None:
            slot["rhyme_group"] = rhyme
        slots.append(slot)
    schedule = _bounded_text(", ".join(
        f"t{item['tick_in_bar']}@{item['target_stress']:.2f}"
        if item["tick_in_bar"] is not None and item["target_stress"] is not None
        else f"t{item['tick_in_bar']}@?"
        for item in slots
    ), _MAX_SUMMARY_BYTES) or ""
    result: dict[str, Any] = {
        "template_id": _bounded_text(_safe_get(value, "template_id"), _MAX_NAME_BYTES),
        "name": _bounded_text(_safe_get(value, "name"), _MAX_NAME_BYTES),
        "slots": slots,
        "slot_stress_schedule": schedule,
        "selected_syllable_schedule": _bounded_text(
            _safe_get(value, "selected_syllable_schedule"), _MAX_SUMMARY_BYTES
        ),
    }
    for key in ("ticks_per_beat", "beats_per_bar"):
        number = _nonnegative_integer(_safe_get(value, key))
        if number is not None:
            result[key] = number
    return result


def _bounded_selected_score(value: Mapping[str, object]) -> dict[str, Any]:
    diagnostics = _mapping(_safe_get(value, "diagnostics"))
    component_source = _mapping(_safe_get(value, "component_scores")) or _mapping(
        _safe_get(diagnostics, "component_scores")
    )
    component_scores = {
        bounded_key: score
        for key, item in _safe_items(component_source, _MAX_COMPONENT_SCORES)
        if (bounded_key := _bounded_text(key, _MAX_NAME_BYTES)) is not None
        if isinstance(key, str) and (score := _finite_number(item)) is not None
    }
    return {
        "bar": _nonnegative_integer(_safe_get(value, "bar")),
        "total": _first_finite_number(value, ("total", "score", "total_score")),
        "component_scores": component_scores,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _mapping_items(value: object, limit: int) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in islice(value, limit) if isinstance(item, Mapping)]


def _first_value(value: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        item = _safe_get(value, key, _MISSING)
        if item is not _MISSING:
            return item
    return None


def _finite_number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    if abs(value) > _MAX_NUMBER_MAGNITUDE:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _normalized_number(value: object) -> float | None:
    number = _finite_number(value)
    return number if number is not None and 0 <= number <= 1 else None


def _nonnegative_number(value: object) -> float | None:
    number = _finite_number(value)
    return number if number is not None and number >= 0 else None


def _first_finite_number(value: Mapping[str, object], keys: tuple[str, ...]) -> float | None:
    return _finite_number(_first_value(value, keys))


def _first_nonnegative_number(
    value: Mapping[str, object], keys: tuple[str, ...]
) -> float | None:
    return _nonnegative_number(_first_value(value, keys))


def _nonnegative_integer(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= _MAX_SAFE_INTEGER else None


def _first_nonnegative_integer(value: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    return _nonnegative_integer(_first_value(value, keys))


def _bounded_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    encoded = value.encode("utf-8", errors="replace")
    return encoded[:limit].decode("utf-8", errors="ignore")


def _bounded_strings(value: object, limit: int, *, text_limit: int = _MAX_LINE_BYTES) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result = []
    for item in islice(value, limit):
        bounded = _bounded_text(item, text_limit)
        if bounded is not None:
            result.append(bounded)
    return result


def _unique_bounded_strings(value: list[str], limit: int) -> list[str]:
    result = []
    for item in value:
        bounded = _bounded_text(item, _MAX_WARNING_BYTES)
        if bounded is None:
            continue
        if bounded not in result:
            result.append(bounded)
        if len(result) == limit:
            break
    return result


def _bounded_integers(value: object, limit: int) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in islice(value, limit) if _nonnegative_integer(item) is not None]


def _bounded_integer_mapping(value: object) -> dict[str, int]:
    mapping = _mapping(value)
    return {
        bounded_key: count
        for key, item in _safe_items(mapping, _MAX_DIAGNOSTIC_ITEMS)
        if (bounded_key := _bounded_text(key, _MAX_NAME_BYTES)) is not None
        if (count := _nonnegative_integer(item)) is not None
    }


def _bounded_string_mapping(
    value: object, *, limit: int = _MAX_DIAGNOSTIC_ITEMS
) -> dict[str, str]:
    mapping = _mapping(value)
    return {
        bounded_key: bounded_item
        for key, item in _safe_items(mapping, limit)
        if (bounded_key := _bounded_text(key, _MAX_NAME_BYTES)) is not None
        if (bounded_item := _bounded_text(item, _MAX_REFERENCE_BYTES)) is not None
    }


def _safe_get(
    value: Mapping[str, object], key: str, default: object = None
) -> object:
    try:
        return value[key]
    except Exception:  # noqa: BLE001 - untrusted diagnostic mappings must not escape.
        return default


def _safe_items(
    value: Mapping[str, object], limit: int
) -> list[tuple[object, object]]:
    result: list[tuple[object, object]] = []
    try:
        iterator = iter(value.items())
    except Exception:  # noqa: BLE001 - untrusted diagnostic mappings must not escape.
        return result
    for _ in range(limit):
        try:
            item = next(iterator)
        except StopIteration:
            break
        except Exception:  # noqa: BLE001 - stop at the first hostile mapping item.
            break
        if isinstance(item, tuple) and len(item) == 2:
            result.append(item)
    return result


def _selected_syllable_schedule(value: Mapping[str, object]) -> str | None:
    bar = _nonnegative_integer(_safe_get(value, "bar"))
    scheduled = _safe_get(value, "scheduled")
    if bar is None or not isinstance(scheduled, (list, tuple)):
        return None
    parts = []
    for item in _mapping_items(scheduled, _MAX_FLOW_SLOTS):
        slot = _mapping(_safe_get(item, "slot"))
        syllable = _mapping(_safe_get(item, "syllable"))
        absolute_tick = _nonnegative_integer(_safe_get(slot, "tick"))
        stress = _nonnegative_integer(_safe_get(syllable, "stress"))
        word = _bounded_text(_safe_get(syllable, "word"), _MAX_NAME_BYTES)
        if absolute_tick is None or stress is None or word is None:
            continue
        relative_tick = absolute_tick - bar * 16
        if not 0 <= relative_tick <= _MAX_SAFE_INTEGER:
            continue
        parts.append(f"t{relative_tick}:{word}/stress{stress}")
    return _bounded_text(", ".join(parts), _MAX_SUMMARY_BYTES) if parts else None


def _encoded_size(value: Mapping[str, object]) -> int:
    try:
        return len(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    except (OverflowError, RecursionError, TypeError, ValueError):
        return MAX_BOUNDED_CHUNK_EVENT_BYTES + 1


def _enforce_serialized_ceiling(value: dict[str, Any]) -> dict[str, Any]:
    if _encoded_size(value) <= MAX_BOUNDED_CHUNK_EVENT_BYTES:
        return value

    compact_flows = []
    for flow in value["flows"]:
        compact_flows.append(
            {
                "template_id": _bounded_text(flow.get("template_id"), _MAX_NAME_BYTES),
                "name": _bounded_text(flow.get("name"), _MAX_NAME_BYTES),
                "slots": flow.get("slots", [])[:8],
                "slot_stress_schedule": _bounded_text(
                    flow.get("slot_stress_schedule"), 1_024
                ),
                "selected_syllable_schedule": _bounded_text(
                    flow.get("selected_syllable_schedule"), 1_024
                ),
            }
        )
    compact_scores = []
    for score in value["selected_scores"]:
        components = list(score.get("component_scores", {}).items())[:4]
        compact_scores.append(
            {
                "bar": score.get("bar"),
                "total": score.get("total"),
                "component_scores": dict(components),
            }
        )
    compact = {
        **value,
        "selected_lines": [
            _bounded_text(item, 256) for item in value["selected_lines"][:2]
        ],
        "flows": compact_flows,
        "selected_scores": compact_scores,
        "prompt_summary": _bounded_text(value.get("prompt_summary"), 1_024),
        "context_lines": [
            _bounded_text(item, 256) for item in value["context_lines"][:2]
        ],
        "stretch_warnings": [
            _bounded_text(item, 256) for item in value["stretch_warnings"][:4]
        ],
        "warnings": [_bounded_text(item, 256) for item in value["warnings"][:4]],
        "hashes": dict(list(value["hashes"].items())[:4]),
        "artifact_refs": dict(
            list(value["artifact_refs"].items())[:_MAX_ARTIFACT_REFS]
        ),
    }
    if _encoded_size(compact) <= MAX_BOUNDED_CHUNK_EVENT_BYTES:
        return compact

    return {
        "state": _bounded_text(value.get("state"), 128),
        "renderer_decision": _bounded_text(value.get("renderer_decision"), 128),
        "coordinator_epoch": value.get("coordinator_epoch"),
        "chunk_index": value.get("chunk_index"),
        "bars": value.get("bars", [])[:2],
        "selected_lines": [
            _bounded_text(item, 128) for item in value.get("selected_lines", [])[:2]
        ],
        "flows": [],
        "candidate_counts": value.get("candidate_counts", {}),
        "selected_scores": [],
        "prompt_summary": _bounded_text(value.get("prompt_summary"), 512),
        "context_lines": [],
        "stage_timings_ms": value.get("stage_timings_ms", {}),
        "request_budget_ms": value.get("request_budget_ms"),
        "elapsed_ms": value.get("elapsed_ms"),
        "deadline_slack_ms": value.get("deadline_slack_ms"),
        "alignment": value.get("alignment", {}),
        "stretch_warnings": [],
        "warnings": [],
        "hashes": {},
        "artifact_refs": value.get("artifact_refs", {}),
        "transfer_bytes": value.get("transfer_bytes"),
        "failure_reason": _bounded_text(value.get("failure_reason"), 256),
    }
