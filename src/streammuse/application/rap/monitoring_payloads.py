"""Stable JSON-ready monitoring payload builders for rap planning."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from streammuse.domain.rap import FlowTemplate, ScheduledSyllable


_MAX_CHUNK_LINES = 2
_MAX_CHUNK_FLOWS = 2
_MAX_FLOW_SLOTS = 32
_MAX_CONTEXT_LINES = 4
_MAX_COMPONENT_SCORES = 16
_MAX_DIAGNOSTIC_ITEMS = 8
_MAX_LINE_CHARACTERS = 512
_MAX_SUMMARY_CHARACTERS = 4_096
_MAX_REFERENCE_CHARACTERS = 1_024
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


def bounded_chunk_event_payload(value: object) -> dict[str, Any]:
    """Reduce remote chunk diagnostics to the fixed live-monitor contract."""

    payload = value if isinstance(value, Mapping) else {}
    manifest = _mapping(payload.get("manifest"))
    manifest_diagnostics = _mapping(manifest.get("diagnostics"))
    selected_bars = _mapping_items(manifest.get("selected_bars"), _MAX_CHUNK_LINES)

    selected_lines = _bounded_strings(payload.get("selected_lines"), _MAX_CHUNK_LINES)
    if not selected_lines:
        selected_lines = [
            text
            for item in selected_bars
            if (text := _bounded_text(item.get("text"), _MAX_LINE_CHARACTERS)) is not None
        ][:_MAX_CHUNK_LINES]

    raw_flows = _mapping_items(payload.get("flows"), _MAX_CHUNK_FLOWS)
    flows = [_bounded_flow(item) for item in raw_flows]
    if not flows:
        flows = [
            _bounded_flow({"template_id": item.get("flow_template_id"), "slots": []})
            for item in selected_bars
            if isinstance(item.get("flow_template_id"), str)
        ][:_MAX_CHUNK_FLOWS]

    candidate_source = _mapping(payload.get("candidate_counts")) or _mapping(
        manifest_diagnostics.get("candidate_stats")
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

    score_items = _mapping_items(payload.get("selected_scores"), _MAX_CHUNK_LINES)
    if not score_items:
        score_items = selected_bars
    selected_scores = [_bounded_selected_score(item) for item in score_items[:_MAX_CHUNK_LINES]]

    timing_source = _mapping(payload.get("stage_timings_ms")) or _mapping(
        manifest_diagnostics.get("stage_timings_ms")
    )
    stage_timings = {
        name: timing
        for name, aliases in _CHUNK_TIMING_ALIASES.items()
        if (timing := _first_finite_number(timing_source, aliases)) is not None
    }
    transfer = _mapping(payload.get("transfer"))
    if "transfer" not in stage_timings:
        transfer_ms = _first_finite_number(transfer, ("total_ms", "response_ms", "download_ms"))
        if transfer_ms is not None:
            stage_timings["transfer"] = transfer_ms
    if "mac" not in stage_timings:
        mac_ms = _first_finite_number(
            payload,
            ("mac_validation_mix_ms", "mac_timing_ms", "mac_ms"),
        )
        if mac_ms is not None:
            stage_timings["mac"] = mac_ms

    alignment_source = _mapping(payload.get("alignment")) or _mapping(
        payload.get("alignment_diagnostics")
    ) or _mapping(manifest_diagnostics.get("alignment_diagnostics"))
    fallback_counts = _bounded_integer_mapping(alignment_source.get("fallback_counts"))
    alignment = {
        "method": _bounded_text(
            _first_value(alignment_source, ("method", "alignment_method", "aligner")),
            _MAX_LINE_CHARACTERS,
        ),
        "confidence": _first_finite_number(
            alignment_source,
            ("confidence", "mean_confidence", "minimum_confidence"),
        ),
        "fallback_counts": fallback_counts,
    }

    diagnostic_warnings = _bounded_strings(
        manifest_diagnostics.get("warnings"),
        _MAX_DIAGNOSTIC_ITEMS,
    )
    warnings = _unique_bounded_strings(
        [*_bounded_strings(payload.get("warnings"), _MAX_DIAGNOSTIC_ITEMS), *diagnostic_warnings],
        _MAX_DIAGNOSTIC_ITEMS,
    )
    stretch_warnings = _unique_bounded_strings(
        [
            *_bounded_strings(payload.get("stretch_warnings"), _MAX_DIAGNOSTIC_ITEMS),
            *(item for item in warnings if any(token in item.lower() for token in ("stretch", "warp", "ratio"))),
        ],
        _MAX_DIAGNOSTIC_ITEMS,
    )

    hashes = _bounded_string_mapping(payload.get("hashes"))
    vocal_hash = _bounded_text(manifest.get("vocal_sha256"), _MAX_REFERENCE_CHARACTERS)
    if vocal_hash is not None:
        hashes.setdefault("vocal_sha256", vocal_hash)

    transfer_bytes = _first_nonnegative_integer(payload, ("transfer_bytes",))
    if transfer_bytes is None:
        transfer_bytes = _first_nonnegative_integer(transfer, ("response_bytes", "bytes"))

    return {
        "state": _bounded_text(payload.get("state"), _MAX_LINE_CHARACTERS),
        "renderer_decision": _bounded_text(
            _first_value(payload, ("renderer_decision", "renderer_commitment", "renderer")),
            _MAX_LINE_CHARACTERS,
        ),
        "chunk_index": _nonnegative_integer(payload.get("chunk_index")),
        "bars": _bounded_integers(payload.get("bars"), _MAX_CHUNK_LINES),
        "selected_lines": selected_lines,
        "flows": flows,
        "candidate_counts": candidate_counts,
        "selected_scores": selected_scores,
        "prompt_summary": _bounded_text(payload.get("prompt_summary"), _MAX_SUMMARY_CHARACTERS),
        "context_lines": _bounded_strings(payload.get("context_lines"), _MAX_CONTEXT_LINES),
        "stage_timings_ms": stage_timings,
        "request_budget_ms": _first_finite_number(
            payload,
            ("request_budget_ms", "accepted_request_budget_ms"),
        )
        or _first_finite_number(manifest_diagnostics, ("accepted_request_budget_ms",)),
        "elapsed_ms": _first_finite_number(payload, ("elapsed_ms", "request_elapsed_ms")),
        "deadline_slack_ms": _first_finite_number(
            payload,
            ("deadline_slack_ms", "final_deadline_slack_ms"),
        ),
        "alignment": alignment,
        "stretch_warnings": stretch_warnings,
        "warnings": warnings,
        "hashes": hashes,
        "artifact_refs": _bounded_string_mapping(payload.get("artifact_refs")),
        "transfer_bytes": transfer_bytes,
        "failure_reason": _bounded_text(
            _first_value(payload, ("failure_reason", "error_message")),
            _MAX_LINE_CHARACTERS,
        ),
    }


def _bounded_flow(value: Mapping[str, object]) -> dict[str, Any]:
    slots = []
    for item in _mapping_items(value.get("slots"), _MAX_FLOW_SLOTS):
        tick = _nonnegative_integer(item.get("tick_in_bar"))
        stress = _finite_number(item.get("target_stress"))
        slot: dict[str, object] = {"tick_in_bar": tick, "target_stress": stress}
        for key in ("duration_ticks", "boundary_strength"):
            number = _finite_number(item.get(key))
            if number is not None:
                slot[key] = number
        rhyme = _bounded_text(item.get("rhyme_group"), _MAX_LINE_CHARACTERS)
        if rhyme is not None:
            slot["rhyme_group"] = rhyme
        slots.append(slot)
    schedule = ", ".join(
        f"t{item['tick_in_bar']}@{item['target_stress']:.2f}"
        if item["tick_in_bar"] is not None and item["target_stress"] is not None
        else f"t{item['tick_in_bar']}@?"
        for item in slots
    )
    result: dict[str, Any] = {
        "template_id": _bounded_text(value.get("template_id"), _MAX_LINE_CHARACTERS),
        "name": _bounded_text(value.get("name"), _MAX_LINE_CHARACTERS),
        "slots": slots,
        "slot_stress_schedule": schedule,
    }
    for key in ("ticks_per_beat", "beats_per_bar"):
        number = _nonnegative_integer(value.get(key))
        if number is not None:
            result[key] = number
    return result


def _bounded_selected_score(value: Mapping[str, object]) -> dict[str, Any]:
    diagnostics = _mapping(value.get("diagnostics"))
    component_source = _mapping(value.get("component_scores")) or _mapping(
        diagnostics.get("component_scores")
    )
    component_scores = {
        str(key)[:_MAX_LINE_CHARACTERS]: score
        for key, item in tuple(component_source.items())[:_MAX_COMPONENT_SCORES]
        if isinstance(key, str) and (score := _finite_number(item)) is not None
    }
    return {
        "bar": _nonnegative_integer(value.get("bar")),
        "total": _first_finite_number(value, ("total", "score", "total_score")),
        "component_scores": component_scores,
    }


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _mapping_items(value: object, limit: int) -> list[Mapping[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, Mapping)][:limit]


def _first_value(value: Mapping[str, object], keys: tuple[str, ...]) -> object:
    return next((value[key] for key in keys if key in value), None)


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        return None
    return float(value)


def _first_finite_number(value: Mapping[str, object], keys: tuple[str, ...]) -> float | None:
    return _finite_number(_first_value(value, keys))


def _nonnegative_integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_nonnegative_integer(value: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    return _nonnegative_integer(_first_value(value, keys))


def _bounded_text(value: object, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def _bounded_strings(value: object, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item[:_MAX_LINE_CHARACTERS] for item in value if isinstance(item, str)][:limit]


def _unique_bounded_strings(value: list[str], limit: int) -> list[str]:
    result = []
    for item in value:
        bounded = item[:_MAX_LINE_CHARACTERS]
        if bounded not in result:
            result.append(bounded)
        if len(result) == limit:
            break
    return result


def _bounded_integers(value: object, limit: int) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if _nonnegative_integer(item) is not None][:limit]


def _bounded_integer_mapping(value: object) -> dict[str, int]:
    mapping = _mapping(value)
    return {
        key[:_MAX_LINE_CHARACTERS]: count
        for key, item in tuple(mapping.items())[:_MAX_DIAGNOSTIC_ITEMS]
        if isinstance(key, str) and (count := _nonnegative_integer(item)) is not None
    }


def _bounded_string_mapping(value: object) -> dict[str, str]:
    mapping = _mapping(value)
    return {
        key[:_MAX_LINE_CHARACTERS]: item[:_MAX_REFERENCE_CHARACTERS]
        for key, item in tuple(mapping.items())[:_MAX_DIAGNOSTIC_ITEMS]
        if isinstance(key, str) and isinstance(item, str)
    }
