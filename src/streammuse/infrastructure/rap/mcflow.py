"""Anonymous structural extraction from user-supplied MCFlow Humdrum files."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from streammuse.domain.rap.flow import FlowProvenance, FlowSlot, FlowTemplate


SCHEMA_VERSION = "streammuse.mcflow_templates.v1"
EXTRACTOR_VERSION = "streammuse.mcflow.v1"
_REQUIRED_SPINES = frozenset({"**recip", "**stress", "**break", "**rhyme", "**lyrics"})
_RECIPROCAL = re.compile(r"^(?P<number>[1-9][0-9]*)(?P<dots>\.*)(?:%(?P<numerator>[1-9][0-9]*))?(?P<rest>r)?$")


@dataclass(frozen=True)
class AnonymousSyllable:
    """Structural syllable data with no lyric or pronunciation text."""

    onset: Fraction
    duration: Fraction
    stress: float
    rhyme_group: str | None


@dataclass(frozen=True)
class PhraseStart:
    """A structural phrase start retained independently of lyric-bearing slots."""

    onset: Fraction
    strength: int


@dataclass(frozen=True)
class ParsedMeasure:
    """One parsed Humdrum measure represented solely by anonymous structure."""

    ordinal: int
    duration: Fraction
    meter: tuple[int, int] | None
    syllables: tuple[AnonymousSyllable, ...]
    phrase_starts: tuple[PhraseStart, ...]


@dataclass(frozen=True)
class ParsedMcFlow:
    """Anonymous contents of one parsed source file."""

    source_hash: str
    measures: tuple[ParsedMeasure, ...]


@dataclass(frozen=True)
class ExtractionRejection:
    """Anonymous reason why one source measure cannot form a template."""

    source_hash: str
    measure_ordinal: int
    error_code: str
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source_hash": self.source_hash,
            "measure_ordinal": self.measure_ordinal,
            "error_code": self.error_code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ExtractionResult:
    """Anonymous accepted templates and rejected measures."""

    templates: tuple[FlowTemplate, ...]
    rejections: tuple[ExtractionRejection, ...]
    parsed_files: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "extractor_version": EXTRACTOR_VERSION,
            "templates": [flow_template_to_dict(template) for template in self.templates],
            "rejections": [rejection.to_dict() for rejection in self.rejections],
            "aggregate": {
                "parsed_files": self.parsed_files,
                "accepted_templates": len(self.templates),
                "rejected_measures": len(self.rejections),
            },
        }


def parse_reciprocal_duration(token: str) -> Fraction:
    """Parse MCFlow reciprocal duration notation into an exact whole-note value."""
    match = _RECIPROCAL.fullmatch(token)
    if match is None:
        raise ValueError("unsupported reciprocal duration")
    denominator = int(match.group("number"))
    numerator = match.group("numerator")
    dots = match.group("dots")
    if numerator is not None:
        if dots:
            raise ValueError("rational reciprocal duration cannot use augmentation dots")
        return Fraction(int(numerator), denominator)
    return Fraction(1, denominator) * (2 - Fraction(1, 2 ** len(dots)))


def parse_mcflow_file(path: str | Path) -> ParsedMcFlow:
    """Parse required Humdrum spines while discarding all source text content."""
    raw = Path(path).read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    lines = raw.decode("utf-8").splitlines()
    exclusive_index, columns = _find_exclusive_interpretations(lines)
    spine_index = {name: index for index, name in enumerate(columns)}

    measures: list[ParsedMeasure] = []
    syllables: list[AnonymousSyllable] = []
    phrase_starts: list[PhraseStart] = []
    meter: tuple[int, int] | None = None
    current_duration = Fraction(0)
    measure_ordinal = 1
    measure_started = False
    previous_stress = 0.0

    def finish_measure() -> None:
        nonlocal syllables, phrase_starts, current_duration
        measures.append(
            ParsedMeasure(
                ordinal=measure_ordinal,
                duration=current_duration,
                meter=meter,
                syllables=tuple(syllables),
                phrase_starts=tuple(phrase_starts),
            )
        )
        syllables = []
        phrase_starts = []
        current_duration = Fraction(0)

    for raw_line in lines[exclusive_index + 1 :]:
        if not raw_line or raw_line.startswith("!"):
            continue
        fields = raw_line.split("\t")
        if len(fields) != len(columns):
            raise ValueError("record width does not match exclusive interpretations")
        if all(field == "*-" for field in fields):
            break
        if raw_line.startswith("*"):
            for field in fields:
                if field.startswith("*M") and len(field) > 2 and field[2].isdigit():
                    meter = _parse_meter(field)
            continue
        if raw_line.startswith("="):
            if measure_started:
                finish_measure()
                measure_ordinal += 1
            measure_started = True
            continue

        measure_started = True
        reciprocal = fields[spine_index["**recip"]]
        duration = parse_reciprocal_duration(reciprocal)
        lyric = fields[spine_index["**lyrics"]]
        is_rest = reciprocal.endswith("r") or lyric == "R"
        stress_token = fields[spine_index["**stress"]]
        if stress_token == ".":
            stress = previous_stress
        else:
            stress = _parse_stress(stress_token)
            previous_stress = stress
        break_strength = _parse_break(fields[spine_index["**break"]])
        if break_strength:
            phrase_starts.append(PhraseStart(onset=current_duration, strength=break_strength))
        if not is_rest:
            rhyme = fields[spine_index["**rhyme"]]
            syllables.append(
                AnonymousSyllable(
                    onset=current_duration,
                    duration=duration,
                    stress=stress,
                    rhyme_group=None if rhyme == "." else rhyme,
                )
            )
        current_duration += duration
    if measure_started:
        finish_measure()
    return ParsedMcFlow(source_hash=source_hash, measures=tuple(measures))


def extract_anonymous_templates(
    path: str | Path, *, max_quantization_error_ticks: float = 0.25
) -> ExtractionResult:
    """Extract anonymous validated flow templates from one user-supplied file."""
    parsed = parse_mcflow_file(path)
    return _extract_parsed((parsed,), max_quantization_error_ticks=max_quantization_error_ticks)


def extract_mcflow_directory(
    path: str | Path, *, max_quantization_error_ticks: float = 0.25
) -> ExtractionResult:
    """Recursively extract user-supplied ``.rap`` files in deterministic order."""
    directory = Path(path)
    if not directory.is_dir():
        raise ValueError("mcflow directory must be an existing directory")
    files = sorted(directory.rglob("*.rap"), key=lambda item: item.relative_to(directory).as_posix())
    parsed = tuple(parse_mcflow_file(file_path) for file_path in files)
    return _extract_parsed(parsed, max_quantization_error_ticks=max_quantization_error_ticks)


def flow_template_to_dict(template: FlowTemplate) -> dict[str, object]:
    """Serialize a template through the anonymous catalog whitelist."""
    _validate_anonymous_template(template)
    return {
        "template_id": template.template_id,
        "name": template.name,
        "ticks_per_beat": template.ticks_per_beat,
        "beats_per_bar": template.beats_per_bar,
        "slots": [
            {
                "tick_in_bar": slot.tick_in_bar,
                "duration_ticks": slot.duration_ticks,
                "target_stress": slot.target_stress,
                "boundary_strength": slot.boundary_strength,
                "rhyme_group": slot.rhyme_group,
            }
            for slot in template.slots
        ],
        "provenance": {
            "kind": template.provenance.kind,
            "source": template.provenance.source,
            "source_hash": template.provenance.source_hash,
            "quantization_error_ticks": template.provenance.quantization_error_ticks,
        },
    }


def write_extracted_templates(extraction: ExtractionResult, path: str | Path) -> None:
    """Write one deterministic anonymous catalog JSON document."""
    Path(path).write_text(json.dumps(extraction.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_extracted_templates(path: str | Path) -> tuple[FlowTemplate, ...]:
    """Load and validate FlowTemplates from an anonymous extracted catalog."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid extracted template catalog") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported extracted template schema")
    _require_keys(payload, {"schema_version", "extractor_version", "templates", "rejections", "aggregate"})
    templates = payload["templates"]
    if not isinstance(templates, list):
        raise ValueError("invalid extracted template catalog")
    _validate_catalog_sections(payload)
    loaded = tuple(_flow_template_from_dict(item) for item in templates)
    for template in loaded:
        _validate_anonymous_template(template)
    return loaded


def _find_exclusive_interpretations(lines: list[str]) -> tuple[int, list[str]]:
    for index, raw_line in enumerate(lines):
        if raw_line.startswith("**"):
            columns = raw_line.split("\t")
            missing = _REQUIRED_SPINES.difference(columns)
            if missing:
                raise ValueError(f"missing required spine: {sorted(missing)[0][2:]}")
            if len(set(columns)) != len(columns):
                raise ValueError("exclusive interpretations must not repeat spines")
            return index, columns
    raise ValueError("missing exclusive interpretations")


def _parse_meter(field: str) -> tuple[int, int]:
    match = re.fullmatch(r"\*M([1-9][0-9]*)/([1-9][0-9]*)", field)
    if match is None:
        raise ValueError("invalid meter interpretation")
    return int(match.group(1)), int(match.group(2))


def _parse_stress(value: str) -> float:
    if value == "0":
        return 0.0
    if value in {"1", "2"}:
        return 1.0
    raise ValueError("invalid stress value")


def _parse_break(value: str) -> int:
    if value in {".", "0"}:
        return 0
    if value in {"1", "2", "3", "4", "5"}:
        return int(value)
    raise ValueError("invalid break value")


def _extract_parsed(
    parsed_files: tuple[ParsedMcFlow, ...], *, max_quantization_error_ticks: float
) -> ExtractionResult:
    if max_quantization_error_ticks < 0:
        raise ValueError("max quantization error ticks must not be negative")
    limit = Fraction(str(max_quantization_error_ticks))
    drafts: list[tuple[ParsedMeasure, list[FlowSlot], float, str]] = []
    rejections: list[ExtractionRejection] = []
    for parsed in parsed_files:
        previous: tuple[ParsedMeasure, list[FlowSlot], float, str] | None = None
        for measure in parsed.measures:
            slots, error, rejection = _quantize_measure(measure, limit)
            if rejection is not None:
                rejections.append(
                    ExtractionRejection(
                        source_hash=parsed.source_hash,
                        measure_ordinal=measure.ordinal,
                        error_code=rejection[0],
                        detail=rejection[1],
                    )
                )
                if measure.phrase_starts:
                    rejections.append(
                        ExtractionRejection(
                            source_hash=parsed.source_hash,
                            measure_ordinal=measure.ordinal,
                            error_code="unrepresentable_phrase_break",
                            detail="phrase break cannot be represented in a rejected measure",
                        )
                    )
                previous = None
            else:
                draft = (measure, slots, error, parsed.source_hash)
                if _apply_phrase_starts(measure, slots, previous):
                    rejections.append(
                        ExtractionRejection(
                            source_hash=parsed.source_hash,
                            measure_ordinal=measure.ordinal,
                            error_code="unrepresentable_phrase_break",
                            detail="phrase break has no immediately preceding lyric-bearing slot",
                        )
                    )
                    previous = None
                else:
                    drafts.append(draft)
                    previous = draft

    templates = tuple(
        FlowTemplate(
            template_id=_template_id(source_hash, measure.ordinal),
            name=f"anonymous_measure_{measure.ordinal}",
            ticks_per_beat=4,
            beats_per_bar=4,
            slots=tuple(slots),
            provenance=FlowProvenance(
                kind="mcflow_extracted_anonymous",
                source="anonymous_mcflow",
                source_hash=source_hash,
                quantization_error_ticks=error,
            ),
        )
        for measure, slots, error, source_hash in drafts
    )
    return ExtractionResult(templates=templates, rejections=tuple(rejections), parsed_files=len(parsed_files))


def _quantize_measure(
    measure: ParsedMeasure, limit: Fraction
) -> tuple[list[FlowSlot], float, tuple[str, str] | None]:
    if measure.meter != (4, 4):
        return [], 0.0, ("non_4_4_meter", "measure meter is not 4/4")
    if not measure.syllables:
        return [], 0.0, ("empty_measure", "measure has no lyric-bearing slots")
    if measure.duration < 1:
        return [], 0.0, ("incomplete_measure", "measure duration is shorter than four beats")
    if measure.duration > 1:
        return [], 0.0, ("overfull_measure", "measure duration is longer than four beats")
    slots: list[FlowSlot] = []
    max_error = Fraction(0)
    for syllable in measure.syllables:
        onset = syllable.onset * 16
        duration = syllable.duration * 16
        onset_tick = _nearest_tick(onset)
        duration_tick = _nearest_tick(duration)
        error = max(abs(onset - onset_tick), abs(duration - duration_tick))
        max_error = max(max_error, error)
        if error > limit:
            return [], 0.0, ("quantization_error", "onset or duration exceeds quantization tolerance")
        if onset_tick < 0 or onset_tick >= 16 or onset_tick + duration_tick > 16:
            return [], 0.0, ("slot_outside_bar", "quantized slot lies outside the bar")
        if duration_tick <= 0:
            return [], 0.0, ("nonpositive_duration", "quantized slot duration is not positive")
        slots.append(
            FlowSlot(
                tick_in_bar=onset_tick,
                duration_ticks=duration_tick,
                target_stress=syllable.stress,
                rhyme_group=syllable.rhyme_group,
            )
        )
    if len({slot.tick_in_bar for slot in slots}) != len(slots):
        return [], 0.0, ("duplicate_quantized_onset", "multiple slots share a quantized onset")
    return slots, float(max_error), None


def _nearest_tick(value: Fraction) -> int:
    return (value.numerator * 2 + value.denominator) // (value.denominator * 2)


def _replace_boundary(slots: list[FlowSlot], index: int, strength: int) -> None:
    slot = slots[index]
    slots[index] = FlowSlot(
        tick_in_bar=slot.tick_in_bar,
        duration_ticks=slot.duration_ticks,
        target_stress=slot.target_stress,
        boundary_strength=strength,
        rhyme_group=slot.rhyme_group,
    )


def _apply_phrase_starts(
    measure: ParsedMeasure,
    slots: list[FlowSlot],
    previous: tuple[ParsedMeasure, list[FlowSlot], float, str] | None,
) -> bool:
    for phrase_start in measure.phrase_starts:
        preceding = [index for index, syllable in enumerate(measure.syllables) if syllable.onset < phrase_start.onset]
        if preceding:
            _replace_boundary(slots, preceding[-1], phrase_start.strength)
        elif previous is not None:
            _replace_boundary(previous[1], len(previous[1]) - 1, phrase_start.strength)
        else:
            return True
    return False


def _template_id(source_hash: str, measure_ordinal: int) -> str:
    identity = f"{source_hash}:{measure_ordinal}".encode("ascii")
    return f"mcflow_{hashlib.sha256(identity).hexdigest()[:20]}"


def _validate_catalog_sections(payload: dict[str, Any]) -> None:
    if not isinstance(payload["extractor_version"], str):
        raise ValueError("invalid extracted template catalog")
    if not isinstance(payload["rejections"], list) or not isinstance(payload["aggregate"], dict):
        raise ValueError("invalid extracted template catalog")
    for rejection in payload["rejections"]:
        _validate_rejection(rejection)
    aggregate = payload["aggregate"]
    _require_keys(aggregate, {"parsed_files", "accepted_templates", "rejected_measures"})
    if any(_nonnegative_integer(aggregate[key]) is None for key in aggregate):
        raise ValueError("invalid extracted template catalog")
    if aggregate["accepted_templates"] != len(payload["templates"]):
        raise ValueError("invalid extracted template catalog")
    if aggregate["rejected_measures"] != len(payload["rejections"]):
        raise ValueError("invalid extracted template catalog")


def _flow_template_from_dict(value: object) -> FlowTemplate:
    if not isinstance(value, dict):
        raise ValueError("invalid extracted template catalog")
    _require_keys(value, {"template_id", "name", "ticks_per_beat", "beats_per_bar", "slots", "provenance"})
    slots_value = value["slots"]
    provenance_value = value["provenance"]
    if not isinstance(slots_value, list) or not isinstance(provenance_value, dict):
        raise ValueError("invalid extracted template catalog")
    _require_keys(provenance_value, {"kind", "source", "source_hash", "quantization_error_ticks"})
    try:
        slots = tuple(_flow_slot_from_dict(slot) for slot in slots_value)
        provenance = FlowProvenance(
            kind=_string(provenance_value["kind"]),
            source=_string(provenance_value["source"]),
            source_hash=_optional_string(provenance_value["source_hash"]),
            quantization_error_ticks=_number(provenance_value["quantization_error_ticks"]),
        )
        return FlowTemplate(
            template_id=_string(value["template_id"]),
            name=_string(value["name"]),
            ticks_per_beat=_integer(value["ticks_per_beat"]),
            beats_per_bar=_integer(value["beats_per_bar"]),
            slots=slots,
            provenance=provenance,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid extracted template catalog") from exc


def _flow_slot_from_dict(value: object) -> FlowSlot:
    if not isinstance(value, dict):
        raise ValueError("invalid extracted template catalog")
    _require_keys(value, {"tick_in_bar", "duration_ticks", "target_stress", "boundary_strength", "rhyme_group"})
    return FlowSlot(
        tick_in_bar=_integer(value["tick_in_bar"]),
        duration_ticks=_integer(value["duration_ticks"]),
        target_stress=_number(value["target_stress"]),
        boundary_strength=_integer(value["boundary_strength"]),
        rhyme_group=_optional_string(value["rhyme_group"]),
    )


def _require_keys(value: dict[str, object], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("invalid extracted template catalog")


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("invalid extracted template catalog")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("invalid extracted template catalog")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("invalid extracted template catalog")
    return result


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid extracted template catalog")
    return value


def _optional_string(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("invalid extracted template catalog")
    return value


def _validate_rejection(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("invalid extracted template catalog")
    _require_keys(value, {"source_hash", "measure_ordinal", "error_code", "detail"})
    if not _is_source_hash(value["source_hash"]):
        raise ValueError("invalid extracted template catalog")
    if _nonnegative_integer(value["measure_ordinal"]) is None or value["measure_ordinal"] < 1:
        raise ValueError("invalid extracted template catalog")
    if not isinstance(value["error_code"], str) or not value["error_code"]:
        raise ValueError("invalid extracted template catalog")
    if not isinstance(value["detail"], str) or not value["detail"]:
        raise ValueError("invalid extracted template catalog")


def _nonnegative_integer(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _is_source_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_anonymous_template(template: FlowTemplate) -> None:
    match = re.fullmatch(r"anonymous_measure_([1-9][0-9]*)", template.name)
    if match is None:
        raise ValueError("invalid anonymous extracted template")
    if template.provenance.kind != "mcflow_extracted_anonymous" or template.provenance.source != "anonymous_mcflow":
        raise ValueError("invalid anonymous extracted template")
    if not _is_source_hash(template.provenance.source_hash):
        raise ValueError("invalid anonymous extracted template")
    error = template.provenance.quantization_error_ticks
    if not math.isfinite(error) or error < 0:
        raise ValueError("invalid anonymous extracted template")
    expected_id = _template_id(template.provenance.source_hash, int(match.group(1)))
    if template.template_id != expected_id:
        raise ValueError("invalid anonymous extracted template")
