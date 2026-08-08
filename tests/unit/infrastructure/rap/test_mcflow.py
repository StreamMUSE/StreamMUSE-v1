"""Tests for anonymous MCFlow structure extraction."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from streammuse.infrastructure.rap.mcflow import (
    extract_anonymous_templates,
    extract_mcflow_directory,
    flow_template_to_dict,
    load_extracted_templates,
    parse_mcflow_file,
    write_extracted_templates,
)


FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "rap" / "mcflow_minimal.rap"


def test_parse_mcflow_file_keeps_exact_anonymous_durations() -> None:
    """Catches a parser that converts reciprocal notation to floats or retains text."""
    parsed = parse_mcflow_file(FIXTURE)

    assert parsed.measures[0].syllables[2].duration == Fraction(3, 16)
    assert parsed.measures[1].syllables[0].duration == Fraction(13, 16)
    assert parsed.measures[1].duration == Fraction(1, 1)
    assert all("lyric" not in field and "ipa" not in field for item in parsed.measures for field in vars(item.syllables[0]))


def test_extract_anonymous_templates_preserves_varied_structure_and_anonymity() -> None:
    """Catches fixed nine-slot extraction, dropped rests, or leaked source text."""
    extraction = extract_anonymous_templates(FIXTURE)
    templates = extraction.templates
    payload = json.dumps(extraction.to_dict(), sort_keys=True)

    assert [len(template.slots) for template in templates] == [4, 1]
    assert [slot.tick_in_bar for slot in templates[0].slots] == [0, 1, 2, 5]
    assert [slot.duration_ticks for slot in templates[0].slots] == [1, 1, 3, 6]
    assert [slot.target_stress for slot in templates[0].slots] == [1.0, 0.0, 1.0, 1.0]
    assert templates[0].slots[-1].boundary_strength == 3
    assert templates[1].slots[0].rhyme_group == "C"
    assert templates[0].provenance.quantization_error_ticks == 0.0
    assert "tav" not in payload
    assert "/ta/" not in payload
    assert FIXTURE.name not in payload
    assert str(FIXTURE) not in payload


def test_same_bytes_produce_stable_anonymous_ids_and_hashes(tmp_path: Path) -> None:
    """Catches identifiers that depend on a supplied file name or location."""
    copied = tmp_path / "renamed.rap"
    copied.write_bytes(FIXTURE.read_bytes())

    original = extract_anonymous_templates(FIXTURE)
    duplicate = extract_anonymous_templates(copied)

    assert [template.template_id for template in duplicate.templates] == [template.template_id for template in original.templates]
    assert [template.provenance.source_hash for template in duplicate.templates] == [
        template.provenance.source_hash for template in original.templates
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    (
        ("**recip\t**stress\t**break\t**rhyme\n", "missing required spine: lyrics"),
        ("**recip\t**stress\t**break\t**rhyme\t**lyrics\n16\t1\t.\tA\n", "record width does not match exclusive interpretations"),
    ),
)
def test_parse_mcflow_file_rejects_missing_or_malformed_spines(tmp_path: Path, content: str, message: str) -> None:
    """Catches silent acceptance of records that cannot be interpreted safely."""
    malformed = tmp_path / "input.rap"
    malformed.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        parse_mcflow_file(malformed)


def test_parse_mcflow_file_ignores_non_meter_tandem_interpretations(tmp_path: Path) -> None:
    """Catches tempo-style tandem records being treated as malformed meters."""
    source = tmp_path / "input.rap"
    tandem = b"\t".join([b"*MM=99"] * 8) + b"\n"
    source.write_bytes(FIXTURE.read_bytes().replace(b"=1\t=1\t=1\t=1\t=1\t=1\t=1\t=1\n", tandem + b"=1\t=1\t=1\t=1\t=1\t=1\t=1\t=1\n", 1))

    parsed = parse_mcflow_file(source)

    assert [measure.duration for measure in parsed.measures] == [Fraction(1, 1), Fraction(1, 1)]


def test_parse_mcflow_file_carries_humdrum_null_stress_values(tmp_path: Path) -> None:
    """Catches a null stress continuation being rejected as a new stress value."""
    source = tmp_path / "input.rap"
    source.write_bytes(FIXTURE.read_bytes().replace(b"16\t0\t.\t.\t.\t/ko/", b"16\t.\t.\t.\t.\t/ko/", 1))

    parsed = parse_mcflow_file(source)

    assert parsed.measures[0].syllables[1].stress == 1.0


def test_parse_mcflow_file_defaults_an_initial_null_stress_to_unaccented(tmp_path: Path) -> None:
    """Catches an initial Humdrum null stress preventing otherwise valid extraction."""
    source = tmp_path / "input.rap"
    source.write_bytes(FIXTURE.read_bytes().replace(b"16\t1\t.\t.\tA\t/ta/", b"16\t.\t.\t.\tA\t/ta/", 1))

    parsed = parse_mcflow_file(source)

    assert parsed.measures[0].syllables[0].stress == 0.0


def test_extract_records_anonymous_quantization_rejections(tmp_path: Path) -> None:
    """Catches measures skipped without a structured explanation."""
    source = tmp_path / "unquantized.rap"
    source.write_text(
        "\n".join(
            (
                "**lyrics\t**rhyme\t**break\t**stress\t**recip",
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4",
                "=1\t=1\t=1\t=1\t=1",
                "vek\tA\t.\t1\t64",
                "R\t.\t.\t.\t64r",
                "*-\t*-\t*-\t*-\t*-",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extraction = extract_anonymous_templates(source, max_quantization_error_ticks=0.0)

    assert not extraction.templates
    assert extraction.rejections[0].error_code == "incomplete_measure"
    assert extraction.rejections[0].measure_ordinal == 1
    assert extraction.rejections[0].source_hash
    assert "unquantized" not in json.dumps(extraction.to_dict())


def test_complete_off_grid_measure_respects_quantization_tolerance_boundary(tmp_path: Path) -> None:
    """Catches off-grid structures being accepted above or rejected at the configured tolerance."""
    source = tmp_path / "off-grid.rap"
    source.write_text(
        "\n".join(
            (
                "**recip\t**stress\t**break\t**rhyme\t**lyrics",
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4",
                "=1\t=1\t=1\t=1\t=1",
                "40%3\t1\t.\tA\tza",
                "5\t0\t.\t.\tzb",
                "40%29r\t.\t.\t.\tR",
                "*-\t*-\t*-\t*-\t*-",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    accepted = extract_anonymous_templates(source, max_quantization_error_ticks=0.2)
    rejected = extract_anonymous_templates(source, max_quantization_error_ticks=0.19)

    assert len(accepted.templates) == 1
    assert accepted.templates[0].provenance.quantization_error_ticks == pytest.approx(0.2)
    assert [(slot.tick_in_bar, slot.duration_ticks) for slot in accepted.templates[0].slots] == [(0, 1), (1, 3)]
    assert not rejected.templates
    assert [(item.measure_ordinal, item.error_code) for item in rejected.rejections] == [(1, "quantization_error")]


def test_extract_records_invalid_break_values_without_stopping_other_measures(tmp_path: Path) -> None:
    """Catches one unsupported prosodic value aborting a whole local corpus extraction."""
    source = tmp_path / "input.rap"
    source.write_text(
        "\n".join(
            (
                "**recip\t**stress\t**break\t**rhyme\t**lyrics",
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4",
                "=1\t=1\t=1\t=1\t=1",
                "4\t1\t?\tA\tza",
                "4\t0\t.\t.\tzb",
                "4\t1\t.\t.\tzc",
                "4\t0\t.\t.\tzd",
                "=2\t=2\t=2\t=2\t=2",
                "4\t1\t.\tB\tze",
                "4\t0\t.\t.\tzf",
                "4\t1\t.\t.\tzg",
                "4\t0\t.\t.\tzh",
                "*-\t*-\t*-\t*-\t*-",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extraction = extract_anonymous_templates(source)

    assert [template.name for template in extraction.templates] == ["anonymous_measure_2"]
    assert [(item.measure_ordinal, item.error_code) for item in extraction.rejections] == [(1, "invalid_break_value")]


def test_extract_records_reciprocal_and_stress_parse_failures_per_measure(tmp_path: Path) -> None:
    """Catches one malformed data record aborting later valid measures from the same source."""
    source = tmp_path / "input.rap"
    source.write_text(
        "\n".join(
            (
                "**recip\t**stress\t**break\t**rhyme\t**lyrics",
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4",
                "=1\t=1\t=1\t=1\t=1",
                "bad_recip\t1\t.\tA\tza",
                "4\t0\t.\t.\tzb",
                "4\t1\t.\t.\tzc",
                "4\t0\t.\t.\tzd",
                "=2\t=2\t=2\t=2\t=2",
                "4\tbad_stress\t.\tB\tze",
                "4\t0\t.\t.\tzf",
                "4\t1\t.\t.\tzg",
                "4\t0\t.\t.\tzh",
                "=3\t=3\t=3\t=3\t=3",
                "4\t1\t.\tC\tzi",
                "4\t0\t.\t.\tzj",
                "4\t1\t.\t.\tzk",
                "4\t0\t.\t.\tzl",
                "*-\t*-\t*-\t*-\t*-",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extraction = extract_anonymous_templates(source)
    payload = json.dumps(extraction.to_dict())

    assert [template.name for template in extraction.templates] == ["anonymous_measure_3"]
    assert [(item.measure_ordinal, item.error_code) for item in extraction.rejections] == [
        (1, "invalid_reciprocal_duration"),
        (2, "invalid_stress_value"),
    ]
    assert "bad_recip" not in payload
    assert "bad_stress" not in payload


def test_extract_records_empty_measure_without_reusing_its_ordinal(tmp_path: Path) -> None:
    """Catches empty measures being omitted and causing ordinal reuse."""
    source = tmp_path / "input.rap"
    empty_then_third = b"=2\t=2\t=2\t=2\t=2\t=2\t=2\t=2\n=3\t=3\t=3\t=3\t=3\t=3\t=3\t=3\n"
    content = FIXTURE.read_bytes().replace(b"=2\t=2\t=2\t=2\t=2\t=2\t=2\t=2\n", empty_then_third, 1)
    source.write_bytes(content.replace(b"16%13\t1\t.\t3\tC", b"16%13\t1\t.\t.\tC", 1))

    extraction = extract_anonymous_templates(source)

    assert [template.name for template in extraction.templates] == ["anonymous_measure_1", "anonymous_measure_3"]
    assert [(item.measure_ordinal, item.error_code) for item in extraction.rejections] == [(2, "empty_measure")]


def test_extract_shifts_a_phrase_break_annotated_on_a_rest(tmp_path: Path) -> None:
    """Catches phrase-break annotations on rests being silently discarded."""
    source = tmp_path / "input.rap"
    content = FIXTURE.read_bytes()
    content = content.replace(b"4r\t.\t.\t.\t.\t.\tR", b"4r\t.\t.\t4\t.\t.\tR", 1)
    content = content.replace(b"16%13\t1\t.\t3\tC", b"16%13\t1\t.\t.\tC", 1)
    source.write_bytes(content)

    extraction = extract_anonymous_templates(source)

    assert extraction.templates[0].slots[-1].boundary_strength == 4


def test_extract_rejects_phrase_break_that_would_cross_a_rejected_measure(tmp_path: Path) -> None:
    """Catches a phrase start after rejection mutating an older accepted template."""
    source = tmp_path / "input.rap"
    source.write_text(
        "\n".join(
            (
                "**recip\t**stress\t**break\t**rhyme\t**lyrics",
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4",
                "=1\t=1\t=1\t=1\t=1",
                "4\t1\t.\tA\tza",
                "4\t0\t.\t.\tzb",
                "4\t1\t.\t.\tzc",
                "4\t0\t.\t.\tzd",
                "=2\t=2\t=2\t=2\t=2",
                "2\t1\t.\tB\tze",
                "2\t0\t.\t.\tzf",
                "4r\t.\t.\t.\tR",
                "=3\t=3\t=3\t=3\t=3",
                "4\t1\t3\tC\tzg",
                "4r\t.\t.\t.\tR",
                "4r\t.\t.\t.\tR",
                "4r\t.\t.\t.\tR",
                "*-\t*-\t*-\t*-\t*-",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extraction = extract_anonymous_templates(source)

    assert [template.name for template in extraction.templates] == ["anonymous_measure_1"]
    assert extraction.templates[0].slots[-1].boundary_strength == 0
    assert [(item.measure_ordinal, item.error_code) for item in extraction.rejections] == [
        (2, "overfull_measure"),
        (3, "unrepresentable_phrase_break"),
    ]


def test_rejected_measure_with_phrase_start_has_one_primary_catalog_rejection(tmp_path: Path) -> None:
    """Catches a failed measure being serialized once per failure reason rather than once per measure."""
    source = tmp_path / "input.rap"
    source.write_text(
        "\n".join(
            (
                "**recip\t**stress\t**break\t**rhyme\t**lyrics",
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4",
                "=1\t=1\t=1\t=1\t=1",
                "4\t1\t.\tA\tza",
                "4\t0\t.\t.\tzb",
                "4\t1\t.\t.\tzc",
                "4\t0\t.\t.\tzd",
                "=2\t=2\t=2\t=2\t=2",
                "2\t1\t3\tB\tze",
                "2\t0\t.\t.\tzf",
                "4r\t.\t.\t.\tR",
                "*-\t*-\t*-\t*-\t*-",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    extraction = extract_anonymous_templates(source)
    output = tmp_path / "catalog.json"
    write_extracted_templates(extraction, output)

    assert [(item.measure_ordinal, item.error_code) for item in extraction.rejections] == [(2, "overfull_measure")]
    assert extraction.to_dict()["aggregate"]["rejected_measures"] == 1
    assert [template.name for template in load_extracted_templates(output)] == ["anonymous_measure_1"]


def test_serialized_catalog_reloads_validated_templates_and_rejects_unknown_schema(tmp_path: Path) -> None:
    """Catches catalog output that cannot be safely reused by TemplateCatalog."""
    output = tmp_path / "catalog.json"
    write_extracted_templates(extract_anonymous_templates(FIXTURE), output)

    restored = load_extracted_templates(output)
    assert flow_template_to_dict(restored[0]) == flow_template_to_dict(extract_anonymous_templates(FIXTURE).templates[0])

    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["schema_version"] = "unknown"
    output.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported extracted template schema"):
        load_extracted_templates(output)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload: payload.update(
            {
                "rejections": [
                    {"source_hash": "invalid", "measure_ordinal": 1, "error_code": "empty_measure", "detail": "empty"}
                ],
                "aggregate": {"parsed_files": 1, "accepted_templates": 2, "rejected_measures": 1},
            }
        ),
        lambda payload: payload.update(
            {
                "rejections": [
                    {
                        "source_hash": payload["templates"][0]["provenance"]["source_hash"],
                        "measure_ordinal": 0,
                        "error_code": "empty_measure",
                        "detail": "empty",
                    }
                ],
                "aggregate": {"parsed_files": 1, "accepted_templates": 2, "rejected_measures": 1},
            }
        ),
        lambda payload: payload["aggregate"].update({"parsed_files": -1}),
        lambda payload: payload["aggregate"].update({"accepted_templates": 99}),
    ),
)
def test_load_extracted_templates_rejects_malformed_rejection_and_aggregate_scalars(tmp_path: Path, mutate) -> None:
    """Catches malformed anonymous catalog metadata passing validation."""
    output = tmp_path / "catalog.json"
    write_extracted_templates(extract_anonymous_templates(FIXTURE), output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    mutate(payload)
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid extracted template catalog"):
        load_extracted_templates(output)


def test_flow_template_to_dict_rejects_nonanonymous_extracted_metadata() -> None:
    """Catches callers serializing arbitrary names or provenance through the public API."""
    template = extract_anonymous_templates(FIXTURE).templates[0]
    nonanonymous = replace(template, name="caller supplied title", provenance=replace(template.provenance, source="caller/path"))

    with pytest.raises(ValueError, match="anonymous extracted template"):
        flow_template_to_dict(nonanonymous)


def test_load_extracted_templates_rejects_nonanonymous_metadata(tmp_path: Path) -> None:
    """Catches catalog input round-tripping a caller supplied name or provenance source."""
    output = tmp_path / "catalog.json"
    write_extracted_templates(extract_anonymous_templates(FIXTURE), output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["templates"][0]["name"] = "caller supplied title"
    payload["templates"][0]["provenance"]["source"] = "caller/path"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="anonymous extracted template"):
        load_extracted_templates(output)


def test_extract_mcflow_directory_is_deterministic_and_anonymous(tmp_path: Path) -> None:
    """Catches nondeterministic traversal and reports that expose input names."""
    inputs = tmp_path / "inputs"
    nested = inputs / "nested"
    nested.mkdir(parents=True)
    (inputs / "z.rap").write_bytes(FIXTURE.read_bytes())
    (nested / "a.rap").write_bytes(FIXTURE.read_bytes())

    extraction = extract_mcflow_directory(inputs)
    repeated = extract_mcflow_directory(inputs)

    assert len(extraction.templates) == 4
    assert [template.template_id for template in extraction.templates] == [
        template.template_id for template in repeated.templates
    ]
    assert "z.rap" not in json.dumps(extraction.to_dict())
    assert "a.rap" not in json.dumps(extraction.to_dict())


def test_directory_extraction_does_not_join_phrase_boundaries_across_files(tmp_path: Path) -> None:
    """Catches phrase starts in one source file modifying another file's template."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.rap").write_bytes(FIXTURE.read_bytes())
    first_break = FIXTURE.read_bytes().replace(b"16\t1\t.\t.\tA", b"16\t1\t.\t5\tA", 1)
    (inputs / "z.rap").write_bytes(first_break)

    extraction = extract_mcflow_directory(inputs)

    assert extraction.templates[1].slots[0].boundary_strength == 0
