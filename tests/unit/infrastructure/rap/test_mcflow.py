"""Tests for anonymous MCFlow structure extraction."""

from __future__ import annotations

import json
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
