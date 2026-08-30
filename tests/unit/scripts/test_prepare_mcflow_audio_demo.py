from __future__ import annotations

import json
from pathlib import Path

import pytest


FIXTURE = Path("tests/fixtures/rap/mcflow_minimal.rap")


def _mcflow_fixture_with_tempo(path: Path) -> None:
    content = FIXTURE.read_text(encoding="utf-8")
    content = content.replace(
        "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\n",
        "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\n"
        "*MM90\t*MM90\t*MM90\t*MM90\t*MM90\t*MM90\t*MM90\t*MM90\n",
    )
    path.write_text(content, encoding="utf-8")


def test_prepare_demo_uses_real_extracted_slots_without_retaining_source_lyrics(
    load_script,
    tmp_path: Path,
) -> None:
    script_path = Path("scripts/prepare_mcflow_audio_demo.py")
    if not script_path.is_file():
        pytest.fail("prepare_mcflow_audio_demo.py is missing")
    module = load_script("prepare_mcflow_audio_demo")
    source_path = tmp_path / "source.rap"
    lyrics_path = tmp_path / "lyrics.txt"
    output_dir = tmp_path / "demo"
    _mcflow_fixture_with_tempo(source_path)
    lyrics_path.write_text("Bright beats move now\nFlow\n", encoding="utf-8")

    requests = module.prepare_demo(
        mcflow_file=source_path,
        lyrics_file=lyrics_path,
        output_dir=output_dir,
        song_id="mcflow_demo",
        source_title="Invented fixture",
        start_measure_ordinal=1,
        bar_count=2,
    )

    assert len(requests) == 1
    request = requests[0]
    assert len(request.syllables) == 5
    assert [item.tick_in_chunk for item in request.syllables] == [0, 1, 2, 5, 16]
    assert [item.target_stress for item in request.syllables] == [1.0, 0.0, 1.0, 1.0, 1.0]
    assert (output_dir / "common" / "mcflow_demo" / "drums.wav").is_file()

    manifest_text = (output_dir / "mcflow_demo_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["source"]["tempo_bpm"] == 90.0
    assert manifest["source"]["measure_ordinals"] == [1, 2]
    assert all(
        template["provenance"]["kind"] == "mcflow_extracted_anonymous"
        for template in manifest["flow_templates"]
    )
    assert "tav" not in manifest_text
    assert "/ta/" not in manifest_text


def test_prepare_demo_rejects_lyrics_with_the_wrong_per_bar_syllable_count(
    load_script,
    tmp_path: Path,
) -> None:
    script_path = Path("scripts/prepare_mcflow_audio_demo.py")
    if not script_path.is_file():
        pytest.fail("prepare_mcflow_audio_demo.py is missing")
    module = load_script("prepare_mcflow_audio_demo")
    source_path = tmp_path / "source.rap"
    lyrics_path = tmp_path / "lyrics.txt"
    _mcflow_fixture_with_tempo(source_path)
    lyrics_path.write_text("Too short\nFlow\n", encoding="utf-8")

    with pytest.raises(ValueError, match="measure 1 requires 4 syllables, got 2"):
        module.prepare_demo(
            mcflow_file=source_path,
            lyrics_file=lyrics_path,
            output_dir=tmp_path / "demo",
            song_id="mcflow_demo",
            source_title="Invented fixture",
            start_measure_ordinal=1,
            bar_count=2,
        )
