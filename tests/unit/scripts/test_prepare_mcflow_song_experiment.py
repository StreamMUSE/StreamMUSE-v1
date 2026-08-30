from __future__ import annotations

import json
from pathlib import Path


def _write_mcflow_song(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "**recip\t**stress\t**break\t**rhyme\t**ipa\t**lyrics",
                "*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4\t*M4/4",
                "*MM133\t*MM133\t*MM133\t*MM133\t*MM133\t*MM133",
                "=1\t=1\t=1\t=1\t=1\t=1",
                "8\t1\t.\tA\t/rI/\trhy-",
                "8\t0\t.\t.\t/ðəm/\t-thm",
                "4\t1\t.\t.\t/braIt/\tbright",
                "4\t0\t.\t.\t/laIts/\tlights",
                "4\t.\t.\t.\tR\t.",
                "=2\t=2\t=2\t=2\t=2\t=2",
                "4\t1\t.\tB\t/oʊk/\tOak-",
                "4\t0\t.\t.\t/taʊn/\t-town",
                "2r\t.\t.\t.\t.\t.",
                "=3\t=3\t=3\t=3\t=3\t=3",
                "4\t1\t.\tC\t/bi/\tbeats",
                "4\t0\t.\t.\t/steI/\tstay",
                "4\t1\t.\t.\t/In/\tin",
                "4\t0\t.\t.\t/taIm/\ttime",
                "*-\t*-\t*-\t*-\t*-\t*-",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_prepare_song_reconstructs_timed_words_normalizes_and_pads_odd_bar(
    load_script,
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = load_script("prepare_mcflow_song_experiment")
    source = tmp_path / "song.rap"
    _write_mcflow_song(source)
    output = tmp_path / "output"
    rendered = []
    monkeypatch.setattr(module, "render_common_drums", lambda requests, **kwargs: rendered.extend(requests))

    requests = module.prepare_song(
        mcflow_file=source,
        output_dir=output,
        song_id="fixture_song",
        source_title="Fixture Song",
        source_commit="abc123",
        normalized_lines={2: "Oak town"},
    )

    assert len(requests) == 2
    assert {request.tempo_bpm for request in requests} == {133.0}
    assert requests[0].duration_seconds == 480 / 133
    assert [request.text for request in requests] == ["rhythm bright lights. Oak town.", "beats stay in time."]
    assert [len(request.syllables) for request in requests] == [6, 4]
    assert requests[-1].end_bar == 4
    assert max(item.tick_in_chunk for item in requests[-1].syllables) < 16
    assert rendered == list(requests)

    records = [
        json.loads(line)
        for line in (output / "common" / "fixture_song" / "chosen_lyrics.jsonl").read_text().splitlines()
    ]
    assert records[0]["source_text"] == "rhythm bright lights"
    assert records[1]["source_text"] == "Oaktown"
    assert records[1]["render_text"] == "Oak town"
    assert records[1]["normalization_applied"] is True

    manifest = json.loads((output / "mcflow_song_manifest.json").read_text())
    assert manifest["source"]["tempo_bpm"] == 133.0
    assert manifest["render"]["tempo_bpm"] == 133.0
    assert manifest["render"]["transcribed_bar_count"] == 3
    assert manifest["render"]["rendered_bar_count"] == 4
    assert manifest["render"]["padding_bar_count"] == 1
    assert manifest["render"]["audio_mode"] == "continuous_onset_r3"


def test_read_transcribed_bars_rejects_unfinished_cross_measure_word(load_script, tmp_path: Path) -> None:
    module = load_script("prepare_mcflow_song_experiment")
    source = tmp_path / "song.rap"
    _write_mcflow_song(source)
    source.write_text(source.read_text().replace("lights\n", "light-\n"), encoding="utf-8")

    try:
        module.read_transcribed_bars(source)
    except ValueError as error:
        assert "unfinished lyric word" in str(error)
    else:
        raise AssertionError("a word crossing a measure boundary must fail explicitly")


def test_read_transcribed_bars_accepts_continuation_without_leading_hyphen(
    load_script,
    tmp_path: Path,
) -> None:
    """Catches real MCFlow internal syllables being mistaken for new words."""
    module = load_script("prepare_mcflow_song_experiment")
    source = tmp_path / "song.rap"
    _write_mcflow_song(source)
    source.write_text(source.read_text().replace("-thm", "thm", 1), encoding="utf-8")

    assert module.read_transcribed_bars(source)[0] == "rhythm bright lights"
