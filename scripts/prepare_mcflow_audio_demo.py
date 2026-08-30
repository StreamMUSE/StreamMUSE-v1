#!/usr/bin/env python3
"""Prepare an auditable variable-density audio demo from an MCFlow transcription."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Sequence

from streammuse.domain.timing import Tempo
from streammuse.experiments.rap_audio_protocols.audio import render_common_drums
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
    canonical_json_dumps,
)
from streammuse.infrastructure.rap.mcflow import (
    extract_anonymous_templates,
    flow_template_to_dict,
)
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer


RENDER_TEMPO = Tempo(90.0, 4, 4)
MCFLOW_REPOSITORY = "https://github.com/Computational-Cognitive-Musicology-Lab/MCFlow"
_TEMPO_TOKEN = re.compile(r"^\*MM(?P<bpm>[0-9]+(?:\.[0-9]+)?)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mcflow-file", required=True, type=Path)
    parser.add_argument("--lyrics-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--song-id", default="mcflow_in_da_club_flow_demo")
    parser.add_argument("--source-title", required=True)
    parser.add_argument("--start-measure-ordinal", type=int, default=2)
    parser.add_argument("--bar-count", type=int, default=8)
    parser.add_argument("--source-commit", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    requests = prepare_demo(
        mcflow_file=args.mcflow_file,
        lyrics_file=args.lyrics_file,
        output_dir=args.output_dir,
        song_id=args.song_id,
        source_title=args.source_title,
        start_measure_ordinal=args.start_measure_ordinal,
        bar_count=args.bar_count,
        source_commit=args.source_commit,
    )
    print(
        f"prepared song={args.song_id} bars={args.bar_count} chunks={len(requests)} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


def prepare_demo(
    *,
    mcflow_file: Path,
    lyrics_file: Path,
    output_dir: Path,
    song_id: str,
    source_title: str,
    start_measure_ordinal: int,
    bar_count: int,
    source_commit: str | None = None,
) -> tuple[TwoBarRenderRequest, ...]:
    if start_measure_ordinal < 1:
        raise ValueError("start measure ordinal must be positive")
    if bar_count <= 0 or bar_count % 2:
        raise ValueError("bar count must be a positive even number")

    source_path = Path(mcflow_file)
    source_bytes = source_path.read_bytes()
    source_tempo = _read_source_tempo(source_bytes.decode("utf-8"))
    if source_tempo != RENDER_TEMPO.bpm:
        raise ValueError(
            f"this demo preserves only 90 BPM MCFlow sources; got {source_tempo:g} BPM"
        )

    extraction = extract_anonymous_templates(source_path)
    template_by_ordinal = {
        int(template.name.removeprefix("anonymous_measure_")): template
        for template in extraction.templates
    }
    ordinals = tuple(range(start_measure_ordinal, start_measure_ordinal + bar_count))
    missing = tuple(ordinal for ordinal in ordinals if ordinal not in template_by_ordinal)
    if missing:
        rejection_by_ordinal = {
            rejection.measure_ordinal: rejection.error_code
            for rejection in extraction.rejections
        }
        details = ", ".join(
            f"{ordinal}:{rejection_by_ordinal.get(ordinal, 'missing')}"
            for ordinal in missing
        )
        raise ValueError(f"selected MCFlow measures are unavailable: {details}")
    templates = tuple(template_by_ordinal[ordinal] for ordinal in ordinals)

    lines = tuple(
        line.strip()
        for line in Path(lyrics_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(lines) != bar_count:
        raise ValueError(f"lyrics file requires {bar_count} nonempty lines, got {len(lines)}")

    analyzer = CmuProsodyAnalyzer()
    analyses = tuple(analyzer.analyze(line) for line in lines)
    for ordinal, template, analysis in zip(ordinals, templates, analyses, strict=True):
        required = len(template.slots)
        actual = len(analysis.syllables)
        if actual != required:
            raise ValueError(f"measure {ordinal} requires {required} syllables, got {actual}")
        if analysis.oov_words:
            raise ValueError(
                f"measure {ordinal} contains CMUdict OOV words: {', '.join(analysis.oov_words)}"
            )

    requests = _build_requests(song_id, lines, analyses, templates)
    output_root = Path(output_dir)
    common_root = output_root / "common" / song_id
    common_root.mkdir(parents=True, exist_ok=True)
    request_path = common_root / "requests.jsonl"
    request_path.write_text(
        "".join(canonical_json_dumps(request.to_payload()) + "\n" for request in requests),
        encoding="utf-8",
    )
    lyric_record_path = common_root / "chosen_lyrics.jsonl"
    lyric_record_path.write_text(
        "".join(
            canonical_json_dumps(
                _lyric_record(
                    bar=bar,
                    measure_ordinal=ordinal,
                    line=line,
                    analysis=analysis,
                    template=template,
                )
            )
            + "\n"
            for bar, (ordinal, line, analysis, template) in enumerate(
                zip(ordinals, lines, analyses, templates, strict=True)
            )
        ),
        encoding="utf-8",
    )
    drums_path = common_root / "drums.wav"
    render_common_drums(
        requests,
        song_index=0,
        allow_smoke_test=True,
        listening_wav_path=drums_path,
    )

    manifest = {
        "schema_version": "streammuse.mcflow_audio_demo.v1",
        "song_id": song_id,
        "bar_count": bar_count,
        "chunk_count": len(requests),
        "lyric_origin": "newly_written_for_this_demo_not_source_song_lyrics",
        "source": {
            "title": source_title,
            "repository": MCFLOW_REPOSITORY,
            "commit": source_commit,
            "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "tempo_bpm": source_tempo,
            "measure_ordinals": list(ordinals),
        },
        "render": {
            "tempo_bpm": RENDER_TEMPO.bpm,
            "ticks_per_beat": RENDER_TEMPO.ticks_per_beat,
            "beats_per_bar": RENDER_TEMPO.beats_per_bar,
            "request_path": str(request_path.relative_to(output_root)),
            "drums_path": str(drums_path.relative_to(output_root)),
            "syllables_per_bar": [len(template.slots) for template in templates],
        },
        "flow_templates": [flow_template_to_dict(template) for template in templates],
    }
    (output_root / "mcflow_demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "lyrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return requests


def _build_requests(song_id, lines, analyses, templates) -> tuple[TwoBarRenderRequest, ...]:
    requests = []
    for chunk_index, first_bar in enumerate(range(0, len(lines), 2)):
        syllables = []
        for local_bar in range(first_bar, first_bar + 2):
            analysis = analyses[local_bar]
            template = templates[local_bar]
            for syllable, slot in zip(analysis.syllables, template.slots, strict=True):
                absolute_tick = local_bar * RENDER_TEMPO.ticks_per_bar + slot.tick_in_bar
                tick_in_chunk = absolute_tick - first_bar * RENDER_TEMPO.ticks_per_bar
                syllables.append(
                    SyllableTarget(
                        word=syllable.word,
                        index_in_word=syllable.index_in_word,
                        phonemes=syllable.phonemes,
                        lexical_stress=syllable.stress,
                        target_stress=slot.target_stress,
                        boundary_strength=slot.boundary_strength,
                        absolute_tick=absolute_tick,
                        tick_in_chunk=tick_in_chunk,
                        target_seconds=tick_in_chunk * RENDER_TEMPO.seconds_per_tick,
                    )
                )
        requests.append(
            TwoBarRenderRequest(
                song_id=song_id,
                chunk_index=chunk_index,
                start_bar=first_bar,
                end_bar=first_bar + 2,
                text=". ".join(lines[first_bar : first_bar + 2]) + ".",
                syllables=tuple(syllables),
            )
        )
    return tuple(requests)


def _lyric_record(*, bar, measure_ordinal, line, analysis, template) -> dict[str, object]:
    return {
        "bar": bar,
        "source_measure_ordinal": measure_ordinal,
        "text": line,
        "syllable_count": len(analysis.syllables),
        "template_id": template.template_id,
        "template_provenance": template.provenance.kind,
        "schedule": [
            {
                "word": syllable.word,
                "index_in_word": syllable.index_in_word,
                "phonemes": list(syllable.phonemes),
                "lexical_stress": syllable.stress,
                "tick_in_bar": slot.tick_in_bar,
                "duration_ticks": slot.duration_ticks,
                "target_stress": slot.target_stress,
                "boundary_strength": slot.boundary_strength,
                "rhyme_group": slot.rhyme_group,
            }
            for syllable, slot in zip(analysis.syllables, template.slots, strict=True)
        ],
    }


def _read_source_tempo(content: str) -> float:
    tempos = set()
    for line in content.splitlines():
        for field in line.split("\t"):
            match = _TEMPO_TOKEN.fullmatch(field)
            if match is not None:
                tempos.add(float(match.group("bpm")))
    if not tempos:
        raise ValueError("MCFlow source has no metronome marking")
    if len(tempos) != 1:
        raise ValueError("MCFlow source changes tempo; this demo requires one tempo")
    return tempos.pop()


if __name__ == "__main__":
    raise SystemExit(main())
