#!/usr/bin/env python3
"""Prepare an auditable full-song audio experiment from timed MCFlow lyrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

from streammuse.domain.timing import Tempo
from streammuse.experiments.rap_audio_protocols.audio import render_common_drums
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
    canonical_json_dumps,
)
from streammuse.infrastructure.rap.mcflow import extract_anonymous_templates, flow_template_to_dict
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer


MCFLOW_REPOSITORY = "https://github.com/Computational-Cognitive-Musicology-Lab/MCFlow"
_TEMPO_TOKEN = re.compile(r"^\*MM(?P<bpm>[0-9]+(?:\.[0-9]+)?)$")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcflow-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--song-id", required=True)
    parser.add_argument("--source-title", required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument(
        "--render-tempo",
        type=float,
        default=None,
        help="Render BPM; defaults to the source MCFlow metronome marking.",
    )
    parser.add_argument(
        "--normalizations-file",
        type=Path,
        help="Optional JSON object mapping one-based measure ordinals to replacement render text.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    normalized_lines = _load_normalized_lines(args.normalizations_file)
    requests = prepare_song(
        mcflow_file=args.mcflow_file,
        output_dir=args.output_dir,
        song_id=args.song_id,
        source_title=args.source_title,
        source_commit=args.source_commit,
        normalized_lines=normalized_lines,
        render_tempo_bpm=args.render_tempo,
    )
    print(
        f"prepared song={args.song_id} chunks={len(requests)} "
        f"rendered_bars={requests[-1].end_bar} output={args.output_dir}",
        flush=True,
    )
    return 0


def prepare_song(
    *,
    mcflow_file: Path,
    output_dir: Path,
    song_id: str,
    source_title: str,
    source_commit: str | None,
    normalized_lines: Mapping[int, str],
    render_tempo_bpm: float | None = None,
) -> tuple[TwoBarRenderRequest, ...]:
    source_path = Path(mcflow_file)
    source_bytes = source_path.read_bytes()
    source_content = source_bytes.decode("utf-8")
    source_tempo = _read_source_tempo(source_content)
    render_tempo = Tempo(
        source_tempo if render_tempo_bpm is None else render_tempo_bpm,
        4,
        4,
    )
    source_lines = read_transcribed_bars(source_path)
    if not source_lines:
        raise ValueError("MCFlow source contains no timed lyric measures")

    extraction = extract_anonymous_templates(source_path)
    if extraction.rejections:
        details = ", ".join(
            f"{item.measure_ordinal}:{item.error_code}" for item in extraction.rejections
        )
        raise ValueError(f"MCFlow source contains rejected measures: {details}")
    templates = tuple(
        sorted(
            extraction.templates,
            key=lambda template: int(template.name.removeprefix("anonymous_measure_")),
        )
    )
    if len(templates) != len(source_lines):
        raise ValueError(
            f"timed lyric/template count mismatch: {len(source_lines)} lyrics, "
            f"{len(templates)} templates"
        )
    unexpected = sorted(set(normalized_lines) - set(range(1, len(source_lines) + 1)))
    if unexpected:
        raise ValueError(f"normalizations reference unavailable measures: {unexpected}")

    render_lines = tuple(
        str(normalized_lines.get(ordinal, source_line)).strip()
        for ordinal, source_line in enumerate(source_lines, start=1)
    )
    if any(not line for line in render_lines):
        raise ValueError("normalized render text must not be empty")

    analyzer = CmuProsodyAnalyzer()
    analyses = tuple(analyzer.analyze(line) for line in render_lines)
    for ordinal, source_line, render_line, template, analysis in zip(
        range(1, len(source_lines) + 1),
        source_lines,
        render_lines,
        templates,
        analyses,
        strict=True,
    ):
        required = len(template.slots)
        source_syllables = len(_timed_lyric_tokens_for_measure(source_content, ordinal))
        if source_syllables != required:
            raise ValueError(
                f"measure {ordinal} source has {source_syllables} lyric syllables but "
                f"its template has {required} slots"
            )
        if len(analysis.syllables) != required:
            raise ValueError(
                f"measure {ordinal} requires {required} syllables, got "
                f"{len(analysis.syllables)} from {render_line!r} (source {source_line!r})"
            )
        if analysis.oov_words:
            raise ValueError(
                f"measure {ordinal} contains CMUdict OOV words: {', '.join(analysis.oov_words)}"
            )

    requests = _build_requests(song_id, render_lines, analyses, templates, render_tempo)
    output_root = Path(output_dir)
    common_root = output_root / "common" / song_id
    common_root.mkdir(parents=True, exist_ok=True)
    request_path = common_root / "requests.jsonl"
    request_path.write_text(
        "".join(canonical_json_dumps(request.to_payload()) + "\n" for request in requests),
        encoding="utf-8",
    )
    lyric_path = common_root / "chosen_lyrics.jsonl"
    lyric_path.write_text(
        "".join(
            canonical_json_dumps(
                _lyric_record(
                    bar=bar,
                    source_text=source_line,
                    render_text=render_line,
                    analysis=analysis,
                    template=template,
                )
            )
            + "\n"
            for bar, (source_line, render_line, analysis, template) in enumerate(
                zip(source_lines, render_lines, analyses, templates, strict=True)
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

    rendered_bar_count = requests[-1].end_bar
    manifest = {
        "schema_version": "streammuse.mcflow_song_experiment.v1",
        "song_id": song_id,
        "lyric_origin": "timed_lyrics_transcribed_in_mcflow_source",
        "source": {
            "title": source_title,
            "repository": MCFLOW_REPOSITORY,
            "commit": source_commit,
            "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "tempo_bpm": source_tempo,
            "measure_ordinals": list(range(1, len(source_lines) + 1)),
        },
        "render": {
            "tempo_bpm": render_tempo.bpm,
            "ticks_per_beat": render_tempo.ticks_per_beat,
            "beats_per_bar": render_tempo.beats_per_bar,
            "audio_mode": "continuous_onset_r3",
            "stress_augmentation": False,
            "timing_regularization": False,
            "transcribed_bar_count": len(source_lines),
            "rendered_bar_count": rendered_bar_count,
            "padding_bar_count": rendered_bar_count - len(source_lines),
            "chunk_count": len(requests),
            "request_path": str(request_path.relative_to(output_root)),
            "lyric_path": str(lyric_path.relative_to(output_root)),
            "drums_path": str(drums_path.relative_to(output_root)),
            "syllables_per_bar": [len(template.slots) for template in templates],
        },
        "normalizations": {
            str(ordinal): render_lines[ordinal - 1]
            for ordinal in sorted(normalized_lines)
        },
        "flow_templates": [flow_template_to_dict(template) for template in templates],
    }
    (output_root / "mcflow_song_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return requests


def read_transcribed_bars(path: Path) -> tuple[str, ...]:
    content = Path(path).read_text(encoding="utf-8")
    lines = content.splitlines()
    try:
        columns = lines[0].split("\t")
        lyric_index = columns.index("**lyrics")
        ipa_index = columns.index("**ipa")
        reciprocal_index = columns.index("**recip")
    except (IndexError, ValueError) as error:
        raise ValueError("MCFlow source requires recip, ipa, and lyrics spines") from error

    bars: list[str] = []
    tokens: list[str] | None = None
    for raw_line in lines[1:]:
        if not raw_line or raw_line.startswith("!"):
            continue
        fields = raw_line.split("\t")
        if raw_line.startswith("="):
            if tokens is not None:
                bars.append(_join_lyric_tokens(tokens, len(bars) + 1))
            tokens = []
            continue
        if raw_line.startswith("*"):
            continue
        if len(fields) != len(columns):
            raise ValueError("record width does not match exclusive interpretations")
        if tokens is None:
            tokens = []
        lyric = fields[lyric_index]
        is_rest = (
            fields[reciprocal_index].endswith("r")
            or fields[ipa_index] == "R"
            or lyric == "R"
        )
        if not is_rest:
            if lyric == ".":
                raise ValueError("sounding MCFlow record has no lyric syllable")
            tokens.append(lyric)
    if tokens is not None:
        bars.append(_join_lyric_tokens(tokens, len(bars) + 1))
    return tuple(bars)


def _join_lyric_tokens(tokens: Sequence[str], measure_ordinal: int) -> str:
    words: list[str] = []
    current = ""
    for token in tokens:
        syllable = token.removeprefix("-")
        if current:
            current += syllable
        else:
            current = syllable
        if current.endswith("-"):
            current = current[:-1]
        else:
            words.append(current)
            current = ""
    if current:
        raise ValueError(f"measure {measure_ordinal} contains an unfinished lyric word")
    return " ".join(words)


def _timed_lyric_tokens_for_measure(content: str, ordinal: int) -> tuple[str, ...]:
    lines = read_transcribed_bars_from_content_tokens(content)
    return lines[ordinal - 1]


def read_transcribed_bars_from_content_tokens(content: str) -> tuple[tuple[str, ...], ...]:
    lines = content.splitlines()
    columns = lines[0].split("\t")
    lyric_index = columns.index("**lyrics")
    ipa_index = columns.index("**ipa")
    reciprocal_index = columns.index("**recip")
    measures: list[tuple[str, ...]] = []
    tokens: list[str] | None = None
    for raw_line in lines[1:]:
        if not raw_line or raw_line.startswith("!"):
            continue
        fields = raw_line.split("\t")
        if raw_line.startswith("="):
            if tokens is not None:
                measures.append(tuple(tokens))
            tokens = []
        elif not raw_line.startswith("*"):
            if tokens is None:
                tokens = []
            lyric = fields[lyric_index]
            if not (
                fields[reciprocal_index].endswith("r")
                or fields[ipa_index] == "R"
                or lyric == "R"
            ):
                tokens.append(lyric)
    if tokens is not None:
        measures.append(tuple(tokens))
    return tuple(measures)


def _build_requests(song_id, lines, analyses, templates, tempo: Tempo) -> tuple[TwoBarRenderRequest, ...]:
    requests = []
    for chunk_index, first_bar in enumerate(range(0, len(lines), 2)):
        syllables = []
        for local_bar in range(first_bar, min(first_bar + 2, len(lines))):
            analysis = analyses[local_bar]
            template = templates[local_bar]
            for syllable, slot in zip(analysis.syllables, template.slots, strict=True):
                absolute_tick = local_bar * tempo.ticks_per_bar + slot.tick_in_bar
                tick_in_chunk = absolute_tick - first_bar * tempo.ticks_per_bar
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
                        target_seconds=tick_in_chunk * tempo.seconds_per_tick,
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
                tempo_bpm=tempo.bpm,
            )
        )
    return tuple(requests)


def _lyric_record(*, bar, source_text, render_text, analysis, template) -> dict[str, object]:
    return {
        "bar": bar,
        "source_measure_ordinal": bar + 1,
        "source_text": source_text,
        "render_text": render_text,
        "normalization_applied": source_text != render_text,
        "syllable_count": len(analysis.syllables),
        "template_id": template.template_id,
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


def _load_normalized_lines(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("normalizations file must contain a JSON object")
    return {int(key): str(value) for key, value in payload.items()}


def _read_source_tempo(content: str) -> float:
    tempos = {
        float(match.group("bpm"))
        for line in content.splitlines()
        for field in line.split("\t")
        if (match := _TEMPO_TOKEN.fullmatch(field)) is not None
    }
    if not tempos:
        raise ValueError("MCFlow source has no metronome marking")
    if len(tempos) != 1:
        raise ValueError("MCFlow source changes tempo")
    return tempos.pop()


if __name__ == "__main__":
    raise SystemExit(main())
