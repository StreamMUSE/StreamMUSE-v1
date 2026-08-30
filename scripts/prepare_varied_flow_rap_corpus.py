#!/usr/bin/env python3
"""Retarget frozen nine-syllable rap lyrics to a balanced varied-flow program."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from streammuse.domain.rap import FlowProvenance, FlowSlot, FlowTemplate
from streammuse.domain.timing import Tempo
from streammuse.experiments.rap_audio_protocols.artifacts import file_sha256
from streammuse.experiments.rap_audio_protocols.audio import render_common_drums
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
    canonical_json_dumps,
)


@dataclass(frozen=True)
class SongDefinition:
    index: int
    song_id: str
    title: str
    topic: str


@dataclass(frozen=True)
class FlowAssignment:
    template: FlowTemplate
    stress_alignment: float


SELECTED_SONGS = (
    SongDefinition(4, "04_city_nights", "Lights After Midnight", "city nights"),
    SongDefinition(8, "08_street_basketball", "Concrete Court", "street basketball"),
    SongDefinition(10, "10_future_music", "The Next Sound", "future music"),
)


_TEMPLATE_SPECS = (
    ("front_sprint", (0, 1, 2, 4, 6, 8, 10, 13, 15), (1.0, 0.1, 0.6, 0.2, 0.9, 0.1, 0.7, 0.2, 1.0), (3,)),
    ("backbeat_ladder", (0, 2, 4, 5, 7, 8, 11, 13, 15), (0.4, 1.0, 0.2, 0.8, 0.2, 1.0, 0.1, 0.7, 0.9), (4,)),
    ("offbeat_entry", (1, 3, 5, 7, 8, 10, 12, 14, 15), (1.0, 0.2, 0.3, 1.0, 0.1, 0.6, 0.9, 0.2, 0.8), (3, 6)),
    ("twin_bursts", (0, 1, 4, 5, 8, 9, 12, 13, 15), (0.6, 0.2, 1.0, 0.1, 0.7, 0.3, 1.0, 0.1, 0.9), (3, 7)),
    ("center_rush", (0, 3, 5, 7, 8, 9, 10, 13, 15), (1.0, 0.1, 0.8, 0.1, 0.4, 1.0, 0.2, 0.7, 0.9), (3, 6)),
    ("closing_cascade", (0, 2, 4, 7, 9, 12, 13, 14, 15), (0.3, 1.0, 0.1, 0.7, 1.0, 0.2, 0.6, 0.2, 1.0), (4,)),
    ("wide_pockets", (0, 3, 4, 7, 8, 11, 12, 14, 15), (1.0, 0.2, 0.6, 0.1, 1.0, 0.2, 0.8, 0.1, 0.9), (4,)),
    ("early_cascade", (0, 1, 2, 3, 6, 9, 11, 13, 15), (0.5, 1.0, 0.1, 0.8, 0.2, 0.9, 0.2, 0.7, 1.0), (3,)),
    ("late_entry", (1, 2, 4, 6, 8, 11, 12, 14, 15), (1.0, 0.1, 0.7, 0.3, 0.9, 0.1, 1.0, 0.2, 0.8), (4, 6)),
    ("crossbeat_push", (0, 2, 3, 6, 7, 10, 11, 14, 15), (0.4, 1.0, 0.2, 0.6, 1.0, 0.1, 0.8, 0.2, 0.9), (2, 6)),
    ("stair_step", (0, 2, 5, 6, 8, 10, 13, 14, 15), (1.0, 0.2, 0.9, 0.1, 0.5, 1.0, 0.1, 0.7, 0.9), (3, 6)),
    ("late_cascade", (0, 3, 6, 8, 10, 11, 12, 14, 15), (0.3, 1.0, 0.2, 0.8, 0.2, 0.7, 1.0, 0.1, 0.9), (3, 5)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-album", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--song",
        action="append",
        choices=tuple(song.song_id for song in SELECTED_SONGS),
        default=None,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_ids = set(args.song or (song.song_id for song in SELECTED_SONGS))
    songs = tuple(song for song in SELECTED_SONGS if song.song_id in selected_ids)
    return prepare_corpus(
        source_album=args.source_album,
        output_dir=args.output_dir,
        songs=songs,
    )


def build_varied_templates() -> tuple[FlowTemplate, ...]:
    templates = []
    for name, ticks, stresses, boundaries in _TEMPLATE_SPECS:
        slots = []
        for index, (tick, stress) in enumerate(zip(ticks, stresses, strict=True)):
            next_tick = ticks[index + 1] if index + 1 < len(ticks) else 16
            slots.append(
                FlowSlot(
                    tick_in_bar=tick,
                    duration_ticks=next_tick - tick,
                    target_stress=stress,
                    boundary_strength=3 if index == len(ticks) - 1 else 2 if index in boundaries else 0,
                    rhyme_group="A" if index == len(ticks) - 1 else None,
                )
            )
        templates.append(
            FlowTemplate(
                template_id=f"varied_{name}_9",
                name=f"Varied flow: {name.replace('_', ' ')}",
                ticks_per_beat=4,
                beats_per_bar=4,
                slots=tuple(slots),
                provenance=FlowProvenance(
                    kind="hand_authored_mcflow_inspired",
                    source="StreamMUSE varied-flow full-song experiment",
                ),
            )
        )
    return tuple(templates)


def assign_templates(
    records: Sequence[dict[str, Any]],
    *,
    templates: Sequence[FlowTemplate],
    song_index: int,
) -> tuple[FlowAssignment, ...]:
    if len(templates) != 12:
        raise ValueError("varied-flow assignment requires exactly twelve templates")
    for record in records:
        schedule = record.get("schedule")
        if not isinstance(schedule, list) or len(schedule) != 9:
            raise ValueError("source bars must contain exactly nine scheduled syllables")

    assignments: list[FlowAssignment] = []
    previous_template_id: str | None = None
    for block_start in range(0, len(records), len(templates)):
        block = records[block_start : block_start + len(templates)]
        score_matrix = np.asarray(
            [
                [_stress_alignment(record, template) for template in templates]
                for record in block
            ],
            dtype=np.float64,
        )
        cost = -score_matrix
        if previous_template_id is not None:
            previous_index = next(
                index for index, template in enumerate(templates) if template.template_id == previous_template_id
            )
            cost[0, previous_index] += 10.0
        tie_break = np.asarray(
            [((index - song_index) % len(templates)) * 1e-9 for index in range(len(templates))],
            dtype=np.float64,
        )
        cost = cost + tie_break[np.newaxis, :]
        row_indices, template_indices = linear_sum_assignment(cost)
        by_row = {
            int(row): int(template_index)
            for row, template_index in zip(row_indices, template_indices, strict=True)
        }
        for row, record in enumerate(block):
            template = templates[by_row[row]]
            assignments.append(
                FlowAssignment(
                    template=template,
                    stress_alignment=_stress_alignment(record, template),
                )
            )
        previous_template_id = assignments[-1].template.template_id
    return tuple(assignments)


def retime_record(record: dict[str, Any], assignment: FlowAssignment) -> dict[str, Any]:
    schedule = record.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 9:
        raise ValueError("source bars must contain exactly nine scheduled syllables")
    bar = int(record["bar"])
    tempo = Tempo(90.0, 4, 4)
    retimed_schedule = []
    for index, (source, slot) in enumerate(zip(schedule, assignment.template.slots, strict=True)):
        absolute_tick = (bar * tempo.ticks_per_bar) + slot.tick_in_bar
        item = copy.deepcopy(source)
        item.update(
            {
                "absolute_tick": absolute_tick,
                "seconds_from_song_start": absolute_tick * tempo.seconds_per_tick,
                "slot_index": index,
                "target_sample_in_bar": round(slot.tick_in_bar * tempo.seconds_per_tick * 48_000),
                "target_stress": slot.target_stress,
                "tick_in_bar": slot.tick_in_bar,
            }
        )
        retimed_schedule.append(item)

    retimed = copy.deepcopy(record)
    retimed["source_template_id"] = str(record.get("template_id", ""))
    retimed["template_id"] = assignment.template.template_id
    retimed["schedule"] = retimed_schedule
    retimed["flow_retiming"] = {
        "method": "balanced_block_assignment_maximizing_lexical_stress_alignment",
        "stress_alignment": assignment.stress_alignment,
        "template_provenance": assignment.template.provenance.kind,
    }
    return retimed


def build_requests(
    song_id: str,
    records: Sequence[dict[str, Any]],
    *,
    templates: Sequence[FlowTemplate],
) -> tuple[TwoBarRenderRequest, ...]:
    if len(records) % 2:
        raise ValueError("varied-flow corpus requires an even number of bars")
    template_by_id = {template.template_id: template for template in templates}
    requests = []
    tempo = Tempo(90.0, 4, 4)
    for chunk_index, start in enumerate(range(0, len(records), 2)):
        bars = records[start : start + 2]
        start_tick = int(bars[0]["bar"]) * tempo.ticks_per_bar
        syllables = []
        for record in bars:
            template = template_by_id[str(record["template_id"])]
            schedule = record["schedule"]
            for slot_index, (item, slot) in enumerate(zip(schedule, template.slots, strict=True)):
                word = str(item["word"])
                index_in_word = 0
                for earlier in reversed(schedule[:slot_index]):
                    if str(earlier["word"]) != word:
                        break
                    index_in_word += 1
                absolute_tick = int(item["absolute_tick"])
                tick_in_chunk = absolute_tick - start_tick
                syllables.append(
                    SyllableTarget(
                        word=word,
                        index_in_word=index_in_word,
                        phonemes=tuple(str(phone) for phone in item["phonemes"]),
                        lexical_stress=int(item["lexical_stress"]),
                        target_stress=float(slot.target_stress),
                        boundary_strength=int(slot.boundary_strength),
                        absolute_tick=absolute_tick,
                        tick_in_chunk=tick_in_chunk,
                        target_seconds=tick_in_chunk * tempo.seconds_per_tick,
                    )
                )
        requests.append(
            TwoBarRenderRequest(
                song_id=song_id,
                chunk_index=chunk_index,
                start_bar=int(bars[0]["bar"]),
                end_bar=int(bars[-1]["bar"]) + 1,
                text=" ".join(str(record["text"]).strip() for record in bars),
                syllables=tuple(syllables),
            )
        )
    return tuple(requests)


def prepare_corpus(
    *,
    source_album: Path,
    output_dir: Path,
    songs: Sequence[SongDefinition],
) -> int:
    source_root = Path(source_album).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    templates = build_varied_templates()
    song_manifests = []
    lyrics_sections = ["# Varied-Flow Rap Corpus", ""]
    for song in songs:
        source_path = source_root / song.song_id / "chosen_lyrics.jsonl"
        source_records = _read_jsonl(source_path)
        if len(source_records) != 50:
            raise ValueError(f"{song.song_id} must contain exactly 50 source bars")
        assignments = assign_templates(source_records, templates=templates, song_index=song.index)
        records = tuple(
            retime_record(record, assignment)
            for record, assignment in zip(source_records, assignments, strict=True)
        )
        requests = build_requests(song.song_id, records, templates=templates)
        if len(requests) != 25:
            raise ValueError(f"{song.song_id} must produce exactly 25 requests")

        common_root = output_root / "common" / song.song_id
        common_root.mkdir(parents=True, exist_ok=True)
        chosen_path = common_root / "chosen_lyrics.jsonl"
        chosen_path.write_text(
            "\n".join(canonical_json_dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        request_path = common_root / "requests.jsonl"
        request_path.write_text(
            "\n".join(
                canonical_json_dumps({**request.to_payload(), "request_sha256": request.sha256})
                for request in requests
            )
            + "\n",
            encoding="utf-8",
        )
        drums_path = common_root / "drums.wav"
        render_common_drums(
            requests,
            song_index=song.index - 1,
            listening_wav_path=drums_path,
        )

        template_counts = Counter(assignment.template.template_id for assignment in assignments)
        transition_count = sum(
            left.template.template_id != right.template.template_id
            for left, right in zip(assignments, assignments[1:])
        )
        stress_scores = [assignment.stress_alignment for assignment in assignments]
        song_manifest = {
            "song_id": song.song_id,
            "song_index": song.index,
            "title": song.title,
            "topic": song.topic,
            "source_lyrics_path": str(source_path),
            "source_lyrics_sha256": file_sha256(source_path),
            "chosen_lyrics_path": str(chosen_path.relative_to(output_root)),
            "chosen_lyrics_sha256": file_sha256(chosen_path),
            "requests_path": str(request_path.relative_to(output_root)),
            "requests_sha256": file_sha256(request_path),
            "drums_path": str(drums_path.relative_to(output_root)),
            "drums_sha256": file_sha256(drums_path),
            "bar_count": len(records),
            "request_count": len(requests),
            "unique_template_count": len(template_counts),
            "template_counts": dict(sorted(template_counts.items())),
            "flow_transition_count": transition_count,
            "mean_stress_alignment": sum(stress_scores) / len(stress_scores),
            "minimum_stress_alignment": min(stress_scores),
        }
        song_manifests.append(song_manifest)
        lyrics_sections.extend([f"## {song.title}", ""])
        lyrics_sections.extend(f"{record['bar'] + 1:02d}. {record['text']}" for record in records)
        lyrics_sections.append("")
        print(
            f"stage=prepare song={song.song_id} bars=50 requests=25 templates={len(template_counts)} "
            f"transitions={transition_count}/49 mean_stress={song_manifest['mean_stress_alignment']:.4f}",
            flush=True,
        )

    template_payload = [_template_payload(template) for template in templates]
    templates_path = output_root / "flow_templates.json"
    templates_path.write_text(
        json.dumps(
            {
                "schema_version": "streammuse.varied_flow_templates.v1",
                "template_count": len(template_payload),
                "templates": template_payload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_root / "lyrics.md").write_text("\n".join(lyrics_sections), encoding="utf-8")
    manifest = {
        "schema_version": "streammuse.varied_flow_rap_corpus.v1",
        "tempo_bpm": 90.0,
        "bars_per_song": 50,
        "flow_assignment_method": "12-bar balanced Hungarian lexical-stress assignment",
        "template_count": len(templates),
        "flow_templates_path": str(templates_path.relative_to(output_root)),
        "flow_templates_sha256": file_sha256(templates_path),
        "song_count": len(song_manifests),
        "total_request_count": sum(song["request_count"] for song in song_manifests),
        "songs": song_manifests,
    }
    (output_root / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def _stress_alignment(record: dict[str, Any], template: FlowTemplate) -> float:
    lexical = [
        1.0 if int(item["lexical_stress"]) == 1 else 0.5 if int(item["lexical_stress"]) == 2 else 0.0
        for item in record["schedule"]
    ]
    errors = [
        abs(actual - slot.target_stress) * (1.0 + slot.target_stress)
        for actual, slot in zip(lexical, template.slots, strict=True)
    ]
    denominator = sum(1.0 + slot.target_stress for slot in template.slots)
    return 1.0 - sum(errors) / denominator


def _template_payload(template: FlowTemplate) -> dict[str, Any]:
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
        },
    }


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ValueError(f"missing source lyric file: {path}")
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


if __name__ == "__main__":
    raise SystemExit(main())
