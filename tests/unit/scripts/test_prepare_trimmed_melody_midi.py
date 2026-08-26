from __future__ import annotations

import csv
import json
from pathlib import Path

import mido
import numpy as np
import pytest

from experiments.lekai_failcase_dataset_v2.prepare_trimmed_melody_midi import (
    prepare_record,
    prepare_selection,
    sha256_file,
    trim_melody_midi_leading_ticks,
)


def _write_npz(path, *, first_onset_step=4):
    measure = np.zeros((4, 88, 16), dtype=np.uint8)
    measure[1, 39, first_onset_step] = 1
    measure[0, 39, first_onset_step : first_onset_step + 3] = 1
    measure[1, 43, first_onset_step + 6] = 1
    measure[0, 43, first_onset_step + 6 : first_onset_step + 8] = 1
    np.savez(
        path,
        metadata={
            "bpm": 120,
            "time_signature": "4/4",
            "time_signature_idx": 0,
            "num_measures": 1,
        },
        measure_0=measure,
    )


def _record(piece_id):
    return {
        "order": 1,
        "style": "anime",
        "id": str(piece_id),
        "title": "Synthetic",
    }


def _absolute_tracks(path):
    midi = mido.MidiFile(str(path))
    tracks = []
    for track in midi.tracks:
        absolute_tick = 0
        events = []
        for message in track:
            absolute_tick += int(message.time)
            events.append((absolute_tick, message))
        tracks.append(events)
    return midi, tracks


def _positive_notes(path):
    _midi, tracks = _absolute_tracks(path)
    return [
        (tick, message.note, message.velocity, message.channel)
        for track in tracks
        for tick, message in track
        if message.type == "note_on" and message.velocity > 0
    ]


def _note_messages(path):
    _midi, tracks = _absolute_tracks(path)
    return [
        (
            tick,
            message.type,
            message.note,
            message.velocity,
            message.channel,
        )
        for track in tracks
        for tick, message in track
        if message.type in {"note_on", "note_off"}
    ]


def _write_detailed_midi(path, *, first_note_tick=120):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("track_name", name="Metadata", time=0))
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(96), time=0))
    meta.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            clocks_per_click=24,
            notated_32nd_notes_per_beat=8,
            time=0,
        )
    )
    meta.append(mido.MetaMessage("end_of_track", time=0))
    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="Melody", time=0))
    notes.append(mido.Message("program_change", channel=2, program=11, time=0))
    notes.append(
        mido.Message(
            "note_on", channel=2, note=60, velocity=91, time=first_note_tick
        )
    )
    notes.append(mido.Message("note_off", channel=2, note=60, velocity=17, time=96))
    notes.append(mido.Message("note_on", channel=2, note=67, velocity=73, time=144))
    notes.append(mido.Message("note_off", channel=2, note=67, velocity=9, time=120))
    notes.append(mido.MetaMessage("end_of_track", time=77))
    midi.tracks.extend([meta, notes])
    midi.save(str(path))


def _message_payloads(path):
    _midi, tracks = _absolute_tracks(path)
    return [
        [message.copy(time=0) for _tick, message in track]
        for track in tracks
    ]


def test_npz_uses_existing_exporter_before_physical_midi_trim(tmp_path):
    npz_root = tmp_path / "npz"
    npz_root.mkdir()
    _write_npz(npz_root / "101.npz", first_onset_step=4)
    output_root = tmp_path / "output"

    row = prepare_record(_record("101"), npz_root=npz_root, output_root=output_root)
    source = output_root / "source_melody_midi" / "101.mid"
    trimmed = output_root / "trimmed_melody_midi" / "101.mid"

    assert source.is_file()
    assert trimmed.is_file()
    assert row["source_melody_midi"] == str(source.resolve())
    assert row["trimmed_melody_midi"] == str(trimmed.resolve())
    assert _positive_notes(source)[0][0] > 0
    assert _positive_notes(trimmed)[0][0] == 0


def test_first_positive_note_on_moves_to_tick_zero(tmp_path):
    source = tmp_path / "source.mid"
    output = tmp_path / "trimmed.mid"
    _write_detailed_midi(source, first_note_tick=120)

    report = trim_melody_midi_leading_ticks(source, output)

    assert report["first_note_tick_original"] == 120
    assert report["offset_ticks"] == 120
    assert _positive_notes(output)[0][0] == 0


def test_note_timing_payload_internal_rest_and_trailing_ticks_are_exact(tmp_path):
    source = tmp_path / "source.mid"
    output = tmp_path / "trimmed.mid"
    _write_detailed_midi(source, first_note_tick=120)

    report = trim_melody_midi_leading_ticks(source, output)
    source_notes = _positive_notes(source)
    output_notes = _positive_notes(output)
    source_note_messages = _note_messages(source)
    output_note_messages = _note_messages(output)

    assert [(n, v, c) for _t, n, v, c in source_notes] == [
        (n, v, c) for _t, n, v, c in output_notes
    ]
    assert [event[1:] for event in source_note_messages] == [
        event[1:] for event in output_note_messages
    ]
    assert [event[0] - 120 for event in source_note_messages] == [
        event[0] for event in output_note_messages
    ]
    assert source_note_messages[1][0] - source_note_messages[0][0] == 96
    assert output_note_messages[1][0] - output_note_messages[0][0] == 96
    assert source_notes[1][0] - source_note_messages[1][0] == 144
    assert output_notes[1][0] - output_note_messages[1][0] == 144
    assert report["source_end_tick"] - source_note_messages[-1][0] == 77
    assert report["output_end_tick"] - output_note_messages[-1][0] == 77
    assert report["source_end_tick"] - report["output_end_tick"] == 120
    assert report["source_note_count"] == report["trimmed_note_count"] == 2
    assert report["exact_tick_shift_verified"] is True


def test_tempo_time_signature_track_name_and_program_are_preserved(tmp_path):
    source = tmp_path / "source.mid"
    output = tmp_path / "trimmed.mid"
    _write_detailed_midi(source, first_note_tick=120)

    trim_melody_midi_leading_ticks(source, output)
    _source_midi, source_tracks = _absolute_tracks(source)
    _output_midi, output_tracks = _absolute_tracks(output)

    assert _message_payloads(source) == _message_payloads(output)
    for message_type in ("set_tempo", "time_signature", "track_name", "program_change"):
        source_events = [
            (tick, message)
            for track in source_tracks
            for tick, message in track
            if message.type == message_type
        ]
        output_events = [
            (tick, message)
            for track in output_tracks
            for tick, message in track
            if message.type == message_type
        ]
        assert len(source_events) == len(output_events)
        assert all(tick == 0 for tick, _message in output_events)


def test_offset_zero_is_byte_exact_copy(tmp_path):
    source = tmp_path / "source.mid"
    output = tmp_path / "trimmed.mid"
    _write_detailed_midi(source, first_note_tick=0)
    source_bytes = source.read_bytes()

    report = trim_melody_midi_leading_ticks(source, output)

    assert report["offset_ticks"] == 0
    assert output.read_bytes() == source_bytes
    assert sha256_file(source) == sha256_file(output)


def test_no_positive_note_on_raises(tmp_path):
    source = tmp_path / "empty.mid"
    output = tmp_path / "trimmed.mid"
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name="Melody", time=0))
    track.append(mido.MetaMessage("end_of_track", time=240))
    midi.tracks.append(track)
    midi.save(str(source))

    with pytest.raises(ValueError, match="no velocity>0 note_on"):
        trim_melody_midi_leading_ticks(source, output)
    assert not output.exists()


def test_small_selection_writes_id_mid_manifests_hashes_and_no_npz(tmp_path):
    npz_root = tmp_path / "npz"
    npz_root.mkdir()
    records = []
    for order, piece_id in enumerate(("201", "202"), start=1):
        _write_npz(npz_root / f"{piece_id}.npz", first_onset_step=order)
        records.append(
            {
                "order": order,
                "style": "film",
                "id": piece_id,
                "title": f"Piece {piece_id}",
            }
        )
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    output_root = tmp_path / "output"

    rows = prepare_selection(
        selection,
        npz_root=npz_root,
        output_root=output_root,
    )

    assert [row["id"] for row in rows] == ["201", "202"]
    assert sorted(path.name for path in (output_root / "source_melody_midi").glob("*.mid")) == [
        "201.mid",
        "202.mid",
    ]
    assert sorted(path.name for path in (output_root / "trimmed_melody_midi").glob("*.mid")) == [
        "201.mid",
        "202.mid",
    ]
    manifest_rows = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    with (output_root / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == len(csv_rows) == 2
    for row in rows:
        assert row["source_sha256"] == sha256_file(Path(row["source_melody_midi"]))
        assert row["output_sha256"] == sha256_file(Path(row["trimmed_melody_midi"]))
        assert row["exact_tick_shift_verified"] is True
    assert list(output_root.rglob("*.npz")) == []
