#!/usr/bin/env python3
"""Pre-freeze and build the 24-clip blinded robustness listening package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.server
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import mido
import numpy as np
from scipy.signal import resample_poly

from streammuse.experiments.melody_robustness import (
    SEEDS,
    build_listening_selection_manifest,
    build_run_schedule,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    validate_campaign_config,
    validate_frozen_qualification,
    validate_listening_selection_manifest,
    validate_staged_input_manifest,
    verify_attempt_verdict,
    write_canonical_json,
)
from streammuse.experiments.robustness_metrics import Roll, load_midi_roll, write_roll_midi
from streammuse.experiments.triangle_midi import build_formal_triangle_excerpt
from streammuse.experiments.triangle_listening import (
    BLOCK_COUNTS as TRIANGLE_BLOCK_COUNTS,
    TRIANGLE_CLIP_BEATS,
    TRIANGLE_CLIP_SECONDS,
    TRIANGLE_COVERAGE_COLLAPSE_MAX_RATIO,
    TRIANGLE_COVERAGE_REFERENCE_MIN_RATIO,
    TRIANGLE_GAIN_POLICY,
    TRIANGLE_LISTENING_ATTEMPT_ID,
    TRIANGLE_PRACTICE_COUNT,
    TRIANGLE_PRESENTATION_COUNT,
    TRIANGLE_PATTERNS,
    TRIANGLE_PROMPT,
    TRIANGLE_RENDER_BPM,
    TRIANGLE_RENDER_SAMPLE_RATE,
    TRIANGLE_SELECTION_SCHEMA_VERSION,
    TRIANGLE_SYNTH_GAIN,
    TRIANGLE_TRIAL_COUNT,
    append_response,
    append_sitting_event,
    build_triangle_control_roll,
    build_triangle_selection_manifest,
    create_snapshot,
    derive_triangle_retry_selection_manifest,
    export_generated_acc,
    progress_summary,
    summarize_unblinded,
    unblind_snapshot,
    validate_response_ledger,
    validate_snapshot,
    validate_sitting_ledger,
    validate_triangle_renderer_identity,
    validate_triangle_selection_manifest,
    triangle_listening_attempt_number,
)


RENDER_BPM = 120
CLIP_SECONDS = 25
CLIP_BEATS = 50  # 25 seconds at 120 BPM
CLIP_TICKS = CLIP_BEATS * 4
RENDER_SAMPLE_RATE = 44100
FIXED_SYNTH_GAIN = 0.5
GAIN_POLICY = "fixed_pair_gain_with_true_peak_protection_only"
TRUE_PEAK_LIMIT_DBTP = -0.1
TRUE_PEAK_OVERSAMPLE = 4
TRUE_PEAK_IMPLEMENTATION = "scipy_resample_poly_4x_kaiser_8.6"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _song(entry: Mapping[str, Any]) -> str:
    return str(entry.get("song", entry.get("source_stem")))


def _song_analysis_horizons(entries: list[dict[str, Any]]) -> dict[str, int]:
    """Return the single frozen clean analysis horizon for each song."""
    grouped: dict[str, set[int]] = {}
    for entry in entries:
        song = _song(entry)
        raw = entry.get("analysis_end_tick")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(
                f"{song}: input manifest requires a positive integer analysis_end_tick"
            )
        grouped.setdefault(song, set()).add(raw)
    inconsistent = {song: sorted(values) for song, values in grouped.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"analysis horizon differs across variants: {inconsistent}")
    return {song: next(iter(values)) for song, values in grouped.items()}


def _validate_excerpt(
    *, song: str, start_beat: Any, end_beat: Any, analysis_end_tick: int
) -> None:
    if (
        isinstance(start_beat, bool)
        or not isinstance(start_beat, int)
        or isinstance(end_beat, bool)
        or not isinstance(end_beat, int)
    ):
        raise ValueError(f"{song}: excerpt beats must be integers")
    if start_beat < 0:
        raise ValueError(f"{song}: excerpt start must be non-negative")
    if end_beat != start_beat + CLIP_BEATS:
        raise ValueError(
            f"{song}: excerpt end must equal start + {CLIP_BEATS} model beats"
        )
    if end_beat * 4 > analysis_end_tick:
        raise ValueError(
            f"{song}: excerpt [{start_beat}, {end_beat}) beats exceeds "
            f"analysis_end_tick={analysis_end_tick}"
        )


def _path(entry: Mapping[str, Any], base: Path, *names: str) -> Path:
    containers = [entry]
    for nested in ("paths", "artifacts", "files"):
        if isinstance(entry.get(nested), Mapping):
            containers.append(entry[nested])  # type: ignore[arg-type]
    for container in containers:
        for name in names:
            value = container.get(name)
            if isinstance(value, str):
                candidate = Path(value)
                return (candidate if candidate.is_absolute() else base / candidate).resolve()
            if isinstance(value, Mapping) and value.get("path"):
                candidate = Path(str(value["path"]))
                return (candidate if candidate.is_absolute() else base / candidate).resolve()
    raise KeyError(f"missing artifact path {names}")


def _validated_build_inputs(
    args: argparse.Namespace,
    selection_path: Path,
    selection: dict[str, Any],
) -> tuple[dict[str, Any], str, list[dict[str, Any]], Path, list[dict[str, Any]]]:
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    if config_sha != args.config_sha256:
        raise RuntimeError("campaign config hash mismatch")
    config = _read_json(config_path)
    validate_campaign_config(config)
    validate_frozen_qualification(config, verify_files=True)
    checkpoint_path = Path(config["checkpoint"]["path"]).resolve()
    if file_sha256(checkpoint_path) != str(config["checkpoint"]["sha256"]):
        raise RuntimeError("checkpoint hash mismatch with frozen campaign config")

    selection_sha = file_sha256(selection_path)
    if selection_sha != config["listening"].get("selection_manifest_sha256"):
        raise RuntimeError("listening selection hash does not match frozen campaign config")
    configured_selection = Path(
        config["listening"].get("selection_manifest_path", "")
    ).resolve()
    if configured_selection != selection_path.resolve():
        raise RuntimeError("listening selection path does not match frozen campaign config")
    for field in ("render_bpm", "clip_seconds", "clip_count", "gain_policy"):
        if selection.get(field) != config["listening"].get(field):
            raise RuntimeError(f"listening selection {field} differs from campaign config")

    manifest_path = Path(selection["input_manifest_path"]).resolve()
    configured_manifest = Path(config["input_manifest"]["path"]).resolve()
    if manifest_path != configured_manifest:
        raise RuntimeError("selection and campaign config reference different input manifests")
    manifest_sha = file_sha256(manifest_path)
    if manifest_sha != selection.get("input_manifest_sha256"):
        raise RuntimeError("selection input manifest hash mismatch")
    if manifest_sha != config["input_manifest"].get("sha256"):
        raise RuntimeError("campaign input manifest hash mismatch")
    input_manifest = _read_json(manifest_path)
    entries = validate_staged_input_manifest(
        input_manifest, manifest_path=manifest_path, verify_files=True
    )
    validate_listening_selection_manifest(
        selection,
        input_manifest,
        manifest_path=manifest_path,
        verify_files=True,
    )

    schedule_path = Path(args.schedule).resolve()
    if file_sha256(schedule_path) != args.schedule_sha256:
        raise RuntimeError("run schedule hash mismatch")
    schedule = read_jsonl(schedule_path)
    if schedule != build_run_schedule(input_manifest, config):
        raise RuntimeError(
            "run schedule is not the deterministic 160-row schedule rebuilt from "
            "the frozen campaign config and input manifest"
        )
    return config, config_sha, schedule, manifest_path, entries


def _validated_campaign_binding(
    args: argparse.Namespace, config: Mapping[str, Any], config_sha: str
) -> dict[str, Any]:
    output_root = Path(args.output_root).resolve()
    binding_path = output_root / "campaign_binding.json"
    expected = {
        "schema_version": "streammuse.melody_robustness.campaign_binding.v1",
        "qualification": False,
        "campaign_config_path": str(Path(args.config).resolve()),
        "campaign_config_sha256": config_sha,
        "run_schedule_path": str(Path(args.schedule).resolve()),
        "run_schedule_sha256": args.schedule_sha256,
        "input_manifest_path": str(Path(config["input_manifest"]["path"]).resolve()),
        "input_manifest_sha256": str(config["input_manifest"]["sha256"]),
        "checkpoint_path": str(Path(config["checkpoint"]["path"]).resolve()),
        "checkpoint_sha256": str(config["checkpoint"]["sha256"]),
        "code_identity": str(config["code_identity"]),
        "qualification_result_sha256": str(config["qualification_result"]["sha256"]),
    }
    if not binding_path.is_file() or _read_json(binding_path) != expected:
        raise RuntimeError("model output root is not bound to this exact formal campaign")
    return {**expected, "campaign_binding_sha256": file_sha256(binding_path)}


def freeze_selection(args: argparse.Namespace) -> None:
    input_path = Path(args.input_manifest).resolve()
    manifest = _read_json(input_path)
    entries = validate_staged_input_manifest(
        manifest, manifest_path=input_path, verify_files=True
    )
    songs = sorted({_song(entry) for entry in entries})
    horizons = _song_analysis_horizons(entries)
    pseed = int(args.perturb_seed)
    sseed = int(args.sample_seed)
    blind_order_seed = int(args.blind_order_seed)
    if pseed not in {int(seed) for seed in SEEDS["perturb"]}:
        raise ValueError(
            f"perturb seed {pseed} is outside the frozen contract: {SEEDS['perturb']}"
        )
    if sseed not in {int(seed) for seed in SEEDS["sample"]}:
        raise ValueError(
            f"sample seed {sseed} is outside the frozen contract: {SEEDS['sample']}"
        )
    if blind_order_seed != int(SEEDS["blind_order"]):
        raise ValueError(
            "blind-order seed must match the frozen contract: "
            f"{SEEDS['blind_order']}"
        )
    if isinstance(args.excerpt_start_beat, bool) or not isinstance(args.excerpt_start_beat, int):
        raise ValueError("--excerpt-start-beat must be an integer")
    starts = {song: args.excerpt_start_beat for song in songs}
    if args.excerpt_starts_json:
        supplied = _read_json(Path(args.excerpt_starts_json))
        unknown = set(map(str, supplied)) - set(songs)
        if unknown:
            raise ValueError(f"excerpt starts name unknown songs: {sorted(unknown)}")
        for song, beat in supplied.items():
            if isinstance(beat, bool) or not isinstance(beat, int):
                raise ValueError(f"{song}: excerpt start must be an integer")
            starts[str(song)] = beat
    for song in songs:
        _validate_excerpt(
            song=song,
            start_beat=starts[song],
            end_beat=starts[song] + CLIP_BEATS,
            analysis_end_tick=horizons[song],
        )
    payload = build_listening_selection_manifest(
        manifest,
        manifest_path=input_path,
        perturb_seed=pseed,
        sample_seed=sseed,
        blind_order_seed=blind_order_seed,
        excerpt_starts=starts,
        verify_files=True,
    )
    digest = write_canonical_json(Path(args.output), payload)
    print(json.dumps({"path": str(Path(args.output).resolve()), "sha256": digest, "clips": 24}))


def _verified_attempt(
    output_root: Path,
    row: Mapping[str, Any],
    expected_binding: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any], set[Path]]:
    run_id = str(row["run_id"])
    run_dir = output_root / "runs" / run_id
    pointer = run_dir / "latest_verdict.json"
    if not pointer.is_file():
        raise RuntimeError(f"listening source has no latest verdict: {run_id}")
    verdict = _read_json(pointer)
    if verdict.get("run_id") != run_id or verdict.get("pipeline") != row.get("pipeline"):
        raise RuntimeError(f"listening verdict identity mismatch: {run_id}")
    if verdict.get("content_valid") is not True or verdict.get("operational_valid") is not True:
        raise RuntimeError(f"listening source run is not content/operational valid: {run_id}")
    if expected_binding is not None:
        for field in (
            "campaign_config_sha256",
            "run_schedule_sha256",
            "input_manifest_sha256",
            "checkpoint_sha256",
            "code_identity",
            "campaign_binding_sha256",
            "qualification_result_sha256",
        ):
            if verdict.get(field) != expected_binding.get(field):
                raise RuntimeError(f"listening verdict {field} campaign binding mismatch: {run_id}")
    attempt_id = str(verdict.get("attempt_id", ""))
    if re.fullmatch(r"attempt-[0-9]{3}", attempt_id) is None:
        raise RuntimeError(f"invalid listening source attempt ID: {run_id}/{attempt_id}")
    attempt = (run_dir / attempt_id).resolve()
    if not attempt.is_dir() or not attempt.is_relative_to(run_dir.resolve()):
        raise RuntimeError(f"listening source attempt escapes run directory: {run_id}")
    immutable = attempt / "verdict.json"
    if not immutable.is_file() or _read_json(immutable) != verdict:
        raise RuntimeError(f"mutable/immutable verdict mismatch: {run_id}")
    raw_index = verdict.get("artifact_index")
    if not isinstance(raw_index, list) or not raw_index:
        raise RuntimeError(f"listening source has no artifact index: {run_id}")
    indexed: set[Path] = set()
    for record in raw_index:
        if not isinstance(record, Mapping):
            raise RuntimeError(f"malformed artifact index: {run_id}")
        relative = record.get("path")
        digest = record.get("sha256")
        size = record.get("size")
        if not isinstance(relative, str) or not relative or not isinstance(digest, str):
            raise RuntimeError(f"malformed artifact index record: {run_id}")
        path = (attempt / relative).resolve()
        if path in indexed or not path.is_relative_to(attempt):
            raise RuntimeError(f"duplicate/escaping indexed artifact: {run_id}/{relative}")
        if (
            not path.is_file()
            or file_sha256(path) != digest
            or isinstance(size, bool)
            or not isinstance(size, int)
            or path.stat().st_size != size
        ):
            raise RuntimeError(f"missing/corrupt indexed artifact: {run_id}/{relative}")
        indexed.add(path)
    return attempt, verdict, indexed


def _single(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {matches}")
    return matches[0]


def _find_run(
    schedule: list[dict[str, Any]], clip: Mapping[str, Any]
) -> dict[str, Any]:
    candidates = [
        row for row in schedule
        if row["pipeline"] == "rt" and row["song"] == clip["song"]
        and row["condition"] == clip["condition"]
        and row["sample_seed"] == clip["sample_seed"]
        and row.get("perturb_seed") == clip.get("perturb_seed")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"listening selector matched {len(candidates)} runs: {clip}")
    return candidates[0]


def _slice(roll: Roll, start: int, length: int = CLIP_TICKS) -> Roll:
    return Roll(
        end_tick=length,
        sustain=frozenset((tick - start, pitch) for tick, pitch in roll.sustain if start <= tick < start + length),
        onsets=frozenset((tick - start, pitch) for tick, pitch in roll.onsets if start <= tick < start + length),
    )


def _exact_model_tick(raw_tick: int, ticks_per_beat: int) -> int:
    """Convert a MIDI tick to the frozen four-logical-ticks-per-beat grid.

    Listening excerpts are evidence artifacts, so silently rounding an event
    onto the model grid would destroy provenance.  Formal model output is
    already quantized to this grid; reject anything that is not exact.
    """

    scaled = int(raw_tick) * 4
    tick, remainder = divmod(scaled, int(ticks_per_beat))
    if remainder:
        raise ValueError(
            f"MIDI event at source tick {raw_tick} is off the four-ticks-per-beat grid"
        )
    return tick


def _theoretical_note_events(
    path: Path, *, start_model_tick: int, end_model_tick: int
) -> list[dict[str, int]]:
    """Extract the frozen theoretical accompaniment window without losing velocity.

    Notes that are already active at ``start_model_tick`` are reconstructed
    with a note-on at local tick zero.  This is the explicit pre-roll policy:
    the excerpt represents the state at the left edge rather than dropping a
    sustain merely because its original onset occurred before the window.
    """

    if start_model_tick < 0 or end_model_tick <= start_model_tick:
        raise ValueError("formal excerpt requires a non-empty non-negative window")
    midi = mido.MidiFile(str(path))
    named_tracks: list[tuple[int, mido.MidiTrack]] = []
    note_tracks: list[tuple[int, mido.MidiTrack]] = []
    for track_index, track in enumerate(midi.tracks):
        names = [
            str(message.name)
            for message in track
            if message.type == "track_name"
        ]
        if "Theoretical Accompaniment" in names:
            named_tracks.append((track_index, track))
        if any(message.type in {"note_on", "note_off"} for message in track):
            note_tracks.append((track_index, track))
    if len(named_tracks) == 1:
        track_index, target = named_tracks[0]
        if any(index != track_index for index, _track in note_tracks):
            raise ValueError(
                "theoretical_model.mid contains notes outside the theoretical track"
            )
    elif not named_tracks and not note_tracks:
        # pretty_midi omits empty instruments when serializing.  An entirely
        # empty theoretical schedule is therefore represented by a valid MIDI
        # with no named/note track and must remain literal silence.
        return []
    elif not named_tracks and len(note_tracks) == 1:
        # Compatibility with minimal test/legacy exporters.  Formal runs are
        # independently required to contain only the theoretical model track.
        track_index, target = note_tracks[0]
    else:
        raise ValueError(
            "theoretical_model.mid must contain exactly one "
            "'Theoretical Accompaniment' note track"
        )

    active: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    completed: list[tuple[int, int, int, int]] = []
    absolute = 0
    for message in target:
        absolute += int(message.time)
        if message.type not in {"note_on", "note_off"}:
            continue
        channel = int(message.channel)
        if channel == 9:
            raise ValueError("theoretical accompaniment unexpectedly uses the drum channel")
        tick = _exact_model_tick(absolute, midi.ticks_per_beat)
        pitch = int(message.note)
        key = (track_index, channel, pitch)
        is_on = message.type == "note_on" and int(message.velocity) > 0
        if is_on:
            active[key].append((tick, int(message.velocity)))
            continue
        if not active.get(key):
            raise ValueError(
                f"unmatched note-off in theoretical accompaniment at tick {tick}, pitch {pitch}"
            )
        note_start, velocity = active[key].pop(0)
        completed.append((note_start, max(note_start + 1, tick), pitch, velocity))

    # A hanging source note is treated as active through the right edge.  The
    # formal run validator normally prevents this, but the deterministic rule
    # makes reconstruction total and keeps the excerpt bounded.
    for (_track, _channel, pitch), starts in active.items():
        for note_start, velocity in starts:
            completed.append(
                (note_start, max(note_start + 1, end_model_tick), pitch, velocity)
            )

    events: list[dict[str, int]] = []
    for note_start, note_end, pitch, velocity in completed:
        if note_start >= end_model_tick or note_end <= start_model_tick:
            continue
        local_start = max(note_start, start_model_tick) - start_model_tick
        local_end = min(note_end, end_model_tick) - start_model_tick
        if local_end <= local_start:
            continue
        events.append(
            {
                "start_model_tick": local_start,
                "end_model_tick": local_end,
                "pitch": pitch,
                "velocity": velocity,
            }
        )
    events.sort(
        key=lambda row: (
            row["start_model_tick"],
            row["pitch"],
            row["end_model_tick"],
            row["velocity"],
        )
    )
    return events


def _note_events_roll(
    events: Sequence[Mapping[str, int]], *, end_model_tick: int
) -> Roll:
    sustain: set[tuple[int, int]] = set()
    onsets: set[tuple[int, int]] = set()
    for event in events:
        start = int(event["start_model_tick"])
        end = int(event["end_model_tick"])
        pitch = int(event["pitch"])
        onsets.add((start, pitch))
        sustain.update((tick, pitch) for tick in range(start, end))
    return Roll(
        end_tick=end_model_tick,
        sustain=frozenset(sustain),
        onsets=frozenset(onsets),
    )


def _write_note_event_midi(
    events: Sequence[Mapping[str, int]], path: Path, *, bpm: int
) -> None:
    """Write deterministic program-0 canonical MIDI while preserving velocities."""

    path.parent.mkdir(parents=True, exist_ok=True)
    ticks_per_beat = 480
    step = ticks_per_beat // 4
    timeline: list[tuple[int, int, int, int, mido.Message]] = []
    for index, event in enumerate(events):
        start = int(event["start_model_tick"])
        end = int(event["end_model_tick"])
        pitch = int(event["pitch"])
        velocity = int(event["velocity"])
        if start < 0 or end <= start:
            raise ValueError("canonical note event has an invalid interval")
        if not 0 <= pitch <= 127 or not 1 <= velocity <= 127:
            raise ValueError("canonical note event has an invalid pitch or velocity")
        timeline.append(
            (
                start * step,
                1,
                pitch,
                index,
                mido.Message("note_on", channel=0, note=pitch, velocity=velocity),
            )
        )
        timeline.append(
            (
                end * step,
                0,
                pitch,
                index,
                mido.Message("note_off", channel=0, note=pitch, velocity=0),
            )
        )
    timeline.sort(key=lambda row: row[:4])
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(meta)
    track = mido.MidiTrack()
    # MIDI defaults to bank zero when bank-select messages are absent.  Keeping
    # the stream minimal also makes it identical to synthetic canonical MIDI.
    track.append(mido.Message("program_change", channel=0, program=0, time=0))
    previous = 0
    for absolute, _priority, _pitch, _index, message in timeline:
        message.time = absolute - previous
        track.append(message)
        previous = absolute
    midi.tracks.append(track)
    midi.save(str(path))


def _canonical_midi_contract(
    path: Path, *, expected_end_model_tick: int
) -> list[dict[str, int]]:
    """Independently parse and validate a frozen listening MIDI contract."""

    midi = mido.MidiFile(str(path))
    if midi.type != 1 or midi.ticks_per_beat != 480:
        raise ValueError("canonical MIDI must be type 1 with 480 ticks per beat")
    tempos: list[tuple[int, int]] = []
    programs: list[tuple[int, int, int]] = []
    bank_messages: list[tuple[int, int, int, int]] = []
    timeline: list[tuple[int, int, mido.Message]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute = 0
        for message in track:
            absolute += int(message.time)
            if message.type == "set_tempo":
                tempos.append((absolute, int(message.tempo)))
            if message.type == "program_change":
                programs.append((absolute, int(message.channel), int(message.program)))
            if message.type == "control_change" and int(message.control) in {0, 32}:
                bank_messages.append(
                    (
                        absolute,
                        int(message.channel),
                        int(message.control),
                        int(message.value),
                    )
                )
            if message.type in {"note_on", "note_off"}:
                timeline.append((absolute, track_index, message))
    expected_tempo = int(mido.bpm2tempo(TRIANGLE_RENDER_BPM))
    if tempos != [(0, expected_tempo)]:
        raise ValueError("canonical MIDI must contain exactly one 120 BPM tempo at tick zero")
    if programs != [(0, 0, 0)]:
        raise ValueError("canonical MIDI must select exactly program 0 on channel 0")
    if any(
        absolute != 0 or channel != 0 or value != 0
        for absolute, channel, _control, value in bank_messages
    ):
        raise ValueError("canonical MIDI bank select must resolve to bank 0 on channel 0")

    timeline.sort(key=lambda row: (row[0], row[1]))
    active: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    notes: list[dict[str, int]] = []
    for absolute, track_index, message in timeline:
        if int(message.channel) != 0:
            raise ValueError("canonical MIDI notes must use channel 0")
        tick = _exact_model_tick(absolute, midi.ticks_per_beat)
        pitch = int(message.note)
        key = (track_index, int(message.channel), pitch)
        if message.type == "note_on" and int(message.velocity) > 0:
            if tick >= expected_end_model_tick:
                raise ValueError("canonical MIDI note-on lies outside the excerpt")
            active[key].append((tick, int(message.velocity)))
        else:
            if not active.get(key):
                raise ValueError("canonical MIDI contains an unmatched note-off")
            start, velocity = active[key].pop(0)
            if tick <= start or tick > expected_end_model_tick:
                raise ValueError("canonical MIDI note interval lies outside the excerpt")
            notes.append(
                {
                    "start_model_tick": start,
                    "end_model_tick": tick,
                    "pitch": pitch,
                    "velocity": velocity,
                }
            )
    if any(starts for starts in active.values()):
        raise ValueError("canonical MIDI contains an unterminated note")
    notes.sort(
        key=lambda row: (
            row["start_model_tick"],
            row["pitch"],
            row["end_model_tick"],
            row["velocity"],
        )
    )
    return notes


def _note_event_sha256(events: Sequence[Mapping[str, int]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(events))).hexdigest()


def _triangle_objective_identity(
    note_events_a: Sequence[Mapping[str, int]],
    note_events_b: Sequence[Mapping[str, int]],
    *,
    wav_a: Path | None = None,
    wav_b: Path | None = None,
) -> bool:
    """Return the frozen note-or-final-audio identity decision.

    MIDI-only development packages can only establish canonical note identity.
    Accepted listening packages additionally treat byte-identical, post-pair-gain
    WAVs as objective identities even when their source note events differ.
    """

    if list(note_events_a) == list(note_events_b):
        return True
    if (wav_a is None) != (wav_b is None):
        raise ValueError("objective identity requires either zero or two final WAVs")
    if wav_a is None:
        return False
    if not wav_a.is_file() or not wav_b.is_file():
        raise FileNotFoundError("objective identity final WAV is missing")
    return wav_a.read_bytes() == wav_b.read_bytes()


def _rebuild_formal_excerpt_midi(
    theoretical: Path,
    destination: Path,
    *,
    start_model_tick: int,
    end_model_tick: int,
) -> tuple[Roll, list[dict[str, int]]]:
    roll, events = build_formal_triangle_excerpt(
        theoretical,
        destination,
        start_model_tick=start_model_tick,
        end_model_tick=end_model_tick,
        bpm=TRIANGLE_RENDER_BPM,
    )
    length = end_model_tick - start_model_tick
    parsed = _canonical_midi_contract(
        destination, expected_end_model_tick=length
    )
    if parsed != events:
        raise RuntimeError("canonical MIDI writer changed the extracted note event stream")
    return roll, events


def _union(left: Roll, right: Roll) -> Roll:
    return Roll(
        end_tick=max(left.end_tick, right.end_tick),
        sustain=frozenset(left.sustain | right.sustain),
        onsets=frozenset(left.onsets | right.onsets),
    )


def _render(
    midi_path: Path, wav_path: Path, *, soundfont: Path, fluidsynth: str,
    sample_rate: int, gain: float,
) -> dict[str, Any]:
    command = [
        fluidsynth, "-ni", str(soundfont), str(midi_path),
        "-F", str(wav_path), "-r", str(sample_rate), "-g", str(gain),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"fluidsynth failed ({result.returncode}): {result.stderr}")
    return {"command": command, "stdout": result.stdout, "stderr": result.stderr}


def _measure_true_peak(samples: np.ndarray) -> float:
    """Measure inter-sample peak with a frozen 4x polyphase reconstruction.

    ``samples`` is frame-major, float64 PCM normalized so digital full scale is
    1.0.  Oversampling each channel together on the time axis exposes peaks
    between the stored samples; using an explicit Kaiser window makes the
    measurement deterministic across all package builds using the frozen
    SciPy environment.
    """
    if samples.ndim != 2:
        raise ValueError("true-peak input must be frame-major [frames, channels]")
    if samples.size == 0:
        return 0.0
    reconstructed = resample_poly(
        samples,
        TRUE_PEAK_OVERSAMPLE,
        1,
        axis=0,
        window=("kaiser", 8.6),
        padtype="constant",
    )
    return float(np.max(np.abs(reconstructed), initial=0.0))


def _dbtp(linear_peak: float) -> float | None:
    return None if linear_peak <= 0.0 else float(20.0 * math.log10(linear_peak))


def _quantize_pcm16(samples: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(samples * 32768.0), -32768, 32767).astype("<i2")


def _fix_wav(path: Path, *, sample_rate: int, seconds: int) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        original_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if width != 2 or original_rate != sample_rate:
        raise RuntimeError(f"render must be 16-bit/{sample_rate}Hz, got {width * 8}/{original_rate}")
    raw = np.frombuffer(frames, dtype="<i2")
    if raw.size % channels:
        raise RuntimeError("rendered PCM does not contain complete channel frames")
    samples = raw.astype(np.float64).reshape(-1, channels) / 32768.0
    target_frames = sample_rate * seconds
    if samples.shape[0] < target_frames:
        samples = np.pad(samples, ((0, target_frames - samples.shape[0]), (0, 0)))
    elif samples.shape[0] > target_frames:
        samples = samples[:target_frames]

    sample_peak_before = int(np.max(np.abs(raw.astype(np.int32)), initial=0))
    true_peak_before = _measure_true_peak(samples)
    limit_linear = float(10.0 ** (TRUE_PEAK_LIMIT_DBTP / 20.0))
    scale = min(1.0, limit_linear / true_peak_before) if true_peak_before > 0 else 1.0
    protected = scale < 1.0
    quantized = _quantize_pcm16(samples * scale)
    normalized_after = quantized.astype(np.float64) / 32768.0
    true_peak_after = _measure_true_peak(normalized_after)
    # Quantization can move a value by half an LSB.  A second attenuation pass
    # keeps the recorded post-quantization WAV itself beneath the frozen limit.
    if true_peak_after > limit_linear:
        correction = (limit_linear / true_peak_after) * (1.0 - 1e-9)
        scale *= correction
        protected = True
        quantized = _quantize_pcm16(samples * scale)
        normalized_after = quantized.astype(np.float64) / 32768.0
        true_peak_after = _measure_true_peak(normalized_after)
    if true_peak_after > limit_linear + 1e-9:
        raise RuntimeError(
            f"true-peak protection failed: {_dbtp(true_peak_after)} dBTP exceeds "
            f"{TRUE_PEAK_LIMIT_DBTP} dBTP"
        )
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(channels)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(quantized.reshape(-1).tobytes())
    return {
        "channels": channels, "sample_width_bits": 16, "sample_rate": sample_rate,
        "frames": sample_rate * seconds, "seconds": seconds,
        "sample_peak_before": sample_peak_before,
        "sample_peak_after": int(
            np.max(np.abs(quantized.astype(np.int32)), initial=0)
        ),
        "true_peak_linear_before": true_peak_before,
        "true_peak_dbtp_before": _dbtp(true_peak_before),
        "true_peak_linear_after": true_peak_after,
        "true_peak_dbtp_after": _dbtp(true_peak_after),
        "true_peak_limit_dbtp": TRUE_PEAK_LIMIT_DBTP,
        "true_peak_protection_applied": protected,
        "applied_gain_scale": scale,
        "peak_measurement": TRUE_PEAK_IMPLEMENTATION,
        "true_peak_oversample_factor": TRUE_PEAK_OVERSAMPLE,
        "true_peak_verified": True,
        "silent": not bool(np.any(quantized)),
    }


def build_package(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    selection = _read_json(selection_path)
    if selection.get("schema_version") != "streammuse.melody_robustness.listening_selection.v1":
        raise ValueError("unsupported listening selection schema")
    if selection.get("frozen_before_formal") is not True:
        raise ValueError("listening selection must be frozen before formal outputs")
    if (
        selection.get("clip_count") != 24
        or selection.get("clip_seconds") != CLIP_SECONDS
        or selection.get("render_bpm") != RENDER_BPM
        or selection.get("gain_policy") != GAIN_POLICY
        or len(selection.get("clips", [])) != 24
    ):
        raise ValueError("selection must contain exactly 24 clips")
    config, config_sha, schedule, manifest_path, entries = _validated_build_inputs(
        args, selection_path, selection
    )
    output_root = Path(args.output_root).resolve()
    package = Path(args.package_dir).resolve()
    horizons = _song_analysis_horizons(entries)
    if selection.get("analysis_horizons_ticks") != horizons:
        raise RuntimeError(
            "selection analysis horizons do not match its hash-pinned input manifest"
        )
    for clip in selection["clips"]:
        song = str(clip.get("song"))
        if song not in horizons:
            raise ValueError(f"selection names unknown song: {song}")
        if clip.get("analysis_end_tick") != horizons[song]:
            raise RuntimeError(f"{song}: selection analysis horizon drifted after freeze")
        _validate_excerpt(
            song=song,
            start_beat=clip.get("excerpt_start_model_beat"),
            end_beat=clip.get("excerpt_end_model_beat"),
            analysis_end_tick=horizons[song],
        )
    campaign_binding = _validated_campaign_binding(args, config, config_sha)
    manifest_dir = manifest_path.parent
    entry_index = {
        (_song(entry), str(entry["condition"]), entry.get("perturb_seed")): entry
        for entry in entries
    }
    soundfont = Path(args.soundfont).resolve() if args.soundfont else None
    if args.sample_rate != RENDER_SAMPLE_RATE:
        raise ValueError(f"accepted listening render requires {RENDER_SAMPLE_RATE} Hz")
    if args.gain != FIXED_SYNTH_GAIN:
        raise ValueError(f"accepted listening render requires fixed synth gain {FIXED_SYNTH_GAIN}")
    if not args.midi_only and (soundfont is None or not soundfont.is_file()):
        raise FileNotFoundError("--soundfont is required for an accepted WAV package")
    synth_version = None
    if not args.midi_only:
        version_result = subprocess.run([args.fluidsynth, "--version"], text=True, capture_output=True, check=False)
        synth_version = (version_result.stdout or version_result.stderr).strip()
        if version_result.returncode:
            raise RuntimeError(f"cannot identify fluidsynth: {synth_version}")
    verified_runs: dict[str, tuple[Path, dict[str, Any], set[Path]]] = {}
    for clip in selection["clips"]:
        if clip.get("anchor_kind") == "known_bad_harmonic_m2":
            continue
        row = _find_run(schedule, clip)
        verified_runs.setdefault(
            str(row["run_id"]),
            _verified_attempt(output_root, row, campaign_binding),
        )
    midi_dir = package / "blind" / "midi"
    wav_dir = package / "blind" / "wav"
    midi_dir.mkdir(parents=True, exist_ok=False)
    wav_dir.mkdir(parents=True, exist_ok=False)
    key_rows = []
    render_rows = []
    for clip in selection["clips"]:
        sample_id = str(clip["sample_id"])
        semantic_index = int(clip["semantic_index"])
        start_tick = int(clip["excerpt_start_model_beat"]) * 4
        if clip.get("anchor_kind") == "known_bad_harmonic_m2":
            source_path = Path(args.controls_root).resolve() / clip["song"] / "harmonic_m2.mid"
            accompaniment = load_midi_roll(source_path, end_tick=start_tick + CLIP_TICKS)
            entry = entry_index[(clip["song"], "sham", None)]
            melody_path = _path(entry, manifest_dir, "output_midi", "melody_midi")
            melody = load_midi_roll(melody_path, end_tick=start_tick + CLIP_TICKS)
            rendered_roll = _union(_slice(melody, start_tick), _slice(accompaniment, start_tick))
            source_run_id = None
        else:
            row = _find_run(schedule, clip)
            attempt, _verdict, indexed_artifacts = verified_runs[str(row["run_id"])]
            if clip["pipeline"] == "rt_theoretical":
                acc_path = _single(attempt, "theoretical_model.mid")
                if acc_path.resolve() not in indexed_artifacts:
                    raise RuntimeError(f"theoretical MIDI is not verdict-indexed: {row['run_id']}")
                accompaniment = load_midi_roll(acc_path, end_tick=start_tick + CLIP_TICKS)
                rendered_roll = _slice(accompaniment, start_tick)
            else:
                acc_path = _single(attempt, "combined.mid")
                if acc_path.resolve() not in indexed_artifacts:
                    raise RuntimeError(f"combined MIDI is not verdict-indexed: {row['run_id']}")
                accompaniment = load_midi_roll(
                    acc_path,
                    end_tick=start_tick + CLIP_TICKS,
                    track_name_contains="Accompaniment",
                )
                entry = entry_index[(clip["song"], clip["condition"], clip.get("perturb_seed"))]
                melody_path = _path(entry, manifest_dir, "output_midi", "melody_midi")
                melody = load_midi_roll(melody_path, end_tick=start_tick + CLIP_TICKS)
                rendered_roll = _union(_slice(melody, start_tick), _slice(accompaniment, start_tick))
            source_path = acc_path
            source_run_id = row["run_id"]
        midi_path = midi_dir / f"{sample_id}.mid"
        write_roll_midi(rendered_roll, midi_path, bpm=RENDER_BPM)
        audio_info = None
        render_info = None
        wav_path = wav_dir / f"{sample_id}.wav"
        if not args.midi_only:
            render_info = _render(
                midi_path, wav_path, soundfont=soundfont, fluidsynth=args.fluidsynth,
                sample_rate=args.sample_rate, gain=args.gain,
            )
            audio_info = _fix_wav(wav_path, sample_rate=args.sample_rate, seconds=CLIP_SECONDS)
        key_rows.append(
            {
                "sample_id": sample_id, "semantic_index": semantic_index,
                "block": clip["block"], "song": clip["song"],
                "condition": clip["condition"], "pipeline": clip["pipeline"],
                "perturb_seed": clip.get("perturb_seed"), "sample_seed": clip["sample_seed"],
                "source_run_id": source_run_id, "source_path": str(source_path),
                "source_sha256": file_sha256(source_path),
                "duplicate_semantic_index": clip.get("duplicate_semantic_index"),
            }
        )
        render_rows.append(
            {
                "sample_id": sample_id, "midi": str(midi_path.relative_to(package)),
                "midi_sha256": file_sha256(midi_path),
                "wav": str(wav_path.relative_to(package)) if wav_path.exists() else None,
                "wav_sha256": file_sha256(wav_path) if wav_path.exists() else None,
                "audio": audio_info, "render": render_info,
            }
        )
    # Repeated trials are literal byte copies of their frozen source clips;
    # this removes synthesizer nondeterminism as a possible consistency cue.
    source_sample_by_semantic = {
        int(row["semantic_index"]): str(row["sample_id"])
        for row in key_rows if row.get("duplicate_semantic_index") is None
    }
    render_by_sample = {str(row["sample_id"]): row for row in render_rows}
    for key_row in key_rows:
        duplicate_index = key_row.get("duplicate_semantic_index")
        if duplicate_index is None:
            continue
        source_id = source_sample_by_semantic[int(duplicate_index)]
        target_id = str(key_row["sample_id"])
        source_render = render_by_sample[source_id]
        target_render = render_by_sample[target_id]
        source_midi = package / source_render["midi"]
        target_midi = package / target_render["midi"]
        shutil.copyfile(source_midi, target_midi)
        target_render["midi_sha256"] = file_sha256(target_midi)
        if not args.midi_only:
            source_wav = package / source_render["wav"]
            target_wav = package / target_render["wav"]
            shutil.copyfile(source_wav, target_wav)
            target_render["wav_sha256"] = file_sha256(target_wav)
            target_render["audio"] = dict(source_render["audio"])
            target_render["render"] = {"byte_copy_of": source_id}
    with (package / "blind" / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "overall_quality_1_to_5", "flaw_note"])
        writer.writeheader()
        for clip in selection["clips"]:
            writer.writerow({"sample_id": clip["sample_id"], "overall_quality_1_to_5": "", "flaw_note": ""})
    write_canonical_json(package / "private_key.json", {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": args.schedule_sha256,
        "campaign_binding_sha256": campaign_binding["campaign_binding_sha256"],
        "qualification_result_sha256": config["qualification_result"]["sha256"],
        "selection_sha256": file_sha256(selection_path), "clips": key_rows,
        "unblind_only_after_scores_sealed": True,
    })
    private_key_sha = file_sha256(package / "private_key.json")
    write_canonical_json(package / "render_manifest.json", {
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": args.schedule_sha256,
        "campaign_binding_sha256": campaign_binding["campaign_binding_sha256"],
        "qualification_result_sha256": config["qualification_result"]["sha256"],
        "input_manifest_sha256": config["input_manifest"]["sha256"],
        "selection_path": str(selection_path), "selection_sha256": file_sha256(selection_path),
        "private_key_sha256": private_key_sha,
        "render_bpm": RENDER_BPM, "sample_rate": args.sample_rate, "bit_depth": 16,
        "gain": args.gain, "gain_policy": selection["gain_policy"],
        "true_peak_requirement": "genuine_inter_sample_true_peak_measurement_and_protection",
        "peak_implementation": TRUE_PEAK_IMPLEMENTATION,
        "soundfont_path": str(soundfont) if soundfont else None,
        "soundfont_sha256": file_sha256(soundfont) if soundfont else None,
        "synth": args.fluidsynth, "synth_version": synth_version,
        "clips": render_rows,
    })
    audit = audit_package_dir(package, require_wav=not args.midi_only)
    write_canonical_json(package / "package_audit.json", audit)
    if not audit["valid"]:
        raise RuntimeError(f"listening package audit failed: {audit['errors']}")


def audit_package_dir(package: Path, *, require_wav: bool = True) -> dict[str, Any]:
    package = package.resolve()
    errors: list[str] = []
    render = _read_json(package / "render_manifest.json")
    if render.get("render_bpm") != RENDER_BPM:
        errors.append(f"render BPM must be {RENDER_BPM}")
    if render.get("sample_rate") != RENDER_SAMPLE_RATE:
        errors.append(f"render sample rate must be {RENDER_SAMPLE_RATE}")
    if render.get("bit_depth") != 16:
        errors.append("render bit depth must be 16")
    if render.get("gain") != FIXED_SYNTH_GAIN:
        errors.append(f"render synth gain must be fixed at {FIXED_SYNTH_GAIN}")
    if render.get("gain_policy") != GAIN_POLICY:
        errors.append("render gain/peak policy does not match the frozen policy")
    if require_wav and render.get("peak_implementation") != TRUE_PEAK_IMPLEMENTATION:
        errors.append(
            "accepted WAV package requires the frozen 4x inter-sample true-peak "
            f"implementation {TRUE_PEAK_IMPLEMENTATION}"
        )
    clips = render.get("clips", [])
    if len(clips) != 24:
        errors.append(f"expected 24 render rows, got {len(clips)}")
    sample_ids = [str(row.get("sample_id")) for row in clips]
    if len(set(sample_ids)) != 24:
        errors.append("sample IDs are not one-to-one")
    key = _read_json(package / "private_key.json")
    key_sha = file_sha256(package / "private_key.json")
    if render.get("private_key_sha256") != key_sha:
        errors.append("private key hash mismatch")
    if key.get("selection_sha256") != render.get("selection_sha256"):
        errors.append("private key and render manifest selection hashes differ")
    if key.get("campaign_config_sha256") != render.get("campaign_config_sha256"):
        errors.append("private key and render manifest campaign hashes differ")
    if key.get("run_schedule_sha256") != render.get("run_schedule_sha256"):
        errors.append("private key and render manifest schedule hashes differ")
    if key.get("campaign_binding_sha256") != render.get("campaign_binding_sha256"):
        errors.append("private key and render manifest campaign binding hashes differ")
    if key.get("qualification_result_sha256") != render.get(
        "qualification_result_sha256"
    ):
        errors.append("private key and render manifest qualification hashes differ")
    selection_path_raw = render.get("selection_path")
    if not isinstance(selection_path_raw, str) or not Path(selection_path_raw).is_file():
        errors.append("selection path is missing from the render manifest")
    elif file_sha256(selection_path_raw) != render.get("selection_sha256"):
        errors.append("selection file hash mismatch")
    if {row["sample_id"] for row in key.get("clips", [])} != set(sample_ids):
        errors.append("blind key and render manifest do not map one-to-one")
    duplicate_groups: dict[int, list[dict[str, Any]]] = {}
    for row in key.get("clips", []):
        identity = int(
            row["duplicate_semantic_index"]
            if row.get("duplicate_semantic_index") is not None
            else row["semantic_index"]
        )
        duplicate_groups.setdefault(identity, []).append(row)
    repeated = [rows for rows in duplicate_groups.values() if len(rows) == 2]
    if len(repeated) != 2:
        errors.append(f"expected two repeated trials, found {len(repeated)}")
    render_by_id = {row["sample_id"]: row for row in clips}
    for pair in repeated:
        hashes = {
            render_by_id[row["sample_id"]].get("wav_sha256")
            or render_by_id[row["sample_id"]].get("midi_sha256")
            for row in pair
        }
        if len(hashes) != 1:
            errors.append(f"repeat trial audio mismatch: {[row['sample_id'] for row in pair]}")
    for row in clips:
        raw_midi = Path(str(row.get("midi", "")))
        midi = (package / raw_midi).resolve()
        if raw_midi.is_absolute() or not midi.is_relative_to(package):
            errors.append(f"MIDI path escapes package: {row.get('sample_id')}")
            continue
        if not midi.is_file() or file_sha256(midi) != row["midi_sha256"]:
            errors.append(f"missing/corrupt MIDI: {row.get('sample_id')}")
        else:
            try:
                midi_file = mido.MidiFile(midi)
                tempos = [
                    message.tempo
                    for track in midi_file.tracks
                    for message in track
                    if message.type == "set_tempo"
                ]
                expected_tempo = int(mido.bpm2tempo(RENDER_BPM))
                if not tempos or any(int(tempo) != expected_tempo for tempo in tempos):
                    errors.append(f"MIDI tempo is not fixed at {RENDER_BPM} BPM: {row['sample_id']}")
            except Exception as exc:
                errors.append(f"unreadable MIDI for tempo audit: {row.get('sample_id')}: {exc}")
        if require_wav:
            if not row.get("wav"):
                errors.append(f"missing WAV path: {row.get('sample_id')}")
                continue
            raw_wav = Path(str(row["wav"]))
            wav = (package / raw_wav).resolve()
            if raw_wav.is_absolute() or not wav.is_relative_to(package):
                errors.append(f"WAV path escapes package: {row.get('sample_id')}")
                continue
            if not wav.is_file() or file_sha256(wav) != row["wav_sha256"]:
                errors.append(f"missing/corrupt WAV: {row.get('sample_id')}")
                continue
            with wave.open(str(wav), "rb") as source:
                wav_rate = source.getframerate()
                wav_channels = source.getnchannels()
                wav_width = source.getsampwidth()
                wav_frame_count = source.getnframes()
                wav_frames = source.readframes(wav_frame_count)
                if wav_rate != RENDER_SAMPLE_RATE:
                    errors.append(f"sample-rate mismatch: {row['sample_id']}")
                if wav_channels not in {1, 2}:
                    errors.append(f"channel mismatch: {row['sample_id']}")
                if wav_width != 2:
                    errors.append(f"bit-depth mismatch: {row['sample_id']}")
                expected_frames = RENDER_SAMPLE_RATE * CLIP_SECONDS
                if wav_frame_count != expected_frames:
                    errors.append(f"duration mismatch: {row['sample_id']}")
            measured_true_peak_dbtp: float | None = None
            measured_silent: bool | None = None
            if wav_width == 2 and wav_channels in {1, 2}:
                wav_pcm = np.frombuffer(wav_frames, dtype="<i2")
                if wav_pcm.size % wav_channels:
                    errors.append(f"incomplete PCM channel frame: {row['sample_id']}")
                else:
                    measured_silent = not bool(np.any(wav_pcm))
                    measured_linear = _measure_true_peak(
                        wav_pcm.astype(np.float64).reshape(-1, wav_channels) / 32768.0
                    )
                    measured_true_peak_dbtp = _dbtp(measured_linear)
                    if (
                        measured_true_peak_dbtp is not None
                        and measured_true_peak_dbtp > TRUE_PEAK_LIMIT_DBTP
                    ):
                        errors.append(
                            f"independently measured true peak exceeds "
                            f"{TRUE_PEAK_LIMIT_DBTP} dBTP: {row['sample_id']} "
                            f"({measured_true_peak_dbtp} dBTP)"
                        )
                    if measured_silent:
                        errors.append(f"WAV contains no nonzero samples: {row['sample_id']}")
            audio = row.get("audio")
            if not isinstance(audio, Mapping):
                errors.append(f"missing audio audit metadata: {row['sample_id']}")
            else:
                if audio.get("silent") is not False:
                    errors.append(f"silent listening clip: {row['sample_id']}")
                if audio.get("true_peak_verified") is not True:
                    errors.append(
                        f"true peak is unverified (sample peak is insufficient): {row['sample_id']}"
                    )
                else:
                    measured = audio.get("true_peak_dbtp_after")
                    if (
                        isinstance(measured, bool)
                        or not isinstance(measured, (int, float))
                        or float(measured) > TRUE_PEAK_LIMIT_DBTP
                    ):
                        errors.append(
                            f"true peak exceeds {TRUE_PEAK_LIMIT_DBTP} dBTP or lacks a "
                            f"numeric measurement: {row['sample_id']}"
                        )
                    elif (
                        measured_true_peak_dbtp is None
                        or abs(float(measured) - measured_true_peak_dbtp) > 1e-7
                    ):
                        errors.append(
                            f"recorded true-peak metadata does not match independent WAV "
                            f"measurement: {row['sample_id']}"
                        )
                if audio.get("peak_measurement") != TRUE_PEAK_IMPLEMENTATION:
                    errors.append(f"true-peak implementation drift: {row['sample_id']}")
                if audio.get("true_peak_oversample_factor") != TRUE_PEAK_OVERSAMPLE:
                    errors.append(f"true-peak oversample factor drift: {row['sample_id']}")
                if audio.get("true_peak_limit_dbtp") != TRUE_PEAK_LIMIT_DBTP:
                    errors.append(f"true-peak limit drift: {row['sample_id']}")
                if measured_silent is not None and bool(audio.get("silent")) != measured_silent:
                    errors.append(f"recorded silence flag does not match WAV: {row['sample_id']}")
    valid = not errors
    return {
        "valid": valid,
        "accepted_final": valid and require_wav,
        "errors": errors,
        "clip_count": len(clips),
        "repeat_trial_count": len(repeated),
        "campaign_config_sha256": render.get("campaign_config_sha256"),
        "run_schedule_sha256": render.get("run_schedule_sha256"),
        "campaign_binding_sha256": render.get("campaign_binding_sha256"),
        "qualification_result_sha256": render.get("qualification_result_sha256"),
        "selection_sha256": render.get("selection_sha256"),
        "private_key_sha256": key_sha,
        "render_manifest_sha256": file_sha256(package / "render_manifest.json"),
        "true_peak_limitation": (
            None if not require_wav or all(
                isinstance(row.get("audio"), Mapping)
                and row["audio"].get("true_peak_verified") is True
                for row in clips
            )
            else "one or more clips lacks a verified post-quantization true-peak measurement"
        ),
    }


def audit_package(args: argparse.Namespace) -> None:
    result = audit_package_dir(Path(args.package_dir).resolve(), require_wav=not args.allow_midi_only)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(2)


def seal_scores(args: argparse.Namespace) -> None:
    package = Path(args.package_dir).resolve()
    package_audit_path = package / "package_audit.json"
    package_audit = _read_json(package_audit_path)
    if package_audit.get("accepted_final") is not True:
        raise RuntimeError("scores can only be sealed for an accepted final listening package")
    scores = package / "blind" / "scores.csv"
    rows = list(csv.DictReader(scores.open("r", encoding="utf-8")))
    if len(rows) != 24:
        raise ValueError("scores.csv must contain 24 rows")
    for row in rows:
        try:
            score = int(row["overall_quality_1_to_5"])
        except Exception as exc:
            raise ValueError(f"missing/invalid score for {row.get('sample_id')}") from exc
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError(f"score outside 1..5 for {row['sample_id']}")
    key_path = package / "private_key.json"
    key = _read_json(key_path)
    expected_ids = {str(row["sample_id"]) for row in key.get("clips", [])}
    score_ids = [str(row.get("sample_id")) for row in rows]
    if len(set(score_ids)) != 24 or set(score_ids) != expected_ids:
        raise ValueError("scores.csv sample IDs do not match the hash-pinned private key")
    payload = {
        "scores_path": str(scores), "scores_sha256": file_sha256(scores),
        "private_key_sha256": file_sha256(key_path),
        "package_audit_sha256": file_sha256(package_audit_path),
        "campaign_config_sha256": package_audit.get("campaign_config_sha256"),
        "run_schedule_sha256": package_audit.get("run_schedule_sha256"),
        "campaign_binding_sha256": package_audit.get("campaign_binding_sha256"),
        "qualification_result_sha256": package_audit.get(
            "qualification_result_sha256"
        ),
        "selection_sha256": package_audit.get("selection_sha256"),
        "sealed_before_unblinding": True, "post_unblinding_followup_separate": True,
    }
    write_canonical_json(package / "sealed_scores.json", payload)


def unblind(args: argparse.Namespace) -> None:
    package = Path(args.package_dir).resolve()
    sealed_path = package / "sealed_scores.json"
    sealed = _read_json(sealed_path)
    scores_path = Path(sealed["scores_path"]).resolve()
    if not scores_path.is_relative_to(package) or scores_path != (package / "blind" / "scores.csv"):
        raise RuntimeError("sealed score path escapes or differs from the package score sheet")
    if file_sha256(scores_path) != sealed["scores_sha256"]:
        raise RuntimeError("scores changed after sealing")
    scores = {row["sample_id"]: row for row in csv.DictReader(scores_path.open("r", encoding="utf-8"))}
    key_path = package / "private_key.json"
    if file_sha256(key_path) != sealed.get("private_key_sha256"):
        raise RuntimeError("private key changed after score sealing")
    key = _read_json(key_path)
    rows = [{**record, **scores[record["sample_id"]]} for record in key["clips"]]
    write_canonical_json(package / "unblinded_scores.json", {
        "sealed_scores_sha256": file_sha256(sealed_path),
        "private_key_sha256": file_sha256(key_path),
        "campaign_config_sha256": sealed.get("campaign_config_sha256"),
        "run_schedule_sha256": sealed.get("run_schedule_sha256"),
        "campaign_binding_sha256": sealed.get("campaign_binding_sha256"),
        "qualification_result_sha256": sealed.get("qualification_result_sha256"),
        "selection_sha256": sealed.get("selection_sha256"),
        "single_listener": True, "interpretation": "exploratory_qualitative_judgement",
        "rows": rows,
    })


# ---------------------------------------------------------------------------
# Acc-only triangle listening v2.  The v1 functions above remain unchanged so
# historical 24-clip quality packages can still be opened and audited.


def freeze_triangle_selection(args: argparse.Namespace) -> None:
    manifest_path = Path(args.input_manifest).resolve()
    manifest = _read_json(manifest_path)
    entries = validate_staged_input_manifest(
        manifest, manifest_path=manifest_path, verify_files=True
    )
    songs = sorted({_song(entry) for entry in entries})
    starts = {song: int(args.excerpt_start_beat) for song in songs}
    if args.excerpt_starts_json:
        supplied = _read_json(Path(args.excerpt_starts_json).resolve())
        if set(map(str, supplied)) != set(songs):
            raise ValueError("--excerpt-starts-json must name all five songs exactly")
        starts = {str(song): value for song, value in supplied.items()}
    selection = build_triangle_selection_manifest(
        manifest,
        manifest_path=manifest_path,
        excerpt_starts=starts,
        blind_order_seed=int(args.blind_order_seed),
        verify_files=True,
    )
    digest = write_canonical_json(Path(args.output), selection)
    print(
        json.dumps(
            {
                "path": str(Path(args.output).resolve()),
                "sha256": digest,
                "trials": TRIANGLE_TRIAL_COUNT,
                "presentations": TRIANGLE_PRESENTATION_COUNT,
                "practice_trials": TRIANGLE_PRACTICE_COUNT,
            },
            sort_keys=True,
        )
    )


def derive_triangle_retry(args: argparse.Namespace) -> None:
    selection, authorization_path, authorization = (
        derive_triangle_retry_selection_manifest(
            args.failed_package,
            args.failed_snapshot,
        )
    )
    output = Path(args.output).resolve()
    if output.exists():
        if _read_json(output) != selection:
            raise FileExistsError(
                "existing retry selection differs from the exact derivation"
            )
        sidecar = output.with_name(output.name + ".sha256")
        parts = (
            sidecar.read_text(encoding="ascii").strip().split()
            if sidecar.is_file()
            else []
        )
        if len(parts) != 2 or parts[0] != file_sha256(output) or parts[1] != output.name:
            raise ValueError("existing retry selection checksum sidecar is invalid")
    else:
        write_canonical_json(output, selection)
    manifest_path = Path(selection["input_manifest_path"]).resolve()
    validate_triangle_selection_manifest(
        selection,
        _read_json(manifest_path),
        manifest_path=manifest_path,
        verify_files=True,
    )
    print(
        json.dumps(
            {
                "selection_path": str(output),
                "selection_sha256": file_sha256(output),
                "listening_attempt_id": selection["listening_attempt_id"],
                "effective_blind_order_seed": selection[
                    "effective_blind_order_seed"
                ],
                "effective_blind_order_seed_sha256": selection[
                    "effective_blind_order_seed_sha256"
                ],
                "retry_authorization_path": str(authorization_path),
                "retry_authorization_sha256": file_sha256(authorization_path),
                "previous_attempt_id": authorization["previous_attempt_id"],
                "previous_qc_status": authorization["previous_qc_status"],
                "new_response_ledger": "must_start_empty_in_a_new_package",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _validated_triangle_build_inputs(
    args: argparse.Namespace,
    selection_path: Path,
    selection: dict[str, Any],
) -> tuple[
    dict[str, Any],
    str,
    list[dict[str, Any]],
    Path,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    config_path = Path(args.config).resolve()
    config_sha = file_sha256(config_path)
    if config_sha != args.config_sha256:
        raise RuntimeError("campaign config hash mismatch")
    config = _read_json(config_path)
    validate_campaign_config(config)
    validate_frozen_qualification(config, verify_files=True)
    checkpoint = Path(config["checkpoint"]["path"]).resolve()
    if file_sha256(checkpoint) != config["checkpoint"]["sha256"]:
        raise RuntimeError("checkpoint hash mismatch with frozen campaign config")

    listening_config = config.get("listening")
    if not isinstance(listening_config, Mapping):
        raise ValueError("campaign config lacks listening contract")
    if listening_config.get("schema_version") != TRIANGLE_SELECTION_SCHEMA_VERSION:
        raise ValueError("campaign config does not freeze the triangle-listening v2 schema")
    required_config_values = {
        "trial_count": TRIANGLE_TRIAL_COUNT,
        "practice_count": TRIANGLE_PRACTICE_COUNT,
        "clip_seconds": TRIANGLE_CLIP_SECONDS,
        "default_excerpt_model_beats": [16, 32],
        "render_bpm": TRIANGLE_RENDER_BPM,
        "gain_policy": TRIANGLE_GAIN_POLICY,
        "flexible_sittings": True,
        "per_trial_persistence": True,
        "partial_snapshot_allowed": True,
        "listening_attempt_id": TRIANGLE_LISTENING_ATTEMPT_ID,
    }
    for field, expected in required_config_values.items():
        if listening_config.get(field) != expected:
            raise ValueError(f"campaign listening.{field} is not frozen to {expected!r}")
    if listening_config.get("listening_attempt_policy") != selection.get(
        "listening_attempt_policy"
    ):
        raise ValueError(
            "campaign listening attempt policy differs from frozen selection"
        )
    if listening_config.get("sitting_policy") != selection.get("sitting_policy"):
        raise ValueError("campaign sitting policy differs from frozen selection")
    renderer_record = listening_config.get("renderer_identity")
    if not isinstance(renderer_record, Mapping):
        raise ValueError("campaign listening contract lacks renderer_identity binding")
    renderer_path = Path(str(renderer_record.get("path", ""))).resolve()
    if not renderer_path.is_file() or file_sha256(renderer_path) != renderer_record.get(
        "sha256"
    ):
        raise ValueError("campaign renderer_identity path/hash binding is invalid")
    validate_triangle_renderer_identity(_read_json(renderer_path), verify_files=True)
    selection_sha = file_sha256(selection_path)
    attempt_number = triangle_listening_attempt_number(
        selection.get("listening_attempt_id")
    )
    frozen_base_path = Path(
        str(listening_config.get("selection_manifest_path", ""))
    ).resolve()
    frozen_base_sha = listening_config.get("selection_manifest_sha256")
    if attempt_number == 1:
        if selection_sha != frozen_base_sha:
            raise RuntimeError("triangle selection hash does not match campaign config")
        if frozen_base_path != selection_path:
            raise RuntimeError("triangle selection path does not match campaign config")
    else:
        lineage = selection.get("retry_lineage")
        if not isinstance(lineage, Mapping):
            raise RuntimeError("retry triangle selection lacks immutable lineage")
        if (
            Path(str(lineage.get("base_selection_path", ""))).resolve()
            != frozen_base_path
            or lineage.get("base_selection_sha256") != frozen_base_sha
        ):
            raise RuntimeError(
                "retry selection lineage does not bind the C5 base selection"
            )

    manifest_path = Path(selection["input_manifest_path"]).resolve()
    if manifest_path != Path(config["input_manifest"]["path"]).resolve():
        raise RuntimeError("selection and campaign reference different input manifests")
    if file_sha256(manifest_path) != selection.get("input_manifest_sha256"):
        raise RuntimeError("triangle selection input-manifest hash mismatch")
    if file_sha256(manifest_path) != config["input_manifest"]["sha256"]:
        raise RuntimeError("campaign input-manifest hash mismatch")
    manifest = _read_json(manifest_path)
    entries = validate_staged_input_manifest(
        manifest, manifest_path=manifest_path, verify_files=True
    )
    validate_triangle_selection_manifest(
        selection,
        manifest,
        manifest_path=manifest_path,
        verify_files=True,
    )

    schedule_path = Path(args.schedule).resolve()
    if file_sha256(schedule_path) != args.schedule_sha256:
        raise RuntimeError("run schedule hash mismatch")
    schedule = read_jsonl(schedule_path)
    if schedule != build_run_schedule(manifest, config):
        raise RuntimeError("schedule is not the exact deterministic 160-row formal schedule")
    binding = _validated_campaign_binding(args, config, config_sha)
    campaign_audit_path = Path(args.campaign_audit).resolve()
    if (
        not campaign_audit_path.is_file()
        or file_sha256(campaign_audit_path) != args.campaign_audit_sha256
    ):
        raise ValueError("formal campaign audit path/hash is invalid")
    campaign_audit = _read_json(campaign_audit_path)
    for field in (
        "campaign_config_sha256",
        "run_schedule_sha256",
        "campaign_binding_sha256",
        "qualification_result_sha256",
    ):
        if campaign_audit.get(field) != binding.get(field):
            raise ValueError(f"formal campaign audit {field} binding mismatch")
    if (
        campaign_audit.get("expected") != 160
        or campaign_audit.get("present") != 160
        or campaign_audit.get("content_valid") != 160
        or campaign_audit.get("missing") != 0
        or campaign_audit.get("invalid") != 0
        or campaign_audit.get("extra_run_ids") != []
    ):
        raise ValueError("formal campaign audit is not complete/valid for all 160 runs")
    readiness = campaign_audit.get("listening_source_readiness")
    if (
        not isinstance(readiness, Mapping)
        or readiness.get("ready") is not True
        or readiness.get("expected_unique_sources") != 80
        or readiness.get("ready_sources") != 80
        or readiness.get("not_ready_sources") != 0
    ):
        raise ValueError("formal campaign audit does not prove all 80 listening sources ready")
    selector_fields = (
        "formal_pipeline",
        "source_artifact",
        "presentation",
        "song",
        "condition",
        "perturb_seed",
        "sample_seed",
    )
    expected_selectors = {
        json.dumps(
            {field: source.get(field) for field in selector_fields},
            sort_keys=True,
            separators=(",", ":"),
        )
        for trial in selection["trials"]
        for source in trial["sources"].values()
        if source.get("kind") == "formal"
    }
    audited_selectors = {
        json.dumps(row.get("selector"), sort_keys=True, separators=(",", ":"))
        for row in readiness.get("sources", [])
        if isinstance(row, Mapping) and row.get("ready") is True
    }
    if audited_selectors != expected_selectors:
        raise ValueError("campaign audit listening selectors differ from frozen selection")
    return (
        config,
        config_sha,
        schedule,
        manifest_path,
        entries,
        binding,
        campaign_audit,
    )


def _triangle_find_run(
    schedule: Sequence[Mapping[str, Any]], source: Mapping[str, Any]
) -> dict[str, Any]:
    if source.get("kind") != "formal":
        raise ValueError("only formal selectors map to campaign runs")
    matches = [
        dict(row)
        for row in schedule
        if row.get("pipeline") == source.get("formal_pipeline")
        and row.get("song") == source.get("song")
        and row.get("condition") == source.get("condition")
        and row.get("perturb_seed") == source.get("perturb_seed")
        and row.get("sample_seed") == source.get("sample_seed")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"triangle formal selector matched {len(matches)} runs: {source}")
    return matches[0]


def _triangle_source_key(source: Mapping[str, Any]) -> str:
    from streammuse.experiments.melody_robustness import canonical_sha256

    return canonical_sha256(dict(source))


def _raw_token_payload_provenance(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError("inferences.json must be a list of objects")
    payload = []
    output_payload = []
    for index, row in enumerate(value):
        response = row.get("response_data")
        metadata = response.get("response_metadata") if isinstance(response, Mapping) else None
        if (
            not isinstance(metadata, Mapping)
            or not isinstance(metadata.get("raw_tokens"), list)
            or not isinstance(metadata.get("structural_tokens"), list)
        ):
            raise ValueError(f"inferences.json row {index} lacks raw/structural tokens")
        item = {
            "raw": metadata["raw_tokens"],
            "structural": metadata["structural_tokens"],
        }
        per_request = hashlib.sha256(
            json.dumps(
                item, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if metadata.get("raw_token_digest") != per_request:
            raise ValueError(f"inferences.json row {index} raw_token_digest mismatch")
        payload.append(item)
        accompaniment = response.get("accompaniment")
        if not isinstance(accompaniment, list) or any(
            not isinstance(event, Mapping) for event in accompaniment
        ):
            raise ValueError(
                f"inferences.json row {index} lacks the full accompaniment event payload"
            )
        output_digest = hashlib.sha256(
            json.dumps(
                accompaniment,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if metadata.get("output_event_digest") != output_digest:
            raise ValueError(f"inferences.json row {index} output_event_digest mismatch")
        output_payload.append(accompaniment)
    digest = hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    output_payload_digest = hashlib.sha256(
        json.dumps(
            output_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "inferences_path": str(path),
        "inferences_sha256": file_sha256(path),
        "inference_count": len(value),
        "raw_token_payload_sha256": digest,
        "output_event_payload_sha256": output_payload_digest,
    }


def _validate_triangle_control_report(
    path: Path,
    expected_sha256: str,
    *,
    selection: Mapping[str, Any],
    selection_sha256: str,
    campaign_config_sha256: str,
    verify_files: bool = True,
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    """Validate the analyzer/package side of the T7.7 three-way closure."""

    path = path.resolve()
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("control report path/hash binding is invalid")
    report = _read_json(path)
    if report.get("campaign_config_sha256") != campaign_config_sha256:
        raise ValueError("control report campaign binding differs from package build")
    known = report.get("listening_known_different")
    if not isinstance(known, Mapping):
        raise ValueError("control report lacks listening_known_different")
    if known.get("selection_sha256") != selection_sha256:
        raise ValueError("control report selection hash differs from frozen selection")
    selected = [
        trial
        for trial in selection.get("trials", [])
        if isinstance(trial, Mapping)
        and trial.get("block") == "known_different_control"
    ]
    records = known.get("controls")
    if (
        len(selected) != 6
        or known.get("expected_count") != 6
        or known.get("actual_count") != 6
        or known.get("all_recipe_bound") is not True
        or known.get("all_source_selectors_bound") is not True
        or known.get("all_not_identical") is not True
        or not isinstance(records, list)
        or len(records) != 6
    ):
        raise ValueError("control report known-different count/status contract failed")
    by_semantic: dict[str, Mapping[str, Any]] = {}
    selected_by_semantic = {str(row["semantic_id"]): row for row in selected}
    for raw_record in records:
        if not isinstance(raw_record, Mapping):
            raise ValueError("control report contains a malformed known-different record")
        semantic_id = str(raw_record.get("semantic_id", ""))
        trial = selected_by_semantic.get(semantic_id)
        if trial is None or semantic_id in by_semantic:
            raise ValueError("control report semantic IDs differ from frozen selection")
        sources = trial["sources"]
        formal = sources["a"]
        synthetic = sources["b"]
        recipe = synthetic.get("recipe")
        if (
            raw_record.get("campaign_config_sha256") != campaign_config_sha256
            or raw_record.get("question_id") != trial.get("question_id")
            or raw_record.get("selection_source_a_sha256")
            != canonical_sha256(dict(formal))
            or raw_record.get("selection_source_b_sha256")
            != canonical_sha256(dict(synthetic))
            or not isinstance(recipe, Mapping)
            or raw_record.get("selection_recipe_sha256")
            != canonical_sha256(dict(recipe))
            or raw_record.get("synthetic_velocity") != 96
            or raw_record.get("not_identical") is not True
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(raw_record.get("formal_comparator_note_events_sha256", "")),
            )
            is None
        ):
            raise ValueError(f"{semantic_id}: analyzer selector/recipe binding differs")
        for label, path_field, hash_field in (
            ("formal source", "formal_source_path", "formal_source_sha256"),
            (
                "formal comparator",
                "formal_comparator_excerpt_path",
                "formal_comparator_excerpt_sha256",
            ),
            ("synthetic excerpt", "synthetic_excerpt_path", "synthetic_excerpt_sha256"),
        ):
            artifact_path = Path(str(raw_record.get(path_field, ""))).resolve()
            digest = raw_record.get(hash_field)
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"{semantic_id}: {label} digest is malformed")
            if verify_files and (
                not artifact_path.is_file() or file_sha256(artifact_path) != digest
            ):
                raise ValueError(f"{semantic_id}: {label} path/hash mismatch")
        by_semantic[semantic_id] = raw_record
    if set(by_semantic) != set(selected_by_semantic):
        raise ValueError("control report omits a frozen known-different semantic ID")
    return report, by_semantic


def _triangle_control_base_selection(
    current_path: Path, current: Mapping[str, Any]
) -> tuple[Path, dict[str, Any], str]:
    """Resolve C5 for attempt 1 and the immutable C5 base for every retry."""

    current_path = current_path.resolve()
    attempt_number = triangle_listening_attempt_number(
        current.get("listening_attempt_id")
    )
    if attempt_number == 1:
        return current_path, dict(current), file_sha256(current_path)
    lineage = current.get("retry_lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("retry selection lacks immutable base-selection lineage")
    base_path = Path(str(lineage.get("base_selection_path", ""))).resolve()
    base_sha = str(lineage.get("base_selection_sha256", ""))
    if not base_path.is_file() or file_sha256(base_path) != base_sha:
        raise ValueError("retry control base selection path/hash mismatch")
    base = _read_json(base_path)
    manifest_path = Path(str(base["input_manifest_path"])).resolve()
    manifest = _read_json(manifest_path)
    validate_triangle_selection_manifest(
        base,
        manifest,
        manifest_path=manifest_path,
        verify_files=True,
    )
    base_known = {
        str(row["semantic_id"]): {
            "sources": row["sources"],
            "excerpt": row["excerpt"],
        }
        for row in base["trials"]
        if row.get("block") == "known_different_control"
    }
    current_known = {
        str(row["semantic_id"]): {
            "sources": row["sources"],
            "excerpt": row["excerpt"],
        }
        for row in current.get("trials", [])
        if row.get("block") == "known_different_control"
    }
    if current_known != base_known:
        raise ValueError("retry changed known-different semantics from C5 base")
    return base_path, base, base_sha


def _validate_known_different_material(
    semantic_id: str,
    formal: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Close analyzer -> canonical source -> blinded pair hashes for one control."""

    expected = {
        "formal_source_sha256": formal.get("source_sha256"),
        "formal_comparator_excerpt_sha256": formal.get("excerpt_midi_sha256"),
        "formal_comparator_note_events_sha256": formal.get(
            "excerpt_note_event_sha256"
        ),
        "synthetic_excerpt_sha256": synthetic.get("excerpt_midi_sha256"),
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{semantic_id}: package {field} differs from analyzer control")
    if (
        record.get("formal_run_id") != formal.get("run_id")
        or Path(str(record.get("formal_source_path", ""))).resolve()
        != Path(str(formal.get("source_path", ""))).resolve()
    ):
        raise ValueError(f"{semantic_id}: package formal source provenance differs")


def _fluidsynth_environment(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    env = dict(os.environ)
    roots: list[Path] = []
    if args.fluidsynth_root:
        root = Path(args.fluidsynth_root).resolve()
        # ``scripts/local_fluidsynth.sh`` intentionally refuses to guess its
        # extracted rootfs.  Keep the CLI's ``--fluidsynth-root`` option as the
        # single source of truth for both the wrapper and the dynamic linker so
        # the documented durable local toolchain works without an ambient
        # shell export.
        env["STREAMMUSE_FLUIDSYNTH_ROOT"] = str(root)
        roots.extend(
            [
                root / "usr" / "lib" / "x86_64-linux-gnu",
                root / "usr" / "lib" / "x86_64-linux-gnu" / "pulseaudio",
            ]
        )
    roots.extend(Path(value).resolve() for value in (args.fluidsynth_lib_dir or []))
    existing = env.get("LD_LIBRARY_PATH")
    library_path = [str(path) for path in roots if path.is_dir()]
    if existing:
        library_path.append(existing)
    if library_path:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(library_path)
    binary = Path(args.fluidsynth)
    resolved_binary = (
        binary.resolve()
        if binary.is_absolute() or binary.parent != Path(".")
        else Path(shutil.which(args.fluidsynth) or args.fluidsynth).resolve()
    )
    if not resolved_binary.is_file():
        raise FileNotFoundError(f"FluidSynth binary does not exist: {resolved_binary}")
    libraries = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.so*")):
            if path.is_file():
                libraries.append(
                    {
                        "path": str(path),
                        "size": path.stat().st_size,
                        "sha256": file_sha256(path),
                    }
                )
    provenance = {
        "binary_path": str(resolved_binary),
        "binary_sha256": file_sha256(resolved_binary),
        "ld_library_path": env.get("LD_LIBRARY_PATH"),
        "library_files": libraries,
    }
    return env, provenance


def _triangle_renderer_identity(
    args: argparse.Namespace, soundfont: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    env, synth = _fluidsynth_environment(args)
    version = subprocess.run(
        [synth["binary_path"], "--version"],
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    version_text = (version.stdout or version.stderr).strip()
    if version.returncode:
        raise RuntimeError(f"cannot identify FluidSynth: {version_text}")
    identity = {
            "schema_version": (
                "streammuse.melody_robustness.listening_triangle_renderer.v2"
            ),
            "fluidsynth": {**synth, "version": version_text},
            "soundfont": {
                "path": str(soundfont),
                "size": soundfont.stat().st_size,
                "sha256": file_sha256(soundfont),
            },
            "midi_program": 0,
            "midi_bank": 0,
            "render_bpm": TRIANGLE_RENDER_BPM,
            "sample_rate": TRIANGLE_RENDER_SAMPLE_RATE,
            "bit_depth": 16,
            "synth_gain": TRIANGLE_SYNTH_GAIN,
            "gain_policy": TRIANGLE_GAIN_POLICY,
            "true_peak_limit_dbtp": TRUE_PEAK_LIMIT_DBTP,
            "true_peak_implementation": TRUE_PEAK_IMPLEMENTATION,
            "command_template": [
                "{fluidsynth_binary}",
                "-ni",
                "{soundfont}",
                "{input_midi}",
                "-F",
                "{output_wav}",
                "-r",
                str(TRIANGLE_RENDER_SAMPLE_RATE),
                "-g",
                str(TRIANGLE_SYNTH_GAIN),
            ],
        }
    validate_triangle_renderer_identity(identity, verify_files=True)
    return identity, env


def freeze_triangle_renderer(args: argparse.Namespace) -> None:
    soundfont = Path(args.soundfont).resolve()
    if not soundfont.is_file():
        raise FileNotFoundError(f"soundfont does not exist: {soundfont}")
    identity, _env = _triangle_renderer_identity(args, soundfont)
    validate_triangle_renderer_identity(identity, verify_files=True)
    digest = write_canonical_json(Path(args.output), identity)
    print(
        json.dumps(
            {"path": str(Path(args.output).resolve()), "sha256": digest},
            indent=2,
            sort_keys=True,
        )
    )


def _render_with_env(
    midi_path: Path,
    wav_path: Path,
    *,
    soundfont: Path,
    fluidsynth: str,
    sample_rate: int,
    gain: float,
    env: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        fluidsynth,
        "-ni",
        str(soundfont),
        str(midi_path),
        "-F",
        str(wav_path),
        "-r",
        str(sample_rate),
        "-g",
        str(gain),
    ]
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=dict(env),
    )
    if result.returncode:
        raise RuntimeError(f"fluidsynth failed ({result.returncode}): {result.stderr}")
    return {"command": command, "stdout": result.stdout, "stderr": result.stderr}


def _read_rendered_pcm(path: Path, *, seconds: int) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if width != 2 or sample_rate != TRIANGLE_RENDER_SAMPLE_RATE or channels not in {1, 2}:
        raise RuntimeError(
            "triangle render must be mono/stereo 16-bit/44100Hz before canonicalization"
        )
    raw = np.frombuffer(frames, dtype="<i2")
    if raw.size % channels:
        raise RuntimeError("triangle render contains incomplete channel frames")
    samples = raw.astype(np.float64).reshape(-1, channels) / 32768.0
    target = TRIANGLE_RENDER_SAMPLE_RATE * seconds
    if samples.shape[0] < target:
        samples = np.pad(samples, ((0, target - samples.shape[0]), (0, 0)))
    else:
        samples = samples[:target]
    return samples, channels


def _write_common_protected_wavs(
    rows: Sequence[tuple[Path, np.ndarray]],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    channels = {samples.shape[1] for _path, samples in rows}
    if len(channels) != 1:
        raise RuntimeError("matched triangle pair renders have different channel counts")
    peaks_before = [_measure_true_peak(samples) for _path, samples in rows]
    limit = float(10.0 ** (TRUE_PEAK_LIMIT_DBTP / 20.0))
    maximum = max(peaks_before, default=0.0)
    scale = min(1.0, limit / maximum) if maximum > 0 else 1.0
    quantized = [_quantize_pcm16(samples * scale) for _path, samples in rows]
    peaks_after = [
        _measure_true_peak(samples.astype(np.float64) / 32768.0) for samples in quantized
    ]
    max_after = max(peaks_after, default=0.0)
    if max_after > limit:
        scale *= (limit / max_after) * (1.0 - 1e-9)
        quantized = [_quantize_pcm16(samples * scale) for _path, samples in rows]
        peaks_after = [
            _measure_true_peak(samples.astype(np.float64) / 32768.0)
            for samples in quantized
        ]
    if any(peak > limit + 1e-9 for peak in peaks_after):
        raise RuntimeError("common pair true-peak protection failed")
    metadata = []
    for (path, original), pcm, before, after in zip(
        rows, quantized, peaks_before, peaks_after, strict=True
    ):
        with wave.open(str(path), "wb") as destination:
            destination.setnchannels(original.shape[1])
            destination.setsampwidth(2)
            destination.setframerate(TRIANGLE_RENDER_SAMPLE_RATE)
            destination.writeframes(pcm.reshape(-1).tobytes())
        metadata.append(
            {
                "channels": original.shape[1],
                "sample_width_bits": 16,
                "sample_rate": TRIANGLE_RENDER_SAMPLE_RATE,
                "frames": TRIANGLE_RENDER_SAMPLE_RATE * TRIANGLE_CLIP_SECONDS,
                "seconds": TRIANGLE_CLIP_SECONDS,
                "true_peak_linear_before": before,
                "true_peak_dbtp_before": _dbtp(before),
                "true_peak_linear_after": after,
                "true_peak_dbtp_after": _dbtp(after),
                "true_peak_limit_dbtp": TRUE_PEAK_LIMIT_DBTP,
                "true_peak_protection_applied": scale < 1.0,
                "common_pair_gain_scale": scale,
                "peak_measurement": TRUE_PEAK_IMPLEMENTATION,
                "true_peak_oversample_factor": TRUE_PEAK_OVERSAMPLE,
                "true_peak_verified": True,
                "silent": not bool(np.any(pcm)),
            }
        )
    return metadata


def _triangle_player_html(public_manifest: Mapping[str, Any], selection_sha: str) -> str:
    # Only opaque IDs and relative audio paths are embedded.  Browser
    # localStorage provides per-question durability on a downloaded package;
    # the exported JSON is imported into the server-side hash-chain ledger.
    manifest_json = json.dumps(public_manifest, ensure_ascii=False).replace("</", "<\\/")
    storage_key = f"streammuse-triangle-{selection_sha}"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>StreamMUSE accompaniment triangle listening</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem}}
button,label,select,input{{font-size:1rem;margin:.35rem}} .clips{{display:flex;gap:.5rem}}
.saved{{color:#176b2c}} .muted{{color:#666}} textarea{{width:100%;min-height:4rem}}
</style></head><body>
<h1>伴奏盲听</h1><p id="prompt"></p><p id="progress"></p>
<div id="trial"><div class="clips"></div><fieldset><legend>哪一段听起来不同？</legend>
<label><input type="radio" name="choice" value="1">1</label>
<label><input type="radio" name="choice" value="2">2</label>
<label><input type="radio" name="choice" value="3">3</label>
<label><input type="radio" name="choice" value="no_difference">听不出区别</label></fieldset>
<label>确信程度 <select id="confidence"><option>1</option><option>2</option><option>3</option><option selected>4</option><option>5</option></select></label>
<div id="tags"></div><textarea id="note" placeholder="可选备注"></textarea>
<button id="save">保存本题</button><span id="saved" class="saved"></span></div>
<fieldset><legend>本次 listening sitting</legend>
<label>设备 <input id="device" placeholder="例如：Sony MDR-7506"></label>
<label>环境 <input id="environment" placeholder="例如：安静办公室"></label>
<button id="startSitting">开始新的 sitting</button><button id="endSitting">结束当前 sitting</button>
<span id="sittingStatus" class="muted"></span></fieldset>
<p><button id="export">导出已答 responses.json</button></p>
<p class="muted">正式题提交前必须先记录 sitting 的设备和环境。每题保存后立即持久化；关闭后重新打开会从下一题继续。</p>
<script>
const manifest={manifest_json}; const key={json.dumps(storage_key)};
const tags=['pitch_harmony','rhythm_timing','density','texture_register','silence_coverage','other'];
let state=JSON.parse(localStorage.getItem(key)||'null')||{{practice_index:0,responses:[],sitting_id:null,sitting_active:false,sitting_events:[]}};
if(state.practice_index===undefined) state.practice_index=0;
if(state.sitting_events===undefined) state.sitting_events=[];if(state.sitting_active===undefined) state.sitting_active=false;
const serverMode=location.protocol==='http:'||location.protocol==='https:';let serverProgress=null;
let started=Date.now(), plays=[0,0,0];
document.getElementById('prompt').textContent=manifest.question_prompt;
document.getElementById('tags').innerHTML=tags.map(t=>`<label><input type="checkbox" value="${{t}}">${{t}}</label>`).join('');
function scoredIndex(){{return serverMode&&serverProgress?serverProgress.answered_count:state.responses.length;}}
function current(){{if(state.practice_index<manifest.practice_trials.length)return {{practice:true,q:manifest.practice_trials[state.practice_index]}};
 return {{practice:false,q:manifest.trials[scoredIndex()]}};}}
function stopAllAudio(){{document.querySelectorAll('.clips audio').forEach(a=>{{a.pause();a.currentTime=0;}});}}
function render(){{const i=scoredIndex();
 stopAllAudio();
 if(state.practice_index<manifest.practice_trials.length)document.getElementById('progress').textContent=`练习 ${{state.practice_index+1}} / ${{manifest.practice_trials.length}}（不计分）`;
 else document.getElementById('progress').textContent=`已答 ${{i}} / ${{manifest.trials.length}}`;
 if(state.practice_index>=manifest.practice_trials.length&&i>=manifest.trials.length){{document.getElementById('trial').innerHTML='<p>95 题已完成，请导出结果。</p>';return;}}
 const q=current().q; plays=[0,0,0]; started=Date.now(); document.getElementById('saved').textContent='';
 document.querySelectorAll('input[name=choice]').forEach(x=>x.checked=false);document.getElementById('note').value='';
 document.querySelectorAll('#tags input').forEach(x=>x.checked=false);
 document.querySelector('.clips').innerHTML=q.clips.map((src,j)=>`<button data-i="${{j}}">播放 ${{j+1}}</button><audio preload="auto" src="${{src}}"></audio>`).join('');
 document.querySelectorAll('.clips button').forEach((b,j)=>b.onclick=()=>{{stopAllAudio();const a=document.querySelectorAll('.clips audio')[j];a.play();plays[j]++;}});}}
document.getElementById('save').onclick=async()=>{{const item=current(),q=item.q;const c=document.querySelector('input[name=choice]:checked');
 if(!c){{alert('请选择 1/2/3 或听不出区别');return}} if(plays.some(x=>x<1)){{alert('请至少播放每段一次');return}}
 if(item.practice){{const ok=c.value===q.correct_choice;state.practice_index++;localStorage.setItem(key,JSON.stringify(state));alert(ok?'练习回答正确':'练习答案不正确，请确认你理解三段中找不同的规则');render();return;}}
 if(!state.sitting_active||!state.sitting_id){{alert('请先记录设备和环境并开始一个 sitting');return}}
 const row={{trial_id:q.question_id,odd_choice:c.value,confidence_1_to_5:Number(document.getElementById('confidence').value),
 difference_tags:[...document.querySelectorAll('#tags input:checked')].map(x=>x.value),note:document.getElementById('note').value,
 play_counts:plays,response_time_ms:Date.now()-started,sitting_id:state.sitting_id,submitted_at:new Date().toISOString()}};
 if(serverMode){{try{{const response=await fetch('/api/response',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(row)}});
   const result=await response.json();if(!response.ok)throw new Error(result.error||'server rejected response');serverProgress=result.progress;
   document.getElementById('saved').textContent='已写入服务器并校验';render();}}catch(error){{alert('保存失败，本题未推进：'+error);}}return;}}
 state.responses.push(row);localStorage.setItem(key,JSON.stringify(state));document.getElementById('saved').textContent='已保存到本机';render();}};
async function sittingPost(route,payload){{if(!serverMode)return null;const response=await fetch(route,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(payload)}});const result=await response.json();if(!response.ok)throw new Error(result.error||'sitting request failed');serverProgress=result.progress;return result;}}
document.getElementById('startSitting').onclick=async()=>{{if(state.sitting_active){{alert('请先结束当前 sitting');return}}const device=document.getElementById('device').value.trim(),environment=document.getElementById('environment').value.trim();if(!device||!environment){{alert('请填写设备和环境');return}}const id='sitting-'+Date.now();const event={{sitting_id:id,device:device,environment:environment,note:'',recorded_at:new Date().toISOString()}};try{{await sittingPost('/api/sitting/start',event);state.sitting_id=id;state.sitting_active=true;state.sitting_events.push({{event:'start',...event}});localStorage.setItem(key,JSON.stringify(state));document.getElementById('sittingStatus').textContent='当前 '+id;}}catch(error){{alert('开始 sitting 失败：'+error)}}}};
document.getElementById('endSitting').onclick=async()=>{{if(!state.sitting_active||!state.sitting_id){{alert('当前没有 active sitting');return}}const raw=prompt('可选：记录异常，多个异常用逗号分隔','')||'';const anomalies=raw.split(',').map(x=>x.trim()).filter(Boolean);const event={{sitting_id:state.sitting_id,anomalies:anomalies,note:'',recorded_at:new Date().toISOString()}};try{{await sittingPost('/api/sitting/end',event);state.sitting_events.push({{event:'end',...event}});state.sitting_active=false;localStorage.setItem(key,JSON.stringify(state));document.getElementById('sittingStatus').textContent='sitting 已结束';}}catch(error){{alert('结束 sitting 失败：'+error)}}}};
document.getElementById('export').onclick=()=>{{const blob=new Blob([JSON.stringify({{selection_sha256:{json.dumps(selection_sha)},sitting_events:state.sitting_events,responses:state.responses}},null,2)],{{type:'application/json'}});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='triangle-responses.json';a.click();URL.revokeObjectURL(a.href);}};
async function init(){{if(serverMode){{const response=await fetch('/api/progress');if(!response.ok)throw new Error('无法读取服务器进度');serverProgress=await response.json();if(state.sitting_id&&serverProgress.sittings&&serverProgress.sittings[state.sitting_id])state.sitting_active=serverProgress.sittings[state.sitting_id].active;}}
 document.getElementById('sittingStatus').textContent=state.sitting_active?'当前 '+state.sitting_id:'尚未开始 sitting';
 render();}}init().catch(error=>{{document.getElementById('trial').innerHTML='<p>启动失败：'+String(error)+'</p>';}});
</script></body></html>"""


def build_triangle_package(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection).resolve()
    selection = _read_json(selection_path)
    if selection.get("schema_version") != TRIANGLE_SELECTION_SCHEMA_VERSION:
        raise ValueError("build-triangle requires listening_triangle_selection.v2")
    if selection.get("frozen_before_formal") is not True:
        raise ValueError("triangle selection must be frozen before formal output")
    (
        config,
        config_sha,
        schedule,
        manifest_path,
        _entries,
        campaign_binding,
        campaign_audit,
    ) = _validated_triangle_build_inputs(args, selection_path, selection)
    (
        control_base_selection_path,
        control_base_selection,
        control_base_selection_sha256,
    ) = _triangle_control_base_selection(selection_path, selection)
    control_report_path = Path(args.control_report).resolve()
    control_report_sha256 = str(args.control_report_sha256)
    _control_report, control_records = _validate_triangle_control_report(
        control_report_path,
        control_report_sha256,
        selection=control_base_selection,
        selection_sha256=control_base_selection_sha256,
        campaign_config_sha256=config_sha,
        verify_files=True,
    )
    output_root = Path(args.output_root).resolve()
    package = Path(args.package_dir).resolve()
    if package.exists():
        raise FileExistsError(f"triangle package already exists: {package}")
    if args.sample_rate != TRIANGLE_RENDER_SAMPLE_RATE:
        raise ValueError("formal triangle render requires 44100 Hz")
    if args.gain != TRIANGLE_SYNTH_GAIN:
        raise ValueError("formal triangle render requires synth gain 0.5")
    soundfont = Path(args.soundfont).resolve() if args.soundfont else None
    if not args.midi_only and (soundfont is None or not soundfont.is_file()):
        raise FileNotFoundError("--soundfont is required for an accepted triangle WAV package")

    synth_env: dict[str, str] | None = None
    synth_provenance: dict[str, Any] | None = None
    synth_version: str | None = None
    renderer_record = config["listening"]["renderer_identity"]
    renderer_identity_path = Path(renderer_record["path"]).resolve()
    frozen_renderer_identity = _read_json(renderer_identity_path)
    validate_triangle_renderer_identity(frozen_renderer_identity, verify_files=True)
    if not args.midi_only:
        current_renderer_identity, synth_env = _triangle_renderer_identity(
            args, soundfont  # type: ignore[arg-type]
        )
        if current_renderer_identity != frozen_renderer_identity:
            raise RuntimeError(
                "current FluidSynth/libs/soundfont/program/render identity differs "
                "from the pre-formal C5 binding"
            )
        synth_provenance = dict(current_renderer_identity["fluidsynth"])
        synth_version = str(synth_provenance["version"])

    private = package / "private"
    canonical_root = private / "canonical_sources"
    pair_root = private / "matched_pairs"
    blind_trials = package / "blind" / "trials"
    blind_practice = package / "blind" / "practice"
    canonical_root.mkdir(parents=True)
    pair_root.mkdir(parents=True)
    blind_trials.mkdir(parents=True)
    blind_practice.mkdir(parents=True)

    # Associate every unique source selector with its frozen song excerpt.
    source_specs: dict[str, tuple[dict[str, Any], Mapping[str, Any] | None]] = {}
    for trial in selection["trials"]:
        for source in trial["sources"].values():
            key = _triangle_source_key(source)
            prior = source_specs.get(key)
            if prior is not None and prior[1] != trial["excerpt"]:
                raise ValueError("the same triangle source was assigned different excerpts")
            source_specs[key] = (dict(source), trial["excerpt"])
    for practice in selection["practice_trials"]:
        for source in practice["sources"].values():
            source_specs.setdefault(_triangle_source_key(source), (dict(source), None))

    attempt_cache: dict[str, tuple[Path, dict[str, Any], set[Path]]] = {}
    source_material: dict[str, dict[str, Any]] = {}
    for index, (key, (source, excerpt)) in enumerate(sorted(source_specs.items()), start=1):
        source_id = f"CS{index:03d}"
        source_dir = canonical_root / source_id
        source_dir.mkdir()
        excerpt_midi = source_dir / "excerpt.mid"
        selector_kind = str(source["kind"])
        provenance: dict[str, Any]
        if selector_kind == "formal":
            if excerpt is None:
                raise ValueError("formal triangle source lacks frozen excerpt")
            row = _triangle_find_run(schedule, source)
            run_id = str(row["run_id"])
            if run_id not in attempt_cache:
                attempt_cache[run_id] = verify_attempt_verdict(
                    output_root / "runs" / run_id,
                    row,
                    campaign_binding,
                    require_content_valid=True,
                )
            attempt, verdict, indexed = attempt_cache[run_id]
            theoretical = _single(attempt, "theoretical_model.mid").resolve()
            if theoretical not in indexed:
                raise RuntimeError(f"theoretical_model.mid is not indexed: {run_id}")
            inferences = _single(attempt, "inferences.json").resolve()
            if inferences not in indexed:
                raise RuntimeError(f"inferences.json is not indexed: {run_id}")
            raw_token_provenance = _raw_token_payload_provenance(inferences)
            start_tick = int(excerpt["start_model_tick"])
            end_tick = int(excerpt["end_model_tick"])
            if end_tick - start_tick != TRIANGLE_CLIP_BEATS * 4:
                raise ValueError("formal triangle excerpt has the wrong logical duration")
            roll, note_events = _rebuild_formal_excerpt_midi(
                theoretical,
                excerpt_midi,
                start_model_tick=start_tick,
                end_model_tick=end_tick,
            )
            provenance = {
                **source,
                "run_id": run_id,
                "attempt_id": attempt.name,
                "verdict_path": str(attempt / "verdict.json"),
                "verdict_sha256": file_sha256(attempt / "verdict.json"),
                "content_valid": verdict["content_valid"],
                "operational_valid": verdict["operational_valid"],
                "source_path": str(theoretical),
                "source_sha256": file_sha256(theoretical),
                **raw_token_provenance,
                "excerpt": dict(excerpt),
            }
        elif selector_kind in {"synthetic_control", "practice_tone"}:
            roll, control_velocity = build_triangle_control_roll(source)
            write_roll_midi(
                roll,
                excerpt_midi,
                bpm=TRIANGLE_RENDER_BPM,
                velocity=control_velocity,
            )
            note_events = _canonical_midi_contract(
                excerpt_midi,
                expected_end_model_tick=TRIANGLE_CLIP_BEATS * 4,
            )
            provenance = {
                **source,
                "run_id": None,
                "attempt_id": None,
                "content_valid": True,
                "operational_valid": True,
                "source_path": None,
                "source_sha256": None,
                "inferences_path": None,
                "inferences_sha256": None,
                "inference_count": None,
                "raw_token_payload_sha256": None,
                "output_event_payload_sha256": None,
                "excerpt": dict(excerpt) if excerpt is not None else None,
            }
        else:
            raise ValueError(f"unknown triangle source kind: {selector_kind}")
        provenance.update(
            {
                "canonical_source_id": source_id,
                "source_empty": not bool(roll.sustain or roll.onsets),
                "excerpt_note_onset_count": len(roll.onsets),
                "excerpt_sustain_cell_count": len(roll.sustain),
                "excerpt_coverage_ratio": len({tick for tick, _pitch in roll.sustain})
                / (TRIANGLE_CLIP_BEATS * 4),
                "excerpt_note_event_sha256": _note_event_sha256(note_events),
                "excerpt_midi_path": str(excerpt_midi),
                "excerpt_midi_sha256": file_sha256(excerpt_midi),
            }
        )
        samples = None
        if not args.midi_only:
            raw_wav = source_dir / "canonical.wav"
            render_record = _render_with_env(
                excerpt_midi,
                raw_wav,
                soundfont=soundfont,  # type: ignore[arg-type]
                fluidsynth=str(synth_provenance["binary_path"]),  # type: ignore[index]
                sample_rate=args.sample_rate,
                gain=args.gain,
                env=synth_env or {},
            )
            samples, _channels = _read_rendered_pcm(
                raw_wav, seconds=TRIANGLE_CLIP_SECONDS
            )
            canonical_audio = _write_common_protected_wavs([(raw_wav, samples)])[0]
            if not provenance["source_empty"] and canonical_audio["silent"]:
                raise RuntimeError(f"non-empty source rendered to silence: {source_id}")
            provenance.update(
                {
                    "canonical_wav_path": str(raw_wav),
                    "canonical_wav_sha256": file_sha256(raw_wav),
                    "canonical_audio": canonical_audio,
                    "render_command": render_record,
                }
            )
        else:
            provenance.update(
                {
                    "canonical_wav_path": None,
                    "canonical_wav_sha256": None,
                    "canonical_audio": None,
                    "render_command": None,
                }
            )
        source_material[key] = {
            "selector": source,
            "roll": roll,
            "note_events": note_events,
            "samples": samples,
            "provenance": provenance,
        }

    for trial in selection["trials"]:
        if trial.get("block") != "known_different_control":
            continue
        semantic_id = str(trial["semantic_id"])
        material_a = source_material[_triangle_source_key(trial["sources"]["a"])]
        material_b = source_material[_triangle_source_key(trial["sources"]["b"])]
        _validate_known_different_material(
            semantic_id,
            material_a["provenance"],
            material_b["provenance"],
            control_records[semantic_id],
        )

    pair_by_semantic: dict[str, dict[str, Any]] = {}
    pair_counter = 0

    def make_pair(
        semantic_id: str,
        source_a: Mapping[str, Any],
        source_b: Mapping[str, Any],
    ) -> dict[str, Any]:
        nonlocal pair_counter
        pair_counter += 1
        pair_id = f"C{pair_counter:03d}"
        destination = pair_root / pair_id
        destination.mkdir()
        material_a = source_material[_triangle_source_key(source_a)]
        material_b = source_material[_triangle_source_key(source_b)]
        midi_a = destination / "a.mid"
        midi_b = destination / "b.mid"
        shutil.copyfile(material_a["provenance"]["excerpt_midi_path"], midi_a)
        shutil.copyfile(material_b["provenance"]["excerpt_midi_path"], midi_b)
        audio_a = audio_b = None
        wav_a = wav_b = None
        if not args.midi_only:
            wav_a = destination / "a.wav"
            wav_b = destination / "b.wav"
            audio_a, audio_b = _write_common_protected_wavs(
                [
                    (wav_a, material_a["samples"]),
                    (wav_b, material_b["samples"]),
                ]
            )
            if not material_a["provenance"]["source_empty"] and audio_a["silent"]:
                raise RuntimeError(f"pair {pair_id} source A unexpectedly rendered silent")
            if not material_b["provenance"]["source_empty"] and audio_b["silent"]:
                raise RuntimeError(f"pair {pair_id} source B unexpectedly rendered silent")
        # Velocity is audible and therefore part of canonical note identity.  A
        # roll-only comparison would incorrectly collapse dynamically distinct
        # excerpts.  The frozen scoring contract is broader still: post-gain,
        # final presentation WAV identity also forces a zero triangle hit.
        objective_identity = _triangle_objective_identity(
            material_a["note_events"],
            material_b["note_events"],
            wav_a=wav_a,
            wav_b=wav_b,
        )
        if source_b.get("kind") == "synthetic_control" and objective_identity:
            raise RuntimeError(
                "known-different control equals its formal comparator in notes or final audio"
            )
        coverage_a = float(material_a["provenance"]["excerpt_coverage_ratio"])
        coverage_b = float(material_b["provenance"]["excerpt_coverage_ratio"])
        coverage_collapse = (
            min(coverage_a, coverage_b) <= TRIANGLE_COVERAGE_COLLAPSE_MAX_RATIO
            and max(coverage_a, coverage_b) >= TRIANGLE_COVERAGE_REFERENCE_MIN_RATIO
        )
        pair = {
            "pair_id": pair_id,
            "semantic_id": semantic_id,
            "source_a": dict(material_a["provenance"]),
            "source_b": dict(material_b["provenance"]),
            "midi_a": str(midi_a),
            "midi_b": str(midi_b),
            "midi_a_sha256": file_sha256(midi_a),
            "midi_b_sha256": file_sha256(midi_b),
            "wav_a": str(wav_a) if wav_a else None,
            "wav_b": str(wav_b) if wav_b else None,
            "wav_a_sha256": file_sha256(wav_a) if wav_a else None,
            "wav_b_sha256": file_sha256(wav_b) if wav_b else None,
            "audio_a": audio_a,
            "audio_b": audio_b,
            "objective_identity": objective_identity,
            "coverage_driven": bool(material_a["provenance"]["source_empty"])
            != bool(material_b["provenance"]["source_empty"])
            or coverage_collapse,
            "coverage_collapse": coverage_collapse,
            "coverage_ratios": {"a": coverage_a, "b": coverage_b},
        }
        pair_by_semantic[semantic_id] = pair
        return pair

    private_rows: list[dict[str, Any]] = []
    render_rows: list[dict[str, Any]] = []
    public_trials: list[dict[str, Any]] = []
    for trial in selection["trials"]:
        semantic_id = str(trial["semantic_id"])
        if trial["block"] == "exact_repeat":
            pair = pair_by_semantic[str(trial["repeat_of"])]
        else:
            pair = make_pair(
                semantic_id,
                trial["sources"]["a"],
                trial["sources"]["b"],
            )
        question_id = str(trial["question_id"])
        question_dir = blind_trials / question_id
        question_dir.mkdir()
        presentations = []
        public_clip_paths = []
        for position, label in enumerate(str(trial["presentation_pattern"]), start=1):
            side = label.lower()
            midi_source = Path(pair[f"midi_{side}"])
            midi_path = question_dir / f"clip_{position}.mid"
            shutil.copyfile(midi_source, midi_path)
            wav_path = None
            audio = pair[f"audio_{side}"]
            if not args.midi_only:
                wav_source = Path(pair[f"wav_{side}"])
                wav_path = question_dir / f"clip_{position}.wav"
                shutil.copyfile(wav_source, wav_path)
                public_clip_paths.append(
                    str(wav_path.relative_to(package / "blind"))
                )
            else:
                public_clip_paths.append(
                    str(midi_path.relative_to(package / "blind"))
                )
            presentations.append(
                {
                    "position": position,
                    "midi": str(midi_path.relative_to(package)),
                    "midi_sha256": file_sha256(midi_path),
                    "wav": str(wav_path.relative_to(package)) if wav_path else None,
                    "wav_sha256": file_sha256(wav_path) if wav_path else None,
                    "audio": audio,
                }
            )
        private_rows.append(
            {
                "question_id": question_id,
                "semantic_id": semantic_id,
                "block": trial["block"],
                "condition": trial["condition"],
                "repeat_of": trial.get("repeat_of"),
                "repeat_distance": trial.get("repeat_distance"),
                "presentation_pattern": trial["presentation_pattern"],
                "correct_choice": trial["correct_choice"],
                "odd_position": trial["odd_position"],
                "source_a": {
                    **pair["source_a"],
                    "rendered_pair_wav_sha256": pair["wav_a_sha256"],
                },
                "source_b": {
                    **pair["source_b"],
                    "rendered_pair_wav_sha256": pair["wav_b_sha256"],
                },
                "pair_id": pair["pair_id"],
                "objective_identity": pair["objective_identity"],
                "coverage_driven": pair["coverage_driven"],
                "coverage_collapse": pair["coverage_collapse"],
                "coverage_ratios": pair["coverage_ratios"],
            }
        )
        render_rows.append(
            {
                "question_id": question_id,
                "presentations": presentations,
            }
        )
        public_trials.append(
            {
                "question_id": question_id,
                "order": int(trial["global_order_index"]) + 1,
                "clips": public_clip_paths,
            }
        )

    practice_private = []
    practice_public = []
    practice_render = []
    for practice in selection["practice_trials"]:
        practice_id = str(practice["practice_id"])
        pair = make_pair(
            f"practice:{practice_id}",
            practice["sources"]["a"],
            practice["sources"]["b"],
        )
        destination = blind_practice / practice_id
        destination.mkdir()
        presentations = []
        public_paths = []
        for position, label in enumerate(practice["presentation_pattern"], start=1):
            side = label.lower()
            midi_path = destination / f"clip_{position}.mid"
            shutil.copyfile(pair[f"midi_{side}"], midi_path)
            wav_path = None
            if not args.midi_only:
                wav_path = destination / f"clip_{position}.wav"
                shutil.copyfile(pair[f"wav_{side}"], wav_path)
                public_paths.append(str(wav_path.relative_to(package / "blind")))
            else:
                public_paths.append(str(midi_path.relative_to(package / "blind")))
            presentations.append(
                {
                    "position": position,
                    "midi": str(midi_path.relative_to(package)),
                    "midi_sha256": file_sha256(midi_path),
                    "wav": str(wav_path.relative_to(package)) if wav_path else None,
                    "wav_sha256": file_sha256(wav_path) if wav_path else None,
                    "audio": pair[f"audio_{side}"],
                }
            )
        practice_private.append(
            {
                "practice_id": practice_id,
                "correct_choice": practice["correct_choice"],
                "presentation_pattern": practice["presentation_pattern"],
                "pair_id": pair["pair_id"],
                "source_a": {
                    **pair["source_a"],
                    "rendered_pair_wav_sha256": pair["wav_a_sha256"],
                },
                "source_b": {
                    **pair["source_b"],
                    "rendered_pair_wav_sha256": pair["wav_b_sha256"],
                },
            }
        )
        practice_public.append(
            {
                "practice_id": practice_id,
                "clips": public_paths,
                "correct_choice": practice["correct_choice"],
                "feedback_allowed": True,
                "scored": False,
            }
        )
        practice_render.append(
            {"practice_id": practice_id, "presentations": presentations}
        )

    selection_sha = file_sha256(selection_path)
    private_key = {
        "schema_version": "streammuse.melody_robustness.listening_triangle_private_key.v2",
        "listening_attempt_id": selection["listening_attempt_id"],
        "retry_lineage": selection.get("retry_lineage"),
        "retry_lineage_sha256": selection.get("retry_lineage_sha256"),
        "selection_sha256": selection_sha,
        "campaign_config_sha256": config_sha,
        "run_schedule_sha256": args.schedule_sha256,
        "campaign_binding_sha256": campaign_binding["campaign_binding_sha256"],
        "qualification_result_sha256": config["qualification_result"]["sha256"],
        "campaign_audit_path": str(Path(args.campaign_audit).resolve()),
        "campaign_audit_sha256": args.campaign_audit_sha256,
        "control_report_path": str(control_report_path),
        "control_report_sha256": control_report_sha256,
        "control_report_base_selection_path": str(control_base_selection_path),
        "control_report_base_selection_sha256": control_base_selection_sha256,
        "control_report_current_selection_sha256": file_sha256(selection_path),
        "renderer_identity_path": str(renderer_identity_path),
        "renderer_identity_sha256": file_sha256(renderer_identity_path),
        "trials": private_rows,
        "practice_trials": practice_private,
        "unblind_only_from_immutable_snapshot": True,
    }
    write_canonical_json(private / "private_key.json", private_key)
    public_manifest = {
        "schema_version": "streammuse.melody_robustness.listening_triangle_public.v2",
        "listening_attempt_id": selection["listening_attempt_id"],
        "question_prompt": TRIANGLE_PROMPT,
        "clip_seconds": TRIANGLE_CLIP_SECONDS,
        "response_choices": ["1", "2", "3", "no_difference"],
        "practice_trials": practice_public,
        "trials": public_trials,
        "semantic_fields_present": False,
    }
    write_canonical_json(package / "blind" / "public_manifest.json", public_manifest)
    (package / "blind" / "player.html").write_text(
        _triangle_player_html(public_manifest, selection_sha), encoding="utf-8"
    )
    render_manifest = {
        "schema_version": "streammuse.melody_robustness.listening_triangle_render.v2",
        "listening_attempt_id": selection["listening_attempt_id"],
        "retry_lineage": selection.get("retry_lineage"),
        "retry_lineage_sha256": selection.get("retry_lineage_sha256"),
        "selection_path": str(selection_path),
        "selection_sha256": selection_sha,
        "input_manifest_path": str(manifest_path),
        "input_manifest_sha256": config["input_manifest"]["sha256"],
        "campaign_config_path": str(Path(args.config).resolve()),
        "campaign_config_sha256": config_sha,
        "run_schedule_path": str(Path(args.schedule).resolve()),
        "run_schedule_sha256": args.schedule_sha256,
        "output_root": str(output_root),
        "campaign_audit_path": str(Path(args.campaign_audit).resolve()),
        "campaign_audit_sha256": args.campaign_audit_sha256,
        "control_report_path": str(control_report_path),
        "control_report_sha256": control_report_sha256,
        "control_report_base_selection_path": str(control_base_selection_path),
        "control_report_base_selection_sha256": control_base_selection_sha256,
        "control_report_current_selection_sha256": file_sha256(selection_path),
        "campaign_binding_sha256": campaign_binding["campaign_binding_sha256"],
        "qualification_result_sha256": config["qualification_result"]["sha256"],
        "renderer_identity_path": str(renderer_identity_path),
        "renderer_identity_sha256": file_sha256(renderer_identity_path),
        "private_key_sha256": file_sha256(private / "private_key.json"),
        "public_manifest_sha256": file_sha256(package / "blind" / "public_manifest.json"),
        "player_sha256": file_sha256(package / "blind" / "player.html"),
        "render_bpm": TRIANGLE_RENDER_BPM,
        "clip_seconds": TRIANGLE_CLIP_SECONDS,
        "sample_rate": args.sample_rate,
        "bit_depth": 16,
        "gain": args.gain,
        "gain_policy": TRIANGLE_GAIN_POLICY,
        "true_peak_limit_dbtp": TRUE_PEAK_LIMIT_DBTP,
        "true_peak_implementation": TRUE_PEAK_IMPLEMENTATION,
        "soundfont_path": str(soundfont) if soundfont else None,
        "soundfont_sha256": file_sha256(soundfont) if soundfont else None,
        "fluidsynth": synth_provenance,
        "fluidsynth_version": synth_version,
        "midi_only_development": bool(args.midi_only),
        "trials": render_rows,
        "practice_trials": practice_render,
    }
    write_canonical_json(package / "render_manifest.json", render_manifest)
    audit = audit_triangle_package_dir(package, require_wav=not args.midi_only)
    write_canonical_json(package / "package_audit.json", audit)
    if not audit["valid"]:
        raise RuntimeError(f"triangle listening package audit failed: {audit['errors']}")
    initial_rows, initial_head = validate_response_ledger(package)
    initial_sittings, initial_sitting_head, _initial_sitting_states = (
        validate_sitting_ledger(package)
    )
    if initial_rows or initial_head is not None or (
        package / "blind" / "response_ledger.jsonl"
    ).exists() or initial_sittings or initial_sitting_head is not None or (
        package / "blind" / "sitting_ledger.jsonl"
    ).exists():
        raise RuntimeError(
            "a newly built listening attempt must start with empty response/sitting ledgers"
        )
    # One pure derivation owns both n=0 initialization and every later resume
    # update; progress is mutable and deliberately has no stale hash sidecar.
    (package / "blind" / "progress_state.json").write_bytes(
        canonical_json_bytes(progress_summary(package))
    )


def _triangle_expected_blind_paths(
    render: Mapping[str, Any],
) -> tuple[set[str], set[str], set[str]]:
    midi_only = render.get("midi_only_development")
    if not isinstance(midi_only, bool):
        raise ValueError("render midi_only_development must be boolean")
    expected_files = {
        "player.html",
        "public_manifest.json",
        "public_manifest.json.sha256",
    }
    expected_dirs = {"trials", "practice"}
    served_files = {"player.html", "public_manifest.json"}
    for prefix, count, id_prefix in (
        ("trials", TRIANGLE_TRIAL_COUNT, "Q"),
        ("practice", TRIANGLE_PRACTICE_COUNT, "P"),
    ):
        for index in range(1, count + 1):
            opaque_id = f"{id_prefix}{index:03d}"
            directory = f"{prefix}/{opaque_id}"
            expected_dirs.add(directory)
            for position in range(1, 4):
                midi = f"{directory}/clip_{position}.mid"
                expected_files.add(midi)
                served_files.add(midi)
                if not midi_only:
                    wav = f"{directory}/clip_{position}.wav"
                    expected_files.add(wav)
                    served_files.add(wav)
    return expected_files, expected_dirs, served_files


def _audit_triangle_blind_file_set(
    package: Path,
    render: Mapping[str, Any],
    public: Mapping[str, Any],
) -> list[str]:
    """Enforce an exact opaque public tree; mutable ledgers are the only extras."""

    errors: list[str] = []
    try:
        expected_files, expected_dirs, _served = _triangle_expected_blind_paths(render)
    except Exception as exc:
        return [f"blind exact-file contract is unavailable: {exc}"]

    expected_public_fields = {
        "schema_version",
        "listening_attempt_id",
        "question_prompt",
        "clip_seconds",
        "response_choices",
        "practice_trials",
        "trials",
        "semantic_fields_present",
    }
    if set(public) != expected_public_fields:
        errors.append("public manifest fields differ from the exact blind schema")
    if public.get("schema_version") != (
        "streammuse.melody_robustness.listening_triangle_public.v2"
    ):
        errors.append("public manifest schema mismatch")
    if public.get("clip_seconds") != TRIANGLE_CLIP_SECONDS:
        errors.append("public manifest clip duration drifted")
    if public.get("response_choices") != ["1", "2", "3", "no_difference"]:
        errors.append("public manifest response choices drifted")

    midi_only = bool(render.get("midi_only_development"))
    public_path = package / "blind" / "public_manifest.json"
    public_sidecar = public_path.with_name(public_path.name + ".sha256")
    expected_sidecar = f"{file_sha256(public_path)}  {public_path.name}\n"
    if not public_sidecar.is_file() or public_sidecar.read_text(
        encoding="ascii"
    ) != expected_sidecar:
        errors.append("public manifest checksum sidecar is missing or invalid")

    for prefix, id_field, count, id_prefix, public_fields in (
        (
            "trials",
            "question_id",
            TRIANGLE_TRIAL_COUNT,
            "Q",
            {"question_id", "order", "clips"},
        ),
        (
            "practice",
            "practice_id",
            TRIANGLE_PRACTICE_COUNT,
            "P",
            {
                "practice_id",
                "clips",
                "correct_choice",
                "feedback_allowed",
                "scored",
            },
        ),
    ):
        render_rows = render.get(prefix if prefix == "trials" else "practice_trials")
        public_rows = public.get(prefix if prefix == "trials" else "practice_trials")
        if not isinstance(render_rows, list) or not isinstance(public_rows, list):
            errors.append(f"{prefix}: render/public rows are missing")
            continue
        expected_ids = [f"{id_prefix}{index:03d}" for index in range(1, count + 1)]
        render_ids = [
            str(row.get(id_field))
            for row in render_rows
            if isinstance(row, Mapping)
        ]
        public_ids = [
            str(row.get(id_field))
            for row in public_rows
            if isinstance(row, Mapping)
        ]
        if render_ids != expected_ids:
            errors.append(f"{prefix}: render IDs/order differ from the opaque contract")
        if public_ids != expected_ids:
            errors.append(f"{prefix}: public IDs/order differ from the opaque contract")
        if len(render_rows) != count or len(public_rows) != count:
            continue
        for index, (render_row, public_row) in enumerate(
            zip(render_rows, public_rows, strict=True), start=1
        ):
            opaque_id = f"{id_prefix}{index:03d}"
            if not isinstance(render_row, Mapping) or not isinstance(public_row, Mapping):
                errors.append(f"{prefix}/{opaque_id}: row is not an object")
                continue
            if set(public_row) != public_fields:
                errors.append(
                    f"{prefix}/{opaque_id}: public row contains semantic/unexpected fields"
                )
            if prefix == "trials" and public_row.get("order") != index:
                errors.append(f"{prefix}/{opaque_id}: public order drifted")
            presentations = render_row.get("presentations")
            if not isinstance(presentations, list) or len(presentations) != 3:
                continue
            public_extension = "mid" if midi_only else "wav"
            expected_public_clips = [
                f"{prefix}/{opaque_id}/clip_{position}.{public_extension}"
                for position in range(1, 4)
            ]
            if public_row.get("clips") != expected_public_clips:
                errors.append(f"{prefix}/{opaque_id}: public clip paths are not exact opaque paths")
            for position, presentation in enumerate(presentations, start=1):
                if not isinstance(presentation, Mapping):
                    continue
                expected_midi = f"blind/{prefix}/{opaque_id}/clip_{position}.mid"
                expected_wav = (
                    None
                    if midi_only
                    else f"blind/{prefix}/{opaque_id}/clip_{position}.wav"
                )
                if presentation.get("midi") != expected_midi:
                    errors.append(
                        f"{prefix}/{opaque_id}/{position}: render MIDI path is not exact/opaque"
                    )
                if presentation.get("wav") != expected_wav:
                    errors.append(
                        f"{prefix}/{opaque_id}/{position}: render WAV path is not exact/opaque"
                    )

    blind = package / "blind"
    if blind.is_symlink() or not blind.is_dir():
        errors.append("blind root must be a real directory, not a symlink")
        return errors
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in blind.rglob("*"):
        relative = path.relative_to(blind).as_posix()
        if path.is_symlink():
            errors.append(f"blind tree contains a symlink: {relative}")
        elif path.is_file():
            actual_files.add(relative)
        elif path.is_dir():
            actual_dirs.add(relative)
        else:
            errors.append(f"blind tree contains a non-file/non-directory node: {relative}")

    mutable_files = {
        "progress_state.json",
        "response_ledger.jsonl",
        "sitting_ledger.jsonl",
    }
    missing_files = sorted(expected_files - actual_files)
    extra_files = sorted(actual_files - expected_files - mutable_files)
    missing_dirs = sorted(expected_dirs - actual_dirs)
    extra_dirs = sorted(actual_dirs - expected_dirs)
    if missing_files:
        errors.append(f"blind tree is missing required files: {missing_files}")
    if extra_files:
        errors.append(f"blind tree contains unexpected files: {extra_files}")
    if missing_dirs:
        errors.append(f"blind tree is missing required directories: {missing_dirs}")
    if extra_dirs:
        errors.append(f"blind tree contains unexpected directories: {extra_dirs}")
    return errors


def audit_triangle_package_dir(
    package_dir: str | Path, *, require_wav: bool = True
) -> dict[str, Any]:
    package = Path(package_dir).resolve()
    errors: list[str] = []
    try:
        render = _read_json(package / "render_manifest.json")
        key = _read_json(package / "private" / "private_key.json")
        public = _read_json(package / "blind" / "public_manifest.json")
    except Exception as exc:
        return {
            "schema_version": "streammuse.melody_robustness.listening_triangle_audit.v2",
            "valid": False,
            "accepted_final": False,
            "errors": [f"missing/unreadable package metadata: {exc}"],
            "trial_count": 0,
            "presentation_count": 0,
        }
    errors.extend(_audit_triangle_blind_file_set(package, render, public))
    selection_path = Path(str(render.get("selection_path", ""))).resolve()
    selection: dict[str, Any] | None = None
    formal_schedule: list[dict[str, Any]] | None = None
    formal_binding: dict[str, Any] | None = None
    formal_output_root: Path | None = None
    renderer_identity: dict[str, Any] | None = None
    control_report_path: Path | None = None
    control_records: dict[str, Mapping[str, Any]] = {}
    rerendered_raw_pcm: dict[str, np.ndarray] = {}
    canonical_note_events: dict[str, list[dict[str, int]]] = {}
    rerendered_source_count = 0
    rerendered_pair_count = 0
    if not selection_path.is_file():
        errors.append("render manifest selection path is missing")
    elif file_sha256(selection_path) != render.get("selection_sha256"):
        errors.append("render manifest selection hash mismatch")
    else:
        try:
            selection = _read_json(selection_path)
            manifest_path = Path(str(selection["input_manifest_path"])).resolve()
            manifest = _read_json(manifest_path)
            validate_triangle_selection_manifest(
                selection,
                manifest,
                manifest_path=manifest_path,
                verify_files=True,
            )
        except Exception as exc:
            errors.append(f"triangle selection exact rebuild failed: {exc}")
    try:
        config_path = Path(str(render["campaign_config_path"])).resolve()
        schedule_path = Path(str(render["run_schedule_path"])).resolve()
        formal_output_root = Path(str(render["output_root"])).resolve()
        if file_sha256(config_path) != render.get("campaign_config_sha256"):
            raise ValueError("campaign config hash mismatch")
        if file_sha256(schedule_path) != render.get("run_schedule_sha256"):
            raise ValueError("run schedule hash mismatch")
        config = _read_json(config_path)
        validate_campaign_config(config)
        validate_frozen_qualification(config, verify_files=True)
        formal_schedule = read_jsonl(schedule_path)
        if selection is None:
            raise ValueError("selection is unavailable")
        manifest = _read_json(Path(selection["input_manifest_path"]))
        if formal_schedule != build_run_schedule(manifest, config):
            raise ValueError("formal schedule differs from frozen exact rebuild")
        formal_binding = _validated_campaign_binding(
            argparse.Namespace(
                config=str(config_path),
                schedule=str(schedule_path),
                output_root=str(formal_output_root),
                schedule_sha256=render["run_schedule_sha256"],
            ),
            config,
            str(render["campaign_config_sha256"]),
        )
        campaign_audit_path = Path(str(render["campaign_audit_path"])).resolve()
        if file_sha256(campaign_audit_path) != render.get("campaign_audit_sha256"):
            raise ValueError("campaign audit path/hash mismatch")
        (
            _validated_config,
            _validated_config_sha,
            formal_schedule,
            _validated_manifest_path,
            _validated_entries,
            formal_binding,
            _validated_campaign_audit,
        ) = _validated_triangle_build_inputs(
            argparse.Namespace(
                config=str(config_path),
                config_sha256=render["campaign_config_sha256"],
                selection=str(selection_path),
                schedule=str(schedule_path),
                schedule_sha256=render["run_schedule_sha256"],
                output_root=str(formal_output_root),
                campaign_audit=str(campaign_audit_path),
                campaign_audit_sha256=render["campaign_audit_sha256"],
            ),
            selection_path,
            selection,
        )
        renderer_path = Path(str(render["renderer_identity_path"])).resolve()
        if file_sha256(renderer_path) != render.get("renderer_identity_sha256"):
            raise ValueError("renderer identity path/hash mismatch")
        renderer_identity = validate_triangle_renderer_identity(
            _read_json(renderer_path), verify_files=True
        )
        if render.get("soundfont_path") != renderer_identity["soundfont"]["path"]:
            if not render.get("midi_only_development"):
                raise ValueError("render soundfont path differs from frozen renderer")
        if render.get("soundfont_sha256") != (
            None
            if render.get("midi_only_development")
            else renderer_identity["soundfont"]["sha256"]
        ):
            raise ValueError("render soundfont hash differs from frozen renderer")
        if not render.get("midi_only_development") and render.get("fluidsynth") != renderer_identity[
            "fluidsynth"
        ]:
            raise ValueError("render FluidSynth provenance differs from frozen renderer")
    except Exception as exc:
        errors.append(f"formal campaign/renderer revalidation failed: {exc}")
    if selection is not None:
        try:
            (
                control_base_path,
                control_base_selection,
                control_base_sha,
            ) = _triangle_control_base_selection(selection_path, selection)
            if (
                Path(str(render["control_report_base_selection_path"])).resolve()
                != control_base_path
                or render.get("control_report_base_selection_sha256")
                != control_base_sha
                or render.get("control_report_current_selection_sha256")
                != render.get("selection_sha256")
            ):
                raise ValueError("control report base/current selection binding differs")
            control_report_path = Path(str(render["control_report_path"])).resolve()
            _control_report, control_records = _validate_triangle_control_report(
                control_report_path,
                str(render["control_report_sha256"]),
                selection=control_base_selection,
                selection_sha256=control_base_sha,
                campaign_config_sha256=str(render["campaign_config_sha256"]),
                verify_files=True,
            )
        except Exception as exc:
            errors.append(f"analyzer control-report revalidation failed: {exc}")
    if render.get("schema_version") != "streammuse.melody_robustness.listening_triangle_render.v2":
        errors.append("triangle render schema mismatch")
    expected_render = {
        "render_bpm": TRIANGLE_RENDER_BPM,
        "clip_seconds": TRIANGLE_CLIP_SECONDS,
        "sample_rate": TRIANGLE_RENDER_SAMPLE_RATE,
        "bit_depth": 16,
        "gain": TRIANGLE_SYNTH_GAIN,
        "gain_policy": TRIANGLE_GAIN_POLICY,
        "true_peak_implementation": TRUE_PEAK_IMPLEMENTATION,
    }
    for field, expected in expected_render.items():
        if render.get(field) != expected:
            errors.append(f"render {field} must equal {expected!r}")
    if render.get("private_key_sha256") != file_sha256(
        package / "private" / "private_key.json"
    ):
        errors.append("private key hash mismatch")
    if render.get("public_manifest_sha256") != file_sha256(
        package / "blind" / "public_manifest.json"
    ):
        errors.append("public manifest hash mismatch")
    player_path = package / "blind" / "player.html"
    if not player_path.is_file() or render.get("player_sha256") != file_sha256(player_path):
        errors.append("blind player is missing or hash-mismatched")
    if key.get("selection_sha256") != render.get("selection_sha256"):
        errors.append("private key and render selection bindings differ")
    if (
        key.get("listening_attempt_id") != render.get("listening_attempt_id")
        or public.get("listening_attempt_id") != render.get("listening_attempt_id")
        or selection is not None
        and selection.get("listening_attempt_id") != render.get("listening_attempt_id")
    ):
        errors.append("public/private/render listening attempt IDs differ")
    if (
        key.get("retry_lineage") != render.get("retry_lineage")
        or key.get("retry_lineage_sha256") != render.get("retry_lineage_sha256")
        or selection is not None
        and (
            selection.get("retry_lineage") != render.get("retry_lineage")
            or selection.get("retry_lineage_sha256")
            != render.get("retry_lineage_sha256")
        )
    ):
        errors.append("private/render/selection retry lineage differs")
    for field in (
        "campaign_config_sha256",
        "run_schedule_sha256",
        "campaign_binding_sha256",
        "qualification_result_sha256",
        "campaign_audit_sha256",
        "control_report_sha256",
        "control_report_base_selection_sha256",
        "control_report_current_selection_sha256",
    ):
        if key.get(field) != render.get(field):
            errors.append(f"private key and render {field} differ")
    if (
        Path(str(key.get("control_report_path", ""))).resolve()
        != Path(str(render.get("control_report_path", ""))).resolve()
        or Path(str(key.get("control_report_base_selection_path", ""))).resolve()
        != Path(str(render.get("control_report_base_selection_path", ""))).resolve()
    ):
        errors.append("private key and render control-report/base-selection paths differ")
    if public.get("semantic_fields_present") is not False:
        errors.append("public manifest does not explicitly prohibit semantic fields")
    if public.get("question_prompt") != TRIANGLE_PROMPT:
        errors.append("public triangle question prompt drifted")
    public_trials = public.get("trials")
    key_trials = key.get("trials")
    render_trials = render.get("trials")
    if not isinstance(public_trials, list) or len(public_trials) != TRIANGLE_TRIAL_COUNT:
        errors.append("public manifest must contain exactly 95 trials")
        public_trials = []
    if not isinstance(key_trials, list) or len(key_trials) != TRIANGLE_TRIAL_COUNT:
        errors.append("private key must contain exactly 95 trials")
        key_trials = []
    if not isinstance(render_trials, list) or len(render_trials) != TRIANGLE_TRIAL_COUNT:
        errors.append("render manifest must contain exactly 95 trials")
        render_trials = []
    ids_public = [str(row.get("question_id")) for row in public_trials]
    ids_key = [str(row.get("question_id")) for row in key_trials]
    ids_render = [str(row.get("question_id")) for row in render_trials]
    expected_ids = [f"Q{index:03d}" for index in range(1, TRIANGLE_TRIAL_COUNT + 1)]
    if ids_public != expected_ids or ids_key != expected_ids or ids_render != expected_ids:
        errors.append("public/private/render trial IDs or order differ from Q001..Q095")
    if any(set(row) != {"question_id", "order", "clips"} for row in public_trials):
        errors.append("public trial rows contain semantic or unexpected fields")
    public_text = json.dumps(public, sort_keys=True)
    if selection is not None:
        forbidden = {
            str(seed) for seed in [*SEEDS["perturb"], *SEEDS["sample"], SEEDS["high_perturb"]]
        }
        forbidden.update({"semantic_id", "condition", "perturb_seed", "sample_seed", "run_id"})
        songs = {
            str(trial["sources"]["a"].get("song"))
            for trial in selection["trials"]
            if trial["sources"]["a"].get("song") is not None
        }
        forbidden.update(song for song in songs if len(song) >= 4)
        leaked = sorted(value for value in forbidden if value in public_text)
        if leaked:
            errors.append(f"public manifest leaks semantic values: {leaked}")
        if player_path.is_file():
            player_text = player_path.read_text(encoding="utf-8")
            player_leaked = sorted(value for value in forbidden if value in player_text)
            if player_leaked:
                errors.append(f"blind player leaks semantic values: {player_leaked}")

    if selection is not None and len(key_trials) == TRIANGLE_TRIAL_COUNT:
        selector_fields = {
            "kind",
            "formal_pipeline",
            "source_artifact",
            "presentation",
            "song",
            "condition",
            "perturb_seed",
            "sample_seed",
            "recipe",
        }
        for selected, private_row in zip(selection["trials"], key_trials, strict=True):
            expected_fields = {
                "question_id": selected["question_id"],
                "semantic_id": selected["semantic_id"],
                "block": selected["block"],
                "condition": selected["condition"],
                "repeat_of": selected.get("repeat_of"),
                "repeat_distance": selected.get("repeat_distance"),
                "presentation_pattern": selected["presentation_pattern"],
                "correct_choice": selected["correct_choice"],
                "odd_position": selected["odd_position"],
            }
            if any(private_row.get(field) != value for field, value in expected_fields.items()):
                errors.append(f"{selected['question_id']}: private semantics differ from selection")
            for selected_side, private_side in (("a", "source_a"), ("b", "source_b")):
                provenance = private_row.get(private_side)
                if not isinstance(provenance, Mapping):
                    errors.append(f"{selected['question_id']}: missing {private_side}")
                    continue
                projected = {
                    field: provenance[field]
                    for field in selector_fields
                    if field in provenance
                }
                if projected != selected["sources"][selected_side]:
                    errors.append(
                        f"{selected['question_id']}: {private_side} selector differs from selection"
                    )
            if selected.get("block") == "known_different_control":
                semantic_id = str(selected["semantic_id"])
                try:
                    record = control_records.get(semantic_id)
                    if not isinstance(record, Mapping):
                        raise ValueError("analyzer control record is unavailable")
                    _validate_known_different_material(
                        semantic_id,
                        private_row["source_a"],
                        private_row["source_b"],
                        record,
                    )
                except Exception as exc:
                    errors.append(
                        f"{selected['question_id']}: T7.7 control closure failed: {exc}"
                    )

    render_by_id = {str(row.get("question_id")): row for row in render_trials}
    key_by_id = {str(row.get("question_id")): row for row in key_trials}
    total_presentations = 0
    for question_id in expected_ids:
        key_row = key_by_id.get(question_id)
        render_row = render_by_id.get(question_id)
        if key_row is None or render_row is None:
            continue
        presentations = render_row.get("presentations")
        if not isinstance(presentations, list) or len(presentations) != 3:
            errors.append(f"{question_id}: expected exactly three presentations")
            continue
        total_presentations += len(presentations)
        pattern = str(key_row.get("presentation_pattern", ""))
        if pattern not in {*TRIANGLE_PATTERNS, "AAA"}:
            errors.append(f"{question_id}: invalid presentation pattern")
            continue
        midi_hashes: list[str | None] = []
        wav_hashes: list[str | None] = []
        for position, presentation in enumerate(presentations, start=1):
            if presentation.get("position") != position:
                errors.append(f"{question_id}: presentation positions are not 1,2,3")
            raw_midi = Path(str(presentation.get("midi", "")))
            midi = (package / raw_midi).resolve()
            if raw_midi.is_absolute() or not midi.is_relative_to(package / "blind"):
                errors.append(f"{question_id}/{position}: MIDI escapes blind package")
                continue
            if not midi.is_file() or file_sha256(midi) != presentation.get("midi_sha256"):
                errors.append(f"{question_id}/{position}: MIDI missing or corrupt")
            else:
                try:
                    _canonical_midi_contract(
                        midi,
                        expected_end_model_tick=TRIANGLE_CLIP_BEATS * 4,
                    )
                except Exception as exc:
                    errors.append(
                        f"{question_id}/{position}: canonical MIDI contract failed: {exc}"
                    )
            midi_hashes.append(presentation.get("midi_sha256"))
            wav_hashes.append(presentation.get("wav_sha256"))
            if not require_wav:
                continue
            raw_wav = Path(str(presentation.get("wav", "")))
            wav = (package / raw_wav).resolve()
            if raw_wav.is_absolute() or not wav.is_relative_to(package / "blind"):
                errors.append(f"{question_id}/{position}: WAV escapes blind package")
                continue
            if not wav.is_file() or file_sha256(wav) != presentation.get("wav_sha256"):
                errors.append(f"{question_id}/{position}: WAV missing or corrupt")
                continue
            try:
                with wave.open(str(wav), "rb") as source:
                    channels = source.getnchannels()
                    width = source.getsampwidth()
                    rate = source.getframerate()
                    frame_count = source.getnframes()
                    frames = source.readframes(frame_count)
                if channels not in {1, 2} or width != 2 or rate != TRIANGLE_RENDER_SAMPLE_RATE:
                    errors.append(f"{question_id}/{position}: WAV format drift")
                    continue
                if frame_count != TRIANGLE_RENDER_SAMPLE_RATE * TRIANGLE_CLIP_SECONDS:
                    errors.append(f"{question_id}/{position}: WAV is not exactly 8 seconds")
                pcm = np.frombuffer(frames, dtype="<i2")
                silent = not bool(np.any(pcm))
                peak = _measure_true_peak(
                    pcm.astype(np.float64).reshape(-1, channels) / 32768.0
                )
                peak_dbtp = _dbtp(peak)
                if peak_dbtp is not None and peak_dbtp > TRUE_PEAK_LIMIT_DBTP:
                    errors.append(f"{question_id}/{position}: true peak exceeds limit")
                audio = presentation.get("audio")
                if not isinstance(audio, Mapping):
                    errors.append(f"{question_id}/{position}: missing audio provenance")
                    continue
                if audio.get("true_peak_verified") is not True:
                    errors.append(f"{question_id}/{position}: true peak not verified")
                if audio.get("peak_measurement") != TRUE_PEAK_IMPLEMENTATION:
                    errors.append(f"{question_id}/{position}: peak implementation drift")
                recorded_peak = audio.get("true_peak_dbtp_after")
                if peak_dbtp is None:
                    if recorded_peak is not None:
                        errors.append(f"{question_id}/{position}: silent peak metadata drift")
                elif not isinstance(recorded_peak, (int, float)) or abs(
                    float(recorded_peak) - peak_dbtp
                ) > 1e-7:
                    errors.append(f"{question_id}/{position}: true peak metadata mismatch")
                side = "source_a" if pattern[position - 1] == "A" else "source_b"
                expected_empty = bool(key_row.get(side, {}).get("source_empty"))
                if silent != expected_empty:
                    errors.append(
                        f"{question_id}/{position}: silence differs from source-empty policy"
                    )
            except Exception as exc:
                errors.append(f"{question_id}/{position}: WAV audit failed: {exc}")
        label_hashes: dict[str, list[tuple[str | None, str | None]]] = defaultdict(list)
        for label, midi_hash, wav_hash in zip(pattern, midi_hashes, wav_hashes, strict=True):
            label_hashes[label].append((midi_hash, wav_hash))
        for label, hashes in label_hashes.items():
            if len(hashes) >= 2 and len(set(hashes)) != 1:
                errors.append(f"{question_id}: duplicate {label} presentations are not literal copies")
        if require_wav and pattern != "AAA":
            audio_by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for label, presentation in zip(pattern, presentations, strict=True):
                audio = presentation.get("audio")
                if isinstance(audio, Mapping):
                    audio_by_label[label].append(audio)
            for label, rows in audio_by_label.items():
                if len(rows) >= 2 and any(dict(row) != dict(rows[0]) for row in rows[1:]):
                    errors.append(
                        f"{question_id}: duplicate {label} audio provenance differs"
                    )
            if audio_by_label.get("A") and audio_by_label.get("B"):
                scale_a = audio_by_label["A"][0].get("common_pair_gain_scale")
                scale_b = audio_by_label["B"][0].get("common_pair_gain_scale")
                if (
                    isinstance(scale_a, bool)
                    or not isinstance(scale_a, (int, float))
                    or isinstance(scale_b, bool)
                    or not isinstance(scale_b, (int, float))
                    or not 0.0 < float(scale_a) <= 1.0
                    or not 0.0 < float(scale_b) <= 1.0
                    or float(scale_a) != float(scale_b)
                ):
                    errors.append(
                        f"{question_id}: A/B common-pair gain scales are inconsistent"
                    )
        if key_row.get("block") == "identity_catch":
            if pattern != "AAA" or len(set(zip(midi_hashes, wav_hashes, strict=True))) != 1:
                errors.append(f"{question_id}: identity catch is not literal A/A/A")
        elif pattern == "AAA":
            errors.append(f"{question_id}: non-identity trial uses A/A/A")

    if total_presentations != TRIANGLE_PRESENTATION_COUNT:
        errors.append(
            f"expected {TRIANGLE_PRESENTATION_COUNT} scored presentations, got {total_presentations}"
        )
    block_counts = Counter(str(row.get("block")) for row in key_trials)
    if dict(block_counts) != TRIANGLE_BLOCK_COUNTS:
        errors.append(f"triangle private-key block counts drifted: {dict(block_counts)}")
    original_by_semantic = {
        str(row.get("semantic_id")): row
        for row in key_trials
        if row.get("block") != "exact_repeat"
    }
    repeats = [row for row in key_trials if row.get("block") == "exact_repeat"]
    for repeat in repeats:
        original = original_by_semantic.get(str(repeat.get("repeat_of")))
        if (
            original is None
            or repeat.get("pair_id") != original.get("pair_id")
            or repeat.get("presentation_pattern") == original.get("presentation_pattern")
            or not isinstance(repeat.get("repeat_distance"), int)
            or repeat["repeat_distance"] < 8
        ):
            errors.append(f"{repeat.get('question_id')}: exact-repeat mapping is invalid")
    canonical_sources: dict[str, Mapping[str, Any]] = {}
    raw_practice_key = key.get("practice_trials")
    practice_key_rows = raw_practice_key if isinstance(raw_practice_key, list) else []
    for trial in [*key_trials, *practice_key_rows]:
        for side in ("source_a", "source_b"):
            source = trial.get(side)
            if not isinstance(source, Mapping):
                continue
            source_id = str(source.get("canonical_source_id", ""))
            if not re.fullmatch(r"CS\d{3}", source_id):
                errors.append(f"invalid canonical source ID: {source_id!r}")
                continue
            prior = canonical_sources.get(source_id)
            prior_canonical = dict(prior) if prior is not None else None
            source_canonical = dict(source)
            if prior_canonical is not None:
                prior_canonical.pop("rendered_pair_wav_sha256", None)
            source_canonical.pop("rendered_pair_wav_sha256", None)
            if prior_canonical is not None and prior_canonical != source_canonical:
                errors.append(f"{source_id}: conflicting source provenance records")
                continue
            canonical_sources[source_id] = source
    for source_id, source in canonical_sources.items():
        try:
            excerpt_midi = Path(str(source["excerpt_midi_path"])).resolve()
            expected_source_dir = (
                package / "private" / "canonical_sources" / source_id
            ).resolve()
            if (
                not excerpt_midi.is_relative_to(expected_source_dir)
                or excerpt_midi.name != "excerpt.mid"
                or not excerpt_midi.is_file()
                or file_sha256(excerpt_midi) != source.get("excerpt_midi_sha256")
            ):
                raise ValueError("canonical excerpt path/hash is not package-bound")
            stored_events = _canonical_midi_contract(
                excerpt_midi,
                expected_end_model_tick=TRIANGLE_CLIP_BEATS * 4,
            )
            selector_kind = source.get("kind")
            with tempfile.TemporaryDirectory(prefix="streammuse-triangle-audit-") as raw_tmp:
                rebuilt_midi = Path(raw_tmp) / "excerpt.mid"
                if selector_kind == "formal":
                    theoretical = Path(str(source["source_path"])).resolve()
                    inferences = Path(str(source["inferences_path"])).resolve()
                    verdict_path = Path(str(source["verdict_path"])).resolve()
                    for label, path, digest in (
                        ("theoretical model", theoretical, source.get("source_sha256")),
                        ("inference trace", inferences, source.get("inferences_sha256")),
                        ("immutable verdict", verdict_path, source.get("verdict_sha256")),
                    ):
                        if not path.is_file() or file_sha256(path) != digest:
                            raise ValueError(f"{label} file/hash mismatch")
                    verdict = _read_json(verdict_path)
                    indexed = verdict.get("artifact_index")
                    if not isinstance(indexed, list):
                        raise ValueError("verdict artifact index is missing")
                    attempt = verdict_path.parent
                    index_by_path = {
                        (attempt / str(record.get("path"))).resolve(): record
                        for record in indexed
                        if isinstance(record, Mapping)
                    }
                    for label, path in (
                        ("theoretical model", theoretical),
                        ("inference trace", inferences),
                    ):
                        record = index_by_path.get(path)
                        if (
                            not isinstance(record, Mapping)
                            or record.get("sha256") != file_sha256(path)
                            or record.get("size") != path.stat().st_size
                        ):
                            raise ValueError(f"{label} is not hash/size-bound by verdict")
                    recomputed = _raw_token_payload_provenance(inferences)
                    for field in (
                        "inferences_sha256",
                        "inference_count",
                        "raw_token_payload_sha256",
                        "output_event_payload_sha256",
                    ):
                        if recomputed[field] != source.get(field):
                            raise ValueError(
                                f"{field} differs from inference-trace recomputation"
                            )
                    if (
                        formal_schedule is None
                        or formal_binding is None
                        or formal_output_root is None
                    ):
                        raise ValueError("formal schedule/binding is unavailable")
                    schedule_row = _triangle_find_run(formal_schedule, source)
                    if str(schedule_row.get("run_id")) != source.get("run_id"):
                        raise ValueError("formal selector run_id differs from provenance")
                    attempt, verified_verdict, indexed_paths = verify_attempt_verdict(
                        formal_output_root / "runs" / str(schedule_row["run_id"]),
                        schedule_row,
                        formal_binding,
                        require_content_valid=True,
                    )
                    if (
                        attempt.name != source.get("attempt_id")
                        or verified_verdict.get("operational_valid")
                        != source.get("operational_valid")
                        or theoretical not in indexed_paths
                        or inferences not in indexed_paths
                    ):
                        raise ValueError(
                            "shared immutable-attempt revalidation differs from provenance"
                        )
                    excerpt = source.get("excerpt")
                    if not isinstance(excerpt, Mapping):
                        raise ValueError("formal source excerpt provenance is missing")
                    rebuilt_roll, rebuilt_events = _rebuild_formal_excerpt_midi(
                        theoretical,
                        rebuilt_midi,
                        start_model_tick=int(excerpt["start_model_tick"]),
                        end_model_tick=int(excerpt["end_model_tick"]),
                    )
                elif selector_kind in {"synthetic_control", "practice_tone"}:
                    rebuilt_roll, velocity = build_triangle_control_roll(source)
                    write_roll_midi(
                        rebuilt_roll,
                        rebuilt_midi,
                        bpm=TRIANGLE_RENDER_BPM,
                        velocity=velocity,
                    )
                    rebuilt_events = _canonical_midi_contract(
                        rebuilt_midi,
                        expected_end_model_tick=TRIANGLE_CLIP_BEATS * 4,
                    )
                else:
                    raise ValueError(f"unsupported scored source kind: {selector_kind!r}")
                if rebuilt_midi.read_bytes() != excerpt_midi.read_bytes():
                    raise ValueError(
                        "canonical excerpt differs byte-for-byte from independent rebuild"
                    )
            if rebuilt_events != stored_events:
                raise ValueError("canonical note events differ from independent rebuild")
            if source.get("excerpt_note_event_sha256") != _note_event_sha256(
                rebuilt_events
            ):
                raise ValueError("canonical note-event digest differs from provenance")
            canonical_note_events[source_id] = list(rebuilt_events)
            coverage = len({tick for tick, _pitch in rebuilt_roll.sustain}) / (
                TRIANGLE_CLIP_BEATS * 4
            )
            if (
                source.get("source_empty")
                != (not bool(rebuilt_roll.sustain or rebuilt_roll.onsets))
                or source.get("excerpt_note_onset_count") != len(rebuilt_roll.onsets)
                or source.get("excerpt_sustain_cell_count") != len(rebuilt_roll.sustain)
                or source.get("excerpt_coverage_ratio") != coverage
            ):
                raise ValueError("canonical excerpt summary differs from independent rebuild")
            canonical_wav_raw = source.get("canonical_wav_path")
            if require_wav:
                canonical_wav = Path(str(canonical_wav_raw)).resolve()
                if (
                    not canonical_wav.is_relative_to(expected_source_dir)
                    or not canonical_wav.is_file()
                    or file_sha256(canonical_wav) != source.get("canonical_wav_sha256")
                ):
                    raise ValueError("canonical source WAV path/hash mismatch")
                if renderer_identity is None:
                    raise ValueError("frozen renderer identity is unavailable")
                synth = renderer_identity["fluidsynth"]
                frozen_env = dict(os.environ)
                frozen_library_path = synth.get("ld_library_path")
                if frozen_library_path is None:
                    frozen_env.pop("LD_LIBRARY_PATH", None)
                else:
                    frozen_env["LD_LIBRARY_PATH"] = str(frozen_library_path)
                with tempfile.TemporaryDirectory(
                    prefix="streammuse-triangle-render-audit-"
                ) as raw_render_tmp:
                    rebuilt_wav = Path(raw_render_tmp) / "canonical.wav"
                    _render_with_env(
                        excerpt_midi,
                        rebuilt_wav,
                        soundfont=Path(renderer_identity["soundfont"]["path"]),
                        fluidsynth=str(synth["binary_path"]),
                        sample_rate=TRIANGLE_RENDER_SAMPLE_RATE,
                        gain=TRIANGLE_SYNTH_GAIN,
                        env=frozen_env,
                    )
                    rebuilt_pcm, _channels = _read_rendered_pcm(
                        rebuilt_wav, seconds=TRIANGLE_CLIP_SECONDS
                    )
                    _write_common_protected_wavs([(rebuilt_wav, rebuilt_pcm)])
                    if rebuilt_wav.read_bytes() != canonical_wav.read_bytes():
                        raise ValueError(
                            "canonical source WAV differs byte-for-byte from frozen re-render"
                        )
                    rerendered_raw_pcm[source_id] = rebuilt_pcm.copy()
                rerendered_source_count += 1
        except Exception as exc:
            errors.append(f"{source_id}: canonical source provenance audit failed: {exc}")

    # Rebuild every unique matched pair from the frozen renderer's raw PCM.
    # This binds pair-level common gain, private a/b.wav, presentation metadata,
    # and every blind copy to the canonical source MIDI rather than to mutable
    # hashes recorded inside the package itself.
    raw_practice_render = render.get("practice_trials")
    practice_render_rows = (
        raw_practice_render if isinstance(raw_practice_render, list) else []
    )
    practice_render_by_id = {
        str(row.get("practice_id")): row for row in practice_render_rows
    }
    pair_specs: dict[str, dict[str, Any]] = {}

    def register_pair(
        private_row: Mapping[str, Any],
        render_row: Mapping[str, Any] | None,
        *,
        context_id: str,
    ) -> None:
        pair_id = str(private_row.get("pair_id", ""))
        source_a = private_row.get("source_a")
        source_b = private_row.get("source_b")
        if (
            re.fullmatch(r"C\d{3}", pair_id) is None
            or not isinstance(source_a, Mapping)
            or not isinstance(source_b, Mapping)
            or not isinstance(render_row, Mapping)
        ):
            errors.append(f"{context_id}: matched-pair provenance is incomplete")
            return
        source_ids = (
            str(source_a.get("canonical_source_id")),
            str(source_b.get("canonical_source_id")),
        )
        spec = pair_specs.setdefault(
            pair_id,
            {
                "source_ids": source_ids,
                "contexts": [],
            },
        )
        if spec["source_ids"] != source_ids:
            errors.append(f"{pair_id}: reused with different canonical sources")
            return
        spec["contexts"].append(
            {
                "context_id": context_id,
                "pattern": str(private_row.get("presentation_pattern", "")),
                "presentations": render_row.get("presentations"),
                "source_a": source_a,
                "source_b": source_b,
                "scored": "question_id" in private_row,
                "block": private_row.get("block"),
                "objective_identity": private_row.get("objective_identity"),
                "coverage_driven": private_row.get("coverage_driven"),
                "coverage_collapse": private_row.get("coverage_collapse"),
                "coverage_ratios": private_row.get("coverage_ratios"),
            }
        )

    for row in key_trials:
        register_pair(
            row,
            render_by_id.get(str(row.get("question_id"))),
            context_id=str(row.get("question_id")),
        )
    for row in practice_key_rows:
        register_pair(
            row,
            practice_render_by_id.get(str(row.get("practice_id"))),
            context_id=str(row.get("practice_id")),
        )

    for pair_id, spec in pair_specs.items():
        try:
            source_id_a, source_id_b = spec["source_ids"]
            source_a = canonical_sources[source_id_a]
            source_b = canonical_sources[source_id_b]
            pair_dir = (package / "private" / "matched_pairs" / pair_id).resolve()
            for side, source in (("a", source_a), ("b", source_b)):
                pair_midi = pair_dir / f"{side}.mid"
                canonical_midi = Path(str(source["excerpt_midi_path"])).resolve()
                if (
                    not pair_midi.is_file()
                    or pair_midi.read_bytes() != canonical_midi.read_bytes()
                ):
                    raise ValueError(f"private {side}.mid differs from canonical source")

            expected_audio: dict[str, Mapping[str, Any] | None] = {
                "A": None,
                "B": None,
            }
            expected_wav_hashes: dict[str, str | None] = {"A": None, "B": None}
            if (
                source_id_a not in canonical_note_events
                or source_id_b not in canonical_note_events
            ):
                raise ValueError("independently rebuilt note events are unavailable")
            expected_objective_identity = _triangle_objective_identity(
                canonical_note_events[source_id_a], canonical_note_events[source_id_b]
            )
            expected_coverage_a = float(source_a["excerpt_coverage_ratio"])
            expected_coverage_b = float(source_b["excerpt_coverage_ratio"])
            expected_coverage_collapse = (
                min(expected_coverage_a, expected_coverage_b)
                <= TRIANGLE_COVERAGE_COLLAPSE_MAX_RATIO
                and max(expected_coverage_a, expected_coverage_b)
                >= TRIANGLE_COVERAGE_REFERENCE_MIN_RATIO
            )
            expected_coverage_driven = (
                bool(source_a["source_empty"]) != bool(source_b["source_empty"])
                or expected_coverage_collapse
            )
            expected_coverage_ratios = {
                "a": expected_coverage_a,
                "b": expected_coverage_b,
            }
            if require_wav:
                if (
                    source_id_a not in rerendered_raw_pcm
                    or source_id_b not in rerendered_raw_pcm
                ):
                    raise ValueError("frozen raw PCM is unavailable for matched-pair rebuild")
                with tempfile.TemporaryDirectory(
                    prefix="streammuse-triangle-pair-audit-"
                ) as raw_pair_tmp:
                    rebuilt_a = Path(raw_pair_tmp) / "a.wav"
                    rebuilt_b = Path(raw_pair_tmp) / "b.wav"
                    audio_a, audio_b = _write_common_protected_wavs(
                        [
                            (rebuilt_a, rerendered_raw_pcm[source_id_a]),
                            (rebuilt_b, rerendered_raw_pcm[source_id_b]),
                        ]
                    )
                    private_a = pair_dir / "a.wav"
                    private_b = pair_dir / "b.wav"
                    if (
                        not private_a.is_file()
                        or rebuilt_a.read_bytes() != private_a.read_bytes()
                        or not private_b.is_file()
                        or rebuilt_b.read_bytes() != private_b.read_bytes()
                    ):
                        raise ValueError(
                            "private pair WAV differs byte-for-byte from frozen PCM rebuild"
                        )
                    expected_audio = {"A": audio_a, "B": audio_b}
                    expected_wav_hashes = {
                        "A": file_sha256(rebuilt_a),
                        "B": file_sha256(rebuilt_b),
                    }
                    expected_objective_identity = _triangle_objective_identity(
                        canonical_note_events[source_id_a],
                        canonical_note_events[source_id_b],
                        wav_a=rebuilt_a,
                        wav_b=rebuilt_b,
                    )
                rerendered_pair_count += 1

            for context in spec["contexts"]:
                pattern = context["pattern"]
                presentations = context["presentations"]
                if not isinstance(presentations, list) or len(presentations) != 3:
                    raise ValueError(
                        f"{context['context_id']}: matched-pair presentations are malformed"
                    )
                if context["scored"]:
                    if context["objective_identity"] is not expected_objective_identity:
                        raise ValueError(
                            f"{context['context_id']}: objective_identity differs from "
                            "independently rebuilt note/final-WAV identity"
                        )
                    if context["block"] == "known_different_control" and (
                        expected_objective_identity
                    ):
                        raise ValueError("known-different control is objectively identical")
                    if (
                        context["coverage_driven"] is not expected_coverage_driven
                        or context["coverage_collapse"]
                        is not expected_coverage_collapse
                        or context["coverage_ratios"] != expected_coverage_ratios
                    ):
                        raise ValueError(
                            f"{context['context_id']}: coverage metadata differs from "
                            "independently rebuilt canonical sources"
                        )

                for label, presentation in zip(pattern, presentations, strict=True):
                    if label not in {"A", "B"} or not isinstance(
                        presentation, Mapping
                    ):
                        raise ValueError(
                            f"{context['context_id']}: presentation pattern is malformed"
                        )
                    row_source = context[f"source_{label.lower()}"]
                    expected_midi_hash = row_source.get("excerpt_midi_sha256")
                    raw_midi = Path(str(presentation.get("midi", "")))
                    blind_midi = (package / raw_midi).resolve()
                    if (
                        raw_midi.is_absolute()
                        or not blind_midi.is_relative_to(package / "blind")
                        or presentation.get("midi_sha256") != expected_midi_hash
                        or not blind_midi.is_file()
                        or file_sha256(blind_midi) != expected_midi_hash
                    ):
                        raise ValueError(
                            f"{context['context_id']}: blind MIDI is not canonical-source bound"
                        )
                    if not require_wav:
                        continue
                    expected_wav_hash = expected_wav_hashes[label]
                    if row_source.get("rendered_pair_wav_sha256") != expected_wav_hash:
                        raise ValueError(
                            f"{context['context_id']}: private pair WAV hash differs from rebuild"
                        )
                    raw_wav = Path(str(presentation.get("wav", "")))
                    blind_wav = (package / raw_wav).resolve()
                    if (
                        raw_wav.is_absolute()
                        or not blind_wav.is_relative_to(package / "blind")
                        or presentation.get("wav_sha256") != expected_wav_hash
                        or not blind_wav.is_file()
                        or file_sha256(blind_wav) != expected_wav_hash
                        or dict(presentation.get("audio", {}))
                        != dict(expected_audio[label] or {})
                    ):
                        raise ValueError(
                            f"{context['context_id']}: blind WAV/audio metadata differs from pair rebuild"
                        )
        except Exception as exc:
            errors.append(f"{pair_id}: matched-pair frozen PCM audit failed: {exc}")

    practices_public = public.get("practice_trials")
    practices_key = key.get("practice_trials")
    practices_render = render.get("practice_trials")
    if not all(
        isinstance(rows, list) and len(rows) == TRIANGLE_PRACTICE_COUNT
        for rows in (practices_public, practices_key, practices_render)
    ):
        errors.append("triangle package must contain exactly three practice trials")
    try:
        validate_sitting_ledger(package)
        validate_response_ledger(package)
        progress_path = package / "blind" / "progress_state.json"
        if progress_path.exists():
            if not progress_path.is_file() or _read_json(
                progress_path
            ) != progress_summary(package):
                raise ValueError(
                    "progress_state differs from the exact ledger-derived summary"
                )
    except Exception as exc:
        errors.append(f"blind mutable ledger/progress validation failed: {exc}")
    valid = not errors
    return {
        "schema_version": "streammuse.melody_robustness.listening_triangle_audit.v2",
        "listening_attempt_id": render.get("listening_attempt_id"),
        "retry_lineage": render.get("retry_lineage"),
        "retry_lineage_sha256": render.get("retry_lineage_sha256"),
        "valid": valid,
        "accepted_final": valid and require_wav,
        "development_midi_only": not require_wav,
        "errors": errors,
        "trial_count": len(render_trials),
        "presentation_count": total_presentations,
        "practice_count": len(practices_render) if isinstance(practices_render, list) else 0,
        "frozen_renderer_rerendered_source_count": rerendered_source_count,
        "frozen_renderer_rebuilt_pair_count": rerendered_pair_count,
        "block_counts": dict(block_counts),
        "selection_sha256": render.get("selection_sha256"),
        "campaign_config_sha256": render.get("campaign_config_sha256"),
        "run_schedule_sha256": render.get("run_schedule_sha256"),
        "campaign_binding_sha256": render.get("campaign_binding_sha256"),
        "qualification_result_sha256": render.get("qualification_result_sha256"),
        "campaign_audit_sha256": render.get("campaign_audit_sha256"),
        "control_report_path": str(control_report_path) if control_report_path else None,
        "control_report_sha256": render.get("control_report_sha256"),
        "control_report_base_selection_path": render.get(
            "control_report_base_selection_path"
        ),
        "control_report_base_selection_sha256": render.get(
            "control_report_base_selection_sha256"
        ),
        "control_report_current_selection_sha256": render.get(
            "control_report_current_selection_sha256"
        ),
        "private_key_sha256": render.get("private_key_sha256"),
        "render_manifest_sha256": file_sha256(package / "render_manifest.json"),
        "public_manifest_sha256": render.get("public_manifest_sha256"),
        "player_sha256": render.get("player_sha256"),
        "empty_source_policy": "literal_silence_accepted_only_when_source_empty",
        "blinding_audited": valid,
    }


def audit_triangle_package(args: argparse.Namespace) -> None:
    result = audit_triangle_package_dir(
        args.package_dir, require_wav=not args.allow_midi_only
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["valid"]:
        raise SystemExit(2)


def resume_triangle(args: argparse.Namespace) -> None:
    summary = progress_summary(args.package_dir)
    if args.new_sitting:
        existing = summary.get("sittings", {})
        summary["suggested_sitting_id"] = f"sitting-{len(existing) + 1:03d}"
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def start_triangle_sitting(args: argparse.Namespace) -> None:
    record = append_sitting_event(
        args.package_dir,
        event="start",
        sitting_id=args.sitting_id,
        device=args.device,
        environment=args.environment,
        note=args.note or "",
    )
    print(
        json.dumps(
            {"sitting_event": record, "progress": progress_summary(args.package_dir)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def end_triangle_sitting(args: argparse.Namespace) -> None:
    record = append_sitting_event(
        args.package_dir,
        event="end",
        sitting_id=args.sitting_id,
        note=args.note or "",
        anomalies=args.anomaly or (),
    )
    print(
        json.dumps(
            {"sitting_event": record, "progress": progress_summary(args.package_dir)},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def record_triangle_response(args: argparse.Namespace) -> None:
    result = append_response(
        args.package_dir,
        trial_id=args.trial_id,
        odd_choice=args.odd_choice,
        confidence_1_to_5=args.confidence,
        sitting_id=args.sitting_id,
        difference_tags=args.tag or (),
        note=args.note or "",
        play_counts=args.play_counts,
        response_time_ms=args.response_time_ms,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def import_triangle_responses(args: argparse.Namespace) -> None:
    package = Path(args.package_dir).resolve()
    payload = _read_json(Path(args.responses).resolve())
    render = _read_json(package / "render_manifest.json")
    if payload.get("selection_sha256") != render.get("selection_sha256"):
        raise ValueError("imported response export belongs to a different selection")
    incoming = payload.get("responses")
    if not isinstance(incoming, list) or not incoming:
        raise ValueError("imported response export must contain at least one response")
    sitting_events = payload.get("sitting_events")
    if not isinstance(sitting_events, list):
        raise ValueError("imported response export must include structured sitting_events")
    starts = [row for row in sitting_events if isinstance(row, Mapping) and row.get("event") == "start"]
    ends = [row for row in sitting_events if isinstance(row, Mapping) and row.get("event") == "end"]
    if len(starts) + len(ends) != len(sitting_events):
        raise ValueError("imported sitting_events contain malformed/unknown events")
    _sitting_rows, _sitting_head, sitting_states = validate_sitting_ledger(package)
    for event in starts:
        sitting_id = str(event.get("sitting_id", ""))
        existing_state = sitting_states.get(sitting_id)
        if existing_state is None:
            append_sitting_event(
                package,
                event="start",
                sitting_id=sitting_id,
                device=event.get("device"),
                environment=event.get("environment"),
                note=str(event.get("note", "")),
                recorded_at=event.get("recorded_at"),
            )
        elif any(
            existing_state["start"].get(field) != event.get(field)
            for field in ("sitting_id", "device", "environment", "note", "recorded_at")
        ):
            raise ValueError("imported sitting start differs from durable metadata")
    _sitting_rows, _sitting_head, sitting_states = validate_sitting_ledger(package)
    existing, _head = validate_response_ledger(package)
    compare_fields = (
        "trial_id",
        "odd_choice",
        "confidence_1_to_5",
        "difference_tags",
        "note",
        "play_counts",
        "response_time_ms",
        "sitting_id",
        "submitted_at",
    )
    if len(incoming) < len(existing):
        raise ValueError("imported response export is older than the server ledger")
    for index, existing_row in enumerate(existing):
        candidate = incoming[index]
        if not isinstance(candidate, Mapping) or any(
            candidate.get(field) != existing_row.get(field) for field in compare_fields
        ):
            raise ValueError("imported responses try to edit an already sealed answer")
    appended = 0
    for row in incoming[len(existing) :]:
        if not isinstance(row, Mapping):
            raise ValueError("imported response row must be an object")
        append_response(
            package,
            trial_id=str(row.get("trial_id")),
            odd_choice=str(row.get("odd_choice")),
            confidence_1_to_5=row.get("confidence_1_to_5"),
            sitting_id=str(row.get("sitting_id", "")),
            difference_tags=row.get("difference_tags", ()),
            note=str(row.get("note", "")),
            play_counts=row.get("play_counts", ()),
            response_time_ms=row.get("response_time_ms"),
            submitted_at=str(row.get("submitted_at")) if row.get("submitted_at") else None,
        )
        appended += 1
    for event in ends:
        sitting_id = str(event.get("sitting_id", ""))
        _rows, _head, current_states = validate_sitting_ledger(package)
        state = current_states.get(sitting_id)
        if state is None:
            raise ValueError("imported sitting end has no matching start")
        if state["end"] is None:
            append_sitting_event(
                package,
                event="end",
                sitting_id=sitting_id,
                note=str(event.get("note", "")),
                anomalies=event.get("anomalies", ()),
                recorded_at=event.get("recorded_at"),
            )
        elif any(
            state["end"].get(field) != event.get(field)
            for field in ("sitting_id", "note", "anomalies", "recorded_at")
        ):
            raise ValueError("imported sitting end differs from durable metadata")
    print(
        json.dumps(
            {
                "imported_new_responses": appended,
                "progress": progress_summary(package),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def snapshot_triangle(args: argparse.Namespace) -> None:
    path, payload = create_snapshot(args.package_dir)
    print(
        json.dumps(
            {
                "snapshot_dir": str(path),
                "snapshot_id": payload["snapshot_id"],
                "answered_count": payload["answered_count"],
                "pending_count": payload["pending_count"],
                "ledger_head_hash": payload["ledger_head_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


def unblind_triangle(args: argparse.Namespace) -> None:
    output, payload = unblind_snapshot(args.package_dir, args.snapshot_dir)
    summary_path, summary = summarize_unblinded(args.package_dir, args.snapshot_dir)
    export_index, exported = export_generated_acc(
        args.package_dir, snapshot_dir=args.snapshot_dir
    )
    print(
        json.dumps(
            {
                "unblinded_scores": str(output),
                "summary": str(summary_path),
                "answered_count": payload["answered_count"],
                "collection_status": summary["collection_status"],
                "generated_acc_index": str(export_index),
                "generated_acc_count": len(exported),
                "warning": (
                    "future responses remain accepted but are marked "
                    "post_partial_unblind_exploratory"
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def summarize_triangle(args: argparse.Namespace) -> None:
    path, payload = summarize_unblinded(args.package_dir, args.snapshot_dir)
    print(
        json.dumps(
            {"summary_path": str(path), **payload},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


class _TriangleRequestHandler(http.server.BaseHTTPRequestHandler):
    server_version = "StreamMUSEBlindTriangle/2"

    @property
    def package(self) -> Path:
        return self.server.package_dir  # type: ignore[attr-defined]

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        route = self.path.split("?", 1)[0]
        if route == "/api/progress":
            try:
                self._json(200, progress_summary(self.package))
            except Exception as exc:
                self._json(409, {"error": str(exc)})
            return
        relative = "player.html" if route in {"/", "/player.html"} else route.lstrip("/")
        allowed = self.server.allowed_blind_paths  # type: ignore[attr-defined]
        if relative not in allowed:
            self.send_error(404)
            return
        candidate = (self.package / "blind" / relative).resolve()
        blind_root = (self.package / "blind").resolve()
        if not candidate.is_relative_to(blind_root) or not candidate.is_file():
            self.send_error(404)
            return
        data = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        route = self.path.split("?", 1)[0]
        if route not in {
            "/api/response",
            "/api/sitting/start",
            "/api/sitting/end",
        }:
            self.send_error(404)
            return
        try:
            raw_length = self.headers.get("Content-Length")
            length = int(raw_length or "")
            if length <= 0 or length > 1_000_000:
                raise ValueError("response request body length is invalid")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, Mapping):
                raise ValueError("response request body must be a JSON object")
            if route == "/api/sitting/start":
                allowed = {"sitting_id", "device", "environment", "note", "recorded_at"}
                if set(payload) - allowed:
                    raise ValueError("sitting-start request contains unexpected fields")
                record = append_sitting_event(
                    self.package,
                    event="start",
                    sitting_id=payload.get("sitting_id"),
                    device=payload.get("device"),
                    environment=payload.get("environment"),
                    note=payload.get("note", ""),
                    recorded_at=payload.get("recorded_at"),
                )
                self._json(
                    201,
                    {"sitting_event": record, "progress": progress_summary(self.package)},
                )
                return
            if route == "/api/sitting/end":
                allowed = {"sitting_id", "note", "anomalies", "recorded_at"}
                if set(payload) - allowed:
                    raise ValueError("sitting-end request contains unexpected fields")
                record = append_sitting_event(
                    self.package,
                    event="end",
                    sitting_id=payload.get("sitting_id"),
                    note=payload.get("note", ""),
                    anomalies=payload.get("anomalies", ()),
                    recorded_at=payload.get("recorded_at"),
                )
                self._json(
                    201,
                    {"sitting_event": record, "progress": progress_summary(self.package)},
                )
                return
            allowed = {
                "trial_id",
                "odd_choice",
                "confidence_1_to_5",
                "difference_tags",
                "note",
                "play_counts",
                "response_time_ms",
                "sitting_id",
                "submitted_at",
            }
            if set(payload) - allowed:
                raise ValueError("response request contains unexpected fields")
            result = append_response(
                self.package,
                trial_id=payload.get("trial_id"),
                odd_choice=payload.get("odd_choice"),
                confidence_1_to_5=payload.get("confidence_1_to_5"),
                sitting_id=payload.get("sitting_id"),
                difference_tags=payload.get("difference_tags", ()),
                note=payload.get("note", ""),
                play_counts=payload.get("play_counts", ()),
                response_time_ms=payload.get("response_time_ms"),
                submitted_at=payload.get("submitted_at"),
            )
            self._json(201, result)
        except Exception as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:
        if not getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            super().log_message(format, *args)


def make_triangle_server(
    package_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    quiet: bool = False,
    require_final: bool = True,
) -> http.server.ThreadingHTTPServer:
    package = Path(package_dir).resolve()
    audit = _read_json(package / "package_audit.json")
    if audit.get("valid") is not True or (
        require_final and audit.get("accepted_final") is not True
    ):
        raise ValueError("serve-triangle requires a valid accepted-final WAV package")
    # Fail before binding a port if the ledger or unblind boundary is corrupt.
    validate_sitting_ledger(package)
    validate_response_ledger(package)
    try:
        _required, _directories, served_paths = _triangle_expected_blind_paths(
            _read_json(package / "render_manifest.json")
        )
    except Exception as exc:
        raise ValueError(f"serve-triangle blind path contract is invalid: {exc}") from exc
    server = http.server.ThreadingHTTPServer((host, port), _TriangleRequestHandler)
    server.package_dir = package  # type: ignore[attr-defined]
    server.quiet = quiet  # type: ignore[attr-defined]
    server.allowed_blind_paths = served_paths  # type: ignore[attr-defined]
    return server


def serve_triangle(args: argparse.Namespace) -> None:
    server = make_triangle_server(
        args.package_dir,
        host=args.host,
        port=args.port,
        quiet=args.quiet,
        require_final=True,
    )
    host, port = server.server_address[:2]
    print(
        json.dumps(
            {
                "url": f"http://{host}:{port}/",
                "package_dir": str(Path(args.package_dir).resolve()),
                "persistence": "per-response server hash-chain + atomic progress",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-selection")
    freeze.add_argument("--input-manifest", required=True)
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--perturb-seed", type=int, default=SEEDS["perturb"][0])
    freeze.add_argument("--sample-seed", type=int, default=SEEDS["sample"][0])
    freeze.add_argument("--blind-order-seed", type=int, default=SEEDS["blind_order"])
    freeze.add_argument("--excerpt-start-beat", type=int, default=0)
    freeze.add_argument("--excerpt-starts-json")
    freeze.set_defaults(func=freeze_selection)
    build = sub.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--config-sha256", required=True)
    build.add_argument("--selection", required=True)
    build.add_argument("--schedule", required=True)
    build.add_argument("--schedule-sha256", required=True)
    build.add_argument("--output-root", required=True)
    build.add_argument("--controls-root", required=True)
    build.add_argument("--package-dir", required=True)
    build.add_argument("--soundfont")
    build.add_argument("--fluidsynth", default="fluidsynth")
    build.add_argument("--sample-rate", type=int, default=RENDER_SAMPLE_RATE)
    build.add_argument("--gain", type=float, default=FIXED_SYNTH_GAIN)
    build.add_argument("--midi-only", action="store_true", help="development only; not an accepted final package")
    build.set_defaults(func=build_package)
    audit = sub.add_parser("audit")
    audit.add_argument("--package-dir", required=True)
    audit.add_argument("--allow-midi-only", action="store_true")
    audit.set_defaults(func=audit_package)
    seal = sub.add_parser("seal-scores")
    seal.add_argument("--package-dir", required=True)
    seal.set_defaults(func=seal_scores)
    reveal = sub.add_parser("unblind")
    reveal.add_argument("--package-dir", required=True)
    reveal.set_defaults(func=unblind)

    triangle_freeze = sub.add_parser(
        "freeze-triangle-selection",
        help="freeze the exact 95-trial acc-only triangle selection",
    )
    triangle_freeze.add_argument("--input-manifest", required=True)
    triangle_freeze.add_argument("--output", required=True)
    triangle_freeze.add_argument(
        "--blind-order-seed", type=int, default=SEEDS["blind_order"]
    )
    triangle_freeze.add_argument("--excerpt-start-beat", type=int, default=16)
    triangle_freeze.add_argument("--excerpt-starts-json")
    triangle_freeze.set_defaults(func=freeze_triangle_selection)

    triangle_retry = sub.add_parser(
        "derive-triangle-retry",
        help=(
            "derive attempt-N+1 only from a full immutable qc_status=fail "
            "triangle snapshot"
        ),
    )
    triangle_retry.add_argument("--failed-package", required=True)
    triangle_retry.add_argument("--failed-snapshot", required=True)
    triangle_retry.add_argument("--output", required=True)
    triangle_retry.set_defaults(func=derive_triangle_retry)

    renderer_freeze = sub.add_parser(
        "freeze-triangle-renderer",
        help="hash-pin FluidSynth, runtime libraries, soundfont and render policy",
    )
    renderer_freeze.add_argument("--output", required=True)
    renderer_freeze.add_argument("--soundfont", required=True)
    renderer_freeze.add_argument("--fluidsynth", default="fluidsynth")
    renderer_freeze.add_argument("--fluidsynth-root")
    renderer_freeze.add_argument("--fluidsynth-lib-dir", action="append")
    renderer_freeze.set_defaults(func=freeze_triangle_renderer)

    triangle_build = sub.add_parser(
        "build-triangle", help="build the frozen acc-only triangle package"
    )
    triangle_build.add_argument("--config", required=True)
    triangle_build.add_argument("--config-sha256", required=True)
    triangle_build.add_argument("--selection", required=True)
    triangle_build.add_argument("--schedule", required=True)
    triangle_build.add_argument("--schedule-sha256", required=True)
    triangle_build.add_argument("--output-root", required=True)
    triangle_build.add_argument("--campaign-audit", required=True)
    triangle_build.add_argument("--campaign-audit-sha256", required=True)
    triangle_build.add_argument("--control-report", required=True)
    triangle_build.add_argument("--control-report-sha256", required=True)
    triangle_build.add_argument("--package-dir", required=True)
    triangle_build.add_argument("--soundfont")
    triangle_build.add_argument("--fluidsynth", default="fluidsynth")
    triangle_build.add_argument(
        "--fluidsynth-root",
        help="root of a locally extracted FluidSynth package tree",
    )
    triangle_build.add_argument(
        "--fluidsynth-lib-dir",
        action="append",
        help="additional runtime library directory; may be repeated",
    )
    triangle_build.add_argument(
        "--sample-rate", type=int, default=TRIANGLE_RENDER_SAMPLE_RATE
    )
    triangle_build.add_argument("--gain", type=float, default=TRIANGLE_SYNTH_GAIN)
    triangle_build.add_argument(
        "--midi-only",
        action="store_true",
        help="development only; never accepted as a final listening package",
    )
    triangle_build.set_defaults(func=build_triangle_package)

    triangle_audit = sub.add_parser("audit-triangle")
    triangle_audit.add_argument("--package-dir", required=True)
    triangle_audit.add_argument("--allow-midi-only", action="store_true")
    triangle_audit.set_defaults(func=audit_triangle_package)

    triangle_resume = sub.add_parser("resume-triangle")
    triangle_resume.add_argument("--package-dir", required=True)
    triangle_resume.add_argument("--new-sitting", action="store_true")
    triangle_resume.set_defaults(func=resume_triangle)

    triangle_sitting_start = sub.add_parser("start-triangle-sitting")
    triangle_sitting_start.add_argument("--package-dir", required=True)
    triangle_sitting_start.add_argument("--sitting-id", required=True)
    triangle_sitting_start.add_argument("--device", required=True)
    triangle_sitting_start.add_argument("--environment", required=True)
    triangle_sitting_start.add_argument("--note")
    triangle_sitting_start.set_defaults(func=start_triangle_sitting)

    triangle_sitting_end = sub.add_parser("end-triangle-sitting")
    triangle_sitting_end.add_argument("--package-dir", required=True)
    triangle_sitting_end.add_argument("--sitting-id", required=True)
    triangle_sitting_end.add_argument("--anomaly", action="append")
    triangle_sitting_end.add_argument("--note")
    triangle_sitting_end.set_defaults(func=end_triangle_sitting)

    triangle_serve = sub.add_parser(
        "serve-triangle",
        help="serve blind assets locally and persist every response immediately",
    )
    triangle_serve.add_argument("--package-dir", required=True)
    triangle_serve.add_argument("--host", default="127.0.0.1")
    triangle_serve.add_argument("--port", type=int, default=8765)
    triangle_serve.add_argument("--quiet", action="store_true")
    triangle_serve.set_defaults(func=serve_triangle)

    triangle_record = sub.add_parser("record-triangle-response")
    triangle_record.add_argument("--package-dir", required=True)
    triangle_record.add_argument("--trial-id")
    triangle_record.add_argument(
        "--odd-choice",
        required=True,
        choices=["1", "2", "3", "no_difference"],
    )
    triangle_record.add_argument("--confidence", required=True, type=int, choices=range(1, 6))
    triangle_record.add_argument("--sitting-id", required=True)
    triangle_record.add_argument("--tag", action="append")
    triangle_record.add_argument("--note")
    triangle_record.add_argument(
        "--play-counts", type=int, nargs=3, default=[1, 1, 1]
    )
    triangle_record.add_argument("--response-time-ms", type=int, default=0)
    triangle_record.set_defaults(func=record_triangle_response)

    triangle_import = sub.add_parser("import-triangle-responses")
    triangle_import.add_argument("--package-dir", required=True)
    triangle_import.add_argument("--responses", required=True)
    triangle_import.set_defaults(func=import_triangle_responses)

    triangle_snapshot = sub.add_parser("snapshot-triangle")
    triangle_snapshot.add_argument("--package-dir", required=True)
    triangle_snapshot.set_defaults(func=snapshot_triangle)
    triangle_seal = sub.add_parser(
        "seal-triangle-scores",
        help="alias of snapshot-triangle; accepts any answered count from 1 to 95",
    )
    triangle_seal.add_argument("--package-dir", required=True)
    triangle_seal.set_defaults(func=snapshot_triangle)

    triangle_unblind = sub.add_parser("unblind-triangle")
    triangle_unblind.add_argument("--package-dir", required=True)
    triangle_unblind.add_argument("--snapshot-dir", required=True)
    triangle_unblind.set_defaults(func=unblind_triangle)

    triangle_summary = sub.add_parser("summarize-triangle")
    triangle_summary.add_argument("--package-dir", required=True)
    triangle_summary.add_argument("--snapshot-dir", required=True)
    triangle_summary.set_defaults(func=summarize_triangle)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
