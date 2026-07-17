#!/usr/bin/env python3
"""Build the frozen melody-perturbation input matrix.

The experiment deliberately uses a raw-MIDI canonical note table.  Matched,
non-drum notes in the model pitch range (21..108) are perturbable; every other
event is preserved verbatim.  Selection and candidate ordering use a keyed
BLAKE2b PRF, so results do not depend on process-global RNG state or
``PYTHONHASHSEED``.

The default condition table produces eight inputs per source song::

    sham
    pitch/onset/both x two perturbation seeds
    high x the first perturbation seed

For the five frozen source songs this is exactly 40 distinct inputs.  A JSON
sidecar contains enough final-state information to replay each output without
running selection or collision logic again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mido
import numpy as np


SCHEMA_VERSION = "streammuse.melody_perturbation.v1"
TOOL_VERSION = "1.0.0"
MODEL_TICKS_PER_BEAT = 4
DEFAULT_PERTURB_SEEDS = (2026071001, 2026071002)
DEFAULT_HIGH_PSEED = 2026071001
MAX_COLLISION_ATTEMPTS = 3


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    pitch_probability: float
    onset_probability: float
    seeds: tuple[int | None, ...]


def default_conditions(
    perturb_seeds: Sequence[int] = DEFAULT_PERTURB_SEEDS,
    high_pseed: int = DEFAULT_HIGH_PSEED,
) -> tuple[ConditionSpec, ...]:
    seeds = tuple(int(seed) for seed in perturb_seeds)
    if len(seeds) != 2 or len(set(seeds)) != 2:
        raise ValueError("the frozen matrix requires exactly two distinct perturbation seeds")
    if int(high_pseed) not in seeds:
        raise ValueError("high_pseed must be one of perturb_seeds so dose masks are nested")
    return (
        ConditionSpec("sham", 0.0, 0.0, (None,)),
        ConditionSpec("pitch", 0.05, 0.0, seeds),
        ConditionSpec("onset", 0.0, 0.15, seeds),
        ConditionSpec("both", 0.05, 0.15, seeds),
        ConditionSpec("high", 0.20, 0.40, (int(high_pseed),)),
    )


@dataclass(frozen=True)
class NoteState:
    pitch: int
    start_tick: int
    end_tick: int

    @property
    def duration_ticks(self) -> int:
        return self.end_tick - self.start_tick


@dataclass(frozen=True)
class CanonicalNote:
    source_note_id: str
    track_index: int
    track_note_ordinal: int
    channel: int
    program: int
    velocity: int
    state: NoteState
    on_event_index: int
    off_event_index: int


@dataclass
class ParsedMidi:
    path: Path
    source_sha256: str
    midi: mido.MidiFile
    notes: list[CanonicalNote]
    stats: dict[str, int]
    dropped_note_event_indices: dict[int, set[int]]


@dataclass(frozen=True)
class LatentDecision:
    note_id: str
    pitch_score: float
    onset_score: float
    pitch_candidates: tuple[int, ...]
    onset_candidates: tuple[int, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _prf_digest(
    *,
    source_sha256: str,
    perturb_seed: int,
    note_id: str,
    decision: str,
    attempt: int = 0,
    candidate: int | None = None,
) -> bytes:
    payload = {
        "attempt": int(attempt),
        "candidate": candidate,
        "decision": str(decision),
        "note_id": str(note_id),
        "perturb_seed": int(perturb_seed),
        "schema_version": SCHEMA_VERSION,
        "source_sha256": str(source_sha256),
    }
    return hashlib.blake2b(canonical_json_bytes(payload), digest_size=16, person=b"SMusePerturbV1").digest()


def keyed_score(
    *, source_sha256: str, perturb_seed: int, note_id: str, decision: str
) -> float:
    """Return a stable value in [0, 1), independent of Python's hash seed."""

    value = int.from_bytes(
        _prf_digest(
            source_sha256=source_sha256,
            perturb_seed=perturb_seed,
            note_id=note_id,
            decision=decision,
        ),
        "big",
    )
    return value / float(1 << 128)


def _keyed_candidate_order(
    candidates: Iterable[int],
    *,
    source_sha256: str,
    perturb_seed: int,
    note_id: str,
    decision: str,
) -> tuple[int, ...]:
    unique = sorted(set(int(candidate) for candidate in candidates))
    return tuple(
        sorted(
            unique,
            key=lambda candidate: _prf_digest(
                source_sha256=source_sha256,
                perturb_seed=perturb_seed,
                note_id=note_id,
                decision=decision,
                candidate=candidate,
            ),
        )
    )


def _message_is_note_on(message: mido.Message) -> bool:
    return message.type == "note_on" and int(message.velocity) > 0


def _message_is_note_off(message: mido.Message) -> bool:
    return message.type == "note_off" or (
        message.type == "note_on" and int(message.velocity) == 0
    )


def parse_canonical_midi(path: str | Path, *, require_grid: bool = True) -> ParsedMidi:
    """Parse the same matched-note universe that PrettyMIDI exposes to the model.

    PrettyMIDI closes all earlier open notes of a given (track, channel, pitch)
    at one note-off, while retaining an onset at the exact off tick when an
    older note was also closed.  Mirroring that detail is necessary for the
    five fixtures, which contain overlapping/retriggered notes.
    """

    path = Path(path).resolve()
    source_sha = sha256_file(path)
    midi = mido.MidiFile(str(path), clip=True)
    if midi.ticks_per_beat <= 0 or midi.ticks_per_beat % MODEL_TICKS_PER_BEAT != 0:
        raise ValueError(
            f"{path}: PPQ {midi.ticks_per_beat} is not divisible by {MODEL_TICKS_PER_BEAT}"
        )
    raw_step = midi.ticks_per_beat // MODEL_TICKS_PER_BEAT

    notes: list[CanonicalNote] = []
    dropped_note_event_indices: dict[int, set[int]] = {}
    stats = {
        "matched_model_visible": 0,
        "matched_drum": 0,
        "matched_out_of_range": 0,
        "dangling_note_on": 0,
        "spurious_note_off": 0,
        "off_grid_model_visible": 0,
    }

    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        programs = [0] * 16
        note_on_ordinal = 0
        # PrettyMIDI stores all opens for the same key and one off can close many.
        open_notes: dict[tuple[int, int], list[dict[str, int]]] = {}
        for event_index, message in enumerate(track):
            absolute_tick += int(message.time)
            if message.type == "program_change":
                programs[int(message.channel)] = int(message.program)
                continue
            if _message_is_note_on(message):
                note_on_ordinal += 1
                key = (int(message.channel), int(message.note))
                open_notes.setdefault(key, []).append(
                    {
                        "start_tick": absolute_tick,
                        "velocity": int(message.velocity),
                        "program": programs[int(message.channel)],
                        "event_index": event_index,
                        "ordinal": note_on_ordinal,
                    }
                )
                continue
            if not _message_is_note_off(message):
                continue
            key = (int(message.channel), int(message.note))
            opens = open_notes.get(key)
            if not opens:
                stats["spurious_note_off"] += 1
                dropped_note_event_indices.setdefault(track_index, set()).add(event_index)
                continue
            to_close = [record for record in opens if record["start_tick"] != absolute_tick]
            to_keep = [record for record in opens if record["start_tick"] == absolute_tick]
            for record in to_close:
                channel = int(message.channel)
                pitch = int(message.note)
                if channel == 9:
                    stats["matched_drum"] += 1
                    continue
                if not 21 <= pitch <= 108:
                    stats["matched_out_of_range"] += 1
                    continue
                start_tick = int(record["start_tick"])
                if start_tick % raw_step or absolute_tick % raw_step:
                    stats["off_grid_model_visible"] += 1
                    if require_grid:
                        raise ValueError(
                            f"{path}: model-visible note at raw ticks "
                            f"[{start_tick},{absolute_tick}) is off the PPQ/4 grid ({raw_step})"
                        )
                model_start = start_tick // raw_step
                model_end = absolute_tick // raw_step
                if model_end <= model_start:
                    raise ValueError(f"{path}: non-positive matched note duration at track {track_index}")
                ordinal = int(record["ordinal"])
                note_id_payload = f"{source_sha}:track={track_index}:note={ordinal}".encode("utf-8")
                note_id = hashlib.sha256(note_id_payload).hexdigest()[:24]
                notes.append(
                    CanonicalNote(
                        source_note_id=note_id,
                        track_index=track_index,
                        track_note_ordinal=ordinal,
                        channel=channel,
                        program=int(record["program"]),
                        velocity=int(record["velocity"]),
                        state=NoteState(pitch=pitch, start_tick=model_start, end_tick=model_end),
                        on_event_index=int(record["event_index"]),
                        off_event_index=event_index,
                    )
                )
                stats["matched_model_visible"] += 1
            if not to_close:
                dropped = dropped_note_event_indices.setdefault(track_index, set())
                dropped.add(event_index)
                dropped.update(int(record["event_index"]) for record in to_keep)
            if to_close and to_keep:
                open_notes[key] = to_keep
            else:
                open_notes.pop(key, None)
        dangling = [record for records in open_notes.values() for record in records]
        stats["dangling_note_on"] += len(dangling)
        dropped_note_event_indices.setdefault(track_index, set()).update(
            int(record["event_index"]) for record in dangling
        )

    notes.sort(key=lambda note: (note.state.start_tick, note.track_index, note.track_note_ordinal))
    if len({note.source_note_id for note in notes}) != len(notes):
        raise AssertionError("source_note_id collision")
    return ParsedMidi(
        path=path,
        source_sha256=source_sha,
        midi=midi,
        notes=notes,
        stats=stats,
        dropped_note_event_indices=dropped_note_event_indices,
    )


def build_latent_decisions(parsed: ParsedMidi, perturb_seed: int) -> dict[str, LatentDecision]:
    decisions: dict[str, LatentDecision] = {}
    for note in parsed.notes:
        pitch_candidates = _keyed_candidate_order(
            (offset for offset in (-2, -1, 1, 2) if 21 <= note.state.pitch + offset <= 108),
            source_sha256=parsed.source_sha256,
            perturb_seed=perturb_seed,
            note_id=note.source_note_id,
            decision="pitch.offset",
        )
        onset_candidates = _keyed_candidate_order(
            (-1, 1),
            source_sha256=parsed.source_sha256,
            perturb_seed=perturb_seed,
            note_id=note.source_note_id,
            decision="onset.delta",
        )
        decisions[note.source_note_id] = LatentDecision(
            note_id=note.source_note_id,
            pitch_score=keyed_score(
                source_sha256=parsed.source_sha256,
                perturb_seed=perturb_seed,
                note_id=note.source_note_id,
                decision="pitch.selection",
            ),
            onset_score=keyed_score(
                source_sha256=parsed.source_sha256,
                perturb_seed=perturb_seed,
                note_id=note.source_note_id,
                decision="onset.selection",
            ),
            pitch_candidates=pitch_candidates,
            onset_candidates=onset_candidates,
        )
    return decisions


def _overlap(first: NoteState, second: NoteState) -> bool:
    return first.start_tick < second.end_tick and second.start_tick < first.end_tick


def _new_same_pitch_collision(
    *,
    note_id: str,
    candidate: NoteState,
    states: Mapping[str, NoteState],
    original_states: Mapping[str, NoteState],
) -> list[str]:
    collisions: list[str] = []
    original = original_states[note_id]
    for other_id, other_state in states.items():
        if other_id == note_id:
            continue
        if candidate.pitch != other_state.pitch or not _overlap(candidate, other_state):
            continue
        other_original = original_states[other_id]
        was_same_pitch_overlap = (
            original.pitch == other_original.pitch and _overlap(original, other_original)
        )
        if not was_same_pitch_overlap:
            collisions.append(other_id)
    return sorted(collisions)


def _unserializable_same_pitch_overlap(
    *,
    note_id: str,
    candidate: NoteState,
    states: Mapping[str, NoteState],
    notes_by_id: Mapping[str, CanonicalNote],
) -> list[str]:
    """Reject spans a raw MIDI note-off cannot represent unambiguously."""
    note = notes_by_id[note_id]
    invalid = []
    for other_id, other_state in states.items():
        if other_id == note_id:
            continue
        other = notes_by_id[other_id]
        if (note.track_index, note.channel) != (other.track_index, other.channel):
            continue
        if candidate.pitch != other_state.pitch or not _overlap(candidate, other_state):
            continue
        # PrettyMIDI closes every open note for one track/channel/pitch at the
        # first off event. Overlapping spans are representable only when they
        # share that off tick.
        if candidate.end_tick != other_state.end_tick:
            invalid.append(other_id)
    return sorted(invalid)


def _candidate_pairs(
    decision: LatentDecision, *, select_pitch: bool, select_onset: bool
) -> list[tuple[int, int]]:
    pitch_candidates = decision.pitch_candidates if select_pitch else (0,)
    onset_candidates = decision.onset_candidates if select_onset else (0,)
    # Attempt 0 is exactly the shared latent proposal.  Later attempts cycle
    # through the independently keyed component orders.  Arms may diverge only
    # after collision handling; such effective mismatches are explicitly logged.
    attempts = max(len(pitch_candidates), len(onset_candidates))
    return [
        (
            pitch_candidates[index % len(pitch_candidates)],
            onset_candidates[index % len(onset_candidates)],
        )
        for index in range(min(MAX_COLLISION_ATTEMPTS, attempts))
    ]


def apply_condition(
    parsed: ParsedMidi,
    *,
    condition: ConditionSpec,
    perturb_seed: int | None,
    decisions: Mapping[str, LatentDecision] | None,
) -> tuple[dict[str, NoteState], list[dict[str, Any]], dict[str, Any]]:
    if condition.name == "sham":
        if perturb_seed is not None or decisions is not None:
            raise ValueError("sham must not have a perturbation seed")
    elif perturb_seed is None or decisions is None:
        raise ValueError(f"{condition.name} requires a perturbation seed and decisions")

    original_states = {note.source_note_id: note.state for note in parsed.notes}
    notes_by_id = {note.source_note_id: note for note in parsed.notes}
    states = dict(original_states)
    note_records: list[dict[str, Any]] = []
    selected_pitch = selected_onset = effective_pitch = effective_onset = giveup = 0

    for note in parsed.notes:
        original = note.state
        if condition.name == "sham":
            pitch_score = onset_score = None
            pitch_candidates: tuple[int, ...] = ()
            onset_candidates: tuple[int, ...] = ()
            choose_pitch = choose_onset = False
            attempts: list[dict[str, Any]] = []
            final = original
            giveup_reason = None
            no_op_reason = None
        else:
            decision = decisions[note.source_note_id]
            pitch_score = decision.pitch_score
            onset_score = decision.onset_score
            pitch_candidates = decision.pitch_candidates
            onset_candidates = decision.onset_candidates
            choose_pitch = pitch_score < condition.pitch_probability
            choose_onset = onset_score < condition.onset_probability
            selected_pitch += int(choose_pitch)
            selected_onset += int(choose_onset)
            attempts = []
            final = original
            giveup_reason = None
            no_op_reason = None
            if choose_pitch or choose_onset:
                accepted = False
                for attempt_index, (pitch_offset, onset_delta) in enumerate(
                    _candidate_pairs(
                        decision, select_pitch=choose_pitch, select_onset=choose_onset
                    )
                ):
                    effective_delta = max(int(onset_delta), -original.start_tick)
                    candidate = NoteState(
                        pitch=original.pitch + int(pitch_offset),
                        start_tick=original.start_tick + effective_delta,
                        end_tick=original.end_tick + effective_delta,
                    )
                    collision_ids = _new_same_pitch_collision(
                        note_id=note.source_note_id,
                        candidate=candidate,
                        states=states,
                        original_states=original_states,
                    )
                    serialization_ids = _unserializable_same_pitch_overlap(
                        note_id=note.source_note_id,
                        candidate=candidate,
                        states=states,
                        notes_by_id=notes_by_id,
                    )
                    attempts.append(
                        {
                            "attempt": attempt_index,
                            "collision_note_ids": collision_ids,
                            "midi_serialization_note_ids": serialization_ids,
                            "effective_onset_delta": effective_delta,
                            "pitch_offset": int(pitch_offset),
                            "requested_onset_delta": int(onset_delta),
                            "state": asdict(candidate),
                        }
                    )
                    if not collision_ids and not serialization_ids:
                        final = candidate
                        accepted = True
                        if choose_onset and effective_delta == 0:
                            no_op_reason = "onset_boundary_clamp"
                        break
                if not accepted:
                    giveup += 1
                    giveup_reason = "collision_or_midi_serialization_after_3_attempts"
                    final = original
            states[note.source_note_id] = final
            effective_pitch += int(final.pitch != original.pitch)
            effective_onset += int(final.start_tick != original.start_tick)

        note_records.append(
            {
                "candidate_order": {
                    "onset_deltas": list(onset_candidates),
                    "pitch_offsets": list(pitch_candidates),
                },
                "effective_edit": {
                    "onset": final.start_tick != original.start_tick,
                    "pitch": final.pitch != original.pitch,
                },
                "final": asdict(final),
                "giveup_reason": giveup_reason,
                "latent_proposal": attempts[0] if attempts else None,
                "no_op_reason": no_op_reason,
                "original": asdict(original),
                "program": note.program,
                "selection": {
                    "onset_probability": condition.onset_probability,
                    "onset_score": onset_score,
                    "onset_selected": choose_onset,
                    "pitch_probability": condition.pitch_probability,
                    "pitch_score": pitch_score,
                    "pitch_selected": choose_pitch,
                },
                "source_note_id": note.source_note_id,
                "track_index": note.track_index,
                "track_note_ordinal": note.track_note_ordinal,
                "velocity": note.velocity,
                "collision_attempts": attempts,
            }
        )

    denominator = len(parsed.notes)
    counts = {
        "effective_onset": effective_onset,
        "effective_pitch": effective_pitch,
        "giveup": giveup,
        "model_visible": denominator,
        "selected_onset": selected_onset,
        "selected_pitch": selected_pitch,
    }
    rates = {
        key: (value / denominator if denominator else 0.0)
        for key, value in counts.items()
        if key != "model_visible"
    }
    return states, note_records, {"counts": counts, "rates": rates}


def _raw_note_priority(message: mido.Message) -> int:
    if _message_is_note_off(message):
        return 1
    if _message_is_note_on(message):
        return 2
    return 0


def write_states_to_midi(
    parsed: ParsedMidi,
    states: Mapping[str, NoteState],
    output_path: str | Path,
) -> None:
    """Rewrite only canonical note events; preserve every other raw MIDI event."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_step = parsed.midi.ticks_per_beat // MODEL_TICKS_PER_BEAT
    by_track: dict[int, list[CanonicalNote]] = {}
    for note in parsed.notes:
        by_track.setdefault(note.track_index, []).append(note)

    output = mido.MidiFile(type=parsed.midi.type, ticks_per_beat=parsed.midi.ticks_per_beat)
    for track_index, source_track in enumerate(parsed.midi.tracks):
        canonical = by_track.get(track_index, [])
        removed_event_indices = {
            event_index
            for note in canonical
            for event_index in (note.on_event_index, note.off_event_index)
        }
        removed_event_indices.update(parsed.dropped_note_event_indices.get(track_index, set()))
        events: list[tuple[int, int, int, str, mido.Message]] = []
        absolute_tick = 0
        original_eot_tick = 0
        for event_index, message in enumerate(source_track):
            absolute_tick += int(message.time)
            if message.type == "end_of_track":
                original_eot_tick = max(original_eot_tick, absolute_tick)
                continue
            if event_index in removed_event_indices:
                continue
            events.append(
                (
                    absolute_tick,
                    _raw_note_priority(message),
                    event_index,
                    "raw",
                    message.copy(time=0),
                )
            )
        for note in canonical:
            state = states[note.source_note_id]
            start_raw = state.start_tick * raw_step
            end_raw = state.end_tick * raw_step
            events.append(
                (
                    start_raw,
                    2,
                    note.on_event_index,
                    note.source_note_id,
                    mido.Message(
                        "note_on",
                        channel=note.channel,
                        note=state.pitch,
                        velocity=note.velocity,
                        time=0,
                    ),
                )
            )
            events.append(
                (
                    end_raw,
                    1,
                    note.off_event_index,
                    note.source_note_id,
                    mido.Message(
                        "note_off",
                        channel=note.channel,
                        note=state.pitch,
                        velocity=0,
                        time=0,
                    ),
                )
            )
        events.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        output_track = mido.MidiTrack()
        previous_tick = 0
        for tick, _priority, _index, _identity, message in events:
            if tick < previous_tick:
                raise AssertionError("event ordering regression")
            output_track.append(message.copy(time=tick - previous_tick))
            previous_tick = tick
        eot_tick = max(original_eot_tick, previous_tick)
        output_track.append(mido.MetaMessage("end_of_track", time=eot_tick - previous_tick))
        output.tracks.append(output_track)
    output.save(str(output_path))


def states_to_roll(states: Iterable[NoteState], *, horizon: int | None = None) -> np.ndarray:
    states = list(states)
    max_end = max((state.end_tick for state in states), default=0)
    if horizon is None:
        horizon = max_end
    if max_end > horizon:
        raise ValueError(f"note end {max_end} exceeds roll horizon {horizon}")
    roll = np.zeros((2, 88, int(horizon)), dtype=np.uint8)
    for state in states:
        pitch_index = state.pitch - 21
        roll[0, pitch_index, state.start_tick : state.end_tick] = 1
        roll[1, pitch_index, state.start_tick] = 1
    return roll


def _roll_diff_counts(
    originals: Mapping[str, NoteState], finals: Mapping[str, NoteState]
) -> dict[str, int]:
    horizon = max(
        max((state.end_tick for state in originals.values()), default=0),
        max((state.end_tick for state in finals.values()), default=0),
    )
    original_roll = states_to_roll(originals.values(), horizon=horizon)
    final_roll = states_to_roll(finals.values(), horizon=horizon)
    return {
        "onset_cells": int(np.count_nonzero(original_roll[1] != final_roll[1])),
        "sustain_cells": int(np.count_nonzero(original_roll[0] != final_roll[0])),
    }


def _time_signature_numerator(midi: mido.MidiFile) -> int:
    for track in midi.tracks:
        for message in track:
            if message.type == "time_signature":
                return int(message.numerator)
    return 4


def _model_max_tick(path: Path) -> int:
    parsed = parse_canonical_midi(path)
    return max((note.state.end_tick for note in parsed.notes), default=0)


def _resource(path: Path, manifest_parent: Path, *, allow_missing: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(manifest_parent.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact {resolved} must live under manifest root {manifest_parent}") from exc
    return {
        "path": relative,
        "sha256": None if allow_missing and not resolved.exists() else sha256_file(resolved),
    }


def replay_sidecar(
    source_midi: str | Path,
    sidecar_path: str | Path,
    output_path: str | Path,
) -> Path:
    source_midi = Path(source_midi)
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    parsed = parse_canonical_midi(source_midi)
    if parsed.source_sha256 != sidecar["source_midi"]["sha256"]:
        raise ValueError("sidecar source SHA256 does not match source MIDI")
    by_id = {note.source_note_id: note for note in parsed.notes}
    records = sidecar["notes"]
    if set(by_id) != {record["source_note_id"] for record in records}:
        raise ValueError("sidecar canonical note set does not match source MIDI")
    states = {
        record["source_note_id"]: NoteState(**record["final"])
        for record in records
    }
    write_states_to_midi(parsed, states, output_path)
    actual_sha = sha256_file(Path(output_path))
    expected_sha = sidecar["output_midi"]["sha256"]
    if actual_sha != expected_sha:
        raise AssertionError(f"replay byte mismatch: expected {expected_sha}, got {actual_sha}")
    return Path(output_path)


def verify_sidecar(source_midi: str | Path, sidecar_path: str | Path) -> None:
    with tempfile.TemporaryDirectory(prefix="streammuse-perturb-replay-") as tmp:
        replay_sidecar(source_midi, sidecar_path, Path(tmp) / "replayed.mid")


def _input_stem(song: str, condition: str, perturb_seed: int | None) -> str:
    if perturb_seed is None:
        return f"{song}__{condition}"
    return f"{song}__{condition}__ps{perturb_seed}"


def _pairing_mismatch(
    arm_states: Mapping[str, Mapping[str, NoteState]],
) -> dict[str, Any] | None:
    if not {"pitch", "onset", "both"}.issubset(arm_states):
        return None
    pitch_mismatch: list[str] = []
    onset_mismatch: list[str] = []
    for note_id, both_state in arm_states["both"].items():
        if both_state.pitch != arm_states["pitch"][note_id].pitch:
            pitch_mismatch.append(note_id)
        onset_state = arm_states["onset"][note_id]
        if (both_state.start_tick, both_state.end_tick) != (
            onset_state.start_tick,
            onset_state.end_tick,
        ):
            onset_mismatch.append(note_id)
    mismatch_count = len(set(pitch_mismatch) | set(onset_mismatch))
    return {
        "collision_interaction": mismatch_count > 0,
        "effective_mismatch_note_count": mismatch_count,
        "effective_onset_mismatch_count": len(onset_mismatch),
        "effective_onset_mismatch_note_ids": onset_mismatch,
        "effective_pitch_mismatch_count": len(pitch_mismatch),
        "effective_pitch_mismatch_note_ids": pitch_mismatch,
        "latent_pairing_verified": True,
    }


def generate_campaign(
    *,
    mel_dir: str | Path,
    acc_dir: str | Path,
    output_root: str | Path,
    manifest_path: str | Path | None = None,
    perturb_seeds: Sequence[int] = DEFAULT_PERTURB_SEEDS,
    high_pseed: int = DEFAULT_HIGH_PSEED,
    expected_song_count: int | None = 5,
) -> dict[str, Any]:
    mel_dir = Path(mel_dir).resolve()
    acc_dir = Path(acc_dir).resolve()
    output_root = Path(output_root).resolve()
    manifest_path = (
        Path(manifest_path).resolve() if manifest_path else output_root / "input_manifest.json"
    )
    if manifest_path.parent != output_root:
        raise ValueError("manifest_path must be directly inside output_root for stable relative paths")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"staging directory is not fresh: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    mel_output = output_root / "mel"
    acc_output = output_root / "acc"
    sidecar_output = output_root / "sidecars"
    npz_output = output_root / "npz"
    for directory in (mel_output, acc_output, sidecar_output, npz_output):
        directory.mkdir()

    source_midis = sorted(mel_dir.glob("*.mid"), key=lambda path: path.name)
    if expected_song_count is not None and len(source_midis) != int(expected_song_count):
        raise ValueError(
            f"expected {expected_song_count} source songs, found {len(source_midis)} in {mel_dir}"
        )
    if not source_midis:
        raise ValueError(f"no .mid files found in {mel_dir}")
    source_stems = [path.stem for path in source_midis]
    if len(source_stems) != len(set(source_stems)):
        raise ValueError("source MIDI stems are not unique")

    conditions = default_conditions(perturb_seeds, high_pseed)
    entries: list[dict[str, Any]] = []
    pairing_by_song_seed: dict[tuple[str, int], dict[str, Any]] = {}

    try:
        for source_path in source_midis:
            song = source_path.stem
            source_acc = acc_dir / source_path.name
            if not source_acc.is_file():
                raise FileNotFoundError(f"missing same-name accompaniment: {source_acc}")
            parsed = parse_canonical_midi(source_path)
            original_states = {note.source_note_id: note.state for note in parsed.notes}
            analysis_end_tick = max((state.end_tick for state in original_states.values()), default=0)
            acc_end_tick = _model_max_tick(source_acc)
            latent_by_seed = {
                int(seed): build_latent_decisions(parsed, int(seed)) for seed in perturb_seeds
            }
            generated_by_seed: dict[int, dict[str, dict[str, NoteState]]] = {
                int(seed): {} for seed in perturb_seeds
            }

            for condition in conditions:
                for perturb_seed in condition.seeds:
                    input_stem = _input_stem(song, condition.name, perturb_seed)
                    output_midi = mel_output / f"{input_stem}.mid"
                    output_acc = acc_output / f"{input_stem}.mid"
                    sidecar_path = sidecar_output / f"{input_stem}.perturbation.json"
                    npz_path = npz_output / f"{input_stem}.npz"
                    decisions = None if perturb_seed is None else latent_by_seed[int(perturb_seed)]
                    states, note_records, edit_stats = apply_condition(
                        parsed,
                        condition=condition,
                        perturb_seed=perturb_seed,
                        decisions=decisions,
                    )
                    write_states_to_midi(parsed, states, output_midi)
                    shutil.copyfile(source_acc, output_acc)
                    if sha256_file(source_acc) != sha256_file(output_acc):
                        raise AssertionError("accompaniment copy hash mismatch")

                    output_parsed = parse_canonical_midi(output_midi)
                    expected_roll = states_to_roll(states.values())
                    actual_states = [note.state for note in output_parsed.notes]
                    actual_roll = states_to_roll(
                        actual_states,
                        horizon=max((state.end_tick for state in states.values()), default=0),
                    )
                    if not np.array_equal(expected_roll, actual_roll):
                        raise AssertionError(f"writer changed model-domain roll for {input_stem}")

                    if perturb_seed is not None and condition.name in {"pitch", "onset", "both"}:
                        generated_by_seed[int(perturb_seed)][condition.name] = states

                    ticks_per_measure = _time_signature_numerator(parsed.midi) * MODEL_TICKS_PER_BEAT
                    input_end_tick = max((state.end_tick for state in states.values()), default=0)
                    raw_horizon = max(input_end_tick, acc_end_tick)
                    validation_horizon = (
                        int(math.ceil(raw_horizon / ticks_per_measure)) * ticks_per_measure
                        if raw_horizon
                        else 0
                    )
                    sidecar = {
                        "collision_policy": {
                            "attempts": MAX_COLLISION_ATTEMPTS,
                            "effective_cross_arm_mismatch": "allowed_and_reported",
                            "intervals": "half-open [start,end)",
                            "scope": "new same-pitch overlaps only",
                            "traversal": "start_tick,track_index,track_note_ordinal",
                            "raw_midi_serialization": "same track/channel/pitch overlaps must share end tick",
                        },
                        "condition": condition.name,
                        "edit_stats": edit_stats,
                        "input_id": input_stem,
                        "metadata_policy": {
                            "cc_pedal_pitch_bend": "preserve raw events",
                            "dangling_note_on": "drop; exclude from canonical universe",
                            "drum_notes": "preserve raw events; exclude from canonical universe",
                            "matched_model_visible_notes": "canonical raw-tick rewrite",
                            "off_range_notes": "preserve raw events; exclude from canonical universe",
                            "spurious_or_zero_duration_note_events": "drop; exclude from canonical universe",
                            "tempo_key_time_signature_track_channel_name": "preserve raw events",
                        },
                        "notes": note_records,
                        "onset_probability": condition.onset_probability,
                        "output_midi": {
                            "path": f"../mel/{output_midi.name}",
                            "sha256": sha256_file(output_midi),
                        },
                        "parser_stats": parsed.stats,
                        "perturb_seed": perturb_seed,
                        "pitch_probability": condition.pitch_probability,
                        "roll_diff_cells": _roll_diff_counts(original_states, states),
                        "schema_version": SCHEMA_VERSION,
                        "selection_semantics": "keyed Bernoulli: score < probability; denominator=model-visible canonical notes",
                        "source_midi": {
                            "path": os.path.relpath(source_path, sidecar_path.parent),
                            "sha256": parsed.source_sha256,
                        },
                        "tool_version": TOOL_VERSION,
                    }
                    write_canonical_json(sidecar_path, sidecar)
                    verify_sidecar(source_path, sidecar_path)

                    entries.append(
                        {
                            "acc_copy": _resource(output_acc, output_root),
                            "analysis_end_tick": analysis_end_tick,
                            "condition": condition.name,
                            "counts": edit_stats["counts"],
                            "input_id": input_stem,
                            "last_input_note_off_tick": input_end_tick,
                            "npz": _resource(npz_path, output_root, allow_missing=True),
                            "onset_probability": condition.onset_probability,
                            "output_midi": _resource(output_midi, output_root),
                            "perturb_seed": perturb_seed,
                            "pitch_probability": condition.pitch_probability,
                            "rates": edit_stats["rates"],
                            "schema_version": SCHEMA_VERSION,
                            "sidecar": _resource(sidecar_path, output_root),
                            "song": song,
                            "source_acc": {
                                "path": os.path.relpath(source_acc, output_root),
                                "sha256": sha256_file(source_acc),
                            },
                            "source_midi": {
                                "path": os.path.relpath(source_path, output_root),
                                "sha256": parsed.source_sha256,
                            },
                            "source_stem": song,
                            "stem": input_stem,
                            "validation_horizon_ticks": validation_horizon,
                        }
                    )

            for seed, arms in generated_by_seed.items():
                pairing = _pairing_mismatch(arms)
                if pairing is None:
                    raise AssertionError(f"missing medium factorial arm for {song}, seed {seed}")
                pairing_by_song_seed[(song, seed)] = pairing
                for entry in entries:
                    if (
                        entry["song"] == song
                        and entry["condition"] in {"pitch", "onset", "both"}
                        and entry["perturb_seed"] == seed
                    ):
                        sidecar_path = output_root / entry["sidecar"]["path"]
                        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                        sidecar["factorial_pairing"] = pairing
                        write_canonical_json(sidecar_path, sidecar)
                        entry["sidecar"]["sha256"] = sha256_file(sidecar_path)

        expected_stems = sorted(entry["stem"] for entry in entries)
        expected_count = len(source_midis) * 8
        if len(entries) != expected_count or len(set(expected_stems)) != expected_count:
            raise AssertionError(
                f"matrix cardinality mismatch: expected {expected_count}, got {len(entries)}"
            )
        if expected_song_count == 5 and len(entries) != 40:
            raise AssertionError(f"frozen five-song campaign must have 40 inputs, got {len(entries)}")

        for entry in entries:
            if entry["condition"] in {"pitch", "onset", "both"}:
                pairing = pairing_by_song_seed.get((entry["song"], int(entry["perturb_seed"])))
                if pairing is not None:
                    entry["factorial_pairing"] = pairing

        manifest = {
            "conditions": [
                {
                    "name": condition.name,
                    "onset_probability": condition.onset_probability,
                    "pitch_probability": condition.pitch_probability,
                    "seeds": list(condition.seeds),
                }
                for condition in conditions
            ],
            "entries": sorted(entries, key=lambda entry: entry["stem"]),
            "exact_stems": expected_stems,
            "expected_input_count": expected_count,
            "high_pseed": int(high_pseed),
            "input_count": len(entries),
            "metadata_policy": "see per-input sidecar; identical across all conditions",
            "model_ticks_per_beat": MODEL_TICKS_PER_BEAT,
            "perturb_seeds": [int(seed) for seed in perturb_seeds],
            "schema_version": SCHEMA_VERSION,
            "selection_semantics": "keyed Bernoulli threshold",
            "source_root": os.path.relpath(mel_dir, output_root),
            "source_song_count": len(source_midis),
            "source_stems": source_stems,
            "tool_version": TOOL_VERSION,
        }
        write_canonical_json(manifest_path, manifest)
        manifest_sha = sha256_file(manifest_path)
        (manifest_path.parent / f"{manifest_path.name}.sha256").write_text(
            f"{manifest_sha}  {manifest_path.name}\n", encoding="ascii"
        )
        return manifest
    except Exception:
        # A partial staging directory must never be mistaken for a valid input set.
        failure_path = output_root / ".FAILED"
        failure_path.write_text("campaign generation failed; discard this staging directory\n")
        raise


def _parse_seed_list(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate the eight-input-per-song matrix")
    generate.add_argument("--mel-dir", required=True, type=Path)
    generate.add_argument("--acc-dir", required=True, type=Path)
    generate.add_argument("--output-root", required=True, type=Path)
    generate.add_argument("--manifest", type=Path)
    generate.add_argument(
        "--perturb-seeds",
        type=_parse_seed_list,
        default=DEFAULT_PERTURB_SEEDS,
        help="two comma-separated seeds",
    )
    generate.add_argument("--high-pseed", type=int, default=DEFAULT_HIGH_PSEED)
    generate.add_argument("--expected-song-count", type=int, default=5)

    verify = subparsers.add_parser("verify", help="replay one sidecar and verify byte identity")
    verify.add_argument("--source-midi", required=True, type=Path)
    verify.add_argument("--sidecar", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "generate":
        manifest = generate_campaign(
            mel_dir=args.mel_dir,
            acc_dir=args.acc_dir,
            output_root=args.output_root,
            manifest_path=args.manifest,
            perturb_seeds=args.perturb_seeds,
            high_pseed=args.high_pseed,
            expected_song_count=args.expected_song_count,
        )
        print(
            json.dumps(
                {
                    "input_count": manifest["input_count"],
                    "manifest": str((args.manifest or args.output_root / "input_manifest.json").resolve()),
                    "status": "ok",
                },
                sort_keys=True,
            )
        )
    else:
        verify_sidecar(args.source_midi, args.sidecar)
        print(json.dumps({"sidecar": str(args.sidecar), "status": "ok"}, sort_keys=True))


if __name__ == "__main__":
    main()
