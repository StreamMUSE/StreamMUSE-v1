"""Canonical 4-ticks-per-beat metrics for the melody robustness pilot."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mido


DISSONANT_INTERVAL_CLASSES = frozenset({1, 2, 6, 10, 11})


@dataclass(frozen=True)
class Roll:
    """Sparse canonical roll; the analysis window is [0, end_tick)."""

    end_tick: int
    sustain: frozenset[tuple[int, int]]
    onsets: frozenset[tuple[int, int]]

    def crop(self, end_tick: int) -> "Roll":
        end = min(int(end_tick), self.end_tick)
        return Roll(
            end_tick=end,
            sustain=frozenset((tick, pitch) for tick, pitch in self.sustain if tick < end),
            onsets=frozenset((tick, pitch) for tick, pitch in self.onsets if tick < end),
        )


def _model_tick(raw_tick: int, source_tpb: int) -> int:
    scaled = raw_tick * 4
    quotient, remainder = divmod(scaled, source_tpb)
    if remainder == 0:
        return quotient
    return int(round(scaled / source_tpb))


def load_midi_roll(
    path: str | Path,
    *,
    end_tick: int | None = None,
    exclude_drums: bool = True,
    track_name_contains: str | None = None,
) -> Roll:
    """Load all pitched tracks onto the model's common 4-tpb grid.

    Retriggers close the oldest matching active note.  Hanging notes close at
    the requested horizon (or final MIDI tick), and tempo headers do not affect
    the musical grid.
    """
    midi = mido.MidiFile(str(path))
    timeline: list[tuple[int, int, Any]] = []
    final_absolute = 0
    for track_index, track in enumerate(midi.tracks):
        track_name = next(
            (str(message.name) for message in track if message.type == "track_name"), ""
        )
        if track_name_contains and track_name_contains.lower() not in track_name.lower():
            continue
        absolute = 0
        for message in track:
            absolute += int(message.time)
            timeline.append((absolute, track_index, message))
        final_absolute = max(final_absolute, absolute)
    timeline.sort(key=lambda item: (item[0], item[1]))
    active: dict[tuple[int, int, int], list[int]] = {}
    notes: list[tuple[int, int, int]] = []
    for absolute, track_index, message in timeline:
        if not hasattr(message, "channel") or not hasattr(message, "note"):
            continue
        channel = int(message.channel)
        if exclude_drums and channel == 9:
            continue
        tick = _model_tick(absolute, midi.ticks_per_beat)
        pitch = int(message.note)
        key = (track_index, channel, pitch)
        is_on = message.type == "note_on" and int(message.velocity) > 0
        is_off = message.type == "note_off" or (
            message.type == "note_on" and int(message.velocity) == 0
        )
        if is_on:
            active.setdefault(key, []).append(tick)
        elif is_off and active.get(key):
            start = active[key].pop(0)
            notes.append((start, max(start + 1, tick), pitch))
    natural_end = max(
        [_model_tick(final_absolute, midi.ticks_per_beat)]
        + [note_end for _start, note_end, _pitch in notes]
        + [start + 1 for starts in active.values() for start in starts]
    )
    horizon = int(end_tick) if end_tick is not None else natural_end
    for (_track, _channel, pitch), starts in active.items():
        for start in starts:
            notes.append((start, max(start + 1, horizon), pitch))
    sustain: set[tuple[int, int]] = set()
    onsets: set[tuple[int, int]] = set()
    for start, stop, pitch in notes:
        if 0 <= start < horizon:
            onsets.add((start, pitch))
        for tick in range(max(0, start), min(horizon, stop)):
            sustain.add((tick, pitch))
    return Roll(horizon, frozenset(sustain), frozenset(onsets))


def write_roll_midi(
    roll: Roll,
    path: str | Path,
    *,
    bpm: int = 120,
    velocity: int = 80,
) -> None:
    """Write a deterministic canonical MIDI from a sparse sustain roll."""
    if isinstance(velocity, bool) or not isinstance(velocity, int) or not 1 <= velocity <= 127:
        raise ValueError("MIDI velocity must be an integer in [1, 127]")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_tpb = 480
    step = source_tpb // 4
    by_pitch: dict[int, set[int]] = {}
    for tick, pitch in roll.sustain:
        by_pitch.setdefault(pitch, set()).add(tick)
    events: list[tuple[int, int, mido.Message]] = []
    for pitch, ticks in sorted(by_pitch.items()):
        ordered = sorted(ticks)
        if not ordered:
            continue
        start = previous = ordered[0]
        for tick in ordered[1:] + [roll.end_tick + 1]:
            if tick != previous + 1 or (tick, pitch) in roll.onsets:
                events.append(
                    (
                        start * step,
                        1,
                        mido.Message("note_on", note=pitch, velocity=velocity),
                    )
                )
                events.append(((previous + 1) * step, 0, mido.Message("note_off", note=pitch, velocity=0)))
                start = tick
            previous = tick
    events.sort(key=lambda item: (item[0], item[1], int(getattr(item[2], "note", 0))))
    midi = mido.MidiFile(type=1, ticks_per_beat=source_tpb)
    meta = mido.MidiTrack()
    meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    midi.tracks.append(meta)
    track = mido.MidiTrack()
    track.append(mido.Message("program_change", program=0, time=0))
    last = 0
    for absolute, _priority, message in events:
        message.time = absolute - last
        track.append(message)
        last = absolute
    midi.tracks.append(track)
    midi.save(str(destination))


def _set_metric(left: frozenset[tuple[int, int]], right: frozenset[tuple[int, int]]) -> dict[str, Any]:
    if not left and not right:
        return {"jaccard": None, "f1": None, "flag": "both_empty"}
    intersection = len(left & right)
    union = len(left | right)
    denominator = len(left) + len(right)
    return {
        "jaccard": intersection / union if union else None,
        "f1": (2 * intersection / denominator) if denominator else None,
        "flag": "one_empty" if not left or not right else "ok",
    }


def _onset_distance(left: Roll, right: Roll) -> dict[str, Any]:
    left_ticks = sorted(tick for tick, _pitch in left.onsets)
    right_ticks = sorted(tick for tick, _pitch in right.onsets)
    if not left_ticks and not right_ticks:
        return {"mean_ticks": None, "flag": "both_empty"}
    if not left_ticks or not right_ticks:
        return {"mean_ticks": None, "flag": "one_empty"}
    distances = [min(abs(tick - other) for other in right_ticks) for tick in left_ticks]
    reverse = [min(abs(tick - other) for other in left_ticks) for tick in right_ticks]
    return {"mean_ticks": statistics.fmean(distances + reverse), "flag": "ok"}


def sensitivity_metrics(left: Roll, right: Roll) -> dict[str, Any]:
    end = min(left.end_tick, right.end_tick)
    left = left.crop(end)
    right = right.crop(end)
    return {
        "analysis_end_tick": end,
        "sustain": _set_metric(left.sustain, right.sustain),
        "onset": _set_metric(left.onsets, right.onsets),
        "onset_distance": _onset_distance(left, right),
    }


def _pitch_map(cells: Iterable[tuple[int, int]]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for tick, pitch in cells:
        result.setdefault(tick, set()).add(pitch)
    return result


def coverage_metrics(accompaniment: Roll, *, end_tick: int | None = None) -> dict[str, Any]:
    end = min(
        accompaniment.end_tick,
        accompaniment.end_tick if end_tick is None else int(end_tick),
    )
    roll = accompaniment.crop(end)
    active = _pitch_map(roll.sustain)
    beats = max(1, math.ceil(end / 4))
    empty_beats = 0
    for beat in range(beats):
        if not any(active.get(tick) for tick in range(beat * 4, min(end, beat * 4 + 4))):
            empty_beats += 1
    return {
        "onsets_per_beat": len(roll.onsets) / beats,
        "active_pitch_ticks_per_beat": len(roll.sustain) / beats,
        "empty_beat_ratio": empty_beats / beats,
        "active_tick_count": len(active),
        "beat_count": beats,
        "fully_empty": not roll.sustain,
    }


def dissonance_metrics(melody: Roll, accompaniment: Roll) -> dict[str, Any]:
    end = min(melody.end_tick, accompaniment.end_tick)
    melody_map = _pitch_map(melody.crop(end).sustain)
    acc_map = _pitch_map(accompaniment.crop(end).sustain)
    pair_count = 0
    dissonant_count = 0
    coactive_tick_values: list[float] = []
    melody_active_ticks = sum(bool(melody_map.get(tick)) for tick in range(end))
    for tick in range(end):
        tick_pairs = 0
        tick_dissonant = 0
        for melody_pitch in melody_map.get(tick, set()):
            for acc_pitch in acc_map.get(tick, set()):
                tick_pairs += 1
                if abs(acc_pitch - melody_pitch) % 12 in DISSONANT_INTERVAL_CLASSES:
                    tick_dissonant += 1
        if tick_pairs:
            pair_count += tick_pairs
            dissonant_count += tick_dissonant
            coactive_tick_values.append(tick_dissonant / tick_pairs)
    coverage = coverage_metrics(accompaniment, end_tick=end)
    coactive_ticks = len(coactive_tick_values)
    harmonic_pair_coverage = (
        coactive_ticks / melody_active_ticks if melody_active_ticks else None
    )
    failure = pair_count == 0
    return {
        "analysis_end_tick": end,
        "dissonant_pair_ticks": dissonant_count,
        "pair_ticks": pair_count,
        "D_micro": dissonant_count / pair_count if pair_count else None,
        "D_macro_coactive_tick": (
            statistics.fmean(coactive_tick_values) if coactive_tick_values else None
        ),
        "harmonic_pair_coverage": harmonic_pair_coverage,
        "coactive_tick_count": coactive_ticks,
        "melody_active_tick_count": melody_active_ticks,
        "coverage_failure": failure,
        "na_reason": "zero_coactive_pair_denominator" if failure else None,
        "coverage": coverage,
    }


def rhythmic_metrics(melody: Roll, accompaniment: Roll) -> dict[str, Any]:
    melody_onsets = sorted({tick for tick, _pitch in melody.onsets})
    acc_onsets = sorted({tick for tick, _pitch in accompaniment.onsets})
    nearest = None
    if melody_onsets and acc_onsets:
        nearest = statistics.fmean(
            min(abs(tick - melody_tick) for melody_tick in melody_onsets)
            for tick in acc_onsets
        )
    melody_ioi = [b - a for a, b in zip(melody_onsets, melody_onsets[1:])]
    acc_ioi = [b - a for a, b in zip(acc_onsets, acc_onsets[1:])]
    count = min(len(melody_ioi), len(acc_ioi))
    correlation = None
    if count >= 2:
        left = melody_ioi[:count]
        right = acc_ioi[:count]
        left_mean = statistics.fmean(left)
        right_mean = statistics.fmean(right)
        numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        correlation = numerator / denominator if denominator else None
    return {
        "nearest_melody_onset_distance_ticks": nearest,
        "ioi_correlation": correlation,
        "melody_onset_count": len(melody_onsets),
        "accompaniment_onset_count": len(acc_onsets),
    }


def transform_roll(roll: Roll, kind: str) -> Roll:
    if kind == "identity":
        return roll
    if kind in {"harmonic_m2", "harmonic_tt"}:
        shift = 1 if kind == "harmonic_m2" else 6
        def shifted(cells: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
            return frozenset((tick, min(127, max(0, pitch + shift))) for tick, pitch in cells)
        return Roll(roll.end_tick, shifted(roll.sustain), shifted(roll.onsets))
    if kind == "rhythm_shift":
        def moved(cells: Iterable[tuple[int, int]]) -> frozenset[tuple[int, int]]:
            return frozenset((tick + 1, pitch) for tick, pitch in cells if tick + 1 < roll.end_tick)
        return Roll(roll.end_tick, moved(roll.sustain), moved(roll.onsets))
    if kind == "coverage_dropout":
        keep = {cell for index, cell in enumerate(sorted(roll.onsets)) if index % 2 == 0}
        retained: set[tuple[int, int]] = set()
        onsets_by_pitch: dict[int, set[int]] = {}
        for tick, pitch in roll.onsets:
            onsets_by_pitch.setdefault(pitch, set()).add(tick)
        for start, pitch in keep:
            tick = start
            while (tick, pitch) in roll.sustain:
                if tick > start and tick in onsets_by_pitch.get(pitch, set()):
                    break
                retained.add((tick, pitch))
                tick += 1
        return Roll(
            roll.end_tick,
            frozenset(retained),
            frozenset(keep),
        )
    if kind == "coverage_empty":
        return Roll(roll.end_tick, frozenset(), frozenset())
    raise ValueError(f"unknown control transform: {kind}")


def bootstrap_song_mean(
    song_effects: Mapping[str, float | None], *, seed: int, iterations: int = 10000
) -> dict[str, Any]:
    songs = sorted(song_effects)
    values = [song_effects[song] for song in songs]
    if not songs:
        return {
            "estimate": None,
            "interval": None,
            "valid_song_count": 0,
            "na_pattern": {},
            "reason": "no_song_blocks",
            "iterations": iterations,
            "seed": seed,
        }
    if any(value is None for value in values):
        return {
            "estimate": None, "interval": None, "valid_song_count": sum(v is not None for v in values),
            "na_pattern": {song: value is None for song, value in zip(songs, values)},
        }
    numeric = [float(value) for value in values if value is not None]
    rng = random.Random(seed)
    samples = sorted(
        statistics.fmean(rng.choice(numeric) for _ in numeric) for _ in range(iterations)
    )
    lower = samples[int(0.025 * (iterations - 1))]
    upper = samples[int(0.975 * (iterations - 1))]
    leave_one_out = [
        statistics.fmean(value for index, value in enumerate(numeric) if index != omitted)
        for omitted in range(len(numeric))
    ] if len(numeric) > 1 else numeric
    return {
        "estimate": statistics.fmean(numeric),
        "interval": [lower, upper],
        "interpretation": "descriptive_song_block_bootstrap",
        "raw_song_effects": dict(song_effects),
        "leave_one_song_out_range": [min(leave_one_out), max(leave_one_out)],
        "valid_song_count": len(numeric),
        "iterations": iterations,
        "seed": seed,
    }
