"""Deterministic beat-domain music features for fail-case analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi


GRID_STEP_BEATS = 0.25
PREFIX_BEATS = 8.0
EPSILON = 1e-8


@dataclass(frozen=True)
class BeatNote:
    start: float
    end: float
    pitch: int


class TrackSelectionError(ValueError):
    """Raised when the required semantic MIDI track cannot be selected safely."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def quantize_beat(value: float, step: float = GRID_STEP_BEATS) -> float:
    """Match MidiConverter: int(round(beat * steps_per_beat)) / steps_per_beat."""
    steps_per_beat = int(round(1.0 / step))
    tick = int(round(max(0.0, value) * steps_per_beat))
    return tick / steps_per_beat


def instrument_to_beat_notes(
    midi: pretty_midi.PrettyMIDI,
    instrument: pretty_midi.Instrument,
    step: float = GRID_STEP_BEATS,
) -> list[BeatNote]:
    """Convert seconds to quarter-note beats, then quantize to the analysis grid."""
    result: list[BeatNote] = []
    for note in instrument.notes:
        start = midi.time_to_tick(note.start) / midi.resolution
        end = midi.time_to_tick(note.end) / midi.resolution
        q_start = quantize_beat(start, step)
        q_end = max(quantize_beat(end, step), q_start + step)
        result.append(BeatNote(q_start, q_end, int(note.pitch)))
    return sorted(result, key=lambda item: (item.start, item.pitch, item.end))


def select_source_melody(midi: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
    named = [inst for inst in midi.instruments if inst.name.strip().casefold() == "melody"]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        raise TrackSelectionError("ambiguous_melody_track", "multiple Melody tracks")
    candidates = [inst for inst in midi.instruments if not inst.is_drum]
    if len(candidates) == 1:
        return candidates[0]
    status = "missing_melody_track" if not candidates else "ambiguous_melody_track"
    raise TrackSelectionError(status, f"expected one non-drum source track, found {len(candidates)}")


def select_prompt_accompaniment(midi: pretty_midi.PrettyMIDI) -> pretty_midi.Instrument:
    named = [
        inst for inst in midi.instruments
        if inst.name.strip().casefold() == "accompaniment"
    ]
    if len(named) == 1:
        return named[0]
    status = "missing_accompaniment_track" if not named else "ambiguous_accompaniment_track"
    raise TrackSelectionError(status, f"expected one Accompaniment track, found {len(named)}")


def load_source_melody(path: Path) -> tuple[str, list[BeatNote]]:
    midi = pretty_midi.PrettyMIDI(str(path))
    instrument = select_source_melody(midi)
    return instrument.name, instrument_to_beat_notes(midi, instrument)


def load_prompt_accompaniment(path: Path) -> tuple[str, list[BeatNote], bool]:
    midi = pretty_midi.PrettyMIDI(str(path))
    named = [
        inst for inst in midi.instruments
        if inst.name.strip().casefold() == "accompaniment"
    ]
    if not named and midi.instruments and all(
        inst.name.strip().casefold() == "melody" for inst in midi.instruments
    ):
        return "", [], False
    instrument = select_prompt_accompaniment(midi)
    return instrument.name, instrument_to_beat_notes(midi, instrument), True


def _clipped_notes(notes: Iterable[BeatNote], horizon: float) -> list[BeatNote]:
    return [
        BeatNote(max(0.0, note.start), min(horizon, note.end), note.pitch)
        for note in notes
        if note.end > 0.0 and note.start < horizon and min(horizon, note.end) > 0.0
    ]


def _onset_notes(notes: Iterable[BeatNote], horizon: float) -> list[BeatNote]:
    return [note for note in notes if 0.0 <= note.start < horizon]


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values.size else float("nan")


def _cv(values: np.ndarray) -> float:
    if not values.size:
        return float("nan")
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean > 0 else float("nan")


def _union_duration(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def _cell_voice_counts(notes: list[BeatNote], horizon: float) -> np.ndarray:
    cell_count = int(np.ceil(horizon / GRID_STEP_BEATS - 1e-12))
    counts = np.zeros(cell_count, dtype=float)
    for index in range(cell_count):
        start = index * GRID_STEP_BEATS
        end = min(horizon, start + GRID_STEP_BEATS)
        counts[index] = sum(note.start < end and note.end > start for note in notes)
    return counts


def _normalized_entropy(weights: np.ndarray, category_count: int) -> float:
    total = float(np.sum(weights))
    if total <= 0:
        return float("nan")
    probabilities = weights[weights > 0] / total
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / float(np.log(category_count))


def pitch_class_distribution(notes: list[BeatNote], horizon: float) -> np.ndarray | None:
    weights = np.zeros(12, dtype=float)
    for note in _clipped_notes(notes, horizon):
        weights[note.pitch % 12] += note.end - note.start
    total = float(np.sum(weights))
    return weights / total if total > 0 else None


def onset_phase_distribution(notes: list[BeatNote], horizon: float) -> np.ndarray | None:
    counts = np.zeros(4, dtype=float)
    distinct = sorted({note.start for note in _onset_notes(notes, horizon)})
    for onset in distinct:
        counts[int(round(onset / GRID_STEP_BEATS)) % 4] += 1.0
    total = float(np.sum(counts))
    return counts / total if total > 0 else None


def js_divergence(left: np.ndarray | None, right: np.ndarray | None) -> float:
    """Jensen-Shannon divergence normalized by log(2) to the interval [0, 1]."""
    if left is None or right is None:
        return float("nan")
    midpoint = (left + right) / 2.0

    def kl(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log(values[mask] / midpoint[mask])))

    return (kl(left) + kl(right)) / (2.0 * float(np.log(2.0)))


def _peak_onset_rate(onsets: np.ndarray, horizon: float, width: float) -> float:
    if horizon <= 0:
        return float("nan")
    final_start = max(0.0, horizon - width)
    starts = np.arange(0.0, final_start + GRID_STEP_BEATS / 2.0, GRID_STEP_BEATS)
    if not starts.size:
        starts = np.array([0.0])
    maximum = max(int(np.sum((onsets >= start) & (onsets < start + width))) for start in starts)
    return maximum / width


def window_features(notes: list[BeatNote], horizon: float) -> dict[str, float]:
    """Compute the frozen feature family over [0, horizon)."""
    if horizon <= 0:
        raise ValueError("feature horizon must be positive")
    onset_notes = _onset_notes(notes, horizon)
    clipped = _clipped_notes(notes, horizon)
    onsets = np.asarray([note.start for note in onset_notes], dtype=float)
    distinct_onsets = np.asarray(sorted(set(onsets.tolist())), dtype=float)
    durations = np.asarray([note.end - note.start for note in clipped], dtype=float)
    pitches = np.asarray([note.pitch for note in onset_notes], dtype=float)
    voices = _cell_voice_counts(clipped, horizon)

    if distinct_onsets.size:
        edge_onsets = np.concatenate(([0.0], distinct_onsets, [horizon]))
        longest_gap = float(np.max(np.diff(edge_onsets)))
    else:
        longest_gap = horizon

    iois = np.diff(distinct_onsets)
    onset_pitch_medians = np.asarray(
        [np.median([note.pitch for note in onset_notes if note.start == onset]) for onset in distinct_onsets],
        dtype=float,
    )
    intervals = np.abs(np.diff(onset_pitch_medians))

    pitch_weights = np.zeros(12, dtype=float)
    for note in clipped:
        pitch_weights[note.pitch % 12] += note.end - note.start
    phase = onset_phase_distribution(notes, horizon)
    beat_count = int(np.ceil(horizon - 1e-12))
    empty_beats = 0
    for beat in range(beat_count):
        beat_start = float(beat)
        beat_end = min(horizon, beat_start + 1.0)
        if not any(note.start < beat_end and note.end > beat_start for note in clipped):
            empty_beats += 1

    return {
        "note_count": float(len(onset_notes)),
        "onset_note_density": len(onset_notes) / horizon,
        "distinct_onset_density": distinct_onsets.size / horizon,
        "peak_onset_rate_4beat": _peak_onset_rate(onsets, horizon, 4.0),
        "peak_onset_rate_8beat": _peak_onset_rate(onsets, horizon, 8.0),
        "ioi_median": _safe_percentile(iois, 50),
        "ioi_p10": _safe_percentile(iois, 10),
        "ioi_p90": _safe_percentile(iois, 90),
        "ioi_cv": _cv(iois),
        "duration_median": _safe_percentile(durations, 50),
        "duration_p10": _safe_percentile(durations, 10),
        "duration_p90": _safe_percentile(durations, 90),
        "duration_cv": _cv(durations),
        "one_step_duration_rate": float(np.mean(durations <= GRID_STEP_BEATS)) if durations.size else float("nan"),
        "active_coverage": _union_duration((note.start, note.end) for note in clipped) / horizon,
        "empty_beat_rate": empty_beats / beat_count,
        "longest_onset_gap": longest_gap,
        "pitch_range": float(np.max(pitches) - np.min(pitches)) if pitches.size else float("nan"),
        "robust_pitch_range": _safe_percentile(pitches, 95) - _safe_percentile(pitches, 5) if pitches.size else float("nan"),
        "pitch_median": _safe_percentile(pitches, 50),
        "pitch_std": float(np.std(pitches)) if pitches.size else float("nan"),
        "abs_interval_median": _safe_percentile(intervals, 50),
        "abs_interval_p90": _safe_percentile(intervals, 90),
        "leap_gt7_rate": float(np.mean(intervals > 7)) if intervals.size else float("nan"),
        "leap_gt12_rate": float(np.mean(intervals > 12)) if intervals.size else float("nan"),
        "pitch_class_entropy": _normalized_entropy(pitch_weights, 12),
        "onset_phase_entropy": _normalized_entropy(phase, 4) if phase is not None else float("nan"),
        "average_voice_number": float(np.mean(voices)) if voices.size else float("nan"),
        "polyphonic_active_ratio": float(np.mean(voices > 1)) if voices.size else float("nan"),
    }


def hard_input_features(notes: list[BeatNote]) -> dict[str, float]:
    features = window_features(notes, PREFIX_BEATS)
    onsets = sorted({note.start for note in _onset_notes(notes, PREFIX_BEATS)})
    onset_counts = [sum(note.start == onset for note in notes) for onset in onsets]
    return {
        "first8_note_count": features["note_count"],
        "first8_active_coverage": features["active_coverage"],
        "first8_leading_blank_beats": onsets[0] if onsets else PREFIX_BEATS,
        "first8_chord_onset_ratio": float(np.mean(np.asarray(onset_counts) > 1)) if onset_counts else float("nan"),
        "first8_polyphonic_active_ratio": features["polyphonic_active_ratio"],
    }


def melody_features(notes: list[BeatNote]) -> dict[str, float]:
    prefix = window_features(notes, PREFIX_BEATS)
    full_horizon = max((note.end for note in notes), default=0.0)
    result = hard_input_features(notes)
    result["full_horizon_beats"] = full_horizon
    for name, value in prefix.items():
        if name not in {"average_voice_number", "polyphonic_active_ratio"}:
            result[f"first8_{name}"] = value
    if full_horizon <= 0:
        full = {name: float("nan") for name in prefix}
    else:
        full = window_features(notes, full_horizon)
    for name, value in full.items():
        if name not in {"average_voice_number", "polyphonic_active_ratio"}:
            result[f"full_{name}"] = value

    result["prefix_full_abs_density_log_ratio"] = _abs_log_ratio(
        prefix["onset_note_density"], full["onset_note_density"]
    )
    result["prefix_full_abs_median_duration_log_ratio"] = _abs_log_ratio(
        prefix["duration_median"], full["duration_median"]
    )
    result["prefix_full_abs_robust_pitch_range_difference"] = _abs_difference(
        prefix["robust_pitch_range"], full["robust_pitch_range"]
    )
    result["prefix_full_pitch_class_js_divergence"] = js_divergence(
        pitch_class_distribution(notes, PREFIX_BEATS),
        pitch_class_distribution(notes, full_horizon) if full_horizon > 0 else None,
    )
    result["prefix_full_onset_phase_js_divergence"] = js_divergence(
        onset_phase_distribution(notes, PREFIX_BEATS),
        onset_phase_distribution(notes, full_horizon) if full_horizon > 0 else None,
    )
    return result


def prompt_features(notes: list[BeatNote]) -> dict[str, float]:
    values = window_features(notes, PREFIX_BEATS)
    allowed = {
        "note_count", "onset_note_density", "distinct_onset_density",
        "active_coverage", "empty_beat_rate", "duration_median", "duration_p10",
        "duration_p90", "duration_cv", "one_step_duration_rate", "pitch_range",
        "robust_pitch_range", "pitch_median", "pitch_class_entropy",
        "onset_phase_entropy", "average_voice_number",
    }
    result = {f"prompt_{name}": values[name] for name in values if name in allowed}
    first = _bar_onset_vector(notes, 0.0)
    second = _bar_onset_vector(notes, 4.0)
    union = np.logical_or(first, second)
    result["prompt_groove_consistency"] = (
        float(np.logical_and(first, second).sum() / union.sum())
        if union.any() else float("nan")
    )
    return result


def relation_features(melody: list[BeatNote], accompaniment: list[BeatNote]) -> dict[str, float]:
    mel = window_features(melody, PREFIX_BEATS)
    acc = window_features(accompaniment, PREFIX_BEATS)
    signed = _signed_log_ratio(acc["onset_note_density"], mel["onset_note_density"])
    return {
        "relation_log_density_ratio": signed,
        "relation_abs_log_density_ratio": abs(signed) if np.isfinite(signed) else float("nan"),
        "relation_abs_median_duration_log_mismatch": _abs_log_ratio(
            acc["duration_median"], mel["duration_median"]
        ),
        "relation_pitch_class_js_divergence": js_divergence(
            pitch_class_distribution(melody, PREFIX_BEATS),
            pitch_class_distribution(accompaniment, PREFIX_BEATS),
        ),
        "relation_onset_phase_js_divergence": js_divergence(
            onset_phase_distribution(melody, PREFIX_BEATS),
            onset_phase_distribution(accompaniment, PREFIX_BEATS),
        ),
        "relation_abs_pitch_median_gap": _abs_difference(
            acc["pitch_median"], mel["pitch_median"]
        ),
    }


def _bar_onset_vector(notes: list[BeatNote], start: float) -> np.ndarray:
    vector = np.zeros(16, dtype=bool)
    for note in notes:
        if start <= note.start < start + 4.0:
            index = int(round((note.start - start) / GRID_STEP_BEATS))
            if 0 <= index < 16:
                vector[index] = True
    return vector


def _signed_log_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    return float(np.log((numerator + EPSILON) / (denominator + EPSILON)))


def _abs_log_ratio(left: float, right: float) -> float:
    value = _signed_log_ratio(left, right)
    return abs(value) if np.isfinite(value) else float("nan")


def _abs_difference(left: float, right: float) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return float("nan")
    return abs(left - right)
