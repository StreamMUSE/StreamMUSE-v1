"""Independent evaluation helpers for the offline rap audio protocol campaign."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, TwoBarRenderRequest


_WORD_PATTERN = re.compile(r"[a-z0-9']+")
_DEFAULT_RMS_WINDOW_SECONDS = 0.040


@dataclass(frozen=True)
class RecognizedWord:
    text: str
    start_seconds: float | None
    end_seconds: float | None


def normalize_word(text: str) -> str:
    tokens = _WORD_PATTERN.findall(text.lower())
    return tokens[0] if tokens else ""


def normalize_words(text: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(text, str):
        return tuple(_WORD_PATTERN.findall(text.lower()))
    return tuple(token for token in (normalize_word(part) for part in text) if token)


def compute_word_error_counts(
    reference_words: Sequence[str],
    hypothesis_words: Sequence[str],
) -> dict[str, float | int]:
    """Return an exact Levenshtein breakdown and WER."""
    reference = tuple(reference_words)
    hypothesis = tuple(hypothesis_words)
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    costs = [[0] * cols for _ in range(rows)]
    ops = [[(0, 0, 0)] * cols for _ in range(rows)]

    for row in range(1, rows):
        costs[row][0] = row
        ops[row][0] = (0, 0, row)
    for col in range(1, cols):
        costs[0][col] = col
        ops[0][col] = (0, col, 0)

    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                costs[row][col] = costs[row - 1][col - 1]
                ops[row][col] = ops[row - 1][col - 1]
                continue

            substitution_cost = costs[row - 1][col - 1] + 1
            insertion_cost = costs[row][col - 1] + 1
            deletion_cost = costs[row - 1][col] + 1
            best_cost = min(substitution_cost, insertion_cost, deletion_cost)
            costs[row][col] = best_cost
            if best_cost == substitution_cost:
                sub, ins, delete = ops[row - 1][col - 1]
                ops[row][col] = (sub + 1, ins, delete)
            elif best_cost == insertion_cost:
                sub, ins, delete = ops[row][col - 1]
                ops[row][col] = (sub, ins + 1, delete)
            else:
                sub, ins, delete = ops[row - 1][col]
                ops[row][col] = (sub, ins, delete + 1)

    substitutions, insertions, deletions = ops[-1][-1]
    reference_word_count = len(reference)
    total_errors = substitutions + insertions + deletions
    wer = 0.0 if reference_word_count == 0 else total_errors / reference_word_count
    return {
        "reference_word_count": reference_word_count,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "word_error_rate": wer,
    }


def estimate_syllable_timing_error_ms(
    request: TwoBarRenderRequest,
    recognized_words: Sequence[RecognizedWord],
) -> tuple[float, ...]:
    """Estimate syllable-anchor timing error from independent word intervals.

    This is explicitly a syllable-level estimate derived from word timing, not a
    phone-level ground truth measurement.
    """
    request_groups = [
        (normalized, syllables)
        for word, syllables in _group_request_syllables(request)
        if (normalized := normalize_word(word))
    ]
    recognized_groups = [
        (normalized, word)
        for word in recognized_words
        if (normalized := normalize_word(word.text))
    ]

    errors: list[float] = []
    matches = _matching_word_indices(
        tuple(word for word, _syllables in request_groups),
        tuple(word for word, _recognized in recognized_groups),
    )
    for request_index, recognized_index in matches:
        syllables = request_groups[request_index][1]
        recognized = recognized_groups[recognized_index][1]
        if recognized.start_seconds is None or recognized.end_seconds is None:
            continue
        if recognized.end_seconds <= recognized.start_seconds:
            continue
        if len(syllables) == 1:
            estimated_seconds = (recognized.start_seconds,)
        else:
            step = (recognized.end_seconds - recognized.start_seconds) / (len(syllables) - 1)
            estimated_seconds = tuple(recognized.start_seconds + step * index for index in range(len(syllables)))
        for syllable, estimate in zip(syllables, estimated_seconds, strict=True):
            errors.append((estimate - syllable.target_seconds) * 1000.0)
    return tuple(errors)


def _matching_word_indices(
    reference_words: Sequence[str],
    hypothesis_words: Sequence[str],
) -> tuple[tuple[int, int], ...]:
    """Return exact word matches from a deterministic edit-distance alignment."""
    rows = len(reference_words) + 1
    cols = len(hypothesis_words) + 1
    costs = [[0] * cols for _ in range(rows)]
    directions = [[""] * cols for _ in range(rows)]

    for row in range(1, rows):
        costs[row][0] = row
        directions[row][0] = "delete"
    for col in range(1, cols):
        costs[0][col] = col
        directions[0][col] = "insert"

    for row in range(1, rows):
        for col in range(1, cols):
            if reference_words[row - 1] == hypothesis_words[col - 1]:
                costs[row][col] = costs[row - 1][col - 1]
                directions[row][col] = "match"
                continue
            candidates = (
                (costs[row - 1][col - 1] + 1, 0, "substitute"),
                (costs[row][col - 1] + 1, 1, "insert"),
                (costs[row - 1][col] + 1, 2, "delete"),
            )
            cost, _priority, direction = min(candidates)
            costs[row][col] = cost
            directions[row][col] = direction

    matches: list[tuple[int, int]] = []
    row = len(reference_words)
    col = len(hypothesis_words)
    while row or col:
        direction = directions[row][col]
        if direction == "match":
            matches.append((row - 1, col - 1))
            row -= 1
            col -= 1
        elif direction == "substitute":
            row -= 1
            col -= 1
        elif direction == "insert":
            col -= 1
        elif direction == "delete":
            row -= 1
        else:  # pragma: no cover - only reachable if the matrix is malformed
            raise RuntimeError("invalid edit-alignment backtrace")
    matches.reverse()
    return tuple(matches)


def compute_signal_metrics(samples: np.ndarray) -> dict[str, int | bool | float]:
    mono = _to_mono_float32(samples)
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64))) if mono.size else 0.0
    return {
        "peak_abs": float(np.max(np.abs(mono), initial=0.0)),
        "clipped_sample_count": int(np.count_nonzero(np.abs(mono) >= np.float32(0.999999))),
        "silent": bool(rms < 1e-6),
        "rms": rms,
    }


def measure_stress_rms_correlation(
    request: TwoBarRenderRequest,
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    window_seconds: float = _DEFAULT_RMS_WINDOW_SECONDS,
) -> float | None:
    mono = _to_mono_float32(samples)
    if sample_rate_hz <= 0 or mono.size == 0:
        return None
    half_window = max(1, int(round(window_seconds * sample_rate_hz / 2)))
    stresses = np.asarray([syllable.target_stress for syllable in request.syllables], dtype=np.float64)
    rms_values = []
    for syllable in request.syllables:
        center = int(round(syllable.target_seconds * sample_rate_hz))
        start = max(0, center - half_window)
        end = min(mono.shape[0], center + half_window)
        if start >= end:
            rms_values.append(0.0)
            continue
        window = mono[start:end]
        rms_values.append(float(np.sqrt(np.mean(np.square(window), dtype=np.float64))))
    if len(rms_values) < 2:
        return None
    rms = np.asarray(rms_values, dtype=np.float64)
    if np.allclose(stresses, stresses[0]) or np.allclose(rms, rms[0]):
        return None
    return float(np.corrcoef(stresses, rms)[0, 1])


def evaluate_protocol_song(
    *,
    protocol_id: ProtocolId,
    song_id: str,
    requests: Sequence[TwoBarRenderRequest],
    chunk_records: Sequence[ChunkRenderRecord],
    transcribe_chunk: Callable[[Path], Sequence[RecognizedWord]],
    progress_callback: Callable[[TwoBarRenderRequest, str], None] | None = None,
) -> dict[str, Any]:
    records_by_chunk = {record.chunk_index: record for record in chunk_records}
    total_word_counts = {
        "reference_word_count": 0,
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
    }
    duration_errors_ms: list[float] = []
    clipped_sample_count = 0
    silent_chunk_count = 0
    failed_chunk_count = 0
    stress_correlations: list[float] = []
    timing_errors_ms: list[float] = []
    successful_chunk_count = 0

    for request in requests:
        record = records_by_chunk.get(request.chunk_index)
        if record is None or not record.success or not record.output_path:
            failed_chunk_count += 1
            if progress_callback is not None:
                progress_callback(request, "failed")
            continue
        try:
            output_path = Path(record.output_path)
            sample_rate_hz, samples = wavfile.read(output_path)
            mono = _to_mono_float32(samples)
            successful_chunk_count += 1

            duration_errors_ms.append(abs((mono.shape[0] / sample_rate_hz) - request.duration_seconds) * 1000.0)
            signal_metrics = compute_signal_metrics(mono)
            clipped_sample_count += int(signal_metrics["clipped_sample_count"])
            silent_chunk_count += int(bool(signal_metrics["silent"]))

            recognized_words = tuple(transcribe_chunk(output_path))
            counts = compute_word_error_counts(
                normalize_words(request.text),
                tuple(word for word in (normalize_word(item.text) for item in recognized_words) if word),
            )
            for key in total_word_counts:
                total_word_counts[key] += int(counts[key])

            stress_correlation = measure_stress_rms_correlation(
                request,
                mono,
                sample_rate_hz=sample_rate_hz,
            )
            if stress_correlation is not None and not math.isnan(stress_correlation):
                stress_correlations.append(stress_correlation)

            timing_errors_ms.extend(estimate_syllable_timing_error_ms(request, recognized_words))
        except Exception:
            if progress_callback is not None:
                progress_callback(request, "error")
            raise
        if progress_callback is not None:
            progress_callback(request, "success")

    reference_words = total_word_counts["reference_word_count"]
    total_errors = (
        total_word_counts["substitutions"]
        + total_word_counts["insertions"]
        + total_word_counts["deletions"]
    )
    return {
        "protocol_id": protocol_id.value,
        "song_id": song_id,
        "chunk_count": len(requests),
        "successful_chunk_count": successful_chunk_count,
        "failed_chunk_count": failed_chunk_count,
        "silent_chunk_count": silent_chunk_count,
        "clipped_sample_count": clipped_sample_count,
        "word_error_counts": total_word_counts,
        "word_error_rate": 0.0 if reference_words == 0 else total_errors / reference_words,
        "duration_error_ms": _duration_summary(duration_errors_ms),
        "estimated_syllable_timing_error_ms": _summarize_values(timing_errors_ms),
        "stress_rms_correlation": (
            float(np.mean(np.asarray(stress_correlations, dtype=np.float64)))
            if stress_correlations
            else None
        ),
    }


def build_faster_whisper_transcriber(
    *,
    model_size: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    whisper_model_factory: Callable[..., Any] | None = None,
) -> Callable[[Path | str], tuple[RecognizedWord, ...]]:
    model: Any | None = None

    def transcribe(path: Path | str) -> tuple[RecognizedWord, ...]:
        nonlocal model
        if model is None:
            factory = whisper_model_factory
            if factory is None:
                module = importlib.import_module("faster_whisper")
                factory = getattr(module, "WhisperModel")
            model = factory(model_size_or_path=model_size, device=device, compute_type=compute_type)
        segments, _info = model.transcribe(str(path), word_timestamps=True, vad_filter=True)
        words: list[RecognizedWord] = []
        for segment in segments:
            for word in getattr(segment, "words", ()) or ():
                words.append(
                    RecognizedWord(
                        text=str(getattr(word, "word", "")),
                        start_seconds=_coerce_optional_float(getattr(word, "start", None)),
                        end_seconds=_coerce_optional_float(getattr(word, "end", None)),
                    )
                )
        return tuple(words)

    return transcribe


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _group_request_syllables(
    request: TwoBarRenderRequest,
) -> tuple[tuple[str, tuple[Any, ...]], ...]:
    groups: list[tuple[str, list[Any]]] = []
    current_word = ""
    current_group: list[Any] = []
    for syllable in request.syllables:
        if current_group and syllable.word != current_word:
            groups.append((current_word, current_group))
            current_group = []
        current_word = syllable.word
        current_group.append(syllable)
    if current_group:
        groups.append((current_word, current_group))
    return tuple((word, tuple(group)) for word, group in groups)


def _summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "measured_count": 0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "measured_count": int(array.size),
    }


def _duration_summary(values: Sequence[float]) -> dict[str, float]:
    summary = _summarize_values(values)
    return {
        "mean": float(summary["mean"]),
        "median": float(summary["median"]),
        "p95": float(summary["p95"]),
        "max": float(summary["max"]),
    }


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), info.max)
        converted = array.astype(np.float32) / np.float32(scale)
    else:
        converted = array.astype(np.float32, copy=False)
    if converted.ndim == 1:
        return converted
    return np.mean(converted, axis=1, dtype=np.float32)
