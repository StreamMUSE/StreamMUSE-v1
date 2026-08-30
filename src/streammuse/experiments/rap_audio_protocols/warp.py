"""Forced-alignment parsing and piecewise warp helpers for Protocol 4."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.optimize import Bounds, LinearConstraint, minimize

from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget


_ARPABET_VOWELS = frozenset(
    {
        "AA",
        "AE",
        "AH",
        "AO",
        "AW",
        "AY",
        "EH",
        "ER",
        "EY",
        "IH",
        "IY",
        "OW",
        "OY",
        "UH",
        "UW",
    }
)
_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LONG_TIER_RE = re.compile(r"^[ \t]*item \[\d+\]:[ \t]*$", re.MULTILINE)
_LONG_INTERVAL_RE = re.compile(
    rf"intervals \[\d+\]:\s*xmin\s*=\s*({_NUMBER_PATTERN})"
    rf"\s*xmax\s*=\s*({_NUMBER_PATTERN})"
    r'\s*text\s*=\s*"((?:""|[^"])*)"',
    re.MULTILINE,
)
_MAX_TIME_MAP_RENDER_ATTEMPTS = 6
_LONG_INTERVAL_SIZE_RE = re.compile(
    r"^[ \t]*intervals:[ \t]*size[ \t]*=[ \t]*(\d+)[ \t]*$",
    re.MULTILINE,
)
_LONG_INTERVAL_HEADER_RE = re.compile(
    r"^[ \t]*intervals \[(\d+)\]:[ \t]*$",
    re.MULTILINE,
)
_LONG_CLASS_RE = re.compile(r'^\s*class\s*=\s*"((?:""|[^"])*)"\s*$', re.MULTILINE)
_LONG_NAME_RE = re.compile(r'^\s*name\s*=\s*"((?:""|[^"])*)"\s*$', re.MULTILINE)
_PHONE_SUFFIX_RE = re.compile(r"\d+$")


StretchRegionFn = Callable[[np.ndarray, int, int], np.ndarray]
FullChunkWarpFn = Callable[
    [np.ndarray, int, int, tuple[tuple[int, int], ...]],
    np.ndarray,
]
WORD_TIER_FALLBACK_PREFIX = "WORD_TIER_FALLBACK:"
UNKNOWN_PLANNED_VOWEL = "UNKNOWN_PLANNED_VOWEL"
MIN_WARP_REGION_SECONDS = 0.010


class PhoneVowelMismatchError(ValueError):
    """Strict phone-tier vowels do not match the planned syllables."""


class MissingPlannedVowelError(PhoneVowelMismatchError):
    """A planned syllable has no ARPAbet vowel for strict matching."""


@dataclass(frozen=True)
class PhoneInterval:
    start_seconds: float
    end_seconds: float
    phone: str

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("phone intervals must have positive duration")
        if not self.phone:
            raise ValueError("phone intervals must not be empty")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class WordInterval:
    start_seconds: float
    end_seconds: float
    word: str

    def __post_init__(self) -> None:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("word intervals must have positive duration")
        if not self.word:
            raise ValueError("word intervals must not be empty")

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True)
class VowelAnchor:
    word: str
    index_in_word: int
    planned_phone: str
    aligned_phone: str
    requested_source_seconds: float
    source_seconds: float
    requested_target_seconds: float
    target_seconds: float
    requested_source_sample: int
    source_sample: int
    target_sample: int
    source_boundary_adjusted: bool
    boundary_adjusted: bool
    anchor_kind: str = "vowel"


@dataclass(frozen=True)
class StretchRegionDiagnostic:
    source_start_sample: int
    source_end_sample: int
    target_start_sample: int
    target_end_sample: int
    stretch_ratio: float


@dataclass(frozen=True)
class WarpedChunk:
    samples: np.ndarray
    sample_rate_hz: int
    source_sha256: str
    anchor_map: tuple[VowelAnchor, ...]
    stretch_regions: tuple[StretchRegionDiagnostic, ...]


@dataclass(frozen=True)
class GentleSparseAnchorSelection:
    anchors: tuple[VowelAnchor, ...]
    regularized_anchors: tuple[VowelAnchor, ...]
    selected_indices: tuple[int, ...]

    @property
    def omitted_indices(self) -> tuple[int, ...]:
        selected = frozenset(self.selected_indices)
        return tuple(
            index
            for index in range(len(self.regularized_anchors))
            if index not in selected
        )


class RubberBandStretcher:
    """Lazy CLI boundary for pitch-preserving local stretching."""

    def __init__(self, *, binary: str = "rubberband", extra_args: Sequence[str] = ()) -> None:
        self._binary = binary
        self._extra_args = tuple(extra_args)

    def __call__(self, samples: np.ndarray, target_frames: int, sample_rate_hz: int) -> np.ndarray:
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        with tempfile.TemporaryDirectory(prefix="streammuse-rubberband-") as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            wavfile.write(input_path, sample_rate_hz, np.asarray(samples, dtype=np.float32))
            target_seconds = target_frames / sample_rate_hz
            try:
                subprocess.run(
                    [
                        self._binary,
                        "--quiet",
                        "--duration",
                        f"{target_seconds:.9f}",
                        *self._extra_args,
                        str(input_path),
                        str(output_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except FileNotFoundError as exc:
                raise RuntimeError("rubberband binary is not installed") from exc
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.strip() if exc.stderr else "unknown rubberband failure"
                raise RuntimeError(f"rubberband stretch failed: {stderr}") from exc
            _, stretched = wavfile.read(output_path)
        return _enforce_target_length(
            _to_mono_float32(stretched),
            target_frames,
            output_name="Rubber Band",
        )


class RubberBandTimeMapStretcher:
    """Apply one pitch-preserving Rubber Band warp to a complete chunk."""

    def __init__(
        self,
        *,
        binary: str = "rubberband",
        engine: str = "r3",
        smoothing: bool = False,
        extra_args: Sequence[str] = (),
    ) -> None:
        if engine not in {"r2", "r3"}:
            raise ValueError("engine must be 'r2' or 'r3'")
        if engine == "r3" and smoothing:
            raise ValueError("Rubber Band smoothing is available only with the R2 engine")
        self._binary = binary
        self._engine = engine
        self._smoothing = smoothing
        self._extra_args = tuple(extra_args)

    def __call__(
        self,
        samples: np.ndarray,
        target_frames: int,
        sample_rate_hz: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        if target_frames <= 0:
            raise ValueError("target_frames must be positive")
        _validate_time_map(time_map)
        with tempfile.TemporaryDirectory(prefix="streammuse-rubberband-timemap-") as temp_dir:
            input_path = Path(temp_dir) / "input.wav"
            output_path = Path(temp_dir) / "output.wav"
            time_map_path = Path(temp_dir) / "time_map.txt"
            wavfile.write(input_path, sample_rate_hz, np.asarray(samples, dtype=np.float32))
            engine_args = ["--fine"] if self._engine == "r3" else []
            if self._smoothing:
                engine_args.append("--smoothing")
            active_time_map = time_map
            for attempt in range(_MAX_TIME_MAP_RENDER_ATTEMPTS):
                time_map_path.write_text(
                    "".join(
                        f"{source} {target}\n"
                        for source, target in active_time_map
                    ),
                    encoding="utf-8",
                )
                output_path.unlink(missing_ok=True)
                try:
                    subprocess.run(
                        [
                            self._binary,
                            "--quiet",
                            "--duration",
                            f"{target_frames / sample_rate_hz:.9f}",
                            "--timemap",
                            str(time_map_path),
                            *engine_args,
                            *self._extra_args,
                            str(input_path),
                            str(output_path),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError("rubberband binary is not installed") from exc
                except subprocess.CalledProcessError as exc:
                    stderr = exc.stderr.strip() if exc.stderr else "unknown rubberband failure"
                    raise RuntimeError(f"rubberband time-map warp failed: {stderr}") from exc
                _, stretched = wavfile.read(output_path)
                stretched_mono = _to_mono_float32(stretched)
                if abs(len(stretched_mono) - target_frames) <= 2:
                    return _enforce_target_length(
                        stretched_mono,
                        target_frames,
                        output_name="Rubber Band",
                    )
                if attempt + 1 < _MAX_TIME_MAP_RENDER_ATTEMPTS:
                    if len(stretched_mono) == 0:
                        continue
                    endpoint_delta = target_frames - len(stretched_mono)
                    source_endpoint, target_endpoint = active_time_map[-1]
                    corrected_endpoint = target_endpoint + endpoint_delta
                    if corrected_endpoint <= active_time_map[-2][1]:
                        return _enforce_target_length(
                            stretched_mono,
                            target_frames,
                            output_name="Rubber Band",
                        )
                    active_time_map = (
                        *active_time_map[:-1],
                        (source_endpoint, corrected_endpoint),
                    )
                    continue
                return _enforce_target_length(
                    stretched_mono,
                    target_frames,
                    output_name="Rubber Band",
                )

        raise RuntimeError("rubberband time-map duration fitting exhausted unexpectedly")


def is_arpabet_vowel(phone: str) -> bool:
    return _normalise_phone(phone) in _ARPABET_VOWELS


def parse_textgrid_phone_intervals(text: str) -> tuple[PhoneInterval, ...]:
    if _LONG_TIER_RE.search(text):
        intervals = _parse_long_textgrid_phone_intervals(text)
    else:
        intervals = _parse_short_textgrid_phone_intervals(text)
    if not intervals:
        raise ValueError("no phone intervals found in TextGrid")
    return intervals


def load_textgrid_phone_intervals(path: Path | str) -> tuple[PhoneInterval, ...]:
    return parse_textgrid_phone_intervals(Path(path).read_text(encoding="utf-8"))


def parse_textgrid_word_intervals(text: str) -> tuple[WordInterval, ...]:
    if _LONG_TIER_RE.search(text):
        intervals = _parse_long_textgrid_word_intervals(text)
    else:
        intervals = _parse_short_textgrid_word_intervals(text)
    if not intervals:
        raise ValueError("no word intervals found in TextGrid")
    return intervals


def load_textgrid_word_intervals(path: Path | str) -> tuple[WordInterval, ...]:
    return parse_textgrid_word_intervals(Path(path).read_text(encoding="utf-8"))


def match_vowel_anchors(
    phone_intervals: Sequence[PhoneInterval],
    syllables: Sequence[SyllableTarget],
    *,
    sample_rate_hz: int,
    target_duration_seconds: float | None = None,
) -> tuple[VowelAnchor, ...]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    source_vowels = tuple(interval for interval in phone_intervals if is_arpabet_vowel(interval.phone))
    planned_vowels = tuple((_primary_vowel_phone(syllable), syllable) for syllable in syllables)
    if len(source_vowels) != len(planned_vowels):
        raise PhoneVowelMismatchError(
            f"aligned vowel count mismatch: expected {len(planned_vowels)}, got {len(source_vowels)}"
        )

    anchors = []
    for index, (source, (planned_phone, syllable)) in enumerate(zip(source_vowels, planned_vowels)):
        if _normalise_phone(planned_phone) != _normalise_phone(source.phone):
            raise PhoneVowelMismatchError(
                "strict vowel-anchor matching failed "
                f"at syllable {index}: expected {planned_phone}, got {source.phone}"
            )
        anchors.append(
            _phone_vowel_anchor(
                source,
                planned_phone=planned_phone,
                syllable=syllable,
                sample_rate_hz=sample_rate_hz,
            )
        )
    return _apply_target_boundary_policy(
        anchors,
        sample_rate_hz=sample_rate_hz,
        target_duration_seconds=target_duration_seconds,
    )


def match_vowel_anchors_with_word_fallback(
    phone_intervals: Sequence[PhoneInterval],
    word_intervals: Sequence[WordInterval],
    syllables: Sequence[SyllableTarget],
    *,
    sample_rate_hz: int,
    request_words: Sequence[str] | None = None,
    target_duration_seconds: float | None = None,
) -> tuple[VowelAnchor, ...]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    syllable_groups = _group_syllables_by_word(syllables)
    if request_words is not None:
        _validate_request_word_sequence(request_words, syllable_groups)
    matched_words = _match_word_intervals(syllable_groups, word_intervals)
    owned_vowels = _assign_vowels_to_matched_words(phone_intervals, matched_words)

    anchors: list[VowelAnchor] = []
    for syllable_group, word_interval, aligned_vowels in zip(
        syllable_groups,
        matched_words,
        owned_vowels,
    ):
        planned_vowels = tuple(
            (_word_fallback_vowel_phone(syllable), syllable)
            for syllable in syllable_group
        )
        phones_match = len(aligned_vowels) == len(planned_vowels) and all(
            planned_phone != UNKNOWN_PLANNED_VOWEL
            and _normalise_phone(source.phone) == _normalise_phone(planned_phone)
            for source, (planned_phone, _) in zip(aligned_vowels, planned_vowels)
        )
        if phones_match:
            anchors.extend(
                _phone_vowel_anchor(
                    source,
                    planned_phone=planned_phone,
                    syllable=syllable,
                    sample_rate_hz=sample_rate_hz,
                )
                for source, (planned_phone, syllable) in zip(aligned_vowels, planned_vowels)
            )
            continue

        for index, (planned_phone, syllable) in enumerate(planned_vowels):
            source_seconds = word_interval.start_seconds + (
                word_interval.duration_seconds * (index + 1) / (len(planned_vowels) + 1)
            )
            anchors.append(
                VowelAnchor(
                    word=syllable.word,
                    index_in_word=syllable.index_in_word,
                    planned_phone=planned_phone,
                    aligned_phone=f"{WORD_TIER_FALLBACK_PREFIX}{word_interval.word}",
                    requested_source_seconds=source_seconds,
                    source_seconds=source_seconds,
                    requested_target_seconds=syllable.target_seconds,
                    target_seconds=syllable.target_seconds,
                    requested_source_sample=round(source_seconds * sample_rate_hz),
                    source_sample=round(source_seconds * sample_rate_hz),
                    target_sample=round(syllable.target_seconds * sample_rate_hz),
                    source_boundary_adjusted=False,
                    boundary_adjusted=False,
                )
            )

    anchors = list(
        _apply_target_boundary_policy(
            anchors,
            sample_rate_hz=sample_rate_hz,
            target_duration_seconds=target_duration_seconds,
        )
    )
    _validate_fallback_anchor_monotonicity(anchors)
    return tuple(anchors)


def promote_vowel_anchors_to_syllable_onsets(
    phone_intervals: Sequence[PhoneInterval],
    syllables: Sequence[SyllableTarget],
    anchors: Sequence[VowelAnchor],
    *,
    sample_rate_hz: int,
) -> tuple[VowelAnchor, ...]:
    """Move strict vowel anchors to aligned syllable onsets when phones agree."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if len(syllables) != len(anchors):
        raise ValueError("syllables and anchors must have equal lengths")

    promoted: list[VowelAnchor] = []
    search_start = 0
    for syllable, anchor in zip(syllables, anchors):
        if anchor.aligned_phone.startswith(WORD_TIER_FALLBACK_PREFIX):
            promoted.append(anchor)
            continue
        vowel_index = _find_aligned_vowel_interval(
            phone_intervals,
            anchor,
            search_start=search_start,
        )
        if vowel_index is None:
            promoted.append(anchor)
            continue
        search_start = vowel_index + 1
        planned_vowel_index = _primary_vowel_index(syllable)
        onset_index = vowel_index - planned_vowel_index
        if onset_index < 0:
            promoted.append(anchor)
            continue
        planned_onset = tuple(
            _normalise_phone(phone) for phone in syllable.phonemes[:planned_vowel_index]
        )
        aligned_onset = tuple(
            _normalise_phone(interval.phone)
            for interval in phone_intervals[onset_index:vowel_index]
        )
        if aligned_onset != planned_onset:
            promoted.append(anchor)
            continue
        source_seconds = phone_intervals[onset_index].start_seconds
        source_sample = round(source_seconds * sample_rate_hz)
        promoted.append(
            replace(
                anchor,
                requested_source_seconds=source_seconds,
                source_seconds=source_seconds,
                requested_source_sample=source_sample,
                source_sample=source_sample,
                source_boundary_adjusted=False,
                anchor_kind="syllable_onset",
            )
        )
    return tuple(promoted)


def regularize_anchor_targets(
    anchors: Sequence[VowelAnchor],
    syllables: Sequence[SyllableTarget],
    *,
    sample_rate_hz: int,
    source_frame_count: int,
    target_frame_count: int,
    min_stretch_ratio: float = 0.5,
    max_stretch_ratio: float = 2.0,
    stress_priority: float = 4.0,
) -> tuple[VowelAnchor, ...]:
    """Move target anchors minimally so local stretch ratios stay bounded."""
    if len(anchors) != len(syllables):
        raise ValueError("anchors and syllables must have equal lengths")
    if not anchors:
        raise ValueError("at least one anchor is required")
    if min_stretch_ratio <= 0 or max_stretch_ratio < min_stretch_ratio:
        raise ValueError("stretch ratio bounds are invalid")
    if stress_priority < 0:
        raise ValueError("stress_priority must not be negative")

    effective_anchors = _apply_source_boundary_policy(
        anchors,
        sample_rate_hz=sample_rate_hz,
        source_frame_count=source_frame_count,
    )
    source_points = np.asarray(
        [0, *(anchor.source_sample for anchor in effective_anchors), source_frame_count - 1],
        dtype=np.float64,
    )
    source_gaps = np.diff(source_points)
    target_end = target_frame_count - 1
    lower_gaps = min_stretch_ratio * source_gaps
    upper_gaps = max_stretch_ratio * source_gaps
    if not float(np.sum(lower_gaps)) <= target_end <= float(np.sum(upper_gaps)):
        raise ValueError("stretch ratio bounds are infeasible for the source and target durations")

    anchor_count = len(effective_anchors)
    difference_matrix = np.zeros((anchor_count + 1, anchor_count), dtype=np.float64)
    difference_matrix[0, 0] = 1.0
    for row in range(1, anchor_count):
        difference_matrix[row, row - 1] = -1.0
        difference_matrix[row, row] = 1.0
    difference_matrix[-1, -1] = -1.0
    lower = np.asarray(lower_gaps / target_end, dtype=np.float64)
    upper = np.asarray(upper_gaps / target_end, dtype=np.float64)
    lower[-1] -= 1.0
    upper[-1] -= 1.0

    requested_samples = np.asarray(
        [round(anchor.requested_target_seconds * sample_rate_hz) for anchor in effective_anchors],
        dtype=np.float64,
    )
    requested = requested_samples / target_end
    weights = np.asarray(
        [1.0 + stress_priority * syllable.target_stress for syllable in syllables],
        dtype=np.float64,
    )
    initial = source_points[1:-1] / (source_frame_count - 1)

    def objective(values: np.ndarray) -> float:
        return float(np.sum(weights * np.square(values - requested)))

    def gradient(values: np.ndarray) -> np.ndarray:
        return 2.0 * weights * (values - requested)

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds(
            np.full(anchor_count, 1.0 / target_end, dtype=np.float64),
            np.full(anchor_count, (target_end - 1) / target_end, dtype=np.float64),
        ),
        constraints=(LinearConstraint(difference_matrix, lower, upper),),
        options={"ftol": 1e-12, "maxiter": 500},
    )
    if not result.success:
        raise ValueError(f"target-anchor regularization failed: {result.message}")

    target_samples = np.rint(result.x * target_end).astype(int)
    regularized = tuple(
        replace(
            anchor,
            target_sample=int(target_sample),
            target_seconds=int(target_sample) / sample_rate_hz,
            boundary_adjusted=(
                anchor.boundary_adjusted or int(target_sample) != anchor.target_sample
            ),
        )
        for anchor, target_sample in zip(effective_anchors, target_samples)
    )
    _validate_anchor_monotonicity(
        tuple(anchor.source_sample for anchor in regularized),
        tuple(anchor.target_sample for anchor in regularized),
        source_frame_count,
        target_frame_count,
    )
    return regularized


def regularize_gentle_sparse_anchors(
    anchors: Sequence[VowelAnchor],
    syllables: Sequence[SyllableTarget],
    *,
    sample_rate_hz: int,
    source_frame_count: int,
    target_frame_count: int,
    min_stretch_ratio: float = 0.75,
    max_stretch_ratio: float = 1.35,
    minimum_target_stress: float = 0.8,
    minimum_boundary_strength: int = 2,
) -> GentleSparseAnchorSelection:
    """Regularize all onsets, then retain only rhythmically salient anchors."""
    regularized = regularize_anchor_targets(
        anchors,
        syllables,
        sample_rate_hz=sample_rate_hz,
        source_frame_count=source_frame_count,
        target_frame_count=target_frame_count,
        min_stretch_ratio=min_stretch_ratio,
        max_stretch_ratio=max_stretch_ratio,
    )
    last_index = len(regularized) - 1
    selected_indices = tuple(
        index
        for index, syllable in enumerate(syllables)
        if index in {0, last_index}
        or syllable.target_stress >= minimum_target_stress
        or syllable.boundary_strength >= minimum_boundary_strength
    )
    return GentleSparseAnchorSelection(
        anchors=tuple(regularized[index] for index in selected_indices),
        regularized_anchors=regularized,
        selected_indices=selected_indices,
    )


def continuous_pitch_preserving_warp(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    anchors: Sequence[VowelAnchor],
    target_frame_count: int,
    stretch_full_chunk: FullChunkWarpFn | None = None,
    source_sha256: str,
) -> WarpedChunk:
    """Warp a chunk in one process using source-to-target anchor pairs."""
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be positive")
    if not anchors:
        raise ValueError("at least one anchor is required")

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(mono) == 0:
        raise ValueError("samples must not be empty")
    effective_anchors = _apply_source_boundary_policy(
        anchors,
        sample_rate_hz=sample_rate_hz,
        source_frame_count=len(mono),
    )
    source_anchors = tuple(anchor.source_sample for anchor in effective_anchors)
    target_anchors = tuple(anchor.target_sample for anchor in effective_anchors)
    _validate_anchor_monotonicity(
        source_anchors,
        target_anchors,
        len(mono),
        target_frame_count,
    )
    time_map = tuple(
        zip(
            (0, *source_anchors, len(mono) - 1),
            (0, *target_anchors, target_frame_count - 1),
        )
    )
    _validate_time_map(time_map)
    warped = (stretch_full_chunk or RubberBandTimeMapStretcher())(
        mono,
        target_frame_count,
        sample_rate_hz,
        time_map,
    )
    warped = _enforce_target_length(warped, target_frame_count)
    diagnostics = tuple(
        StretchRegionDiagnostic(
            source_start_sample=source_start,
            source_end_sample=source_end,
            target_start_sample=target_start,
            target_end_sample=target_end,
            stretch_ratio=(target_end - target_start) / max(1, source_end - source_start),
        )
        for (source_start, target_start), (source_end, target_end) in zip(
            time_map,
            time_map[1:],
        )
    )
    return WarpedChunk(
        samples=warped,
        sample_rate_hz=sample_rate_hz,
        source_sha256=source_sha256,
        anchor_map=effective_anchors,
        stretch_regions=diagnostics,
    )


def piecewise_pitch_preserving_warp(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    anchors: Sequence[VowelAnchor],
    target_frame_count: int,
    stretch_region: StretchRegionFn | None = None,
    crossfade_seconds: float = 0.005,
    min_region_seconds: float = MIN_WARP_REGION_SECONDS,
    source_sha256: str,
) -> WarpedChunk:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be positive")
    if not anchors:
        raise ValueError("at least one anchor is required")

    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(mono) == 0:
        raise ValueError("samples must not be empty")
    effective_anchors = _apply_source_boundary_policy(
        anchors,
        sample_rate_hz=sample_rate_hz,
        source_frame_count=len(mono),
    )
    stretcher = stretch_region or RubberBandStretcher()

    _validate_anchor_monotonicity(
        tuple(anchor.source_sample for anchor in effective_anchors),
        tuple(anchor.target_sample for anchor in effective_anchors),
        len(mono),
        target_frame_count,
    )
    source_points, target_points = _build_control_points(
        tuple(anchor.source_sample for anchor in effective_anchors),
        tuple(anchor.target_sample for anchor in effective_anchors),
        len(mono),
        target_frame_count,
    )

    minimum_region_samples = max(1, round(min_region_seconds * sample_rate_hz))
    _validate_region_lengths(source_points, target_points, minimum_region_samples)

    rendered_regions: list[np.ndarray] = []
    diagnostics: list[StretchRegionDiagnostic] = []
    for region_index, (source_start, source_end, target_start, target_end) in enumerate(
        zip(source_points, source_points[1:], target_points, target_points[1:])
    ):
        source_region = mono[source_start : source_end + 1]
        target_length = (target_end - target_start) + 1
        stretched = _enforce_target_length(stretcher(source_region, target_length, sample_rate_hz), target_length)
        if region_index > 0:
            rendered_regions[-1][-1] = _preserve_boundary_sample(
                rendered_regions[-1][-1],
                stretched[0],
            )
            stretched = stretched[1:]
        rendered_regions.append(stretched)
        diagnostics.append(
            StretchRegionDiagnostic(
                source_start_sample=source_start,
                source_end_sample=source_end,
                target_start_sample=target_start,
                target_end_sample=target_end,
                stretch_ratio=(target_end - target_start) / max(1, source_end - source_start),
            )
        )

    warped = np.concatenate(rendered_regions, axis=0)
    if len(warped) != target_frame_count:
        raise ValueError(f"piecewise warp produced {len(warped)} frames, expected {target_frame_count}")
    warped = _apply_equal_power_crossfade(warped, rendered_regions, tuple(target_points), sample_rate_hz, crossfade_seconds)
    return WarpedChunk(
        samples=warped,
        sample_rate_hz=sample_rate_hz,
        source_sha256=source_sha256,
        anchor_map=effective_anchors,
        stretch_regions=tuple(diagnostics),
    )


def _apply_equal_power_crossfade(
    warped: np.ndarray,
    rendered_regions: Sequence[np.ndarray],
    target_points: tuple[int, ...],
    sample_rate_hz: int,
    crossfade_seconds: float,
) -> np.ndarray:
    crossfade_samples = round(crossfade_seconds * sample_rate_hz)
    if crossfade_samples <= 1:
        return warped

    output = np.array(warped, copy=True)
    for join_index, boundary in enumerate(target_points[1:-1], start=1):
        left_region = rendered_regions[join_index - 1]
        right_region = rendered_regions[join_index]
        overlap = min(crossfade_samples, len(left_region) - 1, len(right_region))
        if overlap <= 1:
            continue
        left_tail = left_region[-overlap - 1 : -1]
        right_head = right_region[:overlap]
        theta = np.linspace(0.0, np.pi / 2.0, overlap + 1, dtype=np.float32)[1:]
        blend = (left_tail * np.cos(theta)) + (right_head * np.sin(theta))
        output[boundary + 1 : boundary + overlap + 1] = blend
    return output.astype(np.float32, copy=False)


def _preserve_boundary_sample(left_sample: np.float32, right_sample: np.float32) -> np.float32:
    if abs(float(right_sample)) > abs(float(left_sample)):
        return np.float32(right_sample)
    return np.float32(left_sample)


def _parse_long_textgrid_phone_intervals(text: str) -> tuple[PhoneInterval, ...]:
    return _phone_intervals_from_values(_parse_long_textgrid_interval_values(text, tier_name="phones"))


def _parse_long_textgrid_word_intervals(text: str) -> tuple[WordInterval, ...]:
    return _word_intervals_from_values(_parse_long_textgrid_interval_values(text, tier_name="words"))


def _parse_long_textgrid_interval_values(
    text: str,
    *,
    tier_name: str,
) -> tuple[tuple[str, str, str], ...]:
    tier_matches = tuple(_LONG_TIER_RE.finditer(text))
    for index, tier_match in enumerate(tier_matches):
        end = tier_matches[index + 1].start() if index + 1 < len(tier_matches) else len(text)
        tier = text[tier_match.end() : end]
        class_match = _LONG_CLASS_RE.search(tier)
        name_match = _LONG_NAME_RE.search(tier)
        if class_match is None or name_match is None:
            continue
        tier_class = _decode_praat_string(class_match.group(1))
        parsed_tier_name = _decode_praat_string(name_match.group(1))
        if tier_class != "IntervalTier" or parsed_tier_name.casefold() != tier_name.casefold():
            continue
        size_matches = tuple(_LONG_INTERVAL_SIZE_RE.finditer(tier))
        if len(size_matches) != 1:
            raise ValueError(
                f"malformed {tier_name} IntervalTier: expected exactly one declared interval count"
            )
        declared_count = int(size_matches[0].group(1))
        interval_headers = tuple(_LONG_INTERVAL_HEADER_RE.finditer(tier))
        if len(interval_headers) != declared_count:
            raise ValueError(
                f"malformed {tier_name} IntervalTier: declared {declared_count} intervals, "
                f"found {len(interval_headers)} interval blocks"
            )

        values = []
        for expected_index, header in enumerate(interval_headers, start=1):
            interval_index = int(header.group(1))
            if interval_index != expected_index:
                raise ValueError(
                    f"malformed {tier_name} IntervalTier: expected interval {expected_index}, "
                    f"found interval {interval_index}"
                )
            block_end = (
                interval_headers[expected_index].start()
                if expected_index < len(interval_headers)
                else len(tier)
            )
            interval_match = _LONG_INTERVAL_RE.fullmatch(tier[header.start() : block_end].strip())
            if interval_match is None:
                raise ValueError(
                    f"malformed {tier_name} IntervalTier: interval {interval_index} is incomplete"
                )
            values.append(interval_match.groups())
        return tuple(values)
    raise ValueError(f"{tier_name} IntervalTier not found in TextGrid")


def _parse_short_textgrid_phone_intervals(text: str) -> tuple[PhoneInterval, ...]:
    return _phone_intervals_from_values(_parse_short_textgrid_interval_values(text, tier_name="phones"))


def _parse_short_textgrid_word_intervals(text: str) -> tuple[WordInterval, ...]:
    return _word_intervals_from_values(_parse_short_textgrid_interval_values(text, tier_name="words"))


def _parse_short_textgrid_interval_values(
    text: str,
    *,
    tier_name: str,
) -> tuple[tuple[str, str, str], ...]:
    values = [line.strip() for line in text.lstrip("\ufeff").splitlines() if line.strip()]
    try:
        cursor = values.index("<exists>") + 1
    except ValueError as exc:
        raise ValueError("short TextGrid is missing tier metadata") from exc

    def take() -> str:
        nonlocal cursor
        if cursor >= len(values):
            raise ValueError(f"short TextGrid ended before the {tier_name} tier was complete")
        value = values[cursor]
        cursor += 1
        return value

    tier_count = _parse_short_count(take(), field_name="tier count")
    tier_values: list[tuple[str, str, str]] | None = None
    for _ in range(tier_count):
        tier_class = _parse_short_string(take())
        parsed_tier_name = _parse_short_string(take())
        _parse_short_number(take(), field_name="tier xmin")
        _parse_short_number(take(), field_name="tier xmax")
        entry_count = _parse_short_count(take(), field_name="tier entry count")
        if tier_class == "IntervalTier":
            entries = []
            for _ in range(entry_count):
                entries.append((take(), take(), _parse_short_string(take())))
            if parsed_tier_name.casefold() == tier_name.casefold():
                tier_values = entries
        elif tier_class == "TextTier":
            for _ in range(entry_count):
                take()
                _parse_short_string(take())
        else:
            raise ValueError(f"unsupported short TextGrid tier class: {tier_class}")
    if tier_values is None:
        raise ValueError(f"{tier_name} IntervalTier not found in TextGrid")
    return tuple(tier_values)


def _phone_intervals_from_values(
    values: Sequence[tuple[str, str, str]],
) -> tuple[PhoneInterval, ...]:
    intervals = []
    for start_seconds, end_seconds, raw_phone in values:
        label = _decode_praat_string(raw_phone).strip()
        if not label:
            continue
        intervals.append(
            PhoneInterval(
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds),
                phone=label,
            )
        )
    return tuple(intervals)


def _word_intervals_from_values(
    values: Sequence[tuple[str, str, str]],
) -> tuple[WordInterval, ...]:
    intervals = []
    for start_seconds, end_seconds, raw_word in values:
        label = _decode_praat_string(raw_word).strip()
        if not label:
            continue
        intervals.append(
            WordInterval(
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds),
                word=label,
            )
        )
    return tuple(intervals)


def _parse_short_string(value: str) -> str:
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise ValueError(f"expected quoted short TextGrid string, got {value!r}")
    return _decode_praat_string(value[1:-1])


def _parse_short_number(value: str, *, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid short TextGrid {field_name}: {value!r}") from exc


def _parse_short_count(value: str, *, field_name: str) -> int:
    number = _parse_short_number(value, field_name=field_name)
    if not number.is_integer() or number < 0:
        raise ValueError(f"invalid short TextGrid {field_name}: {value!r}")
    return int(number)


def _decode_praat_string(value: str) -> str:
    return value.replace('""', '"')


def _build_control_points(
    source_anchor_samples: tuple[int, ...],
    target_anchor_samples: tuple[int, ...],
    source_frame_count: int,
    target_frame_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    source_points = [0]
    target_points = [0]
    for index, (source_anchor, target_anchor) in enumerate(zip(source_anchor_samples, target_anchor_samples)):
        source_points.append(source_anchor)
        target_points.append(target_anchor)
        if index + 1 < len(source_anchor_samples):
            source_points.append(round((source_anchor + source_anchor_samples[index + 1]) / 2))
            target_points.append(round((target_anchor + target_anchor_samples[index + 1]) / 2))
    source_points.append(source_frame_count - 1)
    target_points.append(target_frame_count - 1)
    return tuple(source_points), tuple(target_points)


def _validate_anchor_monotonicity(
    source_anchor_samples: tuple[int, ...],
    target_anchor_samples: tuple[int, ...],
    source_frame_count: int,
    target_frame_count: int,
) -> None:
    if any(sample <= 0 or sample >= source_frame_count - 1 for sample in source_anchor_samples):
        raise ValueError("source anchors must lie strictly inside the source audio")
    if any(sample <= 0 or sample >= target_frame_count - 1 for sample in target_anchor_samples):
        raise ValueError("target anchors must lie strictly inside the target audio")
    if tuple(source_anchor_samples) != tuple(sorted(source_anchor_samples)):
        raise ValueError("source anchors must be strictly monotonic")
    if len(set(source_anchor_samples)) != len(source_anchor_samples):
        raise ValueError("source anchors must be strictly monotonic")
    if tuple(target_anchor_samples) != tuple(sorted(target_anchor_samples)):
        raise ValueError("target anchors must be strictly monotonic")
    if len(set(target_anchor_samples)) != len(target_anchor_samples):
        raise ValueError("target anchors must be strictly monotonic")


def _validate_region_lengths(
    source_points: tuple[int, ...],
    target_points: tuple[int, ...],
    minimum_region_samples: int,
) -> None:
    for source_start, source_end, target_start, target_end in zip(
        source_points,
        source_points[1:],
        target_points,
        target_points[1:],
    ):
        if (source_end - source_start) < minimum_region_samples:
            raise ValueError("source region shorter than 10 ms is impossible to warp explicitly")
        if (target_end - target_start) < minimum_region_samples:
            raise ValueError("target region shorter than 10 ms is impossible to warp explicitly")


def _phone_vowel_anchor(
    source: PhoneInterval,
    *,
    planned_phone: str,
    syllable: SyllableTarget,
    sample_rate_hz: int,
) -> VowelAnchor:
    source_seconds = source.start_seconds + min(0.030, 0.25 * source.duration_seconds)
    return VowelAnchor(
        word=syllable.word,
        index_in_word=syllable.index_in_word,
        planned_phone=planned_phone,
        aligned_phone=source.phone,
        requested_source_seconds=source_seconds,
        source_seconds=source_seconds,
        requested_target_seconds=syllable.target_seconds,
        target_seconds=syllable.target_seconds,
        requested_source_sample=round(source_seconds * sample_rate_hz),
        source_sample=round(source_seconds * sample_rate_hz),
        target_sample=round(syllable.target_seconds * sample_rate_hz),
        source_boundary_adjusted=False,
        boundary_adjusted=False,
    )


def _apply_source_boundary_policy(
    anchors: Sequence[VowelAnchor],
    *,
    sample_rate_hz: int,
    source_frame_count: int,
) -> tuple[VowelAnchor, ...]:
    margin_samples = max(1, round(MIN_WARP_REGION_SECONDS * sample_rate_hz))
    first_interior_sample = margin_samples
    last_interior_sample = (source_frame_count - 1) - margin_samples
    if first_interior_sample >= last_interior_sample:
        raise ValueError("source audio is too short for the source anchor boundary margin")

    adjusted: list[VowelAnchor] = []
    last_anchor_index = len(anchors) - 1
    for index, anchor in enumerate(anchors):
        requested_seconds = anchor.requested_source_seconds
        requested_sample = anchor.requested_source_sample
        if (
            requested_seconds < 0
            or requested_sample < 0
            or requested_sample >= source_frame_count
        ):
            raise ValueError(
                "source anchor lies outside the source audio: "
                f"requested {requested_seconds:.9f}s/sample {requested_sample} "
                f"for {source_frame_count} frames"
            )
        if index == 0 and requested_sample < first_interior_sample:
            effective_sample = first_interior_sample
        elif index == last_anchor_index and requested_sample > last_interior_sample:
            effective_sample = last_interior_sample
        else:
            adjusted.append(anchor)
            continue
        adjusted.append(
            replace(
                anchor,
                source_seconds=effective_sample / sample_rate_hz,
                source_sample=effective_sample,
                source_boundary_adjusted=True,
            )
        )

    if any(anchor.source_boundary_adjusted for anchor in adjusted) and any(
        right.source_sample <= left.source_sample
        for left, right in zip(adjusted, adjusted[1:])
    ):
        raise ValueError("source boundary adjustment collides with an adjacent anchor")
    return tuple(adjusted)


def _apply_target_boundary_policy(
    anchors: Sequence[VowelAnchor],
    *,
    sample_rate_hz: int,
    target_duration_seconds: float | None,
) -> tuple[VowelAnchor, ...]:
    if target_duration_seconds is None:
        return tuple(anchors)
    if target_duration_seconds <= 0:
        raise ValueError("target_duration_seconds must be positive")

    target_frame_count = round(target_duration_seconds * sample_rate_hz)
    margin_samples = max(1, round(MIN_WARP_REGION_SECONDS * sample_rate_hz))
    first_interior_sample = margin_samples
    last_interior_sample = (target_frame_count - 1) - margin_samples
    if first_interior_sample >= last_interior_sample:
        raise ValueError("target audio is too short for the target anchor boundary margin")

    adjusted: list[VowelAnchor] = []
    for anchor in anchors:
        requested_seconds = anchor.requested_target_seconds
        if requested_seconds < 0 or requested_seconds > target_duration_seconds:
            raise ValueError(
                "target anchor lies outside the target audio: "
                f"requested {requested_seconds:.9f}s for duration {target_duration_seconds:.9f}s"
            )
        if requested_seconds == 0:
            effective_sample = first_interior_sample
        elif requested_seconds == target_duration_seconds:
            effective_sample = last_interior_sample
        else:
            adjusted.append(anchor)
            continue
        adjusted.append(
            replace(
                anchor,
                target_seconds=effective_sample / sample_rate_hz,
                target_sample=effective_sample,
                boundary_adjusted=True,
            )
        )

    if any(anchor.boundary_adjusted for anchor in adjusted) and any(
        right.target_sample <= left.target_sample
        for left, right in zip(adjusted, adjusted[1:])
    ):
        raise ValueError("target boundary adjustment collides with an adjacent anchor")
    return tuple(adjusted)


def _group_syllables_by_word(
    syllables: Sequence[SyllableTarget],
) -> tuple[tuple[SyllableTarget, ...], ...]:
    groups: list[tuple[SyllableTarget, ...]] = []
    current: list[SyllableTarget] = []
    for syllable in syllables:
        if syllable.index_in_word == 0:
            if current:
                groups.append(tuple(current))
            current = [syllable]
            continue
        if (
            not current
            or syllable.word != current[0].word
            or syllable.index_in_word != len(current)
        ):
            raise ValueError("request syllables cannot be grouped into monotonic words")
        current.append(syllable)
    if current:
        groups.append(tuple(current))
    if not groups:
        raise ValueError("word-tier fallback requires planned syllables")
    return tuple(groups)


def _match_word_intervals(
    syllable_groups: Sequence[Sequence[SyllableTarget]],
    word_intervals: Sequence[WordInterval],
) -> tuple[WordInterval, ...]:
    matched: list[WordInterval] = []
    aligned_index = 0
    for request_index, group in enumerate(syllable_groups):
        expected = _normalise_word(group[0].word)
        if not expected:
            raise ValueError(f"request word {group[0].word!r} cannot be normalized for matching")
        while aligned_index < len(word_intervals):
            candidate = word_intervals[aligned_index]
            aligned_index += 1
            if _normalise_word(candidate.word) == expected:
                matched.append(candidate)
                break
            if _word_interval_is_ignorable(candidate):
                continue
            raise ValueError(
                "word-tier sequence mismatch "
                f"at request word {request_index}: expected {group[0].word!r}, got {candidate.word!r}"
            )
        else:
            raise ValueError(
                "word-tier sequence mismatch "
                f"at request word {request_index}: expected {group[0].word!r}, reached end of tier"
            )

    unexpected = next(
        (interval for interval in word_intervals[aligned_index:] if not _word_interval_is_ignorable(interval)),
        None,
    )
    if unexpected is not None:
        raise ValueError(f"word-tier sequence has unexpected trailing word {unexpected.word!r}")
    return tuple(matched)


def _assign_vowels_to_matched_words(
    phone_intervals: Sequence[PhoneInterval],
    matched_words: Sequence[WordInterval],
) -> tuple[tuple[PhoneInterval, ...], ...]:
    for left, right in zip(matched_words, matched_words[1:]):
        if right.start_seconds < left.end_seconds:
            raise ValueError("matched word intervals must be chronological and non-overlapping")

    owned: list[list[PhoneInterval]] = [[] for _ in matched_words]
    for phone in phone_intervals:
        if not is_arpabet_vowel(phone.phone):
            continue
        owners = tuple(
            index
            for index, word in enumerate(matched_words)
            if _phone_belongs_to_word(phone, word)
        )
        if len(owners) != 1:
            ownership = "unowned" if not owners else "ambiguously owned"
            raise ValueError(
                "aligned vowel ownership must resolve to exactly one matched word: "
                f"{phone.phone!r} at {phone.start_seconds:.6f}-{phone.end_seconds:.6f}s is {ownership}"
            )
        owned[owners[0]].append(phone)
    return tuple(tuple(intervals) for intervals in owned)


def _validate_request_word_sequence(
    request_words: Sequence[str],
    syllable_groups: Sequence[Sequence[SyllableTarget]],
) -> None:
    if len(request_words) != len(syllable_groups):
        raise ValueError(
            "request word sequence does not match planned syllable groups: "
            f"expected {len(syllable_groups)} words, got {len(request_words)}"
        )
    for index, (request_word, group) in enumerate(zip(request_words, syllable_groups)):
        if _normalise_word(request_word) != _normalise_word(group[0].word):
            raise ValueError(
                "request word sequence does not match planned syllable groups "
                f"at word {index}: request has {request_word!r}, syllables have {group[0].word!r}"
            )


def _normalise_word(word: str) -> str:
    return "".join(character for character in word.casefold() if character.isalnum())


def _word_interval_is_ignorable(interval: WordInterval) -> bool:
    return interval.word.strip().casefold() in {"<eps>", "sil", "sp"}


def _phone_belongs_to_word(phone: PhoneInterval, word: WordInterval) -> bool:
    return word.start_seconds <= phone.start_seconds and phone.end_seconds <= word.end_seconds


def _validate_fallback_anchor_monotonicity(anchors: Sequence[VowelAnchor]) -> None:
    source_samples = tuple(anchor.source_sample for anchor in anchors)
    target_samples = tuple(anchor.target_sample for anchor in anchors)
    if any(right <= left for left, right in zip(source_samples, source_samples[1:])):
        raise ValueError("word-tier fallback produced non-monotonic source anchors")
    if any(right <= left for left, right in zip(target_samples, target_samples[1:])):
        raise ValueError("word-tier fallback produced non-monotonic target anchors")


def _normalise_phone(phone: str) -> str:
    return _PHONE_SUFFIX_RE.sub("", phone.strip().upper())


def _find_aligned_vowel_interval(
    phone_intervals: Sequence[PhoneInterval],
    anchor: VowelAnchor,
    *,
    search_start: int,
) -> int | None:
    expected_phone = _normalise_phone(anchor.aligned_phone)
    for index in range(search_start, len(phone_intervals)):
        interval = phone_intervals[index]
        if _normalise_phone(interval.phone) != expected_phone:
            continue
        if interval.start_seconds <= anchor.requested_source_seconds <= interval.end_seconds:
            return index
    return None


def _primary_vowel_index(syllable: SyllableTarget) -> int:
    for index, phone in enumerate(syllable.phonemes):
        if is_arpabet_vowel(phone):
            return index
    raise MissingPlannedVowelError(
        f"syllable {syllable.word!r} is missing an ARPAbet vowel anchor"
    )


def _primary_vowel_phone(syllable: SyllableTarget) -> str:
    return syllable.phonemes[_primary_vowel_index(syllable)]


def _word_fallback_vowel_phone(syllable: SyllableTarget) -> str:
    try:
        return _primary_vowel_phone(syllable)
    except MissingPlannedVowelError:
        return UNKNOWN_PLANNED_VOWEL


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.dtype.kind in {"i", "u"}:
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        float_samples = array.astype(np.float32) / np.float32(scale)
    else:
        float_samples = array.astype(np.float32, copy=False)
    if float_samples.ndim == 1:
        return float_samples
    return np.mean(float_samples, axis=1, dtype=np.float32)


def _enforce_target_length(
    samples: np.ndarray,
    target_frames: int,
    *,
    output_name: str = "warp",
) -> np.ndarray:
    region = np.asarray(samples, dtype=np.float32).reshape(-1)
    length_difference = len(region) - target_frames
    if abs(length_difference) > 2:
        raise RuntimeError(
            f"{output_name} output length {len(region)} differs from target length "
            f"{target_frames} by {abs(length_difference)} frames"
        )
    if length_difference > 0:
        return region[:target_frames]
    if length_difference < 0:
        return np.pad(region, (0, -length_difference), mode="constant")
    return region


def _validate_time_map(time_map: Sequence[tuple[int, int]]) -> None:
    if len(time_map) < 2:
        raise ValueError("time map must contain at least two points")
    source_points = tuple(source for source, _ in time_map)
    target_points = tuple(target for _, target in time_map)
    if any(point < 0 for point in (*source_points, *target_points)):
        raise ValueError("time-map points must not be negative")
    if any(right <= left for left, right in zip(source_points, source_points[1:])):
        raise ValueError("time-map source points must be strictly monotonic")
    if any(right <= left for left, right in zip(target_points, target_points[1:])):
        raise ValueError("time-map target points must be strictly monotonic")
