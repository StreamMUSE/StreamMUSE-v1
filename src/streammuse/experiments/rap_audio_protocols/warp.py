"""Forced-alignment parsing and piecewise warp helpers for Protocol 4."""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy.io import wavfile

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
_TEXTGRID_INTERVAL_RE = re.compile(
    r"intervals \[\d+\]:\s*xmin = ([0-9.]+)\s*xmax = ([0-9.]+)\s*text = \"([^\"]*)\"",
    re.MULTILINE,
)
_PHONE_SUFFIX_RE = re.compile(r"\d+$")


StretchRegionFn = Callable[[np.ndarray, int, int], np.ndarray]


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
class VowelAnchor:
    word: str
    index_in_word: int
    planned_phone: str
    aligned_phone: str
    source_seconds: float
    target_seconds: float
    source_sample: int
    target_sample: int


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
                        "--pitch",
                        "1.0",
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
        return _coerce_region_length(_to_mono_float32(stretched), target_frames)


def is_arpabet_vowel(phone: str) -> bool:
    return _normalise_phone(phone) in _ARPABET_VOWELS


def parse_textgrid_phone_intervals(text: str) -> tuple[PhoneInterval, ...]:
    intervals = []
    for start_seconds, end_seconds, phone in _TEXTGRID_INTERVAL_RE.findall(text):
        label = phone.strip()
        if not label:
            continue
        intervals.append(
            PhoneInterval(
                start_seconds=float(start_seconds),
                end_seconds=float(end_seconds),
                phone=label,
            )
        )
    if not intervals:
        raise ValueError("no phone intervals found in TextGrid")
    return tuple(intervals)


def load_textgrid_phone_intervals(path: Path | str) -> tuple[PhoneInterval, ...]:
    return parse_textgrid_phone_intervals(Path(path).read_text(encoding="utf-8"))


def match_vowel_anchors(
    phone_intervals: Sequence[PhoneInterval],
    syllables: Sequence[SyllableTarget],
    *,
    sample_rate_hz: int,
) -> tuple[VowelAnchor, ...]:
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    source_vowels = tuple(interval for interval in phone_intervals if is_arpabet_vowel(interval.phone))
    planned_vowels = tuple((_primary_vowel_phone(syllable), syllable) for syllable in syllables)
    if len(source_vowels) != len(planned_vowels):
        raise ValueError(
            f"aligned vowel count mismatch: expected {len(planned_vowels)}, got {len(source_vowels)}"
        )

    anchors = []
    for index, (source, (planned_phone, syllable)) in enumerate(zip(source_vowels, planned_vowels)):
        if _normalise_phone(planned_phone) != _normalise_phone(source.phone):
            raise ValueError(
                "strict vowel-anchor matching failed "
                f"at syllable {index}: expected {planned_phone}, got {source.phone}"
            )
        source_seconds = source.start_seconds + min(0.030, 0.25 * source.duration_seconds)
        anchors.append(
            VowelAnchor(
                word=syllable.word,
                index_in_word=syllable.index_in_word,
                planned_phone=planned_phone,
                aligned_phone=source.phone,
                source_seconds=source_seconds,
                target_seconds=syllable.target_seconds,
                source_sample=round(source_seconds * sample_rate_hz),
                target_sample=round(syllable.target_seconds * sample_rate_hz),
            )
        )
    return tuple(anchors)


def piecewise_pitch_preserving_warp(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    anchors: Sequence[VowelAnchor],
    target_frame_count: int,
    stretch_region: StretchRegionFn | None = None,
    crossfade_seconds: float = 0.005,
    min_region_seconds: float = 0.010,
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
    stretcher = stretch_region or RubberBandStretcher()

    _validate_anchor_monotonicity(
        tuple(anchor.source_sample for anchor in anchors),
        tuple(anchor.target_sample for anchor in anchors),
        len(mono),
        target_frame_count,
    )
    source_points, target_points = _build_control_points(
        tuple(anchor.source_sample for anchor in anchors),
        tuple(anchor.target_sample for anchor in anchors),
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
        stretched = _coerce_region_length(stretcher(source_region, target_length, sample_rate_hz), target_length)
        if region_index > 0:
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
        anchor_map=tuple(anchors),
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
        overlap = min(crossfade_samples, len(left_region), len(right_region), boundary + 1)
        if overlap <= 1:
            continue
        left_tail = left_region[-overlap:]
        right_head = right_region[:overlap]
        theta = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
        blend = (left_tail * np.cos(theta)) + (right_head * np.sin(theta))
        output[boundary - overlap + 1 : boundary + 1] = blend
    return output.astype(np.float32, copy=False)


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


def _normalise_phone(phone: str) -> str:
    return _PHONE_SUFFIX_RE.sub("", phone.strip().upper())


def _primary_vowel_phone(syllable: SyllableTarget) -> str:
    for phone in syllable.phonemes:
        if is_arpabet_vowel(phone):
            return phone
    raise ValueError(f"syllable {syllable.word!r} is missing an ARPAbet vowel anchor")


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


def _coerce_region_length(samples: np.ndarray, target_frames: int) -> np.ndarray:
    region = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(region) == target_frames:
        return region
    if len(region) == 0:
        return np.zeros(target_frames, dtype=np.float32)
    if len(region) == 1:
        return np.full(target_frames, region[0], dtype=np.float32)
    source_positions = np.linspace(0.0, 1.0, len(region), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    return np.interp(target_positions, source_positions, region).astype(np.float32, copy=False)
