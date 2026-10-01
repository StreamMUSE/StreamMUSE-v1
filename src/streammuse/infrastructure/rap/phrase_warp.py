"""Shared phrase time warp: syllable-onset anchors to an exact-length R3 phrase.

The H200 renderer and the Mac client run this same code, so a source phrase and
its onset anchors produce the same warped vocal on either side (given the same
Rubber Band build). Nothing here needs a GPU: it is numpy, scipy, and the
``rubberband`` CLI behind ``RubberBandTimeMapStretcher``.
"""

from __future__ import annotations

import hashlib
import io
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed
from streammuse.experiments.rap_audio_protocols.contracts import TwoBarRenderRequest
from streammuse.experiments.rap_audio_protocols.warp import (
    VowelAnchor,
    regularize_gentle_sparse_anchors,
)


GENTLE_STRETCH_MIN = 0.75
GENTLE_STRETCH_MAX = 1.35


class MossWarpPolicy(str, Enum):
    GENTLE_SPARSE_R3 = "gentle_sparse_r3"
    ALL_ONSETS_R3 = "all_onsets_r3"


class FullChunkStretcher(Protocol):
    def __call__(
        self,
        samples: np.ndarray,
        target_frames: int,
        sample_rate_hz: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class WarpPreparation:
    samples: np.ndarray
    interior_anchors: tuple[VowelAnchor, ...]
    interior_anchor_indices: tuple[int, ...]
    boundary_anchor_indices: tuple[int, ...]
    diagnostic_anchors: tuple[VowelAnchor, ...]
    source_sha256: str
    endpoint_policy: Mapping[str, object]


@dataclass(frozen=True)
class WarpPlan:
    anchors: tuple[VowelAnchor, ...]
    requested_policy: MossWarpPolicy
    resolved_policy: MossWarpPolicy
    selected_request_indices: tuple[int, ...]
    omitted_request_indices: tuple[int, ...]
    timing_regularization_applied: bool
    ratio_bounds: tuple[float, float] | None
    target_drift_seconds: tuple[float, ...]
    warnings: tuple[str, ...]


def prepare_warp_input(
    source_samples: np.ndarray,
    anchors: Sequence[VowelAnchor],
    *,
    sample_rate_hz: int,
    source_sha256: str,
) -> WarpPreparation:
    original = np.asarray(source_samples, dtype=np.float32).reshape(-1)
    original_warp_sha256 = float32le_sha256(original)
    complete_anchors = tuple(anchors)
    if not complete_anchors:
        raise PhraseRenderFailed("warp preparation requires mapped syllable onsets")
    first = complete_anchors[0]
    if first.target_sample != 0:
        return WarpPreparation(
            samples=original,
            interior_anchors=complete_anchors,
            interior_anchor_indices=tuple(range(len(complete_anchors))),
            boundary_anchor_indices=(),
            diagnostic_anchors=complete_anchors,
            source_sha256=original_warp_sha256,
            endpoint_policy={
                "name": "implicit_audio_boundaries",
                "applied": False,
                "target_zero_as_boundary": False,
                "crop_start_source_sample": 0,
                "crop_start_source_seconds": 0.0,
                "original_frame_count": len(original),
                "cropped_frame_count": len(original),
                "original_source_wav_sha256": source_sha256,
                "warp_input_encoding": "float32le",
                "warp_input_float32le_sha256": original_warp_sha256,
            },
        )
    if first.target_seconds != 0.0 or first.requested_target_seconds != 0.0:
        raise PhraseRenderFailed(
            "a target that rounds to sample zero must be exactly the zero boundary"
        )
    crop_start = first.source_sample
    if crop_start < 0 or crop_start >= len(original) - 1:
        raise PhraseRenderFailed(
            "tick-zero acoustic onset must lie within the usable source audio"
        )
    cropped = np.ascontiguousarray(original[crop_start:], dtype=np.float32)
    normalized = tuple(
        replace(
            anchor,
            requested_source_seconds=(anchor.requested_source_sample - crop_start)
            / sample_rate_hz,
            source_seconds=(anchor.source_sample - crop_start) / sample_rate_hz,
            requested_source_sample=anchor.requested_source_sample - crop_start,
            source_sample=anchor.source_sample - crop_start,
        )
        for anchor in complete_anchors
    )
    if (
        normalized[0].source_sample != 0
        or normalized[0].target_sample != 0
        or len(normalized) < 2
    ):
        raise PhraseRenderFailed(
            "tick-zero endpoint policy requires one boundary and an interior anchor"
        )
    warp_input_sha256 = float32le_sha256(cropped)
    return WarpPreparation(
        samples=cropped,
        interior_anchors=normalized[1:],
        interior_anchor_indices=tuple(range(1, len(normalized))),
        boundary_anchor_indices=(0,),
        diagnostic_anchors=normalized,
        source_sha256=warp_input_sha256,
        endpoint_policy={
            "name": "crop_first_acoustic_onset_to_target_boundary",
            "applied": True,
            "target_zero_as_boundary": True,
            "crop_start_source_sample": crop_start,
            "crop_start_source_seconds": crop_start / sample_rate_hz,
            "original_frame_count": len(original),
            "cropped_frame_count": len(cropped),
            "original_source_wav_sha256": source_sha256,
            "warp_input_encoding": "float32le",
            "warp_input_float32le_sha256": warp_input_sha256,
        },
    )


def resolve_warp_plan(
    request: TwoBarRenderRequest,
    preparation: WarpPreparation,
    *,
    target_frame_count: int,
    sample_rate_hz: int,
    requested_policy: MossWarpPolicy,
) -> WarpPlan:
    if len(request.syllables) != len(preparation.diagnostic_anchors):
        raise PhraseRenderFailed(
            "warp preparation no longer matches the planned syllable count"
        )
    all_indices = tuple(range(len(request.syllables)))
    if requested_policy is MossWarpPolicy.ALL_ONSETS_R3:
        return WarpPlan(
            anchors=preparation.interior_anchors,
            requested_policy=requested_policy,
            resolved_policy=requested_policy,
            selected_request_indices=all_indices,
            omitted_request_indices=(),
            timing_regularization_applied=False,
            ratio_bounds=None,
            target_drift_seconds=tuple(0.0 for _ in all_indices),
            warnings=(),
        )

    interior_syllables = tuple(
        request.syllables[index]
        for index in preparation.interior_anchor_indices
    )
    try:
        selection = regularize_gentle_sparse_anchors(
            preparation.interior_anchors,
            interior_syllables,
            sample_rate_hz=sample_rate_hz,
            source_frame_count=len(preparation.samples),
            target_frame_count=target_frame_count,
            min_stretch_ratio=GENTLE_STRETCH_MIN,
            max_stretch_ratio=GENTLE_STRETCH_MAX,
        )
    except ValueError as exc:
        warning = (
            "gentle sparse R3 unavailable; using all-onset R3: "
            f"{exc}"
        )
        return WarpPlan(
            anchors=preparation.interior_anchors,
            requested_policy=requested_policy,
            resolved_policy=MossWarpPolicy.ALL_ONSETS_R3,
            selected_request_indices=all_indices,
            omitted_request_indices=(),
            timing_regularization_applied=False,
            ratio_bounds=None,
            target_drift_seconds=tuple(0.0 for _ in all_indices),
            warnings=(warning,),
        )

    selected_interior_indices = tuple(
        preparation.interior_anchor_indices[index]
        for index in selection.selected_indices
    )
    selected_request_indices = tuple(
        (*preparation.boundary_anchor_indices, *selected_interior_indices)
    )
    selected_set = frozenset(selected_request_indices)
    omitted_request_indices = tuple(
        index for index in all_indices if index not in selected_set
    )
    target_drift = [0.0] * len(all_indices)
    for request_index, anchor in zip(
        preparation.interior_anchor_indices,
        selection.regularized_anchors,
    ):
        target_drift[request_index] = (
            anchor.target_seconds - anchor.requested_target_seconds
        )
    return WarpPlan(
        anchors=selection.anchors,
        requested_policy=requested_policy,
        resolved_policy=requested_policy,
        selected_request_indices=selected_request_indices,
        omitted_request_indices=omitted_request_indices,
        timing_regularization_applied=True,
        ratio_bounds=(GENTLE_STRETCH_MIN, GENTLE_STRETCH_MAX),
        target_drift_seconds=tuple(target_drift),
        warnings=(),
    )


def effective_diagnostic_anchors(
    preparation: WarpPreparation,
    plan: WarpPlan,
    effective_anchors: Sequence[VowelAnchor],
    *,
    source_frame_count: int,
    target_frame_count: int,
    sample_rate_hz: int,
) -> tuple[VowelAnchor, ...]:
    """Return one public anchor per planned syllable.

    Boundary and stretched anchors are reported as warped. A syllable omitted
    from a sparse warp keeps its measured MMS source onset and reports where the
    applied time map actually placed it, so the public contract stays
    one-anchor-per-syllable while the sparse selection stays private.
    """
    boundary_indices = frozenset(preparation.boundary_anchor_indices)
    interior_indices = tuple(
        index
        for index in plan.selected_request_indices
        if index not in boundary_indices
    )
    effective_by_index = dict(zip(interior_indices, effective_anchors, strict=True))
    # Same sample-domain time map continuous_pitch_preserving_warp hands to R3.
    source_points = np.array(
        (0, *(anchor.source_sample for anchor in effective_anchors), source_frame_count - 1),
        dtype=np.float64,
    )
    target_points = np.array(
        (0, *(anchor.target_sample for anchor in effective_anchors), target_frame_count - 1),
        dtype=np.float64,
    )
    anchors: list[VowelAnchor] = []
    for index, anchor in enumerate(preparation.diagnostic_anchors):
        if index in boundary_indices:
            anchors.append(anchor)
        elif index in effective_by_index:
            anchors.append(effective_by_index[index])
        else:
            target_sample = float(
                np.interp(anchor.source_sample, source_points, target_points)
            )
            anchors.append(
                replace(
                    anchor,
                    target_seconds=target_sample / sample_rate_hz,
                    target_sample=int(round(target_sample)),
                )
            )
    return tuple(anchors)


def float32le_sha256(samples: np.ndarray) -> str:
    payload = np.ascontiguousarray(samples, dtype="<f4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def to_float32(samples: np.ndarray) -> np.ndarray:
    if samples.dtype.kind in {"i", "u"}:
        limits = np.iinfo(samples.dtype)
        scale = max(abs(limits.min), limits.max)
        return samples.astype(np.float32) / np.float32(scale)
    return samples.astype(np.float32, copy=False)


def validate_output_samples(
    samples: np.ndarray,
    *,
    expected_frame_count: int,
) -> np.ndarray:
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    if mono.size != expected_frame_count:
        raise PhraseRenderFailed(
            f"warp produced {mono.size} frames, expected {expected_frame_count}"
        )
    if not np.isfinite(mono).all():
        raise PhraseRenderFailed("warp output contains non-finite samples")
    if float(np.max(np.abs(mono.astype(np.float64)))) == 0.0:
        raise PhraseRenderFailed("warp output must not be silent")
    return mono


def encode_pcm16_wav(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
) -> tuple[bytes, Mapping[str, float]]:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = np.rint(clipped * np.float32(32767.0)).astype("<i2")
    if not np.any(pcm16):
        raise PhraseRenderFailed("encoded PCM16 output must not be silent")
    normalized = pcm16.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(normalized)))
    rms = float(np.sqrt(np.mean(np.square(normalized), dtype=np.float64)))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate_hz)
        output.writeframes(pcm16.tobytes())
    return buffer.getvalue(), {"peak": peak, "rms": rms}


def validate_pcm16_wav(
    path: Path,
    *,
    expected_frame_count: int,
    expected_sample_rate_hz: int,
) -> None:
    try:
        with wave.open(str(path), "rb") as rendered:
            properties = (
                rendered.getframerate(),
                rendered.getnchannels(),
                rendered.getsampwidth(),
                rendered.getnframes(),
                rendered.getcomptype(),
            )
    except (OSError, EOFError, wave.Error) as exc:
        raise PhraseRenderFailed("encoded vocal WAV is unreadable") from exc
    expected = (expected_sample_rate_hz, 1, 2, expected_frame_count, "NONE")
    if properties != expected:
        raise PhraseRenderFailed(
            f"encoded vocal WAV format mismatch: expected {expected}, got {properties}"
        )
