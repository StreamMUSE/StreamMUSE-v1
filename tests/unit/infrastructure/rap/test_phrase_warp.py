from __future__ import annotations

import io
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.infrastructure.rap.phrase_warp import (
    RubberBandPhraseWarper,
    source_onset_anchors,
    warp_phrase,
)


RATE = 24_000
POLICIES = ("gentle_sparse_r3", "all_onsets_r3", "all_onsets_stress_r3")


def _request(ticks=(3, 7, 12), stresses=(1.0, 0.0, 0.8)):
    return TwoBarRenderRequest(
        song_id="leading-rest",
        chunk_index=0,
        start_bar=0,
        end_bar=1,
        tempo_bpm=120.0,
        text=" ".join("flow" for _ in ticks),
        syllables=tuple(
            SyllableTarget(
                word="flow",
                index_in_word=0,
                phonemes=("F", "L", "OW1"),
                lexical_stress=1,
                target_stress=stress,
                boundary_strength=0,
                absolute_tick=tick,
                tick_in_chunk=tick,
                target_seconds=tick * 0.125,
            )
            for tick, stress in zip(ticks, stresses, strict=True)
        ),
    )


class _RecordingStretcher:
    """Stand in only for the external R3 process; record the actual warp input."""

    def __init__(self):
        self.calls = []
        self.last_attempt_frame_deviations = (-4, 0)

    def __call__(self, samples, target_frames, sample_rate_hz, time_map):
        self.calls.append((samples.copy(), target_frames, sample_rate_hz, time_map))
        return np.full(target_frames, 0.25, dtype=np.float32)


def _warp(request, onsets, policy, *, source=None, stretcher=None):
    if source is None:
        source = np.full(48_000, 0.25, dtype=np.float32)
    anchors = source_onset_anchors(
        request, onsets, source_frame_count=len(source), sample_rate_hz=RATE
    )
    return warp_phrase(
        request,
        source,
        anchors,
        target_frame_count=48_000,
        sample_rate_hz=RATE,
        source_sha256="original-source",
        policy=policy,
        stretcher=stretcher or _RecordingStretcher(),
    )


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("first_tick, silence_frames", ((3, 9_000), (4, 12_000)))
@pytest.mark.parametrize("first_source, crop_frames", ((0.0, 0), (0.125, 3_000)))
def test_leading_rest_crops_preroll_and_prepends_exact_silence(
    policy, first_tick, silence_frames, first_source, crop_frames
):
    request = _request((first_tick, 7, 12))
    source = np.full(48_000, 0.25, dtype=np.float32)
    source[:crop_frames] = 0.9  # Audible pre-roll must never reach the output.
    stretcher = _RecordingStretcher()

    result = _warp(
        request, (first_source, 0.625, 1.25), policy, source=source, stretcher=stretcher
    )

    assert result.samples.shape == (48_000,)
    assert not np.any(result.samples[:silence_frames])
    assert result.samples[silence_frames] > 0.0
    _, pcm = wavfile.read(io.BytesIO(result.vocal_wav))
    assert not np.any(pcm[:silence_frames])
    assert pcm[silence_frames] > 0
    assert len(stretcher.calls) == 1
    active, target_frames, rate, time_map = stretcher.calls[0]
    np.testing.assert_array_equal(active, source[crop_frames:])
    assert target_frames == 48_000 - silence_frames
    assert rate == RATE
    assert time_map[0] == (0, 0)
    assert time_map[-1] == (47_999 - crop_frames, 47_999 - silence_frames)
    assert result.diagnostic_anchors[0].source_seconds == first_source
    assert result.diagnostic_anchors[0].target_sample == silence_frames
    assert result.plan.target_drift_seconds[0] == 0.0
    assert result.preparation.leading_silence_frames == silence_frames
    assert result.preparation.endpoint_policy["leading_silence_frames"] == silence_frames
    assert result.preparation.endpoint_policy["crop_start_source_sample"] == crop_frames
    assert result.r3_attempt_deviations == (-4, 0)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("first_tick, silence_frames", ((0, 0), (3, 9_000), (4, 12_000)))
@pytest.mark.parametrize("first_source", (0.0, 0.125))
def test_single_slot_capture_uses_linear_pitch_preserving_stretch(
    policy, first_tick, silence_frames, first_source
):
    stretcher = _RecordingStretcher()
    result = _warp(
        _request((first_tick,), (1.0,)), (first_source,), policy, stretcher=stretcher
    )

    assert result.samples.shape == (48_000,)
    assert not np.any(result.samples[:silence_frames])
    assert result.samples[silence_frames] > 0
    assert len(result.diagnostic_anchors) == 1
    assert result.diagnostic_anchors[0].source_seconds == first_source
    assert result.diagnostic_anchors[0].target_sample == silence_frames
    assert result.plan.selected_request_indices == (0,)
    assert result.plan.omitted_request_indices == ()
    assert result.plan.target_drift_seconds == (0.0,)
    time_map = stretcher.calls[0][3]
    assert time_map[0] == (0, 0)
    assert time_map[-1] == (47_999 - round(first_source * RATE), 47_999 - silence_frames)
    ratio = time_map[-1][1] / time_map[-1][0]
    assert all(abs(target - source * ratio) <= 0.5 for source, target in time_map)
    assert len(result.stretch_ratios) == 1


@pytest.mark.parametrize("policy", ("all_onsets_r3", "all_onsets_stress_r3"))
def test_strict_policies_map_every_anchor_in_full_source_and_bar_coordinates(policy):
    stretcher = _RecordingStretcher()
    result = _warp(_request(), (0.125, 0.625, 1.25), policy, stretcher=stretcher)

    assert result.plan.selected_request_indices == (0, 1, 2)
    assert result.plan.omitted_request_indices == ()
    assert not result.plan.timing_regularization_applied
    assert result.plan.target_drift_seconds == (0.0, 0.0, 0.0)
    assert stretcher.calls[0][3] == ((0, 0), (12_000, 12_000), (27_000, 27_000), (44_999, 38_999))
    assert tuple(a.source_seconds for a in result.diagnostic_anchors) == (0.125, 0.625, 1.25)
    assert tuple(a.target_seconds for a in result.diagnostic_anchors) == (0.375, 0.875, 1.5)
    assert tuple(a.requested_source_seconds for a in result.diagnostic_anchors) == (0.125, 0.625, 1.25)
    assert tuple(a.requested_target_seconds for a in result.diagnostic_anchors) == (0.375, 0.875, 1.5)


def test_gentle_regularizes_interior_targets_but_never_moves_first_onset():
    result = _warp(_request(), (0.125, 0.3, 0.5), "gentle_sparse_r3")

    assert result.plan.resolved_policy.value == "gentle_sparse_r3"
    assert result.plan.timing_regularization_applied
    assert result.diagnostic_anchors[0].target_seconds == 0.375
    assert result.plan.target_drift_seconds[0] == 0.0
    assert any(abs(drift) > 0.01 for drift in result.plan.target_drift_seconds[1:])
    assert not np.any(result.samples[:9_000])
    assert tuple(a.source_seconds for a in result.diagnostic_anchors) == (0.125, 0.3, 0.5)


def test_stress_policy_adds_smooth_gain_without_changing_timing_or_duration():
    strict_stretcher, stress_stretcher = _RecordingStretcher(), _RecordingStretcher()
    strict = _warp(_request(), (0.125, 0.625, 1.25), "all_onsets_r3", stretcher=strict_stretcher)
    stressed = _warp(_request(), (0.125, 0.625, 1.25), "all_onsets_stress_r3", stretcher=stress_stretcher)

    assert strict_stretcher.calls[0][1:] == stress_stretcher.calls[0][1:]
    assert strict.samples.shape == stressed.samples.shape
    assert strict.diagnostic_anchors == stressed.diagnostic_anchors
    assert strict.stretch_ratios == stressed.stretch_ratios
    assert np.max(stressed.samples[9_000:21_000]) > np.max(strict.samples[9_000:21_000])
    np.testing.assert_array_equal(stressed.samples[21_000:36_000], strict.samples[21_000:36_000])
    assert np.max(stressed.samples[36_000:]) > np.max(strict.samples[36_000:])
    # No discontinuity is introduced inside the active constant-amplitude signal.
    assert np.max(np.abs(np.diff(stressed.samples[9_000:]))) < 0.001


def test_zero_target_stress_leaves_strict_audio_unchanged():
    request = _request(stresses=(0.0, 0.0, 0.0))
    strict = _warp(request, (0.125, 0.625, 1.25), "all_onsets_r3")
    stressed = _warp(request, (0.125, 0.625, 1.25), "all_onsets_stress_r3")
    assert stressed.vocal_wav == strict.vocal_wav


def test_alignment_policy_snapshot_keeps_original_warper_unchanged():
    original = RubberBandPhraseWarper(rubberband_version="test", stretcher=_RecordingStretcher())
    strict = original.with_policy("all_onsets_r3")
    accented = strict.with_policy("all_onsets_stress_r3")
    assert original.policy == "gentle_sparse_r3"
    assert strict.policy == "all_onsets_r3"
    assert accented.policy == "all_onsets_stress_r3"
    assert strict.rubberband_version == "test"
    with pytest.raises(ValueError):
        original.with_policy("invalid")


def test_alignment_snapshots_isolate_concurrent_stretcher_diagnostics():
    entered, release = threading.Event(), threading.Event()

    class BlockingStretcher(_RecordingStretcher):
        def __call__(self, samples, target_frames, sample_rate_hz, time_map):
            first = not entered.is_set()
            self.last_attempt_frame_deviations = (-7, 0) if first else (-9, 0)
            if first:
                entered.set()
                assert release.wait(3)
            return np.full(target_frames, 0.25, dtype=np.float32)

    source = io.BytesIO()
    wavfile.write(source, RATE, np.full(48_000, 8192, dtype=np.int16))
    original = RubberBandPhraseWarper(policy="all_onsets_r3", rubberband_version="test", stretcher=BlockingStretcher())
    accented = original.with_policy("all_onsets_stress_r3")

    def render(warper):
        return warper.warp(_request(), source.getvalue(), (0.125, 0.625, 1.25), target_frame_count=48_000)

    with ThreadPoolExecutor(max_workers=2) as workers:
        old = workers.submit(render, original)
        try:
            assert entered.wait(3)
            new = workers.submit(render, accented).result(timeout=3)
        finally:
            release.set()
        old = old.result(timeout=3)
    assert old.policy == "all_onsets_r3" and new.policy == "all_onsets_stress_r3"
    assert old.r3_attempt_deviations == (-7, 0)
    assert new.r3_attempt_deviations == (-9, 0)


def test_local_result_reports_resolved_policy_on_gentle_fallback():
    wav = io.BytesIO()
    wavfile.write(wav, RATE, np.full(12_000, 8_192, dtype=np.int16))
    warper = RubberBandPhraseWarper(rubberband_version="test", stretcher=_RecordingStretcher())

    result = warper.warp(_request(), wav.getvalue(), (0.0, 0.1, 0.2), target_frame_count=48_000)

    assert warper.policy == "gentle_sparse_r3"
    assert result.policy == "all_onsets_r3"
    assert result.source_anchors == (0.0, 0.1, 0.2)
    assert result.target_anchors == (0.375, 0.875, 1.5)
    assert result.r3_attempt_deviations == (-4, 0)


@pytest.mark.parametrize("policy", POLICIES)
def test_zero_onset_and_zero_target_keep_existing_time_map(policy):
    stretcher = _RecordingStretcher()
    result = _warp(_request((0, 4, 8)), (0.0, 0.5, 1.0), policy, stretcher=stretcher)

    assert stretcher.calls[0][3] == ((0, 0), (12_000, 12_000), (24_000, 24_000), (47_999, 47_999))
    assert result.samples[0] == 0.25
    assert tuple(a.source_seconds for a in result.diagnostic_anchors) == (0.0, 0.5, 1.0)
    assert tuple(a.target_seconds for a in result.diagnostic_anchors) == (0.0, 0.5, 1.0)


def test_rejects_target_boundary_without_active_audio_remainder():
    request = _request((3,), (1.0,))
    request = replace(request, syllables=(replace(request.syllables[0], target_seconds=2.0),))
    with pytest.raises(PhraseRenderFailed, match="target.*(inside|remainder)"):
        _warp(request, (0.125,), "all_onsets_r3")


@pytest.mark.skipif(shutil.which("rubberband") is None, reason="rubberband CLI not installed")
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("first_source", (0.0, 0.125))
@pytest.mark.parametrize("ticks, silence_frames", (((3,), 9_000), ((4, 8, 12), 12_000)))
def test_real_r3_preserves_pitch_and_leading_rest(policy, first_source, ticks, silence_frames):
    source = (0.2 * np.cos(2 * np.pi * 220 * np.arange(24_000) / RATE)).astype(np.float32)
    wav = io.BytesIO()
    wavfile.write(wav, RATE, source)
    result = RubberBandPhraseWarper(policy=policy, rubberband_version="test").warp(
        _request(ticks, (1.0,) * len(ticks)),
        wav.getvalue(),
        (first_source,) if len(ticks) == 1 else (first_source, 0.4, 0.7),
        target_frame_count=48_000,
    )

    rate, pcm = wavfile.read(io.BytesIO(result.vocal_wav))
    assert rate == RATE
    assert pcm.shape == (48_000,)
    assert not np.any(pcm[:silence_frames])
    assert np.any(pcm[silence_frames:silence_frames + 240])
    active = pcm[18_000:42_000].astype(np.float64)
    spectrum = np.abs(np.fft.rfft(active * np.hanning(len(active))))
    peak_hz = np.fft.rfftfreq(len(active), 1.0 / RATE)[np.argmax(spectrum)]
    assert peak_hz == pytest.approx(220, abs=2)
