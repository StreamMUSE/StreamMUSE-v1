from dataclasses import replace
from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

from streammuse.application.rap.inference.delivery import SpanBuffer
from streammuse.domain.rap.inference import AudioSpan, InferenceRequest


def span(**kwargs):
    values = dict(request_id="request", output_id="output", epoch=1, input_version=1,
                  start_tick=0, duration_ticks=16, sample_rate=100, channels=1,
                  pcm=np.ones((400, 1), dtype=np.float32).tobytes(),
                  safe_split_ticks=tuple(range(1, 16)))
    return AudioSpan(**{**values, **kwargs})


def test_commit_quantum_stays_four_ticks_after_first_split():
    buffer = SpanBuffer(sample_rate=100, channels=1, frame_at=lambda tick: tick * 25)
    buffer.accept(span(), 1)
    assert buffer.select(0, 1, 4)[1] == 100
    assert buffer.select(100, 1, 4)[1] == 200
    assert buffer.select(200, 1, 4)[1] == 300


def test_older_revision_does_not_overwrite_newer_ready_audio():
    buffer = SpanBuffer(sample_rate=100, channels=1, frame_at=lambda tick: tick * 25)
    buffer.accept(span(revision=2), 1)
    assert not buffer.accept(span(output_id="old", revision=1), 2)
    assert buffer.select(0, 1, 1)[0].span.revision == 2


def test_late_audio_cannot_join_mid_phoneme_without_safe_split():
    buffer = SpanBuffer(sample_rate=100, channels=1, frame_at=lambda tick: tick * 25)
    buffer.accept(span(safe_split_ticks=(8,)), 1)
    assert buffer.select(25, 1, 1)[0] is None
    assert buffer.select(200, 1, 1)[0] is not None


def test_frozen_audio_cannot_be_replaced_until_safe_commit_boundary():
    buffer = SpanBuffer(sample_rate=100, channels=1, frame_at=lambda tick: tick * 25)
    buffer.accept(span(), 1)
    buffer.select(0, 1, 4)
    buffer.accept(span(output_id="replacement", input_version=2), 2)
    assert buffer.select(25, 2, 4)[0].span.input_version == 1
    assert buffer.select(100, 2, 4)[0].span.input_version == 2


def make_span(*, start=0, duration=4, frame_at=None, value=0.25, **changes):
    frame_at = frame_at or (lambda tick: tick * 25)
    channels = changes.get("channels", 1)
    frames = frame_at(start + duration) - frame_at(start)
    values = dict(request_id="request", output_id=f"output:{start}", epoch=1,
                  input_version=1, start_tick=start, duration_ticks=duration,
                  sample_rate=100, channels=channels,
                  pcm=np.full((frames, channels), value, dtype=np.float32).tobytes(),
                  safe_split_ticks=())
    return AudioSpan(**{**values, **changes})


@pytest.fixture
def buffer():
    return SpanBuffer(sample_rate=100, channels=1, frame_at=lambda tick: tick * 25)


@pytest.mark.parametrize("pcm", [b"", b"\0", b"\0" * 3, b"\0" * 5,
                                 bytearray(4), memoryview(bytes(4)), "not PCM"])
def test_domain_rejects_empty_partial_or_nonbyte_pcm(pcm):
    with pytest.raises(ValueError):
        make_span(pcm=pcm)


def test_stereo_requires_complete_interleaved_frames():
    with pytest.raises(ValueError):
        make_span(channels=2, pcm=bytes(12))
    assert make_span(channels=2, pcm=bytes(16)).frame_count == 2


def test_domain_rejects_pcm_above_payload_bound():
    with pytest.raises(ValueError):
        make_span(pcm=bytes(64 * 1024 * 1024 + 4))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_pcm_never_enters_playable_buffer(buffer, value):
    with pytest.raises(ValueError, match="finite"):
        buffer.accept(make_span(value=value), 1.0)
    assert buffer.entries == []
    assert buffer.prefix_ticks(1) == 0


@pytest.mark.parametrize("frames", [1, 99, 101])
def test_frame_count_must_match_entire_musical_duration(buffer, frames):
    invalid = make_span(pcm=np.zeros(frames, dtype=np.float32).tobytes())
    with pytest.raises(ValueError, match="frame count"):
        buffer.accept(invalid, 1)
    assert buffer.entries == []
    assert buffer.identities == set()


@pytest.mark.parametrize("changes", [{"sample_rate": 200}, {"channels": 2}])
def test_incompatible_format_is_rejected_before_scheduling(buffer, changes):
    with pytest.raises(ValueError, match="format"):
        buffer.accept(make_span(**changes), 1)
    assert buffer.prefix_ticks(1) == 0


@pytest.mark.parametrize("field", ["duration_ticks", "sample_rate", "channels"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_audio_span_requires_positive_integer_geometry(field, value):
    with pytest.raises(ValueError):
        replace(make_span(), **{field: value})


@pytest.mark.parametrize("splits", [(0,), (4,), (-1,), (2, 1), (1, 1), (1.5,), (True,)])
def test_safe_splits_are_strictly_ordered_interior_integer_offsets(splits):
    with pytest.raises(ValueError):
        make_span(safe_split_ticks=splits)


@pytest.mark.parametrize("onset", [-1, 4, 0.5, True])
def test_first_vocal_onset_is_an_interior_integer_tick(onset):
    with pytest.raises(ValueError):
        make_span(first_vocal_tick=onset)


@pytest.mark.parametrize("field,value", [
    ("duration_ticks", 0), ("ticks_per_beat", True), ("beats_per_bar", 0),
    ("epoch", -1), ("input_version", 1.5), ("start_tick", -1),
    ("bpm", 0), ("bpm", float("nan")), ("bpm", float("inf")),
    ("flow_ticks", (-1,)), ("flow_ticks", (21,)), ("flow_ticks", (True,)),
])
def test_request_rejects_invalid_musical_geometry(field, value):
    values = dict(session_id="session", epoch=1, request_id="request", input_version=1,
                  start_tick=0, duration_ticks=21, bpm=137, ticks_per_beat=7,
                  beats_per_bar=3, topic="test", template_id="test")
    with pytest.raises(ValueError):
        InferenceRequest(**{**values, field: value})


def test_nonstandard_resolution_uses_absolute_rounded_frame_boundaries():
    frame_at = lambda tick: round(Fraction(tick * 11025 * 60, 137 * 7))
    buffer = SpanBuffer(sample_rate=11025, channels=2, frame_at=frame_at)
    first = make_span(start=1, duration=2, frame_at=frame_at, sample_rate=11025, channels=2)
    assert first.frame_count != frame_at(2)
    assert buffer.accept(first, 1)
    entry, end = buffer.select(frame_at(1), 1, 1)
    assert entry.span == first
    assert end == frame_at(3)
    assert entry.audio.shape == (frame_at(3) - frame_at(1), 2)
    bad = replace(first, output_id="bad-rounding", pcm=bytes(frame_at(2) * 2 * 4))
    with pytest.raises(ValueError, match="frame count"):
        buffer.accept(bad, 2)


def test_nonzero_start_safe_split_offsets_are_not_absolute_ticks(buffer):
    candidate = make_span(start=7, duration=5, safe_split_ticks=(2,))
    buffer.accept(candidate, 1)
    assert buffer.select(7 * 25, 1, 1)[1] == 9 * 25
    assert buffer.select(9 * 25, 1, 1)[1] == 12 * 25


def test_quantum_never_invents_a_split_inside_a_phoneme(buffer):
    candidate = make_span(duration=10, safe_split_ticks=(3, 7))
    buffer.accept(candidate, 1)
    assert buffer.select(0, 1, 4)[1] == 7 * 25
    assert buffer.select(7 * 25, 1, 4)[1] == 10 * 25


def test_unsplittable_span_remains_committed_to_its_end(buffer):
    candidate = make_span(duration=8)
    buffer.accept(candidate, 1)
    assert buffer.select(0, 1, 1)[1] == 200
    buffer.accept(replace(candidate, output_id="replacement", input_version=2), 2)
    entry, end = buffer.select(25, 2, 1)
    assert entry.span.input_version == 1
    assert end == 200
    assert buffer.select(200, 2, 1)[0] is None


def test_duplicate_identity_is_idempotent_and_keeps_original_audio(buffer):
    first = make_span()
    assert buffer.accept(first, 1)
    assert not buffer.accept(first, 2)
    assert not buffer.accept(replace(first, pcm=np.ones(100, dtype=np.float32).tobytes()), 3)
    assert len(buffer.entries) == 1
    assert buffer.entries[0].ready_ms == 1
    np.testing.assert_array_equal(buffer.select(0, 1, 1)[0].audio, 0.25)


def test_out_of_order_uneven_spans_only_fill_an_actual_contiguous_prefix(buffer):
    for start, duration, prefix in [(5, 2, 0), (2, 3, 0), (0, 1, 1), (1, 1, 7)]:
        assert buffer.accept(make_span(start=start, duration=duration), 1000 - start)
        assert buffer.prefix_ticks(1) == prefix
    assert buffer.prefix_ticks(2) == 0


@pytest.mark.parametrize("order", list(permutations((0, 1, 2))))
def test_all_delivery_orders_preserve_musical_sample_placement(buffer, order):
    chunks = [make_span(start=0, duration=2, value=0.25),
              make_span(start=2, duration=3, value=0.5),
              make_span(start=5, duration=1, value=0.75)]
    for index in order:
        buffer.accept(chunks[index], index * 17)
    assert buffer.prefix_ticks(1) == 6
    for candidate, value in zip(chunks, (0.25, 0.5, 0.75)):
        entry, end = buffer.select(candidate.start_tick * 25, 1, 1)
        assert entry.span == candidate
        assert end == candidate.end_tick * 25
        np.testing.assert_array_equal(entry.audio, value)


def test_explicit_leading_silence_is_playable_but_empty_pcm_is_not(buffer):
    silence = make_span(duration=2, value=0.0, declared_silence=True)
    assert buffer.accept(silence, 1)
    assert buffer.prefix_ticks(1) == 2
    entry, end = buffer.select(0, 1, 1)
    assert entry.span.declared_silence
    assert end == 50
    assert not entry.audio.any()
    with pytest.raises(ValueError):
        replace(silence, pcm=b"")


def test_missing_prefix_cannot_freeze_all_the_way_to_an_out_of_order_span(buffer):
    buffer.accept(make_span(start=8), 1)
    entry, end = buffer.select(0, 1, 1)
    assert entry is None
    assert 0 < end <= 25
    buffer.accept(make_span(start=1), 2)
    entry, _ = buffer.select(25, 1, 1)
    assert entry is not None
    assert entry.span.start_tick == 1


@pytest.mark.parametrize("tick", [1, 48, 4000])
def test_missing_coverage_quantum_uses_absolute_fractional_tick_geometry(tick):
    frame_at = lambda position: round(Fraction(position * 25, 2))
    buffer = SpanBuffer(sample_rate=100, channels=1, frame_at=frame_at)
    entry, end = buffer.select(frame_at(tick), 1, 1)
    assert entry is None
    assert end == frame_at(tick + 1)


def test_newer_revision_replaces_only_mutable_coverage(buffer):
    first = make_span(duration=8, safe_split_ticks=(4,))
    assert buffer.accept(first, 1)
    assert buffer.select(0, 1, 1)[0].span.revision == 0
    revision = replace(first, revision=1, pcm=np.ones(200, dtype=np.float32).tobytes())
    assert buffer.accept(revision, 2)
    assert buffer.select(25, 1, 1)[0].span.revision == 0
    selected, end = buffer.select(100, 1, 1)
    assert selected.span.revision == 1
    assert end == 200
    np.testing.assert_array_equal(selected.audio, 1.0)


@pytest.mark.parametrize("start,duration", [(1, 4), (0, 3), (4, 4)])
def test_revision_cannot_change_an_output_identity_target_coverage(buffer, start, duration):
    first = make_span()
    buffer.accept(first, 1)
    moved = make_span(start=start, duration=duration, output_id=first.output_id, revision=1)
    with pytest.raises(ValueError, match="coverage"):
        buffer.accept(moved, 2)
    assert len(buffer.entries) == 1


def test_retained_older_context_is_labeled_stale_not_current(buffer):
    buffer.accept(make_span(input_version=1), 1)
    selected, _ = buffer.select(0, 2, 1)
    assert selected.span.input_version == 1
    assert selected.state == "stale"


@pytest.mark.parametrize("ready_ms", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_readiness_timestamps_are_rejected(buffer, ready_ms):
    with pytest.raises(ValueError):
        buffer.accept(make_span(), ready_ms)
    assert buffer.entries == []


@pytest.mark.parametrize("ready_ms,timings", [
    (0, {}), (137, {"tts_ms": 9}),
    (1000000, {"llm_ttft_ms": None, "server_duration_ms": 50000}),
])
def test_frame_geometry_is_independent_of_latency_metrics(buffer, ready_ms, timings):
    buffer.accept(make_span(start=3, duration=5, stage_timings=timings,
                            safe_split_ticks=(2,), first_vocal_tick=2), ready_ms)
    assert buffer.prefix_ticks(1) == 0
    selected, end = buffer.select(75, 1, 1)
    assert end == 125
    assert selected.audio.shape == (125, 1)
    assert selected.span.first_vocal_tick == 2
    selected, end = buffer.select(125, 1, 1)
    assert end == 200


@pytest.mark.parametrize("chunks", [(12,), (4, 4, 4), (1,) * 12, (3, 1, 6, 2)])
def test_packet_geometry_preserves_exact_pcm_and_duration(buffer, chunks):
    start = 0
    for duration in chunks:
        buffer.accept(make_span(start=start, duration=duration,
                                safe_split_ticks=tuple(range(1, duration))), start * 999)
        start += duration
    assert buffer.prefix_ticks(1) == 12
    rendered = []
    frame = 0
    while frame < 300:
        entry, end = buffer.select(frame, 1, 1)
        assert entry is not None
        offset = frame - entry.span.start_tick * 25
        rendered.append(entry.audio[offset:offset + end - frame])
        frame = end
    assert frame == 300
    np.testing.assert_array_equal(np.concatenate(rendered), np.full((300, 1), 0.25))


def test_expired_unsplittable_audio_is_not_replayed(buffer):
    buffer.accept(make_span(duration=1), 1)
    buffer.discard_before(25)
    assert buffer.select(25, 1, 1)[0] is None
    assert buffer.snapshot()[0]["state"] == "late"


def test_diagnostic_compaction_does_not_discard_unplayed_accepted_coverage(buffer):
    for start in range(130):
        assert buffer.accept(make_span(start=start, duration=1), start)
    assert buffer.prefix_ticks(1) == 130
    buffer.discard_before(0)
    assert buffer.prefix_ticks(1) == 130
    assert buffer.select(0, 1, 1)[0] is not None
