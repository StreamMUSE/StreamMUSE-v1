"""Per-bar MOSS preparation retains timing evidence and reusable stems."""

from dataclasses import replace
import hashlib
import io
import wave

import numpy as np
import pytest

from streammuse.application.rap.bar_audio import BarAudioPreparer, PreparedBarVocal
from streammuse.application.rap.chunk_audio import LocalPhraseWarp
from streammuse.application.rap.audio_rendering import bar_frame_count
from streammuse.domain.rap import AudioFormat, FlowProvenance, FlowSlot, FlowTemplate, PcmAudio, ScheduledSyllable, Syllable, materialize_flow
from streammuse.domain.rap.audio import RapMixConfig
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
from streammuse.domain.rap.timeline import RapTimeline
from streammuse.domain.timing import Tempo


def wav_bytes(frames: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24_000)
        output.writeframes(np.full(frames, 1_000, dtype="<i2").tobytes())
    return buffer.getvalue()


def record_and_manifest(*, bar: int = 0, bpm: float = 120.0):
    flow = FlowTemplate("four", "Four", 4, 4, tuple(FlowSlot(t, 1, 0.5) for t in (0, 4, 8, 12)), FlowProvenance("test", "test"))
    record = BarLyricRecord(
        "session", bar, 0, "space", flow, "one two three four",
        tuple(ScheduledSyllable(slot, Syllable(word, 0, 1, 1, ("AH1",), "test"))
              for slot, word in zip(materialize_flow(flow, bar), ("one", "two", "three", "four"), strict=True)),
        0.5,
    )
    wav = wav_bytes(48_000)
    manifest = BarAudioManifest(
        record.lyric_id, bar, bpm, 48_000, hashlib.sha256(wav).hexdigest(),
        (0.1, 0.55, 0.98, 1.5), (0.9, 0.6, 0.95, 0.8), {},
    )
    return record, manifest, wav


class Drums:
    def render(self, template, tempo, audio_format, bar):
        frames = bar_frame_count(bar, tempo, audio_format)
        return PcmAudio(audio_format, frames, np.full((frames, 2), 0.2, dtype=np.float32).tobytes())


class Warper:
    def __init__(self, *, diagnostics=True):
        self.calls = []
        self.diagnostics = diagnostics

    def warp(self, request, source_wav, source_onsets, *, target_frame_count):
        self.calls.append(request)
        wav = wav_bytes(target_frame_count)
        if not self.diagnostics:
            return PreparedBarVocal(wav, 12.0)
        targets = tuple(item.target_seconds + (0.15 if index == 1 else 0) for index, item in enumerate(request.syllables))
        return LocalPhraseWarp(wav, (0.0, 0.45, 0.88, 1.4), targets, (0.8, 1.2), 12.0, "test", "gentle_sparse_r3", (2, 0))


def preparer(warper, *, bpm=120.0):
    return BarAudioPreparer(warper=warper, drums=Drums(), tempo=Tempo(bpm, 4, 4), audio_format=AudioFormat())


def test_moss_preparation_retains_sparse_warp_drift_source_alignment_and_confidence():
    record, manifest, source = record_and_manifest()
    prepared = preparer(Warper()).prepare(record, manifest, source)

    assert prepared.audio.frame_count == 96_000
    assert [item.requested_sample for item in prepared.onset_estimates] == [0, 24_000, 48_000, 72_000]
    assert [item.estimated_sample for item in prepared.onset_estimates] == [0, 31_200, 48_000, 72_000]
    assert [item.confidence for item in prepared.onset_estimates] == [0.9, 0.6, 0.95, 0.8]
    assert all(item.method == "source_alignment_warp_estimate" for item in prepared.onset_estimates)
    assert prepared.warp_diagnostics.source_onsets == manifest.source_onsets
    assert prepared.warp_diagnostics.source_anchors == (0.0, 0.45, 0.88, 1.4)
    assert prepared.warp_diagnostics.target_anchors == (0.0, 0.65, 1.0, 1.5)
    assert prepared.warp_diagnostics.r3_attempt_deviations == (2, 0)


def test_moss_missing_warp_diagnostics_stays_explicitly_unavailable():
    record, manifest, source = record_and_manifest()
    prepared = preparer(Warper(diagnostics=False)).prepare(record, manifest, source)

    assert len(prepared.onset_estimates) == 4
    assert all(item.estimated_sample is None and item.method == "unavailable" for item in prepared.onset_estimates)
    assert all(item.unavailable_reason for item in prepared.onset_estimates)
    assert prepared.warp_diagnostics is None


def test_moss_initial_preparation_includes_selected_chord_loop():
    record, manifest, source = record_and_manifest()
    renderer = preparer(Warper())
    renderer.set_mix_config(RapMixConfig(chord_loop="minor_cycle", chord_gain=0.4))
    prepared = renderer.prepare(record, manifest, source)
    assert prepared.chord_stem is not None
    assert prepared.chord_stem.frame_count == prepared.audio.frame_count
    assert np.max(np.abs(np.frombuffer(prepared.chord_stem.data, dtype=np.float32))) > 0


def test_output_rate_conversion_cannot_leak_vocals_into_the_leading_rest():
    record, manifest, source = record_and_manifest()
    flow = replace(record.flow_template, slots=tuple(
        replace(slot, tick_in_bar=slot.tick_in_bar + 2) for slot in record.flow_template.slots))
    shifted = replace(record, flow_template=flow, scheduled=tuple(
        replace(item, slot=slot) for item, slot in zip(record.scheduled, materialize_flow(flow, record.bar))))
    manifest = replace(manifest, lyric_id=shifted.lyric_id)
    prepared = preparer(Warper()).prepare(shifted, manifest, source)
    samples = np.frombuffer(prepared.vocal_stem.data, dtype=np.float32).reshape(-1, 2)
    assert np.count_nonzero(samples[:12000]) == 0
    assert np.max(np.abs(samples[12000:])) > 0


def test_moss_gain_only_remix_does_not_warp_again_or_modify_prepared_audio():
    record, manifest, source = record_and_manifest()
    warper = Warper()
    renderer = preparer(warper)
    original = renderer.prepare(record, manifest, source)
    original_bytes = original.audio.data
    config = RapMixConfig(vocal_gain=0.0, drum_gain=0.5, revision=2)
    renderer.set_mix_config(config)
    remixed = renderer.remix(original)

    assert len(warper.calls) == 1
    assert np.frombuffer(remixed.audio.data, dtype=np.float32) == pytest.approx(0.1)
    assert original.audio.data == original_bytes
    assert remixed.vocal_stem is original.vocal_stem
    assert remixed.onset_estimates is original.onset_estimates
    assert remixed.mix_config == config


def test_moss_preparation_uses_exact_timeline_snapshot_and_rejects_old_tempo_source():
    timing = RapTimeline(Tempo(92.0, 4, 4)).with_tempo(120.0, 2).snapshot(3, 48_000)
    record, manifest, source = record_and_manifest(bar=3)
    warper = Warper()
    renderer = preparer(warper, bpm=92.0)
    prepared = renderer.prepare(record, manifest, source, timing=timing)

    assert prepared.timing is timing
    assert prepared.audio.frame_count == 96_000
    assert warper.calls[0].tempo_bpm == 120.0
    with pytest.raises(ValueError, match="tempo"):
        renderer.prepare(record, replace(manifest, tempo_bpm=92.0), source, timing=timing)
    assert len(warper.calls) == 1


def test_moss_rejects_wrong_destination_before_warp():
    record, manifest, source = record_and_manifest()
    warper = Warper()
    timing = RapTimeline(Tempo(120.0, 4, 4)).snapshot(1, 48_000)
    with pytest.raises(ValueError, match="destination"):
        preparer(warper).prepare(record, manifest, source, timing=timing)
    assert not warper.calls


@pytest.mark.parametrize(("bar", "frames"), [(1, 125_218), (2, 125_217), (3, 125_218)])
def test_moss_preserves_exact_noninteger_tempo_lengths_after_remix(bar, frames):
    record, manifest, source = record_and_manifest(bar=bar, bpm=92.0)
    renderer = preparer(Warper(), bpm=92.0)
    prepared = renderer.prepare(record, manifest, source)
    remixed = renderer.remix(prepared, RapMixConfig.with_drum_boost(revision=1))
    assert prepared.audio.frame_count == remixed.audio.frame_count == frames
    assert prepared.vocal_stem.frame_count == prepared.drum_stem.frame_count == frames


def test_server_continuation_and_workspace_are_keyed_by_source_tempo(tmp_path):
    from streammuse.application.rap.bar_audio import MossBarAudioRenderer
    from streammuse.application.rap.chunk_orchestration import PhraseSourceResult
    from streammuse.infrastructure.rap.bar_package import PcmBarPackageEncoder

    class Source:
        def __init__(self):
            self.calls = []

        def render_source(self, request, workspace, *, prefix=None):
            self.calls.append((workspace, prefix))
            return PhraseSourceResult(
                wav_bytes(48_000),
                {"source_onsets": (0.1, 0.55, 0.98, 1.5), "onset_confidence": (0.9,) * 4},
                {"frame_count": 48_000}, {}, (), {"moss": 0.0, "aligner": 0.0}, {},
            )

    source = Source()
    renderer = MossBarAudioRenderer(source, PcmBarPackageEncoder(), workspace_root=tmp_path)
    first, _, _ = record_and_manifest()
    second, _, _ = record_and_manifest(bar=1)
    renderer.render(first, tempo_bpm=120.0)
    other_tempo = renderer.render(second, tempo_bpm=90.0, previous=first)
    assert source.calls[1][1] is None
    assert other_tempo.manifest.diagnostics["continuation"]["used"] is False
    renderer.render(first, tempo_bpm=90.0)
    assert source.calls[0][0] != source.calls[2][0]
    matching = renderer.render(second, tempo_bpm=90.0, previous=first)
    assert matching.manifest.diagnostics["continuation"]["used"] is True
    assert source.calls[3][1].text == first.text


def test_alignment_preparer_snapshots_do_not_mutate_existing_policy_or_mix():
    class PolicyWarper(Warper):
        def __init__(self, policy="gentle_sparse_r3"):
            super().__init__()
            self.policy = policy

        def with_policy(self, policy):
            if policy not in ("gentle_sparse_r3", "all_onsets_r3", "all_onsets_stress_r3"):
                raise ValueError("invalid alignment policy")
            return PolicyWarper(policy)

        def warp(self, *args, **kwargs):
            return replace(super().warp(*args, **kwargs), policy=self.policy)

    record, manifest, source = record_and_manifest()
    original = preparer(PolicyWarper())
    original.set_mix_config(RapMixConfig(drum_gain=0.3))
    strict = original.with_alignment_policy("all_onsets_r3")
    accented = strict.with_alignment_policy("all_onsets_stress_r3")
    assert original.prepare(record, manifest, source).warp_diagnostics.policy == "gentle_sparse_r3"
    assert strict.prepare(record, manifest, source).warp_diagnostics.policy == "all_onsets_r3"
    result = accented.prepare(record, manifest, source)
    assert result.warp_diagnostics.policy == "all_onsets_stress_r3"
    assert result.mix_config.drum_gain == 0.3
    with pytest.raises(ValueError):
        original.with_alignment_policy("invalid")
