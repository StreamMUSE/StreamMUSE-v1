from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import (
    append_chunk_record,
    chunk_record_is_complete,
    file_sha256,
)
from streammuse.experiments.rap_audio_protocols.contracts import ProtocolId, SyllableTarget, TwoBarRenderRequest


ROOT = Path(__file__).resolve().parents[3]


def _load_backend() -> ModuleType:
    module_name = "_streammuse_test_script_aligned_moss_backend"
    script_path = ROOT / "scripts" / "rap_audio_backends" / "aligned_moss_backend.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _request() -> TwoBarRenderRequest:
    syllables = []
    words = [f"word{chr(97 + index)}" for index in range(18)]
    for index, word in enumerate(words):
        syllables.append(
            SyllableTarget(
                word=word,
                index_in_word=0,
                phonemes=("AA1",),
                lexical_stress=1,
                target_stress=1.0,
                boundary_strength=0,
                absolute_tick=index,
                tick_in_chunk=index,
                target_seconds=0.15 + (index * 0.25),
            )
        )
    return TwoBarRenderRequest(
        song_id="01_space_exploration",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text=" ".join(words),
        syllables=tuple(syllables),
    )


def _liftoff_request(*, first_target_seconds: float = 0.15) -> TwoBarRenderRequest:
    words = (
        "ignite",
        "the",
        "night",
        "we",
        "rise",
        "for",
        "liftoff",
        "and",
        "carry",
        "the",
        "signal",
        "through",
        "the",
        "stars",
        "until",
        "the",
        "dawn",
    )
    syllables = []
    for word in words:
        pronunciations = (("L", "IH1", "F"), ("T", "AO1", "F")) if word == "liftoff" else (("AA1",),)
        for index_in_word, phonemes in enumerate(pronunciations):
            syllable_index = len(syllables)
            syllables.append(
                SyllableTarget(
                    word=word,
                    index_in_word=index_in_word,
                    phonemes=phonemes,
                    lexical_stress=1,
                    target_stress=1.0,
                    boundary_strength=0,
                    absolute_tick=syllable_index,
                    tick_in_chunk=syllable_index,
                    target_seconds=first_target_seconds + (syllable_index * 0.25),
                )
            )
    return TwoBarRenderRequest(
        song_id="01_space_exploration",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text=" ".join(words),
        syllables=tuple(syllables),
    )


def _write_source_wav(path: Path, *, sample_rate_hz: int = 1_000) -> tuple[np.ndarray, str]:
    frame_count = round((16 / 3) * sample_rate_hz)
    samples = np.zeros(frame_count, dtype=np.float32)
    for syllable in _request().syllables:
        samples[round((syllable.target_seconds - 0.04) * sample_rate_hz)] = 1.0
    wavfile.write(path, sample_rate_hz, samples)
    return samples, hashlib.sha256(path.read_bytes()).hexdigest()


def _write_textgrid(
    path: Path,
    *,
    missing_last_vowel: bool = False,
    first_vowel_interval: tuple[float, float] | None = None,
) -> None:
    intervals = []
    interval_count = 18 - int(missing_last_vowel)
    for index in range(interval_count):
        if index == 0 and first_vowel_interval is not None:
            start, end = first_vowel_interval
        else:
            start = 0.10 + (index * 0.25)
            end = start + 0.04
        intervals.append(
            f"""
            intervals [{index + 1}]:
                xmin = {start:.3f}
                xmax = {end:.3f}
                text = "AA1"
            """
        )
    path.write_text(
        f"""
        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0
        xmax = 5.50
        tiers? <exists>
        size = 1
        item []:
            item [1]:
                class = "IntervalTier"
                name = "phones"
                xmin = 0
                xmax = 5.50
                intervals: size = {interval_count}
        """
        + "".join(intervals),
        encoding="utf-8",
    )


def _write_liftoff_textgrid(path: Path, *, malformed_phone_index: int | None = None) -> None:
    request = _liftoff_request()
    words = []
    phones = []
    for index, word in enumerate(request.text.split()):
        word_start = 0.05 + (index * 0.25)
        word_end = word_start + 0.18
        words.append(
            f"""
            intervals [{index + 1}]:
                xmin = {word_start:.3f}
                xmax = {word_end:.3f}
                text = "{word}"
            """
        )
        phone_start = word_start if word == "liftoff" else word_start + 0.04
        phone_end = word_end if word == "liftoff" else phone_start + 0.04
        phone = "spn" if word == "liftoff" else "AA1"
        label_field = "label" if index == malformed_phone_index else "text"
        phones.append(
            f"""
            intervals [{index + 1}]:
                xmin = {phone_start:.3f}
                xmax = {phone_end:.3f}
                {label_field} = "{phone}"
            """
        )
    path.write_text(
        """
        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0
        xmax = 5.50
        tiers? <exists>
        size = 2
        item []:
            item [1]:
                class = "IntervalTier"
                name = "words"
                xmin = 0
                xmax = 5.50
                intervals: size = 17
        """
        + "".join(words)
        + """
            item [2]:
                class = "IntervalTier"
                name = "phones"
                xmin = 0
                xmax = 5.50
                intervals: size = 17
        """
        + "".join(phones),
        encoding="utf-8",
    )


def _write_owned_textgrid(
    path: Path,
    *,
    request: TwoBarRenderRequest | None = None,
    extra_gap_vowel: bool = False,
    mismatched_phone_index: int | None = None,
    overlapping_words: bool = False,
    phone_labels: tuple[str, ...] | None = None,
    spn_words: frozenset[str] = frozenset(),
    swap_first_two_phone_entries: bool = False,
) -> None:
    aligned_request = request or _request()
    request_words = aligned_request.text.split()
    words = []
    phone_values = []
    for index, word in enumerate(request_words):
        word_start = 0.05 + (index * 0.25)
        word_end = 0.38 if overlapping_words and index == 0 else word_start + 0.18
        words.append(
            f"""
            intervals [{index + 1}]:
                xmin = {word_start:.3f}
                xmax = {word_end:.3f}
                text = "{word}"
            """
        )
        if word in spn_words:
            phone_start = word_start
            phone_end = word_end
            phone = "spn"
        else:
            phone_start = 0.10 + (index * 0.25)
            phone_end = phone_start + 0.04
            phone = phone_labels[index] if phone_labels is not None else "AA1"
        if index == mismatched_phone_index:
            phone = "AE1"
        phone_values.append((phone_start, phone_end, phone))
    if extra_gap_vowel:
        phone_values.append((0.25, 0.27, "AA1"))
    phone_values.sort()
    if swap_first_two_phone_entries:
        phone_values[0], phone_values[1] = phone_values[1], phone_values[0]
    phones = [
        f"""
        intervals [{index + 1}]:
            xmin = {start:.3f}
            xmax = {end:.3f}
            text = "{phone}"
        """
        for index, (start, end, phone) in enumerate(phone_values)
    ]
    path.write_text(
        f"""
        File type = "ooTextFile"
        Object class = "TextGrid"

        xmin = 0
        xmax = 5.50
        tiers? <exists>
        size = 2
        item []:
            item [1]:
                class = "IntervalTier"
                name = "words"
                xmin = 0
                xmax = 5.50
                intervals: size = {len(request_words)}
        """
        + "".join(words)
        + f"""
            item [2]:
                class = "IntervalTier"
                name = "phones"
                xmin = 0
                xmax = 5.50
                intervals: size = {len(phone_values)}
        """
        + "".join(phones),
        encoding="utf-8",
    )


def _impulse_stretcher(samples: np.ndarray, target_frames: int, sample_rate_hz: int) -> np.ndarray:
    del sample_rate_hz
    output = np.zeros(target_frames, dtype=np.float32)
    if len(samples) == 0 or target_frames == 0:
        return output
    if len(samples) == 1:
        output[:] = samples[0]
        return output
    nonzero = np.flatnonzero(np.abs(samples) > 0.5)
    for index in nonzero:
        mapped = round(index * (target_frames - 1) / (len(samples) - 1))
        output[mapped] = samples[index]
    return output


def _linear_full_chunk_stretcher(
    samples: np.ndarray,
    target_frames: int,
    sample_rate_hz: int,
    time_map: tuple[tuple[int, int], ...],
) -> np.ndarray:
    del sample_rate_hz, time_map
    source_positions = np.linspace(0.0, 1.0, len(samples), dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, target_frames, dtype=np.float32)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def test_stage_alignment_inputs_enforces_source_sha_and_writes_lab_pairs(tmp_path: Path) -> None:
    backend = _load_backend()
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    pending = (
        backend.PendingAlignedChunk(
            request=_request(),
            source_wav_path=source_wav,
            expected_source_sha256=source_sha256,
        ),
    )
    alignment_output_dir = tmp_path / "mfa-output"

    staged = backend.stage_alignment_inputs(
        pending,
        tmp_path / "mfa-corpus",
        output_dir=alignment_output_dir,
    )

    assert len(staged) == 1
    assert staged[0].source_sha256 == source_sha256
    assert staged[0].staged_wav_path.read_bytes() == source_wav.read_bytes()
    assert staged[0].staged_lab_path.read_text(encoding="utf-8") == _request().text
    assert staged[0].expected_textgrid_path == alignment_output_dir / "01_space_exploration__chunk_00.TextGrid"

    with pytest.raises(ValueError, match="SHA-256"):
        backend.stage_alignment_inputs(
            (
                backend.PendingAlignedChunk(
                    request=_request(),
                    source_wav_path=source_wav,
                    expected_source_sha256="0" * 64,
                ),
            ),
            tmp_path / "bad-corpus",
            output_dir=tmp_path / "bad-output",
        )


def test_run_forced_alignment_uses_injected_mfa_command_boundary(tmp_path: Path) -> None:
    backend = _load_backend()
    calls = []

    def fake_mfa_command(*, corpus_dir: Path, output_dir: Path, dictionary_name: str, acoustic_model_name: str) -> None:
        calls.append((corpus_dir, output_dir, dictionary_name, acoustic_model_name))

    backend.run_forced_alignment(
        tmp_path / "corpus",
        tmp_path / "aligned",
        mfa_command=fake_mfa_command,
    )

    assert calls == [
        (
            tmp_path / "corpus",
            tmp_path / "aligned",
            "english_us_arpa",
            "english_us_arpa",
        )
    ]


def test_render_aligned_chunk_propagates_source_sha_and_logged_stretch_ratios(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _request()
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    output_wav = tmp_path / "warped.wav"
    _write_textgrid(textgrid_path)

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        attempts=2,
        stretch_region=_impulse_stretcher,
    )

    sample_rate_hz, warped = wavfile.read(output_wav)

    assert result.record.protocol_id is ProtocolId.MOSS_ALIGNED
    assert result.record.success
    assert result.record.source_chunk_sha256 == source_sha256
    assert result.record.attempts == 2
    assert sample_rate_hz == 1_000
    assert len(warped) == round(request.duration_seconds * sample_rate_hz)
    assert len(result.anchor_map) == 18
    assert len(result.stretch_ratios) > 0
    assert all(ratio > 0 for ratio in result.stretch_ratios)
    assert all(
        warped[round(syllable.target_seconds * sample_rate_hz)] == pytest.approx(1.0)
        for syllable in request.syllables
    )
    diagnostics_path = output_wav.with_suffix(output_wav.suffix + ".alignment.json")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert result.fallback_count == 0
    assert result.boundary_adjustment_count == 0
    assert result.source_boundary_adjustment_count == 0
    assert result.diagnostics_path == diagnostics_path
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["boundary_adjustment_count"] == 0
    assert diagnostics["source_boundary_adjustment_count"] == 0
    assert all(not anchor["aligned_phone"].startswith("WORD_TIER_FALLBACK:") for anchor in diagnostics["anchor_map"])


def test_render_aligned_chunk_supports_auditable_constrained_onset_stress_mode(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    request = _request()
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    output_wav = tmp_path / "warped.wav"
    _write_textgrid(textgrid_path)
    calls: list[tuple[tuple[int, int], ...]] = []

    def recording_stretcher(
        samples: np.ndarray,
        target_frames: int,
        sample_rate_hz: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        calls.append(time_map)
        return _linear_full_chunk_stretcher(samples, target_frames, sample_rate_hz, time_map)

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        mode="continuous_onset_constrained_r3_stress",
        stretch_full_chunk=recording_stretcher,
    )

    diagnostics = json.loads(
        output_wav.with_suffix(output_wav.suffix + ".alignment.json").read_text(encoding="utf-8")
    )
    assert result.record.success
    assert len(calls) == 1
    assert result.mode == "continuous_onset_constrained_r3_stress"
    assert result.stress_applied
    assert all(anchor.anchor_kind == "syllable_onset" for anchor in result.anchor_map)
    assert diagnostics["mode"] == "continuous_onset_constrained_r3_stress"
    assert diagnostics["stress"]["applied"] is True
    assert diagnostics["stress"]["target_seconds"] == pytest.approx(
        [anchor.target_seconds for anchor in result.anchor_map]
    )
    assert diagnostics["stress"]["requested_target_seconds"] == pytest.approx(
        [syllable.target_seconds for syllable in request.syllables]
    )
    assert diagnostics["timing_regularization"]["applied"] is True
    assert len(diagnostics["timing_regularization"]["target_drift_seconds"]) == 18


def test_render_aligned_chunk_supports_gentle_sparse_r3_without_stress(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    base_request = _request()
    selected_indices = (0, 4, 10, 17)
    request = replace(
        base_request,
        syllables=tuple(
            replace(
                syllable,
                target_stress=0.8 if index == 4 else 0.2,
                boundary_strength=2 if index == 10 else 0,
            )
            for index, syllable in enumerate(base_request.syllables)
        ),
    )
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    output_wav = tmp_path / "warped.wav"
    _write_textgrid(textgrid_path)
    calls: list[tuple[tuple[int, int], ...]] = []

    def recording_stretcher(
        samples: np.ndarray,
        target_frames: int,
        sample_rate_hz: int,
        time_map: tuple[tuple[int, int], ...],
    ) -> np.ndarray:
        calls.append(time_map)
        return _linear_full_chunk_stretcher(
            samples,
            target_frames,
            sample_rate_hz,
            time_map,
        )

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        mode="continuous_onset_gentle_sparse_r3",
        stretch_full_chunk=recording_stretcher,
    )

    diagnostics = json.loads(
        output_wav.with_suffix(output_wav.suffix + ".alignment.json").read_text(
            encoding="utf-8"
        )
    )
    assert result.record.success
    assert result.mode == "continuous_onset_gentle_sparse_r3"
    assert not result.stress_applied
    assert len(calls) == 1
    assert len(calls[0]) == len(selected_indices) + 2
    assert len(result.anchor_map) == len(selected_indices)
    assert all(anchor.anchor_kind == "syllable_onset" for anchor in result.anchor_map)
    assert min(result.stretch_ratios) >= 0.75 - 0.01
    assert max(result.stretch_ratios) <= 1.35 + 0.01
    assert diagnostics["stress"]["applied"] is False
    regularization = diagnostics["timing_regularization"]
    assert regularization["applied"] is True
    assert regularization["policy"] == "gentle_sparse_r3"
    assert regularization["min_stretch_ratio"] == 0.75
    assert regularization["max_stretch_ratio"] == 1.35
    assert regularization["stress_priority"] == 4.0
    assert regularization["minimum_target_stress"] == 0.8
    assert regularization["minimum_boundary_strength"] == 2
    assert regularization["input_anchor_count"] == 18
    assert regularization["effective_anchor_count"] == 4
    assert regularization["selected_anchor_indices"] == list(selected_indices)
    assert regularization["omitted_anchor_indices"] == [
        index for index in range(18) if index not in selected_indices
    ]
    assert len(regularization["target_drift_seconds"]) == 18
    assert regularization["max_absolute_target_drift_seconds"] == pytest.approx(
        max(abs(value) for value in regularization["target_drift_seconds"])
    )


def test_render_aligned_chunk_falls_back_only_for_liftoff_spn_word(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _liftoff_request()
    source_wav = tmp_path / "source.wav"
    wavfile.write(source_wav, 1_000, np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32))
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_liftoff_textgrid(textgrid_path)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert result.record.success
    assert len(result.anchor_map) == 18
    fallback_anchors = [
        anchor
        for anchor in result.anchor_map
        if anchor.aligned_phone.startswith("WORD_TIER_FALLBACK:")
    ]
    assert [(anchor.word, anchor.index_in_word) for anchor in fallback_anchors] == [
        ("liftoff", 0),
        ("liftoff", 1),
    ]
    assert all(
        anchor.aligned_phone == "AA1"
        for anchor in result.anchor_map
        if anchor.word != "liftoff"
    )
    diagnostics_path = output_wav.with_suffix(output_wav.suffix + ".alignment.json")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert result.diagnostics_path == diagnostics_path
    assert result.fallback_count == 2
    assert result.boundary_adjustment_count == 0
    assert diagnostics["source_sha256"] == source_sha256
    assert diagnostics["fallback_count"] == 2
    assert diagnostics["boundary_adjustment_count"] == 0
    assert len(diagnostics["anchor_map"]) == 18
    assert diagnostics["stretch_ratios"] == list(result.stretch_ratios)
    assert [
        anchor["aligned_phone"]
        for anchor in diagnostics["anchor_map"]
        if anchor["aligned_phone"].startswith("WORD_TIER_FALLBACK:")
    ] == ["WORD_TIER_FALLBACK:liftoff", "WORD_TIER_FALLBACK:liftoff"]
    assert list(tmp_path.glob(".warped.wav.alignment.json.*.tmp")) == []


def test_render_aligned_chunk_falls_back_for_single_syllable_oov_possessive(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    request = _request()
    syllables = list(request.syllables)
    syllables[0] = replace(
        syllables[0],
        word="gravity's",
        phonemes=("G", "R", "V", "T", "Y"),
    )
    request = replace(
        request,
        text=request.text.replace("worda", "gravity's", 1),
        syllables=tuple(syllables),
    )
    source_wav = tmp_path / "source.wav"
    wavfile.write(
        source_wav,
        1_000,
        np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32),
    )
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_owned_textgrid(
        textgrid_path,
        request=request,
        spn_words=frozenset({"gravity's"}),
    )
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert result.record.success
    fallback_anchors = [
        anchor
        for anchor in result.anchor_map
        if anchor.aligned_phone.startswith("WORD_TIER_FALLBACK:")
    ]
    assert len(fallback_anchors) == 1
    assert fallback_anchors[0].word == "gravity's"
    assert fallback_anchors[0].planned_phone == "UNKNOWN_PLANNED_VOWEL"
    assert fallback_anchors[0].aligned_phone == "WORD_TIER_FALLBACK:gravity's"
    assert result.fallback_count == 1

    diagnostics = json.loads(
        output_wav.with_suffix(output_wav.suffix + ".alignment.json").read_text(encoding="utf-8")
    )
    assert diagnostics["fallback_count"] == 1
    assert diagnostics["anchor_map"][0]["planned_phone"] == "UNKNOWN_PLANNED_VOWEL"
    assert diagnostics["anchor_map"][0]["aligned_phone"] == "WORD_TIER_FALLBACK:gravity's"


def test_render_aligned_chunk_falls_back_for_multisyllable_oov_possessive(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    request = _liftoff_request()
    syllables = tuple(
        replace(
            syllable,
            word="darkness's",
            phonemes=("D", "R", "K") if syllable.index_in_word == 0 else ("N", "S", "Z"),
        )
        if syllable.word == "liftoff"
        else syllable
        for syllable in request.syllables
    )
    request = replace(
        request,
        text=request.text.replace("liftoff", "darkness's", 1),
        syllables=syllables,
    )
    source_wav = tmp_path / "source.wav"
    wavfile.write(
        source_wav,
        1_000,
        np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32),
    )
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_owned_textgrid(
        textgrid_path,
        request=request,
        spn_words=frozenset({"darkness's"}),
    )
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    fallback_anchors = [
        anchor
        for anchor in result.anchor_map
        if anchor.aligned_phone.startswith("WORD_TIER_FALLBACK:")
    ]
    assert result.record.success
    assert [(anchor.word, anchor.index_in_word) for anchor in fallback_anchors] == [
        ("darkness's", 0),
        ("darkness's", 1),
    ]
    assert all(
        anchor.planned_phone == "UNKNOWN_PLANNED_VOWEL"
        for anchor in fallback_anchors
    )
    assert all(
        anchor.aligned_phone == "WORD_TIER_FALLBACK:darkness's"
        for anchor in fallback_anchors
    )
    assert (
        1.55
        < fallback_anchors[0].source_seconds
        < fallback_anchors[1].source_seconds
        < 1.73
    )
    assert result.fallback_count == 2

    diagnostics = json.loads(
        output_wav.with_suffix(output_wav.suffix + ".alignment.json").read_text(encoding="utf-8")
    )
    diagnostic_fallbacks = [
        anchor
        for anchor in diagnostics["anchor_map"]
        if anchor["aligned_phone"].startswith("WORD_TIER_FALLBACK:")
    ]
    assert diagnostics["fallback_count"] == 2
    assert [anchor["planned_phone"] for anchor in diagnostic_fallbacks] == [
        "UNKNOWN_PLANNED_VOWEL",
        "UNKNOWN_PLANNED_VOWEL",
    ]


def test_render_aligned_chunk_rejects_malformed_phone_interval_before_fallback(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _liftoff_request()
    source_wav = tmp_path / "source.wav"
    wavfile.write(source_wav, 1_000, np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32))
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_liftoff_textgrid(textgrid_path, malformed_phone_index=0)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "malformed phones IntervalTier" in (result.record.error or "")
    assert np.count_nonzero(wavfile.read(output_wav)[1]) == 0


def test_render_aligned_chunk_rejects_unowned_gap_vowel_instead_of_reporting_zero_fallback(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    request = _request()
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_owned_textgrid(textgrid_path, extra_gap_vowel=True)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "aligned vowel ownership" in (result.record.error or "")
    diagnostics = json.loads(
        output_wav.with_suffix(output_wav.suffix + ".alignment.json").read_text(encoding="utf-8")
    )
    assert diagnostics["success"] is False
    assert diagnostics["fallback_count"] == 0


def test_render_aligned_chunk_rejects_overlapping_word_intervals_before_ambiguous_ownership(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    request = _request()
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_owned_textgrid(
        textgrid_path,
        mismatched_phone_index=17,
        overlapping_words=True,
    )
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "matched word intervals must be chronological and non-overlapping" in (result.record.error or "")
    assert np.count_nonzero(wavfile.read(output_wav)[1]) == 0


def test_render_aligned_chunk_rejects_zero_fallback_after_strict_phone_order_failure(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    request = _request()
    syllables = list(request.syllables)
    syllables[0] = replace(syllables[0], phonemes=("IY1",))
    syllables[1] = replace(syllables[1], phonemes=("OW1",))
    request = replace(request, syllables=tuple(syllables))
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_owned_textgrid(
        textgrid_path,
        phone_labels=("IY1", "OW1", *(("AA1",) * 16)),
        swap_first_two_phone_entries=True,
    )
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "word-tier fallback produced no synthesized anchors" in (result.record.error or "")
    assert np.count_nonzero(wavfile.read(output_wav)[1]) == 0


def test_render_aligned_chunk_clamps_tick_zero_target_and_audits_adjustment(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _liftoff_request(first_target_seconds=0.0)
    source_wav = tmp_path / "source.wav"
    wavfile.write(source_wav, 1_000, np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32))
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_liftoff_textgrid(textgrid_path)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert result.record.success
    assert result.boundary_adjustment_count == 1
    assert result.anchor_map[0].requested_target_seconds == 0.0
    assert result.anchor_map[0].target_seconds == pytest.approx(0.010)
    assert result.anchor_map[0].target_sample == 10
    assert result.anchor_map[0].boundary_adjusted
    assert result.anchor_map[1].requested_target_seconds == pytest.approx(0.25)
    assert result.anchor_map[1].target_seconds == pytest.approx(0.25)
    assert not result.anchor_map[1].boundary_adjusted

    diagnostics_path = output_wav.with_suffix(output_wav.suffix + ".alignment.json")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["boundary_adjustment_count"] == 1
    assert diagnostics["anchor_map"][0]["requested_target_seconds"] == 0.0
    assert diagnostics["anchor_map"][0]["effective_target_seconds"] == pytest.approx(0.010)
    assert diagnostics["anchor_map"][0]["boundary_adjusted"] is True
    assert diagnostics["anchor_map"][1]["requested_target_seconds"] == pytest.approx(0.25)
    assert diagnostics["anchor_map"][1]["effective_target_seconds"] == pytest.approx(0.25)
    assert diagnostics["anchor_map"][1]["boundary_adjusted"] is False


def test_render_aligned_chunk_clamps_early_source_anchor_and_audits_adjustment(
    tmp_path: Path,
) -> None:
    backend = _load_backend()
    request = _request()
    syllables = list(request.syllables)
    syllables[0] = replace(syllables[0], target_seconds=0.0)
    request = replace(request, syllables=tuple(syllables))
    source_wav = tmp_path / "source.wav"
    wavfile.write(
        source_wav,
        1_000,
        np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32),
    )
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_textgrid(textgrid_path, first_vowel_interval=(0.005, 0.015))
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert result.record.success
    assert result.boundary_adjustment_count == 1
    assert result.source_boundary_adjustment_count == 1
    assert result.anchor_map[0].requested_source_seconds == pytest.approx(0.0075)
    assert result.anchor_map[0].source_seconds == pytest.approx(0.010)
    assert result.anchor_map[0].requested_source_sample == 8
    assert result.anchor_map[0].source_sample == 10
    assert result.anchor_map[0].source_boundary_adjusted
    assert result.anchor_map[1].requested_source_seconds == pytest.approx(0.360)
    assert result.anchor_map[1].source_seconds == pytest.approx(0.360)
    assert result.anchor_map[1].requested_source_sample == 360
    assert result.anchor_map[1].source_sample == 360
    assert not result.anchor_map[1].source_boundary_adjusted

    diagnostics = json.loads(
        output_wav.with_suffix(output_wav.suffix + ".alignment.json").read_text(encoding="utf-8")
    )
    first_diagnostic = diagnostics["anchor_map"][0]
    assert diagnostics["boundary_adjustment_count"] == 1
    assert diagnostics["source_boundary_adjustment_count"] == 1
    assert first_diagnostic["requested_source_seconds"] == pytest.approx(0.0075)
    assert first_diagnostic["effective_source_seconds"] == pytest.approx(0.010)
    assert first_diagnostic["requested_source_sample"] == 8
    assert first_diagnostic["effective_source_sample"] == 10
    assert first_diagnostic["source_boundary_adjusted"] is True


def test_render_aligned_chunk_fails_closed_when_boundary_clamp_collides(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _liftoff_request(first_target_seconds=0.0)
    syllables = list(request.syllables)
    syllables[1] = replace(syllables[1], target_seconds=0.010)
    request = replace(request, syllables=tuple(syllables))
    source_wav = tmp_path / "source.wav"
    wavfile.write(source_wav, 1_000, np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32))
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_liftoff_textgrid(textgrid_path)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "target boundary adjustment collides" in (result.record.error or "")
    assert np.count_nonzero(wavfile.read(output_wav)[1]) == 0


def test_render_aligned_chunk_fails_closed_for_negative_target_time(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _liftoff_request(first_target_seconds=-0.100)
    source_wav = tmp_path / "source.wav"
    wavfile.write(source_wav, 1_000, np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32))
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_liftoff_textgrid(textgrid_path)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "target anchor lies outside the target audio" in (result.record.error or "")
    assert np.count_nonzero(wavfile.read(output_wav)[1]) == 0


def test_render_aligned_chunk_fails_closed_for_target_beyond_duration(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _liftoff_request()
    syllables = list(request.syllables)
    syllables[-1] = replace(
        syllables[-1],
        target_seconds=request.duration_seconds + 0.100,
    )
    request = replace(request, syllables=tuple(syllables))
    source_wav = tmp_path / "source.wav"
    wavfile.write(source_wav, 1_000, np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32))
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_liftoff_textgrid(textgrid_path)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "target anchor lies outside the target audio" in (result.record.error or "")
    assert np.count_nonzero(wavfile.read(output_wav)[1]) == 0


def test_render_aligned_chunk_fails_closed_when_request_word_sequence_cannot_match(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _liftoff_request()
    request = replace(request, text=request.text.replace("liftoff", "launch"))
    source_wav = tmp_path / "source.wav"
    wavfile.write(source_wav, 1_000, np.zeros(round(request.duration_seconds * 1_000), dtype=np.float32))
    source_sha256 = file_sha256(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_liftoff_textgrid(textgrid_path)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=request,
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )

    assert not result.record.success
    assert "request word sequence" in (result.record.error or "")
    assert np.count_nonzero(wavfile.read(output_wav)[1]) == 0


def test_request_words_split_hyphenated_compounds_and_preserve_contractions() -> None:
    backend = _load_backend()

    assert backend._request_words("byte-strewn shores, don't stop") == (
        "byte",
        "strewn",
        "shores",
        "don't",
        "stop",
    )


def test_render_aligned_chunk_returns_explicit_failure_for_missing_vowel_anchor(tmp_path: Path) -> None:
    backend = _load_backend()
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_textgrid(textgrid_path, missing_last_vowel=True)
    output_wav = tmp_path / "warped.wav"

    result = backend.render_aligned_chunk(
        request=_request(),
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=output_wav,
        stretch_region=_impulse_stretcher,
    )
    sample_rate_hz, silence = wavfile.read(output_wav)

    assert not result.record.success
    assert result.record.source_chunk_sha256 == source_sha256
    assert "aligned vowel count" in (result.record.error or "")
    assert result.record.output_path == str(output_wav)
    assert result.record.output_sha256 == file_sha256(output_wav)
    assert result.record.sample_rate_hz == 1_000
    assert result.output_wav_path == output_wav
    assert sample_rate_hz == 1_000
    assert len(silence) == round(_request().duration_seconds * sample_rate_hz)
    assert np.count_nonzero(silence) == 0
    diagnostics_path = output_wav.with_suffix(output_wav.suffix + ".alignment.json")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert result.diagnostics_path == diagnostics_path
    assert diagnostics["success"] is False
    assert diagnostics["source_sha256"] == source_sha256
    assert diagnostics["anchor_map"] == []
    assert diagnostics["stretch_ratios"] == []
    assert diagnostics["fallback_count"] == 0
    assert diagnostics["boundary_adjustment_count"] == 0
    assert diagnostics["source_boundary_adjustment_count"] == 0
    assert "aligned vowel count" in diagnostics["error"]

    ledger_path = tmp_path / "records.jsonl"
    append_chunk_record(ledger_path, result.record)
    assert not chunk_record_is_complete(
        ledger_path,
        output_wav,
        request=_request(),
        protocol_id=ProtocolId.MOSS_ALIGNED,
    )
