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


def _liftoff_request() -> TwoBarRenderRequest:
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
                    target_seconds=0.15 + (syllable_index * 0.25),
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


def _write_textgrid(path: Path, *, missing_last_vowel: bool = False) -> None:
    intervals = []
    for index in range(18 - int(missing_last_vowel)):
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
        """
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
                intervals: size = 18
        """
        + "".join(intervals),
        encoding="utf-8",
    )


def _write_liftoff_textgrid(path: Path) -> None:
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
        phones.append(
            f"""
            intervals [{index + 1}]:
                xmin = {phone_start:.3f}
                xmax = {phone_end:.3f}
                text = "{phone}"
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
    assert result.diagnostics_path == diagnostics_path
    assert diagnostics["fallback_count"] == 0
    assert all(not anchor["aligned_phone"].startswith("WORD_TIER_FALLBACK:") for anchor in diagnostics["anchor_map"])


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
    assert diagnostics["source_sha256"] == source_sha256
    assert diagnostics["fallback_count"] == 2
    assert len(diagnostics["anchor_map"]) == 18
    assert diagnostics["stretch_ratios"] == list(result.stretch_ratios)
    assert [
        anchor["aligned_phone"]
        for anchor in diagnostics["anchor_map"]
        if anchor["aligned_phone"].startswith("WORD_TIER_FALLBACK:")
    ] == ["WORD_TIER_FALLBACK:liftoff", "WORD_TIER_FALLBACK:liftoff"]
    assert list(tmp_path.glob(".warped.wav.alignment.json.*.tmp")) == []


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
    assert "aligned vowel count" in diagnostics["error"]

    ledger_path = tmp_path / "records.jsonl"
    append_chunk_record(ledger_path, result.record)
    assert not chunk_record_is_complete(
        ledger_path,
        output_wav,
        request=_request(),
        protocol_id=ProtocolId.MOSS_ALIGNED,
    )
