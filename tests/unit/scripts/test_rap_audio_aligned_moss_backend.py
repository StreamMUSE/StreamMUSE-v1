from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.io import wavfile

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

    staged = backend.stage_alignment_inputs(pending, tmp_path / "mfa-corpus")

    assert len(staged) == 1
    assert staged[0].source_sha256 == source_sha256
    assert staged[0].staged_wav_path.read_bytes() == source_wav.read_bytes()
    assert staged[0].staged_lab_path.read_text(encoding="utf-8") == _request().text

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
        crossfade_seconds=0.0,
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


def test_render_aligned_chunk_returns_explicit_failure_for_missing_vowel_anchor(tmp_path: Path) -> None:
    backend = _load_backend()
    source_wav = tmp_path / "source.wav"
    _, source_sha256 = _write_source_wav(source_wav)
    textgrid_path = tmp_path / "aligned.TextGrid"
    _write_textgrid(textgrid_path, missing_last_vowel=True)

    result = backend.render_aligned_chunk(
        request=_request(),
        source_wav_path=source_wav,
        expected_source_sha256=source_sha256,
        textgrid_path=textgrid_path,
        output_wav_path=tmp_path / "warped.wav",
        stretch_region=_impulse_stretcher,
        crossfade_seconds=0.0,
    )

    assert not result.record.success
    assert result.record.source_chunk_sha256 == source_sha256
    assert "aligned vowel count" in (result.record.error or "")
