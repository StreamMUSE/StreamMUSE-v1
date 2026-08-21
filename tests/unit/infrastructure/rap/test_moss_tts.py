from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from scripts.rap_audio_backends import moss_backend
from streammuse.experiments.rap_audio_protocols.contracts import (
    SyllableTarget,
    TwoBarRenderRequest,
)
from streammuse.experiments.rap_audio_protocols.timing import moss_token_target
from streammuse.infrastructure.rap.moss_tts import (
    MossSynthesisFailed,
    PersistentMossSynthesizer,
)


def _request() -> TwoBarRenderRequest:
    return TwoBarRenderRequest(
        song_id="test-song",
        chunk_index=0,
        start_bar=0,
        end_bar=2,
        text="steady motion",
        syllables=(
            SyllableTarget(
                word="steady",
                index_in_word=0,
                phonemes=("S", "T", "EH1"),
                lexical_stress=1,
                target_stress=1.0,
                boundary_strength=0,
                absolute_tick=1,
                tick_in_chunk=1,
                target_seconds=0.25,
            ),
            SyllableTarget(
                word="steady",
                index_in_word=1,
                phonemes=("D", "IY0"),
                lexical_stress=0,
                target_stress=0.5,
                boundary_strength=0,
                absolute_tick=5,
                tick_in_chunk=5,
                target_seconds=0.75,
            ),
            SyllableTarget(
                word="motion",
                index_in_word=0,
                phonemes=("M", "OW1"),
                lexical_stress=1,
                target_stress=1.0,
                boundary_strength=0,
                absolute_tick=9,
                tick_in_chunk=9,
                target_seconds=1.5,
            ),
            SyllableTarget(
                word="motion",
                index_in_word=1,
                phonemes=("SH", "AH0", "N"),
                lexical_stress=0,
                target_stress=0.5,
                boundary_strength=2,
                absolute_tick=13,
                tick_in_chunk=13,
                target_seconds=2.0,
            ),
        ),
    )


def test_loads_moss_once_and_reuses_exact_offline_settings_for_two_phrases(
    tmp_path: Path,
) -> None:
    reference_wav = tmp_path / "reference.wav"
    reference_wav.write_bytes(b"stable reference voice")
    runtime = SimpleNamespace(
        model_id=moss_backend.MODEL_ID,
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash="model-revision")),
        processor=SimpleNamespace(model_config=SimpleNamespace(sampling_rate=24_000)),
    )
    load_calls: list[dict[str, object]] = []
    generation_calls: list[tuple[object, Path, Path]] = []

    def load_runtime(**kwargs: object) -> object:
        load_calls.append(dict(kwargs))
        return runtime

    def generate_phrase(
        *,
        request: TwoBarRenderRequest,
        output_path: Path,
        reference_wav: Path,
        runtime: object,
    ) -> None:
        generation_calls.append((runtime, output_path, reference_wav))
        samples = np.linspace(-0.25, 0.25, 240, dtype=np.float32)
        wavfile.write(output_path, 24_000, samples)

    synthesizer = PersistentMossSynthesizer.load(
        model_id=moss_backend.MODEL_ID,
        device="cuda:3",
        reference_wav=reference_wav,
        runtime_loader=load_runtime,
        phrase_generator=generate_phrase,
    )

    first = synthesizer.synthesize(_request(), tmp_path / "request-a" / "source.wav")
    second = synthesizer.synthesize(_request(), tmp_path / "request-b" / "source.wav")

    assert load_calls == [{"model_id": moss_backend.MODEL_ID, "device": "cuda:3"}]
    assert generation_calls == [
        (runtime, tmp_path / "request-a" / ".source.partial.wav", reference_wav),
        (runtime, tmp_path / "request-b" / ".source.partial.wav", reference_wav),
    ]
    assert first.output_wav == tmp_path / "request-a" / "source.wav"
    assert second.output_wav == tmp_path / "request-b" / "source.wav"
    assert first.sample_rate_hz == second.sample_rate_hz == 24_000
    assert first.frame_count == second.frame_count == 240
    assert first.model_revision == second.model_revision == "model-revision"
    assert (
        first.reference_voice_sha256
        == hashlib.sha256(b"stable reference voice").hexdigest()
    )
    assert first.resolved_generation_settings == {
        "language": moss_backend.LANGUAGE,
        "instruction": moss_backend.RAP_INSTRUCTION,
        "generation_mode": moss_backend.GENERATION_MODE,
        "generation_kwargs": moss_backend.GENERATION_KWARGS,
        "token_target": moss_token_target(_request()),
    }
    with pytest.raises(TypeError):
        first.resolved_generation_settings["language"] = "changed"  # type: ignore[index]


def test_failed_synthesis_removes_stale_and_partial_outputs(tmp_path: Path) -> None:
    reference_wav = tmp_path / "reference.wav"
    reference_wav.write_bytes(b"reference")
    output_wav = tmp_path / "request" / "source.wav"
    output_wav.parent.mkdir()
    wavfile.write(output_wav, 24_000, np.ones(32, dtype=np.float32))

    def fail_generation(**_: object) -> None:
        raise RuntimeError("generation failed")

    synthesizer = PersistentMossSynthesizer.load(
        model_id=moss_backend.MODEL_ID,
        device="cuda:0",
        reference_wav=reference_wav,
        runtime_loader=lambda **_: SimpleNamespace(model=object()),
        phrase_generator=fail_generation,
    )

    with pytest.raises(MossSynthesisFailed, match="generation failed"):
        synthesizer.synthesize(_request(), output_wav)

    assert not output_wav.exists()
    assert not (output_wav.parent / ".source.partial.wav").exists()


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (np.zeros(32, dtype=np.float32), "silent"),
        (np.array([0.0, np.nan], dtype=np.float32), "finite"),
        (np.array([0.0, np.inf], dtype=np.float32), "finite"),
    ],
)
def test_rejects_silent_or_non_finite_moss_audio(
    tmp_path: Path,
    samples: np.ndarray,
    message: str,
) -> None:
    reference_wav = tmp_path / "reference.wav"
    reference_wav.write_bytes(b"reference")
    output_wav = tmp_path / "request" / "source.wav"

    def write_invalid(*, output_path: Path, **_: object) -> None:
        wavfile.write(output_path, 24_000, samples)

    synthesizer = PersistentMossSynthesizer.load(
        model_id=moss_backend.MODEL_ID,
        device="cuda:0",
        reference_wav=reference_wav,
        runtime_loader=lambda **_: SimpleNamespace(model=object()),
        phrase_generator=write_invalid,
    )

    with pytest.raises(MossSynthesisFailed, match=message):
        synthesizer.synthesize(_request(), output_wav)

    assert not output_wav.exists()


def test_warmup_executes_real_phrase_generation_without_reloading_model(
    tmp_path: Path,
) -> None:
    reference_wav = tmp_path / "reference.wav"
    reference_wav.write_bytes(b"reference")
    runtime = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(_commit_hash="revision"))
    )
    load_count = 0
    generated: list[tuple[TwoBarRenderRequest, Path]] = []

    def load_runtime(**_: object) -> object:
        nonlocal load_count
        load_count += 1
        return runtime

    def generate_phrase(
        *, request: TwoBarRenderRequest, output_path: Path, **_: object
    ) -> None:
        generated.append((request, output_path))
        wavfile.write(
            output_path,
            24_000,
            np.linspace(-0.1, 0.1, 64, dtype=np.float32),
        )

    synthesizer = PersistentMossSynthesizer.load(
        model_id=moss_backend.MODEL_ID,
        device="cuda:0",
        reference_wav=reference_wav,
        runtime_loader=load_runtime,
        phrase_generator=generate_phrase,
    )

    diagnostics = synthesizer.warmup()

    assert load_count == 1
    assert len(generated) == 1
    assert generated[0][0].text == "warm voice"
    assert generated[0][1].suffix == ".wav"
    assert diagnostics["generated"] is True
    assert diagnostics["sample_rate_hz"] == 24_000
    assert diagnostics["frame_count"] == 64
