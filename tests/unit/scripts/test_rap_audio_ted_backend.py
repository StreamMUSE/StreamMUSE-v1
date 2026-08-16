"""Tests for the TED-TTS local-duration backend."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import read_chunk_record_index
from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus
from streammuse.experiments.rap_audio_protocols.timing import TimedTextSegment, build_ted_segments


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "rap_audio_protocols" / "two_bar_records.jsonl"


def _load_backend() -> ModuleType:
    module_name = "_streammuse_test_script_rap_audio_ted_backend"
    script_path = ROOT / "scripts" / "rap_audio_backends" / "ted_backend.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _request():
    corpus = load_song_corpus(FIXTURE_PATH, song_id="01_space_exploration", expected_bars=2)
    return corpus.two_bar_requests()[0]


class _FakeIndexTTS2:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.failures_remaining = 0

    def infer(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("simulated TED failure")
        output_path = Path(str(kwargs["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, 22_050, np.ones(128, dtype=np.int16))


def test_render_pending_requests_invokes_ted_with_exact_local_duration_arguments(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _request()
    model = _FakeIndexTTS2()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = backend.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=tmp_path / "chunks",
        reference_wav_path=tmp_path / "reference.wav",
        ted_model=model,
    )

    assert len(records) == 1
    assert len(model.calls) == 1
    expected_segments = build_ted_segments(request)
    expected_text = "|".join(segment.text_with_spacing for segment in expected_segments)
    expected_emotions = "|".join(backend.TED_SEGMENT_DESCRIPTION for _ in expected_segments)
    expected_duration_tokens = [int(segment.target_seconds / 0.02) for segment in expected_segments]
    call = model.calls[0]
    assert call == {
        "spk_audio_prompt": str(tmp_path / "reference.wav"),
        "text": expected_text,
        "output_path": str(tmp_path / "chunks" / "01_space_exploration" / "chunk-000.wav"),
        "emo_audio_prompt": None,
        "emo_alpha": 0,
        "use_emo_text": True,
        "emo_text": expected_emotions,
        "target_duration_tokens": expected_duration_tokens,
        "duration_mode": "both",
        "verbose": True,
        "use_random": False,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "length_penalty": 0.0,
        "max_mel_tokens": 850,
        "method": "hmm",
    }
    stored = read_chunk_record_index(record_path)[(records[0].protocol_id, request.song_id, request.chunk_index)]
    assert stored.success is True
    assert stored.sample_rate_hz == 22_050
    assert stored.attempts == 1
    assert stored.output_sha256


def test_render_request_rejects_segment_count_mismatch_before_model_invocation(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _request()
    model = _FakeIndexTTS2()

    with pytest.raises(ValueError, match="segment counts must match"):
        backend.render_request(
            request=request,
            ted_model=model,
            reference_wav_path=tmp_path / "reference.wav",
            output_path=tmp_path / "chunk.wav",
            segment_builder=lambda _: (
                TimedTextSegment(text_with_spacing="first|second", target_seconds=0.40),
            ),
        )

    assert model.calls == []


def test_render_pending_requests_retries_and_logs_silence_after_bounded_failures(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _request()
    model = _FakeIndexTTS2()
    model.failures_remaining = 3
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = backend.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=tmp_path / "chunks",
        reference_wav_path=tmp_path / "reference.wav",
        ted_model=model,
        max_attempts=3,
    )

    assert len(model.calls) == 3
    assert len(records) == 1
    record = records[0]
    assert record.success is False
    assert record.attempts == 3
    assert "simulated TED failure" in str(record.error)
    assert record.output_path is not None
    assert record.output_sha256
    assert record.sample_rate_hz == 22_050

    sample_rate_hz, samples = wavfile.read(Path(record.output_path))
    assert sample_rate_hz == 22_050
    assert samples.shape == (backend.chunk_frame_count(request),)
    assert np.count_nonzero(samples) == 0

    stored = read_chunk_record_index(record_path)[(record.protocol_id, request.song_id, request.chunk_index)]
    assert stored.success is False
    assert stored.output_sha256 == record.output_sha256


def test_main_uses_explicit_paths_and_fake_indextts2_factory(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _request()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    created_models: list[_FakeIndexTTS2] = []

    def factory(*, model_dir: Path, cfg_path: Path, ted_checkout: Path):
        assert model_dir == tmp_path / "checkpoints"
        assert cfg_path == tmp_path / "checkpoints" / "config.yaml"
        assert ted_checkout == tmp_path / "TED-TTS"
        model = _FakeIndexTTS2()
        created_models.append(model)
        return model

    exit_code = backend.main(
        [
            "--requests-jsonl",
            str(request_path),
            "--records-jsonl",
            str(record_path),
            "--output-dir",
            str(tmp_path / "chunks"),
            "--reference-wav",
            str(tmp_path / "reference.wav"),
            "--ted-checkout",
            str(tmp_path / "TED-TTS"),
            "--model-dir",
            str(tmp_path / "checkpoints"),
            "--cfg-path",
            str(tmp_path / "checkpoints" / "config.yaml"),
        ],
        ted_model_factory=factory,
    )

    assert exit_code == 0
    assert len(created_models) == 1
    assert len(created_models[0].calls) == 1
