"""Tests for the TED-TTS local-duration backend."""

from __future__ import annotations

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
    chunk_record_is_complete,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.contracts import ProtocolId
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


class _H200CompatibilityIndexTTS2:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def infer(self, **kwargs: object) -> None:
        self.calls.append(kwargs)
        method = kwargs["method"]
        if method == "hmm":
            raise ValueError("cannot convert float NaN to integer")
        if method != "max_head":
            raise ValueError(f"unsupported test method: {method}")
        output_path = Path(str(kwargs["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(output_path, 22_050, np.ones(128, dtype=np.int16))


class _ParentCheckingIndexTTS2:
    def infer(self, **kwargs: object) -> None:
        output_path = Path(str(kwargs["output_path"]))
        assert output_path.parent.is_dir()
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


def test_render_request_creates_output_parent_before_inference(tmp_path: Path) -> None:
    backend = _load_backend()
    output_path = tmp_path / "missing" / "parents" / "chunk.wav"

    record = backend.render_request(
        request=_request(),
        ted_model=_ParentCheckingIndexTTS2(),
        reference_wav_path=tmp_path / "reference.wav",
        output_path=output_path,
    )

    assert record.success is True
    assert output_path.is_file()


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
    diagnostic = json.loads(str(record.error))
    assert diagnostic["infer_kwargs"]["method"] == "hmm"

    sample_rate_hz, samples = wavfile.read(Path(record.output_path))
    assert sample_rate_hz == 22_050
    assert samples.shape == (backend.chunk_frame_count(request),)
    assert np.count_nonzero(samples) == 0

    stored = read_chunk_record_index(record_path)[(record.protocol_id, request.song_id, request.chunk_index)]
    assert stored.success is False
    assert stored.output_sha256 == record.output_sha256


@pytest.mark.parametrize("stale_kind", ["missing_wav", "changed_wav", "changed_request"])
def test_render_pending_requests_replaces_incomplete_record_without_duplicate_ledger_rows(
    tmp_path: Path,
    stale_kind: str,
) -> None:
    backend = _load_backend()
    original_request = _request()
    current_request = original_request
    model = _FakeIndexTTS2()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "chunks"
    output_path = output_dir / original_request.song_id / "chunk-000.wav"
    request_path.write_text(json.dumps(original_request.to_payload()) + "\n", encoding="utf-8")

    first_records = backend.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        reference_wav_path=tmp_path / "reference.wav",
        ted_model=model,
    )
    assert len(first_records) == 1

    if stale_kind == "missing_wav":
        output_path.unlink()
    elif stale_kind == "changed_wav":
        wavfile.write(output_path, 22_050, np.full(128, 2, dtype=np.int16))
    else:
        current_request = replace(
            original_request,
            start_bar=original_request.start_bar + 2,
            end_bar=original_request.end_bar + 2,
        )
        request_path.write_text(json.dumps(current_request.to_payload()) + "\n", encoding="utf-8")

    second_records = backend.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        reference_wav_path=tmp_path / "reference.wav",
        ted_model=model,
    )

    assert len(second_records) == 1
    assert len(model.calls) == 2
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 1
    stored = read_chunk_record_index(record_path)[
        (ProtocolId.TED_LOCAL, current_request.song_id, current_request.chunk_index)
    ]
    assert stored.request_sha256 == current_request.sha256
    assert chunk_record_is_complete(
        record_path,
        output_path,
        request=current_request,
        protocol_id=ProtocolId.TED_LOCAL,
    )


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


def test_main_rerenders_prior_failure_and_reports_failure_again(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _request()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")
    created_models: list[_FakeIndexTTS2] = []

    def failing_factory(**_: Path) -> _FakeIndexTTS2:
        model = _FakeIndexTTS2()
        model.failures_remaining = 1
        created_models.append(model)
        return model

    argv = [
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
        "--max-attempts",
        "1",
    ]

    assert backend.main(argv, ted_model_factory=failing_factory) == 1
    assert backend.main(argv, ted_model_factory=failing_factory) == 1
    assert [len(model.calls) for model in created_models] == [1, 1]
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 1


def test_main_h200_max_head_override_recovers_from_hmm_nan_failure_and_records_selection(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backend = _load_backend()
    request = _request()
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")
    created_models: list[_H200CompatibilityIndexTTS2] = []

    def factory(**_: Path) -> _H200CompatibilityIndexTTS2:
        model = _H200CompatibilityIndexTTS2()
        created_models.append(model)
        return model

    common_argv = [
        "--requests-jsonl",
        str(request_path),
        "--reference-wav",
        str(tmp_path / "reference.wav"),
        "--ted-checkout",
        str(tmp_path / "TED-TTS"),
        "--model-dir",
        str(tmp_path / "checkpoints"),
        "--cfg-path",
        str(tmp_path / "checkpoints" / "config.yaml"),
        "--max-attempts",
        "1",
    ]
    hmm_root = tmp_path / "hmm"

    assert backend.main(
        [
            *common_argv,
            "--records-jsonl",
            str(hmm_root / "render_chunks.jsonl"),
            "--output-dir",
            str(hmm_root),
        ],
        ted_model_factory=factory,
    ) == 1

    hmm_record = next(iter(read_chunk_record_index(hmm_root / "render_chunks.jsonl").values()))
    assert "cannot convert float NaN to integer" in str(hmm_record.error)
    assert created_models[0].calls[0]["method"] == "hmm"
    assert "inference_method=hmm" in capsys.readouterr().out

    max_head_root = tmp_path / "max-head"
    assert backend.main(
        [
            *common_argv,
            "--records-jsonl",
            str(max_head_root / "render_chunks.jsonl"),
            "--output-dir",
            str(max_head_root),
            "--inference-method",
            "max_head",
        ],
        ted_model_factory=factory,
    ) == 0

    max_head_record = next(iter(read_chunk_record_index(max_head_root / "render_chunks.jsonl").values()))
    assert max_head_record.success is True
    assert created_models[1].calls[0]["method"] == "max_head"
    assert "inference_method=max_head" in capsys.readouterr().out
    assert json.loads((max_head_root / "inference_config.json").read_text(encoding="utf-8")) == {
        "determinism_note": "use_random=False does not guarantee determinism because TED still samples",
        "duration_mode": "both",
        "generation_settings": {
            "do_sample": True,
            "emo_alpha": 0,
            "length_penalty": 0.0,
            "max_mel_tokens": 850,
            "num_beams": 3,
            "repetition_penalty": 10.0,
            "temperature": 0.8,
            "top_k": 30,
            "top_p": 0.8,
            "use_emo_text": True,
            "use_random": False,
            "verbose": True,
        },
        "method": "max_head",
    }


def test_render_pending_requests_keeps_matching_inference_config_and_rejects_conflicts(tmp_path: Path) -> None:
    backend = _load_backend()
    request = _request()
    model = _FakeIndexTTS2()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "song" / "render_chunks.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")
    kwargs = {
        "request_path": request_path,
        "record_path": record_path,
        "output_dir": tmp_path / "chunks",
        "reference_wav_path": tmp_path / "reference.wav",
        "ted_model": model,
    }

    backend.render_pending_requests(**kwargs)
    config_path = record_path.with_name("inference_config.json")
    original_config = config_path.read_text(encoding="utf-8")

    assert backend.render_pending_requests(**kwargs) == ()
    assert config_path.read_text(encoding="utf-8") == original_config
    with pytest.raises(ValueError, match="inference config conflicts"):
        backend.render_pending_requests(**kwargs, inference_method="max_head")
    assert len(model.calls) == 1


def test_create_ted_model_uses_exact_checkout_and_restores_cached_modules(tmp_path: Path) -> None:
    backend = _load_backend()
    checkout = tmp_path / "requested-checkout"
    package = checkout / "indextts"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "checkout_marker.py").write_text("TAG = 'requested'\n", encoding="utf-8")
    (package / "infer_v2.py").write_text(
        "from indextts.checkout_marker import TAG\n"
        "class IndexTTS2:\n"
        "    def __init__(self, *, model_dir, cfg_path, is_fp16):\n"
        "        self.tag = TAG\n"
        "        self.model_dir = model_dir\n"
        "        self.cfg_path = cfg_path\n"
        "        self.is_fp16 = is_fp16\n",
        encoding="utf-8",
    )

    cached_package = ModuleType("indextts")
    cached_package.__path__ = [str(tmp_path / "cached-checkout" / "indextts")]  # type: ignore[attr-defined]
    cached_infer = ModuleType("indextts.infer_v2")

    class CachedIndexTTS2:
        tag = "cached"

    cached_infer.IndexTTS2 = CachedIndexTTS2  # type: ignore[attr-defined]
    original_path = list(sys.path)
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "indextts" or name.startswith("indextts.")
    }
    try:
        for name in tuple(original_modules):
            sys.modules.pop(name, None)
        sys.modules["indextts"] = cached_package
        sys.modules["indextts.infer_v2"] = cached_infer

        model = backend.create_ted_model(
            model_dir=tmp_path / "model",
            cfg_path=tmp_path / "model" / "config.yaml",
            ted_checkout=checkout,
        )

        assert model.tag == "requested"
        assert model.model_dir == str(tmp_path / "model")
        assert model.cfg_path == str(tmp_path / "model" / "config.yaml")
        assert model.is_fp16 is True
        assert sys.path == original_path
        assert sys.modules["indextts"] is cached_package
        assert sys.modules["indextts.infer_v2"] is cached_infer
        assert "indextts.checkout_marker" not in sys.modules
    finally:
        for name in tuple(sys.modules):
            if name == "indextts" or name.startswith("indextts."):
                sys.modules.pop(name, None)
        sys.modules.update(original_modules)
        sys.path[:] = original_path
