from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import ANY

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import (
    append_chunk_record,
    chunk_record_is_complete,
    file_sha256,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.contracts import (
    ChunkRenderRecord,
    ProtocolId,
    SyllableTarget,
    TwoBarRenderRequest,
)


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "rap_audio_backends" / "moss_backend.py"


def _load_module(module_name: str = "scripts.rap_audio_backends.moss_backend"):
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module at {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _request(chunk_index: int = 0, *, song_id: str = "01_space_exploration") -> TwoBarRenderRequest:
    start_bar = chunk_index * 2
    syllables = tuple(
        SyllableTarget(
            word=f"word{index}",
            index_in_word=0,
            phonemes=("W", "ER1", "D"),
            lexical_stress=1 if index % 3 == 0 else 0,
            target_stress=1.0 if index % 4 == 0 else 0.5,
            boundary_strength=2 if index in {8, 17} else 0,
            absolute_tick=start_bar * 16 + index,
            tick_in_chunk=index,
            target_seconds=(index + 1) / 3.0,
        )
        for index in range(18)
    )
    return TwoBarRenderRequest(
        song_id=song_id,
        chunk_index=chunk_index,
        start_bar=start_bar,
        end_bar=start_bar + 2,
        text="clear words lock to the pocket with precise downbeats",
        syllables=syllables,
    )


class FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.to_calls: list[str] = []

    def to(self, device: str):
        self.to_calls.append(device)
        return self


class FakeDecodedMessage:
    def __init__(self, audio) -> None:
        self.audio_codes_list = [audio]


class FakeAudioTokenizer:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str):
        self.devices.append(device)
        return self


class FakeProcessor:
    def __init__(self, *, sample_rate_hz: int = 24_000) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.model_config = type("Config", (), {"sampling_rate": sample_rate_hz})()
        self.audio_tokenizer = FakeAudioTokenizer()
        self.build_user_message_calls: list[dict[str, object]] = []
        self.call_modes: list[str] = []
        self.conversations: list[list[dict[str, object]]] = []

    def build_user_message(self, **kwargs):
        self.build_user_message_calls.append(dict(kwargs))
        return {"role": "user", **kwargs}

    def __call__(self, conversations, *, mode: str):
        self.call_modes.append(mode)
        self.conversations.append(conversations)
        return {
            "input_ids": FakeTensor("input_ids"),
            "attention_mask": FakeTensor("attention_mask"),
        }

    def decode(self, outputs):
        return [FakeDecodedMessage(np.linspace(-0.2, 0.2, 48, dtype=np.float32))]


class FakeModel:
    def __init__(self, *, failures_before_success: int = 0) -> None:
        self.failures_before_success = failures_before_success
        self.generate_calls: list[dict[str, object]] = []
        self.to_device: str | None = None
        self.eval_called = False

    def to(self, device: str):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(dict(kwargs))
        if len(self.generate_calls) <= self.failures_before_success:
            raise RuntimeError(f"synthetic generate failure {len(self.generate_calls)}")
        return {"audio_tokens": [1, 2, 3]}


class FakeCuda:
    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.manual_seed_all_calls: list[int] = []
        self.cudnn_sdp_flags: list[bool] = []
        self.flash_sdp_flags: list[bool] = []
        self.mem_efficient_sdp_flags: list[bool] = []
        self.math_sdp_flags: list[bool] = []
        self.device_capability_calls: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)

    def get_device_capability(self, device: str):
        self.device_capability_calls.append(device)
        return (9, 0)

    def enable_cudnn_sdp(self, flag: bool) -> None:
        self.cudnn_sdp_flags.append(flag)

    def enable_flash_sdp(self, flag: bool) -> None:
        self.flash_sdp_flags.append(flag)

    def enable_mem_efficient_sdp(self, flag: bool) -> None:
        self.mem_efficient_sdp_flags.append(flag)

    def enable_math_sdp(self, flag: bool) -> None:
        self.math_sdp_flags.append(flag)


class FakeBackends:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


class FakeTorch:
    def __init__(self, *, cuda_available: bool = True) -> None:
        self.cuda = FakeCuda(available=cuda_available)
        self.backends = FakeBackends(self.cuda)
        self.manual_seed_calls: list[int] = []
        self.bfloat16 = "bfloat16"
        self.float32 = "float32"
        self.float16 = "float16"

    def manual_seed(self, seed: int) -> None:
        self.manual_seed_calls.append(seed)

    def as_tensor(self, value, *, dtype=None):
        return FakeTorchTensor(value)


class FakeTorchTensor:
    def __init__(self, samples) -> None:
        self.samples = np.asarray(samples, dtype=np.float32)

    @property
    def ndim(self) -> int:
        return self.samples.ndim

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def unsqueeze(self, axis: int):
        return FakeTorchTensor(np.expand_dims(self.samples, axis))

    def numpy(self) -> np.ndarray:
        return self.samples


class FakeTorchaudio:
    def __init__(self) -> None:
        self.save_calls: list[tuple[str, int]] = []

    def save(self, path: str, audio, sample_rate: int) -> None:
        array = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio, dtype=np.float32)
        if array.ndim == 2:
            array = np.swapaxes(array, 0, 1)
        wavfile.write(path, sample_rate, array)
        self.save_calls.append((path, sample_rate))


def _runtime(module, *, model=None, torch_module=None, torchaudio_module=None, device: str = "cuda:0"):
    active_torch = torch_module if torch_module is not None else FakeTorch()
    return module.MossBackendRuntime(
        processor=FakeProcessor(),
        model=model if model is not None else FakeModel(),
        torch_module=active_torch,
        torchaudio_module=torchaudio_module if torchaudio_module is not None else FakeTorchaudio(),
        device=device,
        model_id=module.MODEL_ID,
        attn_implementation="sdpa",
        dtype=active_torch.bfloat16 if device.startswith("cuda") else active_torch.float32,
    )


def test_moss_backend_module_import_is_dependency_light(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = {"torch", "torchaudio", "transformers"}
    real_import = __import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"heavy import attempted during module import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guard)

    module = _load_module("scripts.rap_audio_backends.moss_backend_import_light")

    assert module.MODEL_ID == "OpenMOSS-Team/MOSS-TTS-v1.5"


def test_render_requests_uses_documented_prompt_generation_args_and_manifest(tmp_path: Path) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_prompt")
    request = _request(0)
    processor = FakeProcessor()
    model = FakeModel()
    torch_module = FakeTorch()
    torchaudio_module = FakeTorchaudio()
    reference_wav = tmp_path / "reference.wav"
    manifest_path = tmp_path / "campaign_manifest.json"
    wavfile.write(reference_wav, 24_000, np.zeros(32, dtype=np.float32))

    runtime = module.MossBackendRuntime(
        processor=processor,
        model=model,
        torch_module=torch_module,
        torchaudio_module=torchaudio_module,
        device="cuda:3",
        model_id=module.MODEL_ID,
        attn_implementation="flash_attention_2",
        dtype=torch_module.bfloat16,
    )

    records = module.render_requests(
        requests=(request,),
        output_dir=tmp_path / "chunks",
        record_path=tmp_path / "render_chunks.jsonl",
        reference_wav=reference_wav,
        runtime=runtime,
        campaign_manifest_path=manifest_path,
    )

    assert len(records) == 1
    assert processor.build_user_message_calls == [
        {
            "text": request.text,
            "language": "English",
            "reference": [str(reference_wav)],
            "tokens": 67,
            "instruction": "clear, rhythmically spoken rap with restrained pitch",
        }
    ]
    assert processor.call_modes == ["generation"]
    assert model.generate_calls == [
        {
            "input_ids": ANY,
            "attention_mask": ANY,
            "max_new_tokens": 256,
            "audio_temperature": 1.7,
            "audio_top_p": 0.8,
            "audio_top_k": 25,
            "audio_repetition_penalty": 1.0,
        }
    ]
    assert records[0].protocol_id is ProtocolId.MOSS_GLOBAL
    assert records[0].request_sha256 == request.sha256
    assert records[0].sample_rate_hz == 24_000
    assert records[0].attempts == 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["protocol_id"] == "moss_global"
    assert "instruction" in manifest["style_instruction_caveat"]
    assert "unsupported" in manifest["deterministic_seed_caveat"]
    record_index = read_chunk_record_index(tmp_path / "render_chunks.jsonl")
    assert record_index[(ProtocolId.MOSS_GLOBAL, request.song_id, request.chunk_index)].output_sha256 == records[0].output_sha256


def test_render_requests_persists_fingerprinted_default_config_before_generation(tmp_path: Path) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_config_before_render")
    request = _request(0)
    ledger_path = tmp_path / "render_chunks.jsonl"
    config_path = tmp_path / "campaign_manifest.json"

    class ConfigCheckingModel(FakeModel):
        def generate(self, **kwargs):
            assert config_path.is_file()
            return super().generate(**kwargs)

    module.render_requests(
        requests=(request,),
        output_dir=tmp_path / "chunks",
        record_path=ledger_path,
        reference_wav=tmp_path / "reference.wav",
        runtime=_runtime(module, model=ConfigCheckingModel()),
    )

    manifest = json.loads(config_path.read_text(encoding="utf-8"))
    fingerprint = manifest.pop("configuration_sha256")
    assert manifest["schema_version"] == 1
    assert fingerprint == hashlib.sha256(module.canonical_json_dumps(manifest).encode("utf-8")).hexdigest()


@pytest.mark.parametrize("changed_config", ["model_id", "reference_wav", "generation_kwargs"])
def test_render_requests_refuses_changed_resume_config_without_overwriting_manifest(
    tmp_path: Path,
    changed_config: str,
) -> None:
    module = _load_module(f"scripts.rap_audio_backends.moss_backend_config_conflict_{changed_config}")
    request = _request(0)
    ledger_path = tmp_path / "render_chunks.jsonl"
    manifest_path = tmp_path / "campaign_manifest.json"
    reference_wav = tmp_path / "reference.wav"

    module.render_requests(
        requests=(request,),
        output_dir=tmp_path / "chunks",
        record_path=ledger_path,
        reference_wav=reference_wav,
        runtime=_runtime(module),
        campaign_manifest_path=manifest_path,
    )
    original_manifest = manifest_path.read_text(encoding="utf-8")
    original_ledger = ledger_path.read_text(encoding="utf-8")
    resumed_model = FakeModel()
    resumed_runtime = _runtime(module, model=resumed_model)
    resumed_reference = reference_wav
    if changed_config == "model_id":
        resumed_runtime = replace(resumed_runtime, model_id="OpenMOSS-Team/MOSS-TTS-v2")
    elif changed_config == "reference_wav":
        resumed_reference = tmp_path / "different-reference.wav"
    else:
        module.GENERATION_KWARGS = {**module.GENERATION_KWARGS, "audio_top_k": 26}

    with pytest.raises(ValueError, match="campaign manifest conflicts"):
        module.render_requests(
            requests=(request,),
            output_dir=tmp_path / "chunks",
            record_path=ledger_path,
            reference_wav=resumed_reference,
            runtime=resumed_runtime,
            campaign_manifest_path=manifest_path,
        )

    assert resumed_model.generate_calls == []
    assert manifest_path.read_text(encoding="utf-8") == original_manifest
    assert ledger_path.read_text(encoding="utf-8") == original_ledger


def test_render_requests_upgrades_matching_legacy_manifest_before_resume(tmp_path: Path) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_legacy_manifest")
    request = _request(0)
    ledger_path = tmp_path / "render_chunks.jsonl"
    manifest_path = tmp_path / "campaign_manifest.json"
    runtime = _runtime(module)

    first = module.render_requests(
        requests=(request,),
        output_dir=tmp_path / "chunks",
        record_path=ledger_path,
        reference_wav=tmp_path / "reference.wav",
        runtime=runtime,
        campaign_manifest_path=manifest_path,
    )
    legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy_manifest.pop("configuration_sha256")
    legacy_manifest.pop("schema_version")
    manifest_path.write_text(json.dumps(legacy_manifest, indent=2, sort_keys=True), encoding="utf-8")
    resumed_model = FakeModel()

    resumed = module.render_requests(
        requests=(request,),
        output_dir=tmp_path / "chunks",
        record_path=ledger_path,
        reference_wav=tmp_path / "reference.wav",
        runtime=replace(runtime, model=resumed_model),
        campaign_manifest_path=manifest_path,
    )

    migrated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed == first
    assert resumed_model.generate_calls == []
    assert migrated["schema_version"] == 1
    assert "configuration_sha256" in migrated


def test_render_requests_retries_with_same_identity_and_best_effort_seeding(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_retry")
    request = _request(0)
    processor = FakeProcessor()
    model = FakeModel(failures_before_success=1)
    torch_module = FakeTorch()
    torchaudio_module = FakeTorchaudio()
    reference_wav = tmp_path / "reference.wav"
    wavfile.write(reference_wav, 24_000, np.zeros(32, dtype=np.float32))

    runtime = module.MossBackendRuntime(
        processor=processor,
        model=model,
        torch_module=torch_module,
        torchaudio_module=torchaudio_module,
        device="cuda:0",
        model_id=module.MODEL_ID,
        attn_implementation="sdpa",
        dtype=torch_module.bfloat16,
    )

    records = module.render_requests(
        requests=(request,),
        output_dir=tmp_path / "chunks",
        record_path=tmp_path / "render_chunks.jsonl",
        reference_wav=reference_wav,
        runtime=runtime,
        max_retries=2,
        base_seed=9000,
    )

    assert [call["text"] for call in processor.build_user_message_calls] == [request.text, request.text]
    assert [call["tokens"] for call in processor.build_user_message_calls] == [67, 67]
    assert torch_module.manual_seed_calls == [9000, 9001]
    assert torch_module.cuda.manual_seed_all_calls == [9000, 9001]
    assert records[0].success is True
    assert records[0].attempts == 2
    output = capsys.readouterr().out.lower()
    assert "official deterministic seeding is unsupported" in output
    assert "attempt=2/2" in output


def test_render_requests_records_exact_length_silence_and_error_after_bounded_failures(tmp_path: Path) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_failure_silence")
    request = _request(0)
    output_path = tmp_path / "chunks" / request.song_id / "chunk-000.wav"

    records = module.render_requests(
        requests=(request,),
        output_dir=tmp_path / "chunks",
        record_path=tmp_path / "render_chunks.jsonl",
        reference_wav=tmp_path / "reference.wav",
        runtime=_runtime(module, model=FakeModel(failures_before_success=1)),
        max_retries=1,
    )

    record = records[0]
    sample_rate_hz, samples = wavfile.read(output_path)
    assert record.success is False
    assert record.output_path == str(output_path)
    assert record.output_sha256 == file_sha256(output_path)
    assert record.sample_rate_hz == 24_000
    assert record.error == "synthetic generate failure 1"
    assert sample_rate_hz == 24_000
    assert samples.shape == (round(request.duration_seconds * 24_000),)
    assert np.count_nonzero(samples) == 0


@pytest.mark.parametrize("stale_kind", ["failed", "missing_wav", "request_mismatch", "hash_mismatch"])
def test_render_requests_replaces_incomplete_record_without_losing_other_rows(tmp_path: Path, stale_kind: str) -> None:
    module = _load_module(f"scripts.rap_audio_backends.moss_backend_resume_{stale_kind}")
    request = _request(0)
    other_request = _request(1)
    output_root = tmp_path / "chunks"
    output_path = output_root / request.song_id / "chunk-000.wav"
    ledger_path = tmp_path / "render_chunks.jsonl"

    if stale_kind in {"request_mismatch", "hash_mismatch"}:
        output_path.parent.mkdir(parents=True)
        wavfile.write(output_path, 24_000, np.linspace(-0.1, 0.1, 24, dtype=np.float32))

    stale_record = ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256="0" * 64 if stale_kind == "request_mismatch" else request.sha256,
        success=stale_kind != "failed",
        output_path=str(output_path) if stale_kind != "failed" else None,
        output_sha256="0" * 64 if stale_kind == "missing_wav" else None,
        sample_rate_hz=24_000,
        attempts=1,
        error="old failure" if stale_kind == "failed" else None,
    )
    append_chunk_record(ledger_path, stale_record)
    if stale_kind == "hash_mismatch":
        wavfile.write(output_path, 24_000, np.linspace(-0.2, 0.2, 24, dtype=np.float32))

    other_record = ChunkRenderRecord(
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=other_request.song_id,
        chunk_index=other_request.chunk_index,
        request_sha256=other_request.sha256,
        success=False,
        sample_rate_hz=24_000,
        attempts=2,
        error="unrelated failure",
    )
    append_chunk_record(ledger_path, other_record)
    model = FakeModel()

    records = module.render_requests(
        requests=(request,),
        output_dir=output_root,
        record_path=ledger_path,
        reference_wav=tmp_path / "reference.wav",
        runtime=_runtime(module, model=model),
        max_retries=1,
    )

    index = read_chunk_record_index(ledger_path)
    key = (ProtocolId.MOSS_GLOBAL, request.song_id, request.chunk_index)
    other_key = (ProtocolId.MOSS_GLOBAL, other_request.song_id, other_request.chunk_index)
    assert len(model.generate_calls) == 1
    assert records[0] == index[key]
    assert records[0].success is True
    assert index[other_key] == other_record
    assert len([line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]) == 2
    assert chunk_record_is_complete(
        ledger_path,
        output_path,
        request=request,
        protocol_id=ProtocolId.MOSS_GLOBAL,
    )


def test_render_requests_skips_only_a_complete_chunk_record(tmp_path: Path) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_complete_resume")
    request = _request(0)
    output_root = tmp_path / "chunks"
    ledger_path = tmp_path / "render_chunks.jsonl"
    first_model = FakeModel()
    second_model = FakeModel()

    first = module.render_requests(
        requests=(request,),
        output_dir=output_root,
        record_path=ledger_path,
        reference_wav=tmp_path / "reference.wav",
        runtime=_runtime(module, model=first_model),
    )
    resumed = module.render_requests(
        requests=(request,),
        output_dir=output_root,
        record_path=ledger_path,
        reference_wav=tmp_path / "reference.wav",
        runtime=_runtime(module, model=second_model),
    )

    assert len(first_model.generate_calls) == 1
    assert second_model.generate_calls == []
    assert resumed == first


def test_render_requests_namespaces_same_chunk_index_by_song(tmp_path: Path) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_song_paths")
    requests = (
        _request(0, song_id="01_space_exploration"),
        _request(0, song_id="02_pressure_lines"),
    )
    output_root = tmp_path / "chunks"

    records = module.render_requests(
        requests=requests,
        output_dir=output_root,
        record_path=tmp_path / "render_chunks.jsonl",
        reference_wav=tmp_path / "reference.wav",
        runtime=_runtime(module),
    )

    assert [Path(record.output_path or "") for record in records] == [
        output_root / "01_space_exploration" / "chunk-000.wav",
        output_root / "02_pressure_lines" / "chunk-000.wav",
    ]
    assert all(Path(record.output_path or "").is_file() for record in records)


def test_attention_capability_query_uses_requested_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_attention_device")
    torch_module = FakeTorch()
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())

    implementation = module._resolve_attn_implementation(
        torch_module,
        device="cuda:3",
        dtype=torch_module.bfloat16,
    )

    assert implementation == "flash_attention_2"
    assert torch_module.cuda.device_capability_calls == ["cuda:3"]


def test_render_requests_passes_real_torch_tensor_to_save_and_records_valid_wav(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    module = _load_module("scripts.rap_audio_backends.moss_backend_real_tensor")

    class TensorOnlyTorchaudio:
        def __init__(self) -> None:
            self.waveforms = []

        def save(self, path: str, waveform, sample_rate: int) -> None:
            if not isinstance(waveform, torch.Tensor):
                raise TypeError("torchaudio.save requires a torch.Tensor")
            destination = Path(path)
            assert destination.parent.is_dir()
            assert waveform.dtype is torch.float32
            assert tuple(waveform.shape) == (1, 48)
            wavfile.write(destination, sample_rate, waveform.squeeze(0).numpy())
            self.waveforms.append(waveform)

    request = _request(0)
    output_root = tmp_path / "deep" / "chunks"
    ledger_path = tmp_path / "records" / "render_chunks.jsonl"
    torchaudio_module = TensorOnlyTorchaudio()
    runtime = module.MossBackendRuntime(
        processor=FakeProcessor(),
        model=FakeModel(),
        torch_module=torch,
        torchaudio_module=torchaudio_module,
        device="cpu",
        model_id=module.MODEL_ID,
        attn_implementation="eager",
        dtype=torch.float32,
    )

    records = module.render_requests(
        requests=(request,),
        output_dir=output_root,
        record_path=ledger_path,
        reference_wav=tmp_path / "reference.wav",
        runtime=runtime,
        max_retries=1,
    )

    output_path = output_root / request.song_id / "chunk-000.wav"
    sample_rate_hz, samples = wavfile.read(output_path)
    assert len(torchaudio_module.waveforms) == 1
    assert sample_rate_hz == 24_000
    assert samples.ndim == 1
    assert records[0].output_sha256 == file_sha256(output_path)
    assert chunk_record_is_complete(
        ledger_path,
        output_path,
        request=request,
        protocol_id=ProtocolId.MOSS_GLOBAL,
    )


def test_render_requests_rejects_non_mono_saved_wav(tmp_path: Path) -> None:
    module = _load_module("scripts.rap_audio_backends.moss_backend_stereo_validation")

    class StereoTorchaudio:
        def save(self, path: str, waveform, sample_rate: int) -> None:
            wavfile.write(path, sample_rate, np.zeros((48, 2), dtype=np.float32))

    records = module.render_requests(
        requests=(_request(0),),
        output_dir=tmp_path / "chunks",
        record_path=tmp_path / "render_chunks.jsonl",
        reference_wav=tmp_path / "reference.wav",
        runtime=_runtime(module, torchaudio_module=StereoTorchaudio()),
        max_retries=1,
    )

    assert records[0].success is False
    assert "channel mismatch" in (records[0].error or "")
