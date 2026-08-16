from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import ANY

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import read_chunk_record_index
from streammuse.experiments.rap_audio_protocols.contracts import ProtocolId, SyllableTarget, TwoBarRenderRequest


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "rap_audio_backends" / "moss_backend.py"


def _load_module(module_name: str = "scripts.rap_audio_backends.moss_backend"):
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module at {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _request(chunk_index: int = 0) -> TwoBarRenderRequest:
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
        song_id="01_space_exploration",
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


class FakeAudioTensor:
    def __init__(self, samples: np.ndarray) -> None:
        self.samples = np.asarray(samples, dtype=np.float32)

    def unsqueeze(self, axis: int) -> np.ndarray:
        return np.expand_dims(self.samples, axis)


class FakeDecodedMessage:
    def __init__(self, samples: np.ndarray) -> None:
        self.audio_codes_list = [FakeAudioTensor(samples)]


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

    def is_available(self) -> bool:
        return self.available

    def manual_seed_all(self, seed: int) -> None:
        self.manual_seed_all_calls.append(seed)

    def get_device_capability(self):
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


class FakeTorchaudio:
    def __init__(self) -> None:
        self.save_calls: list[tuple[str, int]] = []

    def save(self, path: str, audio, sample_rate: int) -> None:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 2:
            array = np.swapaxes(array, 0, 1)
        wavfile.write(path, sample_rate, array)
        self.save_calls.append((path, sample_rate))


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
