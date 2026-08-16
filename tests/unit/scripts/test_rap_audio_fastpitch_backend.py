"""Tests for the NeMo FastPitch explicit-phoneme backend."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import read_chunk_record_index
from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus


ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "rap_audio_protocols" / "two_bar_records.jsonl"
CAMPAIGN_OUTPUT_DIR = ROOT / "output" / "rap_album_10x50_90bpm_20260816_v4"
MODULE_PATH = ROOT / "scripts" / "rap_audio_backends" / "fastpitch_backend.py"
OOV_PHONE_LABELS = {
    "gravity's": ("G", "R", "AE1", "V", "AH0", "T", "IY0", "Z"),
}


def _load_backend(module_name: str = "scripts.rap_audio_backends.fastpitch_backend") -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load module at {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_request():
    corpus = load_song_corpus(FIXTURE_PATH, song_id="01_space_exploration", expected_bars=2)
    return corpus.two_bar_requests()[0]


def _campaign_request(song_id: str, chunk_index: int):
    corpus = load_song_corpus(CAMPAIGN_OUTPUT_DIR / song_id / "chosen_lyrics.jsonl", song_id=song_id)
    return corpus.two_bar_requests()[chunk_index]


def _tokenizer_labels(request, overrides: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    overrides = overrides or {}
    labels: list[str] = ["<blk>"]
    words = list(_word_syllables(request.syllables))
    for word_index, (word, syllables) in enumerate(words):
        if all(syllable.phonemes for syllable in syllables):
            phones = tuple(phone for syllable in syllables for phone in syllable.phonemes)
        else:
            phones = overrides.get(word.lower(), OOV_PHONE_LABELS[word.lower()])
        for phone in phones:
            labels.append(phone)
            labels.append("<blk>")
        if word_index + 1 < len(words):
            labels.append(" ")
    return tuple(labels)


def _word_syllables(syllables) -> Iterable[tuple[str, tuple[object, ...]]]:
    start = 0
    while start < len(syllables):
        word = syllables[start].word
        end = start
        while end + 1 < len(syllables) and syllables[end + 1].word == word:
            end += 1
        yield word, syllables[start : end + 1]
        start = end + 1


class FakeRow:
    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    def tolist(self) -> list[int]:
        return list(self._values)


class FakeTensor:
    def __init__(self, values: list[list[int]] | list[list[float]]) -> None:
        self.values = [list(row) for row in values]
        self.shape = (len(self.values), len(self.values[0]) if self.values else 0)
        self.to_calls: list[dict[str, object]] = []

    def to(self, device=None, dtype=None):  # type: ignore[no-untyped-def]
        self.to_calls.append({"device": device, "dtype": dtype})
        return self

    def __getitem__(self, index: int) -> FakeRow:
        return FakeRow(self.values[index])

    def numpy(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float32)


class FakeTorch:
    def __init__(self) -> None:
        self.long = "long"
        self.float32 = "float32"
        self.tensor_calls: list[dict[str, object]] = []

    def tensor(self, data, *, device=None, dtype=None):  # type: ignore[no-untyped-def]
        tensor = FakeTensor(data)
        self.tensor_calls.append(
            {
                "data": [list(row) for row in data],
                "device": device,
                "dtype": dtype,
                "tensor": tensor,
            }
        )
        return tensor

    def no_grad(self):
        return nullcontext()


class FakeVocab:
    def __init__(self, labels_by_id: dict[int, str]) -> None:
        self._labels_by_id = dict(labels_by_id)
        self.ids_to_tokens_calls: list[list[int]] = []

    def ids_to_tokens(self, ids: list[int]) -> tuple[str, ...]:
        self.ids_to_tokens_calls.append(list(ids))
        return tuple(self._labels_by_id[index] for index in ids)


class FakeFastPitch:
    def __init__(
        self,
        *,
        labels_by_text: dict[str, tuple[str, ...]],
        failures_remaining: int = 0,
        reject_pitch_energy: bool = False,
    ) -> None:
        self.labels_by_text = dict(labels_by_text)
        self.failures_remaining = failures_remaining
        self.reject_pitch_energy = reject_pitch_energy
        self.parse_calls: list[str] = []
        self.calls: list[dict[str, object]] = []
        self.generate_spectrogram_calls = 0
        self.to_calls: list[str] = []
        self.eval_called = False
        self.vocab = FakeVocab({})

    def parse(self, text: str) -> FakeTensor:
        self.parse_calls.append(text)
        labels = self.labels_by_text[text]
        ids = list(range(len(labels)))
        self.vocab = FakeVocab(dict(enumerate(labels)))
        return FakeTensor([ids])

    def generate_spectrogram(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.generate_spectrogram_calls += 1
        raise AssertionError("generate_spectrogram() must not be used for explicit durations")

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **kwargs: object):
        self.calls.append(dict(kwargs))
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise RuntimeError("synthetic FastPitch failure")
        if self.reject_pitch_energy and (kwargs.get("pitch") is not None or kwargs.get("energy") is not None):
            raise TypeError("pitch/energy shape unsupported in this NeMo build")
        spect = FakeTensor([[0.25, 0.0, -0.25]])
        return spect, None, None


class FakeHiFiGan:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.to_calls: list[str] = []
        self.eval_called = False

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def convert_spectrogram_to_audio(self, *, spec: object) -> np.ndarray:
        self.calls.append(spec)
        return np.linspace(-0.25, 0.25, 384, dtype=np.float32).reshape(1, -1)


def test_fastpitch_backend_module_import_is_dependency_light(monkeypatch: pytest.MonkeyPatch) -> None:
    blocked = {"torch", "nemo"}
    real_import = __import__

    def guard(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"heavy import attempted during module import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", guard)

    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_import_light")

    assert module.FASTPITCH_MODEL_ID == "tts_en_fastpitch"
    assert module.HIFIGAN_MODEL_ID == "tts_en_hifigan"


def test_render_pending_requests_uses_parse_ids_to_tokens_duration_tensors_and_direct_forward(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_render")
    request = _fixture_request()
    labels = _tokenizer_labels(request)
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}, reject_pitch_energy=True),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:5",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "render_chunks.jsonl"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        progress_stream=progress,
    )

    assert len(records) == 1
    record = records[0]
    assert record.success is True
    assert record.sample_rate_hz == 22_050
    assert record.attempts == 1
    assert runtime.fastpitch.parse_calls == [request.text]
    assert runtime.fastpitch.vocab.ids_to_tokens_calls == [list(range(len(labels)))]
    assert runtime.fastpitch.generate_spectrogram_calls == 0
    assert len(runtime.fastpitch.calls) == 2
    assert runtime.fastpitch.calls[0]["text"].shape == runtime.fastpitch.calls[0]["durs"].shape
    assert runtime.fastpitch.calls[0] == {
        "text": runtime.fastpitch.calls[0]["text"],
        "durs": runtime.fastpitch.calls[0]["durs"],
        "pitch": None,
        "energy": None,
        "speaker": None,
        "pace": 1.0,
    }
    assert runtime.fastpitch.calls[1]["pitch"] is not None
    assert runtime.fastpitch.calls[1]["energy"] is not None
    assert len(runtime.hifigan.calls) == 1
    assert "prosody_controls=duration_only_version_guard" in progress.getvalue()

    stored = read_chunk_record_index(record_path)[(record.protocol_id, request.song_id, request.chunk_index)]
    assert stored.output_sha256 == record.output_sha256


def test_build_render_plan_uses_tokenizer_derived_oov_recovery_from_timing_api() -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_plan")
    request = _campaign_request("01_space_exploration", 9)
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )

    plan = module.build_render_plan(request=request, runtime=runtime)

    gravity_groups = tuple(
        plan.syllable_phone_groups[index]
        for index, syllable in enumerate(request.syllables)
        if syllable.word == "gravity's"
    )
    assert gravity_groups == (
        ("G", "R", "AE1"),
        ("V", "AH0"),
        ("T", "IY0", "Z"),
    )
    assert plan.duration_tensor.shape == plan.tokens.shape
    assert sum(plan.duration_frames) == round(request.duration_seconds * 22050 / 256)
    assert len(plan.anchor_error_frames) == len(request.syllables)


def test_render_pending_requests_retries_and_logs_silence_after_bounded_failures(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_retry")
    request = _fixture_request()
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}, failures_remaining=3),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:1",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "render_chunks.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        max_attempts=3,
    )

    assert len(records) == 1
    record = records[0]
    assert len(runtime.fastpitch.calls) == 3
    assert record.success is False
    assert record.attempts == 3
    assert record.output_path is not None
    assert record.output_sha256
    assert record.sample_rate_hz == 22_050

    sample_rate_hz, samples = wavfile.read(Path(record.output_path))
    assert sample_rate_hz == 22_050
    assert samples.shape == (module.chunk_frame_count(request),)
    assert np.count_nonzero(samples) == 0

    error_payload = json.loads(str(record.error))
    assert error_payload["message"] == "synthetic FastPitch failure"
    assert error_payload["attempts"] == 3
    assert error_payload["prosody_controls"] == "not_started"
    assert error_payload["tokenizer_labels"]
    assert error_payload["duration_frames"]
    assert error_payload["anchor_error_frames"]


def test_main_uses_lazy_runtime_factory_and_h200_friendly_cli_arguments(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_main")
    request = _fixture_request()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "render_chunks.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")
    created_runtimes = []

    def runtime_factory(*, fastpitch_model_id: str, hifigan_model_id: str, device: str):
        assert fastpitch_model_id == "custom_fastpitch.nemo"
        assert hifigan_model_id == "custom_hifigan.nemo"
        assert device == "cuda:7"
        runtime = module.FastPitchBackendRuntime(
            fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}, reject_pitch_energy=True),
            hifigan=FakeHiFiGan(),
            torch_module=FakeTorch(),
            device=device,
            fastpitch_model_id=fastpitch_model_id,
            hifigan_model_id=hifigan_model_id,
        )
        created_runtimes.append(runtime)
        return runtime

    exit_code = module.main(
        [
            "--requests-jsonl",
            str(request_path),
            "--records-jsonl",
            str(record_path),
            "--output-dir",
            str(tmp_path / "chunks"),
            "--fastpitch-model",
            "custom_fastpitch.nemo",
            "--hifigan-model",
            "custom_hifigan.nemo",
            "--device",
            "cuda:7",
            "--song",
            request.song_id,
        ],
        runtime_factory=runtime_factory,
    )

    assert exit_code == 0
    assert len(created_runtimes) == 1
