"""Tests for the NeMo FastPitch explicit-phoneme backend."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import (
    append_chunk_record,
    chunk_record_is_complete,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId
from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus
from streammuse.experiments.rap_audio_protocols.timing import build_fastpitch_phone_plan


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
        if word.lower() in overrides:
            phones = overrides[word.lower()]
        elif all(syllable.phonemes for syllable in syllables):
            phones = tuple(phone for syllable in syllables for phone in syllable.phonemes)
        else:
            phones = OOV_PHONE_LABELS[word.lower()]
        for phone in phones:
            labels.append(phone)
            labels.append("<blk>")
        if word_index + 1 < len(words):
            labels.append(" ")
    return tuple(labels)


def _request_with_replaced_word(request, old_word: str, new_word: str, phoneme_groups: tuple[tuple[str, ...], ...]):
    matching_indices = [index for index, syllable in enumerate(request.syllables) if syllable.word == old_word]
    assert len(matching_indices) == len(phoneme_groups)
    syllables = list(request.syllables)
    for index, phonemes in zip(matching_indices, phoneme_groups):
        syllables[index] = replace(syllables[index], word=new_word, phonemes=phonemes)
    text_start = request.text.lower().index(old_word.lower())
    text_end = text_start + len(old_word)
    return replace(
        request,
        text=request.text[:text_start] + new_word + request.text[text_end:],
        syllables=tuple(syllables),
    )


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


class FakeNeMo27Vocab:
    """Matches the NeMo 2.7.3 EnglishPhonemesTokenizer vocabulary surface."""

    def __init__(self, labels_by_id: dict[int, str]) -> None:
        self.tokens = tuple(labels_by_id[index] for index in sorted(labels_by_id))
        self._id2token = dict(labels_by_id)


class FakeFastPitch:
    def __init__(
        self,
        *,
        labels_by_text: dict[str, tuple[str, ...]],
        failures_remaining: int = 0,
        reject_pitch_energy: bool = False,
        vocab_factory: Callable[[dict[int, str]], object] = FakeVocab,
    ) -> None:
        self.labels_by_text = dict(labels_by_text)
        self.failures_remaining = failures_remaining
        self.reject_pitch_energy = reject_pitch_energy
        self.parse_calls: list[str] = []
        self.calls: list[dict[str, object]] = []
        self.generate_spectrogram_calls = 0
        self.to_calls: list[str] = []
        self.eval_called = False
        self.vocab_factory = vocab_factory
        self.vocab = vocab_factory({})

    def parse(self, text: str) -> FakeTensor:
        self.parse_calls.append(text)
        labels = self.labels_by_text[text]
        ids = list(range(len(labels)))
        self.vocab = self.vocab_factory(dict(enumerate(labels)))
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
    def __init__(self, *, sample_rate: int = 22_050) -> None:
        self.sample_rate = sample_rate
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


def test_default_render_uses_parse_ids_to_tokens_and_duration_only_direct_forward(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_render")
    request = _fixture_request()
    labels = _tokenizer_labels(request)
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}),
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
    assert len(runtime.fastpitch.calls) == 1
    assert runtime.fastpitch.calls[0]["text"].shape == runtime.fastpitch.calls[0]["durs"].shape
    assert runtime.fastpitch.calls[0] == {
        "text": runtime.fastpitch.calls[0]["text"],
        "durs": runtime.fastpitch.calls[0]["durs"],
        "pitch": None,
        "energy": None,
        "speaker": None,
        "pace": 1.0,
    }
    assert len(runtime.hifigan.calls) == 1
    assert "prosody_controls=duration_only_default" in progress.getvalue()

    stored = read_chunk_record_index(record_path)[(record.protocol_id, request.song_id, request.chunk_index)]
    assert stored.output_sha256 == record.output_sha256


def test_smoke_validated_prosody_opt_in_preserves_duration_only_version_guard(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_prosody_guard")
    request = _fixture_request()
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}, reject_pitch_energy=True),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        prosody_mode="smoke-validated",
        progress_stream=progress,
    )

    assert records[0].success is True
    assert len(runtime.fastpitch.calls) == 2
    assert runtime.fastpitch.calls[1]["pitch"] is not None
    assert runtime.fastpitch.calls[1]["energy"] is not None
    assert "prosody_controls=duration_only_version_guard" in progress.getvalue()


@pytest.mark.parametrize("stale_kind", ["failed", "missing_wav", "request_mismatch", "hash_mismatch"])
def test_render_pending_requests_replaces_incomplete_record_without_losing_other_rows(
    tmp_path: Path,
    stale_kind: str,
) -> None:
    module = _load_backend(f"scripts.rap_audio_backends.fastpitch_backend_resume_{stale_kind}")
    original_request = _fixture_request()
    current_request = original_request
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "chunks"
    output_path = output_dir / original_request.song_id / "chunk-000.wav"
    request_path.write_text(json.dumps(original_request.to_payload()) + "\n", encoding="utf-8")
    first_runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={original_request.text: _tokenizer_labels(original_request)}),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    first = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=first_runtime,
    )
    primary = first[0]
    other = ChunkRenderRecord(
        protocol_id=ProtocolId.FASTPITCH_PHONEME,
        song_id="unrelated_song",
        chunk_index=7,
        request_sha256="1" * 64,
        success=False,
        sample_rate_hz=22_050,
        attempts=2,
        error="unrelated failure",
    )
    append_chunk_record(record_path, other)

    if stale_kind == "failed":
        failed = replace(primary, success=False, error="old failure")
        record_path.write_text(
            "\n".join(json.dumps(item.to_payload()) for item in (failed, other)) + "\n",
            encoding="utf-8",
        )
    elif stale_kind == "missing_wav":
        output_path.unlink()
    elif stale_kind == "hash_mismatch":
        wavfile.write(output_path, 22_050, np.full(64, 0.5, dtype=np.float32))
    else:
        current_request = replace(
            original_request,
            start_bar=original_request.start_bar + 2,
            end_bar=original_request.end_bar + 2,
        )
        request_path.write_text(json.dumps(current_request.to_payload()) + "\n", encoding="utf-8")

    second_runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={current_request.text: _tokenizer_labels(current_request)}),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    rerendered = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=second_runtime,
    )

    index = read_chunk_record_index(record_path)
    key = (ProtocolId.FASTPITCH_PHONEME, current_request.song_id, current_request.chunk_index)
    other_key = (other.protocol_id, other.song_id, other.chunk_index)
    assert len(second_runtime.fastpitch.calls) == 1
    assert rerendered == (index[key],)
    assert index[other_key] == other
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 2
    assert chunk_record_is_complete(
        record_path,
        output_path,
        request=current_request,
        protocol_id=ProtocolId.FASTPITCH_PHONEME,
    )


def test_render_pending_requests_skips_only_complete_matching_record(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_complete_resume")
    request = _fixture_request()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "chunks"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    def make_runtime():
        return module.FastPitchBackendRuntime(
            fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}),
            hifigan=FakeHiFiGan(),
            torch_module=FakeTorch(),
            device="cuda:0",
            fastpitch_model_id=module.FASTPITCH_MODEL_ID,
            hifigan_model_id=module.HIFIGAN_MODEL_ID,
        )

    first_runtime = make_runtime()
    second_runtime = make_runtime()
    first = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=first_runtime,
    )
    resumed = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=second_runtime,
    )

    assert first[0].success is True
    assert resumed == ()
    assert second_runtime.fastpitch.parse_calls == [request.text]
    assert second_runtime.fastpitch.calls == []


@pytest.mark.parametrize(
    "stale_sidecar",
    [
        "missing",
        "corrupt_json",
        "request_mismatch",
        "invalid_plan_shape",
        "missing_pronunciation_diagnostic",
        "missing_grapheme_split_diagnostic",
    ],
)
def test_render_pending_requests_rerenders_success_with_invalid_timing_sidecar(
    tmp_path: Path,
    stale_sidecar: str,
) -> None:
    module = _load_backend(f"scripts.rap_audio_backends.fastpitch_backend_sidecar_{stale_sidecar}")
    request = _fixture_request()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "chunks"
    output_path = output_dir / request.song_id / "chunk-000.wav"
    sidecar_path = output_path.with_suffix(".timing.json")
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    def make_runtime():
        return module.FastPitchBackendRuntime(
            fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}),
            hifigan=FakeHiFiGan(),
            torch_module=FakeTorch(),
            device="cuda:0",
            fastpitch_model_id=module.FASTPITCH_MODEL_ID,
            hifigan_model_id=module.HIFIGAN_MODEL_ID,
        )

    first_runtime = make_runtime()
    first = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=first_runtime,
    )
    assert first[0].success is True

    if stale_sidecar == "missing":
        sidecar_path.unlink()
    elif stale_sidecar == "corrupt_json":
        sidecar_path.write_text("{not-json\n", encoding="utf-8")
    else:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if stale_sidecar == "request_mismatch":
            sidecar["request_sha256"] = "0" * 64
        elif stale_sidecar == "missing_pronunciation_diagnostic":
            sidecar.pop("pronunciation_fallback_words", None)
        elif stale_sidecar == "missing_grapheme_split_diagnostic":
            sidecar.pop("grapheme_split_words", None)
        else:
            sidecar["syllable_label_indices"] = sidecar["syllable_label_indices"][:-1]
        sidecar_path.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")

    second_runtime = make_runtime()
    rerendered = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=second_runtime,
    )

    repaired_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert len(second_runtime.fastpitch.calls) == 1
    assert len(rerendered) == 1
    assert rerendered[0].success is True
    assert repaired_sidecar["request_sha256"] == request.sha256
    assert len(repaired_sidecar["syllable_label_indices"]) == len(request.syllables)


def test_render_pending_requests_rerenders_canonical_sidecar_with_runtime_tokenizer_mismatch(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_sidecar_runtime_mismatch")
    request = _request_with_replaced_word(_fixture_request(), "blasts", "where", (("W", "EH1", "R"),))
    runtime_labels = _tokenizer_labels(request, overrides={"where": tuple("where")})
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    output_dir = tmp_path / "chunks"
    output_path = output_dir / request.song_id / "chunk-000.wav"
    sidecar_path = output_path.with_suffix(".timing.json")
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    def make_runtime():
        return module.FastPitchBackendRuntime(
            fastpitch=FakeFastPitch(labels_by_text={request.text: runtime_labels}),
            hifigan=FakeHiFiGan(),
            torch_module=FakeTorch(),
            device="cuda:0",
            fastpitch_model_id=module.FASTPITCH_MODEL_ID,
            hifigan_model_id=module.HIFIGAN_MODEL_ID,
        )

    first = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=make_runtime(),
    )
    assert first[0].success is True

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    where_syllable_index = next(
        index for index, syllable in enumerate(request.syllables) if syllable.word == "where"
    )
    where_label_indices = sidecar["syllable_label_indices"][where_syllable_index]
    assert [sidecar["tokenizer_labels"][index] for index in where_label_indices] == list("where")
    altered_labels = list(sidecar["tokenizer_labels"])
    for label_index, label in zip(where_label_indices, "there"):
        altered_labels[label_index] = label
    altered_plan = build_fastpitch_phone_plan(request, tuple(altered_labels))
    sidecar.update(
        {
            "tokenizer_labels": list(altered_plan.tokenizer_labels),
            "duration_frames": list(altered_plan.duration_frames),
            "spoken_label_indices": list(altered_plan.spoken_label_indices),
            "vowel_label_indices": list(altered_plan.vowel_label_indices),
            "syllable_phone_groups": [list(group) for group in altered_plan.syllable_phone_groups],
            "syllable_label_indices": [list(group) for group in altered_plan.syllable_label_indices],
            "anchor_error_frames": list(altered_plan.anchor_error_frames),
            "compressed_consonant_regions": list(altered_plan.compressed_consonant_regions),
            "grapheme_fallback_words": list(altered_plan.grapheme_fallback_words),
            "grapheme_split_words": list(altered_plan.grapheme_split_words),
            "pronunciation_fallback_words": list(altered_plan.pronunciation_fallback_words),
        }
    )
    sidecar_path.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")

    second_runtime = make_runtime()
    rerendered = module.render_pending_requests(
        request_path=request_path,
        record_path=record_path,
        output_dir=output_dir,
        runtime=second_runtime,
    )

    repaired_sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert len(rerendered) == 1
    assert rerendered[0].success is True
    assert len(second_runtime.fastpitch.calls) == 1
    assert repaired_sidecar["tokenizer_labels"] == list(runtime_labels)


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


def test_build_render_plan_supports_nemo_27_vocab_without_ids_to_tokens() -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_nemo_27_vocab")
    request = _fixture_request()
    labels = _tokenizer_labels(request)
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(
            labels_by_text={request.text: labels},
            vocab_factory=FakeNeMo27Vocab,
        ),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )

    plan = module.build_render_plan(request=request, runtime=runtime)

    assert plan.tokenizer_labels == labels
    assert plan.duration_tensor.shape == plan.tokens.shape


def test_grapheme_fallback_writes_auditable_timing_sidecar_and_progress(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_grapheme_sidecar")
    request = _request_with_replaced_word(_fixture_request(), "blasts", "where", (("W", "EH1", "R"),))
    labels = _tokenizer_labels(request, overrides={"where": tuple("where")})
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "chunks" / request.song_id / "chunk-000.wav"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        progress_stream=progress,
    )

    sidecar_path = output_path.with_suffix(".timing.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_plan = build_fastpitch_phone_plan(request, labels)
    assert records[0].success is True
    assert set(sidecar) == {
        "schema_version",
        "protocol_id",
        "song_id",
        "chunk_index",
        "request_sha256",
        "output_path",
        "tokenizer_labels",
        "duration_frames",
        "spoken_label_indices",
        "vowel_label_indices",
        "syllable_phone_groups",
        "syllable_label_indices",
        "anchor_error_frames",
        "compressed_consonant_regions",
        "grapheme_fallback_words",
        "grapheme_split_words",
        "pronunciation_fallback_words",
    }
    assert sidecar["schema_version"] == 1
    assert sidecar["protocol_id"] == ProtocolId.FASTPITCH_PHONEME.value
    assert sidecar["song_id"] == request.song_id
    assert sidecar["chunk_index"] == request.chunk_index
    assert sidecar["request_sha256"] == request.sha256
    assert sidecar["output_path"] == str(output_path)
    assert sidecar["tokenizer_labels"] == list(labels)
    assert sidecar["duration_frames"] == list(expected_plan.duration_frames)
    assert sidecar["spoken_label_indices"] == list(expected_plan.spoken_label_indices)
    assert sidecar["vowel_label_indices"] == list(expected_plan.vowel_label_indices)
    assert sidecar["syllable_phone_groups"] == [list(group) for group in expected_plan.syllable_phone_groups]
    assert sidecar["syllable_label_indices"] == [list(group) for group in expected_plan.syllable_label_indices]
    assert sidecar["anchor_error_frames"] == list(expected_plan.anchor_error_frames)
    assert sidecar["compressed_consonant_regions"] == list(expected_plan.compressed_consonant_regions)
    assert sidecar["grapheme_fallback_words"] == list(expected_plan.grapheme_fallback_words)
    assert sidecar["grapheme_split_words"] == list(expected_plan.grapheme_split_words)
    assert sidecar["pronunciation_fallback_words"] == list(expected_plan.pronunciation_fallback_words)
    assert sum(sidecar["duration_frames"]) == round(request.duration_seconds * 22050 / 256)
    assert all(
        [sidecar["tokenizer_labels"][label_index] for label_index in label_group] == phone_group
        for phone_group, label_group in zip(
            sidecar["syllable_phone_groups"],
            sidecar["syllable_label_indices"],
        )
    )
    assert "grapheme_fallback_words=where" in progress.getvalue()
    assert "grapheme_split_words=none" in progress.getvalue()
    assert "pronunciation_fallback_words=none" in progress.getvalue()


def test_grapheme_split_writes_auditable_timing_sidecar_and_progress(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_grapheme_split_sidecar")
    request = _request_with_replaced_word(
        _fixture_request(),
        "rocket",
        "ruins",
        (("R", "UW1"), ("AH0", "N", "Z")),
    )
    labels = _tokenizer_labels(request, overrides={"ruins": tuple("ruins")})
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "chunks" / request.song_id / "chunk-000.wav"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        progress_stream=progress,
    )

    sidecar = json.loads(output_path.with_suffix(".timing.json").read_text(encoding="utf-8"))
    assert records[0].success is True
    assert sidecar["syllable_phone_groups"][:2] == [["r", "u"], ["i", "n", "s"]]
    assert sidecar["grapheme_fallback_words"] == ["ruins"]
    assert sidecar["grapheme_split_words"] == ["ruins"]
    assert sidecar["pronunciation_fallback_words"] == []
    assert "grapheme_split_words=ruins" in progress.getvalue()


def test_pronunciation_fallback_writes_auditable_timing_sidecar_and_progress(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_pronunciation_sidecar")
    request = _request_with_replaced_word(
        _fixture_request(),
        "rocket",
        "secrets",
        (("S", "IY1"), ("K", "R", "AH0", "T", "S")),
    )
    labels = _tokenizer_labels(
        request,
        overrides={"secrets": ("S", "IY1", "K", "R", "IH0", "T", "S")},
    )
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "chunks" / request.song_id / "chunk-000.wav"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        progress_stream=progress,
    )

    sidecar = json.loads(output_path.with_suffix(".timing.json").read_text(encoding="utf-8"))
    assert records[0].success is True
    assert sidecar["syllable_phone_groups"][:2] == [
        ["S", "IY1"],
        ["K", "R", "IH0", "T", "S"],
    ]
    assert sidecar["grapheme_fallback_words"] == []
    assert sidecar["pronunciation_fallback_words"] == ["secrets"]
    assert "pronunciation_fallback_words=secrets" in progress.getvalue()


def test_grapheme_fallback_is_reported_for_failed_render_without_success_sidecar(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_grapheme_failure")
    request = _request_with_replaced_word(_fixture_request(), "blasts", "where", (("W", "EH1", "R"),))
    labels = _tokenizer_labels(request, overrides={"where": tuple("where")})
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}, failures_remaining=1),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "chunks" / request.song_id / "chunk-000.wav"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        max_attempts=1,
        progress_stream=progress,
    )

    error = json.loads(str(records[0].error))
    assert records[0].success is False
    assert error["grapheme_fallback_words"] == ["where"]
    assert error["grapheme_split_words"] == []
    assert error["pronunciation_fallback_words"] == []
    assert "grapheme_fallback_words=where" in progress.getvalue()
    assert "grapheme_split_words=none" in progress.getvalue()
    assert "pronunciation_fallback_words=none" in progress.getvalue()
    assert not output_path.with_suffix(".timing.json").exists()


def test_grapheme_split_is_reported_for_failed_render_without_success_sidecar(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_grapheme_split_failure")
    request = _request_with_replaced_word(
        _fixture_request(),
        "rocket",
        "ruins",
        (("R", "UW1"), ("AH0", "N", "Z")),
    )
    labels = _tokenizer_labels(request, overrides={"ruins": tuple("ruins")})
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}, failures_remaining=1),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "chunks" / request.song_id / "chunk-000.wav"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        max_attempts=1,
        progress_stream=progress,
    )

    error = json.loads(str(records[0].error))
    assert records[0].success is False
    assert error["grapheme_fallback_words"] == ["ruins"]
    assert error["grapheme_split_words"] == ["ruins"]
    assert error["pronunciation_fallback_words"] == []
    assert "grapheme_split_words=ruins" in progress.getvalue()
    assert not output_path.with_suffix(".timing.json").exists()


def test_pronunciation_fallback_is_reported_for_failed_render_without_success_sidecar(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_pronunciation_failure")
    request = _request_with_replaced_word(_fixture_request(), "blasts", "bots", (("B", "AO1", "T", "S"),))
    labels = _tokenizer_labels(request, overrides={"bots": ("B", "AA1", "T", "Z")})
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: labels}, failures_remaining=1),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    output_path = tmp_path / "chunks" / request.song_id / "chunk-000.wav"
    progress = io.StringIO()
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
        max_attempts=1,
        progress_stream=progress,
    )

    error = json.loads(str(records[0].error))
    assert records[0].success is False
    assert error["grapheme_fallback_words"] == []
    assert error["grapheme_split_words"] == []
    assert error["pronunciation_fallback_words"] == ["bots"]
    assert "pronunciation_fallback_words=bots" in progress.getvalue()
    assert not output_path.with_suffix(".timing.json").exists()


def test_timing_sidecar_replacement_is_atomic_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_atomic_sidecar")
    request = _fixture_request()
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}),
        hifigan=FakeHiFiGan(),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    output_path = tmp_path / "chunk-000.wav"
    sidecar_path = output_path.with_suffix(".timing.json")
    sidecar_path.write_text('{"previous":true}\n', encoding="utf-8")
    plan = module.build_render_plan(request=request, runtime=runtime)

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("synthetic atomic replace failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic atomic replace failure"):
        module._write_timing_sidecar(output_path, request, plan)

    assert sidecar_path.read_text(encoding="utf-8") == '{"previous":true}\n'
    assert list(tmp_path.glob(".chunk-000.timing.json.*.tmp")) == []


def test_runtime_sample_rate_controls_successful_wav_and_record(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_success_sample_rate")
    request = _fixture_request()
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}),
        hifigan=FakeHiFiGan(sample_rate=44_100),
        torch_module=FakeTorch(),
        device="cuda:0",
        fastpitch_model_id=module.FASTPITCH_MODEL_ID,
        hifigan_model_id=module.HIFIGAN_MODEL_ID,
    )
    request_path = tmp_path / "requests.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")

    records = module.render_pending_requests(
        request_path=request_path,
        record_path=tmp_path / "records.jsonl",
        output_dir=tmp_path / "chunks",
        runtime=runtime,
    )

    sample_rate_hz, samples = wavfile.read(Path(records[0].output_path or ""))
    assert records[0].success is True
    assert records[0].sample_rate_hz == 44_100
    assert sample_rate_hz == 44_100
    assert samples.shape == (384,)


def test_render_pending_requests_retries_and_logs_silence_after_bounded_failures(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_retry")
    request = _fixture_request()
    runtime = module.FastPitchBackendRuntime(
        fastpitch=FakeFastPitch(labels_by_text={request.text: _tokenizer_labels(request)}, failures_remaining=3),
        hifigan=FakeHiFiGan(sample_rate=44_100),
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
    assert record.sample_rate_hz == 44_100

    sample_rate_hz, samples = wavfile.read(Path(record.output_path))
    assert sample_rate_hz == 44_100
    assert samples.shape == (module.chunk_frame_count(request, sample_rate_hz=44_100),)
    assert np.count_nonzero(samples) == 0

    error_payload = json.loads(str(record.error))
    assert error_payload["message"] == "synthetic FastPitch failure"
    assert error_payload["attempts"] == 3
    assert error_payload["prosody_controls"] == "not_started"
    assert error_payload["tokenizer_labels"]
    assert error_payload["duration_frames"]
    assert error_payload["anchor_error_frames"]


def test_main_rerenders_prior_failure_and_reports_failure_again(tmp_path: Path) -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_failure_resume")
    request = _fixture_request()
    request_path = tmp_path / "requests.jsonl"
    record_path = tmp_path / "records.jsonl"
    request_path.write_text(json.dumps(request.to_payload()) + "\n", encoding="utf-8")
    created_runtimes = []

    def failing_factory(*, fastpitch_model_id: str, hifigan_model_id: str, device: str):
        runtime = module.FastPitchBackendRuntime(
            fastpitch=FakeFastPitch(
                labels_by_text={request.text: _tokenizer_labels(request)},
                failures_remaining=1,
            ),
            hifigan=FakeHiFiGan(),
            torch_module=FakeTorch(),
            device=device,
            fastpitch_model_id=fastpitch_model_id,
            hifigan_model_id=hifigan_model_id,
        )
        created_runtimes.append(runtime)
        return runtime

    argv = [
        "--requests-jsonl",
        str(request_path),
        "--records-jsonl",
        str(record_path),
        "--output-dir",
        str(tmp_path / "chunks"),
        "--max-attempts",
        "1",
    ]

    assert module.main(argv, runtime_factory=failing_factory) == 1
    assert module.main(argv, runtime_factory=failing_factory) == 1
    assert [len(runtime.fastpitch.calls) for runtime in created_runtimes] == [1, 1]
    assert len(record_path.read_text(encoding="utf-8").splitlines()) == 1


def test_pretrained_checkpoint_loading_uses_requested_map_location_then_device() -> None:
    module = _load_backend("scripts.rap_audio_backends.fastpitch_backend_model_loading")
    from_pretrained_calls: list[dict[str, object]] = []

    class FakeModel:
        def __init__(self) -> None:
            self.to_calls: list[str] = []
            self.eval_called = False

        def to(self, device: str):
            self.to_calls.append(device)
            return self

        def eval(self):
            self.eval_called = True
            return self

    model = FakeModel()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, **kwargs: object) -> FakeModel:
            from_pretrained_calls.append(dict(kwargs))
            return model

    loaded = module._load_nemo_model(FakeModelClass, "tts_en_fastpitch", device="cuda:6")

    assert loaded is model
    assert from_pretrained_calls == [{"model_name": "tts_en_fastpitch", "map_location": "cuda:6"}]
    assert model.to_calls == ["cuda:6"]
    assert model.eval_called is True


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
    assert len(created_runtimes[0].fastpitch.calls) == 1
