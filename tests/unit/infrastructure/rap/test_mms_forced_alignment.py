from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed
from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget
from streammuse.infrastructure.rap.mms_forced_alignment import (
    CharacterSpan,
    MmsAlignmentResult,
    MmsForcedAligner,
    WordSpan,
    map_syllable_onsets,
    normalize_mms_transcript,
)


class _FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.to_calls: list[str] = []

    @property
    def shape(self) -> tuple[int, ...]:
        return self.values.shape

    def unsqueeze(self, axis: int) -> "_FakeTensor":
        return _FakeTensor(np.expand_dims(self.values, axis))

    def to(self, device: str) -> "_FakeTensor":
        self.to_calls.append(device)
        return self

    def __getitem__(self, item: object) -> "_FakeTensor":
        return _FakeTensor(self.values[item])


def test_normalizes_supported_text_to_ascii_mms_words() -> None:
    assert normalize_mms_transcript("Can't, CAF\u00c9!") == ("cant", "cafe")
    with pytest.raises(PhraseRenderFailed, match="digits"):
        normalize_mms_transcript("version 2")
    with pytest.raises(PhraseRenderFailed, match="empty"):
        normalize_mms_transcript("?!")


class _InferenceMode(AbstractContextManager[None]):
    def __init__(self, torch_module: "_FakeTorch") -> None:
        self._torch = torch_module

    def __enter__(self) -> None:
        self._torch.inference_entries.append("enter")
        self._torch.inference_active = True
        return None

    def __exit__(self, *args: object) -> None:
        self._torch.inference_active = False
        self._torch.inference_entries.append("exit")


class _FakeTorch:
    def __init__(self) -> None:
        self.from_numpy_lengths: list[int] = []
        self.inference_entries: list[str] = []
        self.inference_active = False

    def from_numpy(self, values: np.ndarray) -> _FakeTensor:
        self.from_numpy_lengths.append(len(values))
        return _FakeTensor(values)

    def inference_mode(self) -> _InferenceMode:
        return _InferenceMode(self)


class _FakeModel:
    def __init__(self, *, expected_waveform_frames: int = 16_000) -> None:
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.call_count = 0
        self.expected_waveform_frames = expected_waveform_frames

    def to(self, device: str) -> "_FakeModel":
        self.to_calls.append(device)
        return self

    def eval(self) -> "_FakeModel":
        self.eval_calls += 1
        return self

    def __call__(self, waveform: _FakeTensor) -> tuple[_FakeTensor, None]:
        assert waveform.shape == (1, self.expected_waveform_frames)
        self.call_count += 1
        return _FakeTensor(np.zeros((1, 100, 32), dtype=np.float32)), None


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, words: list[str]) -> list[list[int]]:
        self.calls.append(tuple(words))
        return [[ord(character) for character in word] for word in words]


class _FakeCtcAligner:
    def __init__(self, torch_module: _FakeTorch) -> None:
        self._torch = torch_module
        self.call_count = 0

    def __call__(
        self,
        emission: _FakeTensor,
        tokens: list[list[int]],
    ) -> list[list[SimpleNamespace]]:
        assert emission.shape == (100, 32)
        assert self._torch.inference_active
        self.call_count += 1
        cursor = 5
        aligned: list[list[SimpleNamespace]] = []
        for word_tokens in tokens:
            spans = []
            for token in word_tokens:
                spans.append(
                    SimpleNamespace(
                        token=token, start=cursor, end=cursor + 1, score=0.9
                    )
                )
                cursor += 2
            aligned.append(spans)
        return aligned


def test_loads_mms_once_and_reuses_it_with_inference_only_resampling(
    tmp_path: Path,
) -> None:
    source_wav = tmp_path / "source.wav"
    wavfile.write(
        source_wav,
        24_000,
        np.sin(np.linspace(0.0, 20.0, 24_000, dtype=np.float32)),
    )
    torch_module = _FakeTorch()
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    ctc_aligner = _FakeCtcAligner(torch_module)
    runtime = SimpleNamespace(
        model=model,
        tokenizer=tokenizer,
        aligner=ctc_aligner,
        torch_module=torch_module,
        sample_rate_hz=16_000,
        identity="torchaudio.pipelines.MMS_FA",
        version="torchaudio 2.8.0+cu128 / MMS_FA",
    )
    load_calls: list[str] = []

    def load_runtime() -> object:
        load_calls.append("load")
        return runtime

    aligner = MmsForcedAligner.load(device="cuda:3", runtime_loader=load_runtime)

    first = aligner.align(source_wav, "Steady, MOTION!")
    second = aligner.align(source_wav, "steady motion")
    warmup = aligner.warmup(source_wav, "warm voice")

    assert load_calls == ["load"]
    assert model.to_calls == ["cuda:3"]
    assert model.eval_calls == 1
    assert model.call_count == 3
    assert tokenizer.calls == [
        ("steady", "motion"),
        ("steady", "motion"),
        ("warm", "voice"),
    ]
    assert ctc_aligner.call_count == 3
    assert torch_module.from_numpy_lengths == [16_000, 16_000, 16_000]
    assert torch_module.inference_entries == [
        "enter",
        "exit",
        "enter",
        "exit",
        "enter",
        "exit",
    ]
    assert (
        first.normalized_transcript == second.normalized_transcript == "steady motion"
    )
    assert tuple(span.word for span in first.word_spans) == ("steady", "motion")
    assert "".join(span.character for span in first.character_spans) == "steadymotion"
    assert first.duration_seconds == second.duration_seconds == 1.0
    assert first.aligner_identity == "torchaudio.pipelines.MMS_FA"
    assert first.aligner_version == "torchaudio 2.8.0+cu128 / MMS_FA"
    assert warmup["aligned"] is True
    assert warmup["normalized_transcript"] == "warm voice"
    assert warmup["confidence"] == pytest.approx(0.9)


def test_ctc_seconds_use_actual_post_resample_waveform_and_keep_source_bounds(
    tmp_path: Path,
) -> None:
    source_wav = tmp_path / "source.wav"
    wavfile.write(
        source_wav,
        24_000,
        np.sin(np.linspace(0.0, 20.0, 24_001, dtype=np.float32)),
    )
    torch_module = _FakeTorch()
    aligner = MmsForcedAligner.load(
        device="cuda:0",
        runtime_loader=lambda: SimpleNamespace(
            model=_FakeModel(expected_waveform_frames=16_001),
            tokenizer=_FakeTokenizer(),
            aligner=_FakeCtcAligner(torch_module),
            torch_module=torch_module,
            sample_rate_hz=16_000,
            identity="torchaudio.pipelines.MMS_FA",
            version="torchaudio 2.8.0 / MMS_FA",
        ),
    )

    result = aligner.align(source_wav, "steady motion")

    inference_duration = 16_001 / 16_000
    assert torch_module.from_numpy_lengths == [16_001]
    assert result.character_spans[0].start_seconds == pytest.approx(
        5 * inference_duration / 100,
        abs=1e-12,
    )
    assert result.duration_seconds == inference_duration
    assert result.source_sample_rate_hz == 24_000
    assert result.source_frame_count == 24_001
    assert result.source_duration_seconds == 24_001 / 24_000
    assert result.inference_sample_rate_hz == 16_000
    assert result.inference_frame_count == 16_001
    assert result.emission_frame_count == 100


def test_repeated_words_keep_distinct_positional_character_spans(
    tmp_path: Path,
) -> None:
    source_wav = tmp_path / "source.wav"
    wavfile.write(
        source_wav,
        16_000,
        np.sin(np.linspace(0.0, 20.0, 16_000, dtype=np.float32)),
    )
    torch_module = _FakeTorch()
    aligner = MmsForcedAligner.load(
        device="cuda:0",
        runtime_loader=lambda: SimpleNamespace(
            model=_FakeModel(),
            tokenizer=_FakeTokenizer(),
            aligner=_FakeCtcAligner(torch_module),
            torch_module=torch_module,
            sample_rate_hz=16_000,
            identity="torchaudio.pipelines.MMS_FA",
            version="torchaudio 2.8.0 / MMS_FA",
        ),
    )

    result = aligner.align(source_wav, "echo echo")

    assert tuple(span.word for span in result.word_spans) == ("echo", "echo")
    assert tuple(span.word_index for span in result.word_spans) == (0, 1)
    assert all(span.word_index == 0 for span in result.word_spans[0].characters)
    assert all(span.word_index == 1 for span in result.word_spans[1].characters)
    assert result.word_spans[1].start_seconds > result.word_spans[0].end_seconds


def _word_span(
    word: str,
    word_index: int,
    *,
    start_seconds: float,
    character_step: float,
    score: float = 0.9,
) -> WordSpan:
    characters = tuple(
        CharacterSpan(
            word=word,
            word_index=word_index,
            character=character,
            character_index=index,
            start_seconds=start_seconds + index * character_step,
            end_seconds=start_seconds + (index + 1) * character_step,
            score=score,
        )
        for index, character in enumerate(word)
    )
    return WordSpan(
        word=word,
        word_index=word_index,
        start_seconds=characters[0].start_seconds,
        end_seconds=characters[-1].end_seconds,
        score=score,
        characters=characters,
    )


def _syllable(
    word: str,
    index_in_word: int,
    phonemes: tuple[str, ...],
    target_seconds: float,
) -> SyllableTarget:
    return SyllableTarget(
        word=word,
        index_in_word=index_in_word,
        phonemes=phonemes,
        lexical_stress=1 if index_in_word == 0 else 0,
        target_stress=1.0 if index_in_word == 0 else 0.5,
        boundary_strength=0,
        absolute_tick=round(target_seconds * 4),
        tick_in_chunk=round(target_seconds * 4),
        target_seconds=target_seconds,
    )


def test_maps_every_syllable_with_ordered_exact_weighted_and_proportional_methods() -> (
    None
):
    words = (
        _word_span("steady", 0, start_seconds=0.10, character_step=0.04),
        _word_span("rhythm", 1, start_seconds=0.50, character_step=0.04),
        _word_span("a", 2, start_seconds=0.90, character_step=0.20),
    )
    alignment = MmsAlignmentResult(
        normalized_transcript="steady rhythm a",
        character_spans=tuple(
            character for word in words for character in word.characters
        ),
        word_spans=words,
        duration_seconds=1.20,
        alignment_time_ms=12.0,
        aligner_identity="torchaudio.pipelines.MMS_FA",
        aligner_version="torchaudio 2.8.0 / MMS_FA",
        warnings=(),
    )
    targets = (
        _syllable("steady", 0, ("S", "T", "EH1"), 0.20),
        _syllable("steady", 1, ("D", "IY0"), 0.50),
        _syllable("rhythm", 0, ("R", "IH1", "DH"), 0.80),
        _syllable("rhythm", 1, ("AH0", "M"), 1.10),
        _syllable("a", 0, ("EY1",), 1.40),
        _syllable("a", 1, ("AH0",), 1.70),
    )

    mapped = map_syllable_onsets(
        alignment,
        targets,
        source_sample_rate_hz=1_000,
        source_frame_count=1_200,
    )

    assert [item["method"] for item in mapped.anchor_diagnostics] == [
        "orthographic_vowel_groups",
        "orthographic_vowel_groups",
        "phoneme_weighted_character",
        "phoneme_weighted_character",
        "word_duration_proportional",
        "word_duration_proportional",
    ]
    assert [anchor.source_sample for anchor in mapped.anchors] == [
        100,
        300,
        500,
        620,
        900,
        1_000,
    ]
    assert [anchor.target_seconds for anchor in mapped.anchors] == [
        target.target_seconds for target in targets
    ]
    assert all(anchor.anchor_kind == "syllable_onset" for anchor in mapped.anchors)
    assert mapped.method_counts == {
        "orthographic_vowel_groups": 2,
        "phoneme_weighted_character": 2,
        "word_duration_proportional": 2,
    }
    assert mapped.coverage == 1.0
    assert any("phoneme-weighted" in warning for warning in mapped.warnings)
    assert any("proportional" in warning for warning in mapped.warnings)


def test_low_confidence_complete_mapping_warns_and_continues() -> None:
    word = _word_span(
        "steady",
        0,
        start_seconds=0.10,
        character_step=0.04,
        score=0.2,
    )
    alignment = MmsAlignmentResult(
        normalized_transcript="steady",
        character_spans=word.characters,
        word_spans=(word,),
        duration_seconds=0.50,
        alignment_time_ms=2.0,
        aligner_identity="torchaudio.pipelines.MMS_FA",
        aligner_version="torchaudio 2.8.0 / MMS_FA",
        warnings=(),
    )
    targets = (
        _syllable("steady", 0, ("S", "T", "EH1"), 0.20),
        _syllable("steady", 1, ("D", "IY0"), 0.50),
    )

    mapped = map_syllable_onsets(
        alignment,
        targets,
        source_sample_rate_hz=1_000,
        source_frame_count=500,
    )

    assert len(mapped.anchors) == len(targets)
    assert mapped.coverage == 1.0
    assert any("low MMS confidence" in warning for warning in mapped.warnings)


def test_rejects_missing_or_reordered_aligned_words() -> None:
    steady = _word_span("steady", 0, start_seconds=0.10, character_step=0.03)
    motion = _word_span("motion", 1, start_seconds=0.50, character_step=0.03)
    alignment = MmsAlignmentResult(
        normalized_transcript="motion steady",
        character_spans=motion.characters + steady.characters,
        word_spans=(replace(motion, word_index=0), replace(steady, word_index=1)),
        duration_seconds=0.90,
        alignment_time_ms=1.0,
        aligner_identity="torchaudio.pipelines.MMS_FA",
        aligner_version="torchaudio 2.8.0 / MMS_FA",
        warnings=(),
    )
    targets = (
        _syllable("steady", 0, ("S", "T", "EH1"), 0.20),
        _syllable("steady", 1, ("D", "IY0"), 0.50),
        _syllable("motion", 0, ("M", "OW1"), 0.80),
        _syllable("motion", 1, ("SH", "AH0", "N"), 1.10),
    )

    with pytest.raises(PhraseRenderFailed, match="exact planned transcript"):
        map_syllable_onsets(
            alignment,
            targets,
            source_sample_rate_hz=1_000,
            source_frame_count=900,
        )


def test_rejects_impossible_duplicate_source_onsets() -> None:
    word = _word_span("a", 0, start_seconds=0.1000, character_step=0.0004)
    alignment = MmsAlignmentResult(
        normalized_transcript="a",
        character_spans=word.characters,
        word_spans=(word,),
        duration_seconds=0.20,
        alignment_time_ms=1.0,
        aligner_identity="torchaudio.pipelines.MMS_FA",
        aligner_version="torchaudio 2.8.0 / MMS_FA",
        warnings=(),
    )
    targets = (
        _syllable("a", 0, ("EY1",), 0.20),
        _syllable("a", 1, ("AH0",), 0.50),
    )

    with pytest.raises(PhraseRenderFailed, match="unique"):
        map_syllable_onsets(
            alignment,
            targets,
            source_sample_rate_hz=1_000,
            source_frame_count=200,
        )


@pytest.mark.parametrize(
    ("start_seconds", "end_seconds"),
    [
        (np.nan, 0.20),
        (0.10, np.inf),
        (0.10, 1.10),
    ],
)
def test_rejects_non_finite_or_out_of_bounds_character_spans(
    start_seconds: float,
    end_seconds: float,
) -> None:
    character = CharacterSpan(
        word="a",
        word_index=0,
        character="a",
        character_index=0,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        score=0.9,
    )
    word = WordSpan(
        word="a",
        word_index=0,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        score=0.9,
        characters=(character,),
    )
    alignment = MmsAlignmentResult(
        normalized_transcript="a",
        character_spans=(character,),
        word_spans=(word,),
        duration_seconds=1.0,
        alignment_time_ms=1.0,
        aligner_identity="torchaudio.pipelines.MMS_FA",
        aligner_version="torchaudio 2.8.0 / MMS_FA",
        warnings=(),
    )

    with pytest.raises(PhraseRenderFailed, match="finite|bounds"):
        map_syllable_onsets(
            alignment,
            (_syllable("a", 0, ("EY1",), 0.20),),
            source_sample_rate_hz=1_000,
            source_frame_count=1_000,
        )


def test_rejects_non_monotonic_character_spans_across_words() -> None:
    first = _word_span("one", 0, start_seconds=0.30, character_step=0.03)
    second = _word_span("two", 1, start_seconds=0.20, character_step=0.03)
    alignment = MmsAlignmentResult(
        normalized_transcript="one two",
        character_spans=first.characters + second.characters,
        word_spans=(first, second),
        duration_seconds=0.60,
        alignment_time_ms=1.0,
        aligner_identity="torchaudio.pipelines.MMS_FA",
        aligner_version="torchaudio 2.8.0 / MMS_FA",
        warnings=(),
    )

    with pytest.raises(PhraseRenderFailed, match="ordered"):
        map_syllable_onsets(
            alignment,
            (
                _syllable("one", 0, ("W", "AH1", "N"), 0.20),
                _syllable("two", 0, ("T", "UW1"), 0.50),
            ),
            source_sample_rate_hz=1_000,
            source_frame_count=600,
        )


def test_rejects_ctc_token_sequence_that_does_not_cover_normalized_words(
    tmp_path: Path,
) -> None:
    source_wav = tmp_path / "source.wav"
    wavfile.write(
        source_wav,
        16_000,
        np.sin(np.linspace(0.0, 20.0, 16_000, dtype=np.float32)),
    )
    torch_module = _FakeTorch()

    class WrongTokenAligner(_FakeCtcAligner):
        def __call__(
            self,
            emission: _FakeTensor,
            tokens: list[list[int]],
        ) -> list[list[SimpleNamespace]]:
            aligned = super().__call__(emission, tokens)
            aligned[0][0] = SimpleNamespace(
                token=-1,
                start=aligned[0][0].start,
                end=aligned[0][0].end,
                score=aligned[0][0].score,
            )
            return aligned

    runtime = SimpleNamespace(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        aligner=WrongTokenAligner(torch_module),
        torch_module=torch_module,
        sample_rate_hz=16_000,
        identity="torchaudio.pipelines.MMS_FA",
        version="torchaudio 2.8.0 / MMS_FA",
    )
    aligner = MmsForcedAligner.load(
        device="cuda:0",
        runtime_loader=lambda: runtime,
    )

    with pytest.raises(PhraseRenderFailed, match="token coverage"):
        aligner.align(source_wav, "steady motion")
