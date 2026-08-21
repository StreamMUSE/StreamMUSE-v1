"""Resident torchaudio MMS forced alignment and syllable-onset mapping."""

from __future__ import annotations

import importlib
import re
import threading
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import gcd, isfinite
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Protocol

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from streammuse.application.rap.chunk_orchestration import PhraseRenderFailed
from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget
from streammuse.experiments.rap_audio_protocols.warp import VowelAnchor


_ASCII_WORDS = re.compile(r"[a-z]+(?:'[a-z]+)*")
_VOWEL_GROUPS = re.compile(r"[aeiouy]+")
_LOW_CONFIDENCE_SCORE = 0.5


@dataclass(frozen=True)
class CharacterSpan:
    """One transcript character aligned to source-audio seconds."""

    word: str
    word_index: int
    character: str
    character_index: int
    start_seconds: float
    end_seconds: float
    score: float


@dataclass(frozen=True)
class WordSpan:
    """One exact normalized word and its ordered character evidence."""

    word: str
    word_index: int
    start_seconds: float
    end_seconds: float
    score: float
    characters: tuple[CharacterSpan, ...]


@dataclass(frozen=True)
class MmsAlignmentResult:
    """Immutable known-transcript MMS alignment in source-relative seconds."""

    normalized_transcript: str
    character_spans: tuple[CharacterSpan, ...]
    word_spans: tuple[WordSpan, ...]
    duration_seconds: float
    alignment_time_ms: float
    aligner_identity: str
    aligner_version: str
    warnings: tuple[str, ...]
    source_sample_rate_hz: int | None = None
    source_frame_count: int | None = None
    inference_sample_rate_hz: int | None = None
    inference_frame_count: int | None = None
    emission_frame_count: int | None = None

    @property
    def confidence(self) -> float:
        if not self.character_spans:
            return 0.0
        return float(
            sum(span.score for span in self.character_spans) / len(self.character_spans)
        )

    @property
    def source_duration_seconds(self) -> float:
        if self.source_sample_rate_hz and self.source_frame_count is not None:
            return self.source_frame_count / self.source_sample_rate_hz
        return self.duration_seconds


@dataclass(frozen=True)
class _MmsInferenceWaveform:
    tensor: object
    source_sample_rate_hz: int
    source_frame_count: int
    inference_sample_rate_hz: int
    inference_frame_count: int

    @property
    def inference_duration_seconds(self) -> float:
        return self.inference_frame_count / self.inference_sample_rate_hz


class PhraseForcedAligner(Protocol):
    """Replaceable known-transcript phrase alignment boundary."""

    def align(self, source_wav: Path, transcript: str) -> MmsAlignmentResult: ...


@dataclass(frozen=True)
class SyllableOnsetMap:
    """Complete source-to-target syllable onset map and mapping evidence."""

    anchors: tuple[VowelAnchor, ...]
    anchor_diagnostics: tuple[Mapping[str, object], ...]
    method_counts: Mapping[str, int]
    warnings: tuple[str, ...]
    coverage: float


def map_syllable_onsets(
    alignment: MmsAlignmentResult,
    syllables: Sequence[SyllableTarget],
    *,
    source_sample_rate_hz: int,
    source_frame_count: int,
) -> SyllableOnsetMap:
    if source_sample_rate_hz <= 0 or source_frame_count <= 0:
        raise PhraseRenderFailed("source audio bounds must be positive")
    if not syllables:
        raise PhraseRenderFailed("syllable onset mapping requires planned syllables")

    grouped_targets = _group_planned_syllables(syllables)
    expected_words = tuple(word for word, _ in grouped_targets)
    aligned_words = tuple(word.word for word in alignment.word_spans)
    normalized_words = tuple(alignment.normalized_transcript.split())
    if aligned_words != expected_words or normalized_words != expected_words:
        raise PhraseRenderFailed(
            "MMS aligned word order does not cover the exact planned transcript"
        )
    _validate_alignment_evidence(
        alignment,
        source_duration_seconds=source_frame_count / source_sample_rate_hz,
    )

    anchors: list[VowelAnchor] = []
    diagnostics: list[Mapping[str, object]] = []
    methods: Counter[str] = Counter()
    warnings = list(alignment.warnings)
    previous_source_sample = -1
    previous_target_sample = -1
    for word_span, (_, word_targets) in zip(alignment.word_spans, grouped_targets):
        source_times, character_indices, method = _map_word_onsets(
            word_span,
            word_targets,
        )
        methods[method] += len(word_targets)
        if method == "phoneme_weighted_character":
            warnings.append(
                f"MMS used phoneme-weighted character mapping for '{word_span.word}'"
            )
        elif method == "word_duration_proportional":
            warnings.append(
                f"MMS used proportional word-duration mapping for '{word_span.word}'"
            )

        for target, source_seconds, character_index in zip(
            word_targets,
            source_times,
            character_indices,
        ):
            target_seconds = float(target.target_seconds)
            if not isfinite(target_seconds) or target_seconds < 0:
                raise PhraseRenderFailed(
                    "planned target onset must be finite and non-negative"
                )
            source_sample = round(source_seconds * source_sample_rate_hz)
            target_sample = round(target_seconds * source_sample_rate_hz)
            if (
                not isfinite(source_seconds)
                or source_sample < 0
                or source_sample >= source_frame_count
                or source_sample <= previous_source_sample
            ):
                raise PhraseRenderFailed(
                    "MMS syllable source onsets must be unique, in bounds, and strictly increasing"
                )
            if target_sample <= previous_target_sample:
                raise PhraseRenderFailed(
                    "planned syllable target onsets must be unique and strictly increasing"
                )

            character_span = (
                word_span.characters[character_index]
                if character_index is not None
                else None
            )
            confidence = (
                character_span.score if character_span is not None else word_span.score
            )
            if confidence < _LOW_CONFIDENCE_SCORE:
                warnings.append(
                    f"low MMS confidence for '{word_span.word}' ({confidence:.3f}); mapping retained"
                )
            aligned_evidence = (
                f"MMS:{character_span.character}"
                if character_span is not None
                else "MMS:word_duration"
            )
            anchor = VowelAnchor(
                word=target.word,
                index_in_word=target.index_in_word,
                planned_phone=_planned_nucleus(target.phonemes),
                aligned_phone=aligned_evidence,
                requested_source_seconds=source_seconds,
                source_seconds=source_seconds,
                requested_target_seconds=target_seconds,
                target_seconds=target_seconds,
                requested_source_sample=source_sample,
                source_sample=source_sample,
                target_sample=target_sample,
                source_boundary_adjusted=False,
                boundary_adjusted=False,
                anchor_kind="syllable_onset",
            )
            anchors.append(anchor)
            diagnostics.append(
                MappingProxyType(
                    {
                        "word": target.word,
                        "index_in_word": target.index_in_word,
                        "method": method,
                        "character_index": character_index,
                        "character": (
                            character_span.character
                            if character_span is not None
                            else None
                        ),
                        "confidence": confidence,
                        "source_seconds": source_seconds,
                        "source_sample": source_sample,
                        "target_seconds": target_seconds,
                        "target_sample": target_sample,
                    }
                )
            )
            previous_source_sample = source_sample
            previous_target_sample = target_sample

    if len(anchors) != len(syllables):
        raise PhraseRenderFailed("MMS mapping did not cover every planned syllable")
    return SyllableOnsetMap(
        anchors=tuple(anchors),
        anchor_diagnostics=tuple(diagnostics),
        method_counts=MappingProxyType(dict(methods)),
        warnings=tuple(dict.fromkeys(warnings)),
        coverage=len(anchors) / len(syllables),
    )


def _group_planned_syllables(
    syllables: Sequence[SyllableTarget],
) -> tuple[tuple[str, tuple[SyllableTarget, ...]], ...]:
    grouped: list[tuple[str, list[SyllableTarget]]] = []
    for syllable in syllables:
        normalized = normalize_mms_transcript(syllable.word)
        if len(normalized) != 1:
            raise PhraseRenderFailed(
                "planned syllable word must normalize to one MMS word"
            )
        word = normalized[0]
        if syllable.index_in_word == 0:
            grouped.append((word, [syllable]))
            continue
        if not grouped or grouped[-1][0] != word:
            raise PhraseRenderFailed("planned syllable words or indices are incomplete")
        if syllable.index_in_word != len(grouped[-1][1]):
            raise PhraseRenderFailed(
                "planned syllable indices must be contiguous and unique"
            )
        grouped[-1][1].append(syllable)
    if not grouped:
        raise PhraseRenderFailed("planned syllable groups must not be empty")
    return tuple((word, tuple(items)) for word, items in grouped)


def _validate_alignment_evidence(
    alignment: MmsAlignmentResult,
    *,
    source_duration_seconds: float,
) -> None:
    if not isfinite(alignment.duration_seconds) or alignment.duration_seconds <= 0:
        raise PhraseRenderFailed("MMS alignment duration must be finite and positive")
    if (
        alignment.source_sample_rate_hz is not None
        and alignment.source_frame_count is not None
        and abs(alignment.source_duration_seconds - source_duration_seconds) > 1e-12
    ):
        raise PhraseRenderFailed(
            "MMS alignment source bounds do not match the source WAV"
        )
    flattened: list[CharacterSpan] = []
    previous_end = 0.0
    for word_index, word_span in enumerate(alignment.word_spans):
        if (
            word_span.word_index != word_index
            or not word_span.characters
            or "".join(item.character for item in word_span.characters)
            != word_span.word
        ):
            raise PhraseRenderFailed("MMS word/character coverage is incomplete")
        for character_index, span in enumerate(word_span.characters):
            if (
                span.word != word_span.word
                or span.word_index != word_index
                or span.character_index != character_index
                or not all(
                    isfinite(value)
                    for value in (span.start_seconds, span.end_seconds, span.score)
                )
                or span.start_seconds < previous_end
                or span.end_seconds <= span.start_seconds
                or span.end_seconds > source_duration_seconds
            ):
                raise PhraseRenderFailed(
                    "MMS character spans must be finite, ordered, complete, and in bounds"
                )
            flattened.append(span)
            previous_end = span.end_seconds
        if (
            word_span.start_seconds != word_span.characters[0].start_seconds
            or word_span.end_seconds != word_span.characters[-1].end_seconds
            or not isfinite(word_span.score)
        ):
            raise PhraseRenderFailed(
                "MMS word span does not match its character bounds"
            )
    if tuple(flattened) != alignment.character_spans:
        raise PhraseRenderFailed(
            "MMS phrase character coverage is incomplete or duplicated"
        )


def _map_word_onsets(
    word_span: WordSpan,
    targets: tuple[SyllableTarget, ...],
) -> tuple[tuple[float, ...], tuple[int | None, ...], str]:
    count = len(targets)
    vowel_starts = [match.start() for match in _VOWEL_GROUPS.finditer(word_span.word)]
    if (
        len(vowel_starts) == count + 1
        and word_span.word.endswith("e")
        and vowel_starts[-1] == len(word_span.word) - 1
    ):
        vowel_starts.pop()
    if len(vowel_starts) == count:
        indices = (0, *vowel_starts[1:])
        return (
            tuple(word_span.characters[index].start_seconds for index in indices),
            tuple(indices),
            "orthographic_vowel_groups",
        )

    weighted_indices = _phoneme_weighted_indices(
        character_count=len(word_span.characters),
        phoneme_counts=tuple(len(target.phonemes) for target in targets),
    )
    if weighted_indices is not None:
        return (
            tuple(
                word_span.characters[index].start_seconds for index in weighted_indices
            ),
            tuple(weighted_indices),
            "phoneme_weighted_character",
        )

    duration = word_span.end_seconds - word_span.start_seconds
    if not isfinite(duration) or duration <= 0:
        raise PhraseRenderFailed(
            f"MMS word duration cannot map planned syllables for '{word_span.word}'"
        )
    times = tuple(
        word_span.start_seconds + duration * index / count for index in range(count)
    )
    return times, tuple(None for _ in targets), "word_duration_proportional"


def _phoneme_weighted_indices(
    *,
    character_count: int,
    phoneme_counts: tuple[int, ...],
) -> tuple[int, ...] | None:
    if character_count < len(phoneme_counts) or any(
        count <= 0 for count in phoneme_counts
    ):
        return None
    total = sum(phoneme_counts)
    indices = [0]
    cumulative = 0
    for count in phoneme_counts[:-1]:
        cumulative += count
        index = int(cumulative * character_count / total)
        if index <= indices[-1] or index >= character_count:
            return None
        indices.append(index)
    return tuple(indices)


def _planned_nucleus(phonemes: Sequence[str]) -> str:
    for phoneme in phonemes:
        if phoneme[-1:].isdigit():
            return phoneme
    return phonemes[0] if phonemes else "unknown"


def normalize_mms_transcript(transcript: str) -> tuple[str, ...]:
    """Normalize to lowercase ASCII letters, deleting internal apostrophes."""
    if not isinstance(transcript, str):
        raise PhraseRenderFailed("MMS transcript must be text")
    ascii_text = (
        unicodedata.normalize("NFKD", transcript)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    if any(character.isdigit() for character in ascii_text):
        raise PhraseRenderFailed("MMS transcript normalization does not support digits")
    words = tuple(
        match.group(0).replace("'", "") for match in _ASCII_WORDS.finditer(ascii_text)
    )
    if not words or any(not word for word in words):
        raise PhraseRenderFailed("MMS transcript is empty after ASCII normalization")
    return words


class MmsForcedAligner:
    """Reuse one torchaudio MMS_FA model for known-transcript alignment."""

    def __init__(
        self,
        *,
        runtime: object,
        device: str,
        clock: Callable[[], float],
    ) -> None:
        model = getattr(runtime, "model")
        moved_model = model.to(device)
        self._model = moved_model if moved_model is not None else model
        evaluated_model = self._model.eval()
        if evaluated_model is not None:
            self._model = evaluated_model
        self._tokenizer = getattr(runtime, "tokenizer")
        self._ctc_aligner = getattr(runtime, "aligner")
        self._torch = getattr(runtime, "torch_module")
        self._sample_rate_hz = int(getattr(runtime, "sample_rate_hz"))
        self._identity = str(getattr(runtime, "identity"))
        self._version = str(getattr(runtime, "version"))
        self._labels = frozenset(str(item) for item in getattr(runtime, "labels", ()))
        self._device = device
        self._clock = clock
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        device: str,
        runtime_loader: Callable[[], object] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> "MmsForcedAligner":
        runtime = (runtime_loader or _load_mms_runtime)()
        sample_rate_hz = int(getattr(runtime, "sample_rate_hz"))
        if sample_rate_hz <= 0:
            raise PhraseRenderFailed("MMS bundle sample rate must be positive")
        return cls(runtime=runtime, device=device, clock=clock)

    def align(self, source_wav: Path, transcript: str) -> MmsAlignmentResult:
        words = normalize_mms_transcript(transcript)
        if self._labels:
            unsupported = sorted(set("".join(words)) - self._labels)
            if unsupported:
                raise PhraseRenderFailed(
                    f"MMS transcript contains unsupported labels: {''.join(unsupported)}"
                )

        waveform = _load_resampled_waveform(
            Path(source_wav),
            target_sample_rate_hz=self._sample_rate_hz,
            torch_module=self._torch,
        )
        started = self._clock()
        with self._lock:
            with self._torch.inference_mode():
                tokens = self._tokenizer(list(words))
                model_output = self._model(waveform.tensor.to(self._device))
                emission = model_output[0]
                aligned_words = self._ctc_aligner(emission[0], tokens)
        alignment_time_ms = max(0.0, (self._clock() - started) * 1000.0)

        emission_frames = int(emission.shape[1])
        if emission_frames <= 0:
            raise PhraseRenderFailed("MMS model emitted no alignment frames")
        duration_seconds = waveform.inference_duration_seconds
        seconds_per_emission_frame = duration_seconds / emission_frames
        character_spans, word_spans = _materialize_spans(
            words,
            tokens,
            aligned_words,
            seconds_per_emission_frame=seconds_per_emission_frame,
            duration_seconds=duration_seconds,
        )
        warnings = tuple(
            f"low MMS CTC score for {span.word}[{span.character_index}]={span.score:.3f}"
            for span in character_spans
            if span.score < _LOW_CONFIDENCE_SCORE
        )
        return MmsAlignmentResult(
            normalized_transcript=" ".join(words),
            character_spans=character_spans,
            word_spans=word_spans,
            duration_seconds=duration_seconds,
            alignment_time_ms=alignment_time_ms,
            aligner_identity=self._identity,
            aligner_version=self._version,
            warnings=warnings,
            source_sample_rate_hz=waveform.source_sample_rate_hz,
            source_frame_count=waveform.source_frame_count,
            inference_sample_rate_hz=waveform.inference_sample_rate_hz,
            inference_frame_count=waveform.inference_frame_count,
            emission_frame_count=emission_frames,
        )

    def warmup(self, source_wav: Path, transcript: str) -> Mapping[str, object]:
        """Execute one real resident acoustic and CTC alignment pass."""
        result = self.align(Path(source_wav), transcript)
        return MappingProxyType(
            {
                "aligner": self._identity,
                "version": self._version,
                "device": self._device,
                "sample_rate_hz": self._sample_rate_hz,
                "aligned": True,
                "normalized_transcript": result.normalized_transcript,
                "alignment_time_ms": result.alignment_time_ms,
                "confidence": result.confidence,
            }
        )


def _load_mms_runtime() -> object:
    torch_module = importlib.import_module("torch")
    torchaudio_module = importlib.import_module("torchaudio")
    bundle = torchaudio_module.pipelines.MMS_FA
    version = getattr(torchaudio_module, "__version__", "unknown")
    return SimpleNamespace(
        model=bundle.get_model(),
        tokenizer=bundle.get_tokenizer(),
        aligner=bundle.get_aligner(),
        torch_module=torch_module,
        sample_rate_hz=int(bundle.sample_rate),
        identity="torchaudio.pipelines.MMS_FA",
        version=f"torchaudio {version} / MMS_FA",
        labels=tuple(bundle.get_labels()),
    )


def _load_resampled_waveform(
    path: Path,
    *,
    target_sample_rate_hz: int,
    torch_module: object,
) -> _MmsInferenceWaveform:
    try:
        source_sample_rate_hz, samples = wavfile.read(path)
    except Exception as exc:
        raise PhraseRenderFailed(
            f"unable to read source WAV for MMS alignment: {path}"
        ) from exc
    if source_sample_rate_hz <= 0:
        raise PhraseRenderFailed("MMS source WAV sample rate must be positive")
    mono = _to_mono_float32(samples)
    if mono.size == 0:
        raise PhraseRenderFailed("MMS source WAV must not be empty")
    if not np.isfinite(mono).all():
        raise PhraseRenderFailed("MMS source WAV must contain only finite samples")
    source_frame_count = int(mono.size)
    duration_seconds = source_frame_count / source_sample_rate_hz
    if source_sample_rate_hz != target_sample_rate_hz:
        divisor = gcd(source_sample_rate_hz, target_sample_rate_hz)
        mono = resample_poly(
            mono,
            target_sample_rate_hz // divisor,
            source_sample_rate_hz // divisor,
        ).astype(np.float32, copy=False)
    expected_frames = round(duration_seconds * target_sample_rate_hz)
    if mono.size > expected_frames:
        mono = mono[:expected_frames]
    elif mono.size < expected_frames:
        mono = np.pad(mono, (0, expected_frames - mono.size))
    tensor = getattr(torch_module, "from_numpy")(
        np.ascontiguousarray(mono, dtype=np.float32)
    ).unsqueeze(0)
    return _MmsInferenceWaveform(
        tensor=tensor,
        source_sample_rate_hz=int(source_sample_rate_hz),
        source_frame_count=source_frame_count,
        inference_sample_rate_hz=target_sample_rate_hz,
        inference_frame_count=int(mono.size),
    )


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise PhraseRenderFailed("MMS source WAV must be mono")
    if array.dtype.kind in {"i", "u"}:
        limits = np.iinfo(array.dtype)
        scale = max(abs(limits.min), limits.max)
        return array.astype(np.float32) / np.float32(scale)
    return array.astype(np.float32, copy=False)


def _materialize_spans(
    words: tuple[str, ...],
    expected_tokens: Sequence[Sequence[object]],
    aligned_words: Sequence[Sequence[object]],
    *,
    seconds_per_emission_frame: float,
    duration_seconds: float,
) -> tuple[tuple[CharacterSpan, ...], tuple[WordSpan, ...]]:
    if len(expected_tokens) != len(words) or len(aligned_words) != len(words):
        raise PhraseRenderFailed(
            f"MMS word coverage mismatch: expected {len(words)}, got {len(aligned_words)}"
        )
    all_characters: list[CharacterSpan] = []
    word_spans: list[WordSpan] = []
    previous_end = 0.0
    for word_index, (word, word_tokens, raw_spans) in enumerate(
        zip(words, expected_tokens, aligned_words)
    ):
        if len(word_tokens) != len(word) or len(raw_spans) != len(word):
            raise PhraseRenderFailed(
                f"MMS character coverage mismatch for {word}: expected {len(word)}, got {len(raw_spans)}"
            )
        characters: list[CharacterSpan] = []
        for character_index, (character, expected_token, raw_span) in enumerate(
            zip(word, word_tokens, raw_spans)
        ):
            if int(getattr(raw_span, "token")) != int(expected_token):
                raise PhraseRenderFailed(
                    f"MMS token coverage mismatch for {word}[{character_index}]"
                )
            start_seconds = (
                float(getattr(raw_span, "start")) * seconds_per_emission_frame
            )
            end_seconds = float(getattr(raw_span, "end")) * seconds_per_emission_frame
            score = float(getattr(raw_span, "score"))
            if (
                not all(
                    isfinite(value) for value in (start_seconds, end_seconds, score)
                )
                or start_seconds < previous_end
                or end_seconds <= start_seconds
                or end_seconds > duration_seconds
            ):
                raise PhraseRenderFailed(
                    f"MMS produced invalid or non-monotonic character span for {word}"
                )
            span = CharacterSpan(
                word=word,
                word_index=word_index,
                character=character,
                character_index=character_index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                score=score,
            )
            characters.append(span)
            all_characters.append(span)
            previous_end = end_seconds
        word_score = sum(item.score for item in characters) / len(characters)
        word_spans.append(
            WordSpan(
                word=word,
                word_index=word_index,
                start_seconds=characters[0].start_seconds,
                end_seconds=characters[-1].end_seconds,
                score=word_score,
                characters=tuple(characters),
            )
        )
    return tuple(all_characters), tuple(word_spans)
