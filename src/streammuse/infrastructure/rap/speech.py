"""Lazy eSpeak adapter for independently rendered rap syllables."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import ctypes
import ctypes.util
from math import isfinite
import re
import struct
import subprocess
from time import perf_counter
from threading import Lock
from typing import Protocol

import numpy as np
from scipy.signal import resample

from streammuse.application.rap.audio_service import SpeechSynthesizer
from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    PcmAudio,
    ProsodyAnalysis,
    RenderedSyllable,
    SyllableRenderRequest,
)


_PHONEME_EVENT = 7
_WORD_EVENT = 1
_TRIM_THRESHOLD_DBFS = -45.0
_TRIM_PADDING_MS = 5.0


_ARPABET_TO_ESPEAK = {
    "AA": "A:", "AE": "a", "AH": "V", "AO": "O:",
    "AW": "aU", "AY": "aI", "EH": "E", "ER": "3:",
    "EY": "eI", "IH": "I", "IY": "i:", "OW": "oU",
    "OY": "OI", "UH": "U", "UW": "u:",
    "B": "b", "CH": "tS", "D": "d", "DH": "D", "F": "f",
    "G": "g", "HH": "h", "JH": "dZ", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "N", "P": "p", "R": "r",
    "S": "s", "SH": "S", "T": "t", "TH": "T", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "Z",
}


class CommandRunner(Protocol):
    """Minimal process boundary, injectable for tests and alternate backends."""

    def run(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]: ...


class _SubprocessCommandRunner:
    def __init__(self, *, timeout_s: float, max_output_bytes: int) -> None:
        self._timeout_s = timeout_s
        self._max_output_bytes = max_output_bytes

    def run(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self._timeout_s,
        )
        stdout = result.stdout or b""
        stderr = result.stderr or b""
        if len(stdout) > self._max_output_bytes or len(stderr) > self._max_output_bytes:
            return subprocess.CompletedProcess(command, 1, b"", b"eSpeak output exceeded configured limit")
        return result


@dataclass(frozen=True)
class _RenderKey:
    voice: str
    speed_wpm: int
    pitch: int
    render_mode: str
    render_text: str


@dataclass(frozen=True)
class EspeakEventRecord:
    event_type: int
    text_position: int
    length: int
    sample: int
    phoneme: str


@dataclass(frozen=True)
class RenderedContinuousPhrase:
    audio: PcmAudio
    onset_frames: tuple[int, ...]
    synthesis_latency_ms: float
    pronunciation_source: str = "espeak_continuous_events"
    warnings: tuple[AudioWarning, ...] = ()


class _EspeakEventId(ctypes.Union):
    _fields_ = [
        ("number", ctypes.c_int),
        ("name", ctypes.c_char_p),
        ("string", ctypes.c_char * 8),
    ]


class _EspeakEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("unique_identifier", ctypes.c_uint),
        ("text_position", ctypes.c_int),
        ("length", ctypes.c_int),
        ("audio_position", ctypes.c_int),
        ("sample", ctypes.c_int),
        ("user_data", ctypes.c_void_p),
        ("id", _EspeakEventId),
    ]


_ESPEAK_CALLBACK = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_short),
    ctypes.c_int,
    ctypes.POINTER(_EspeakEvent),
)


class EspeakEventPhraseSynthesizer:
    """Synthesize one connected phrase and retain eSpeak phoneme timestamps."""

    def __init__(
        self,
        analyzer,
        *,
        output_sample_rate_hz: int = 48_000,
        library_path: str | None = None,
    ) -> None:
        if output_sample_rate_hz <= 0:
            raise ValueError("output sample rate must be positive")
        resolved = library_path or ctypes.util.find_library("espeak-ng")
        if resolved is None:
            resolved = "/opt/homebrew/lib/libespeak-ng.dylib"
        try:
            self._library = ctypes.CDLL(resolved)
        except OSError as exc:
            raise OSError("the espeak-ng shared library is required for adaptive audio") from exc
        self._analyzer = analyzer
        self._output_sample_rate_hz = output_sample_rate_hz
        self._lock = Lock()
        self._chunks: list[np.ndarray] = []
        self._events: list[EspeakEventRecord] = []

        @_ESPEAK_CALLBACK
        def callback(
            wav: ctypes.POINTER(ctypes.c_short),
            sample_count: int,
            event_pointer: ctypes.POINTER(_EspeakEvent),
        ) -> int:
            if wav and sample_count > 0:
                self._chunks.append(
                    np.ctypeslib.as_array(wav, shape=(sample_count,)).copy()
                )
            if event_pointer:
                index = 0
                while True:
                    event = event_pointer[index]
                    if event.type == 0:
                        break
                    phoneme = ""
                    if event.type == _PHONEME_EVENT:
                        phoneme = bytes(event.id.string).split(b"\0", 1)[0].decode(
                            "utf-8",
                            errors="replace",
                        )
                    self._events.append(
                        EspeakEventRecord(
                            event_type=event.type,
                            text_position=event.text_position,
                            length=event.length,
                            sample=event.sample,
                            phoneme=phoneme,
                        )
                    )
                    index += 1
            return 0

        self._callback = callback
        self._configure_signatures()
        self.source_sample_rate_hz = self._library.espeak_Initialize(2, 0, None, 1)
        if self.source_sample_rate_hz <= 0:
            raise RuntimeError(f"eSpeak initialization failed: {self.source_sample_rate_hz}")
        self._library.espeak_SetSynthCallback(self._callback)

    def synthesize(
        self,
        text: str,
        *,
        voice: str,
        speed_wpm: int,
        pitch: int,
    ) -> RenderedContinuousPhrase:
        with self._lock:
            self._chunks.clear()
            self._events.clear()
            if self._library.espeak_SetVoiceByName(voice.encode("utf-8")) != 0:
                raise RuntimeError(f"eSpeak voice selection failed: {voice}")
            if self._library.espeak_SetParameter(1, speed_wpm, 0) != 0:
                raise RuntimeError(f"eSpeak rate selection failed: {speed_wpm}")
            if self._library.espeak_SetParameter(3, pitch, 0) != 0:
                raise RuntimeError(f"eSpeak pitch selection failed: {pitch}")

            encoded = text.encode("utf-8") + b"\0"
            buffer = ctypes.create_string_buffer(encoded)
            started = perf_counter()
            result = self._library.espeak_Synth(
                ctypes.cast(buffer, ctypes.c_void_p),
                len(encoded),
                0,
                1,
                0,
                1,
                None,
                None,
            )
            if result != 0:
                raise RuntimeError(f"eSpeak synthesis failed: {result}")
            self._library.espeak_Synchronize()
            latency_ms = (perf_counter() - started) * 1000.0
            if not self._chunks:
                raise RuntimeError("eSpeak emitted no phrase audio")

            source = np.concatenate(self._chunks).astype(np.float32) / np.float32(32768.0)
            output_frames = round(
                len(source) * self._output_sample_rate_hz / self.source_sample_rate_hz
            )
            output = resample(source, output_frames).astype(np.float32)
            phone_events = tuple(
                event
                for event in self._events
                if event.event_type == _PHONEME_EVENT
                and event.phoneme
                and not event.phoneme.startswith("_")
            )
            if not phone_events:
                raise RuntimeError("eSpeak emitted no phrase phoneme events")
            first_phone_frame = _convert_sample_rate_frame(
                phone_events[0].sample,
                self.source_sample_rate_hz,
                self._output_sample_rate_hz,
            )
            active = np.flatnonzero(
                np.abs(output) >= 10 ** (_TRIM_THRESHOLD_DBFS / 20.0)
            )
            if active.size == 0:
                raise RuntimeError("eSpeak emitted silent phrase audio")
            padding = round(_TRIM_PADDING_MS / 1000.0 * self._output_sample_rate_hz)
            end_frame = min(len(output), int(active[-1]) + padding + 1)
            cropped = output[first_phone_frame:end_frame].copy()
            if not len(cropped):
                raise RuntimeError("eSpeak phrase is empty after onset trimming")
            analysis = self._analyzer.analyze(text)
            onset_frames = map_espeak_events_to_syllable_onsets(
                analysis,
                tuple(self._events),
                source_sample_rate_hz=self.source_sample_rate_hz,
                output_sample_rate_hz=self._output_sample_rate_hz,
                crop_start_output_frame=first_phone_frame,
            )
            if onset_frames[0] != 0 or onset_frames[-1] >= len(cropped):
                raise RuntimeError("eSpeak phrase onsets lie outside cropped audio")
            audio = PcmAudio(
                AudioFormat(self._output_sample_rate_hz, 1, 4),
                len(cropped),
                cropped.tobytes(),
            )
            return RenderedContinuousPhrase(
                audio=audio,
                onset_frames=onset_frames,
                synthesis_latency_ms=latency_ms,
            )

    def _configure_signatures(self) -> None:
        library = self._library
        library.espeak_Initialize.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        library.espeak_Initialize.restype = ctypes.c_int
        library.espeak_SetSynthCallback.argtypes = [_ESPEAK_CALLBACK]
        library.espeak_SetSynthCallback.restype = None
        library.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        library.espeak_SetVoiceByName.restype = ctypes.c_int
        library.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        library.espeak_SetParameter.restype = ctypes.c_int
        library.espeak_Synth.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_void_p,
        ]
        library.espeak_Synth.restype = ctypes.c_int
        library.espeak_Synchronize.argtypes = []
        library.espeak_Synchronize.restype = ctypes.c_int


def map_espeak_events_to_syllable_onsets(
    analysis: ProsodyAnalysis,
    events: tuple[EspeakEventRecord, ...],
    *,
    source_sample_rate_hz: int,
    output_sample_rate_hz: int,
    crop_start_output_frame: int,
) -> tuple[int, ...]:
    groups: list[list] = []
    for syllable in analysis.syllables:
        if syllable.index_in_word == 0:
            groups.append([])
        if not groups:
            raise RuntimeError("prosody analysis begins inside a word")
        groups[-1].append(syllable)
    word_events = tuple(event for event in events if event.event_type == _WORD_EVENT)
    if len(word_events) != len(groups):
        raise RuntimeError(
            f"eSpeak word-event mismatch: {len(word_events)} != {len(groups)}"
        )

    onsets = []
    for syllable_group, word_event in zip(groups, word_events, strict=True):
        phones = tuple(
            event
            for event in events
            if event.event_type == _PHONEME_EVENT
            and event.text_position == word_event.text_position
            and event.phoneme
            and not event.phoneme.startswith("_")
        )
        vowel_indices = tuple(
            index
            for index, event in enumerate(phones)
            if _is_espeak_event_vowel(event.phoneme)
        )
        if len(vowel_indices) != len(syllable_group):
            word = syllable_group[0].word
            raise RuntimeError(
                f"eSpeak vowel-event mismatch for {word!r}: "
                f"{len(vowel_indices)} != {len(syllable_group)}"
            )
        for syllable_index, vowel_index in enumerate(vowel_indices):
            onset_index = 0 if syllable_index == 0 else vowel_indices[syllable_index - 1] + 1
            onset_index = min(onset_index, vowel_index)
            source_frame = _convert_sample_rate_frame(
                phones[onset_index].sample,
                source_sample_rate_hz,
                output_sample_rate_hz,
            )
            onsets.append(max(0, source_frame - crop_start_output_frame))
    if len(onsets) != len(analysis.syllables):
        raise RuntimeError(
            f"continuous phrase onset mismatch: {len(onsets)} != {len(analysis.syllables)}"
        )
    if any(right <= left for left, right in zip(onsets, onsets[1:])):
        raise RuntimeError(f"continuous phrase onsets are not strictly increasing: {onsets}")
    return tuple(onsets)


def _convert_sample_rate_frame(frame: int, source_rate: int, target_rate: int) -> int:
    return round(frame * target_rate / source_rate)


def _is_espeak_event_vowel(phoneme: str) -> bool:
    core = phoneme.lstrip("',")
    return core.startswith("0") or _is_espeak_vowel(phoneme)


def arpabet_syllable_to_espeak(phonemes: tuple[str, ...]) -> tuple[str, ...]:
    """Map one CMUdict syllable to eSpeak phonemes with leading syllable stress."""
    mapped: list[str] = []
    stress_marker = ""
    for phoneme in phonemes:
        match = re.fullmatch(r"([A-Z]+)([012])?", phoneme)
        if match is None or match.group(1) not in _ARPABET_TO_ESPEAK:
            raise ValueError(f"Unsupported ARPAbet phoneme: {phoneme}")
        if not stress_marker:
            stress_marker = {"1": "'", "2": ","}.get(match.group(2), "")
        mapped.append(_ARPABET_TO_ESPEAK[match.group(1)])
    if mapped and stress_marker:
        mapped[0] = stress_marker + mapped[0]
    return tuple(mapped)


class EspeakPhonemeSynthesizer(SpeechSynthesizer):
    """Render CMUdict syllables first, with best-effort eSpeak fallbacks."""

    def __init__(
        self,
        command: str = "espeak-ng",
        runner: CommandRunner | None = None,
        *,
        cache_size: int = 256,
        command_timeout_s: float = 2.0,
        max_output_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size <= 0:
            raise ValueError("cache_size must be a positive integer")
        if not isfinite(command_timeout_s) or command_timeout_s <= 0:
            raise ValueError("command_timeout_s must be positive and finite")
        if not isinstance(max_output_bytes, int) or isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")
        self._command = command
        self._runner = runner or _SubprocessCommandRunner(
            timeout_s=command_timeout_s,
            max_output_bytes=max_output_bytes,
        )
        self._cache_size = cache_size
        self._cache_lock = Lock()
        self._pcm_cache: OrderedDict[_RenderKey, PcmAudio] = OrderedDict()

    def synthesize(self, request: SyllableRenderRequest) -> RenderedSyllable:
        started = perf_counter()
        if request.phonemes:
            try:
                phonemes = arpabet_syllable_to_espeak(request.phonemes)
            except ValueError:
                phonemes = ()
            else:
                audio = self._render(request, "phonemes", "".join(phonemes))
                if audio is not None:
                    return self._result(request, audio, phonemes, "cmudict_arpabet", started, ())

        warnings = [self._fallback_warning(request)]
        g2p_tokens = self._g2p_tokens(request.word, request.voice)
        selected = _select_syllable_tokens(g2p_tokens, request.syllable_count, request.index_in_word)
        if selected is not None:
            audio = self._render(request, "phonemes", "".join(selected))
            if audio is not None:
                return self._result(request, audio, selected, "espeak_g2p", started, tuple(warnings))

        fragment = _grapheme_fragment(request.word, request.syllable_count, request.index_in_word)
        audio = self._render(request, "text", fragment)
        if audio is not None:
            return self._result(request, audio, (fragment,), "grapheme_fragment", started, tuple(warnings))

        warnings.append(self._synthesis_failed_warning(request))
        return self._result(
            request,
            PcmAudio(AudioFormat(channels=1), 0, b""),
            (),
            "synthesis_failed",
            started,
            tuple(warnings),
        )

    def _result(
        self,
        request: SyllableRenderRequest,
        audio: PcmAudio,
        renderer_phonemes: tuple[str, ...],
        pronunciation_source: str,
        started: float,
        warnings: tuple[AudioWarning, ...],
    ) -> RenderedSyllable:
        return RenderedSyllable(
            request=request,
            audio=audio,
            renderer_phonemes=renderer_phonemes,
            pronunciation_source=pronunciation_source,
            synthesis_latency_ms=(perf_counter() - started) * 1000,
            warnings=warnings,
        )

    def _g2p_tokens(self, word: str, voice: str) -> tuple[str, ...]:
        try:
            result = self._runner.run((self._command, "-q", "-x", "--sep=_", "-v", voice, word))
        except (OSError, subprocess.SubprocessError):
            return ()
        if result.returncode != 0:
            return ()
        return tuple(token for token in result.stdout.decode("ascii", errors="ignore").strip().split("_") if token)

    def _render(self, request: SyllableRenderRequest, render_mode: str, render_text: str) -> PcmAudio | None:
        key = _RenderKey(request.voice, request.speed_wpm, request.pitch, render_mode, render_text)
        with self._cache_lock:
            cached = self._pcm_cache.get(key)
            if cached is not None:
                self._pcm_cache.move_to_end(key)
                return cached
        if render_mode == "phonemes":
            rendered_input = f"[[{render_text}]]"
        else:
            rendered_input = render_text
        command = (
            self._command,
            "-D",
            "-z",
            "-v",
            request.voice,
            "-s",
            str(request.speed_wpm),
            "-p",
            str(request.pitch),
            "--stdout",
            rendered_input,
        )
        try:
            result = self._runner.run(command)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        try:
            decoded = _decode_mono_float32_wav(result.stdout)
        except (EOFError, ValueError):
            return None
        with self._cache_lock:
            self._pcm_cache[key] = decoded
            self._pcm_cache.move_to_end(key)
            while len(self._pcm_cache) > self._cache_size:
                self._pcm_cache.popitem(last=False)
        return decoded

    @staticmethod
    def _fallback_warning(request: SyllableRenderRequest) -> AudioWarning:
        return AudioWarning(
            code=AudioWarningCode.PRONUNCIATION_FALLBACK,
            severity=AudioWarningSeverity.WARNING,
            message="CMUdict phonemes unavailable; using best-effort pronunciation fallback",
            bar=request.bar,
            slot_index=request.slot_index,
            word=request.word,
            action="fallback",
        )

    @staticmethod
    def _synthesis_failed_warning(request: SyllableRenderRequest) -> AudioWarning:
        return AudioWarning(
            code=AudioWarningCode.SYNTHESIS_FAILED,
            severity=AudioWarningSeverity.ERROR,
            message="eSpeak could not synthesize the requested syllable",
            bar=request.bar,
            slot_index=request.slot_index,
            word=request.word,
            action="empty_pcm",
        )


def _select_syllable_tokens(
    tokens: tuple[str, ...], syllable_count: int, index_in_word: int
) -> tuple[str, ...] | None:
    if syllable_count < 1 or not 0 <= index_in_word < syllable_count:
        return None
    groups: list[list[str]] = [[]]
    vowel_count = 0
    for token in tokens:
        groups[-1].append(token)
        if _is_espeak_vowel(token):
            vowel_count += 1
            if vowel_count < syllable_count:
                groups.append([])
    if vowel_count != syllable_count or any(not group for group in groups):
        return None
    return tuple(groups[index_in_word])


def _is_espeak_vowel(token: str) -> bool:
    core = token.lstrip("',")
    return any(symbol in core for symbol in "aeiouAEIOUV@3")


def _grapheme_fragment(word: str, syllable_count: int, index_in_word: int) -> str:
    spelling = re.sub(r"[^a-z]", "", word.lower())
    if syllable_count < 1 or not spelling:
        return spelling
    boundaries = [match.end() for match in re.finditer(r"[aeiouy]+", spelling)]
    fragments: list[str] = []
    start = 0
    for boundary in boundaries[: max(syllable_count - 1, 0)]:
        fragments.append(spelling[start:boundary])
        start = boundary
    fragments.append(spelling[start:])
    while len(fragments) < syllable_count:
        fragment = fragments.pop()
        split_at = max(1, len(fragment) // 2)
        fragments.extend((fragment[:split_at], fragment[split_at:]))
    return fragments[min(index_in_word, len(fragments) - 1)]


def _decode_mono_float32_wav(wav_bytes: bytes) -> PcmAudio:
    channels, width, sample_rate_hz, raw = _read_pcm_wav_payload(wav_bytes)
    if channels != 1:
        raise ValueError(f"Expected mono WAV output, received {channels} channels")
    if width not in (1, 2, 3, 4):
        raise ValueError(f"Unsupported WAV sample width: {width}")
    if sample_rate_hz <= 0:
        raise ValueError(f"Invalid WAV sample rate: {sample_rate_hz}")
    bytes_per_frame = channels * width
    if not raw or len(raw) % bytes_per_frame:
        raise ValueError("WAV PCM payload is empty or not aligned to complete frames")
    frames = len(raw) // bytes_per_frame
    samples = _pcm_samples_to_float32(raw, width).reshape(frames, channels)
    mono = samples.mean(axis=1, dtype=np.float32)
    return PcmAudio(AudioFormat(sample_rate_hz, 1), frames, mono.astype(np.float32).tobytes())


_ESPEAK_STREAMING_RIFF_SIZE = 2_147_479_588
_ESPEAK_STREAMING_DATA_SIZE = 2_147_479_552


def _read_pcm_wav_payload(wav_bytes: bytes) -> tuple[int, int, int, memoryview]:
    """Validate a complete RIFF/WAVE payload and return its declared PCM data."""
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("Expected RIFF/WAVE output")

    riff_size = struct.unpack_from("<I", wav_bytes, 4)[0]
    streaming_espeak = riff_size == _ESPEAK_STREAMING_RIFF_SIZE
    if not streaming_espeak and riff_size != len(wav_bytes) - 8:
        raise ValueError("RIFF size does not match the complete WAV payload")

    cursor = 12
    format_fields: tuple[int, int, int, int, int, int] | None = None
    data_payload: memoryview | None = None
    payload = memoryview(wav_bytes)
    while cursor < len(wav_bytes):
        if len(wav_bytes) - cursor < 8:
            raise ValueError("WAV contains trailing bytes outside a complete chunk")
        chunk_id = wav_bytes[cursor : cursor + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, cursor + 4)[0]
        data_offset = cursor + 8

        if chunk_id == b"data" and streaming_espeak and chunk_size == _ESPEAK_STREAMING_DATA_SIZE:
            if data_payload is not None:
                raise ValueError("WAV contains multiple data chunks")
            # Homebrew eSpeak 1.52 emits these sentinel RIFF/data lengths while
            # stdout contains the complete PCM stream through EOF.
            data_payload = payload[data_offset:]
            cursor = len(wav_bytes)
            continue

        data_end = data_offset + chunk_size
        padded_end = data_end + (chunk_size & 1)
        if padded_end > len(wav_bytes):
            raise EOFError("WAV chunk exceeds the emitted payload")

        if chunk_id == b"fmt ":
            if format_fields is not None:
                raise ValueError("WAV contains multiple fmt chunks")
            if chunk_size < 16:
                raise ValueError("WAV fmt chunk is incomplete")
            format_fields = struct.unpack_from("<HHIIHH", wav_bytes, data_offset)
        elif chunk_id == b"data":
            if data_payload is not None:
                raise ValueError("WAV contains multiple data chunks")
            data_payload = payload[data_offset:data_end]

        cursor = padded_end

    if format_fields is None or data_payload is None:
        raise ValueError("WAV requires one fmt chunk and one data chunk")

    format_tag, channels, sample_rate_hz, byte_rate, block_align, bits_per_sample = format_fields
    if format_tag != 1:
        raise ValueError("compressed WAV output is unsupported")
    if bits_per_sample % 8:
        raise ValueError("WAV sample width must be a whole number of bytes")
    sample_width = bits_per_sample // 8
    if block_align != channels * sample_width:
        raise ValueError("WAV block alignment does not match its format")
    if byte_rate != sample_rate_hz * block_align:
        raise ValueError("WAV byte rate does not match its format")
    return channels, sample_width, sample_rate_hz, data_payload


def _pcm_samples_to_float32(raw: bytes | memoryview, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = packed[:, 0].astype(np.int32) | (packed[:, 1].astype(np.int32) << 8) | (packed[:, 2].astype(np.int32) << 16)
        values[values & 0x800000 != 0] -= 1 << 24
        return values.astype(np.float32) / (1 << 23)
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / (1 << 31)
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")
