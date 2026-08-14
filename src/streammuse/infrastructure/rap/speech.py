"""Lazy eSpeak adapter for independently rendered rap syllables."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import re
import subprocess
from time import perf_counter
from typing import Protocol
import wave

import numpy as np

from streammuse.application.rap.audio_service import SpeechSynthesizer
from streammuse.domain.rap import (
    AudioFormat,
    AudioWarning,
    AudioWarningCode,
    AudioWarningSeverity,
    PcmAudio,
    RenderedSyllable,
    SyllableRenderRequest,
)


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
    def run(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


@dataclass(frozen=True)
class _RenderKey:
    voice: str
    speed_wpm: int
    pitch: int
    render_mode: str
    render_text: str


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

    def __init__(self, command: str = "espeak-ng", runner: CommandRunner | None = None) -> None:
        self._command = command
        self._runner = runner or _SubprocessCommandRunner()
        self._pcm_cache: dict[_RenderKey, PcmAudio] = {}

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
        except OSError:
            return ()
        if result.returncode != 0:
            return ()
        return tuple(token for token in result.stdout.decode("ascii", errors="ignore").strip().split("_") if token)

    def _render(self, request: SyllableRenderRequest, render_mode: str, render_text: str) -> PcmAudio | None:
        key = _RenderKey(request.voice, request.speed_wpm, request.pitch, render_mode, render_text)
        cached = self._pcm_cache.get(key)
        if cached is not None:
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
        except OSError:
            return None
        if result.returncode != 0:
            return None
        try:
            decoded = _decode_mono_float32_wav(result.stdout)
        except (EOFError, ValueError, wave.Error):
            return None
        self._pcm_cache[key] = decoded
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
    with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError("compressed WAV output is unsupported")
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        sample_rate_hz = wav_file.getframerate()
        raw = wav_file.readframes(frames)
    samples = _pcm_samples_to_float32(raw, width).reshape(frames, channels)
    mono = samples.mean(axis=1, dtype=np.float32)
    return PcmAudio(AudioFormat(sample_rate_hz, 1), frames, mono.astype(np.float32).tobytes())


def _pcm_samples_to_float32(raw: bytes, sample_width: int) -> np.ndarray:
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
