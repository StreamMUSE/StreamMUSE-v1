"""Tests for the replaceable eSpeak phoneme speech adapter."""

from __future__ import annotations

from io import BytesIO
import subprocess
import wave

import numpy as np
import pytest

from streammuse.domain.rap import AudioWarningCode, SyllableRenderRequest
from streammuse.infrastructure.rap.speech import EspeakPhonemeSynthesizer, arpabet_syllable_to_espeak


class FakeEspeakRunner:
    def __init__(
        self,
        *,
        wav_bytes: bytes = b"",
        phoneme_stdout: str = "",
        fail: bool = False,
        unavailable: bool = False,
    ) -> None:
        self.wav_bytes = wav_bytes
        self.phoneme_stdout = phoneme_stdout
        self.fail = fail
        self.unavailable = unavailable
        self.commands: list[tuple[str, ...]] = []

    @property
    def wav_command_count(self) -> int:
        return sum("--stdout" in command for command in self.commands)

    def run(self, command: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(command)
        if self.unavailable:
            raise FileNotFoundError(command[0])
        if self.fail:
            return subprocess.CompletedProcess(command, 1, b"", b"failed")
        if "-x" in command:
            return subprocess.CompletedProcess(command, 0, self.phoneme_stdout.encode("ascii"), b"")
        return subprocess.CompletedProcess(command, 0, self.wav_bytes, b"")


def wav_bytes(*, frames: int, sample_rate_hz: int = 24_000) -> bytes:
    samples = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    pcm = np.round(samples * 32767).astype("<i2").tobytes()
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(pcm)
    return buffer.getvalue()


def cmu_request(phonemes: tuple[str, ...]) -> SyllableRenderRequest:
    return SyllableRenderRequest(
        bar=2,
        slot_index=3,
        word="move",
        index_in_word=0,
        syllable_count=1,
        phonemes=phonemes,
        stress=1,
        analysis_source="cmudict",
        voice="en-us",
        speed_wpm=175,
        pitch=50,
    )


def oov_request(word: str, *, index_in_word: int = 0, syllable_count: int = 2) -> SyllableRenderRequest:
    return SyllableRenderRequest(
        bar=2,
        slot_index=3,
        word=word,
        index_in_word=index_in_word,
        syllable_count=syllable_count,
        phonemes=(),
        stress=0,
        analysis_source="heuristic",
        voice="en-us",
        speed_wpm=175,
        pitch=50,
    )


@pytest.mark.parametrize(
    ("arpabet", "expected"),
    [
        (("M", "UW1", "V"), ("'m", "u:", "v")),
        (("IH0", "NG"), ("I", "N")),
        (("S", "IH1", "T"), ("'s", "I", "t")),
        (("TH", "R", "UW2"), (",T", "r", "u:")),
    ],
)
def test_arpabet_syllable_maps_to_espeak_with_syllable_stress(
    arpabet: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    assert arpabet_syllable_to_espeak(arpabet) == expected


def test_espeak_synthesizer_uses_explicit_phonemes() -> None:
    runner = FakeEspeakRunner(wav_bytes=wav_bytes(frames=240))
    synthesizer = EspeakPhonemeSynthesizer(command="espeak-ng", runner=runner)

    result = synthesizer.synthesize(cmu_request(("M", "UW1", "V")))

    command = runner.commands[0]
    assert "-D" in command
    assert "-z" in command
    assert "-v" in command and command[command.index("-v") + 1] == "en-us"
    assert "-s" in command and command[command.index("-s") + 1] == "175"
    assert "-p" in command and command[command.index("-p") + 1] == "50"
    assert "--stdout" in command
    assert "[['mu:v]]" in " ".join(command)
    assert result.pronunciation_source == "cmudict_arpabet"
    assert result.audio.format.channels == 1
    assert result.audio.format.sample_width_bytes == 4
    assert result.audio.frame_count == 240


def test_missing_cmu_phonemes_use_espeak_g2p_and_warn() -> None:
    runner = FakeEspeakRunner(
        phoneme_stdout="s_t_r_'i:_m_m_j_'u:_z",
        wav_bytes=wav_bytes(frames=240),
    )

    result = EspeakPhonemeSynthesizer(command="espeak-ng", runner=runner).synthesize(oov_request("StreamMUSE"))

    assert result.pronunciation_source == "espeak_g2p"
    assert result.renderer_phonemes == ("s", "t", "r", "'i:")
    assert result.warnings[0].code == AudioWarningCode.PRONUNCIATION_FALLBACK
    assert "-q" in runner.commands[0]
    assert "-x" in runner.commands[0]
    assert "--sep=_" in runner.commands[0]
    assert "-v" in runner.commands[0] and runner.commands[0][runner.commands[0].index("-v") + 1] == "en-us"


def test_g2p_vowel_mismatch_uses_deterministic_grapheme_fragment() -> None:
    runner = FakeEspeakRunner(phoneme_stdout="r_'I_d_m", wav_bytes=wav_bytes(frames=120))

    result = EspeakPhonemeSynthesizer(command="espeak-ng", runner=runner).synthesize(
        oov_request("rhythm", index_in_word=1, syllable_count=2)
    )

    assert result.pronunciation_source == "grapheme_fragment"
    assert result.renderer_phonemes == ("thm",)
    assert result.warnings[0].code == AudioWarningCode.PRONUNCIATION_FALLBACK
    assert runner.commands[-1][-1] == "thm"


def test_identical_phoneme_requests_use_cached_pcm() -> None:
    runner = FakeEspeakRunner(wav_bytes=wav_bytes(frames=240))
    synthesizer = EspeakPhonemeSynthesizer(command="espeak-ng", runner=runner)
    request = cmu_request(("M", "UW1", "V"))

    synthesizer.synthesize(request)
    synthesizer.synthesize(request)

    assert runner.wav_command_count == 1


def test_failed_commands_return_empty_pcm_and_warnings() -> None:
    result = EspeakPhonemeSynthesizer(command="espeak-ng", runner=FakeEspeakRunner(fail=True)).synthesize(
        oov_request("missing")
    )

    assert result.audio.format.channels == 1
    assert result.audio.format.sample_width_bytes == 4
    assert result.audio.frame_count == 0
    assert [warning.code for warning in result.warnings] == [
        AudioWarningCode.PRONUNCIATION_FALLBACK,
        AudioWarningCode.SYNTHESIS_FAILED,
    ]


def test_unavailable_espeak_returns_empty_pcm_and_warnings() -> None:
    result = EspeakPhonemeSynthesizer(
        command="missing-espeak", runner=FakeEspeakRunner(unavailable=True)
    ).synthesize(oov_request("missing"))

    assert result.audio.frame_count == 0
    assert [warning.code for warning in result.warnings] == [
        AudioWarningCode.PRONUNCIATION_FALLBACK,
        AudioWarningCode.SYNTHESIS_FAILED,
    ]
