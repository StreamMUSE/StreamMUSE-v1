from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from streammuse.application.tasks import VoiceInputConfig
from streammuse.domain.tasks import ZipZapZopTask
from streammuse.infrastructure.voice import FasterWhisperRecognizer


def test_opt_in_local_faster_whisper_model_loads_warms_and_transcribes() -> None:
    model = os.environ.get("STREAMMUSE_TEST_VOICE_MODEL")
    if not model:
        pytest.skip("set STREAMMUSE_TEST_VOICE_MODEL to run the offline real-model test")

    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(
            model=model,
            model_cache=os.environ.get("STREAMMUSE_TEST_VOICE_MODEL_CACHE"),
            model_revision=os.environ.get("STREAMMUSE_TEST_VOICE_MODEL_REVISION"),
            local_files_only=True,
        )
    )
    try:
        recognizer.start()
        result = recognizer.transcribe(np.zeros(16_000, dtype=np.float32))
    finally:
        recognizer.close()

    assert isinstance(result.text, str)
    assert result.latency_ms >= 0.0
    assert recognizer.provenance["model_load_ms"] is not None
    assert recognizer.provenance["warmup_ms"] is not None


def test_opt_in_prerecorded_game_audio_runs_real_asr_and_task_parser() -> None:
    model = os.environ.get("STREAMMUSE_TEST_VOICE_MODEL")
    audio_path = os.environ.get("STREAMMUSE_TEST_VOICE_AUDIO")
    expected = os.environ.get("STREAMMUSE_TEST_VOICE_EXPECTED")
    if not model or not audio_path or not expected:
        pytest.skip(
            "set STREAMMUSE_TEST_VOICE_MODEL, STREAMMUSE_TEST_VOICE_AUDIO, and "
            "STREAMMUSE_TEST_VOICE_EXPECTED to run the prerecorded speech test"
        )
    audio_file = Path(audio_path).expanduser().resolve()
    if not audio_file.is_file():
        pytest.fail(f"STREAMMUSE_TEST_VOICE_AUDIO is not a file: {audio_file}")

    from faster_whisper.audio import decode_audio

    task = ZipZapZopTask()
    context = task.build_speech_context(task.initial_state(), [])
    recognizer = FasterWhisperRecognizer(
        VoiceInputConfig(
            model=model,
            model_cache=os.environ.get("STREAMMUSE_TEST_VOICE_MODEL_CACHE"),
            model_revision=os.environ.get("STREAMMUSE_TEST_VOICE_MODEL_REVISION"),
            local_files_only=True,
        )
    )
    try:
        recognizer.start()
        result = recognizer.transcribe(
            np.asarray(decode_audio(str(audio_file), sampling_rate=16_000), dtype=np.float32),
            speech_context=context,
        )
    finally:
        recognizer.close()

    parsed = task.parse_spoken_response(task.initial_state(), [], result.text)
    assert parsed.status == "ok", result.text
    assert parsed.canonical_text == expected
