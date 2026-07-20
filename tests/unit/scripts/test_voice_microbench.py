from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "voice_microbench.py"
_SPEC = importlib.util.spec_from_file_location("voice_microbench", _MODULE_PATH)
assert _SPEC is not None
voice_microbench = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["voice_microbench"] = voice_microbench
_SPEC.loader.exec_module(voice_microbench)

BackendResult = voice_microbench.BackendResult
Measurement = voice_microbench.Measurement
SampleInfo = voice_microbench.SampleInfo
render_markdown_table = voice_microbench.render_markdown_table
render_sample_table = voice_microbench.render_sample_table
summarize_measurements = voice_microbench.summarize_measurements
build_piper_tts_batch_code = voice_microbench.build_piper_tts_batch_code
build_kokoro_tts_batch_code = voice_microbench.build_kokoro_tts_batch_code
build_vosk_stt_batch_code = voice_microbench.build_vosk_stt_batch_code
find_system_espeak = voice_microbench.find_system_espeak
score_stt_measurements = voice_microbench.score_stt_measurements
levenshtein_distance = voice_microbench.levenshtein_distance
summarize_measurements_with_setup = voice_microbench.summarize_measurements_with_setup


def test_summarize_measurements_reports_latency_and_resource_stats() -> None:
    summary = summarize_measurements(
        [
            Measurement(latency_ms=10.0, peak_rss_mb=100.0, gpu_peak_mb=200.0, ok=True, output="cat"),
            Measurement(latency_ms=30.0, peak_rss_mb=120.0, gpu_peak_mb=250.0, ok=True, output="cat"),
        ]
    )

    assert summary["available"] is True
    assert summary["runs"] == 2
    assert summary["latency_mean_ms"] == 20.0
    assert summary["latency_median_ms"] == 20.0
    assert summary["latency_min_ms"] == 10.0
    assert summary["latency_max_ms"] == 30.0
    assert summary["peak_rss_mb"] == 120.0
    assert summary["gpu_peak_mb"] == 250.0


def test_summarize_measurements_marks_backend_unavailable_without_successful_runs() -> None:
    summary = summarize_measurements(
        [Measurement(latency_ms=0.0, peak_rss_mb=None, gpu_peak_mb=None, ok=False, output="", error="missing")]
    )

    assert summary["available"] is False
    assert summary["runs"] == 0
    assert summary["error"] == "missing"


def test_render_markdown_table_uses_x_for_unavailable_backend_metrics() -> None:
    table = render_markdown_table(
        "STT",
        [
            BackendResult(
                device="mac",
                technology="whisper.cpp",
                kind="stt",
                summary={"available": False, "runs": 0, "error": "not installed"},
            ),
            BackendResult(
                device="h200",
                technology="faster-whisper",
                kind="stt",
                summary={
                    "available": True,
                    "runs": 10,
                    "latency_mean_ms": 42.123,
                    "latency_min_ms": 30.0,
                    "latency_max_ms": 55.0,
                    "peak_rss_mb": 512.0,
                    "gpu_peak_mb": 1024.0,
                },
            ),
        ],
    )

    assert "| mac | whisper.cpp | X | X | X | X | X | X | X | not installed |" in table
    assert "| h200 | faster-whisper | 10 | 42.1 | 30.0 | 55.0 | X | 512.0 | 1024.0 |  |" in table


def test_render_sample_table_reports_sample_lengths_and_paths() -> None:
    table = render_sample_table(
        [
            SampleInfo(
                phrase="red apple",
                path=Path("/tmp/red_apple.wav"),
                duration_s=0.75,
                sample_rate_hz=16000,
                frame_count=12000,
            )
        ]
    )

    assert "| red apple | /tmp/red_apple.wav | 0.750 | 16000 | 12000 |" in table


def test_piper_batch_code_uses_python_api_with_explicit_espeak_data_path() -> None:
    code = build_piper_tts_batch_code(
        phrases=("cat",),
        output_dir=Path("/tmp/out"),
        model_path=Path("/tmp/model.onnx"),
        espeak_data_dir=Path("/tmp/espeak-ng-data"),
    )

    assert "PiperVoice.load" in code
    assert "espeak_data_dir='/tmp/espeak-ng-data'" in code
    assert "voice.synthesize" in code


def test_kokoro_batch_code_can_override_espeak_loader_paths() -> None:
    code = build_kokoro_tts_batch_code(
        phrases=("cat",),
        output_dir=Path("/tmp/out"),
        espeak_library_path=Path("/opt/homebrew/lib/libespeak-ng.dylib"),
        espeak_data_path=Path("/opt/homebrew/share/espeak-ng-data"),
    )

    assert "espeakng_loader.get_library_path=lambda:" in code
    assert "espeakng_loader.get_data_path=lambda:" in code
    assert "KPipeline" in code


def test_find_system_espeak_returns_none_when_paths_are_missing() -> None:
    assert find_system_espeak(
        library_candidates=(Path("/not/libespeak-ng.dylib"),),
        data_candidates=(Path("/not/espeak-ng-data"),),
    ) is None


def test_vosk_batch_code_uses_short_phrase_model_once_for_all_samples() -> None:
    code = build_vosk_stt_batch_code(
        sample_paths=(Path("/tmp/cat.wav"), Path("/tmp/dog.wav")),
        model_path=Path("/tmp/vosk-model"),
    )

    assert "Model('/tmp/vosk-model')" in code
    assert "KaldiRecognizer" in code
    assert "paths=['/tmp/cat.wav', '/tmp/dog.wav']" in code


def test_score_stt_measurements_normalizes_text_and_counts_exact_matches() -> None:
    scores = score_stt_measurements(
        [
            Measurement(latency_ms=1.0, peak_rss_mb=None, gpu_peak_mb=None, ok=True, output=" Red, apple! ", sample="/tmp/04_red_apple.wav"),
            Measurement(latency_ms=1.0, peak_rss_mb=None, gpu_peak_mb=None, ok=True, output="blue", sample="/tmp/07_blue_car.wav"),
            Measurement(latency_ms=1.0, peak_rss_mb=None, gpu_peak_mb=None, ok=False, output="", sample="/tmp/01_cat.wav"),
        ]
    )

    assert scores["accuracy_available"] is True
    assert scores["accuracy_runs"] == 2
    assert scores["exact_match_count"] == 1
    assert scores["exact_match_rate"] == 0.5
    assert scores["char_distance_mean"] == 2.0
    assert scores["word_distance_mean"] == 0.5


def test_levenshtein_distance_counts_simple_edits() -> None:
    assert levenshtein_distance(["zip", "zap"], ["zip", "sap"]) == 1
    assert levenshtein_distance(list("apple"), list("apples")) == 1


def test_summarize_measurements_with_setup_separates_load_and_warmup() -> None:
    summary = summarize_measurements_with_setup(
        [
            Measurement(latency_ms=100.0, peak_rss_mb=None, gpu_peak_mb=None, ok=True, output="a"),
            Measurement(latency_ms=20.0, peak_rss_mb=None, gpu_peak_mb=None, ok=True, output="b"),
            Measurement(latency_ms=40.0, peak_rss_mb=None, gpu_peak_mb=None, ok=True, output="c"),
        ],
        setup_ms=500.0,
    )

    assert summary["setup_ms"] == 500.0
    assert summary["first_run_ms"] == 100.0
    assert summary["steady_state_mean_ms"] == 30.0


def test_build_length_sweep_phrases_has_requested_variants_and_exact_word_counts() -> None:
    phrases = voice_microbench.build_length_sweep_phrases((1, 2, 4), variants_per_length=5)

    assert len(phrases) == 15
    assert {phrase.word_count for phrase in phrases} == {1, 2, 4}
    assert all(len(phrase.text.split()) == phrase.word_count for phrase in phrases)
    assert len({phrase.phrase_id for phrase in phrases}) == len(phrases)


def test_summarize_sweep_trials_separates_tts_generation_time_from_audio_duration() -> None:
    trials = (
        voice_microbench.SweepTrial(
            kind="tts",
            phrase_id="w1-v1",
            text="cat",
            word_count=1,
            repeat_index=1,
            latency_ms=20.0,
            audio_duration_s=0.5,
            output="one.wav",
        ),
        voice_microbench.SweepTrial(
            kind="tts",
            phrase_id="w1-v1",
            text="cat",
            word_count=1,
            repeat_index=2,
            latency_ms=40.0,
            audio_duration_s=0.5,
            output="two.wav",
        ),
    )

    summary = voice_microbench.summarize_sweep_trials(trials)

    bucket = summary["by_word_count"]["1"]
    assert bucket["generation_p50_ms"] == 30.0
    assert bucket["generation_p95_ms"] == 40.0
    assert bucket["audio_duration_mean_s"] == 0.5
    assert bucket["rtf_p50"] == 0.06


def test_build_length_sweep_schedule_is_deterministic_and_repeats_every_phrase() -> None:
    phrases = voice_microbench.build_length_sweep_phrases((1, 2), variants_per_length=2)

    first = voice_microbench.build_length_sweep_schedule(phrases, repetitions=3, seed=42)
    second = voice_microbench.build_length_sweep_schedule(phrases, repetitions=3, seed=42)

    assert first == second
    assert len(first) == 12
    assert {(phrase.phrase_id, repeat_index) for phrase, repeat_index in first} == {
        (phrase.phrase_id, repeat_index) for phrase in phrases for repeat_index in range(1, 4)
    }


def test_write_length_sweep_outputs_creates_raw_data_summary_and_tts_plots(tmp_path: Path) -> None:
    result = voice_microbench.SweepRunResult(
        device="test-mac",
        technology="espeak-ng",
        kind="tts",
        setup_ms=None,
        first_request_ms=50.0,
        trials=(
            voice_microbench.SweepTrial(
                kind="tts",
                phrase_id="w1-v1",
                text="cat",
                word_count=1,
                repeat_index=1,
                latency_ms=20.0,
                audio_duration_s=0.4,
                output="cat.wav",
            ),
        ),
    )

    paths = voice_microbench.write_length_sweep_outputs(tmp_path, (result,))

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "length_sweep_trials.json").exists()
    assert (tmp_path / "length_sweep_trials.csv").exists()
    assert (tmp_path / "length_sweep_summary.json").exists()
    assert (tmp_path / "length_sweep_report.md").exists()
    assert {path.name for path in paths} == {
        "tts_generation_by_words.png",
        "tts_audio_duration_by_words.png",
        "tts_rtf_by_words.png",
    }


def test_faster_whisper_length_sweep_code_loads_once_and_separates_first_request() -> None:
    code = voice_microbench.build_faster_whisper_length_sweep_code(
        model="tiny.en",
        device="cpu",
        compute_type="int8",
        first_request={"phrase_id": "w1-v1", "path": "/tmp/cat.wav"},
        warmup_requests=({"phrase_id": "w1-v1", "path": "/tmp/cat.wav"},),
        requests=({"phrase_id": "w1-v1", "path": "/tmp/cat.wav"},),
    )

    assert "model=WhisperModel('tiny.en', device='cpu', compute_type='int8')" in code
    assert "first_request_ms" in code
    assert "setup_ms" in code
    assert "rows=[transcribe(request) for request in requests]" in code


def test_piper_length_sweep_code_loads_once_and_records_audio_duration() -> None:
    code = voice_microbench.build_piper_length_sweep_code(
        model_path=Path("/tmp/voice.onnx"),
        espeak_data_dir=Path("/tmp/espeak-ng-data"),
        output_dir=Path("/tmp/output"),
        first_request={"phrase_id": "w1-v1", "text": "cat"},
        warmup_requests=({"phrase_id": "w1-v1", "text": "cat"},),
        requests=({"phrase_id": "w1-v1", "text": "cat"},),
    )

    assert "voice=PiperVoice.load('/tmp/voice.onnx', espeak_data_dir='/tmp/espeak-ng-data')" in code
    assert "audio_duration_s" in code
    assert "first_request_ms" in code
    assert "rows=[synthesize(request, index)" in code


def test_sweep_run_result_from_payload_preserves_stt_input_duration_and_transcript() -> None:
    result = voice_microbench.sweep_run_result_from_payload(
        device="h200",
        technology="faster-whisper tiny.en float16",
        kind="stt",
        payload={
            "setup_ms": 10.0,
            "first_request_ms": 20.0,
            "rows": [
                {
                    "phrase_id": "w1-v1",
                    "text": "cat",
                    "word_count": 1,
                    "repeat_index": 1,
                    "input_audio_duration_s": 0.5,
                    "latency_ms": 30.0,
                    "output": "cat",
                }
            ],
        },
    )

    assert result.setup_ms == 10.0
    assert result.first_request_ms == 20.0
    assert result.trials[0].audio_duration_s == 0.5
    assert result.trials[0].output == "cat"


def test_audio_duration_s_reads_wav_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    voice_microbench._write_placeholder_wav(path)

    assert voice_microbench.audio_duration_s(path) == 0.25


def test_length_sweep_refuses_to_generate_missing_files_in_an_explicit_samples_directory(tmp_path: Path) -> None:
    phrase = voice_microbench.SweepPhrase(phrase_id="w1-v1", text="cat", word_count=1)

    with pytest.raises(FileNotFoundError, match="w1-v1.wav"):
        voice_microbench.ensure_length_sweep_samples((phrase,), tmp_path, require_existing=True)
