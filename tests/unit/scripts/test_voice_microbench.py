from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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
