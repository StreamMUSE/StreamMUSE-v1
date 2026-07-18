"""Microbenchmark local speech-to-text and text-to-speech backends.

This intentionally benchmarks speech technologies outside the StreamMUSE runtime.
It answers: for 1-3 word phrases, how fast is each backend on this device?
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import platform
import random
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence


Kind = Literal["stt", "tts"]

DEFAULT_PHRASES = (
    "cat",
    "dog",
    "apple",
    "red apple",
    "zip zap",
    "green tea",
    "blue car",
    "hello world",
    "animal name",
    "coffee cup",
)
DEFAULT_SWEEP_WORD_COUNTS = (1, 2, 4, 8, 16, 32, 64)
_LENGTH_SWEEP_STREAMS = (
    "cat watches soft rain fall across the quiet garden beside a warm window while evening lights glow over the small street after sunset today",
    "dog follows bright leaves along the calm river near the old bridge while gentle wind moves through tall trees before dinner tonight",
    "bird carries a small twig toward the hidden nest above green branches as morning sun reaches the peaceful park beside our familiar school",
    "child finds red shells beside clear water on a sandy beach then walks slowly home with a cheerful friend before lunch today",
    "friend brings fresh bread from the local market through the busy square and shares a warm meal with family after a long afternoon outside",
)


@dataclass(frozen=True)
class Measurement:
    latency_ms: float
    peak_rss_mb: float | None
    gpu_peak_mb: float | None
    ok: bool
    output: str
    error: str | None = None
    sample: str | None = None


@dataclass(frozen=True)
class BackendResult:
    device: str
    technology: str
    kind: Kind
    summary: dict[str, Any]
    measurements: tuple[Measurement, ...] = ()


@dataclass(frozen=True)
class SampleInfo:
    phrase: str
    path: Path
    duration_s: float
    sample_rate_hz: int
    frame_count: int


@dataclass(frozen=True)
class SweepPhrase:
    phrase_id: str
    text: str
    word_count: int


@dataclass(frozen=True)
class SweepTrial:
    kind: Kind
    phrase_id: str
    text: str
    word_count: int
    repeat_index: int
    latency_ms: float
    audio_duration_s: float | None
    output: str
    error: str | None = None

    @property
    def real_time_factor(self) -> float | None:
        if self.audio_duration_s is None or self.audio_duration_s <= 0:
            return None
        return self.latency_ms / (self.audio_duration_s * 1000.0)


@dataclass(frozen=True)
class SweepRunResult:
    device: str
    technology: str
    kind: Kind
    setup_ms: float | None
    first_request_ms: float | None
    trials: tuple[SweepTrial, ...]


def sweep_run_result_from_payload(
    *,
    device: str,
    technology: str,
    kind: Kind,
    payload: dict[str, Any],
) -> SweepRunResult:
    duration_key = "input_audio_duration_s" if kind == "stt" else "audio_duration_s"
    trials = tuple(
        SweepTrial(
            kind=kind,
            phrase_id=str(row["phrase_id"]),
            text=str(row["text"]),
            word_count=int(row["word_count"]),
            repeat_index=int(row["repeat_index"]),
            latency_ms=float(row["latency_ms"]),
            audio_duration_s=float(row[duration_key]) if row.get(duration_key) is not None else None,
            output=str(row.get("output", "")),
            error=str(row["error"]) if row.get("error") else None,
        )
        for row in payload.get("rows", [])
    )
    setup_ms = payload.get("setup_ms")
    first_request_ms = payload.get("first_request_ms")
    return SweepRunResult(
        device=device,
        technology=technology,
        kind=kind,
        setup_ms=float(setup_ms) if setup_ms is not None else None,
        first_request_ms=float(first_request_ms) if first_request_ms is not None else None,
        trials=trials,
    )


def build_length_sweep_phrases(
    word_counts: tuple[int, ...] = DEFAULT_SWEEP_WORD_COUNTS,
    *,
    variants_per_length: int = 5,
) -> tuple[SweepPhrase, ...]:
    if variants_per_length < 1 or variants_per_length > len(_LENGTH_SWEEP_STREAMS):
        raise ValueError(f"variants_per_length must be between 1 and {len(_LENGTH_SWEEP_STREAMS)}")
    if any(word_count < 1 for word_count in word_counts):
        raise ValueError("word_counts must contain positive integers")

    phrases: list[SweepPhrase] = []
    for word_count in word_counts:
        for variant, stream in enumerate(_LENGTH_SWEEP_STREAMS[:variants_per_length], start=1):
            tokens = stream.split()
            repeated = (tokens * ((word_count + len(tokens) - 1) // len(tokens)))[:word_count]
            phrases.append(SweepPhrase(f"w{word_count}-v{variant}", " ".join(repeated), word_count))
    return tuple(phrases)


def build_length_sweep_schedule(
    phrases: Sequence[SweepPhrase],
    *,
    repetitions: int,
    seed: int,
) -> tuple[tuple[SweepPhrase, int], ...]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    schedule = [(phrase, repeat_index) for phrase in phrases for repeat_index in range(1, repetitions + 1)]
    random.Random(seed).shuffle(schedule)
    return tuple(schedule)


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0.0, 1.0]")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def summarize_sweep_trials(trials: Sequence[SweepTrial]) -> dict[str, Any]:
    successful = [trial for trial in trials if trial.error is None]
    if not successful:
        return {"runs": 0, "by_word_count": {}}

    kind = successful[0].kind
    by_word_count: dict[str, dict[str, Any]] = {}
    for word_count in sorted({trial.word_count for trial in successful}):
        bucket = [trial for trial in successful if trial.word_count == word_count]
        latencies = [trial.latency_ms for trial in bucket]
        audio_durations = [trial.audio_duration_s for trial in bucket if trial.audio_duration_s is not None]
        row: dict[str, Any] = {
            "runs": len(bucket),
            "latency_mean_ms": round(statistics.fmean(latencies), 3),
            "latency_p50_ms": round(statistics.median(latencies), 3),
            "latency_p95_ms": round(nearest_rank_percentile(latencies, 0.95), 3),
            "latency_max_ms": round(max(latencies), 3),
            "audio_duration_mean_s": round(statistics.fmean(audio_durations), 6) if audio_durations else None,
        }
        if kind == "tts":
            rtfs = [trial.real_time_factor for trial in bucket if trial.real_time_factor is not None]
            row.update(
                {
                    "generation_mean_ms": row["latency_mean_ms"],
                    "generation_p50_ms": row["latency_p50_ms"],
                    "generation_p95_ms": row["latency_p95_ms"],
                    "generation_max_ms": row["latency_max_ms"],
                    "rtf_p50": round(statistics.median(rtfs), 6) if rtfs else None,
                    "rtf_p95": round(nearest_rank_percentile(rtfs, 0.95), 6) if rtfs else None,
                }
            )
        else:
            row["transcription_p50_ms"] = row["latency_p50_ms"]
            row["transcription_p95_ms"] = row["latency_p95_ms"]
        by_word_count[str(word_count)] = row
    return {"kind": kind, "runs": len(successful), "by_word_count": by_word_count}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=platform.node() or platform.system().lower())
    parser.add_argument("--output-dir", default="voice_bench_runs")
    parser.add_argument("--samples-dir", default=None)
    parser.add_argument("--phrases", default=",".join(DEFAULT_PHRASES))
    parser.add_argument("--kinds", default="stt,tts", help="Comma-separated: stt,tts")
    parser.add_argument("--stt-backends", default="whisper_cpp,faster_whisper,sherpa_onnx")
    parser.add_argument("--tts-backends", default="system_tts,piper,kokoro,sherpa_onnx")
    parser.add_argument("--length-sweep", action="store_true", help="Run the reproducible 1-64 word latency sweep.")
    parser.add_argument("--word-counts", default=",".join(str(value) for value in DEFAULT_SWEEP_WORD_COUNTS))
    parser.add_argument("--variants-per-length", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args(argv)

    run_prefix = "voice_length_sweep" if args.length_sweep else "voice_microbench"
    run_dir = Path(args.output_dir).expanduser().resolve() / time.strftime(f"{run_prefix}_%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.length_sweep:
        return _run_length_sweep(args, run_dir)

    phrases = tuple(part.strip() for part in str(args.phrases).split(",") if part.strip())
    samples_dir = Path(args.samples_dir).expanduser().resolve() if args.samples_dir else run_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    results: list[BackendResult] = []
    kinds = {part.strip() for part in str(args.kinds).split(",") if part.strip()}
    if "tts" in kinds:
        for backend in _split_csv(args.tts_backends):
            results.append(run_tts_backend(args.device, backend, phrases, run_dir))
    sample_infos: tuple[SampleInfo, ...] = ()
    if "stt" in kinds:
        sample_infos = ensure_samples(phrases, samples_dir)
        sample_paths = tuple(sample.path for sample in sample_infos)
        for backend in _split_csv(args.stt_backends):
            results.append(run_stt_backend(args.device, backend, sample_paths, run_dir))

    write_outputs(run_dir, results, sample_infos)
    print(f"wrote: {run_dir}")
    if sample_infos:
        print(render_sample_table(sample_infos))
    print(render_markdown_table("STT", [result for result in results if result.kind == "stt"]))
    print(render_markdown_table("TTS", [result for result in results if result.kind == "tts"]))
    return 0


def _run_length_sweep(args: argparse.Namespace, run_dir: Path) -> int:
    word_counts = _parse_positive_int_csv(args.word_counts, option="--word-counts")
    phrases = build_length_sweep_phrases(word_counts, variants_per_length=args.variants_per_length)
    schedule = build_length_sweep_schedule(phrases, repetitions=args.repetitions, seed=args.seed)
    requests = tuple(
        {
            "phrase_id": phrase.phrase_id,
            "text": phrase.text,
            "word_count": phrase.word_count,
            "repeat_index": repeat_index,
        }
        for phrase, repeat_index in schedule
    )
    warmup_requests = _select_warmup_requests(requests, args.warmup_requests)
    kinds = {part.strip() for part in str(args.kinds).split(",") if part.strip()}
    unsupported_kinds = kinds.difference({"stt", "tts"})
    if unsupported_kinds:
        raise ValueError(f"unknown sweep kinds: {', '.join(sorted(unsupported_kinds))}")

    results: list[SweepRunResult] = []
    sample_rows: list[dict[str, Any]] = []
    if "stt" in kinds:
        samples_dir = Path(args.samples_dir).expanduser().resolve() if args.samples_dir else run_dir / "samples"
        samples = ensure_length_sweep_samples(
            phrases,
            samples_dir,
            require_existing=args.samples_dir is not None,
        )
        samples_by_phrase_id = {phrase_id: sample for phrase_id, _, sample in samples}
        sample_rows = [
            {
                "phrase_id": phrase_id,
                "text": phrase.text,
                "word_count": phrase.word_count,
                "path": str(sample.path),
                "audio_duration_s": sample.duration_s,
                "sample_rate_hz": sample.sample_rate_hz,
            }
            for phrase_id, phrase, sample in samples
        ]
        stt_requests = tuple(
            {
                **request,
                "path": str(samples_by_phrase_id[str(request["phrase_id"])].path),
                "input_audio_duration_s": samples_by_phrase_id[str(request["phrase_id"])].duration_s,
            }
            for request in requests
        )
        stt_warmups = _select_warmup_requests(stt_requests, args.warmup_requests)
        for backend in _split_csv(args.stt_backends):
            if backend != "faster_whisper":
                raise ValueError("--length-sweep currently supports only --stt-backends faster_whisper")
            results.append(
                run_faster_whisper_length_sweep(
                    device=args.device,
                    requests=stt_requests,
                    warmup_requests=stt_warmups,
                )
            )

    if "tts" in kinds:
        for backend in _split_csv(args.tts_backends):
            if backend == "espeak_ng":
                results.append(
                    run_espeak_ng_length_sweep(
                        device=args.device,
                        requests=requests,
                        warmup_requests=warmup_requests,
                        output_dir=run_dir / "tts" / backend,
                    )
                )
            elif backend == "piper":
                results.append(
                    run_piper_length_sweep(
                        device=args.device,
                        requests=requests,
                        warmup_requests=warmup_requests,
                        output_dir=run_dir / "tts" / backend,
                    )
                )
            else:
                raise ValueError("--length-sweep currently supports only --tts-backends espeak_ng,piper")

    plot_paths = write_length_sweep_outputs(run_dir, results)
    _augment_length_sweep_manifest(
        run_dir,
        {
            "word_counts": list(word_counts),
            "variants_per_length": args.variants_per_length,
            "repetitions": args.repetitions,
            "warmup_requests": args.warmup_requests,
            "seed": args.seed,
            "runner_python": _benchmark_python(),
            "samples": sample_rows,
        },
    )
    print(f"wrote: {run_dir}")
    print(f"warmed trials: {sum(len(result.trials) for result in results)}")
    for path in plot_paths:
        print(f"plot: {path}")
    return 0


def _parse_positive_int_csv(raw: str, *, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{option} must be a comma-separated list of positive integers") from exc
    if not values or any(value < 1 for value in values):
        raise ValueError(f"{option} must contain at least one positive integer")
    return values


def _select_warmup_requests(requests: Sequence[dict[str, Any]], count: int) -> tuple[dict[str, Any], ...]:
    if count < 0:
        raise ValueError("warmup request count must not be negative")
    if not requests:
        raise ValueError("cannot select warmups without requests")
    return tuple(requests[index % len(requests)] for index in range(1, count + 1))


def ensure_length_sweep_samples(
    phrases: Sequence[SweepPhrase],
    samples_dir: Path,
    *,
    require_existing: bool = False,
) -> tuple[tuple[str, SweepPhrase, SampleInfo], ...]:
    rows: list[tuple[str, SweepPhrase, SampleInfo]] = []
    for phrase in phrases:
        path = samples_dir / f"{phrase.phrase_id}.wav"
        if not path.exists():
            if require_existing:
                raise FileNotFoundError(f"required length-sweep sample is missing: {path}")
            _write_sample_wav(phrase.text, path)
        rows.append((phrase.phrase_id, phrase, _sample_info(phrase.text, path)))
    return tuple(rows)


def run_faster_whisper_length_sweep(
    *,
    device: str,
    requests: Sequence[dict[str, Any]],
    warmup_requests: Sequence[dict[str, Any]],
) -> SweepRunResult:
    if not requests:
        raise ValueError("faster-whisper sweep needs at least one request")
    model = os.environ.get("FASTER_WHISPER_MODEL", "tiny.en")
    inference_device = os.environ.get("FASTER_WHISPER_DEVICE", "auto")
    compute_type = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")
    payload = _run_sweep_json_command(
        [
            _benchmark_python(),
            "-c",
            build_faster_whisper_length_sweep_code(
                model=model,
                device=inference_device,
                compute_type=compute_type,
                first_request=requests[0],
                warmup_requests=warmup_requests,
                requests=requests,
            ),
        ]
    )
    return sweep_run_result_from_payload(
        device=device,
        technology=f"faster-whisper {model} {inference_device} {compute_type}",
        kind="stt",
        payload=payload,
    )


def run_piper_length_sweep(
    *,
    device: str,
    requests: Sequence[dict[str, Any]],
    warmup_requests: Sequence[dict[str, Any]],
    output_dir: Path,
) -> SweepRunResult:
    if not requests:
        raise ValueError("Piper sweep needs at least one request")
    model = os.environ.get("PIPER_MODEL")
    if not model:
        raise RuntimeError("PIPER_MODEL must identify a Piper .onnx voice model")
    espeak_data_dir = _piper_espeak_data_dir_for_benchmark_python()
    if espeak_data_dir is None:
        raise RuntimeError("Piper espeak-ng-data directory was not found; set PIPER_ESPEAK_DATA_DIR")
    payload = _run_sweep_json_command(
        [
            _benchmark_python(),
            "-c",
            build_piper_length_sweep_code(
                model_path=Path(model),
                espeak_data_dir=espeak_data_dir,
                output_dir=output_dir,
                first_request=requests[0],
                warmup_requests=warmup_requests,
                requests=requests,
            ),
        ]
    )
    return sweep_run_result_from_payload(
        device=device,
        technology=f"Piper {Path(model).stem}",
        kind="tts",
        payload=payload,
    )


def run_espeak_ng_length_sweep(
    *,
    device: str,
    requests: Sequence[dict[str, Any]],
    warmup_requests: Sequence[dict[str, Any]],
    output_dir: Path,
) -> SweepRunResult:
    if not requests:
        raise ValueError("espeak-ng sweep needs at least one request")
    command = shutil.which("espeak-ng") or shutil.which("espeak")
    if not command:
        raise RuntimeError("espeak-ng/espeak command was not found")
    output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize(request: dict[str, Any], index: int) -> SweepTrial:
        output_path = output_dir / f"{index:04d}.wav"
        measurement = _run_command([command, "-w", str(output_path), str(request["text"])], sample=str(request["text"]))
        duration = audio_duration_s(output_path) if measurement.ok else None
        return SweepTrial(
            kind="tts",
            phrase_id=str(request["phrase_id"]),
            text=str(request["text"]),
            word_count=int(request["word_count"]),
            repeat_index=int(request["repeat_index"]),
            latency_ms=measurement.latency_ms,
            audio_duration_s=duration,
            output=str(output_path),
            error=measurement.error,
        )

    first_request = synthesize(requests[0], 0)
    for index, request in enumerate(warmup_requests, start=-len(warmup_requests)):
        synthesize(request, index)
    trials = tuple(synthesize(request, index) for index, request in enumerate(requests, start=1))
    return SweepRunResult(
        device=device,
        technology=Path(command).name,
        kind="tts",
        setup_ms=None,
        first_request_ms=first_request.latency_ms,
        trials=trials,
    )


def _benchmark_python() -> str:
    return os.environ.get("VOICE_BENCH_PYTHON", sys.executable)


def _piper_espeak_data_dir_for_benchmark_python() -> Path | None:
    override = os.environ.get("PIPER_ESPEAK_DATA_DIR")
    if override:
        path = Path(override)
        return path if path.exists() else None
    if _benchmark_python() == sys.executable:
        return _piper_espeak_data_dir()
    probe = (
        "import importlib.util\n"
        "from pathlib import Path\n"
        "spec=importlib.util.find_spec('piper')\n"
        "print(Path(spec.origin).parent / 'espeak-ng-data' if spec and spec.origin else '')\n"
    )
    proc = subprocess.run([_benchmark_python(), "-c", probe], text=True, capture_output=True, check=False)
    path = Path(proc.stdout.strip()) if proc.returncode == 0 and proc.stdout.strip() else None
    return path if path is not None and path.exists() else None


def _run_sweep_json_command(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=float(os.environ.get("VOICE_BENCH_TIMEOUT_S", "900")),
        check=False,
    )
    output = (proc.stdout or "").strip()
    if proc.returncode != 0:
        detail = (proc.stderr or output).strip()
        raise RuntimeError(f"sweep backend failed ({proc.returncode}): {detail}")
    try:
        payload = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"sweep backend did not emit JSON: {output or proc.stderr}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise RuntimeError("sweep backend emitted an invalid JSON payload")
    return payload


def _augment_length_sweep_manifest(run_dir: Path, benchmark: dict[str, Any]) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["benchmark"] = benchmark
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def ensure_samples(phrases: Iterable[str], samples_dir: Path) -> tuple[SampleInfo, ...]:
    samples: list[SampleInfo] = []
    for index, phrase in enumerate(phrases, start=1):
        path = samples_dir / f"{index:02d}_{_slug(phrase)}.wav"
        if not path.exists():
            _write_sample_wav(phrase, path)
        samples.append(_sample_info(phrase, path))
    return tuple(samples)


def run_tts_backend(device: str, backend: str, phrases: tuple[str, ...], run_dir: Path) -> BackendResult:
    if backend == "piper":
        measurements = _run_piper_tts_batch(phrases, run_dir / "tts" / backend)
        return BackendResult(
            device,
            backend,
            "tts",
            summarize_measurements_with_setup(measurements, setup_ms=_pop_last_batch_setup_ms()),
            tuple(measurements),
        )
    if backend == "espeak_ng":
        measurements = _run_many(lambda phrase, index: _run_espeak_ng_tts(phrase, run_dir / "tts" / backend, index), phrases)
        return BackendResult(device, backend, "tts", summarize_measurements(measurements), tuple(measurements))
    if backend == "kokoro":
        measurements = _run_kokoro_tts_batch(phrases, run_dir / "tts" / backend)
        return BackendResult(
            device,
            backend,
            "tts",
            summarize_measurements_with_setup(measurements, setup_ms=_pop_last_batch_setup_ms()),
            tuple(measurements),
        )
    runner = {
        "system_tts": _run_system_tts,
        "sherpa_onnx": _run_sherpa_tts,
    }.get(backend)
    if runner is None:
        return _unavailable(device, backend, "tts", f"unknown tts backend: {backend}")

    measurements = _run_many(lambda phrase, index: runner(phrase, run_dir / "tts" / backend, index), phrases)
    return BackendResult(device, backend, "tts", summarize_measurements(measurements), tuple(measurements))


def run_stt_backend(device: str, backend: str, sample_paths: tuple[Path, ...], run_dir: Path) -> BackendResult:
    if backend == "vosk":
        measurements = _run_vosk_stt_batch(sample_paths, run_dir / "stt" / backend)
        return _stt_result(device, backend, measurements, setup_ms=_pop_last_batch_setup_ms())
    if backend == "faster_whisper":
        measurements = _run_faster_whisper_stt_batch(sample_paths, run_dir / "stt" / backend)
        return _stt_result(device, backend, measurements, setup_ms=_pop_last_batch_setup_ms())
    runner = {
        "whisper_cpp": _run_whisper_cpp_stt,
        "sherpa_onnx": _run_sherpa_stt,
    }.get(backend)
    if runner is None:
        return _unavailable(device, backend, "stt", f"unknown stt backend: {backend}")

    measurements = _run_many(lambda path, index: runner(path, run_dir / "stt" / backend, index), sample_paths)
    return _stt_result(device, backend, measurements)


def _run_many(run_one: Callable[[Any, int], Measurement], samples: Iterable[Any]) -> list[Measurement]:
    measurements: list[Measurement] = []
    for index, sample in enumerate(samples, start=1):
        try:
            measurements.append(run_one(sample, index))
        except Exception as exc:  # pragma: no cover - defensive for optional backends
            measurements.append(
                Measurement(
                    latency_ms=0.0,
                    peak_rss_mb=None,
                    gpu_peak_mb=_gpu_memory_mb(),
                    ok=False,
                    output="",
                    error=str(exc),
                    sample=str(sample),
                )
            )
    return measurements


def summarize_measurements(measurements: list[Measurement]) -> dict[str, Any]:
    return summarize_measurements_with_setup(measurements)


def summarize_measurements_with_setup(measurements: list[Measurement], setup_ms: float | None = None) -> dict[str, Any]:
    successful = [item for item in measurements if item.ok]
    if not successful:
        error = next((item.error for item in measurements if item.error), "no successful runs")
        return {"available": False, "runs": 0, "error": error}

    latencies = [item.latency_ms for item in successful]
    steady_latencies = latencies[1:]
    rss_values = [item.peak_rss_mb for item in successful if item.peak_rss_mb is not None]
    gpu_values = [item.gpu_peak_mb for item in successful if item.gpu_peak_mb is not None]
    summary = {
        "available": True,
        "runs": len(successful),
        "latency_mean_ms": round(statistics.fmean(latencies), 3),
        "latency_median_ms": round(statistics.median(latencies), 3),
        "latency_min_ms": round(min(latencies), 3),
        "latency_max_ms": round(max(latencies), 3),
        "first_run_ms": round(latencies[0], 3),
        "steady_state_mean_ms": round(statistics.fmean(steady_latencies), 3) if steady_latencies else None,
        "peak_rss_mb": round(max(rss_values), 3) if rss_values else None,
        "gpu_peak_mb": round(max(gpu_values), 3) if gpu_values else None,
    }
    if setup_ms is not None:
        summary["setup_ms"] = round(float(setup_ms), 3)
    return summary


def _stt_result(device: str, backend: str, measurements: list[Measurement], setup_ms: float | None = None) -> BackendResult:
    summary = summarize_measurements_with_setup(measurements, setup_ms=setup_ms)
    summary.update(score_stt_measurements(measurements))
    return BackendResult(device, backend, "stt", summary, tuple(measurements))


def score_stt_measurements(measurements: list[Measurement]) -> dict[str, Any]:
    rows = []
    for item in measurements:
        if not item.ok or not item.sample:
            continue
        expected = _expected_phrase_from_sample(item.sample)
        if not expected:
            continue
        actual_norm = normalize_text(item.output)
        expected_norm = normalize_text(expected)
        rows.append(
            {
                "sample": item.sample,
                "expected": expected,
                "actual": item.output,
                "expected_normalized": expected_norm,
                "actual_normalized": actual_norm,
                "exact_match": actual_norm == expected_norm,
                "char_distance": levenshtein_distance(list(expected_norm), list(actual_norm)),
                "word_distance": levenshtein_distance(expected_norm.split(), actual_norm.split()),
            }
        )
    if not rows:
        return {"accuracy_available": False, "accuracy_runs": 0}
    exact_count = sum(1 for row in rows if row["exact_match"])
    return {
        "accuracy_available": True,
        "accuracy_runs": len(rows),
        "exact_match_count": exact_count,
        "exact_match_rate": round(exact_count / float(len(rows)), 3),
        "char_distance_mean": round(statistics.fmean(row["char_distance"] for row in rows), 3),
        "word_distance_mean": round(statistics.fmean(row["word_distance"] for row in rows), 3),
        "accuracy_rows": rows,
    }


def levenshtein_distance(left: list[Any], right: list[Any]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            insert_cost = current[right_index - 1] + 1
            delete_cost = previous[right_index] + 1
            replace_cost = previous[right_index - 1] + (0 if left_item == right_item else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def normalize_text(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _expected_phrase_from_sample(sample: str) -> str | None:
    stem = Path(sample).stem
    cleaned = re.sub(r"^\d+_", "", stem)
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned or None


def render_markdown_table(title: str, results: list[BackendResult]) -> str:
    lines = [
        f"### {title}",
        "",
        "| device | technology | runs | mean ms | min ms | max ms | exact match | peak RSS MB | GPU peak MB | error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        summary = result.summary
        if not summary.get("available"):
            error = str(summary.get("error", "")).replace("\n", " ")
            lines.append(f"| {result.device} | {result.technology} | X | X | X | X | X | X | X | {error} |")
            continue
        lines.append(
            "| {device} | {technology} | {runs} | {mean} | {min_} | {max_} | {accuracy} | {rss} | {gpu} |  |".format(
                device=result.device,
                technology=result.technology,
                runs=summary.get("runs", "X"),
                mean=_fmt(summary.get("latency_mean_ms")),
                min_=_fmt(summary.get("latency_min_ms")),
                max_=_fmt(summary.get("latency_max_ms")),
                accuracy=_accuracy_fmt(summary),
                rss=_fmt(summary.get("peak_rss_mb")),
                gpu=_fmt(summary.get("gpu_peak_mb")),
            )
        )
    return "\n".join(lines)


def render_sample_table(samples: Iterable[SampleInfo]) -> str:
    lines = [
        "### Input Samples",
        "",
        "| phrase | path | duration s | sample rate Hz | frames |",
        "|---|---:|---:|---:|---:|",
    ]
    for sample in samples:
        lines.append(
            f"| {sample.phrase} | {sample.path} | {sample.duration_s:.3f} | "
            f"{sample.sample_rate_hz} | {sample.frame_count} |"
        )
    return "\n".join(lines)


def write_outputs(run_dir: Path, results: list[BackendResult], samples: tuple[SampleInfo, ...] = ()) -> None:
    raw = [asdict(result) for result in results]
    (run_dir / "results.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    (run_dir / "samples.json").write_text(
        json.dumps(
            [
                {
                    "phrase": sample.phrase,
                    "path": str(sample.path),
                    "duration_s": sample.duration_s,
                    "sample_rate_hz": sample.sample_rate_hz,
                    "frame_count": sample.frame_count,
                }
                for sample in samples
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Voice Microbenchmark Report\n\n"
        + (render_sample_table(samples) + "\n\n" if samples else "")
        + render_markdown_table("STT", [result for result in results if result.kind == "stt"])
        + "\n\n"
        + render_markdown_table("TTS", [result for result in results if result.kind == "tts"])
        + "\n",
        encoding="utf-8",
    )
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device",
                "kind",
                "technology",
                "available",
                "runs",
                "latency_mean_ms",
                "latency_median_ms",
                "latency_min_ms",
                "latency_max_ms",
                "first_run_ms",
                "steady_state_mean_ms",
                "setup_ms",
                "exact_match_rate",
                "char_distance_mean",
                "word_distance_mean",
                "peak_rss_mb",
                "gpu_peak_mb",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            row = {"device": result.device, "kind": result.kind, "technology": result.technology, **result.summary}
            writer.writerow({field: row.get(field) for field in writer.fieldnames})


def write_length_sweep_outputs(run_dir: Path, results: Sequence[SweepRunResult]) -> tuple[Path, ...]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "runs": [
            {
                "device": result.device,
                "technology": result.technology,
                "kind": result.kind,
                "setup_ms": result.setup_ms,
                "first_request_ms": result.first_request_ms,
                "warmed_trials": len(result.trials),
            }
            for result in results
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    raw_payload = {"runs": [asdict(result) for result in results]}
    (run_dir / "length_sweep_trials.json").write_text(json.dumps(raw_payload, indent=2), encoding="utf-8")

    summaries = [
        {
            "device": result.device,
            "technology": result.technology,
            "kind": result.kind,
            "setup_ms": result.setup_ms,
            "first_request_ms": result.first_request_ms,
            **summarize_sweep_trials(result.trials),
        }
        for result in results
    ]
    (run_dir / "length_sweep_summary.json").write_text(json.dumps({"runs": summaries}, indent=2), encoding="utf-8")

    with (run_dir / "length_sweep_trials.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "device",
            "technology",
            "kind",
            "setup_ms",
            "first_request_ms",
            "phrase_id",
            "text",
            "word_count",
            "repeat_index",
            "latency_ms",
            "audio_duration_s",
            "real_time_factor",
            "output",
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for trial in result.trials:
                writer.writerow(
                    {
                        "device": result.device,
                        "technology": result.technology,
                        "kind": result.kind,
                        "setup_ms": result.setup_ms,
                        "first_request_ms": result.first_request_ms,
                        **asdict(trial),
                        "real_time_factor": trial.real_time_factor,
                    }
                )

    report_lines = [
        "# Voice Length-Sweep Report",
        "",
        "| device | backend | kind | setup ms | first request ms | warmed trials |",
        "|---|---|---|---:|---:|---:|",
    ]
    for result in results:
        report_lines.append(
            f"| {result.device} | {result.technology} | {result.kind} | {_fmt(result.setup_ms)} | "
            f"{_fmt(result.first_request_ms)} | {len(result.trials)} |"
        )
    report_lines.extend(["", "The plots show individual warmed trials, p50, and p95. TTS generation time and audio duration use separate plots."])
    (run_dir / "length_sweep_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    plots_dir = run_dir / "plots"
    plot_paths: list[Path] = []
    for result in results:
        plot_paths.extend(render_length_sweep_plots(result, plots_dir))
    return tuple(plot_paths)


def render_length_sweep_plots(result: SweepRunResult, plots_dir: Path) -> tuple[Path, ...]:
    if not result.trials:
        return ()
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_sweep_trials(result.trials)["by_word_count"]
    word_ticks = [float(word_count) for word_count in summary]
    suffix = "" if result.kind not in {"stt", "tts"} else ""
    prefix = f"{result.device} | {result.technology}"
    paths: list[Path] = []

    if result.kind == "stt":
        path = plots_dir / f"stt_latency_by_words{suffix}.png"
        _plot_trials_and_summary(
            plt,
            result.trials,
            x_values=[float(trial.word_count) for trial in result.trials],
            trial_y_values=[trial.latency_ms for trial in result.trials],
            summary_x_values=[float(word_count) for word_count in summary],
            p50_values=[summary[word_count]["latency_p50_ms"] for word_count in summary],
            p95_values=[summary[word_count]["latency_p95_ms"] for word_count in summary],
            title=f"{prefix}: STT latency by words",
            x_label="Input words",
            y_label="Transcription latency (ms)",
            path=path,
            x_tick_values=word_ticks,
        )
        paths.append(path)

        path = plots_dir / f"stt_latency_by_audio_duration{suffix}.png"
        with_duration = [trial for trial in result.trials if trial.audio_duration_s is not None]
        _plot_trials_and_summary(
            plt,
            with_duration,
            x_values=[float(trial.audio_duration_s) for trial in with_duration],
            trial_y_values=[trial.latency_ms for trial in with_duration],
            summary_x_values=[summary[word_count]["audio_duration_mean_s"] for word_count in summary],
            p50_values=[summary[word_count]["latency_p50_ms"] for word_count in summary],
            p95_values=[summary[word_count]["latency_p95_ms"] for word_count in summary],
            title=f"{prefix}: STT latency by input audio duration",
            x_label="Input audio duration (s)",
            y_label="Transcription latency (ms)",
            path=path,
        )
        paths.append(path)
    elif result.kind == "tts":
        path = plots_dir / f"tts_generation_by_words{suffix}.png"
        _plot_trials_and_summary(
            plt,
            result.trials,
            x_values=[float(trial.word_count) for trial in result.trials],
            trial_y_values=[trial.latency_ms for trial in result.trials],
            summary_x_values=[float(word_count) for word_count in summary],
            p50_values=[summary[word_count]["generation_p50_ms"] for word_count in summary],
            p95_values=[summary[word_count]["generation_p95_ms"] for word_count in summary],
            title=f"{prefix}: TTS generation time by words",
            x_label="Input words",
            y_label="Generation time (ms)",
            path=path,
            x_tick_values=word_ticks,
        )
        paths.append(path)

        path = plots_dir / f"tts_audio_duration_by_words{suffix}.png"
        with_duration = [trial for trial in result.trials if trial.audio_duration_s is not None]
        _plot_trials_and_summary(
            plt,
            with_duration,
            x_values=[float(trial.word_count) for trial in with_duration],
            trial_y_values=[float(trial.audio_duration_s) for trial in with_duration],
            summary_x_values=[float(word_count) for word_count in summary],
            p50_values=[summary[word_count]["audio_duration_mean_s"] for word_count in summary],
            p95_values=[summary[word_count]["audio_duration_mean_s"] for word_count in summary],
            title=f"{prefix}: generated audio duration by words",
            x_label="Input words",
            y_label="Generated audio duration (s)",
            path=path,
            x_tick_values=word_ticks,
            p95_label="Mean duration",
        )
        paths.append(path)

        path = plots_dir / f"tts_rtf_by_words{suffix}.png"
        with_rtf = [trial for trial in result.trials if trial.real_time_factor is not None]
        _plot_trials_and_summary(
            plt,
            with_rtf,
            x_values=[float(trial.word_count) for trial in with_rtf],
            trial_y_values=[float(trial.real_time_factor) for trial in with_rtf],
            summary_x_values=[float(word_count) for word_count in summary],
            p50_values=[summary[word_count]["rtf_p50"] for word_count in summary],
            p95_values=[summary[word_count]["rtf_p95"] for word_count in summary],
            title=f"{prefix}: TTS real-time factor by words",
            x_label="Input words",
            y_label="Generation time / audio duration",
            path=path,
            x_tick_values=word_ticks,
        )
        paths.append(path)
    return tuple(paths)


def _plot_trials_and_summary(
    plt: Any,
    trials: Sequence[SweepTrial],
    *,
    x_values: Sequence[float],
    trial_y_values: Sequence[float],
    summary_x_values: Sequence[float | None],
    p50_values: Sequence[float | None],
    p95_values: Sequence[float | None],
    title: str,
    x_label: str,
    y_label: str,
    path: Path,
    x_tick_values: Sequence[float] | None = None,
    p95_label: str = "p95",
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    axis.scatter(x_values, trial_y_values, alpha=0.2, s=18, color="#4C78A8", label="Warmed trials")
    points = [
        (x, p50, p95)
        for x, p50, p95 in zip(summary_x_values, p50_values, p95_values)
        if x is not None and p50 is not None and p95 is not None
    ]
    if points:
        summary_x, p50, p95 = zip(*points)
        axis.plot(summary_x, p50, marker="o", color="#F58518", label="p50")
        axis.plot(summary_x, p95, marker="s", linestyle="--", color="#54A24B", label=p95_label)
    if x_tick_values:
        ticks = sorted(set(x_tick_values))
        axis.set_xticks(ticks)
        axis.set_xlim(0, ticks[-1] * 1.05)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _run_system_tts(text: str, output_dir: Path, index: int) -> Measurement:
    output_dir.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    if system == "darwin" and shutil.which("say"):
        out = output_dir / f"{index:02d}.aiff"
        return _run_command(["say", "-o", str(out), text], sample=text)
    command = shutil.which("espeak-ng") or shutil.which("espeak")
    if command:
        out = output_dir / f"{index:02d}.wav"
        return _run_command([command, "-w", str(out), text], sample=text)
    return Measurement(0.0, None, _gpu_memory_mb(), False, "", "no system TTS command found", text)


def _run_espeak_ng_tts(text: str, output_dir: Path, index: int) -> Measurement:
    command = shutil.which("espeak-ng") or shutil.which("espeak")
    if not command:
        return Measurement(0.0, None, _gpu_memory_mb(), False, "", "espeak-ng/espeak command not found", text)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{index:02d}.wav"
    return _run_command([command, "-w", str(out), text], sample=text)


def _run_piper_tts_batch(phrases: tuple[str, ...], output_dir: Path) -> list[Measurement]:
    if importlib.util.find_spec("piper") is None:
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", "piper package not installed")]
    model = os.environ.get("PIPER_MODEL")
    if not model:
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", "PIPER_MODEL is not set")]
    espeak_data_dir = _piper_espeak_data_dir()
    if espeak_data_dir is None:
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", "piper espeak-ng-data directory not found")]
    output_dir.mkdir(parents=True, exist_ok=True)
    code = build_piper_tts_batch_code(
        phrases=phrases,
        output_dir=output_dir,
        model_path=Path(model),
        espeak_data_dir=espeak_data_dir,
    )
    return _run_json_batch_command([sys.executable, "-c", code])


def build_piper_tts_batch_code(
    *,
    phrases: tuple[str, ...],
    output_dir: Path,
    model_path: Path,
    espeak_data_dir: Path,
) -> str:
    return (
        "import json, time, wave\n"
        "from piper import PiperVoice\n"
        "setup_start=time.perf_counter()\n"
        f"voice=PiperVoice.load({str(model_path)!r}, espeak_data_dir={str(espeak_data_dir)!r})\n"
        "setup_ms=(time.perf_counter()-setup_start)*1000.0\n"
        f"phrases={list(phrases)!r}\n"
        f"out_dir={str(output_dir)!r}\n"
        "rows=[]\n"
        "for index, text in enumerate(phrases, start=1):\n"
        "    out=f'{out_dir}/{index:02d}.wav'\n"
        "    start=time.perf_counter()\n"
        "    with wave.open(out, 'wb') as wav_file:\n"
        "        params_set=False\n"
        "        for chunk in voice.synthesize(text):\n"
        "            if not params_set:\n"
        "                wav_file.setframerate(chunk.sample_rate)\n"
        "                wav_file.setsampwidth(chunk.sample_width)\n"
        "                wav_file.setnchannels(chunk.sample_channels)\n"
        "                params_set=True\n"
        "            wav_file.writeframes(chunk.audio_int16_bytes)\n"
        "    rows.append({'sample': text, 'latency_ms': (time.perf_counter()-start)*1000.0, 'output': out})\n"
        "print(json.dumps({'setup_ms': setup_ms, 'rows': rows}))\n"
    )


def build_piper_length_sweep_code(
    *,
    model_path: Path,
    espeak_data_dir: Path,
    output_dir: Path,
    first_request: dict[str, Any],
    warmup_requests: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
) -> str:
    return (
        "import json, time, wave\n"
        "from pathlib import Path\n"
        "from piper import PiperVoice\n"
        "setup_start=time.perf_counter()\n"
        f"voice=PiperVoice.load({str(model_path)!r}, espeak_data_dir={str(espeak_data_dir)!r})\n"
        "setup_ms=(time.perf_counter()-setup_start)*1000.0\n"
        f"out_dir=Path({str(output_dir)!r})\n"
        "out_dir.mkdir(parents=True, exist_ok=True)\n"
        f"first_request={dict(first_request)!r}\n"
        f"warmup_requests={list(warmup_requests)!r}\n"
        f"requests={list(requests)!r}\n"
        "def synthesize(request, index):\n"
        "    path=out_dir/f'{index:04d}.wav'\n"
        "    start=time.perf_counter()\n"
        "    frame_count=0\n"
        "    sample_rate=None\n"
        "    with wave.open(str(path), 'wb') as wav_file:\n"
        "        params_set=False\n"
        "        for chunk in voice.synthesize(request['text']):\n"
        "            if not params_set:\n"
        "                wav_file.setframerate(chunk.sample_rate)\n"
        "                wav_file.setsampwidth(chunk.sample_width)\n"
        "                wav_file.setnchannels(chunk.sample_channels)\n"
        "                sample_rate=chunk.sample_rate\n"
        "                params_set=True\n"
        "            wav_file.writeframes(chunk.audio_int16_bytes)\n"
        "            frame_count += len(chunk.audio_int16_bytes)//(chunk.sample_width*chunk.sample_channels)\n"
        "    if sample_rate is None:\n"
        "        raise RuntimeError('Piper produced no audio')\n"
        "    return {**request, 'latency_ms': (time.perf_counter()-start)*1000.0, 'audio_duration_s': frame_count/sample_rate, 'output': str(path)}\n"
        "first_request_ms=synthesize(first_request, 0)['latency_ms']\n"
        "for index, request in enumerate(warmup_requests, start=-len(warmup_requests)):\n"
        "    synthesize(request, index)\n"
        "rows=[synthesize(request, index) for index, request in enumerate(requests, start=1)]\n"
        "print(json.dumps({'setup_ms': setup_ms, 'first_request_ms': first_request_ms, 'rows': rows}))\n"
    )


def _piper_espeak_data_dir() -> Path | None:
    spec = importlib.util.find_spec("piper")
    if spec is None or spec.origin is None:
        return None
    data_dir = Path(spec.origin).parent / "espeak-ng-data"
    return data_dir if data_dir.exists() else None


def _run_kokoro_tts_batch(phrases: tuple[str, ...], output_dir: Path) -> list[Measurement]:
    if importlib.util.find_spec("kokoro") is None:
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", "kokoro package not installed")]
    output_dir.mkdir(parents=True, exist_ok=True)
    system_espeak = find_system_espeak()
    code = build_kokoro_tts_batch_code(
        phrases=phrases,
        output_dir=output_dir,
        espeak_library_path=system_espeak[0] if system_espeak else None,
        espeak_data_path=system_espeak[1] if system_espeak else None,
    )
    return _run_json_batch_command([sys.executable, "-c", code])


def build_kokoro_tts_batch_code(
    *,
    phrases: tuple[str, ...],
    output_dir: Path,
    espeak_library_path: Path | None = None,
    espeak_data_path: Path | None = None,
) -> str:
    override = ""
    if espeak_library_path is not None and espeak_data_path is not None:
        override = (
            "import espeakng_loader\n"
            f"espeakng_loader.get_library_path=lambda: {str(espeak_library_path)!r}\n"
            f"espeakng_loader.get_data_path=lambda: {str(espeak_data_path)!r}\n"
        )
    return (
        "import json, time\n"
        "import soundfile as sf\n"
        f"{override}"
        "from kokoro import KPipeline\n"
        "setup_start=time.perf_counter()\n"
        "pipeline=KPipeline(lang_code='a')\n"
        "setup_ms=(time.perf_counter()-setup_start)*1000.0\n"
        f"phrases={list(phrases)!r}\n"
        f"out_dir={str(output_dir)!r}\n"
        "rows=[]\n"
        "for index, text in enumerate(phrases, start=1):\n"
        "    start=time.perf_counter()\n"
        "    gen=pipeline(text, voice='af_heart')\n"
        "    for _,_,audio in gen:\n"
        "        sf.write(f'{out_dir}/{index:02d}.wav', audio, 24000)\n"
        "        break\n"
        "    rows.append({'sample': text, 'latency_ms': (time.perf_counter()-start)*1000.0, 'output': f'{out_dir}/{index:02d}.wav'})\n"
        "print(json.dumps({'setup_ms': setup_ms, 'rows': rows}))\n"
    )


def find_system_espeak(
    *,
    library_candidates: tuple[Path, ...] | None = None,
    data_candidates: tuple[Path, ...] | None = None,
) -> tuple[Path, Path] | None:
    if library_candidates is None:
        library_candidates = (
            Path("/opt/homebrew/lib/libespeak-ng.dylib"),
            Path("/usr/local/lib/libespeak-ng.dylib"),
            Path("/usr/lib/libespeak-ng.so"),
            Path("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1"),
        )
    if data_candidates is None:
        data_candidates = (
            Path("/opt/homebrew/share/espeak-ng-data"),
            Path("/usr/local/share/espeak-ng-data"),
            Path("/usr/share/espeak-ng-data"),
        )
    library = next((path for path in library_candidates if path.exists()), None)
    data = next((path for path in data_candidates if (path / "phontab").exists()), None)
    if library is None or data is None:
        return None
    return library, data


def _run_sherpa_tts(text: str, output_dir: Path, index: int) -> Measurement:
    if importlib.util.find_spec("sherpa_onnx") is None:
        return Measurement(0.0, None, _gpu_memory_mb(), False, "", "sherpa_onnx package not installed", text)
    return Measurement(0.0, None, _gpu_memory_mb(), False, "", "sherpa_onnx TTS requires explicit model config", text)


def _run_whisper_cpp_stt(wav_path: Path, output_dir: Path, index: int) -> Measurement:
    command = os.environ.get("WHISPER_CPP_BIN") or shutil.which("whisper-cli") or shutil.which("main")
    model = os.environ.get("WHISPER_CPP_MODEL")
    if not command:
        return Measurement(0.0, None, _gpu_memory_mb(), False, "", "whisper.cpp command not found", str(wav_path))
    if not model:
        return Measurement(0.0, None, _gpu_memory_mb(), False, "", "WHISPER_CPP_MODEL is not set", str(wav_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run_command([command, "-m", model, "-f", str(wav_path), "-nt"], sample=str(wav_path))


def _run_vosk_stt_batch(sample_paths: tuple[Path, ...], output_dir: Path) -> list[Measurement]:
    if importlib.util.find_spec("vosk") is None:
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", "vosk package not installed")]
    model_path = os.environ.get("VOSK_MODEL")
    if not model_path:
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", "VOSK_MODEL is not set")]
    if not Path(model_path).exists():
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", f"VOSK_MODEL does not exist: {model_path}")]
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run_json_batch_command(
        [sys.executable, "-c", build_vosk_stt_batch_code(sample_paths=sample_paths, model_path=Path(model_path))]
    )


def build_vosk_stt_batch_code(*, sample_paths: tuple[Path, ...], model_path: Path) -> str:
    return (
        "import json, time, wave\n"
        "from vosk import Model, KaldiRecognizer, SetLogLevel\n"
        "SetLogLevel(-1)\n"
        "setup_start=time.perf_counter()\n"
        f"model=Model({str(model_path)!r})\n"
        "setup_ms=(time.perf_counter()-setup_start)*1000.0\n"
        f"paths={[str(path) for path in sample_paths]!r}\n"
        "rows=[]\n"
        "for path in paths:\n"
        "    start=time.perf_counter()\n"
        "    with wave.open(path, 'rb') as wf:\n"
        "        rec=KaldiRecognizer(model, wf.getframerate())\n"
        "        while True:\n"
        "            data=wf.readframes(4000)\n"
        "            if len(data)==0:\n"
        "                break\n"
        "            rec.AcceptWaveform(data)\n"
        "        result=json.loads(rec.FinalResult())\n"
        "    rows.append({'sample': path, 'latency_ms': (time.perf_counter()-start)*1000.0, 'output': result.get('text', '')})\n"
        "print(json.dumps({'setup_ms': setup_ms, 'rows': rows}))\n"
    )


def _run_faster_whisper_stt_batch(sample_paths: tuple[Path, ...], output_dir: Path) -> list[Measurement]:
    if importlib.util.find_spec("faster_whisper") is None:
        return [Measurement(0.0, None, _gpu_memory_mb(), False, "", "faster_whisper package not installed")]
    model = os.environ.get("FASTER_WHISPER_MODEL", "tiny.en")
    device = os.environ.get("FASTER_WHISPER_DEVICE", "auto")
    compute_type = os.environ.get("FASTER_WHISPER_COMPUTE_TYPE", "int8")
    code = (
        "import json, time\n"
        "from faster_whisper import WhisperModel\n"
        "setup_start=time.perf_counter()\n"
        f"model=WhisperModel({model!r}, device={device!r}, compute_type={compute_type!r})\n"
        "setup_ms=(time.perf_counter()-setup_start)*1000.0\n"
        f"paths={[str(path) for path in sample_paths]!r}\n"
        "rows=[]\n"
        "for path in paths:\n"
        "    start=time.perf_counter()\n"
        "    segments,_=model.transcribe(path, beam_size=1, vad_filter=False)\n"
        "    text=' '.join(s.text.strip() for s in segments)\n"
        "    rows.append({'sample': path, 'latency_ms': (time.perf_counter()-start)*1000.0, 'output': text})\n"
        "print(json.dumps({'setup_ms': setup_ms, 'rows': rows}))\n"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return _run_json_batch_command([sys.executable, "-c", code])


def build_faster_whisper_length_sweep_code(
    *,
    model: str,
    device: str,
    compute_type: str,
    first_request: dict[str, Any],
    warmup_requests: Sequence[dict[str, Any]],
    requests: Sequence[dict[str, Any]],
) -> str:
    return (
        "import json, time\n"
        "from faster_whisper import WhisperModel\n"
        "setup_start=time.perf_counter()\n"
        f"model=WhisperModel({model!r}, device={device!r}, compute_type={compute_type!r})\n"
        "setup_ms=(time.perf_counter()-setup_start)*1000.0\n"
        f"first_request={dict(first_request)!r}\n"
        f"warmup_requests={list(warmup_requests)!r}\n"
        f"requests={list(requests)!r}\n"
        "def transcribe(request):\n"
        "    start=time.perf_counter()\n"
        "    segments,_=model.transcribe(request['path'], beam_size=1, vad_filter=False)\n"
        "    text=' '.join(segment.text.strip() for segment in segments)\n"
        "    return {**request, 'latency_ms': (time.perf_counter()-start)*1000.0, 'output': text}\n"
        "first_request_ms=transcribe(first_request)['latency_ms']\n"
        "for request in warmup_requests:\n"
        "    transcribe(request)\n"
        "rows=[transcribe(request) for request in requests]\n"
        "print(json.dumps({'setup_ms': setup_ms, 'first_request_ms': first_request_ms, 'rows': rows}))\n"
    )


def _run_sherpa_stt(wav_path: Path, output_dir: Path, index: int) -> Measurement:
    if importlib.util.find_spec("sherpa_onnx") is None:
        return Measurement(0.0, None, _gpu_memory_mb(), False, "", "sherpa_onnx package not installed", str(wav_path))
    return Measurement(0.0, None, _gpu_memory_mb(), False, "", "sherpa_onnx ASR requires explicit model config", str(wav_path))


def _run_command(command: list[str], *, sample: str, input_text: str | None = None) -> Measurement:
    before_gpu = _gpu_memory_mb()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=float(os.environ.get("VOICE_BENCH_TIMEOUT_S", "120")),
        check=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    after_gpu = _gpu_memory_mb()
    output = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode != 0:
        return Measurement(elapsed_ms, _rss_to_mb(max(usage_before, usage_after)), after_gpu or before_gpu, False, output, output, sample)
    return Measurement(elapsed_ms, _rss_to_mb(max(usage_before, usage_after)), after_gpu or before_gpu, True, output, sample=sample)


def _run_json_batch_command(command: list[str]) -> list[Measurement]:
    before_gpu = _gpu_memory_mb()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=float(os.environ.get("VOICE_BENCH_TIMEOUT_S", "300")),
        check=False,
    )
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    after_gpu = _gpu_memory_mb()
    peak_rss = _rss_to_mb(max(usage_before, usage_after))
    gpu_peak = after_gpu or before_gpu
    if proc.returncode != 0:
        output = (proc.stdout or proc.stderr or "").strip()
        return [Measurement(0.0, peak_rss, gpu_peak, False, output, output)]
    try:
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        output = (proc.stdout or proc.stderr or "").strip()
        return [Measurement(0.0, peak_rss, gpu_peak, False, output, f"failed to parse JSON output: {exc}")]
    setup_ms = None
    if isinstance(payload, dict):
        setup_ms = payload.get("setup_ms")
        rows = payload.get("rows", [])
    else:
        rows = payload
    measurements = [
        Measurement(
            latency_ms=float(row["latency_ms"]),
            peak_rss_mb=peak_rss,
            gpu_peak_mb=gpu_peak,
            ok=True,
            output=str(row.get("output", "")),
            sample=str(row.get("sample", "")),
        )
        for row in rows
    ]
    summary_hint = getattr(_run_json_batch_command, "_last_setup_ms", None)
    _ = summary_hint
    for item in measurements:
        pass
    _run_json_batch_command._last_setup_ms = setup_ms  # type: ignore[attr-defined]
    return measurements


def _pop_last_batch_setup_ms() -> float | None:
    value = getattr(_run_json_batch_command, "_last_setup_ms", None)
    _run_json_batch_command._last_setup_ms = None  # type: ignore[attr-defined]
    return float(value) if value is not None else None


def _gpu_memory_mb() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    values = []
    for line in proc.stdout.splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return max(values) if values else None


def _rss_to_mb(value: int) -> float:
    if platform.system().lower() == "darwin":
        return round(float(value) / (1024.0 * 1024.0), 3)
    return round(float(value) / 1024.0, 3)


def _write_sample_wav(phrase: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    system = platform.system().lower()
    if system == "darwin" and shutil.which("say"):
        proc = subprocess.run(
            ["say", "-o", str(path), "--data-format=LEI16@16000", phrase],
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return
    command = shutil.which("espeak-ng") or shutil.which("espeak")
    if command:
        proc = subprocess.run([command, "-w", str(path), phrase], text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            return
    _write_placeholder_wav(path)


def _write_placeholder_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    frames = sample_rate // 4
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


def audio_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_count = int(handle.getnframes())
        sample_rate = int(handle.getframerate())
    if sample_rate <= 0:
        raise ValueError(f"WAV has invalid sample rate: {path}")
    return round(frame_count / float(sample_rate), 6)


def _sample_info(phrase: str, path: Path) -> SampleInfo:
    with wave.open(str(path), "rb") as handle:
        frame_count = int(handle.getnframes())
        sample_rate = int(handle.getframerate())
    return SampleInfo(
        phrase=phrase,
        path=path,
        duration_s=audio_duration_s(path),
        sample_rate_hz=sample_rate,
        frame_count=frame_count,
    )


def _unavailable(device: str, technology: str, kind: Kind, error: str) -> BackendResult:
    summary = summarize_measurements([Measurement(0.0, None, _gpu_memory_mb(), False, "", error)])
    return BackendResult(device, technology, kind, summary, ())


def _fmt(value: Any) -> str:
    if value is None:
        return "X"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _accuracy_fmt(summary: dict[str, Any]) -> str:
    if not summary.get("accuracy_available"):
        return "X"
    return f"{summary.get('exact_match_count', 0)}/{summary.get('accuracy_runs', 0)} ({float(summary.get('exact_match_rate', 0.0)):.1%})"


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.lower()).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
