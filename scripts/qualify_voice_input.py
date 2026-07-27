#!/usr/bin/env python3
"""Replay a consented local speech corpus through the production voice stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import shutil
import sys
import uuid
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
from urllib.parse import urlparse

import numpy as np

from streammuse.application.tasks import VoiceInputConfig
from streammuse.domain.tasks import SpeechContext, ZipZapZopTask
from streammuse.infrastructure.voice import (
    FasterWhisperRecognizer,
    TranscriptionResult,
    VoiceInfrastructureError,
)


SCHEMA_VERSION = "streammuse.voice_input_qualification.v1"
RESULTS_FILENAME = "samples.jsonl"
SUMMARY_FILENAME = "summary.json"
_NEGATIVE_LABEL = "<negative>"
_EMPTY_LABEL = "<empty>"
_UNRECOGNIZED_LABEL = "<unrecognized>"
_POSITIVE_CATEGORIES = frozenset({"zip", "zap", "zop", "combination", "number"})
_NEGATIVE_CATEGORIES = ("silence", "noise", "playback", "non_command")
_SAFE_CATEGORIES = _POSITIVE_CATEGORIES | frozenset(_NEGATIVE_CATEGORIES)
_DEVELOPMENT_SPLIT = "dev"
_HF_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
_SAFE_CATEGORY_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_EXPECTED_HF_REPOSITORY = "Systran/faster-whisper-tiny.en"
_VOICE_DEPENDENCIES = (
    "faster-whisper",
    "ctranslate2",
    "av",
    "onnxruntime",
    "tokenizers",
    "huggingface-hub",
    "sounddevice",
    "webrtcvad-wheels",
    "numpy",
    "scipy",
)
_MODEL_CRITICAL_FILES = frozenset(
    {
        "model.bin",
        "config.json",
        "tokenizer.json",
        "vocabulary.json",
        "vocabulary.txt",
        "preprocessor_config.json",
    }
)
_MODEL_REQUIRED_FILES = frozenset({"model.bin", "config.json"})


class QualificationError(RuntimeError):
    """Raised for an invalid qualification input or configuration."""


class Recognizer(Protocol):
    @property
    def provenance(self) -> Mapping[str, Any]: ...

    def start(self) -> None: ...

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        speech_context: SpeechContext | None = None,
    ) -> TranscriptionResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class CorpusSample:
    source_index: int
    audio_path: Path
    audio_sha256: str
    expected_raw: str
    expected_canonical: str | None
    split: str
    category: str
    semantic_category: str
    speaker: str | None = None
    session: str | None = None
    distance: str | None = None
    environment: str | None = None


@dataclass(frozen=True)
class GateThresholds:
    min_overall_accuracy: float = 0.95
    min_base_word_accuracy: float = 0.95
    min_combo_number_accuracy: float = 0.95
    max_negative_false_command_rate: float = 0.01
    max_positive_empty_transcript_rate: float = 0.05
    max_asr_p95_ms: float = 300.0
    min_speakers: int = 5
    min_base_word_samples: int = 100
    min_combination_samples: int = 100
    min_number_samples: int = 150
    min_negative_samples: int = 200
    min_negative_category_samples: int = 50

    def as_dict(self) -> dict[str, float | int]:
        return {
            "min_overall_accuracy": self.min_overall_accuracy,
            "min_base_word_accuracy": self.min_base_word_accuracy,
            "min_combo_number_accuracy": self.min_combo_number_accuracy,
            "max_negative_false_command_rate": self.max_negative_false_command_rate,
            "max_positive_empty_transcript_rate": self.max_positive_empty_transcript_rate,
            "max_asr_p95_ms": self.max_asr_p95_ms,
            "min_speakers": self.min_speakers,
            "min_base_word_samples": self.min_base_word_samples,
            "min_combination_samples": self.min_combination_samples,
            "min_number_samples": self.min_number_samples,
            "min_negative_samples": self.min_negative_samples,
            "min_negative_category_samples": self.min_negative_category_samples,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _optional_sha256(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _verify_file_hash(path: Path, expected: str | None, *, label: str) -> None:
    actual = _optional_sha256(path)
    if actual != expected:
        raise QualificationError(
            f"{label} changed during qualification: expected sha256={expected}, "
            f"found sha256={actual}"
        )


def _verify_frozen_inputs(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    samples: Sequence[CorpusSample],
    lock_path: Path,
    lock_sha256: str | None,
    verify_all_audio: bool,
) -> None:
    _verify_file_hash(manifest_path, manifest_sha256, label="manifest")
    _verify_file_hash(lock_path, lock_sha256, label="dependency lock")
    if verify_all_audio:
        for sample in samples:
            _verify_file_hash(
                sample.audio_path,
                sample.audio_sha256,
                label=f"sample {sample.source_index} audio",
            )


def _critical_model_tree(path: Path) -> dict[str, Any]:
    files = sorted(
        (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.name in _MODEL_CRITICAL_FILES
        ),
        key=lambda candidate: candidate.relative_to(path).as_posix(),
    )
    names = {candidate.name for candidate in files}
    missing_required = sorted(_MODEL_REQUIRED_FILES - names)
    if not files:
        return {
            "sha256": None,
            "file_count": 0,
            "files": [],
            "missing_required_files": sorted(_MODEL_REQUIRED_FILES),
        }
    digest = hashlib.sha256()
    evidence_files: list[dict[str, str]] = []
    for candidate in files:
        relative = candidate.relative_to(path).as_posix()
        file_sha256 = _sha256(candidate)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
        evidence_files.append({"path": relative, "sha256": file_sha256})
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "files": evidence_files,
        "missing_required_files": missing_required,
    }


def _static_source_baseline() -> dict[str, dict[str, str]]:
    values = {
        "qualification_script": Path(__file__).resolve(),
        "zip_zap_zop_parser": inspect.getsourcefile(ZipZapZopTask.parse_spoken_response),
    }
    baseline: dict[str, dict[str, str]] = {}
    for name, value in values.items():
        if value is None:
            raise QualificationError(f"could not resolve {name} source file")
        path = Path(value).resolve()
        if not path.is_file():
            raise QualificationError(f"{name} source is not a file: {path}")
        baseline[name] = {"path": str(path), "sha256": _sha256(path)}
    return baseline


def _source_baseline(
    recognizer: Recognizer,
    static_baseline: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    _verify_source_baseline(static_baseline)
    value = inspect.getsourcefile(type(recognizer))
    if value is None:
        raise QualificationError("could not resolve recognizer source file")
    path = Path(value).resolve()
    if not path.is_file():
        raise QualificationError(f"recognizer source is not a file: {path}")
    return {
        **{name: dict(evidence) for name, evidence in static_baseline.items()},
        "recognizer": {"path": str(path), "sha256": _sha256(path)},
    }


def _verify_source_baseline(baseline: Mapping[str, Mapping[str, str]]) -> None:
    for name, evidence in baseline.items():
        _verify_file_hash(
            Path(evidence["path"]),
            evidence["sha256"],
            label=f"{name} source",
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_output_isolation(
    output_dir: Path,
    staging_dir: Path,
    *,
    frozen_inputs: Sequence[Path],
) -> None:
    for candidate_name, candidate in (
        ("output directory", output_dir),
        ("staging directory", staging_dir),
    ):
        for frozen in frozen_inputs:
            if _paths_overlap(candidate, frozen):
                raise QualificationError(
                    f"{candidate_name} collides with or contains frozen input: {frozen}"
                )
    if output_dir.exists():
        raise QualificationError(f"output directory must be new: {output_dir}")
    if staging_dir.exists():
        raise QualificationError(f"staging directory already exists: {staging_dir}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(_canonical_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
    temporary.replace(path)


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise QualificationError(f"manifest does not exist or is not a file: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise QualificationError(
                    f"invalid JSON on manifest line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise QualificationError(f"manifest line {line_number} must be an object")
            rows.append(value)
        if not rows:
            raise QualificationError("manifest JSONL must contain at least one sample")
        return rows

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise QualificationError(f"invalid JSON manifest: {exc.msg}") from exc
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = payload.get("samples", payload.get("entries"))
        if values is None:
            raise QualificationError("JSON manifest object must contain 'samples' or 'entries'")
    else:
        raise QualificationError("JSON manifest must be a list or an object with samples/entries")
    if not isinstance(values, list) or not values:
        raise QualificationError("JSON manifest must contain at least one sample")
    if not all(isinstance(value, dict) for value in values):
        raise QualificationError("every manifest sample must be an object")
    return list(values)


def _local_audio_path(value: object, *, manifest_dir: Path, index: int) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise QualificationError(f"sample {index}: audio_path must be a non-empty string")
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc:
        raise QualificationError(
            f"sample {index}: audio_path must be a local filesystem path, not a URI"
        )
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise QualificationError(f"sample {index}: audio file does not exist: {path}") from exc
    if not resolved.is_file():
        raise QualificationError(f"sample {index}: audio_path is not a file: {resolved}")
    return resolved


def _nonempty_optional(value: object, *, field: str, index: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise QualificationError(f"sample {index}: {field} must be a string/integer or null")
    rendered = str(value).strip()
    if not rendered:
        raise QualificationError(f"sample {index}: {field} must not be empty")
    return rendered


def _safe_category_token(value: object, *, field: str, index: int) -> str | None:
    rendered = _nonempty_optional(value, field=field, index=index)
    if rendered is None:
        return None
    if not _SAFE_CATEGORY_TOKEN_PATTERN.fullmatch(rendered):
        raise QualificationError(
            f"sample {index}: {field} must match "
            f"{_SAFE_CATEGORY_TOKEN_PATTERN.pattern!r}"
        )
    return rendered


def _parse_expected(
    task: ZipZapZopTask,
    value: object,
    *,
    index: int,
) -> tuple[str, str | None]:
    if value is None:
        return "", None
    if not isinstance(value, str):
        raise QualificationError(f"sample {index}: expected must be a string or null")
    rendered = value.strip()
    if not rendered:
        return "", None
    parsed = task.parse_spoken_response(task.initial_state(), [], rendered)
    if parsed.canonical_text is None:
        raise QualificationError(
            f"sample {index}: expected value is not valid Zip-Zap-Zop speech: {value!r}"
        )
    return rendered, parsed.canonical_text


def _semantic_category(expected: str | None) -> str:
    if expected is None:
        return "negative"
    if expected in {"Zip", "Zap", "Zop"}:
        return expected.lower()
    if expected.isdigit() or (expected.startswith("-") and expected[1:].isdigit()):
        return "number"
    if expected.lower() in {"zipzap", "zipzop", "zapzop", "zipzapzop"}:
        return "combination"
    return "other"


def load_manifest(path: Path, task: ZipZapZopTask | None = None) -> list[CorpusSample]:
    """Load and validate a local JSON/JSONL corpus manifest."""

    path = path.expanduser().resolve()
    task = task or ZipZapZopTask()
    samples: list[CorpusSample] = []
    seen_audio_paths: dict[Path, int] = {}
    seen_audio_hashes: dict[str, int] = {}
    for index, row in enumerate(_read_manifest_rows(path), 1):
        missing = {"audio_path", "expected", "split"} - set(row)
        if missing:
            raise QualificationError(
                f"sample {index}: missing required field(s): {', '.join(sorted(missing))}"
            )
        split = _nonempty_optional(row["split"], field="split", index=index)
        assert split is not None
        audio_path = _local_audio_path(
            row["audio_path"], manifest_dir=path.parent, index=index
        )
        duplicate_path_index = seen_audio_paths.get(audio_path)
        if duplicate_path_index is not None:
            raise QualificationError(
                f"sample {index}: audio_path duplicates sample {duplicate_path_index}: "
                f"{audio_path}"
            )
        audio_sha256 = _sha256(audio_path)
        duplicate_content_index = seen_audio_hashes.get(audio_sha256)
        if duplicate_content_index is not None:
            raise QualificationError(
                f"sample {index}: audio content duplicates sample "
                f"{duplicate_content_index} (sha256={audio_sha256})"
            )
        seen_audio_paths[audio_path] = index
        seen_audio_hashes[audio_sha256] = index
        expected_raw, expected_canonical = _parse_expected(task, row["expected"], index=index)
        semantic_category = _semantic_category(expected_canonical)
        explicit_category = _nonempty_optional(row.get("category"), field="category", index=index)
        if explicit_category is not None and explicit_category not in _SAFE_CATEGORIES:
            allowed = ", ".join(sorted(_SAFE_CATEGORIES))
            raise QualificationError(
                f"sample {index}: category must be one of: {allowed}"
            )
        if expected_canonical is None:
            if explicit_category not in _NEGATIVE_CATEGORIES:
                allowed = ", ".join(_NEGATIVE_CATEGORIES)
                raise QualificationError(
                    f"sample {index}: a negative sample requires category one of: {allowed}"
                )
            safe_category = explicit_category
        else:
            if explicit_category is not None and explicit_category != semantic_category:
                raise QualificationError(
                    f"sample {index}: category {explicit_category!r} does not match "
                    f"the derived category {semantic_category!r}"
                )
            safe_category = semantic_category
        samples.append(
            CorpusSample(
                source_index=index,
                audio_path=audio_path,
                audio_sha256=audio_sha256,
                expected_raw=expected_raw,
                expected_canonical=expected_canonical,
                split=split,
                category=safe_category,
                semantic_category=semantic_category,
                speaker=_nonempty_optional(row.get("speaker"), field="speaker", index=index),
                session=_nonempty_optional(row.get("session"), field="session", index=index),
                distance=_safe_category_token(row.get("distance"), field="distance", index=index),
                environment=_safe_category_token(
                    row.get("environment"), field="environment", index=index
                ),
            )
        )
    return samples


def _pcm24_to_float32(data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size % 3:
        raise QualificationError("24-bit WAV data has an incomplete PCM sample")
    triples = raw.reshape(-1, 3).astype(np.int32)
    values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    values = np.where(values & 0x800000, values - 0x1000000, values)
    return values.astype(np.float32) / 8388608.0


def _load_pcm_wav(path: Path) -> np.ndarray:
    try:
        with wave.open(str(path), "rb") as handle:
            if handle.getcomptype() != "NONE":
                raise QualificationError(f"compressed WAV is not supported: {path}")
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frame_count = handle.getnframes()
            data = handle.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise QualificationError(f"could not decode PCM WAV {path}: {exc}") from exc
    if channels <= 0 or sample_rate <= 0:
        raise QualificationError(f"WAV has invalid channel count or sample rate: {path}")
    if sample_width == 1:
        audio = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        audio = _pcm24_to_float32(data)
    elif sample_width == 4:
        audio = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise QualificationError(f"unsupported WAV sample width ({sample_width} bytes): {path}")
    if audio.size % channels:
        raise QualificationError(f"WAV sample data is not aligned to its channel count: {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    if sample_rate != 16_000 and audio.size:
        try:
            from scipy.signal import resample_poly
        except ImportError as exc:  # pragma: no cover - base dependency
            raise QualificationError("audio resampling requires scipy") from exc
        divisor = gcd(sample_rate, 16_000)
        audio = resample_poly(audio, 16_000 // divisor, sample_rate // divisor)
    return np.ascontiguousarray(audio, dtype=np.float32)


def load_audio(path: Path) -> np.ndarray:
    """Decode a local audio file into the recognizer's 16-kHz mono format."""

    if path.suffix.lower() in {".wav", ".wave"}:
        audio = _load_pcm_wav(path)
    else:
        try:
            from faster_whisper.audio import decode_audio  # type: ignore[import-not-found]
        except ImportError as exc:
            raise QualificationError(
                "non-WAV corpus audio requires the voice extra; use "
                "`uv run --frozen --extra voice`"
            ) from exc
        try:
            audio = decode_audio(str(path), sampling_rate=16_000)
        except Exception as exc:
            raise QualificationError(f"could not decode local audio file {path}: {exc}") from exc
        audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise QualificationError(f"decoded audio must be mono: {path}")
    if audio.size == 0:
        return np.empty(0, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise QualificationError(f"decoded audio contains non-finite samples: {path}")
    return np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32)


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float] | None:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return {"lower": max(0.0, center - margin), "upper": min(1.0, center + margin)}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _confusion_label(row: Mapping[str, Any], *, expected: bool) -> str:
    if expected:
        return str(row["expected_canonical"] or _NEGATIVE_LABEL)
    prediction = row.get("predicted_canonical")
    if prediction:
        return str(prediction)
    return _EMPTY_LABEL if not row.get("raw_transcript") else _UNRECOGNIZED_LABEL


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    positive = [row for row in rows if row.get("expected_canonical") is not None]
    negative = [row for row in rows if row.get("expected_canonical") is None]
    raw_successes = sum(bool(row["raw_exact"]) for row in rows)
    canonical_successes = sum(bool(row["canonical_exact"]) for row in rows)
    false_commands = sum(bool(row["false_game_command"]) for row in negative)
    empty_transcripts = sum(bool(row["positive_empty_transcript"]) for row in positive)
    quality_rejected = [
        row for row in rows if bool(row.get("transcript_quality_rejected"))
    ]
    positive_quality_rejected = [
        row for row in positive if bool(row.get("transcript_quality_rejected"))
    ]
    negative_quality_rejected = [
        row for row in negative if bool(row.get("transcript_quality_rejected"))
    ]
    rejection_reasons = Counter(
        str(reason)
        for row in quality_rejected
        for reason in row.get("transcript_rejection_reasons", [])
    )
    confusion = Counter(
        (_confusion_label(row, expected=True), _confusion_label(row, expected=False))
        for row in rows
    )

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "count": count,
        "positive_count": len(positive),
        "negative_count": len(negative),
        "raw_exact": {
            "correct": raw_successes,
            "accuracy": rate(raw_successes, count),
            "wilson_95": wilson_interval(raw_successes, count),
        },
        "canonical_exact": {
            "correct": canonical_successes,
            "accuracy": rate(canonical_successes, count),
            "wilson_95": wilson_interval(canonical_successes, count),
        },
        "negative_false_game_command": {
            "count": false_commands,
            "rate": rate(false_commands, len(negative)),
            "wilson_95": wilson_interval(false_commands, len(negative)),
        },
        "positive_empty_transcript": {
            "count": empty_transcripts,
            "rate": rate(empty_transcripts, len(positive)),
            "wilson_95": wilson_interval(empty_transcripts, len(positive)),
        },
        "transcript_quality_rejected": {
            "count": len(quality_rejected),
            "rate": rate(len(quality_rejected), count),
            "positive_count": len(positive_quality_rejected),
            "positive_rate": rate(len(positive_quality_rejected), len(positive)),
            "negative_count": len(negative_quality_rejected),
            "negative_rate": rate(len(negative_quality_rejected), len(negative)),
            "reason_counts": dict(sorted(rejection_reasons.items())),
        },
        "asr_latency_ms": {
            "p50": _percentile([float(row["asr_latency_ms"]) for row in rows], 50),
            "p95": _percentile([float(row["asr_latency_ms"]) for row in rows], 95),
        },
        "confusion_counts": [
            {"expected": expected, "predicted": predicted, "count": value}
            for (expected, predicted), value in sorted(confusion.items())
        ],
    }


def _group_summaries(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: summarize_rows(grouped[name]) for name in sorted(grouped)}


def _check(
    name: str,
    actual: float | int | None,
    operator: str,
    threshold: float | int,
    *,
    detail: str | None = None,
) -> dict[str, Any]:
    if actual is None:
        passed = False
    elif operator == ">=":
        passed = actual >= threshold
    elif operator == "<=":
        passed = actual <= threshold
    elif operator == "==":
        passed = actual == threshold
    else:  # pragma: no cover - internal contract
        raise AssertionError(f"unsupported gate operator: {operator}")
    result: dict[str, Any] = {
        "name": name,
        "passed": passed,
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
    }
    if detail:
        result["detail"] = detail
    return result


def evaluate_gates(
    samples: Sequence[CorpusSample],
    rows: Sequence[Mapping[str, Any]],
    *,
    acceptance_split: str,
    thresholds: GateThresholds,
    reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    selected_samples = [sample for sample in samples if sample.split == acceptance_split]
    selected = [row for row in rows if row["split"] == acceptance_split]
    semantic_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in selected:
        semantic_rows[str(row["semantic_category"])].append(row)

    speakers = {sample.speaker for sample in selected_samples if sample.speaker is not None}
    development_samples = [sample for sample in samples if sample.split == _DEVELOPMENT_SPLIT]
    development_speakers = {
        sample.speaker for sample in development_samples if sample.speaker is not None
    }
    missing_speakers = sum(sample.speaker is None for sample in samples)
    missing_sessions = sum(sample.session is None for sample in samples)
    missing_distances = sum(sample.distance is None for sample in samples)
    missing_environments = sum(sample.environment is None for sample in samples)
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.speaker is not None:
            speaker_splits[sample.speaker].add(sample.split)
    overlapping_speakers = sum(
        speaker in speakers and any(split != acceptance_split for split in splits)
        for speaker, splits in speaker_splits.items()
    )
    acceptance_development_overlap = len(speakers & development_speakers)
    acceptance_sessions = {
        sample.session for sample in selected_samples if sample.session is not None
    }
    development_sessions = {
        sample.session for sample in development_samples if sample.session is not None
    }
    session_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.session is not None:
            session_splits[sample.session].add(sample.split)
    overlapping_sessions = sum(len(splits) > 1 for splits in session_splits.values())
    acceptance_development_session_overlap = len(
        acceptance_sessions & development_sessions
    )
    acceptance_metrics = summarize_rows(selected)
    model_evidence = reproducibility["model"]
    lock_evidence = reproducibility["dependency_lock"]
    resolved_model_path = model_evidence.get("resolved_path")
    resolved_path_exists = bool(
        isinstance(resolved_model_path, str)
        and resolved_model_path
        and Path(resolved_model_path).is_dir()
    )
    lock_sha256 = lock_evidence.get("sha256")
    is_hf_snapshot = model_evidence.get("evidence_mode") == "huggingface_snapshot"
    local_tree_sha256 = model_evidence.get("critical_file_tree", {}).get("sha256")
    formal_threshold_profile = thresholds == GateThresholds()
    formal_qualification_profile = bool(
        formal_threshold_profile
        and reproducibility.get("decode_context_profile") == "task"
        and model_evidence.get("configured") == "tiny.en"
        and model_evidence.get("device") == "cpu"
        and model_evidence.get("compute_type") == "int8"
        and model_evidence.get("local_files_only") is True
        and is_hf_snapshot
        and model_evidence.get("snapshot_repository") == _EXPECTED_HF_REPOSITORY
        and model_evidence.get("evidence_valid")
    )

    checks: list[dict[str, Any]] = []
    checks.append(
        _check("formal_threshold_profile", int(formal_threshold_profile), "==", 1)
    )
    checks.append(
        _check(
            "production_decode_context_profile",
            int(reproducibility.get("decode_context_profile") == "task"),
            "==",
            1,
        )
    )
    checks.append(_check("acceptance_sample_count", len(selected), ">=", 1))
    checks.append(
        _check(
            "development_split_is_distinct",
            int(acceptance_split != _DEVELOPMENT_SPLIT),
            "==",
            1,
        )
    )
    checks.append(_check("development_sample_count", len(development_samples), ">=", 1))
    checks.append(_check("development_speaker_count", len(development_speakers), ">=", 1))
    checks.append(_check("speaker_count", len(speakers), ">=", thresholds.min_speakers))
    checks.append(_check("samples_missing_speaker", missing_speakers, "==", 0))
    checks.append(_check("samples_missing_session", missing_sessions, "==", 0))
    checks.append(_check("samples_missing_distance", missing_distances, "==", 0))
    checks.append(_check("samples_missing_environment", missing_environments, "==", 0))
    checks.append(_check("speaker_split_overlap_count", overlapping_speakers, "==", 0))
    checks.append(_check("session_split_overlap_count", overlapping_sessions, "==", 0))
    checks.append(
        _check(
            "acceptance_development_speaker_overlap_count",
            acceptance_development_overlap,
            "==",
            0,
        )
    )
    checks.append(
        _check(
            "acceptance_development_session_overlap_count",
            acceptance_development_session_overlap,
            "==",
            0,
        )
    )
    checks.append(
        _check(
            "configured_model_is_tiny_en",
            int(model_evidence.get("configured") == "tiny.en"),
            "==",
            1,
        )
    )
    checks.append(
        _check("configured_device_is_cpu", int(model_evidence.get("device") == "cpu"), "==", 1)
    )
    checks.append(
        _check(
            "configured_compute_type_is_int8",
            int(model_evidence.get("compute_type") == "int8"),
            "==",
            1,
        )
    )
    checks.append(
        _check(
            "configured_local_files_only",
            int(model_evidence.get("local_files_only") is True),
            "==",
            1,
        )
    )
    checks.append(
        _check("model_is_huggingface_snapshot", int(is_hf_snapshot), "==", 1)
    )
    checks.append(
        _check(
            "model_huggingface_repository",
            int(model_evidence.get("snapshot_repository") == _EXPECTED_HF_REPOSITORY),
            "==",
            1,
        )
    )
    checks.append(
        _check(
            "resolved_model_path_recorded",
            int(isinstance(resolved_model_path, str) and bool(resolved_model_path)),
            "==",
            1,
        )
    )
    checks.append(_check("resolved_model_path_exists", int(resolved_path_exists), "==", 1))
    checks.append(
        _check(
            "model_reproducibility_evidence_valid",
            int(bool(model_evidence.get("evidence_valid"))),
            "==",
            1,
        )
    )
    checks.append(
        _check(
            "hf_snapshot_commit_valid",
            int(not is_hf_snapshot or bool(model_evidence.get("snapshot_commit_valid"))),
            "==",
            1,
        )
    )
    checks.append(
        _check(
            "hf_requested_revision_pinned",
            int(not is_hf_snapshot or bool(model_evidence.get("requested_revision_pinned"))),
            "==",
            1,
        )
    )
    checks.append(
        _check(
            "model_critical_file_tree_recorded",
            int(isinstance(local_tree_sha256, str) and len(local_tree_sha256) == 64),
            "==",
            1,
        )
    )
    checks.append(
        _check(
            "dependency_lock_sha256_recorded",
            int(isinstance(lock_sha256, str) and len(lock_sha256) == 64),
            "==",
            1,
        )
    )
    for name in _VOICE_DEPENDENCIES:
        package_evidence = lock_evidence["packages"][name]
        checks.append(
            _check(
                f"dependency_{name.replace('-', '_')}_matches_lock",
                int(bool(package_evidence.get("matches_lock"))),
                "==",
                1,
                detail=(
                    f"installed={package_evidence.get('installed')!r}, "
                    f"locked={package_evidence.get('locked')!r}"
                ),
            )
        )
    for category in ("zip", "zap", "zop"):
        category_rows = semantic_rows[category]
        checks.append(
            _check(
                f"{category}_sample_count",
                len(category_rows),
                ">=",
                thresholds.min_base_word_samples,
            )
        )
        checks.append(
            _check(
                f"{category}_canonical_accuracy",
                summarize_rows(category_rows)["canonical_exact"]["accuracy"],
                ">=",
                thresholds.min_base_word_accuracy,
            )
        )
    combinations = semantic_rows["combination"]
    numbers = semantic_rows["number"]
    negatives = semantic_rows["negative"]
    negative_category_counts = Counter(
        sample.category
        for sample in selected_samples
        if sample.semantic_category == "negative"
    )
    checks.extend(
        [
            _check(
                "combination_sample_count",
                len(combinations),
                ">=",
                thresholds.min_combination_samples,
            ),
            _check("number_sample_count", len(numbers), ">=", thresholds.min_number_samples),
            _check("negative_sample_count", len(negatives), ">=", thresholds.min_negative_samples),
            _check(
                "overall_canonical_accuracy",
                acceptance_metrics["canonical_exact"]["accuracy"],
                ">=",
                thresholds.min_overall_accuracy,
            ),
            _check(
                "combination_number_canonical_accuracy",
                summarize_rows([*combinations, *numbers])["canonical_exact"]["accuracy"],
                ">=",
                thresholds.min_combo_number_accuracy,
            ),
            _check(
                "negative_false_game_command_rate",
                summarize_rows(negatives)["negative_false_game_command"]["rate"],
                "<=",
                thresholds.max_negative_false_command_rate,
            ),
            _check(
                "positive_empty_transcript_rate",
                acceptance_metrics["positive_empty_transcript"]["rate"],
                "<=",
                thresholds.max_positive_empty_transcript_rate,
            ),
            _check(
                "asr_latency_p95_ms",
                acceptance_metrics["asr_latency_ms"]["p95"],
                "<=",
                thresholds.max_asr_p95_ms,
            ),
        ]
    )
    for category in _NEGATIVE_CATEGORIES:
        category_rows = [row for row in negatives if row["category"] == category]
        category_false_command_rate = summarize_rows(category_rows)[
            "negative_false_game_command"
        ]["rate"]
        checks.append(
            _check(
                f"{category}_negative_sample_count",
                negative_category_counts[category],
                ">=",
                thresholds.min_negative_category_samples,
            )
        )
        checks.append(
            _check(
                f"{category}_negative_false_game_command_rate",
                category_false_command_rate,
                "<=",
                thresholds.max_negative_false_command_rate,
            )
        )
    return {
        "qualification_profile": "formal" if formal_qualification_profile else "exploratory",
        "acceptance_split": acceptance_split,
        "thresholds": thresholds.as_dict(),
        "protocol": {
            "sample_count": len(selected_samples),
            "speaker_count": len(speakers),
            "development_split": _DEVELOPMENT_SPLIT,
            "development_sample_count": len(development_samples),
            "development_speaker_count": len(development_speakers),
            "samples_missing_speaker": missing_speakers,
            "samples_missing_session": missing_sessions,
            "samples_missing_distance": missing_distances,
            "samples_missing_environment": missing_environments,
            "speaker_split_overlap_count": overlapping_speakers,
            "session_split_overlap_count": overlapping_sessions,
            "acceptance_development_speaker_overlap_count": acceptance_development_overlap,
            "acceptance_development_session_overlap_count": (
                acceptance_development_session_overlap
            ),
            "acceptance_session_count": len(acceptance_sessions),
            "development_session_count": len(development_sessions),
            "distance_category_counts": dict(
                sorted(
                    Counter(
                        sample.distance
                        for sample in selected_samples
                        if sample.distance is not None
                    ).items()
                )
            ),
            "environment_category_counts": dict(
                sorted(
                    Counter(
                        sample.environment
                        for sample in selected_samples
                        if sample.environment is not None
                    ).items()
                )
            ),
            "semantic_category_counts": dict(
                sorted(Counter(sample.semantic_category for sample in selected_samples).items())
            ),
            "negative_category_counts": {
                category: negative_category_counts[category]
                for category in _NEGATIVE_CATEGORIES
            },
            "negative_category_false_game_command_rates": {
                category: summarize_rows(
                    [row for row in negatives if row["category"] == category]
                )["negative_false_game_command"]["rate"]
                for category in _NEGATIVE_CATEGORIES
            },
        },
        "checks": checks,
        "not_measured": [
            "endpoint-caused onset truncation",
            "capture queue overflow",
            "last-voice-frame-to-text latency including endpoint lag",
            "scripted microphone deadline success rate",
            "positive false-no-speech rate from microphone/VAD endpointing",
        ],
        "passed": all(bool(check["passed"]) for check in checks),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _marker_applies(values: object) -> bool:
    if values is None:
        return True
    markers = values if isinstance(values, list) else [values]
    try:
        from packaging.markers import InvalidMarker, Marker
    except ImportError as exc:  # pragma: no cover - packaging is a base dependency
        raise QualificationError("parsing uv.lock markers requires packaging") from exc
    for value in markers:
        if not isinstance(value, str):
            raise QualificationError("uv.lock resolution-markers must be strings")
        try:
            if Marker(value).evaluate():
                return True
        except InvalidMarker as exc:
            raise QualificationError(f"invalid uv.lock resolution marker: {value}") from exc
    return False


def _read_lock_packages(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {name: [] for name in _VOICE_DEPENDENCIES}
    try:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - Python 3.10
            import tomli as tomllib  # type: ignore[no-redef]

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise QualificationError(f"could not parse dependency lock {path}: {exc}") from exc
    packages = payload.get("package")
    if not isinstance(packages, list):
        raise QualificationError("uv.lock must contain a package array")
    applicable: dict[str, list[str]] = {name: [] for name in _VOICE_DEPENDENCIES}
    for package in packages:
        if not isinstance(package, dict):
            continue
        name = str(package.get("name", "")).lower().replace("_", "-")
        if name not in applicable or not _marker_applies(package.get("resolution-markers")):
            continue
        version = package.get("version")
        if isinstance(version, str) and version not in applicable[name]:
            applicable[name].append(version)
    return applicable


def _dependency_lock(
    path: Path,
    *,
    package_version_getter: Callable[[str], str | None],
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    lock_sha256 = _optional_sha256(path)
    locked_versions = _read_lock_packages(path)
    _verify_file_hash(path, lock_sha256, label="dependency lock")
    packages: dict[str, dict[str, Any]] = {}
    for name in _VOICE_DEPENDENCIES:
        installed = package_version_getter(name)
        applicable = locked_versions[name]
        packages[name] = {
            "installed": installed,
            "locked": applicable,
            "matches_lock": bool(
                installed is not None
                and len(applicable) == 1
                and installed == applicable[0]
            ),
        }
    return {
        "path": str(path),
        "sha256": lock_sha256,
        "packages": packages,
    }


def _environment(
    dependency_lock: Mapping[str, Any],
    *,
    package_version_getter: Callable[[str], str | None],
) -> dict[str, Any]:
    voice_versions = {
        name: dependency_lock["packages"][name]["installed"]
        for name in _VOICE_DEPENDENCIES
    }
    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": {"streammuse": package_version_getter("streammuse"), **voice_versions},
        "voice_dependencies": voice_versions,
    }


def _snapshot_commit(path: Path) -> str | None:
    indexes = [index for index, part in enumerate(path.parts) if part == "snapshots"]
    if not indexes:
        return None
    index = indexes[-1]
    return path.parts[index + 1] if index + 1 < len(path.parts) else ""


def _snapshot_repository(path: Path) -> str | None:
    indexes = [index for index, part in enumerate(path.parts) if part == "snapshots"]
    if not indexes:
        return None
    index = indexes[-1]
    if index == 0:
        return None
    encoded = path.parts[index - 1]
    if not encoded.startswith("models--"):
        return None
    return encoded.removeprefix("models--").replace("--", "/")


def _model_baseline(
    provenance: Mapping[str, Any],
    config: VoiceInputConfig,
) -> dict[str, Any]:
    resolved_path_value = provenance.get("model_path_resolved")
    resolved_path = (
        Path(resolved_path_value).expanduser().resolve()
        if isinstance(resolved_path_value, str) and resolved_path_value
        else None
    )
    path_exists = bool(resolved_path is not None and resolved_path.is_dir())
    snapshot_commit = _snapshot_commit(resolved_path) if resolved_path is not None else None
    snapshot_repository = (
        _snapshot_repository(resolved_path) if resolved_path is not None else None
    )
    is_hf_snapshot = snapshot_commit is not None
    resolved_revision = provenance.get("model_revision_resolved")
    requested_revision = provenance.get("model_revision_requested")
    commit_valid = bool(
        is_hf_snapshot
        and isinstance(snapshot_commit, str)
        and _HF_COMMIT_PATTERN.fullmatch(snapshot_commit)
        and resolved_revision == snapshot_commit
    )
    requested_revision_pinned = bool(
        commit_valid
        and isinstance(requested_revision, str)
        and _HF_COMMIT_PATTERN.fullmatch(requested_revision)
        and requested_revision == snapshot_commit
    )
    critical_tree = (
        _critical_model_tree(resolved_path)
        if path_exists and resolved_path is not None
        else {
            "sha256": None,
            "file_count": 0,
            "files": [],
            "missing_required_files": sorted(_MODEL_REQUIRED_FILES),
        }
    )
    critical_tree_valid = bool(
        path_exists
        and critical_tree["sha256"]
        and not critical_tree["missing_required_files"]
    )
    return {
        "configured": config.model,
        "device": config.device,
        "compute_type": config.compute_type,
        "local_files_only": config.local_files_only,
        "resolved_path": str(resolved_path) if resolved_path is not None else None,
        "path_exists": path_exists,
        "evidence_mode": "huggingface_snapshot" if is_hf_snapshot else "local_directory",
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "snapshot_repository": snapshot_repository,
        "snapshot_commit": snapshot_commit,
        "snapshot_commit_valid": commit_valid,
        "requested_revision_pinned": requested_revision_pinned,
        "critical_file_tree": critical_tree,
        "evidence_valid": bool(
            critical_tree_valid
            and (requested_revision_pinned if is_hf_snapshot else True)
        ),
    }


def _verify_model_baseline(
    baseline: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    resolved_path_value = provenance.get("model_path_resolved")
    resolved_path = (
        str(Path(resolved_path_value).expanduser().resolve())
        if isinstance(resolved_path_value, str) and resolved_path_value
        else None
    )
    if resolved_path != baseline.get("resolved_path"):
        raise QualificationError("resolved model path changed during qualification")
    if provenance.get("model_revision_resolved") != baseline.get("resolved_revision"):
        raise QualificationError("resolved model revision changed during qualification")
    if provenance.get("model_revision_requested") != baseline.get("requested_revision"):
        raise QualificationError("requested model revision changed during qualification")
    if resolved_path is None:
        raise QualificationError("resolved model path is unavailable")
    current_tree = _critical_model_tree(Path(resolved_path))
    if current_tree != baseline.get("critical_file_tree"):
        raise QualificationError("resolved model critical file tree changed during qualification")


def _reproducibility(
    model_baseline: Mapping[str, Any],
    dependency_lock: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": dict(model_baseline),
        "dependency_lock": dict(dependency_lock),
    }


def _speech_context(args: argparse.Namespace, task: ZipZapZopTask) -> SpeechContext | None:
    if args.context_profile == "baseline":
        if args.initial_prompt or args.hotword:
            raise QualificationError(
                "--initial-prompt/--hotword require --context-profile custom"
            )
        return None
    if args.context_profile == "task":
        if args.initial_prompt or args.hotword:
            raise QualificationError("task context cannot be combined with custom prompt/hotwords")
        return task.build_speech_context(task.initial_state(), [])
    return SpeechContext(
        initial_prompt=args.initial_prompt,
        hotwords=tuple(args.hotword or ()),
    )


def _thresholds(args: argparse.Namespace) -> GateThresholds:
    return GateThresholds(
        min_overall_accuracy=args.min_overall_accuracy,
        min_base_word_accuracy=args.min_base_word_accuracy,
        min_combo_number_accuracy=args.min_combo_number_accuracy,
        max_negative_false_command_rate=args.max_negative_false_command_rate,
        max_positive_empty_transcript_rate=args.max_positive_empty_transcript_rate,
        max_asr_p95_ms=args.max_asr_p95_ms,
        min_speakers=args.min_speakers,
        min_base_word_samples=args.min_base_word_samples,
        min_combination_samples=args.min_combination_samples,
        min_number_samples=args.min_number_samples,
        min_negative_samples=args.min_negative_samples,
        min_negative_category_samples=args.min_negative_category_samples,
    )


def run_qualification(
    args: argparse.Namespace,
    *,
    recognizer_factory: Callable[[VoiceInputConfig], Recognizer] = FasterWhisperRecognizer,
    audio_loader: Callable[[Path], np.ndarray] = load_audio,
    dependency_lock_path: Path | None = None,
    package_version_getter: Callable[[str], str | None] = _package_version,
) -> dict[str, Any]:
    """Execute one deterministic decode configuration and write its evidence."""

    static_source_baseline = _static_source_baseline()
    manifest_path = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise QualificationError(f"output directory must be new: {output_dir}")
    artifact_set_id = uuid.uuid4().hex
    staging_dir = output_dir.with_name(
        f".{output_dir.name}.staging-{artifact_set_id}"
    )
    lock_path = (
        dependency_lock_path or (Path(__file__).resolve().parents[1] / "uv.lock")
    ).expanduser().resolve()
    manifest_sha256 = _sha256(manifest_path)
    dependency_lock = _dependency_lock(
        lock_path,
        package_version_getter=package_version_getter,
    )
    task = ZipZapZopTask()
    samples = load_manifest(manifest_path, task)
    _verify_frozen_inputs(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        samples=samples,
        lock_path=lock_path,
        lock_sha256=dependency_lock["sha256"],
        verify_all_audio=True,
    )
    initial_frozen_paths = [
        manifest_path,
        lock_path,
        *(sample.audio_path for sample in samples),
        *(Path(value["path"]) for value in static_source_baseline.values()),
    ]
    _validate_output_isolation(
        output_dir,
        staging_dir,
        frozen_inputs=initial_frozen_paths,
    )
    context = _speech_context(args, task)
    voice_config = VoiceInputConfig(
        model=args.model,
        device=args.device,
        compute_type=args.compute_type,
        model_cache=args.model_cache,
        model_revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    recognizer = recognizer_factory(voice_config)
    source_baseline = _source_baseline(recognizer, static_source_baseline)
    frozen_paths = [
        manifest_path,
        lock_path,
        *(sample.audio_path for sample in samples),
        *(Path(value["path"]) for value in source_baseline.values()),
    ]
    _validate_output_isolation(
        output_dir,
        staging_dir,
        frozen_inputs=frozen_paths,
    )
    results: list[dict[str, Any]] = []
    primary_error: BaseException | None = None
    model_baseline: dict[str, Any] | None = None
    provenance: dict[str, Any] = {}
    try:
        recognizer.start()
        provenance = dict(recognizer.provenance)
        model_baseline = _model_baseline(provenance, voice_config)
        resolved_model_path = model_baseline.get("resolved_path")
        if isinstance(resolved_model_path, str):
            _validate_output_isolation(
                output_dir,
                staging_dir,
                frozen_inputs=[*frozen_paths, Path(resolved_model_path)],
            )
        for sample in samples:
            _verify_frozen_inputs(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                samples=samples,
                lock_path=lock_path,
                lock_sha256=dependency_lock["sha256"],
                verify_all_audio=False,
            )
            _verify_file_hash(
                sample.audio_path,
                sample.audio_sha256,
                label=f"sample {sample.source_index} audio",
            )
            audio = audio_loader(sample.audio_path)
            _verify_file_hash(
                sample.audio_path,
                sample.audio_sha256,
                label=f"sample {sample.source_index} audio",
            )
            transcription = recognizer.transcribe(audio, speech_context=context)
            _verify_frozen_inputs(
                manifest_path=manifest_path,
                manifest_sha256=manifest_sha256,
                samples=samples,
                lock_path=lock_path,
                lock_sha256=dependency_lock["sha256"],
                verify_all_audio=False,
            )
            _verify_file_hash(
                sample.audio_path,
                sample.audio_sha256,
                label=f"sample {sample.source_index} audio",
            )
            raw_text = transcription.text.strip()
            rejection_reasons = tuple(
                str(reason)
                for reason in getattr(transcription, "rejection_reasons", ())
            )
            if rejection_reasons:
                predicted = None
                parse_status = "rejected_transcript"
                parse_reason = ",".join(rejection_reasons)
            else:
                parsed = task.parse_spoken_response(task.initial_state(), [], raw_text)
                predicted = parsed.canonical_text
                parse_status = parsed.status
                parse_reason = parsed.reason
            is_negative = sample.expected_canonical is None
            results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "sample_id": f"sample-{sample.source_index:06d}",
                    "source_index": sample.source_index,
                    "audio_sha256": sample.audio_sha256,
                    "split": sample.split,
                    "category": sample.category,
                    "semantic_category": sample.semantic_category,
                    "expected_raw": sample.expected_raw,
                    "expected_canonical": sample.expected_canonical,
                    "raw_transcript": raw_text,
                    "predicted_canonical": predicted,
                    "parse_status": parse_status,
                    "parse_reason": parse_reason,
                    "transcript_quality_rejected": bool(rejection_reasons),
                    "transcript_rejection_reasons": list(rejection_reasons),
                    "raw_exact": raw_text == sample.expected_raw,
                    "canonical_exact": predicted == sample.expected_canonical,
                    "false_game_command": is_negative and predicted is not None,
                    "positive_empty_transcript": not is_negative and not raw_text,
                    "asr_latency_ms": float(transcription.latency_ms),
                    "asr_diagnostics": transcription.diagnostics,
                }
            )
        provenance = dict(recognizer.provenance)
        _verify_model_baseline(model_baseline, provenance)
        _verify_source_baseline(source_baseline)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            recognizer.close()
        except BaseException:
            if primary_error is None:
                raise

    assert model_baseline is not None
    _verify_frozen_inputs(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        samples=samples,
        lock_path=lock_path,
        lock_sha256=dependency_lock["sha256"],
        verify_all_audio=True,
    )
    _verify_model_baseline(model_baseline, provenance)
    _verify_source_baseline(source_baseline)
    reproducibility = _reproducibility(model_baseline, dependency_lock)
    reproducibility["decode_context_profile"] = args.context_profile
    gates = evaluate_gates(
        samples,
        results,
        acceptance_split=args.acceptance_split,
        thresholds=_thresholds(args),
        reproducibility=reproducibility,
    )
    by_split_category: dict[str, dict[str, Any]] = {}
    for split in sorted({row["split"] for row in results}):
        split_rows = [row for row in results if row["split"] == split]
        by_split_category[str(split)] = _group_summaries(split_rows, "category")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_set_id": artifact_set_id,
        "qualification_profile": gates["qualification_profile"],
        "source_code": source_baseline,
        "tool": source_baseline["qualification_script"],
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "sample_count": len(samples),
        },
        "configuration": {
            "context_profile": args.context_profile,
            "initial_prompt": context.initial_prompt if context else None,
            "hotwords": list(context.hotwords) if context else [],
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type,
            "model_cache": args.model_cache,
            "model_revision": args.model_revision,
            "local_files_only": args.local_files_only,
        },
        "environment": _environment(
            dependency_lock,
            package_version_getter=package_version_getter,
        ),
        "recognizer_provenance": provenance,
        "reproducibility": reproducibility,
        "metrics": {
            "overall": summarize_rows(results),
            "by_split": _group_summaries(results, "split"),
            "by_category": _group_summaries(results, "category"),
            "by_split_category": by_split_category,
        },
        "acceptance": gates,
        "qualification_scope": "offline_asr_corpus",
        "full_feature_qualification": False,
        "privacy": {
            "speaker_or_session_written_to_results": False,
            "distance_or_environment_written_to_results": False,
            "audio_persisted_by_tool": False,
            "note": (
                "Identity fields remain in memory; safe distance/environment tokens "
                "are reported only as aggregate counts."
            ),
        },
        "passed": bool(gates["passed"]),
    }
    verification_args = {
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "samples": samples,
        "lock_path": lock_path,
        "lock_sha256": dependency_lock["sha256"],
        "verify_all_audio": True,
    }
    def verify_publication_baselines() -> None:
        _verify_frozen_inputs(**verification_args)
        _verify_model_baseline(model_baseline, provenance)
        _verify_source_baseline(source_baseline)

    published = False
    try:
        verify_publication_baselines()
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_dir.mkdir()
        staged_results = staging_dir / RESULTS_FILENAME
        staged_summary = staging_dir / SUMMARY_FILENAME
        _write_jsonl(staged_results, results)
        verify_publication_baselines()
        samples_sha256 = _sha256(staged_results)
        summary["artifact_set"] = {
            "id": artifact_set_id,
            "samples": {
                "path": RESULTS_FILENAME,
                "sha256": samples_sha256,
            },
            "summary": {"path": SUMMARY_FILENAME},
        }
        summary["results_jsonl"] = RESULTS_FILENAME
        _write_json(staged_summary, summary)
        verify_publication_baselines()
        if output_dir.exists():
            raise QualificationError(f"output directory appeared during run: {output_dir}")
        staging_dir.rename(output_dir)
        published = True
        verify_publication_baselines()
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        if published and output_dir.exists():
            shutil.rmtree(output_dir)
        raise
    return summary


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a local voice corpus through FasterWhisperRecognizer and "
            "ZipZapZopTask without recording a microphone."
        )
    )
    parser.add_argument("--manifest", required=True, help="local JSON or JSONL corpus manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--model-cache")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forbid model downloads (default: true)",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_false",
        dest="local_files_only",
        help="explicitly permit faster-whisper to populate its cache",
    )
    parser.add_argument(
        "--context-profile",
        choices=("baseline", "task", "custom"),
        default="baseline",
    )
    parser.add_argument("--initial-prompt")
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--acceptance-split", default="acceptance")

    parser.add_argument("--min-overall-accuracy", type=_unit_interval, default=0.95)
    parser.add_argument("--min-base-word-accuracy", type=_unit_interval, default=0.95)
    parser.add_argument("--min-combo-number-accuracy", type=_unit_interval, default=0.95)
    parser.add_argument(
        "--max-negative-false-command-rate", type=_unit_interval, default=0.01
    )
    parser.add_argument(
        "--max-positive-empty-transcript-rate",
        type=_unit_interval,
        default=0.05,
    )
    parser.add_argument("--max-asr-p95-ms", type=_positive_float, default=300.0)
    parser.add_argument("--min-speakers", type=_nonnegative_int, default=5)
    parser.add_argument("--min-base-word-samples", type=_nonnegative_int, default=100)
    parser.add_argument("--min-combination-samples", type=_nonnegative_int, default=100)
    parser.add_argument("--min-number-samples", type=_nonnegative_int, default=150)
    parser.add_argument("--min-negative-samples", type=_nonnegative_int, default=200)
    parser.add_argument(
        "--min-negative-category-samples",
        type=_nonnegative_int,
        default=50,
        help="minimum samples required in each negative category",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_qualification(args)
    except (QualificationError, VoiceInfrastructureError, OSError, ValueError) as exc:
        print(f"voice qualification failed: {exc}", file=sys.stderr)
        return 2
    output = Path(args.output_dir).expanduser().resolve() / SUMMARY_FILENAME
    print(f"Voice qualification {'passed' if summary['passed'] else 'failed'}: {output}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
