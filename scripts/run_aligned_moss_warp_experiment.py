#!/usr/bin/env python3
"""Compare aligned-MOSS warp and stress strategies on fixed source chunks."""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from math import gcd
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from streammuse.experiments.rap_audio_protocols.artifacts import file_sha256
from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget, TwoBarRenderRequest
from streammuse.experiments.rap_audio_protocols.evaluation import (
    build_faster_whisper_transcriber,
    compute_signal_metrics,
    compute_word_error_counts,
    estimate_syllable_timing_error_ms,
    measure_stress_rms_correlation,
    normalize_word,
    normalize_words,
)


BASELINE_MODE = "piecewise_vowel_r2"
EXPERIMENT_MODES = (
    BASELINE_MODE,
    "continuous_vowel_r3",
    "continuous_onset_r3",
    "continuous_onset_constrained_r3_stress",
    "continuous_onset_r2_smooth",
)


@dataclass(frozen=True)
class ChunkCandidate:
    song_id: str
    chunk_index: int
    distortion_score: float
    baseline_wav_path: Path
    diagnostics_path: Path


@dataclass(frozen=True)
class SelectedChunk:
    category: str
    candidate: ChunkCandidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clean-count", type=int, default=3)
    parser.add_argument("--median-count", type=int, default=3)
    parser.add_argument("--worst-count", type=int, default=6)
    parser.add_argument("--evaluate-asr", action="store_true")
    parser.add_argument("--asr-only", action="store_true")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--whisper-device", default="cuda")
    parser.add_argument("--whisper-compute-type", default="float16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.asr_only:
        evaluate_existing_asr(args)
        return 0
    backend = _load_backend_module()
    candidates = discover_candidates(args.input_dir)
    selected = select_chunks(
        candidates,
        clean_count=args.clean_count,
        median_count=args.median_count,
        worst_count=args.worst_count,
    )
    requests = _load_selected_requests(args.input_dir, selected)
    transcribe = None
    if args.evaluate_asr:
        transcribe = build_faster_whisper_transcriber(
            model_size=args.whisper_model,
            device=args.whisper_device,
            compute_type=args.whisper_compute_type,
        )

    manifest_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    mix_paths_by_mode: dict[str, list[Path]] = {mode: [] for mode in EXPERIMENT_MODES}
    for ordinal, selected_chunk in enumerate(selected, start=1):
        candidate = selected_chunk.candidate
        request = requests[(candidate.song_id, candidate.chunk_index)]
        source_path = (
            args.input_dir
            / "moss_global"
            / candidate.song_id
            / f"chunk-{candidate.chunk_index:03d}.wav"
        )
        textgrid_path = (
            args.input_dir
            / "moss_aligned"
            / candidate.song_id
            / "mfa-output"
            / f"{candidate.song_id}__chunk_{candidate.chunk_index:02d}.TextGrid"
        )
        baseline_diagnostics = _load_json(candidate.diagnostics_path)
        source_sha256 = str(baseline_diagnostics["source_sha256"])
        if file_sha256(source_path) != source_sha256:
            raise ValueError(f"source SHA mismatch for {source_path}")

        chunk_root = (
            args.output_dir
            / "chunks"
            / f"{ordinal:02d}_{selected_chunk.category}"
            / candidate.song_id
            / f"chunk-{candidate.chunk_index:03d}"
        )
        chunk_root.mkdir(parents=True, exist_ok=True)
        for mode in EXPERIMENT_MODES:
            vocal_path = chunk_root / f"{mode}.wav"
            if mode == BASELINE_MODE:
                shutil.copy2(candidate.baseline_wav_path, vocal_path)
                diagnostics = baseline_diagnostics
            else:
                result = backend.render_aligned_chunk(
                    request=request,
                    source_wav_path=source_path,
                    expected_source_sha256=source_sha256,
                    textgrid_path=textgrid_path,
                    output_wav_path=vocal_path,
                    mode=mode,
                )
                if not result.record.success:
                    raise RuntimeError(result.record.error or f"{mode} render failed")
                diagnostics = _load_json(result.diagnostics_path)

            mix_path = vocal_path.with_name(f"{mode}.mix.wav")
            _mix_with_chunk_drums(
                vocal_path,
                args.input_dir / "common" / candidate.song_id / "drums.wav",
                request,
                mix_path,
            )
            mix_paths_by_mode[mode].append(mix_path)
            metrics = evaluate_render(
                request,
                vocal_path,
                diagnostics,
                transcribe=transcribe,
            )
            metric_rows.append(
                {
                    "category": selected_chunk.category,
                    "song_id": candidate.song_id,
                    "chunk_index": candidate.chunk_index,
                    "mode": mode,
                    **metrics,
                }
            )
            manifest_rows.append(
                {
                    "ordinal": ordinal,
                    "category": selected_chunk.category,
                    "song_id": candidate.song_id,
                    "chunk_index": candidate.chunk_index,
                    "text": request.text,
                    "mode": mode,
                    "source_sha256": source_sha256,
                    "vocal_path": str(vocal_path.relative_to(args.output_dir)),
                    "vocal_sha256": file_sha256(vocal_path),
                    "mix_path": str(mix_path.relative_to(args.output_dir)),
                    "mix_sha256": file_sha256(mix_path),
                }
            )
            print(
                f"RENDER count={ordinal}/{len(selected)} category={selected_chunk.category} "
                f"song={candidate.song_id} chunk={candidate.chunk_index} mode={mode}",
                flush=True,
            )

    aggregate_paths = {}
    for mode, paths in mix_paths_by_mode.items():
        aggregate_path = args.output_dir / "listening" / f"{mode}.wav"
        _concatenate_audio(paths, aggregate_path)
        aggregate_paths[mode] = aggregate_path

    selection_payload = [
        {
            "category": item.category,
            **{
                **asdict(item.candidate),
                "baseline_wav_path": str(item.candidate.baseline_wav_path),
                "diagnostics_path": str(item.candidate.diagnostics_path),
            },
        }
        for item in selected
    ]
    _write_json(args.output_dir / "selection.json", {"selected": selection_payload})
    _write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": "streammuse.aligned_moss_warp_experiment.v1",
            "input_dir": str(args.input_dir.resolve()),
            "modes": list(EXPERIMENT_MODES),
            "asr_enabled": bool(args.evaluate_asr),
            "artifacts": manifest_rows,
        },
    )
    _write_json(
        args.output_dir / "metrics.json",
        {
            "rows": metric_rows,
            "aggregate_by_mode": aggregate_metrics(metric_rows),
        },
    )
    _write_listening_html(
        args.output_dir / "index.html",
        output_root=args.output_dir,
        selected=selected,
        requests=requests,
        aggregate_paths=aggregate_paths,
        manifest_rows=manifest_rows,
        metric_rows=metric_rows,
    )
    print(f"COMPLETE output={args.output_dir.resolve()}", flush=True)
    return 0


def discover_candidates(input_dir: Path) -> tuple[ChunkCandidate, ...]:
    candidates = []
    for diagnostics_path in sorted((input_dir / "moss_aligned").glob("*/chunk-*.wav.alignment.json")):
        payload = _load_json(diagnostics_path)
        ratios = tuple(float(value) for value in payload.get("stretch_ratios", ()))
        if not payload.get("success") or not ratios or any(value <= 0 for value in ratios):
            continue
        baseline_wav_path = Path(str(diagnostics_path).removesuffix(".alignment.json"))
        if not baseline_wav_path.is_file():
            continue
        song_id = diagnostics_path.parent.name
        chunk_index = int(diagnostics_path.name.split("-", 1)[1].split(".", 1)[0])
        distortion_score = max(max(value, 1.0 / value) for value in ratios)
        candidates.append(
            ChunkCandidate(
                song_id=song_id,
                chunk_index=chunk_index,
                distortion_score=distortion_score,
                baseline_wav_path=baseline_wav_path,
                diagnostics_path=diagnostics_path,
            )
        )
    if not candidates:
        raise ValueError(f"no successful baseline diagnostics found under {input_dir}")
    return tuple(candidates)


def select_chunks(
    candidates: Sequence[ChunkCandidate],
    *,
    clean_count: int,
    median_count: int,
    worst_count: int,
) -> tuple[SelectedChunk, ...]:
    requested_count = clean_count + median_count + worst_count
    if min(clean_count, median_count, worst_count) < 0:
        raise ValueError("selection counts must not be negative")
    if requested_count > len(candidates):
        raise ValueError("selection counts exceed available candidates")
    ordered = sorted(candidates, key=lambda item: (item.distortion_score, item.song_id, item.chunk_index))
    clean = ordered[:clean_count]
    worst = ordered[len(ordered) - worst_count :] if worst_count else []
    excluded = set(clean) | set(worst)
    remaining = [item for item in ordered if item not in excluded]
    median_score = float(np.median([item.distortion_score for item in ordered]))
    median = sorted(
        sorted(
            remaining,
            key=lambda item: (abs(item.distortion_score - median_score), item.distortion_score),
        )[:median_count],
        key=lambda item: item.distortion_score,
    )
    return tuple(
        [*(SelectedChunk("clean", item) for item in clean)]
        + [*(SelectedChunk("median", item) for item in median)]
        + [*(SelectedChunk("worst", item) for item in worst)]
    )


def boundary_jump_metrics(samples: np.ndarray, boundary_samples: Sequence[int]) -> dict[str, Any]:
    mono = _to_mono_float32(samples)
    jumps = [
        abs(float(mono[index + 1]) - float(mono[index]))
        for index in boundary_samples
        if 0 <= index < len(mono) - 1
    ]
    return {
        "boundary_count": len(jumps),
        "mean_absolute_jump": float(np.mean(jumps)) if jumps else 0.0,
        "max_absolute_jump": max(jumps, default=0.0),
    }


def evaluate_render(
    request: TwoBarRenderRequest,
    vocal_path: Path,
    diagnostics: dict[str, Any],
    *,
    transcribe: Any | None,
) -> dict[str, Any]:
    sample_rate_hz, samples = wavfile.read(vocal_path)
    mono = _to_mono_float32(samples)
    ratios = tuple(float(value) for value in diagnostics.get("stretch_ratios", ()))
    anchor_map = diagnostics.get("anchor_map", ())
    boundaries = tuple(int(anchor["target_sample"]) for anchor in anchor_map)
    stress_correlation = measure_stress_rms_correlation(
        request,
        mono,
        sample_rate_hz=sample_rate_hz,
    )
    result: dict[str, Any] = {
        "sample_rate_hz": sample_rate_hz,
        "duration_seconds": len(mono) / sample_rate_hz,
        "signal": compute_signal_metrics(mono),
        "stretch_ratio_min": min(ratios, default=0.0),
        "stretch_ratio_max": max(ratios, default=0.0),
        "extreme_ratio_count": sum(value < 0.5 or value > 2.0 for value in ratios),
        "reciprocal_distortion_max": max(
            (max(value, 1.0 / value) for value in ratios if value > 0),
            default=0.0,
        ),
        "boundary_jumps": boundary_jump_metrics(mono, boundaries),
        "stress_rms_correlation": stress_correlation,
        "target_drift_seconds": diagnostics.get("timing_regularization", {}).get(
            "target_drift_seconds",
            [float(anchor["target_seconds"]) - float(anchor["requested_target_seconds"]) for anchor in anchor_map],
        ),
    }
    if transcribe is not None:
        result["asr"] = evaluate_asr_metrics(request, tuple(transcribe(vocal_path)))
    return result


def evaluate_existing_asr(args: argparse.Namespace) -> None:
    manifest_path = args.output_dir / "manifest.json"
    metrics_path = args.output_dir / "metrics.json"
    manifest = _load_json(manifest_path)
    metrics_payload = _load_json(metrics_path)
    artifacts = tuple(manifest.get("artifacts", ()))
    if not artifacts:
        raise ValueError("ASR-only evaluation requires a completed artifact manifest")
    requests = _load_requests_by_song_ids(
        args.input_dir,
        tuple(sorted({str(item["song_id"]) for item in artifacts})),
    )
    transcribe = build_faster_whisper_transcriber(
        model_size=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )
    asr_by_key = {}
    for ordinal, artifact in enumerate(artifacts, start=1):
        key = (
            str(artifact["song_id"]),
            int(artifact["chunk_index"]),
            str(artifact["mode"]),
        )
        vocal_path = args.output_dir / str(artifact["vocal_path"])
        recognized = tuple(transcribe(vocal_path))
        asr_by_key[key] = evaluate_asr_metrics(requests[key[:2]], recognized)
        print(
            f"ASR count={ordinal}/{len(artifacts)} song={key[0]} chunk={key[1]} mode={key[2]}",
            flush=True,
        )
    rows = merge_asr_metrics(tuple(metrics_payload["rows"]), asr_by_key)
    _write_json(
        metrics_path,
        {
            "rows": rows,
            "aggregate_by_mode": aggregate_metrics(rows),
        },
    )
    manifest["asr_enabled"] = True
    manifest["asr_config"] = {
        "model": args.whisper_model,
        "device": args.whisper_device,
        "compute_type": args.whisper_compute_type,
    }
    _write_json(manifest_path, manifest)
    print(f"ASR_COMPLETE output={args.output_dir.resolve()}", flush=True)


def evaluate_asr_metrics(
    request: TwoBarRenderRequest,
    recognized: Sequence[Any],
) -> dict[str, Any]:
    counts = compute_word_error_counts(
        normalize_words(request.text),
        tuple(
            word
            for word in (normalize_word(item.text) for item in recognized)
            if word
        ),
    )
    timing_errors = estimate_syllable_timing_error_ms(request, recognized)
    return {
        **counts,
        "transcript": " ".join(item.text.strip() for item in recognized),
        "timing_error_ms_mean_absolute": (
            float(np.mean(np.abs(timing_errors))) if timing_errors else None
        ),
        "timing_error_count": len(timing_errors),
    }


def merge_asr_metrics(
    rows: Sequence[dict[str, Any]],
    asr_by_key: dict[tuple[str, int, str], dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    merged = []
    for row in rows:
        key = (str(row["song_id"]), int(row["chunk_index"]), str(row["mode"]))
        if key not in asr_by_key:
            raise ValueError(f"missing ASR metrics for {key}")
        merged.append({**row, "asr": asr_by_key[key]})
    return tuple(merged)


def aggregate_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregates = {}
    for mode in EXPERIMENT_MODES:
        selected = [row for row in rows if row["mode"] == mode]
        aggregates[mode] = {
            "chunk_count": len(selected),
            "mean_extreme_ratio_count": _mean(row["extreme_ratio_count"] for row in selected),
            "mean_reciprocal_distortion_max": _mean(
                row["reciprocal_distortion_max"] for row in selected
            ),
            "mean_boundary_jump": _mean(
                row["boundary_jumps"]["mean_absolute_jump"] for row in selected
            ),
            "mean_stress_rms_correlation": _mean(
                row["stress_rms_correlation"]
                for row in selected
                if row["stress_rms_correlation"] is not None
            ),
            "mean_word_error_rate": _mean(
                row["asr"]["word_error_rate"] for row in selected if "asr" in row
            ),
            "mean_asr_timing_error_ms": _mean(
                row["asr"]["timing_error_ms_mean_absolute"]
                for row in selected
                if "asr" in row and row["asr"]["timing_error_ms_mean_absolute"] is not None
            ),
        }
    return aggregates


def _load_selected_requests(
    input_dir: Path,
    selected: Sequence[SelectedChunk],
) -> dict[tuple[str, int], TwoBarRenderRequest]:
    song_ids = sorted({item.candidate.song_id for item in selected})
    return _load_requests_by_song_ids(input_dir, song_ids)


def _load_requests_by_song_ids(
    input_dir: Path,
    song_ids: Sequence[str],
) -> dict[tuple[str, int], TwoBarRenderRequest]:
    requests = {}
    for song_id in song_ids:
        request_path = input_dir / "common" / song_id / "requests.jsonl"
        for line_number, raw in enumerate(request_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            request = _request_from_payload(payload)
            expected_sha = payload.get("request_sha256")
            if expected_sha is not None and expected_sha != request.sha256:
                raise ValueError(f"{request_path}:{line_number}: request SHA mismatch")
            requests[(song_id, request.chunk_index)] = request
    return requests


def _request_from_payload(payload: dict[str, Any]) -> TwoBarRenderRequest:
    return TwoBarRenderRequest(
        song_id=str(payload["song_id"]),
        chunk_index=int(payload["chunk_index"]),
        start_bar=int(payload["start_bar"]),
        end_bar=int(payload["end_bar"]),
        text=str(payload["text"]),
        syllables=tuple(
            SyllableTarget(
                word=str(item["word"]),
                index_in_word=int(item["index_in_word"]),
                phonemes=tuple(str(phone) for phone in item["phonemes"]),
                lexical_stress=int(item["lexical_stress"]),
                target_stress=float(item["target_stress"]),
                boundary_strength=int(item["boundary_strength"]),
                absolute_tick=int(item["absolute_tick"]),
                tick_in_chunk=int(item["tick_in_chunk"]),
                target_seconds=float(item["target_seconds"]),
            )
            for item in payload["syllables"]
        ),
    )


def _mix_with_chunk_drums(
    vocal_path: Path,
    drums_path: Path,
    request: TwoBarRenderRequest,
    output_path: Path,
) -> None:
    vocal_rate, vocal = wavfile.read(vocal_path)
    drum_rate, drums = wavfile.read(drums_path)
    vocal_mono = _to_mono_float32(vocal)
    drum_mono = _to_mono_float32(drums)
    start = round(request.chunk_index * request.duration_seconds * drum_rate)
    source_frame_count = round((len(vocal_mono) / vocal_rate) * drum_rate)
    drum_chunk = drum_mono[start : start + source_frame_count]
    drum_chunk = resample_mono(
        drum_chunk,
        source_rate_hz=drum_rate,
        target_rate_hz=vocal_rate,
    )
    if len(drum_chunk) < len(vocal_mono):
        drum_chunk = np.pad(drum_chunk, (0, len(vocal_mono) - len(drum_chunk)))
    elif len(drum_chunk) > len(vocal_mono):
        drum_chunk = drum_chunk[: len(vocal_mono)]
    mixed = (0.9 * vocal_mono) + (0.35 * drum_chunk)
    peak = float(np.max(np.abs(mixed), initial=0.0))
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, vocal_rate, np.asarray(mixed, dtype=np.float32))


def resample_mono(
    samples: np.ndarray,
    *,
    source_rate_hz: int,
    target_rate_hz: int,
) -> np.ndarray:
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    mono = _to_mono_float32(samples)
    if source_rate_hz == target_rate_hz:
        return np.asarray(mono, dtype=np.float32)
    divisor = gcd(source_rate_hz, target_rate_hz)
    resampled = resample_poly(
        mono,
        up=target_rate_hz // divisor,
        down=source_rate_hz // divisor,
    )
    return np.asarray(resampled, dtype=np.float32)


def _concatenate_audio(paths: Sequence[Path], output_path: Path) -> None:
    if not paths:
        raise ValueError("at least one audio path is required")
    sample_rate_hz: int | None = None
    parts = []
    for path in paths:
        current_rate, samples = wavfile.read(path)
        if sample_rate_hz is None:
            sample_rate_hz = current_rate
        elif current_rate != sample_rate_hz:
            raise ValueError("aggregate audio sample rates do not match")
        parts.append(_to_mono_float32(samples))
        parts.append(np.zeros(round(0.25 * current_rate), dtype=np.float32))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, int(sample_rate_hz), np.concatenate(parts).astype(np.float32))


def _write_listening_html(
    path: Path,
    *,
    output_root: Path,
    selected: Sequence[SelectedChunk],
    requests: dict[tuple[str, int], TwoBarRenderRequest],
    aggregate_paths: dict[str, Path],
    manifest_rows: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
) -> None:
    artifacts = {
        (row["song_id"], row["chunk_index"], row["mode"]): row for row in manifest_rows
    }
    metrics = {
        (row["song_id"], row["chunk_index"], row["mode"]): row for row in metric_rows
    }
    aggregate_html = "".join(
        f'<section><h3>{html.escape(mode)}</h3><audio controls preload="none" '
        f'src="{html.escape(str(audio_path.relative_to(output_root)))}"></audio></section>'
        for mode, audio_path in aggregate_paths.items()
    )
    rows = []
    for item in selected:
        candidate = item.candidate
        request = requests[(candidate.song_id, candidate.chunk_index)]
        cells = []
        for mode in EXPERIMENT_MODES:
            key = (candidate.song_id, candidate.chunk_index, mode)
            artifact = artifacts[key]
            metric = metrics[key]
            cells.append(
                "<td>"
                f'<audio controls preload="none" src="{html.escape(artifact["mix_path"])}"></audio>'
                f'<a href="{html.escape(artifact["vocal_path"])}">dry vocal</a>'
                f'<small>extreme={metric["extreme_ratio_count"]} '
                f'dist={metric["reciprocal_distortion_max"]:.2f} '
                f'stress={_format_optional(metric["stress_rms_correlation"])}</small>'
                "</td>"
            )
        rows.append(
            "<tr>"
            f'<th><b>{html.escape(item.category)}</b><br>{html.escape(candidate.song_id)} '
            f'#{candidate.chunk_index}<small>{html.escape(request.text)}</small></th>'
            + "".join(cells)
            + "</tr>"
        )
    headers = "".join(f"<th>{html.escape(mode)}</th>" for mode in EXPERIMENT_MODES)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aligned MOSS warp experiment</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#17202a;background:#f7f8fa}}h1,h2,h3{{margin:0 0 10px}}
.aggregate{{display:grid;grid-template-columns:repeat(5,minmax(220px,1fr));gap:8px;margin:14px 0 24px}}section{{background:white;border:1px solid #ccd2d8;padding:10px;border-radius:6px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #ccd2d8;padding:8px;vertical-align:top}}th{{text-align:left;min-width:190px}}td{{min-width:220px}}audio{{width:100%;height:34px}}a,small{{display:block;margin-top:5px}}small{{color:#53606c;line-height:1.35}}
</style></head><body><h1>Aligned MOSS warp experiment</h1>
<p>Same MOSS sources, MFA alignments, lyrics, and drum slices. Compare timing, intelligibility, and artifacts.</p>
<h2>Aggregate A/B</h2><div class="aggregate">{aggregate_html}</div>
<h2>Per chunk</h2><table><thead><tr><th>Selection</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _load_backend_module() -> ModuleType:
    module_name = "_streammuse_aligned_moss_experiment_backend"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent / "rap_audio_backends" / "aligned_moss_backend.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load aligned MOSS backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _to_mono_float32(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples)
    if array.dtype.kind in {"i", "u"}:
        scale = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / np.float32(scale)
    else:
        array = array.astype(np.float32, copy=False)
    if array.ndim == 1:
        return array
    return np.mean(array, axis=1, dtype=np.float32)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("diagnostics path is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mean(values: Any) -> float | None:
    collected = tuple(float(value) for value in values)
    return float(np.mean(collected)) if collected else None


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
