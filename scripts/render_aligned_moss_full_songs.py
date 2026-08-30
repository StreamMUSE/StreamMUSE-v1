#!/usr/bin/env python3
"""Render gap-free full songs with the gentle sparse aligned-MOSS mode."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
from scipy.io import wavfile

from streammuse.domain.rap import PcmAudio
from streammuse.experiments.rap_audio_protocols.artifacts import (
    build_protocol_artifact_manifest,
    chunk_record_is_complete,
    file_sha256,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.audio import (
    TARGET_SAMPLE_RATE_HZ,
    TARGET_STEREO_FORMAT,
    assemble_vocal_stem,
    mix_stems,
)
from streammuse.experiments.rap_audio_protocols.contracts import (
    ChunkRenderRecord,
    ProtocolId,
    SyllableTarget,
    TwoBarRenderRequest,
    canonical_json_dumps,
)


MODE = "continuous_onset_gentle_sparse_r3"
_STRESS_MODES = frozenset({"continuous_onset_constrained_r3_stress"})
_REGULARIZED_MODES = frozenset(
    {
        "continuous_onset_constrained_r3_stress",
        "continuous_onset_gentle_sparse_r3",
    }
)
DEFAULT_SONGS = (
    "01_space_exploration",
    "02_deep_ocean",
    "03_artificial_intelligence",
)
SONGS = DEFAULT_SONGS + (
    "04_city_nights",
    "05_climate_resilience",
    "06_human_memory",
    "07_quantum_physics",
    "08_street_basketball",
    "09_renewable_energy",
    "10_future_music",
)
SONG_TITLES = {
    "01_space_exploration": "Signals Beyond Earth",
    "02_deep_ocean": "Pressure Below",
    "03_artificial_intelligence": "Learning Machines",
    "04_city_nights": "Lights After Midnight",
    "05_climate_resilience": "Stronger Than The Storm",
    "06_human_memory": "Rooms Inside The Mind",
    "07_quantum_physics": "Probability Steps",
    "08_street_basketball": "Concrete Court",
    "09_renewable_energy": "Power From Tomorrow",
    "10_future_music": "The Next Sound",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--song", action="append", choices=SONGS, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_campaign(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        song_ids=tuple(args.song or DEFAULT_SONGS),
        backend=_load_backend_module(),
    )


def run_campaign(
    *,
    input_dir: Path,
    output_dir: Path,
    song_ids: Sequence[str],
    backend: Any,
    allow_smoke_test: bool = False,
    expected_chunk_count: int = 25,
) -> int:
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not song_ids:
        raise ValueError("at least one song is required")
    if len(set(song_ids)) != len(song_ids):
        raise ValueError("song IDs must be unique")
    if expected_chunk_count <= 0:
        raise ValueError("expected_chunk_count must be positive")
    unknown = sorted(set(song_ids) - set(SONGS))
    if unknown:
        raise ValueError(f"unsupported song IDs: {', '.join(unknown)}")

    song_manifests = []
    total_chunk_count = 0
    for song_id in song_ids:
        requests = _load_requests(input_root / "common" / song_id / "requests.jsonl")
        if not allow_smoke_test and len(requests) != expected_chunk_count:
            raise ValueError(
                f"{song_id} must contain exactly {expected_chunk_count} two-bar requests"
            )
        song_manifest = _render_song(
            input_root=input_root,
            output_root=output_root,
            song_id=song_id,
            requests=requests,
            backend=backend,
            allow_smoke_test=allow_smoke_test or expected_chunk_count != 25,
        )
        song_manifests.append(song_manifest)
        total_chunk_count += len(requests)

    tempos = {float(item["tempo_bpm"]) for item in song_manifests}
    bar_counts = {int(item["bar_count"]) for item in song_manifests}
    manifest = {
        "schema_version": "streammuse.aligned_moss_full_songs.v1",
        "mode": MODE,
        "tempo_bpm": next(iter(tempos)) if len(tempos) == 1 else None,
        "bars_per_song": next(iter(bar_counts)) if len(bar_counts) == 1 else None,
        "input_dir": str(input_root),
        "song_count": len(song_manifests),
        "total_chunk_count": total_chunk_count,
        "songs": song_manifests,
    }
    _write_json_atomic(output_root / "manifest.json", manifest)
    return 0


def _render_song(
    *,
    input_root: Path,
    output_root: Path,
    song_id: str,
    requests: tuple[TwoBarRenderRequest, ...],
    backend: Any,
    allow_smoke_test: bool,
) -> dict[str, Any]:
    source_ledger_path = input_root / "moss_global" / song_id / "render_chunks.jsonl"
    source_records = read_chunk_record_index(source_ledger_path)
    song_root = output_root / song_id
    chunks_root = song_root / "chunks"
    ledger_path = song_root / "render_chunks.jsonl"
    rendered_records = read_chunk_record_index(ledger_path)

    ordered_records: list[ChunkRenderRecord] = []
    diagnostics_payloads: list[dict[str, Any]] = []
    for ordinal, request in enumerate(requests, start=1):
        source = source_records.get((ProtocolId.MOSS_GLOBAL, song_id, request.chunk_index))
        if source is None or not source.success or not source.output_sha256:
            raise ValueError(f"missing successful MOSS source record for {song_id}/{request.chunk_index:03d}")
        if not _request_sha256_matches(request, source.request_sha256):
            raise ValueError(f"MOSS source request mismatch for {song_id}/{request.chunk_index:03d}")

        source_path = input_root / "moss_global" / song_id / f"chunk-{request.chunk_index:03d}.wav"
        if not source_path.is_file():
            raise ValueError(f"missing MOSS source WAV: {source_path}")
        if file_sha256(source_path) != source.output_sha256:
            raise ValueError(f"MOSS source SHA mismatch: {source_path}")
        textgrid_path = (
            input_root
            / "moss_aligned"
            / song_id
            / "mfa-output"
            / f"{song_id}__chunk_{request.chunk_index:02d}.TextGrid"
        )
        if not textgrid_path.is_file():
            raise ValueError(f"missing MFA TextGrid: {textgrid_path}")

        output_path = chunks_root / f"chunk-{request.chunk_index:03d}.wav"
        diagnostics_path = output_path.with_suffix(output_path.suffix + ".alignment.json")
        if _chunk_is_current(
            ledger_path=ledger_path,
            output_path=output_path,
            diagnostics_path=diagnostics_path,
            request=request,
            expected_source_sha256=source.output_sha256,
        ):
            record = rendered_records[(ProtocolId.MOSS_ALIGNED, song_id, request.chunk_index)]
            status = "skip"
        else:
            result = backend.render_aligned_chunk(
                request=request,
                source_wav_path=source_path,
                expected_source_sha256=source.output_sha256,
                textgrid_path=textgrid_path,
                output_wav_path=output_path.resolve(),
                mode=MODE,
            )
            if not result.record.success:
                raise RuntimeError(result.record.error or f"v5 render failed for {song_id}/{request.chunk_index:03d}")
            if result.diagnostics_path is None:
                raise RuntimeError(f"v5 render emitted no diagnostics for {song_id}/{request.chunk_index:03d}")
            record = result.record
            rendered_records[(ProtocolId.MOSS_ALIGNED, song_id, request.chunk_index)] = record
            _write_record_ledger(ledger_path, rendered_records)
            status = "render"

        diagnostics = _load_verified_diagnostics(
            diagnostics_path,
            request=request,
            expected_source_sha256=source.output_sha256,
            expected_output_sha256=record.output_sha256,
        )
        ordered_records.append(record)
        diagnostics_payloads.append(diagnostics)
        print(
            f"stage=render song={song_id} mode={MODE} chunk={request.chunk_index:03d} "
            f"count={ordinal}/{len(requests)} status={status} output_sha256={record.output_sha256}",
            flush=True,
        )

    vocal_path = song_root / "vocals.wav"
    mix_path = song_root / "mix.wav"
    drums_path = input_root / "common" / song_id / "drums.wav"
    if not drums_path.is_file():
        raise ValueError(f"missing common drum stem: {drums_path}")
    vocal_stem = assemble_vocal_stem(
        requests,
        chunk_paths_by_index={
            record.chunk_index: Path(record.output_path or "") for record in ordered_records
        },
        allow_smoke_test=allow_smoke_test,
        listening_wav_path=vocal_path,
    )
    drums = _load_stereo_float32_audio(drums_path)
    mix = mix_stems(vocal_stem.audio, drums, listening_wav_path=mix_path)
    artifact_manifest = build_protocol_artifact_manifest(
        ProtocolId.MOSS_ALIGNED,
        requests=requests,
        chunk_records=ordered_records,
        vocal_stem_path=vocal_path,
        drums_path=drums_path,
        mix_path=mix_path,
        allow_smoke_test=allow_smoke_test,
    )
    _write_json_atomic(song_root / "artifact_manifest.json", artifact_manifest)

    diagnostics_summary = _summarize_diagnostics(diagnostics_payloads)
    song_manifest = {
        "song_id": song_id,
        "title": SONG_TITLES[song_id],
        "tempo_bpm": requests[0].tempo_bpm,
        "chunk_count": len(requests),
        "bar_count": requests[-1].end_bar,
        "frame_count": mix.audio.frame_count,
        "duration_seconds": mix.audio.frame_count / TARGET_SAMPLE_RATE_HZ,
        "sample_rate_hz": TARGET_SAMPLE_RATE_HZ,
        "separator_silence_seconds": 0.0,
        "vocal_path": str(vocal_path.relative_to(output_root)),
        "vocal_sha256": file_sha256(vocal_path),
        "drums_path": str(drums_path),
        "drums_sha256": file_sha256(drums_path),
        "mix_path": str(mix_path.relative_to(output_root)),
        "mix_sha256": file_sha256(mix_path),
        "mix_peak_before_limiter": mix.peak_before_limiter,
        "mix_applied_gain": mix.applied_gain,
        "diagnostics": diagnostics_summary,
    }
    _write_json_atomic(song_root / "summary.json", song_manifest)
    print(
        f"stage=assemble song={song_id} mode={MODE} chunks={len(requests)} "
        f"frames={mix.audio.frame_count} seconds={song_manifest['duration_seconds']:.6f} "
        f"separator_silence_seconds=0 mix_sha256={song_manifest['mix_sha256']}",
        flush=True,
    )
    return song_manifest


def _chunk_is_current(
    *,
    ledger_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    request: TwoBarRenderRequest,
    expected_source_sha256: str,
) -> bool:
    if not chunk_record_is_complete(
        ledger_path,
        output_path,
        request=request,
        protocol_id=ProtocolId.MOSS_ALIGNED,
        expected_source_sha256=expected_source_sha256,
    ):
        return False
    record = read_chunk_record_index(ledger_path)[
        (ProtocolId.MOSS_ALIGNED, request.song_id, request.chunk_index)
    ]
    try:
        _load_verified_diagnostics(
            diagnostics_path,
            request=request,
            expected_source_sha256=expected_source_sha256,
            expected_output_sha256=record.output_sha256,
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _load_verified_diagnostics(
    path: Path,
    *,
    request: TwoBarRenderRequest,
    expected_source_sha256: str,
    expected_output_sha256: str | None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "success": True,
        "mode": MODE,
        "request_sha256": request.sha256,
        "source_sha256": expected_source_sha256,
        "output_sha256": expected_output_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: diagnostics {key} mismatch")
    expected_stress = MODE in _STRESS_MODES
    if payload.get("stress", {}).get("applied") is not expected_stress:
        raise ValueError(f"{path}: stress application mismatch for mode {MODE}")
    expected_regularization = MODE in _REGULARIZED_MODES
    if payload.get("timing_regularization", {}).get("applied") is not expected_regularization:
        raise ValueError(f"{path}: timing regularization mismatch for mode {MODE}")
    return payload


def _summarize_diagnostics(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    stretch_ratios = [
        float(value)
        for payload in payloads
        for value in payload.get("stretch_ratios", [])
    ]
    target_drift = [
        abs(float(value))
        for payload in payloads
        for value in payload.get("timing_regularization", {}).get("target_drift_seconds", [])
    ]
    sorted_drift = sorted(target_drift)
    p95_index = max(0, min(len(sorted_drift) - 1, round(0.95 * len(sorted_drift)) - 1))
    return {
        "rendered_chunk_count": len(payloads),
        "fallback_count": sum(int(payload.get("fallback_count", 0)) for payload in payloads),
        "boundary_adjustment_count": sum(
            int(payload.get("boundary_adjustment_count", 0)) for payload in payloads
        ),
        "source_boundary_adjustment_count": sum(
            int(payload.get("source_boundary_adjustment_count", 0)) for payload in payloads
        ),
        "peak_limited_chunk_count": sum(
            payload.get("stress", {}).get("peak_limited") is True for payload in payloads
        ),
        "minimum_stretch_ratio": min(stretch_ratios) if stretch_ratios else None,
        "maximum_stretch_ratio": max(stretch_ratios) if stretch_ratios else None,
        "mean_absolute_target_drift_ms": (
            1000.0 * sum(target_drift) / len(target_drift) if target_drift else None
        ),
        "p95_absolute_target_drift_ms": (
            1000.0 * sorted_drift[p95_index] if sorted_drift else None
        ),
        "maximum_absolute_target_drift_ms": (
            1000.0 * max(target_drift) if target_drift else None
        ),
    }


def _load_requests(path: Path) -> tuple[TwoBarRenderRequest, ...]:
    if not path.is_file():
        raise ValueError(f"missing request ledger: {path}")
    requests = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        request = TwoBarRenderRequest(
            song_id=str(payload["song_id"]),
            chunk_index=int(payload["chunk_index"]),
            start_bar=int(payload["start_bar"]),
            end_bar=int(payload["end_bar"]),
            text=str(payload["text"]),
            tempo_bpm=float(payload.get("tempo_bpm", 90.0)),
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
        stored_sha256 = payload.get("request_sha256")
        if stored_sha256 is not None and not _request_sha256_matches(
            request,
            str(stored_sha256),
        ):
            raise ValueError(f"{path}:{line_number}: request_sha256 mismatch")
        requests.append(request)
    requests.sort(key=lambda item: item.chunk_index)
    return tuple(requests)


def _request_sha256_matches(request: TwoBarRenderRequest, candidate: str) -> bool:
    if candidate == request.sha256:
        return True
    if request.tempo_bpm != 90.0:
        return False
    legacy_payload = request.to_payload()
    legacy_payload.pop("tempo_bpm")
    legacy_sha256 = hashlib.sha256(
        canonical_json_dumps(legacy_payload).encode("utf-8")
    ).hexdigest()
    return candidate == legacy_sha256


def _load_stereo_float32_audio(path: Path) -> PcmAudio:
    sample_rate_hz, samples = wavfile.read(path)
    if sample_rate_hz != TARGET_SAMPLE_RATE_HZ:
        raise ValueError(f"{path} must be 48 kHz")
    array = np.asarray(samples)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{path} must be stereo")
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), info.max)
        float32 = array.astype(np.float32) / np.float32(scale)
    else:
        float32 = array.astype(np.float32, copy=False)
    return PcmAudio(TARGET_STEREO_FORMAT, int(float32.shape[0]), float32.tobytes())


def _write_record_ledger(
    path: Path,
    records: dict[tuple[ProtocolId, str, int], ChunkRenderRecord],
) -> None:
    lines = [
        canonical_json_dumps(records[key].to_payload())
        for key in sorted(records, key=lambda item: (item[1], item[2], item[0].value))
    ]
    _write_text_atomic(path, "\n".join(lines) + "\n")


def _write_json_atomic(path: Path, payload: Any) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_backend_module() -> ModuleType:
    module_name = "_streammuse_full_song_aligned_moss_backend"
    path = Path(__file__).resolve().parent / "rap_audio_backends" / "aligned_moss_backend.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load aligned MOSS backend: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
