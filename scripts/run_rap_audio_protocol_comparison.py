#!/usr/bin/env python3
"""Prepare, assemble, evaluate, and package the rap audio protocol comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
from scipy.io import wavfile

from streammuse.domain.rap import PcmAudio
from streammuse.experiments.rap_audio_protocols.artifacts import build_protocol_artifact_manifest, file_sha256
from streammuse.experiments.rap_audio_protocols.audio import (
    SONG_FRAME_COUNT,
    TARGET_SAMPLE_RATE_HZ,
    TARGET_STEREO_FORMAT,
    assemble_vocal_stem,
    mix_stems,
    render_common_drums,
    write_listening_wav,
)
from streammuse.experiments.rap_audio_protocols.contracts import ChunkRenderRecord, ProtocolId, SyllableTarget, TwoBarRenderRequest, canonical_json_dumps
from streammuse.experiments.rap_audio_protocols.corpus import load_song_corpus
from streammuse.experiments.rap_audio_protocols.evaluation import build_faster_whisper_transcriber, evaluate_protocol_song
from streammuse.experiments.rap_audio_protocols.listening import ListeningAsset, write_listening_package


_SONG_IDS = (
    "01_space_exploration",
    "02_deep_ocean",
    "03_artificial_intelligence",
)
_PROTOCOL_IDS = tuple(protocol.value for protocol in ProtocolId)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-album", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "assemble", "evaluate", "package"), required=True)
    parser.add_argument("--song", choices=_SONG_IDS, default=None)
    parser.add_argument("--protocol", choices=_PROTOCOL_IDS, default=None)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--whisper-device", default="cpu")
    parser.add_argument("--whisper-compute-type", default="int8")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    transcriber_factory: Callable[..., Callable[[Path], Sequence[Any]]] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "prepare":
            prepare_campaign(args)
        elif args.stage == "assemble":
            assemble_campaign(args)
        elif args.stage == "evaluate":
            evaluate_campaign(args, transcriber_factory=transcriber_factory)
        else:
            package_campaign(args)
    except Exception as error:
        _append_campaign_error(args.output_dir / "campaign_errors.jsonl", args=args, error=error)
        _write_progress(
            stage=args.stage,
            song_id=args.song,
            protocol_id=args.protocol,
            chunk_index=None,
            ordinal=None,
            total=None,
            status="error",
            error_type=type(error).__name__,
            message=str(error),
        )
        raise
    return 0


def prepare_campaign(args: argparse.Namespace) -> None:
    selected_songs = _selected_songs(args.song)
    common_root = args.output_dir / "common"
    common_root.mkdir(parents=True, exist_ok=True)
    manifest_songs = []
    total_request_count = 0

    for song_index, song_id in enumerate(selected_songs):
        chosen_path = args.source_album / song_id / "chosen_lyrics.jsonl"
        corpus = load_song_corpus(chosen_path, song_id=song_id)
        requests = corpus.two_bar_requests()
        if len(requests) != 25:
            raise ValueError(f"{song_id} must yield exactly 25 requests")
        total_request_count += len(requests)

        song_root = common_root / song_id
        song_root.mkdir(parents=True, exist_ok=True)
        requests_path = song_root / "requests.jsonl"
        request_lines = [
            canonical_json_dumps(
                {
                    **request.to_payload(),
                    "request_sha256": request.sha256,
                }
            )
            for request in requests
        ]
        rendered_requests = "\n".join(request_lines) + "\n"
        if requests_path.exists() and requests_path.read_text(encoding="utf-8") != rendered_requests:
            raise ValueError(f"mismatched existing request set for {song_id}")
        requests_path.write_text(rendered_requests, encoding="utf-8")
        for ordinal, request in enumerate(requests, start=1):
            _write_progress(
                stage="prepare",
                song_id=song_id,
                protocol_id="common",
                chunk_index=request.chunk_index,
                ordinal=ordinal,
                total=len(requests),
                status="ready",
                request_sha256=request.sha256,
            )

        drums_path = song_root / "drums.wav"
        if not drums_path.exists():
            drums = render_common_drums(requests, song_index=song_index)
            write_listening_wav(drums_path, drums)

        manifest_songs.append(
            {
                "song_id": song_id,
                "chosen_lyrics_path": str(chosen_path.resolve()),
                "chosen_lyrics_sha256": file_sha256(chosen_path),
                "request_count": len(requests),
                "requests_path": str(requests_path.relative_to(args.output_dir)),
                "requests_sha256": file_sha256(requests_path),
                "drums_path": str(drums_path.relative_to(args.output_dir)),
                "drums_sha256": file_sha256(drums_path),
            }
        )

    if total_request_count != 75 and args.song is None:
        raise ValueError(f"prepare must emit exactly 75 requests, got {total_request_count}")

    manifest = {
        "schema_version": "streammuse.rap_audio_protocols.corpus_manifest.v1",
        "source_album": str(args.source_album.resolve()),
        "song_count": len(manifest_songs),
        "total_request_count": total_request_count,
        "songs": manifest_songs,
    }
    manifest_path = common_root / "corpus_manifest.json"
    rendered_manifest = json.dumps(manifest, indent=2, sort_keys=True)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != rendered_manifest:
        raise ValueError("mismatched existing manifest")
    manifest_path.write_text(rendered_manifest, encoding="utf-8")


def assemble_campaign(args: argparse.Namespace) -> None:
    protocol_ids = _selected_protocols(args.protocol)
    for song_id in _selected_songs(args.song):
        requests = _load_requests(args.output_dir / "common" / song_id / "requests.jsonl")
        drums_path = args.output_dir / "common" / song_id / "drums.wav"
        drums = _load_stereo_float32_audio(drums_path)
        for protocol_name in protocol_ids:
            protocol_id = ProtocolId(protocol_name)
            protocol_root = args.output_dir / protocol_name / song_id
            records = _load_chunk_records(protocol_root / "render_chunks.jsonl")
            if len(records) != 25:
                raise ValueError(f"{protocol_name}/{song_id} must contain exactly 25 chunk records")
            record_by_chunk = {record.chunk_index: record for record in records}
            chunk_paths = {}
            ordered_records = []
            for ordinal, request in enumerate(requests, start=1):
                record = record_by_chunk.get(request.chunk_index)
                if record is None or not record.success or not record.output_path:
                    raise ValueError(f"missing successful chunk record for {protocol_name}/{song_id}/{request.chunk_index}")
                chunk_path = Path(record.output_path)
                if not chunk_path.is_file():
                    raise ValueError(f"missing rendered WAV for {protocol_name}/{song_id}/{request.chunk_index}: {chunk_path}")
                chunk_paths[request.chunk_index] = chunk_path
                ordered_records.append(record)
                _write_progress(
                    stage="assemble",
                    song_id=song_id,
                    protocol_id=protocol_name,
                    chunk_index=request.chunk_index,
                    ordinal=ordinal,
                    total=len(requests),
                    status="verified",
                    output_sha256=record.output_sha256 or "-",
                )
            vocal_stem = assemble_vocal_stem(
                requests,
                chunk_paths_by_index=chunk_paths,
                listening_wav_path=protocol_root / "vocals.wav",
            )
            mix = mix_stems(vocal_stem.audio, drums, listening_wav_path=protocol_root / "mix.wav")
            if vocal_stem.audio.frame_count != SONG_FRAME_COUNT or mix.audio.frame_count != SONG_FRAME_COUNT:
                raise ValueError("assembled artifacts must contain exactly 6,400,000 frames")
            manifest = build_protocol_artifact_manifest(
                protocol_id,
                requests=requests,
                chunk_records=ordered_records,
                vocal_stem_path=protocol_root / "vocals.wav",
                drums_path=drums_path,
                mix_path=protocol_root / "mix.wav",
            )
            (protocol_root / "artifact_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )


def evaluate_campaign(
    args: argparse.Namespace,
    *,
    transcriber_factory: Callable[..., Callable[[Path], Sequence[Any]]] | None = None,
) -> None:
    transcriber_builder = transcriber_factory or build_faster_whisper_transcriber
    transcriber = transcriber_builder(
        model_size=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )
    aggregate = []
    for song_id in _selected_songs(args.song):
        requests = _load_requests(args.output_dir / "common" / song_id / "requests.jsonl")
        for protocol_name in _selected_protocols(args.protocol):
            protocol_root = args.output_dir / protocol_name / song_id
            ordinal_by_chunk = {
                request.chunk_index: ordinal
                for ordinal, request in enumerate(requests, start=1)
            }

            def report_progress(request: TwoBarRenderRequest, status: str) -> None:
                _write_progress(
                    stage="evaluate",
                    song_id=song_id,
                    protocol_id=protocol_name,
                    chunk_index=request.chunk_index,
                    ordinal=ordinal_by_chunk[request.chunk_index],
                    total=len(requests),
                    status=status,
                )

            metrics = evaluate_protocol_song(
                protocol_id=ProtocolId(protocol_name),
                song_id=song_id,
                requests=requests,
                chunk_records=_load_chunk_records(protocol_root / "render_chunks.jsonl"),
                transcribe_chunk=transcriber,
                progress_callback=report_progress,
            )
            (protocol_root / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
            aggregate.append(metrics)
    (args.output_dir / "experiment_metrics.json").write_text(
        json.dumps({"metrics": aggregate}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def package_campaign(args: argparse.Namespace) -> None:
    selected_songs = _selected_songs(args.song)
    selected_protocols = _selected_protocols(args.protocol)
    selected_matrix = tuple(
        (song_id, protocol_name)
        for song_id in selected_songs
        for protocol_name in selected_protocols
    )
    missing_mixes = tuple(
        args.output_dir / protocol_name / song_id / "mix.wav"
        for song_id, protocol_name in selected_matrix
        if not (args.output_dir / protocol_name / song_id / "mix.wav").is_file()
    )
    if missing_mixes:
        missing_paths = ", ".join(str(path.relative_to(args.output_dir)) for path in missing_mixes)
        raise ValueError(
            "package stage requires complete selected matrix: "
            f"missing {len(missing_mixes)} mix.wav files: {missing_paths}"
        )

    assets = []
    comparison_rows = []
    for song_id in selected_songs:
        for protocol_name in selected_protocols:
            protocol_root = args.output_dir / protocol_name / song_id
            mix_path = protocol_root / "mix.wav"
            assets.append(
                ListeningAsset(
                    song_id=song_id,
                    protocol_id=protocol_name,
                    title=song_id.replace("_", " "),
                    audio_path=mix_path,
                )
            )
            metrics_path = protocol_root / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                metrics = {"failed_chunk_count": "n/a"}
            comparison_rows.append(
                {
                    "song_id": song_id,
                    "protocol_id": protocol_name,
                    "mix_sha256": file_sha256(mix_path),
                    "failed_chunk_count": metrics.get("failed_chunk_count", "n/a"),
                }
            )
            _write_progress(
                stage="package",
                song_id=song_id,
                protocol_id=protocol_name,
                chunk_index=None,
                ordinal=len(assets),
                total=None,
                status="ready",
                mix_sha256=comparison_rows[-1]["mix_sha256"],
            )
    outputs = write_listening_package(output_dir=args.output_dir, assets=assets)
    (args.output_dir / "COMPARISON.md").write_text(_render_comparison_markdown(comparison_rows), encoding="utf-8")
    # write_listening_package already writes the audit file; keep the returned paths live for callers
    _ = outputs


def _render_comparison_markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Rap Audio Protocol Comparison",
        "",
        "| Song | Protocol | Mix SHA-256 | Failed chunks |",
        "| --- | --- | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['song_id']} | {row['protocol_id']} | `{row['mix_sha256']}` | {row['failed_chunk_count']} |"
        )
    return "\n".join(lines) + "\n"


def _selected_songs(selected_song: str | None) -> tuple[str, ...]:
    return (selected_song,) if selected_song is not None else _SONG_IDS


def _selected_protocols(selected_protocol: str | None) -> tuple[str, ...]:
    return (selected_protocol,) if selected_protocol is not None else _PROTOCOL_IDS


def _append_campaign_error(path: Path, *, args: argparse.Namespace, error: Exception) -> None:
    payload = {
        "schema_version": "streammuse.rap_audio_protocols.campaign_error.v1",
        "stage": args.stage,
        "song_id": args.song,
        "protocol_id": args.protocol,
        "chunk_index": None,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json_dumps(payload))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_progress(
    *,
    stage: str,
    song_id: str | None,
    protocol_id: str | None,
    chunk_index: int | None,
    ordinal: int | None,
    total: int | None,
    status: str,
    **details: Any,
) -> None:
    fields = [
        f"stage={stage}",
        f"song={song_id or '-'}",
        f"protocol={protocol_id or '-'}",
        f"chunk={chunk_index:03d}" if chunk_index is not None else "chunk=-",
        f"count={ordinal}/{total}" if ordinal is not None and total is not None else "count=-",
        f"status={status}",
    ]
    fields.extend(
        f"{key}={_progress_value(value)}"
        for key, value in details.items()
    )
    print(" ".join(fields), file=sys.stdout, flush=True)


def _progress_value(value: Any) -> str:
    return str(value).replace("\n", "\\n").replace("\r", "\\r")


def _load_requests(path: Path) -> tuple[TwoBarRenderRequest, ...]:
    requests = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        requests.append(_request_from_payload(payload, path=path, line_number=line_number))
    return tuple(requests)


def _request_from_payload(payload: dict[str, Any], *, path: Path, line_number: int) -> TwoBarRenderRequest:
    syllables = tuple(
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
    )
    request = TwoBarRenderRequest(
        song_id=str(payload["song_id"]),
        chunk_index=int(payload["chunk_index"]),
        start_bar=int(payload["start_bar"]),
        end_bar=int(payload["end_bar"]),
        text=str(payload["text"]),
        syllables=syllables,
    )
    expected_sha = payload.get("request_sha256")
    if expected_sha is not None and expected_sha != request.sha256:
        raise ValueError(f"{path}:{line_number}: request_sha256 mismatch")
    return request


def _load_chunk_records(path: Path) -> tuple[ChunkRenderRecord, ...]:
    if not path.exists():
        raise ValueError(f"missing chunk record ledger: {path}")
    records = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        output_path = payload.get("output_path")
        actual_sha = file_sha256(output_path) if output_path and Path(output_path).is_file() else payload.get("output_sha256")
        sample_rate_hz = payload.get("sample_rate_hz")
        if output_path and Path(output_path).is_file() and sample_rate_hz is None:
            sample_rate_hz = int(wavfile.read(Path(output_path))[0])
        records.append(
            ChunkRenderRecord(
                protocol_id=ProtocolId(str(payload["protocol_id"])),
                song_id=str(payload["song_id"]),
                chunk_index=int(payload["chunk_index"]),
                request_sha256=str(payload["request_sha256"]),
                success=bool(payload["success"]),
                output_path=output_path,
                output_sha256=actual_sha,
                source_chunk_sha256=payload.get("source_chunk_sha256"),
                sample_rate_hz=sample_rate_hz,
                attempts=int(payload.get("attempts", 0)),
                error=payload.get("error"),
            )
        )
    return tuple(sorted(records, key=lambda item: item.chunk_index))


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
