"""Artifact integrity helpers for offline rap protocol comparisons."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from streammuse.experiments.rap_audio_protocols.audio import validate_wav_metadata
from streammuse.experiments.rap_audio_protocols.contracts import (
    ChunkRenderRecord,
    ProtocolId,
    TwoBarRenderRequest,
    canonical_json_dumps,
    sha256_hex,
)


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_chunk_record_index(path: Path | str) -> dict[tuple[ProtocolId, str, int], ChunkRenderRecord]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return {}
    index: dict[tuple[ProtocolId, str, int], ChunkRenderRecord] = {}
    payloads: dict[tuple[ProtocolId, str, int], str] = {}
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{ledger_path}:{line_number}: expected JSON object")
            record = _record_from_payload(payload)
            key = (record.protocol_id, record.song_id, record.chunk_index)
            rendered = canonical_json_dumps(record.to_payload())
            if key in index:
                if payloads[key] == rendered:
                    raise ValueError(f"duplicate chunk record for {key}")
                raise ValueError(f"conflicting chunk record for {key}")
            index[key] = record
            payloads[key] = rendered
    return index


def append_chunk_record(path: Path | str, record: ChunkRenderRecord) -> str:
    ledger_path = Path(path)
    existing = read_chunk_record_index(ledger_path)
    key = (record.protocol_id, record.song_id, record.chunk_index)
    if key in existing:
        if existing[key].to_payload() == record.to_payload():
            raise ValueError(f"duplicate chunk record for {key}")
        raise ValueError(f"conflicting chunk record for {key}")

    final_record = _hydrate_record(record)
    rendered = canonical_json_dumps(final_record.to_payload())
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.write("\n")
    return rendered


def chunk_record_is_complete(
    ledger_path: Path | str,
    wav_path: Path | str,
    *,
    request: TwoBarRenderRequest,
    protocol_id: ProtocolId,
) -> bool:
    wav = Path(wav_path)
    if not wav.is_file():
        return False
    index = read_chunk_record_index(ledger_path)
    key = (protocol_id, request.song_id, request.chunk_index)
    record = index.get(key)
    if record is None or not record.success:
        return False
    if record.request_sha256 != request.sha256 or record.source_chunk_sha256 != request.sha256:
        return False
    if Path(record.output_path or "").resolve() != wav.resolve():
        return False
    return record.output_sha256 == file_sha256(wav)


def build_protocol_artifact_manifest(
    protocol_id: ProtocolId,
    *,
    requests: tuple[TwoBarRenderRequest, ...] | list[TwoBarRenderRequest],
    chunk_records: tuple[ChunkRenderRecord, ...] | list[ChunkRenderRecord],
    vocal_stem_path: Path | str,
    drums_path: Path | str,
    mix_path: Path | str,
) -> dict[str, Any]:
    payload = {
        "schema_version": "streammuse.rap_audio_protocols.artifact_manifest.v1",
        "protocol_id": protocol_id.value,
        "request_sha256": [request.sha256 for request in requests],
        "source_chunks": [_source_chunk_manifest(record) for record in chunk_records],
        "vocal_stem": _file_manifest(vocal_stem_path),
        "drums": _file_manifest(drums_path),
        "mix": _file_manifest(mix_path),
    }
    return {**payload, "artifact_manifest_sha256": sha256_hex(payload)}


def _file_manifest(path: Path | str) -> dict[str, Any]:
    file_path = Path(path)
    metadata = validate_wav_metadata(file_path)
    return {
        "path": str(file_path),
        "size_bytes": file_path.stat().st_size,
        "sha256": file_sha256(file_path),
        "sample_rate_hz": metadata.sample_rate_hz,
        "channels": metadata.channels,
        "frame_count": metadata.frame_count,
        "metadata_sha256": sha256_hex(
            {
                "sample_rate_hz": metadata.sample_rate_hz,
                "channels": metadata.channels,
                "frame_count": metadata.frame_count,
                "dtype": metadata.dtype,
            }
        ),
    }


def _hydrate_record(record: ChunkRenderRecord) -> ChunkRenderRecord:
    if not record.success:
        return record
    output_path = record.output_path
    if not output_path:
        raise ValueError("successful chunk records require output_path")
    path = Path(output_path)
    resolved_sha = record.output_sha256
    if path.is_file():
        actual_sha = file_sha256(path)
        if resolved_sha is None or resolved_sha == "":
            resolved_sha = actual_sha
        elif resolved_sha != actual_sha:
            raise ValueError("successful chunk record output_sha256 does not match its WAV")
    elif resolved_sha in (None, ""):
        raise ValueError("successful chunk records require output_sha256 when the WAV is unavailable")
    return ChunkRenderRecord(
        protocol_id=record.protocol_id,
        song_id=record.song_id,
        chunk_index=record.chunk_index,
        request_sha256=record.request_sha256,
        success=record.success,
        output_path=record.output_path,
        output_sha256=resolved_sha,
        source_chunk_sha256=record.source_chunk_sha256,
        sample_rate_hz=record.sample_rate_hz,
        attempts=record.attempts,
        error=record.error,
    )


def _record_from_payload(payload: dict[str, Any]) -> ChunkRenderRecord:
    return ChunkRenderRecord(
        protocol_id=ProtocolId(str(payload["protocol_id"])),
        song_id=str(payload["song_id"]),
        chunk_index=int(payload["chunk_index"]),
        request_sha256=str(payload["request_sha256"]),
        success=bool(payload["success"]),
        output_path=payload.get("output_path"),
        output_sha256=payload.get("output_sha256"),
        source_chunk_sha256=payload.get("source_chunk_sha256"),
        sample_rate_hz=payload.get("sample_rate_hz"),
        attempts=int(payload.get("attempts", 0)),
        error=payload.get("error"),
    )


def _source_chunk_manifest(record: ChunkRenderRecord) -> dict[str, Any]:
    payload = record.to_payload()
    return {
        **payload,
        "record_sha256": sha256_hex(payload),
    }
