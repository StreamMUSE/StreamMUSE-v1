"""Artifact integrity helpers for offline rap protocol comparisons."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from streammuse.experiments.rap_audio_protocols.audio import (
    TARGET_SAMPLE_RATE_HZ,
    TARGET_STEREO_FORMAT,
    TARGET_VOCAL_FORMAT,
    validate_request_set,
    validate_wav_metadata,
)
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


def listening_artifact_filename(
    song_title: str,
    renderer: str,
    stem: str,
    *,
    extension: str = ".wav",
) -> str:
    """Build a stable listener-facing filename with explicit provenance."""
    suffix = extension if extension.startswith(".") else f".{extension}"
    if suffix == ".":
        raise ValueError("artifact extension must not be empty")
    parts = tuple(_artifact_slug(value) for value in (song_title, renderer, stem))
    return "_".join(parts) + suffix.lower()


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
    expected_source_sha256: str | None = None,
) -> bool:
    wav = Path(wav_path)
    if not wav.is_file():
        return False
    index = read_chunk_record_index(ledger_path)
    key = (protocol_id, request.song_id, request.chunk_index)
    record = index.get(key)
    if record is None or not record.success:
        return False
    if record.request_sha256 != request.sha256:
        return False
    if not _source_chunk_sha256_is_valid(record):
        return False
    if expected_source_sha256 is not None and record.source_chunk_sha256 != expected_source_sha256:
        return False
    if Path(record.output_path or "").resolve() != wav.resolve():
        return False
    metadata = validate_wav_metadata(wav)
    if record.sample_rate_hz is not None and metadata.sample_rate_hz != record.sample_rate_hz:
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
    allow_smoke_test: bool = False,
) -> dict[str, Any]:
    shape = validate_request_set(requests, allow_smoke_test=allow_smoke_test)
    _validate_chunk_records(protocol_id, requests, chunk_records)
    payload = {
        "schema_version": "streammuse.rap_audio_protocols.artifact_manifest.v1",
        "protocol_id": protocol_id.value,
        "request_sha256": [request.sha256 for request in requests],
        "source_chunks": [_source_chunk_manifest(record) for record in chunk_records],
        "vocal_stem": _file_manifest(
            vocal_stem_path,
            expected_channels=TARGET_VOCAL_FORMAT.channels,
            expected_frame_count=shape.expected_frame_count,
        ),
        "drums": _file_manifest(
            drums_path,
            expected_channels=TARGET_STEREO_FORMAT.channels,
            expected_frame_count=shape.expected_frame_count,
        ),
        "mix": _file_manifest(
            mix_path,
            expected_channels=TARGET_STEREO_FORMAT.channels,
            expected_frame_count=shape.expected_frame_count,
        ),
    }
    return {**payload, "artifact_manifest_sha256": sha256_hex(payload)}


def _file_manifest(
    path: Path | str,
    *,
    expected_channels: int,
    expected_frame_count: int,
) -> dict[str, Any]:
    file_path = Path(path)
    metadata = validate_wav_metadata(
        file_path,
        expected_sample_rate_hz=TARGET_SAMPLE_RATE_HZ,
        expected_channels=expected_channels,
        expected_frame_count=expected_frame_count,
        expected_dtype="int16",
    )
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
    resolved_source_sha = _normalise_optional_sha256(record.source_chunk_sha256, field_name="source_chunk_sha256")
    if path.is_file():
        metadata = validate_wav_metadata(path)
        actual_sha = file_sha256(path)
        if resolved_sha is None or resolved_sha == "":
            resolved_sha = actual_sha
        elif resolved_sha != actual_sha:
            raise ValueError("successful chunk record output_sha256 does not match its WAV")
        if record.sample_rate_hz is None:
            sample_rate_hz = metadata.sample_rate_hz
        elif record.sample_rate_hz != metadata.sample_rate_hz:
            raise ValueError("successful chunk record sample_rate_hz does not match its WAV")
        else:
            sample_rate_hz = record.sample_rate_hz
    elif resolved_sha in (None, ""):
        raise ValueError("successful chunk records require output_sha256 when the WAV is unavailable")
    else:
        sample_rate_hz = record.sample_rate_hz
    return ChunkRenderRecord(
        protocol_id=record.protocol_id,
        song_id=record.song_id,
        chunk_index=record.chunk_index,
        request_sha256=record.request_sha256,
        success=record.success,
        output_path=record.output_path,
        output_sha256=resolved_sha,
        source_chunk_sha256=resolved_source_sha,
        sample_rate_hz=sample_rate_hz,
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
        source_chunk_sha256=_normalise_optional_sha256(payload.get("source_chunk_sha256"), field_name="source_chunk_sha256"),
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


def _validate_chunk_records(
    protocol_id: ProtocolId,
    requests: tuple[TwoBarRenderRequest, ...] | list[TwoBarRenderRequest],
    chunk_records: tuple[ChunkRenderRecord, ...] | list[ChunkRenderRecord],
) -> None:
    if len(chunk_records) != len(requests):
        raise ValueError("chunk records must match the request set exactly")
    request_by_chunk = {request.chunk_index: request for request in requests}
    seen: set[int] = set()
    for record in chunk_records:
        if record.protocol_id != protocol_id:
            raise ValueError("chunk record protocol mismatch")
        if not _source_chunk_sha256_is_valid(record):
            raise ValueError(
                "moss_aligned chunk records require a nonempty valid source_chunk_sha256"
                if record.protocol_id == ProtocolId.MOSS_ALIGNED
                else "source_chunk_sha256 must be a 64-character lowercase SHA-256 hex digest"
            )
        request = request_by_chunk.get(record.chunk_index)
        if request is None or record.song_id != request.song_id or record.request_sha256 != request.sha256:
            raise ValueError("chunk records must match the request set exactly")
        if record.chunk_index in seen:
            raise ValueError("chunk records must be unique by chunk_index")
        seen.add(record.chunk_index)


def _normalise_optional_sha256(value: Any, *, field_name: str) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not _is_sha256_hex(value):
        raise ValueError(f"{field_name} must be a 64-character lowercase SHA-256 hex digest")
    return value


def _is_sha256_hex(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _artifact_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("artifact filename labels must contain letters or digits")
    return slug


def _source_chunk_sha256_is_valid(record: ChunkRenderRecord) -> bool:
    source_sha = record.source_chunk_sha256
    if source_sha in (None, ""):
        return record.protocol_id != ProtocolId.MOSS_ALIGNED
    return isinstance(source_sha, str) and _is_sha256_hex(source_sha)
