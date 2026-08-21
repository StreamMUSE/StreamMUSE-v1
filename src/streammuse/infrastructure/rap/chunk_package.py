"""Safe binary ZIP codec for remote rap chunks."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import io
import json
from math import isfinite
from pathlib import PurePosixPath
from typing import Protocol
import wave
import zipfile
import zlib

from streammuse.domain.rap.remote_chunk import (
    REMOTE_CHUNK_PACKAGE_MAX_BYTES,
    REMOTE_CHUNK_SAMPLE_RATE_HZ,
    RemoteRapChunkManifest,
)


RAP_CHUNK_PACKAGE_MEDIA_TYPE = "application/vnd.streammuse.rap-chunk+zip"
RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE = "application/vnd.streammuse.rap-chunk-opus+zip"
MAX_RAP_CHUNK_PACKAGE_BYTES = REMOTE_CHUNK_PACKAGE_MAX_BYTES
_MANIFEST_MEMBER = "manifest.json"
_VOCALS_MEMBER = "vocals.wav"
_MEMBERS = (_MANIFEST_MEMBER, _VOCALS_MEMBER)
_OPUS_MEMBER = "vocals.opus"
_TRANSPORT_MEMBER = "transport.json"
_OPUS_MEMBERS = (_MANIFEST_MEMBER, _TRANSPORT_MEMBER, _OPUS_MEMBER)
_ALLOWED_COMPRESSIONS = {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
_OPUS_TRANSPORT_SCHEMA_VERSION = "streammuse.rap_chunk_opus.v1"


@dataclass(frozen=True)
class DecodedRapChunkPackage:
    manifest: RemoteRapChunkManifest
    vocal_wav: bytes
    transport_codec: str = "pcm"

    def __post_init__(self) -> None:
        if self.transport_codec not in {"pcm", "opus"}:
            raise ValueError("transport_codec must be pcm or opus")


class _OpusCodec(Protocol):
    encoder_identity: str

    def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes: ...

    def decode_to_pcm16_mono_24khz(self, encoded: bytes, *, expected_frame_count: int) -> bytes: ...


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def _validate_wav(vocal_wav: bytes, manifest: RemoteRapChunkManifest) -> None:
    try:
        with wave.open(io.BytesIO(vocal_wav), "rb") as wav:
            if wav.getcomptype() != "NONE" or wav.getsampwidth() != 2:
                raise ValueError("vocals WAV must be uncompressed PCM16")
            if wav.getnchannels() != 1:
                raise ValueError("vocals WAV must be mono")
            if wav.getframerate() != REMOTE_CHUNK_SAMPLE_RATE_HZ:
                raise ValueError("vocals WAV sample rate does not match the 24 kHz contract")
            if wav.getnframes() != manifest.expected_frame_count:
                raise ValueError("vocals WAV frame count does not match the manifest")
            samples_bytes = wav.readframes(wav.getnframes())
    except (EOFError, wave.Error) as error:
        raise ValueError("invalid vocals WAV") from error

    expected_bytes = manifest.expected_frame_count * 2
    if len(samples_bytes) != expected_bytes:
        raise ValueError("vocals WAV data payload is truncated")

    samples = array("h")
    samples.frombytes(samples_bytes)
    if not samples or not any(sample != 0 for sample in samples):
        raise ValueError("vocals WAV must not be silent")
    if not all(isfinite(float(sample)) for sample in samples):
        raise ValueError("vocals WAV contains non-finite decoded samples")


def _pcm_from_wav(vocal_wav: bytes, manifest: RemoteRapChunkManifest) -> bytes:
    _validate_wav(vocal_wav, manifest)
    with wave.open(io.BytesIO(vocal_wav), "rb") as wav:
        return wav.readframes(wav.getnframes())


def _wav_from_pcm(pcm: bytes, manifest: RemoteRapChunkManifest) -> bytes:
    if len(pcm) != manifest.expected_frame_count * 2:
        raise ValueError("decoded Opus frame count does not match the manifest")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(REMOTE_CHUNK_SAMPLE_RATE_HZ)
        wav.writeframes(pcm)
    vocal_wav = buffer.getvalue()
    _validate_wav(vocal_wav, manifest)
    return vocal_wav


def _validate_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("package contains a path traversal member name")


def encode_chunk_package(manifest: RemoteRapChunkManifest, vocal_wav: bytes) -> bytes:
    """Encode a canonical manifest and exact WAV bytes into a deterministic ZIP."""
    if not isinstance(manifest, RemoteRapChunkManifest):
        raise ValueError("manifest must be a RemoteRapChunkManifest")
    if not isinstance(vocal_wav, bytes):
        raise ValueError("vocal_wav must be bytes")
    _validate_wav(vocal_wav, manifest)
    if hashlib.sha256(vocal_wav).hexdigest() != manifest.vocal_sha256:
        raise ValueError("vocals WAV SHA-256 does not match the manifest")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=True) as archive:
        archive.writestr(_zip_info(_MANIFEST_MEMBER), manifest.canonical_json_bytes())
        archive.writestr(_zip_info(_VOCALS_MEMBER), vocal_wav)
    package = buffer.getvalue()
    if len(package) > MAX_RAP_CHUNK_PACKAGE_BYTES:
        raise ValueError("rap chunk package exceeds 4 MiB")
    return package


def encode_opus_chunk_package(
    manifest: RemoteRapChunkManifest, vocal_wav: bytes, codec: _OpusCodec
) -> bytes:
    """Encode a lossily transported Opus variant while retaining the canonical manifest."""
    if not isinstance(manifest, RemoteRapChunkManifest):
        raise ValueError("manifest must be a RemoteRapChunkManifest")
    if hashlib.sha256(vocal_wav).hexdigest() != manifest.vocal_sha256:
        raise ValueError("vocals WAV SHA-256 does not match the manifest")
    pcm = _pcm_from_wav(vocal_wav, manifest)
    encoded = codec.encode_pcm16_mono_24khz(pcm, expected_frame_count=manifest.expected_frame_count)
    if not isinstance(encoded, bytes) or not encoded or len(encoded) > MAX_RAP_CHUNK_PACKAGE_BYTES:
        raise ValueError("Opus encoder returned invalid or oversized output")
    identity = getattr(codec, "encoder_identity", None)
    if not isinstance(identity, str) or not identity or len(identity) > 128 or any(ord(char) < 32 for char in identity):
        raise ValueError("Opus encoder identity is invalid")
    transport = {
        "schema_version": _OPUS_TRANSPORT_SCHEMA_VERSION,
        "codec": "opus",
        "container": "ogg",
        "sample_rate_hz": REMOTE_CHUNK_SAMPLE_RATE_HZ,
        "channels": 1,
        "expected_frame_count": manifest.expected_frame_count,
        "bitrate_bps": 48_000,
        "encoder": identity,
        "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=True) as archive:
        archive.writestr(_zip_info(_MANIFEST_MEMBER), manifest.canonical_json_bytes())
        archive.writestr(_zip_info(_TRANSPORT_MEMBER), json.dumps(transport, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        archive.writestr(_zip_info(_OPUS_MEMBER), encoded)
    package = buffer.getvalue()
    if len(package) > MAX_RAP_CHUNK_PACKAGE_BYTES:
        raise ValueError("rap chunk Opus package exceeds 4 MiB")
    return package


def decode_chunk_package(package: bytes, *, expected_request_id: str) -> DecodedRapChunkPackage:
    """Decode and validate an in-memory chunk package without extracting files."""
    if not isinstance(package, bytes):
        raise ValueError("rap chunk package must be bytes")
    if len(package) > MAX_RAP_CHUNK_PACKAGE_BYTES:
        raise ValueError("rap chunk package exceeds 4 MiB")
    if not isinstance(expected_request_id, str) or not expected_request_id:
        raise ValueError("expected_request_id must be a non-empty string")
    try:
        with zipfile.ZipFile(io.BytesIO(package), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            for name in names:
                _validate_name(name)
            if len(names) != len(set(names)):
                raise ValueError("package contains duplicate members")
            if set(names) != set(_MEMBERS):
                raise ValueError("package contains unexpected or missing members")
            if any(member.flag_bits & 0x1 for member in members):
                raise ValueError("package members must not be encrypted")
            if any(member.compress_type not in _ALLOWED_COMPRESSIONS for member in members):
                raise ValueError("package members use unsupported compression")
            if any(member.file_size > MAX_RAP_CHUNK_PACKAGE_BYTES for member in members):
                raise ValueError("package member exceeds 4 MiB")
            if sum(member.file_size for member in members) > MAX_RAP_CHUNK_PACKAGE_BYTES:
                raise ValueError("package contents exceed 4 MiB")
            manifest_bytes = archive.read(_MANIFEST_MEMBER)
            vocal_wav = archive.read(_VOCALS_MEMBER)
    except (zipfile.BadZipFile, EOFError, NotImplementedError, OSError, RuntimeError, zlib.error) as error:
        raise ValueError("invalid rap chunk package ZIP") from error

    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid manifest JSON") from error
    try:
        manifest = RemoteRapChunkManifest.from_payload(payload)
    except (RecursionError, TypeError, ValueError) as error:
        raise ValueError("invalid manifest JSON contract") from error
    if manifest.request_id != expected_request_id:
        raise ValueError("manifest request_id does not match the requested chunk")
    _validate_wav(vocal_wav, manifest)
    if hashlib.sha256(vocal_wav).hexdigest() != manifest.vocal_sha256:
        raise ValueError("vocals WAV SHA-256 does not match the manifest")
    return DecodedRapChunkPackage(manifest, vocal_wav)


def decode_opus_chunk_package(
    package: bytes, *, expected_request_id: str, codec: _OpusCodec
) -> DecodedRapChunkPackage:
    """Validate a strict Opus ZIP before decoding it into canonical PCM WAV bytes."""
    manifest_bytes, transport_bytes, encoded = _read_strict_members(package, _OPUS_MEMBERS)
    manifest = _decode_manifest(manifest_bytes, expected_request_id)
    try:
        transport = json.loads(transport_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid Opus transport JSON") from error
    _validate_opus_transport(transport, manifest)
    if hashlib.sha256(encoded).hexdigest() != transport["encoded_sha256"]:
        raise ValueError("Opus encoded SHA-256 does not match transport metadata")
    try:
        pcm = codec.decode_to_pcm16_mono_24khz(encoded, expected_frame_count=manifest.expected_frame_count)
    except Exception as error:
        raise ValueError("Opus payload could not be decoded") from error
    return DecodedRapChunkPackage(manifest, _wav_from_pcm(pcm, manifest), "opus")


def _read_strict_members(package: bytes, expected_members: tuple[str, ...]) -> tuple[bytes, ...]:
    if not isinstance(package, bytes):
        raise ValueError("rap chunk package must be bytes")
    if len(package) > MAX_RAP_CHUNK_PACKAGE_BYTES:
        raise ValueError("rap chunk package exceeds 4 MiB")
    try:
        with zipfile.ZipFile(io.BytesIO(package), "r") as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            for name in names:
                _validate_name(name)
            if len(names) != len(set(names)):
                raise ValueError("package contains duplicate members")
            if set(names) != set(expected_members):
                raise ValueError("package contains unexpected or missing members")
            if any(member.flag_bits & 0x1 for member in members):
                raise ValueError("package members must not be encrypted")
            if any(member.compress_type not in _ALLOWED_COMPRESSIONS for member in members):
                raise ValueError("package members use unsupported compression")
            if any(member.file_size > MAX_RAP_CHUNK_PACKAGE_BYTES for member in members):
                raise ValueError("package member exceeds 4 MiB")
            if sum(member.file_size for member in members) > MAX_RAP_CHUNK_PACKAGE_BYTES:
                raise ValueError("package contents exceed 4 MiB")
            return tuple(archive.read(name) for name in expected_members)
    except (zipfile.BadZipFile, EOFError, NotImplementedError, OSError, RuntimeError, zlib.error) as error:
        raise ValueError("invalid rap chunk package ZIP") from error


def _decode_manifest(manifest_bytes: bytes, expected_request_id: str) -> RemoteRapChunkManifest:
    if not isinstance(expected_request_id, str) or not expected_request_id:
        raise ValueError("expected_request_id must be a non-empty string")
    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
        manifest = RemoteRapChunkManifest.from_payload(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as error:
        raise ValueError("invalid manifest JSON contract") from error
    if manifest.request_id != expected_request_id:
        raise ValueError("manifest request_id does not match the requested chunk")
    return manifest


def _validate_opus_transport(payload: object, manifest: RemoteRapChunkManifest) -> None:
    required = {
        "schema_version", "codec", "container", "sample_rate_hz", "channels", "expected_frame_count",
        "bitrate_bps", "encoder", "encoded_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Opus transport JSON has an invalid schema")
    if (
        payload["schema_version"] != _OPUS_TRANSPORT_SCHEMA_VERSION
        or payload["codec"] != "opus"
        or payload["container"] != "ogg"
        or payload["sample_rate_hz"] != REMOTE_CHUNK_SAMPLE_RATE_HZ
        or payload["channels"] != 1
        or payload["expected_frame_count"] != manifest.expected_frame_count
        or payload["bitrate_bps"] != 48_000
    ):
        raise ValueError("Opus transport JSON does not match the manifest contract")
    identity = payload["encoder"]
    digest = payload["encoded_sha256"]
    if not isinstance(identity, str) or not identity or len(identity) > 128 or any(ord(char) < 32 for char in identity):
        raise ValueError("Opus transport encoder identity is invalid")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Opus transport SHA-256 is invalid")
