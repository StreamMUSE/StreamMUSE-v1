"""Safe binary ZIP codec for remote rap chunks."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import hashlib
import io
import json
from math import isfinite
from pathlib import PurePosixPath
import wave
import zipfile

from streammuse.domain.rap.remote_chunk import REMOTE_CHUNK_SAMPLE_RATE_HZ, RemoteRapChunkManifest


RAP_CHUNK_PACKAGE_MEDIA_TYPE = "application/vnd.streammuse.rap-chunk+zip"
MAX_RAP_CHUNK_PACKAGE_BYTES = 4 * 1024 * 1024
_MANIFEST_MEMBER = "manifest.json"
_VOCALS_MEMBER = "vocals.wav"
_MEMBERS = (_MANIFEST_MEMBER, _VOCALS_MEMBER)


@dataclass(frozen=True)
class DecodedRapChunkPackage:
    manifest: RemoteRapChunkManifest
    vocal_wav: bytes


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
    except wave.Error as error:
        raise ValueError("invalid vocals WAV") from error

    samples = array("h")
    samples.frombytes(samples_bytes)
    if not samples or not any(sample != 0 for sample in samples):
        raise ValueError("vocals WAV must not be silent")
    if not all(isfinite(float(sample)) for sample in samples):
        raise ValueError("vocals WAV contains non-finite decoded samples")


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
            if any(member.file_size > MAX_RAP_CHUNK_PACKAGE_BYTES for member in members):
                raise ValueError("package member exceeds 4 MiB")
            if sum(member.file_size for member in members) > MAX_RAP_CHUNK_PACKAGE_BYTES:
                raise ValueError("package contents exceed 4 MiB")
            manifest_bytes = archive.read(_MANIFEST_MEMBER)
            vocal_wav = archive.read(_VOCALS_MEMBER)
    except zipfile.BadZipFile as error:
        raise ValueError("invalid rap chunk package ZIP") from error

    try:
        payload = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid manifest JSON") from error
    try:
        manifest = RemoteRapChunkManifest.from_payload(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid manifest JSON contract") from error
    if manifest.request_id != expected_request_id:
        raise ValueError("manifest request_id does not match the requested chunk")
    _validate_wav(vocal_wav, manifest)
    if hashlib.sha256(vocal_wav).hexdigest() != manifest.vocal_sha256:
        raise ValueError("vocals WAV SHA-256 does not match the manifest")
    return DecodedRapChunkPackage(manifest, vocal_wav)
