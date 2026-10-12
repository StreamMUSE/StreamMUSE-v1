"""Protocol v3 bar audio package: manifest plus the raw MOSS line, as PCM16 or Opus."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from typing import Protocol
import wave
import zipfile

from streammuse.domain.rap.lyric_stream import BAR_AUDIO_SAMPLE_RATE_HZ, BarAudioManifest

BAR_PACKAGE_MEDIA_TYPE = "application/vnd.streammuse.rap-bar+zip"
BAR_OPUS_PACKAGE_MEDIA_TYPE = "application/vnd.streammuse.rap-bar-opus+zip"
_MAX_PACKAGE_BYTES = 4 * 1024 * 1024
_MANIFEST = "manifest.json"
_SOURCE_WAV = "source.wav"
_SOURCE_OPUS = "source.opus"


class OpusCodec(Protocol):
    encoder_identity: str

    def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes: ...

    def decode_to_pcm16_mono_24khz(self, encoded: bytes, *, expected_frame_count: int) -> bytes: ...


@dataclass(frozen=True)
class DecodedBarPackage:
    manifest: BarAudioManifest
    source_wav: bytes
    transport: str  # "pcm" | "opus"


def _pcm16(source_wav: bytes, manifest: BarAudioManifest) -> bytes:
    with wave.open(io.BytesIO(source_wav)) as handle:
        if (
            handle.getnchannels() != 1
            or handle.getsampwidth() != 2
            or handle.getframerate() != BAR_AUDIO_SAMPLE_RATE_HZ
            or handle.getnframes() != manifest.frame_count
        ):
            raise ValueError("bar source must be mono 24 kHz PCM16 with the manifest's frame count")
        return handle.readframes(handle.getnframes())


def _wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(BAR_AUDIO_SAMPLE_RATE_HZ)
        handle.writeframes(pcm)
    return buffer.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, data)
    data = buffer.getvalue()
    if len(data) > _MAX_PACKAGE_BYTES:
        raise ValueError("bar package exceeds 4 MiB")
    return data


class PcmBarPackageEncoder:
    media_type = BAR_PACKAGE_MEDIA_TYPE

    def encode(self, manifest: BarAudioManifest, source_wav: bytes) -> bytes:
        if hashlib.sha256(source_wav).hexdigest() != manifest.source_sha256:
            raise ValueError("bar source does not match the manifest hash")
        _pcm16(source_wav, manifest)
        return _zip({_MANIFEST: manifest.canonical_json_bytes(), _SOURCE_WAV: source_wav})


class OpusBarPackageEncoder:
    media_type = BAR_OPUS_PACKAGE_MEDIA_TYPE

    def __init__(self, codec: OpusCodec) -> None:
        self._codec = codec

    def encode(self, manifest: BarAudioManifest, source_wav: bytes) -> bytes:
        if hashlib.sha256(source_wav).hexdigest() != manifest.source_sha256:
            raise ValueError("bar source does not match the manifest hash")
        encoded = self._codec.encode_pcm16_mono_24khz(
            _pcm16(source_wav, manifest), expected_frame_count=manifest.frame_count
        )
        return _zip({_MANIFEST: manifest.canonical_json_bytes(), _SOURCE_OPUS: encoded})


def decode_bar_package(
    package: bytes, *, expected_lyric_id: str, codec: OpusCodec | None = None
) -> DecodedBarPackage:
    if len(package) > _MAX_PACKAGE_BYTES:
        raise ValueError("bar package exceeds 4 MiB")
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
        if names not in ({_MANIFEST, _SOURCE_WAV}, {_MANIFEST, _SOURCE_OPUS}):
            raise ValueError("bar package has unexpected members")
        manifest = BarAudioManifest.from_payload(json.loads(archive.read(_MANIFEST)))
        if manifest.lyric_id != expected_lyric_id:
            raise ValueError("bar package is for a different lyric")
        if _SOURCE_WAV in names:
            source_wav = archive.read(_SOURCE_WAV)
            if hashlib.sha256(source_wav).hexdigest() != manifest.source_sha256:
                raise ValueError("bar source does not match the manifest hash")
            _pcm16(source_wav, manifest)
            return DecodedBarPackage(manifest, source_wav, "pcm")
        if codec is None:
            raise ValueError("an Opus bar package needs an Opus codec")
        pcm = codec.decode_to_pcm16_mono_24khz(archive.read(_SOURCE_OPUS), expected_frame_count=manifest.frame_count)
        return DecodedBarPackage(manifest, _wav(pcm), "opus")
