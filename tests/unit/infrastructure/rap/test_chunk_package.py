from __future__ import annotations

import hashlib
import io
import json
import struct
import wave
import zipfile
from contextlib import nullcontext

import pytest

from streammuse.domain.rap import (
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    RemoteCandidatePolicy,
    RemoteCandidateStats,
    RemoteRapBarRequest,
    RemoteRapChunkDiagnostics,
    RemoteRapChunkManifest,
    RemoteRapChunkRequest,
    RemoteSelectedBar,
    ScheduledSyllable,
    Syllable,
    materialize_flow,
)
from streammuse.infrastructure.rap.chunk_package import (
    RAP_CHUNK_PACKAGE_MEDIA_TYPE,
    decode_chunk_package,
    encode_chunk_package,
)


def _flow() -> FlowTemplate:
    return FlowTemplate(
        template_id="test_flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(FlowSlot(tick_in_bar=0, duration_ticks=1, target_stress=1.0),),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


def _wav_bytes(
    *,
    frame_count: int = 128_000,
    sample_rate_hz: int = 24_000,
    channels: int = 1,
    sample_width: int = 2,
    sample: int = 1_000,
) -> bytes:
    frame = struct.pack("<h", sample) if sample_width == 2 else b"\x00" * sample_width
    payload = frame * frame_count * channels
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate_hz)
        wav.writeframes(payload)
    return buffer.getvalue()


@pytest.fixture
def manifest() -> RemoteRapChunkManifest:
    flow = _flow()
    request = RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=0,
        bars=(
            RemoteRapBarRequest(0, "space", flow),
            RemoteRapBarRequest(1, "space", flow),
        ),
        tempo_bpm=90.0,
        remaining_budget_ms=5_000,
        policy=RemoteCandidatePolicy.realtime_default(),
        context_lines=(),
        seed=7,
    )
    selected = tuple(
        RemoteSelectedBar.create(
            request.bars[bar],
            text="orbit",
            scheduled=tuple(
                ScheduledSyllable(slot=slot, syllable=Syllable("orbit", 0, 1, 1))
                for slot in materialize_flow(flow, bar=bar)
            ),
            score=0.9,
        )
        for bar in range(2)
    )
    vocal_wav = _wav_bytes()
    return RemoteRapChunkManifest(
        request_id=request.request_id,
        chunk_index=request.chunk_index,
        tempo_bpm=request.tempo_bpm,
        output_sample_rate_hz=request.output_sample_rate_hz,
        expected_frame_count=request.expected_frame_count,
        selected_bars=selected,
        diagnostics=RemoteRapChunkDiagnostics(
            accepted_request_budget_ms=5_000,
            resolved_policy=RemoteCandidatePolicy.realtime_default(),
            candidate_stats=RemoteCandidateStats(
                6,
                5,
                3,
                2,
                (
                    {
                        "bar": 0,
                        "candidate_id": "candidate-1",
                        "text": "orbit",
                        "score": 0.9,
                        "component_scores": {"flow": 0.9},
                        "source_order": 0,
                    },
                ),
                (),
            ),
            stage_timings_ms={
                "generation": 1.0,
                "evaluation": 1.0,
                "moss": 1.0,
                "aligner": 1.0,
                "warp": 1.0,
                "packaging": 1.0,
                "total": 7.0,
            },
            alignment_diagnostics={
                "fallback_counts": {"word": 0},
                "source_anchors": [0.0],
                "target_anchors": [0.0],
                "local_warp_ratios": [1.0],
            },
            audio_diagnostics={
                "sample_rate_hz": 24_000,
                "frame_count": 128_000,
                "duration_seconds": 128_000 / 24_000,
                "peak": 0.5,
            },
            model_tool_versions={"moss": "test", "aligner": "test", "rubberband": "test"},
            warnings=(),
        ),
        vocal_sha256=hashlib.sha256(vocal_wav).hexdigest(),
    )


@pytest.fixture
def vocal_wav_bytes() -> bytes:
    return _wav_bytes()


def _archive(members: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with pytest.warns(UserWarning, match="Duplicate name") if len({name for name, _ in members}) != len(members) else nullcontext():
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in members:
                archive.writestr(name, data)
    return buffer.getvalue()


def _with_unsupported_compression(package: bytes) -> bytes:
    modified = bytearray(package)
    local_header = modified.index(b"PK\x03\x04")
    central_header = modified.index(b"PK\x01\x02")
    struct.pack_into("<H", modified, local_header + 8, zipfile.ZIP_BZIP2)
    struct.pack_into("<H", modified, central_header + 10, zipfile.ZIP_BZIP2)
    return bytes(modified)


def _nested_json(depth: int) -> object:
    value: object = "leaf"
    for _ in range(depth):
        value = {"next": value}
    return value


def test_package_round_trip(manifest: RemoteRapChunkManifest, vocal_wav_bytes: bytes) -> None:
    encoded = encode_chunk_package(manifest, vocal_wav_bytes)
    decoded = decode_chunk_package(encoded, expected_request_id=manifest.request_id)

    assert decoded.manifest == manifest
    assert decoded.vocal_wav == vocal_wav_bytes


def test_package_uses_deterministic_member_names_and_canonical_manifest_bytes(
    manifest: RemoteRapChunkManifest, vocal_wav_bytes: bytes
) -> None:
    first = encode_chunk_package(manifest, vocal_wav_bytes)
    second = encode_chunk_package(manifest, vocal_wav_bytes)

    assert RAP_CHUNK_PACKAGE_MEDIA_TYPE == "application/vnd.streammuse.rap-chunk+zip"
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == ["manifest.json", "vocals.wav"]
        assert archive.read("manifest.json") == manifest.canonical_json_bytes()


def test_package_rejects_hash_mismatch(manifest: RemoteRapChunkManifest, vocal_wav_bytes: bytes) -> None:
    bad_manifest = RemoteRapChunkManifest(
        **{**manifest.__dict__, "vocal_sha256": "0" * 64}
    )
    encoded = _archive(
        [("manifest.json", bad_manifest.canonical_json_bytes()), ("vocals.wav", vocal_wav_bytes)]
    )

    with pytest.raises(ValueError, match="SHA-256"):
        decode_chunk_package(encoded, expected_request_id=manifest.request_id)


@pytest.mark.parametrize(
    ("members", "message"),
    (
        (
            [("manifest.json", b"{}"), ("manifest.json", b"{}"), ("vocals.wav", b"wav")],
            "duplicate",
        ),
        ([("../manifest.json", b"{}"), ("vocals.wav", b"wav")], "path traversal"),
        ([("manifest.json", b"{}"), ("vocals.wav", b"wav"), ("extra.txt", b"x")], "unexpected"),
    ),
)
def test_package_rejects_unsafe_members(members: list[tuple[str, bytes]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_chunk_package(_archive(members), expected_request_id="request-1")


def test_package_rejects_oversized_archive() -> None:
    with pytest.raises(ValueError, match="4 MiB"):
        decode_chunk_package(b"x" * (4 * 1024 * 1024 + 1), expected_request_id="request-1")


def test_package_rejects_oversized_compressed_member() -> None:
    encoded = _archive([("manifest.json", b"{}"), ("vocals.wav", b"\0" * (4 * 1024 * 1024 + 1))])

    with pytest.raises(ValueError, match="member exceeds 4 MiB"):
        decode_chunk_package(encoded, expected_request_id="request-1")


def test_package_rejects_oversized_combined_uncompressed_contents() -> None:
    encoded = _archive(
        [
            ("manifest.json", b"\0" * (2 * 1024 * 1024 + 1)),
            ("vocals.wav", b"\0" * (2 * 1024 * 1024)),
        ]
    )

    with pytest.raises(ValueError, match="contents exceed 4 MiB"):
        decode_chunk_package(encoded, expected_request_id="request-1")


def test_package_rejects_malformed_manifest_json(vocal_wav_bytes: bytes) -> None:
    encoded = _archive([("manifest.json", b"{"), ("vocals.wav", vocal_wav_bytes)])

    with pytest.raises(ValueError, match="manifest JSON"):
        decode_chunk_package(encoded, expected_request_id="request-1")


def test_package_rejects_unsupported_member_compression(
    manifest: RemoteRapChunkManifest, vocal_wav_bytes: bytes
) -> None:
    encoded = _with_unsupported_compression(encode_chunk_package(manifest, vocal_wav_bytes))

    with pytest.raises(ValueError, match="compression"):
        decode_chunk_package(encoded, expected_request_id=manifest.request_id)


def test_package_normalizes_excessively_nested_manifest_json(vocal_wav_bytes: bytes) -> None:
    nested_json = b"[" * 1_100 + b"0" + b"]" * 1_100
    encoded = _archive([("manifest.json", nested_json), ("vocals.wav", vocal_wav_bytes)])

    with pytest.raises(ValueError, match="manifest JSON"):
        decode_chunk_package(encoded, expected_request_id="request-1")


def test_package_normalizes_post_parse_nested_manifest_diagnostics(
    manifest: RemoteRapChunkManifest, vocal_wav_bytes: bytes
) -> None:
    payload = manifest.to_payload()
    payload["selected_bars"][0]["diagnostics"]["detail"] = _nested_json(33)  # type: ignore[index,union-attr]
    encoded = _archive(
        [("manifest.json", json.dumps(payload, allow_nan=False).encode("utf-8")), ("vocals.wav", vocal_wav_bytes)]
    )

    with pytest.raises(ValueError, match="manifest JSON contract"):
        decode_chunk_package(encoded, expected_request_id=manifest.request_id)


@pytest.mark.parametrize(
    ("wav_bytes", "message"),
    (
        (_wav_bytes(sample_width=1), "PCM16"),
        (_wav_bytes(channels=2), "mono"),
        (_wav_bytes(sample_rate_hz=48_000), "sample rate"),
        (_wav_bytes(frame_count=127_999), "frame count"),
        (_wav_bytes(sample=0), "silent"),
    ),
)
def test_package_rejects_invalid_wav(
    manifest: RemoteRapChunkManifest, wav_bytes: bytes, message: str
) -> None:
    encoded = _archive([("manifest.json", manifest.canonical_json_bytes()), ("vocals.wav", wav_bytes)])

    with pytest.raises(ValueError, match=message):
        decode_chunk_package(encoded, expected_request_id=manifest.request_id)


def test_package_rejects_non_finite_decoded_samples(manifest: RemoteRapChunkManifest) -> None:
    float_wav = io.BytesIO()
    with wave.open(float_wav, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(4)
        wav.setframerate(24_000)
        wav.writeframes(struct.pack("<f", float("nan")) * 128_000)
    encoded = _archive(
        [("manifest.json", manifest.canonical_json_bytes()), ("vocals.wav", float_wav.getvalue())]
    )

    with pytest.raises(ValueError, match="PCM16|non-finite"):
        decode_chunk_package(encoded, expected_request_id=manifest.request_id)


def test_package_rejects_wav_with_truncated_data_payload(
    manifest: RemoteRapChunkManifest, vocal_wav_bytes: bytes
) -> None:
    encoded = _archive(
        [("manifest.json", manifest.canonical_json_bytes()), ("vocals.wav", vocal_wav_bytes[:-2])]
    )

    with pytest.raises(ValueError, match="truncated"):
        decode_chunk_package(encoded, expected_request_id=manifest.request_id)


def test_package_normalizes_truncated_riff_header_to_value_error(manifest: RemoteRapChunkManifest) -> None:
    encoded = _archive([("manifest.json", manifest.canonical_json_bytes()), ("vocals.wav", b"RIFF")])

    with pytest.raises(ValueError, match="invalid vocals WAV"):
        decode_chunk_package(encoded, expected_request_id=manifest.request_id)
