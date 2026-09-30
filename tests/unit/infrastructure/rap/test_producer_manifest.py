from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from streammuse.infrastructure.rap.producer_manifest import (
    PRODUCER_FINGERPRINT_FILE,
    PRODUCER_MANIFEST_FILE,
    ProducerManifestError,
    ProducerManifestV1,
    canonical_json_bytes,
    initialize_producer_namespace,
    verify_producer_namespace,
)


def _manifest(**overrides: object) -> ProducerManifestV1:
    values = {
        "backend": "sglang-omni",
        "backend_implementation_revision": "sglang-moss-adapter.v1",
        "streammuse_revision": {"commit": "abc123", "patch_sha256": "1" * 64},
        "model": {"id": "OpenMOSS-Team/MOSS-TTS-v1.5", "revision": "model-1"},
        "runtime": {
            "identity": "sglang-omni",
            "version": "0.1.4",
            "revision": "runtime-1",
            "config_sha256": "2" * 64,
        },
        "generation": {
            "seed_policy_version": "streammuse.moss_seed.v1",
            "settings": {"audio_top_k": 25, "audio_top_p": 0.8},
        },
        "reference": {"audio_sha256": "3" * 64, "text_sha256": "4" * 64},
        "alignment": {
            "identity": "torchaudio.pipelines.MMS_FA",
            "version": "2.8.0",
            "warp_policy": "gentle_sparse_r3",
        },
        "output": {
            "sample_rate_hz": 24_000,
            "wire_audio_codec": "pcm",
            "public_schema_version": "streammuse.rap_chunk.v1",
        },
    }
    values.update(overrides)
    return ProducerManifestV1(**values)  # type: ignore[arg-type]


def test_canonical_json_and_fingerprint_ignore_mapping_order() -> None:
    left = _manifest()
    right = _manifest(
        generation={
            "settings": {"audio_top_p": 0.8, "audio_top_k": 25},
            "seed_policy_version": "streammuse.moss_seed.v1",
        }
    )

    assert left.canonical_bytes() == right.canonical_bytes()
    assert left.fingerprint == right.fingerprint
    assert len(left.fingerprint) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("backend", "inprocess"),
        ("model", {"id": "OpenMOSS-Team/MOSS-TTS-v1.5", "revision": "model-2"}),
        (
            "runtime",
            {
                "identity": "sglang-omni",
                "version": "0.1.4",
                "revision": "runtime-2",
                "config_sha256": "9" * 64,
            },
        ),
        (
            "reference",
            {"audio_sha256": "8" * 64, "text_sha256": "4" * 64},
        ),
        (
            "alignment",
            {"identity": "MMS_FA", "version": "2.8.0", "warp_policy": "all_onsets_r3"},
        ),
        (
            "generation",
            {
                "seed_policy_version": "streammuse.moss_seed.v1",
                "settings": {"audio_top_k": 26, "audio_top_p": 0.8},
            },
        ),
    ),
)
def test_output_affecting_change_changes_fingerprint(field: str, value: object) -> None:
    assert _manifest().fingerprint != _manifest(**{field: value}).fingerprint


def test_non_json_and_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _manifest(generation={"seed_policy_version": "v1", "settings": {"x": float("nan")}})
    with pytest.raises(ValueError, match="non-JSON"):
        _manifest(generation={"seed_policy_version": "v1", "settings": {"x": object()}})


def test_namespace_is_full_fingerprint_and_is_idempotently_verified(tmp_path: Path) -> None:
    manifest = _manifest()
    namespace = initialize_producer_namespace(tmp_path, manifest)

    assert namespace == tmp_path / manifest.fingerprint
    assert json.loads((namespace / PRODUCER_MANIFEST_FILE).read_text()) == manifest.to_payload()
    assert (namespace / PRODUCER_FINGERPRINT_FILE).read_text() == f"{manifest.fingerprint}\n"
    assert initialize_producer_namespace(tmp_path, manifest) == namespace
    verify_producer_namespace(namespace, manifest)


def test_concurrent_namespace_initialization_is_atomic(tmp_path: Path) -> None:
    manifest = _manifest()

    with ThreadPoolExecutor(max_workers=8) as executor:
        namespaces = list(
            executor.map(
                lambda _: initialize_producer_namespace(tmp_path, manifest),
                range(32),
            )
        )

    assert namespaces == [tmp_path / manifest.fingerprint] * 32
    verify_producer_namespace(namespaces[0], manifest)
    assert json.loads(
        (namespaces[0] / PRODUCER_MANIFEST_FILE).read_text(encoding="utf-8")
    ) == manifest.to_payload()


@pytest.mark.parametrize("corruption", (b"", b"{}", b"not-json"))
def test_existing_corrupt_manifest_fails_closed(tmp_path: Path, corruption: bytes) -> None:
    manifest = _manifest()
    namespace = initialize_producer_namespace(tmp_path, manifest)
    (namespace / PRODUCER_MANIFEST_FILE).write_bytes(corruption)

    with pytest.raises(ProducerManifestError, match="does not match"):
        initialize_producer_namespace(tmp_path, manifest)


def test_namespace_with_artifacts_but_no_manifest_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest()
    namespace = tmp_path / manifest.fingerprint
    (namespace / "request-1").mkdir(parents=True)

    with pytest.raises(ProducerManifestError, match="no trusted manifest"):
        initialize_producer_namespace(tmp_path, manifest)


def test_wrong_namespace_and_manifest_are_rejected(tmp_path: Path) -> None:
    manifest = _manifest()
    namespace = initialize_producer_namespace(tmp_path, manifest)

    with pytest.raises(ProducerManifestError, match="fingerprint mismatch"):
        verify_producer_namespace(namespace, replace(manifest, backend="inprocess"))


def test_canonical_encoder_rejects_bytes_and_accepts_tuple() -> None:
    assert canonical_json_bytes({"items": (2, 1)}) == b'{"items":[2,1]}'
    with pytest.raises(ValueError, match="non-JSON"):
        canonical_json_bytes({"secret": b"bytes"})
