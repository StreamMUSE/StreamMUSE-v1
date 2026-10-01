"""Canonical producer identity and isolated RAP artifact namespaces."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias


PRODUCER_MANIFEST_SCHEMA_VERSION = "streammuse.rap_producer.v1"
PRODUCER_MANIFEST_FILE = "_producer_manifest.v1.json"
PRODUCER_FINGERPRINT_FILE = "_producer_fingerprint.sha256"
_LOCK_FILE = ".producer_manifest.lock"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BACKENDS = {"inprocess", "sglang-omni", "mlx"}

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ProducerManifestError(RuntimeError):
    """Raised when a producer namespace cannot be trusted."""


@dataclass(frozen=True)
class ProducerManifestV1:
    """All output-affecting producer inputs, excluding operational placement."""

    backend: str
    backend_implementation_revision: str
    streammuse_revision: Mapping[str, JsonValue]
    model: Mapping[str, JsonValue]
    runtime: Mapping[str, JsonValue]
    generation: Mapping[str, JsonValue]
    reference: Mapping[str, JsonValue]
    alignment: Mapping[str, JsonValue]
    output: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError("producer backend is unsupported")
        _require_identity(
            self.backend_implementation_revision,
            "backend implementation revision",
        )
        for field_name in (
            "streammuse_revision",
            "model",
            "runtime",
            "generation",
            "reference",
            "alignment",
            "output",
        ):
            value = _canonicalize(getattr(self, field_name), path=field_name)
            if not isinstance(value, dict) or not value:
                raise ValueError(f"producer {field_name} must be a non-empty object")
            object.__setattr__(self, field_name, _freeze(value))
        _validate_required_contract(self.to_payload())

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": PRODUCER_MANIFEST_SCHEMA_VERSION,
            "backend": self.backend,
            "backend_implementation_revision": self.backend_implementation_revision,
            "streammuse_revision": _thaw(self.streammuse_revision),
            "model": _thaw(self.model),
            "runtime": _thaw(self.runtime),
            "generation": _thaw(self.generation),
            "reference": _thaw(self.reference),
            "alignment": _thaw(self.alignment),
            "output": _thaw(self.output),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_payload())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def canonical_json_bytes(value: object) -> bytes:
    canonical = _canonicalize(value, path="manifest")
    return json.dumps(
        canonical,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def initialize_producer_namespace(
    artifact_root: str | Path,
    manifest: ProducerManifestV1,
) -> Path:
    """Create or verify one immutable producer namespace and return its root."""
    if not isinstance(manifest, ProducerManifestV1):
        raise ValueError("producer namespace requires ProducerManifestV1")
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    namespace = root / manifest.fingerprint
    if namespace.exists() and not namespace.is_dir():
        raise ProducerManifestError("producer namespace path is not a directory")
    namespace.mkdir(mode=0o750, exist_ok=True)

    lock_path = namespace / _LOCK_FILE
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            _initialize_locked(namespace, manifest)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return namespace


def verify_producer_namespace(
    namespace: str | Path,
    manifest: ProducerManifestV1,
) -> None:
    path = Path(namespace)
    expected_name = manifest.fingerprint
    if path.name != expected_name:
        raise ProducerManifestError("producer namespace fingerprint mismatch")
    _verify_file(path / PRODUCER_MANIFEST_FILE, manifest.canonical_bytes())
    _verify_file(
        path / PRODUCER_FINGERPRINT_FILE,
        f"{expected_name}\n".encode("ascii"),
    )


def _initialize_locked(namespace: Path, manifest: ProducerManifestV1) -> None:
    manifest_path = namespace / PRODUCER_MANIFEST_FILE
    fingerprint_path = namespace / PRODUCER_FINGERPRINT_FILE
    expected_manifest = manifest.canonical_bytes()
    expected_fingerprint = f"{manifest.fingerprint}\n".encode("ascii")
    existing_entries = {
        path.name
        for path in namespace.iterdir()
        if path.name != _LOCK_FILE and not path.name.startswith(".producer-init-")
    }

    if manifest_path.exists() or fingerprint_path.exists():
        _verify_file(manifest_path, expected_manifest)
        _verify_file(fingerprint_path, expected_fingerprint)
        return
    if existing_entries:
        raise ProducerManifestError(
            "producer namespace has artifacts but no trusted manifest"
        )

    _publish_new_file(manifest_path, expected_manifest)
    try:
        _publish_new_file(fingerprint_path, expected_fingerprint)
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        _fsync_directory(namespace)
        raise
    verify_producer_namespace(namespace, manifest)


def _publish_new_file(path: Path, data: bytes) -> None:
    temporary = path.parent / f".producer-init-{uuid.uuid4().hex}"
    try:
        with temporary.open("xb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _verify_file(path, data)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_file(path: Path, expected: bytes) -> None:
    try:
        actual = path.read_bytes()
    except OSError as exc:
        raise ProducerManifestError("producer namespace manifest is missing") from exc
    if actual != expected:
        raise ProducerManifestError("producer namespace manifest does not match")


def _validate_required_contract(payload: Mapping[str, JsonValue]) -> None:
    streammuse = _require_mapping(
        payload["streammuse_revision"], "streammuse revision"
    )
    _require_identity(streammuse.get("commit"), "StreamMUSE commit")
    _require_sha256(streammuse.get("patch_sha256"), "StreamMUSE patch")
    model = _require_mapping(payload["model"], "model")
    _require_identity(model.get("id"), "model id")
    _require_identity(model.get("revision"), "model revision")
    runtime = _require_mapping(payload["runtime"], "runtime")
    for name in ("identity", "version", "revision"):
        _require_identity(runtime.get(name), f"runtime {name}")
    _require_sha256(runtime.get("config_sha256"), "runtime config")
    generation = _require_mapping(payload["generation"], "generation")
    _require_identity(generation.get("seed_policy_version"), "seed policy version")
    settings = generation.get("settings")
    if not isinstance(settings, Mapping) or not settings:
        raise ValueError("producer generation settings must be a non-empty object")
    reference = _require_mapping(payload["reference"], "reference")
    _require_sha256(reference.get("audio_sha256"), "reference audio")
    text_hash = reference.get("text_sha256")
    if text_hash is not None:
        _require_sha256(text_hash, "reference text")
    alignment = _require_mapping(payload["alignment"], "alignment")
    for name in ("identity", "version", "warp_policy"):
        _require_identity(alignment.get(name), f"alignment {name}")
    output = _require_mapping(payload["output"], "output")
    rate = output.get("sample_rate_hz")
    if isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0:
        raise ValueError("producer output sample rate must be positive")
    for name in ("wire_audio_codec", "public_schema_version"):
        _require_identity(output.get(name), f"output {name}")


def _require_mapping(value: object, name: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError(f"producer {name} must be an object")
    return value


def _require_identity(value: object, name: str) -> None:
    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"producer {name} must not be empty")
        return
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(f"producer {name} is invalid")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"producer {name} SHA-256 is invalid")


def _canonicalize(value: object, *, path: str) -> JsonValue:
    if value is None or type(value) in {bool, int, str}:
        return value  # type: ignore[return-value]
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"producer {path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"producer {path} contains an invalid key")
            result[key] = _canonicalize(item, path=f"{path}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _canonicalize(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ValueError(f"producer {path} contains a non-JSON value")


def _freeze(value: JsonValue) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value  # type: ignore[return-value]
    raise ValueError("producer manifest contains an invalid frozen value")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
