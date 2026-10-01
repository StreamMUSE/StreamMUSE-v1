"""Bounded HTTP adapter for the Mac-local MLX MOSS service (scripts/mlx_moss_server.py).

The service speaks the same speech API subset as SGLang-Omni, so this adapter
reuses ``SglangMossSynthesizer`` for timeouts, cancellation, size limits and WAV
validation. It differs in what it pins: instead of an SGLang build and runtime
config file, the render server pins the MLX runtime identity (mlx and mlx-audio
versions, the mlx-audio commit, quantization, a hash of the converted weights,
and the audio tokenizer revision), and startup refuses a service that reports
anything else.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from streammuse.infrastructure.rap.moss_generation import DEFAULT_BASE_SEED
from streammuse.infrastructure.rap.moss_tts import MossBackendUnavailable
from streammuse.infrastructure.rap.sglang_moss_tts import (
    SglangMossSynthesizer,
    _require_bounded_text,
    _require_pinned_identity,
    _require_sha256,
    _validate_reference_uri,
    _validated_base_url,
)


MLX_MOSS_ADAPTER_REVISION = "streammuse.mlx_moss_http.v1"
MLX_RUNTIME_IDENTITY_FIELDS = (
    "mlx_version",
    "mlx_audio_version",
    "mlx_audio_commit",
    "quantization",
    "weights_sha256",
    "audio_tokenizer_revision",
)


@dataclass(frozen=True)
class MlxMossConfig:
    base_url: str
    model_id: str
    model_revision: str
    reference_audio_uri: str
    reference_audio_sha256: str
    runtime_identity: Mapping[str, str]
    request_timeout_seconds: float = 120.0
    connect_timeout_seconds: float = 5.0
    pool_timeout_seconds: float = 5.0
    cancellation_grace_seconds: float = 2.0
    response_byte_limit: int = 64 * 1024 * 1024
    maximum_audio_seconds: float = 30.0
    base_seed: int = DEFAULT_BASE_SEED

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_url", _validated_base_url(self.base_url))
        _require_bounded_text(self.model_id, "MLX MOSS model", maximum=512)
        _require_bounded_text(self.model_revision, "MLX MOSS model revision", maximum=512)
        _require_pinned_identity(self.model_revision, "MLX MOSS model revision")
        _validate_reference_uri(self.reference_audio_uri)
        _require_sha256(self.reference_audio_sha256, "reference audio")
        identity = validated_runtime_identity(self.runtime_identity)
        object.__setattr__(self, "runtime_identity", MappingProxyType(identity))
        for name in (
            "request_timeout_seconds",
            "connect_timeout_seconds",
            "pool_timeout_seconds",
            "cancellation_grace_seconds",
            "maximum_audio_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"MLX MOSS {name} must be finite and positive")
        if (
            isinstance(self.response_byte_limit, bool)
            or not isinstance(self.response_byte_limit, int)
            or self.response_byte_limit < 1024
        ):
            raise ValueError("MLX MOSS response byte limit is invalid")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int) or self.base_seed < 0:
            raise ValueError("MLX MOSS base seed must be non-negative")

    @property
    def reference_text_sha256(self) -> str:
        # MOSS generation mode clones the voice from audio only.
        return "unavailable"

    @property
    def runtime_config_sha256(self) -> str:
        return runtime_identity_sha256(self.runtime_identity)


def validated_runtime_identity(value: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("MLX MOSS runtime identity must be an object")
    missing = [name for name in MLX_RUNTIME_IDENTITY_FIELDS if name not in value]
    if missing:
        raise ValueError(f"MLX MOSS runtime identity is missing {', '.join(missing)}")
    identity: dict[str, str] = {}
    for name in MLX_RUNTIME_IDENTITY_FIELDS:
        field_value = value[name]
        if not isinstance(field_value, str):
            raise ValueError(f"MLX MOSS runtime identity {name} must be a string")
        _require_bounded_text(field_value, f"MLX MOSS runtime identity {name}", maximum=256)
        _require_pinned_identity(field_value, f"MLX MOSS runtime identity {name}")
        identity[name] = field_value
    _require_sha256(identity["weights_sha256"], "MLX weights")
    return identity


def runtime_identity_sha256(identity: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(identity), ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()


class MlxMossSynthesizer(SglangMossSynthesizer):
    """MOSS synthesis through the loopback MLX service."""

    _BACKEND = "mlx"
    _LABEL = "MLX MOSS"

    @classmethod
    def _config_type(cls) -> type:
        return MlxMossConfig

    def _extra_payload(self) -> dict[str, object]:
        return {}

    def _validate_model_entry(self, model: Mapping[str, object]) -> None:
        reported = model.get("streammuse_runtime")
        if not isinstance(reported, Mapping):
            raise MossBackendUnavailable("MLX MOSS service does not report its runtime identity")
        for name in MLX_RUNTIME_IDENTITY_FIELDS:
            if reported.get(name) != self._config.runtime_identity[name]:
                raise MossBackendUnavailable(
                    f"MLX MOSS runtime {name} does not match the configured pin"
                )
