from __future__ import annotations

from typing import Optional

import torch


def is_mps_available() -> bool:
    """Return True when MPS backend is available and built."""
    if not hasattr(torch.backends, "mps"):
        return False
    return bool(torch.backends.mps.is_available()) and bool(torch.backends.mps.is_built())


def _normalize_preference(preference: str) -> str:
    value = (preference or "auto").strip().lower()
    if value not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device preference: {preference}")
    return value


def resolve_device(preference: str = "auto") -> str:
    """Resolve runtime device based on preference and backend availability."""
    pref = _normalize_preference(preference)
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if is_mps_available():
            return "mps"
        return "cpu"

    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine")
    if pref == "mps" and not is_mps_available():
        raise RuntimeError("MPS is not available on this machine")
    return pref


def resolve_dtype(device: str, preference: str = "auto") -> torch.dtype:
    """Resolve dtype based on device and preference."""
    pref = (preference or "auto").strip().lower()
    if pref not in {"auto", "float32", "float16"}:
        raise ValueError(f"Unsupported dtype preference: {preference}")

    if pref == "float32":
        return torch.float32
    if pref == "float16":
        if device == "cpu":
            raise RuntimeError("float16 is not supported for CPU inference in this runtime policy")
        return torch.float16

    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)


def parse_env_bool(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
