"""Small JSON normalization helpers shared by optional infrastructure."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a defensive value accepted by JSON with non-finite floats disabled."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, bytes):
        return value.hex()
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if type(value).__module__.startswith("numpy"):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return json_safe(tolist())
        item = getattr(value, "item", None)
        if callable(item):
            return json_safe(item())
    return str(value)
