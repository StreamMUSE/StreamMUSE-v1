from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Sequence


MODEL_OUTPUT_TRACE_ENV = "LEKAI_MODEL_OUTPUT_TRACE_PATH"
MODEL_INPUT_TRACE_ENV = "LEKAI_MODEL_INPUT_TRACE_PATH"


def token_digest(tokens: Sequence[int]) -> str:
    payload = json.dumps([int(token) for token in tokens], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelOutputTracer:
    """Optional JSONL trace written immediately after one model beat is generated."""

    def __init__(self, mode: str, path: str | os.PathLike[str] | None = None) -> None:
        configured_path = path if path is not None else os.environ.get(MODEL_OUTPUT_TRACE_ENV)
        self._path = Path(configured_path).expanduser() if configured_path else None
        self._mode = str(mode)
        self._sequence_index = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def record(
        self,
        *,
        output_tokens: Sequence[int],
        target_beat: int | None,
        input_file: str,
        prompt_tokens: Sequence[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._path is None:
            return

        normalized_output = [int(token) for token in output_tokens]
        record: dict[str, Any] = {
            "schema_version": 1,
            "mode": self._mode,
            "sequence_index": self._sequence_index,
            "input_file": str(input_file),
            "target_beat": int(target_beat) if target_beat is not None else None,
            "output_tokens": normalized_output,
            "output_token_count": len(normalized_output),
            "output_digest": token_digest(normalized_output),
            "metadata": dict(metadata or {}),
        }
        if prompt_tokens is not None:
            normalized_prompt = [int(token) for token in prompt_tokens]
            record["prompt_token_count"] = len(normalized_prompt)
            record["prompt_digest"] = token_digest(normalized_prompt)

        line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._sequence_index += 1


class ModelInputTracer:
    """Optional JSONL trace written immediately before a model generation call."""

    def __init__(self, mode: str, path: str | os.PathLike[str] | None = None) -> None:
        configured_path = path if path is not None else os.environ.get(MODEL_INPUT_TRACE_ENV)
        self._path = Path(configured_path).expanduser() if configured_path else None
        self._mode = str(mode)
        self._sequence_index = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def record(
        self,
        *,
        input_tokens: Sequence[int],
        target_beat: int | None,
        input_file: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._path is None:
            return

        normalized_input = [int(token) for token in input_tokens]
        record: dict[str, Any] = {
            "schema_version": 1,
            "record_type": "model_input",
            "mode": self._mode,
            "sequence_index": self._sequence_index,
            "input_file": str(input_file),
            "target_beat": int(target_beat) if target_beat is not None else None,
            "input_tokens": normalized_input,
            "input_token_count": len(normalized_input),
            "input_digest": token_digest(normalized_input),
            "input_shape": [1, len(normalized_input)],
            "input_dtype": "torch.int64",
            "metadata": dict(metadata or {}),
        }

        line = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self._sequence_index += 1
