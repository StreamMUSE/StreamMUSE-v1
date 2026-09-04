"""HTTP client for Lekai prompt-continuation realtime control endpoints."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import requests

from streammuse.domain.musical import MusicalEvent
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import (
    event_representation_summary,
)
from streammuse.infrastructure.inference.serialization import event_from_dict, event_to_dict


@dataclass(frozen=True)
class PromptContinuationHttpClientConfig:
    base_url: str
    timeout_s: float = 30.0
    model_name: str = "lekai_prompt_continuation"
    inference_mode: str = "sliding_window"
    checkpoint_path: Optional[str] = None


class PromptContinuationHttpClient:
    """Client for `/prompt_continuation/*` endpoints.

    This is for the real frontend/backend interaction path. It is deliberately
    not named fake-offline and does not implement the generic InferenceEngine
    protocol because prompt-continuation uses start/append/poll/playable control
    flow instead of one request per generation window.
    """

    def __init__(self, config: PromptContinuationHttpClientConfig) -> None:
        self._config = config
        self._base_url = normalize_prompt_continuation_base_url(config.base_url)

    @property
    def base_url(self) -> str:
        return self._base_url

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if method == "GET":
            response = requests.get(self._url(path), timeout=float(self._config.timeout_s))
        elif method == "POST":
            response = requests.post(self._url(path), json=payload, timeout=float(self._config.timeout_s))
        else:
            raise ValueError(f"unsupported method: {method}")
        response.raise_for_status()
        return response.json()

    def clear_history(self) -> dict[str, Any]:
        return self._request_json("POST", "/clear_history", payload={})

    def start(
        self,
        *,
        melody_events: list[MusicalEvent],
        prompt_length_ticks: int,
        generation_interval_ticks: int,
        observed_until_tick: int,
        bpm: Optional[int] = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "melody_notes": [event_to_dict(event) for event in melody_events],
            "prompt_length_ticks": int(prompt_length_ticks),
            "generation_interval_ticks": int(generation_interval_ticks),
            "observed_until_tick": int(observed_until_tick),
            "inference_mode": self._config.inference_mode,
            "model_name": self._config.model_name,
        }
        if self._config.checkpoint_path:
            payload["checkpoint_path"] = self._config.checkpoint_path
        if bpm is not None:
            if int(bpm) <= 0:
                raise ValueError("bpm must be > 0")
            payload["bpm"] = int(bpm)
        return self._request_json("POST", "/prompt_continuation/start", payload=payload)

    def append_melody(
        self,
        *,
        melody_events: list[MusicalEvent],
        observed_until_tick: int,
    ) -> dict[str, Any]:
        payload = {
            "melody_notes": [event_to_dict(event) for event in melody_events],
            "observed_until_tick": int(observed_until_tick),
        }
        return self._request_json("POST", "/prompt_continuation/append_melody", payload=payload)

    def status(self) -> dict[str, Any]:
        return self._request_json("GET", "/prompt_continuation/status")

    def playable(self) -> tuple[list[MusicalEvent], dict[str, Any]]:
        data = self._request_json("GET", "/prompt_continuation/playable")
        accompaniment = [event_from_dict(event) for event in data.get("accompaniment", [])]
        status = dict(data.get("status", {}))
        self._attach_representation_loop_status(
            status,
            server_representation=data.get("representation", {}),
            events=accompaniment,
        )
        return accompaniment, status

    def raw_history(self) -> tuple[list[MusicalEvent], dict[str, Any]]:
        data = self._request_json("GET", "/prompt_continuation/raw_history")
        accompaniment = [event_from_dict(event) for event in data.get("accompaniment", [])]
        status = dict(data.get("status", {}))
        self._attach_representation_loop_status(
            status,
            server_representation=data.get("representation", {}),
            events=accompaniment,
        )
        return accompaniment, status

    def prompt_history(self) -> tuple[list[MusicalEvent], dict[str, Any]]:
        data = self._request_json("GET", "/prompt_continuation/prompt_history")
        accompaniment = [event_from_dict(event) for event in data.get("accompaniment", [])]
        status = dict(data.get("status", {}))
        self._attach_representation_loop_status(
            status,
            server_representation=data.get("representation", {}),
            events=accompaniment,
        )
        return accompaniment, status

    def _attach_representation_loop_status(
        self,
        status: dict[str, Any],
        *,
        server_representation: Any,
        events: list[MusicalEvent],
    ) -> None:
        client_representation = event_representation_summary(
            [event_to_dict(event) for event in events],
            include_keys=_env_truthy("LEKAI_PROMPT_CONTINUATION_REPRESENTATION_TRACE_KEYS"),
        )
        server_representation = dict(server_representation or {})
        match = (
            bool(server_representation)
            and server_representation.get("digest") == client_representation.get("digest")
            and server_representation.get("event_count") == client_representation.get("event_count")
        )
        status["server_playable_representation"] = server_representation
        status["client_playable_representation"] = client_representation
        status["playable_representation_match"] = bool(match)
        if _env_truthy("LEKAI_PROMPT_CONTINUATION_STRICT_REPRESENTATION_LOOP") and not match:
            raise RuntimeError(
                "prompt-continuation representation mismatch: "
                f"server={server_representation} client={client_representation}"
            )

    def wait_until_terminal(
        self,
        *,
        poll_interval_s: float = 0.1,
        max_wait_s: float = 60.0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        deadline = time.monotonic() + float(max_wait_s)
        polls: list[dict[str, Any]] = []
        while True:
            status = self.status()
            polls.append(status)
            if status.get("is_playback_ready") or status.get("is_failed") or not status.get("is_running"):
                return status, polls
            if time.monotonic() >= deadline:
                raise TimeoutError(f"prompt-continuation did not finish within {max_wait_s}s; last_status={status}")
            time.sleep(float(poll_interval_s))


def normalize_prompt_continuation_base_url(server_url: str) -> str:
    """Return a server base URL from either a base URL or a known endpoint URL."""
    raw = str(server_url).rstrip("/")
    parsed = urlsplit(raw)
    known_suffixes = (
        "/generate_accompaniment",
        "/prompt_continuation/start",
        "/prompt_continuation/append_melody",
        "/prompt_continuation/status",
        "/prompt_continuation/playable",
        "/prompt_continuation/raw_history",
        "/prompt_continuation/prompt_history",
    )
    path = parsed.path.rstrip("/")
    for suffix in known_suffixes:
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parsed.scheme, parsed.netloc, path.rstrip("/"), "", ""))


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
