"""Local-server chat model client for generic realtime tasks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

from streammuse.domain.tasks import ChatModelResponse


@dataclass(frozen=True)
class LocalChatModelClientConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "local-model"
    api_key: str | None = None
    timeout_s: float = 30.0
    max_retries: int = 0
    retry_delay_s: float = 0.25
    top_p: float | None = None
    extra_payload: dict[str, Any] | None = None


class LocalChatModelClient:
    def __init__(self, config: LocalChatModelClientConfig) -> None:
        self.config = config
        self._session = requests.Session()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 32,
        temperature: float = 0.0,
        timeout_s: float | None = None,
    ) -> ChatModelResponse:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if self.config.top_p is not None:
            payload["top_p"] = float(self.config.top_p)
        if self.config.extra_payload:
            collisions = sorted(set(payload).intersection(self.config.extra_payload))
            if collisions:
                joined = ", ".join(collisions)
                raise ValueError(f"extra_payload cannot override chat completion payload keys: {joined}")
            payload.update(self.config.extra_payload)
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        attempts = max(1, int(self.config.max_retries) + 1)
        last_error: BaseException | None = None
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                response = self._session.post(
                    self._chat_url(),
                    json=payload,
                    headers=headers,
                    timeout=float(self.config.timeout_s if timeout_s is None else timeout_s),
                )
                response.raise_for_status()
                data = response.json()
                latency_ms = (time.perf_counter() - start) * 1000.0
                return self._parse_response(data, latency_ms=latency_ms)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(float(self.config.retry_delay_s))
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("local chat client exhausted attempts without a response")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "LocalChatModelClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _chat_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    @staticmethod
    def _parse_response(data: dict[str, Any], *, latency_ms: float) -> ChatModelResponse:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Unexpected chat completion response: {data}")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError(f"Unexpected chat completion response: {data}")
        message = first.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError(f"Unexpected chat completion response: {data}")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        return ChatModelResponse(
            text=str(message.get("content") or "").strip(),
            latency_ms=latency_ms,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
            raw=data,
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
