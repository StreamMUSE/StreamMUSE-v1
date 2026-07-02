from __future__ import annotations

import pytest
import requests

from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatModelClient,
    LocalChatModelClientConfig,
)


class FakeResponse:
    def __init__(self, payload: dict[str, object], *, status_error: Exception | None = None) -> None:
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def json(self) -> dict[str, object]:
        return self.payload


def test_local_chat_client_reads_openai_compatible_response(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse(
            {
                "choices": [{"message": {"content": "Zip"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }
        )

    monkeypatch.setattr(requests, "post", fake_post)
    client = LocalChatModelClient(LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma"))

    response = client.generate([{"role": "user", "content": "3:"}], max_tokens=4, temperature=0.0)

    assert response.text == "Zip"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 1
    assert calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    assert calls[0]["json"]["model"] == "gemma"  # type: ignore[index]


def test_local_chat_client_raises_on_malformed_response(monkeypatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *_args, **_kwargs: FakeResponse({"choices": []}))
    client = LocalChatModelClient(LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma"))

    with pytest.raises(ValueError, match="Unexpected chat completion response"):
        client.generate([{"role": "user", "content": "3:"}])


def test_local_chat_client_retry_disabled_surfaces_request_error(monkeypatch) -> None:
    def fake_post(*_args: object, **_kwargs: object) -> FakeResponse:
        raise requests.Timeout("slow")

    monkeypatch.setattr(requests, "post", fake_post)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma", max_retries=0)
    )

    with pytest.raises(requests.Timeout):
        client.generate([{"role": "user", "content": "3:"}])
