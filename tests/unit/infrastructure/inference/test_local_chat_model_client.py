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


class FakeSession:
    def __init__(self, response: FakeResponse | None = None, error: requests.RequestException | None = None) -> None:
        self.response = response or FakeResponse({"choices": [{"message": {"content": "1"}}]})
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response

    def close(self) -> None:
        self.closed = True


def test_local_chat_client_reads_openai_compatible_response(monkeypatch) -> None:
    session = FakeSession(
        FakeResponse(
            {
                "choices": [{"message": {"content": "Zip"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }
        )
    )
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma"))

    response = client.generate([{"role": "user", "content": "3:"}], max_tokens=4, temperature=0.0)

    assert response.text == "Zip"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 1
    assert session.calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    payload = session.calls[0]["json"]
    assert payload["model"] == "gemma"  # type: ignore[index]
    assert "top_p" not in payload  # type: ignore[operator]
    assert session.calls[0]["timeout"] == 30.0


def test_local_chat_client_allows_per_call_timeout_override(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma", timeout_s=30.0)
    )

    client.generate([{"role": "user", "content": "1:"}], timeout_s=0.25)

    assert session.calls[0]["timeout"] == 0.25


def test_local_chat_client_includes_top_p_and_extra_payload_when_configured(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url="http://localhost:8000/v1",
            model="gemma",
            top_p=0.8,
            extra_payload={"chat_template_kwargs": {"enable_thinking": False}},
        )
    )

    client.generate([{"role": "user", "content": "1:"}], max_tokens=8, temperature=0.7)

    payload = session.calls[0]["json"]
    assert payload["top_p"] == 0.8  # type: ignore[index]
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}  # type: ignore[index]


def test_local_chat_client_rejects_extra_payload_key_collisions(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url="http://localhost:8000/v1",
            model="gemma",
            extra_payload={"model": "other"},
        )
    )

    with pytest.raises(ValueError, match="extra_payload cannot override"):
        client.generate([{"role": "user", "content": "1:"}])
    assert session.calls == []


def test_local_chat_client_reuses_and_closes_session(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma"))

    client.generate([{"role": "user", "content": "1:"}])
    client.generate([{"role": "user", "content": "2:"}])
    client.close()

    assert len(session.calls) == 2
    assert session.closed is True


def test_local_chat_client_raises_on_malformed_response(monkeypatch) -> None:
    session = FakeSession(FakeResponse({"choices": []}))
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma"))

    with pytest.raises(ValueError, match="Unexpected chat completion response"):
        client.generate([{"role": "user", "content": "3:"}])


def test_local_chat_client_retry_disabled_surfaces_request_error(monkeypatch) -> None:
    session = FakeSession(error=requests.Timeout("slow"))
    monkeypatch.setattr(requests, "Session", lambda: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma", max_retries=0)
    )

    with pytest.raises(requests.Timeout):
        client.generate([{"role": "user", "content": "3:"}])
