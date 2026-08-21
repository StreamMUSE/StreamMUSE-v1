from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import httpx
import pytest
import requests

import streammuse.infrastructure.inference.local_chat_client as local_chat_module
from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatChoicesResponse,
    LocalChatModelClient,
    LocalChatModelClientConfig,
)


class FakeResponse:
    def __init__(
        self, payload: dict[str, object], *, status_error: Exception | None = None
    ) -> None:
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self) -> None:
        if self.status_error is not None:
            raise self.status_error

    def json(self) -> dict[str, object]:
        return self.payload


class FakeSession:
    def __init__(
        self, response: FakeResponse | None = None, error: BaseException | None = None
    ) -> None:
        self.response = response or FakeResponse(
            {"choices": [{"message": {"content": "1"}}]}
        )
        self.error = error
        self.calls: list[dict[str, object]] = []
        self.closed = False

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if self.error is not None:
            raise self.error
        return self.response

    async def __aenter__(self) -> "FakeSession":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.close()

    async def aclose(self) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class ControlledChatServer:
    def __init__(self) -> None:
        self.request_started = Event()
        self.release = Event()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                owner.request_started.set()
                owner.release.wait()
                body = b'{"choices":[{"message":{"content":"one"}}]}'
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass

            def log_message(self, _format: str, *_args: object) -> None:
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "ControlledChatServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release.set()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=0.5)


def test_local_chat_client_reads_openai_compatible_response(monkeypatch) -> None:
    session = FakeSession(
        FakeResponse(
            {
                "choices": [{"message": {"content": "Zip"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            }
        )
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma")
    )

    response = client.generate(
        [{"role": "user", "content": "3:"}], max_tokens=4, temperature=0.0
    )

    assert response.text == "Zip"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 1
    assert session.calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    payload = session.calls[0]["json"]
    assert payload["model"] == "gemma"  # type: ignore[index]
    assert "top_p" not in payload  # type: ignore[operator]
    assert session.calls[0]["timeout"] == 30.0
    client.close()


def test_local_chat_client_allows_per_call_timeout_override(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url="http://localhost:8000/v1", model="gemma", timeout_s=30.0
        )
    )

    client.generate([{"role": "user", "content": "1:"}], timeout_s=0.25)

    assert session.calls[0]["timeout"] == 0.25
    client.close()


def test_local_chat_client_blank_timeout_keeps_safe_target_and_exception_diagnostics(
    monkeypatch,
) -> None:
    request = httpx.Request("POST", "http://127.0.0.1:18001/v1/chat/completions")
    session = FakeSession(error=httpx.ReadTimeout("", request=request))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url="http://user:password@127.0.0.1:18001/v1?access_token=do-not-record",
            model="qwen-rap",
            api_key="do-not-record",
        )
    )

    with pytest.raises(requests.Timeout) as error:
        client.generate([{"role": "user", "content": "do-not-record"}])

    message = str(error.value)
    assert "target_url=http://127.0.0.1:18001/v1/chat/completions" in message
    assert "exception_class=ReadTimeout" in message
    assert "exception_repr=ReadTimeout('')" in message
    for secret in ("user", "password", "access_token", "do-not-record"):
        assert secret not in message
    client.close()


def test_local_chat_client_includes_top_p_and_extra_payload_when_configured(
    monkeypatch,
) -> None:
    session = FakeSession()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
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
    client.close()


def test_local_chat_client_rejects_extra_payload_key_collisions(monkeypatch) -> None:
    session = FakeSession()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
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
    client.close()


def test_local_chat_client_reuses_and_closes_session(monkeypatch) -> None:
    session = FakeSession()
    constructed: list[FakeSession] = []

    def build_session(**_kwargs) -> FakeSession:
        constructed.append(session)
        return session

    monkeypatch.setattr(httpx, "AsyncClient", build_session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma")
    )

    client.generate([{"role": "user", "content": "1:"}])
    client.generate([{"role": "user", "content": "2:"}])
    client.close()

    assert constructed == [session]
    assert len(session.calls) == 2
    assert session.closed is True


def test_local_chat_client_raises_on_malformed_response(monkeypatch) -> None:
    session = FakeSession(FakeResponse({"choices": []}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(base_url="http://localhost:8000/v1", model="gemma")
    )

    with pytest.raises(ValueError, match="Unexpected chat completion response"):
        client.generate([{"role": "user", "content": "3:"}])
    client.close()


def test_local_chat_client_retry_disabled_surfaces_request_error(monkeypatch) -> None:
    session = FakeSession(error=requests.Timeout("slow"))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(
            base_url="http://localhost:8000/v1", model="gemma", max_retries=0
        )
    )

    with pytest.raises(requests.Timeout):
        client.generate([{"role": "user", "content": "3:"}])
    client.close()


def test_abort_interrupts_an_inflight_http_request_without_waiting_for_timeout() -> (
    None
):
    with ControlledChatServer() as server:
        client = LocalChatModelClient(
            LocalChatModelClientConfig(
                base_url=server.base_url, model="gemma", timeout_s=60.0
            )
        )
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                client.generate([{"role": "user", "content": "3:"}])
            except BaseException as exc:
                errors.append(exc)

        worker = Thread(target=generate, daemon=True)
        worker.start()
        assert server.request_started.wait(timeout=0.5)

        client.abort()
        worker.join(timeout=0.5)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
        assert "aborted" in str(errors[0]).lower()
        client.close()


def test_abort_interrupts_retry_delay_after_retry_sleep_has_started(
    monkeypatch,
) -> None:
    sleep_started = Event()

    async def retry_sleep(_delay: float) -> None:
        sleep_started.set()
        await asyncio.Event().wait()

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    monkeypatch.setattr(local_chat_module, "_retry_sleep", retry_sleep, raising=False)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="gemma", max_retries=2, retry_delay_s=60.0),
        transport=httpx.MockTransport(handler),
    )
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            client.generate([{"role": "user", "content": "3:"}])
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=generate, daemon=True)
    worker.start()
    assert sleep_started.wait(timeout=0.5)

    client.abort()
    worker.join(timeout=0.5)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "aborted" in str(errors[0]).lower()
    client.close()


def test_abort_waits_for_loop_side_request_cleanup() -> None:
    request_started = Event()
    cleanup_finished = Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.05)
            cleanup_finished.set()
        raise AssertionError("unreachable")

    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="gemma", timeout_s=60.0),
        transport=httpx.MockTransport(handler),
    )
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            client.generate([{"role": "user", "content": "3:"}])
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=generate, daemon=True)
    worker.start()
    assert request_started.wait(timeout=0.5)

    client.abort()
    worker.join(timeout=0.5)

    assert cleanup_finished.is_set()
    assert not worker.is_alive()
    assert len(errors) == 1
    assert "aborted" in str(errors[0]).lower()
    client.close()


def test_close_does_not_cancel_request_cleanup_a_second_time() -> None:
    request_started = Event()
    cleanup_started = Event()
    cleanup_finished = Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await asyncio.sleep(0.05)
            cleanup_finished.set()
        raise AssertionError("unreachable")

    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="gemma", timeout_s=60.0),
        transport=httpx.MockTransport(handler),
    )
    errors: list[BaseException] = []

    def generate() -> None:
        try:
            client.generate([{"role": "user", "content": "3:"}])
        except BaseException as exc:
            errors.append(exc)

    worker = Thread(target=generate, daemon=True)
    worker.start()
    assert request_started.wait(timeout=0.5)

    client.abort()
    assert cleanup_started.wait(timeout=0.5)
    closer = Thread(target=client.close, daemon=True)
    closer.start()
    closer.join(timeout=0.5)
    worker.join(timeout=0.5)

    assert cleanup_finished.is_set()
    assert not closer.is_alive()
    assert not worker.is_alive()
    assert len(errors) == 1
    assert "aborted" in str(errors[0]).lower()


def test_keyboard_interrupt_cancels_request_before_generate_returns(
    monkeypatch,
) -> None:
    request_started = Event()
    cleanup_finished = Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        request_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_finished.set()
        raise AssertionError("unreachable")

    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="gemma", timeout_s=60.0),
        transport=httpx.MockTransport(handler),
    )
    original_result = local_chat_module.Future.result
    interrupted = Event()

    def interrupt_result(self, timeout=None):
        if not interrupted.is_set():
            assert request_started.wait(timeout=0.5)
            interrupted.set()
            raise KeyboardInterrupt
        return original_result(self, timeout=timeout)

    monkeypatch.setattr(local_chat_module.Future, "result", interrupt_result)
    with pytest.raises(KeyboardInterrupt):
        client.generate([{"role": "user", "content": "3:"}])
    monkeypatch.setattr(local_chat_module.Future, "result", original_result)

    assert cleanup_finished.wait(timeout=0.5)
    client.close()


def test_concurrent_close_callers_wait_for_transport_shutdown(monkeypatch) -> None:
    close_started = Event()
    release_close = Event()

    class BlockingCloseSession(FakeSession):
        async def aclose(self) -> None:
            close_started.set()
            while not release_close.is_set():
                await asyncio.sleep(0.01)
            self.close()

    session = BlockingCloseSession()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(model="gemma"))
    returned = [Event(), Event()]

    def close(index: int) -> None:
        client.close()
        returned[index].set()

    first = Thread(target=close, args=(0,), daemon=True)
    second = Thread(target=close, args=(1,), daemon=True)
    first.start()
    assert close_started.wait(timeout=0.5)
    second.start()

    assert not returned[1].wait(timeout=0.05)
    release_close.set()
    first.join(timeout=0.5)
    second.join(timeout=0.5)

    assert returned[0].is_set()
    assert returned[1].is_set()
    assert session.closed is True


def test_exhausted_httpx_timeout_preserves_requests_timeout_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="gemma", max_retries=1, retry_delay_s=0.0),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(requests.Timeout, match="slow"):
        client.generate([{"role": "user", "content": "3:"}])
    client.close()


def test_generate_choices_sends_independent_n_and_returns_all_choices(
    monkeypatch,
) -> None:
    session = FakeSession(
        FakeResponse(
            {
                "choices": [
                    {"index": 0, "message": {"content": "first line"}},
                    {"index": 1, "message": {"content": "second line"}},
                ],
                "usage": {"prompt_tokens": 19, "completion_tokens": 8},
            }
        )
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="qwen-rap", top_p=0.95)
    )

    response = client.generate_choices(
        [{"role": "user", "content": "one line"}],
        n=2,
        max_tokens=24,
        temperature=1.0,
        timeout_s=2.5,
    )

    assert isinstance(response, LocalChatChoicesResponse)
    assert tuple(choice.text for choice in response.choices) == (
        "first line",
        "second line",
    )
    assert tuple(choice.index for choice in response.choices) == (0, 1)
    assert response.prompt_tokens == 19
    assert response.completion_tokens == 8
    assert response.malformed_choice_indices == ()
    assert response.warnings == ()
    payload = session.calls[0]["json"]
    assert payload == {
        "model": "qwen-rap",
        "messages": [{"role": "user", "content": "one line"}],
        "temperature": 1.0,
        "max_tokens": 24,
        "n": 2,
        "top_p": 0.95,
    }
    assert session.calls[0]["timeout"] == 2.5
    client.close()


def test_generate_choices_n_one_is_compatible_without_changing_generate_payload(
    monkeypatch,
) -> None:
    session = FakeSession(
        FakeResponse({"choices": [{"index": 0, "message": {"content": "one"}}]})
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(model="qwen-rap"))

    choices = client.generate_choices(
        [{"role": "user", "content": "a"}], n=1, max_tokens=9, temperature=0.3
    )
    legacy = client.generate(
        [{"role": "user", "content": "b"}], max_tokens=9, temperature=0.3
    )

    assert tuple(item.text for item in choices.choices) == ("one",)
    assert legacy.text == "one"
    assert session.calls[0]["json"]["n"] == 1  # type: ignore[index]
    assert "n" not in session.calls[1]["json"]  # type: ignore[operator]
    client.close()


@pytest.mark.parametrize("n", (0, -1, 1.5, True, "2"))
def test_generate_choices_rejects_invalid_n_without_transport(
    monkeypatch, n: object
) -> None:
    session = FakeSession()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(model="qwen-rap"))

    with pytest.raises(ValueError, match="positive integer"):
        client.generate_choices([], n=n, max_tokens=8, temperature=1.0)  # type: ignore[arg-type]

    assert session.calls == []
    client.close()


def test_generate_choices_retains_valid_partial_response_with_bounded_diagnostics(
    monkeypatch,
) -> None:
    malformed = [{"index": index, "message": {}} for index in range(40)]
    session = FakeSession(
        FakeResponse(
            {
                "choices": [
                    {"index": 8, "message": {"content": "  usable one  "}},
                    *malformed,
                    {"index": 49, "message": {"content": "usable two"}},
                ]
            }
        )
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(model="qwen-rap"))

    response = client.generate_choices([], n=42, max_tokens=12, temperature=1.0)

    assert tuple(item.text for item in response.choices) == ("usable one", "usable two")
    assert len(response.malformed_choice_indices) == 32
    assert response.malformed_choice_indices[:2] == (0, 1)
    assert len(response.warnings) <= 33
    assert any("8 additional malformed" in warning for warning in response.warnings)
    client.close()


@pytest.mark.parametrize(
    "payload",
    (
        {"choices": []},
        {"choices": [{"index": 0, "message": {}}]},
        {"choices": [{"index": 0, "message": {"content": "  "}}]},
        {"choices": "not-a-list"},
    ),
)
def test_generate_choices_rejects_wholly_malformed_or_empty_responses(
    monkeypatch, payload
) -> None:
    session = FakeSession(FakeResponse(payload))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: session)
    client = LocalChatModelClient(LocalChatModelClientConfig(model="qwen-rap"))

    with pytest.raises(ValueError, match="Unexpected chat completion response"):
        client.generate_choices([], n=1, max_tokens=12, temperature=1.0)

    client.close()


def test_generate_choices_retries_with_the_same_payload() -> None:
    attempts: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(dict(__import__("json").loads(request.content)))
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(
            200, json={"choices": [{"index": 0, "message": {"content": "ready"}}]}
        )

    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="qwen-rap", max_retries=1, retry_delay_s=0.0),
        transport=httpx.MockTransport(handler),
    )

    response = client.generate_choices([], n=1, max_tokens=12, temperature=1.0)

    assert response.choices[0].text == "ready"
    assert attempts[0] == attempts[1]
    client.close()


def test_generate_choices_preserves_timeout_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("choice timeout", request=request)

    client = LocalChatModelClient(
        LocalChatModelClientConfig(model="qwen-rap"),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(requests.Timeout, match="choice timeout"):
        client.generate_choices([], n=2, max_tokens=12, temperature=1.0)

    client.close()


def test_generate_choices_abort_interrupts_inflight_request() -> None:
    with ControlledChatServer() as server:
        client = LocalChatModelClient(
            LocalChatModelClientConfig(
                base_url=server.base_url, model="qwen-rap", timeout_s=60.0
            )
        )
        errors: list[BaseException] = []

        def generate() -> None:
            try:
                client.generate_choices([], n=4, max_tokens=12, temperature=1.0)
            except BaseException as exc:
                errors.append(exc)

        worker = Thread(target=generate, daemon=True)
        worker.start()
        assert server.request_started.wait(timeout=0.5)

        client.abort()
        worker.join(timeout=0.5)

        assert not worker.is_alive()
        assert len(errors) == 1
        assert "aborted" in str(errors[0]).lower()
        client.close()
