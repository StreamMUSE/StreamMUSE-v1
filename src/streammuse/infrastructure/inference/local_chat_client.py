"""Local-server chat model client for generic realtime tasks."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from threading import Event, RLock, Thread
from typing import Any

import httpx
import requests

from streammuse.domain.tasks import ChatModelResponse


_retry_sleep = asyncio.sleep


class LocalChatRequestAborted(RuntimeError):
    """Raised when local chat generation is aborted or permanently stopped."""


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


@dataclass
class _ActiveRequest:
    aborted: bool = False
    cancel_sent: bool = False
    result: Future[ChatModelResponse] = field(default_factory=Future)
    task: asyncio.Task[ChatModelResponse] | None = None


class LocalChatModelClient:
    """Synchronous client with an abortable asynchronous HTTP transport."""

    def __init__(
        self,
        config: LocalChatModelClientConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self._transport = transport
        self._lock = RLock()
        self._active: _ActiveRequest | None = None
        self._active_cleared = Event()
        self._active_cleared.set()
        self._stopped = False
        self._closed = False
        self._close_complete = Event()
        self._close_error: BaseException | None = None
        self._loop = asyncio.new_event_loop()
        self._ready = Event()
        self._client: httpx.AsyncClient | None = None
        self._startup_error: BaseException | None = None
        self._loop_thread = Thread(
            target=self._run_event_loop,
            name="streammuse-local-chat-http",
            daemon=True,
        )
        self._loop_thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError("failed to initialize local chat HTTP client") from self._startup_error

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

        active = _ActiveRequest()
        with self._lock:
            if self._stopped:
                raise LocalChatRequestAborted("local chat client is stopped")
            if self._active is not None:
                raise RuntimeError("local chat client already has an active request")
            self._active = active
            self._active_cleared.clear()
        try:
            self._loop.call_soon_threadsafe(
                self._start_request,
                active,
                payload,
                headers,
                float(self.config.timeout_s if timeout_s is None else timeout_s),
            )
        except BaseException:
            self._finish_registration_failure(active)
            raise
        try:
            return active.result.result()
        except BaseException:
            with self._lock:
                active.aborted = True
            self._cancel(active)
            raise

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._client = httpx.AsyncClient(transport=self._transport)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            self._loop.close()
            return
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    def abort(self) -> None:
        """Abort current work while allowing a later generation request."""
        with self._lock:
            active = self._active
            if active is not None:
                active.aborted = True
        self._cancel(active)

    def stop_accepting_and_abort(self) -> None:
        """Permanently reject new work and abort any registered request."""
        with self._lock:
            self._stopped = True
            active = self._active
            if active is not None:
                active.aborted = True
        self._cancel(active)

    def close(self) -> None:
        self.stop_accepting_and_abort()
        self._active_cleared.wait()
        with self._lock:
            if self._closed:
                wait_for_close = True
            else:
                self._closed = True
                wait_for_close = False
        if wait_for_close:
            self._close_complete.wait()
            if self._close_error is not None:
                raise self._close_error
            return
        future = asyncio.run_coroutine_threadsafe(self._close_async_client(), self._loop)
        try:
            future.result()
        except BaseException as exc:
            self._close_error = exc
            raise
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
            self._close_complete.set()

    def __enter__(self) -> "LocalChatModelClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _cancel(self, active: _ActiveRequest | None) -> None:
        if active is None:
            return
        with self._lock:
            task = active.task
            if task is None or task.done() or active.cancel_sent:
                return
            active.cancel_sent = True
        try:
            self._loop.call_soon_threadsafe(task.cancel)
        except RuntimeError:
            pass

    def _start_request(
        self,
        active: _ActiveRequest,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_s: float,
    ) -> None:
        task = self._loop.create_task(
            self._generate_async(payload=payload, headers=headers, timeout_s=timeout_s)
        )
        with self._lock:
            active.task = task
            should_cancel = active.aborted and not active.cancel_sent
            if should_cancel:
                active.cancel_sent = True
        task.add_done_callback(lambda completed: self._finish_request(active, completed))
        if should_cancel:
            task.cancel()

    def _finish_request(
        self,
        active: _ActiveRequest,
        task: asyncio.Task[ChatModelResponse],
    ) -> None:
        with self._lock:
            if self._active is active:
                self._active = None
                self._active_cleared.set()
        if task.cancelled():
            active.result.set_exception(LocalChatRequestAborted("local chat request aborted"))
            return
        error = task.exception()
        if error is not None:
            active.result.set_exception(error)
            return
        active.result.set_result(task.result())

    def _finish_registration_failure(self, active: _ActiveRequest) -> None:
        with self._lock:
            if self._active is active:
                self._active = None
                self._active_cleared.set()

    async def _close_async_client(self) -> None:
        assert self._client is not None
        await self._client.aclose()

    async def _generate_async(
        self,
        *,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout_s: float,
    ) -> ChatModelResponse:
        attempts = max(1, int(self.config.max_retries) + 1)
        try:
            assert self._client is not None
            for attempt in range(attempts):
                start = time.perf_counter()
                try:
                    response = await self._client.post(
                        self._chat_url(),
                        json=payload,
                        headers=headers,
                        timeout=timeout_s,
                    )
                    response.raise_for_status()
                    data = response.json()
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    return self._parse_response(data, latency_ms=latency_ms)
                except httpx.HTTPError as exc:
                    if attempt < attempts - 1:
                        await _retry_sleep(float(self.config.retry_delay_s))
                        continue
                    if isinstance(exc, httpx.TimeoutException):
                        raise requests.Timeout(str(exc)) from exc
                    raise
        except asyncio.CancelledError as exc:
            raise LocalChatRequestAborted("local chat request aborted") from exc

        raise RuntimeError("local chat client exhausted attempts without a response")

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
