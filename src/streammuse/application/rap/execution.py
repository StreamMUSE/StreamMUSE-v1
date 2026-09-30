"""Deadline and cancellation primitives shared by the RAP render pipeline."""

from __future__ import annotations

import math
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from typing import Iterator


_CORRELATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CANCELLATION_OUTCOMES = {
    "not_requested",
    "cancel_requested",
    "cancel_requested_but_not_interruptible",
    "transport_closed_abort_unconfirmed",
    "upstream_abort_confirmed",
}


class ExecutionStopped(RuntimeError):
    """Base class for an explicitly stopped render execution."""


class ExecutionDeadlineExceeded(ExecutionStopped):
    """Raised when an absolute monotonic execution deadline has elapsed."""


class ExecutionCancelled(ExecutionStopped):
    """Raised after cancellation has been requested for an execution."""


class SynthesisExecutionContext:
    """One immutable deadline plus a thread-safe cooperative cancellation signal."""

    def __init__(
        self,
        *,
        deadline_monotonic: float,
        correlation_id: str,
        cancellation_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        validation_time_monotonic: float | None = None,
    ) -> None:
        if not callable(clock):
            raise ValueError("execution clock must be callable")
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(deadline_monotonic)
        ):
            raise ValueError("execution deadline must be a finite monotonic value")
        if not isinstance(correlation_id, str) or not _CORRELATION_ID.fullmatch(
            correlation_id
        ):
            raise ValueError("execution correlation_id is invalid")
        if cancellation_event is not None and not isinstance(
            cancellation_event, threading.Event
        ):
            raise ValueError("execution cancellation_event must be threading.Event")
        now = float(
            clock()
            if validation_time_monotonic is None
            else validation_time_monotonic
        )
        if not math.isfinite(now):
            raise ValueError("execution clock returned a non-finite value")
        if float(deadline_monotonic) <= now:
            raise ExecutionDeadlineExceeded("render execution deadline has elapsed")

        self._deadline_monotonic = float(deadline_monotonic)
        self._correlation_id = correlation_id
        self._event = cancellation_event or threading.Event()
        self._clock = clock
        self._callback_lock = threading.Lock()
        self._callbacks: dict[str, Callable[[], None]] = {}
        self._cancel_reason = "not_cancelled"
        self._cancellation_outcome = "not_requested"
        self._cancellation_grace_exceeded = False
        self._uninterruptible_depth = 0

    @classmethod
    def from_timeout(
        cls,
        timeout_seconds: float,
        *,
        correlation_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> "SynthesisExecutionContext":
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("execution timeout must be finite and positive")
        now = float(clock())
        if not math.isfinite(now):
            raise ValueError("execution clock returned a non-finite value")
        return cls(
            deadline_monotonic=now + float(timeout_seconds),
            correlation_id=(
                correlation_id
                if correlation_id is not None
                else f"rap-{uuid.uuid4().hex}"
            ),
            clock=clock,
            validation_time_monotonic=now,
        )

    @property
    def deadline_monotonic(self) -> float:
        return self._deadline_monotonic

    @property
    def correlation_id(self) -> str:
        return self._correlation_id

    @property
    def cancel_reason(self) -> str:
        with self._callback_lock:
            return self._cancel_reason

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def cancellation_outcome(self) -> str:
        with self._callback_lock:
            return self._cancellation_outcome

    @property
    def upstream_abort_confirmed(self) -> bool:
        with self._callback_lock:
            return self._cancellation_outcome == "upstream_abort_confirmed"

    @property
    def cancellation_grace_exceeded(self) -> bool:
        with self._callback_lock:
            return self._cancellation_grace_exceeded

    @property
    def recovery_outcome(self) -> str:
        with self._callback_lock:
            if not self._event.is_set():
                return "not_required"
            if self._cancellation_grace_exceeded:
                return "restart_required"
            if self._cancellation_outcome == "upstream_abort_confirmed":
                return "upstream_released"
            return "abort_unconfirmed"

    @property
    def deadline_outcome(self) -> str:
        return (
            "deadline_exceeded"
            if self.remaining_seconds() <= 0
            else "within_deadline"
        )

    def remaining_seconds(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise RuntimeError("execution clock returned a non-finite value")
        return max(0.0, self._deadline_monotonic - now)

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise ExecutionCancelled("render execution was cancelled")

    def raise_if_expired(self) -> None:
        if self.remaining_seconds() <= 0:
            raise ExecutionDeadlineExceeded("render execution deadline has elapsed")

    def checkpoint(self) -> None:
        # Cancellation wins when both conditions become true at the same checkpoint.
        self.raise_if_cancelled()
        self.raise_if_expired()

    def cancel(self, reason: str = "cancelled") -> bool:
        bounded_reason = _bounded_reason(reason)
        with self._callback_lock:
            if self._event.is_set():
                return False
            self._cancel_reason = bounded_reason
            self._cancellation_outcome = (
                "cancel_requested_but_not_interruptible"
                if self._uninterruptible_depth
                else "cancel_requested"
            )
            self._event.set()
            callbacks = tuple(self._callbacks.values())
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass
        return True

    def add_cancel_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        if not callable(callback):
            raise ValueError("cancellation callback must be callable")
        token = uuid.uuid4().hex
        call_now = False
        with self._callback_lock:
            if self._event.is_set():
                call_now = True
            else:
                self._callbacks[token] = callback
        if call_now:
            callback()

        def remove() -> None:
            with self._callback_lock:
                self._callbacks.pop(token, None)

        return remove

    def record_cancellation_outcome(self, outcome: str) -> None:
        if outcome not in _CANCELLATION_OUTCOMES - {"not_requested"}:
            raise ValueError("execution cancellation outcome is invalid")
        with self._callback_lock:
            if not self._event.is_set():
                raise RuntimeError(
                    "cannot record a cancellation outcome before cancellation"
                )
            self._cancellation_outcome = outcome

    def record_cancellation_grace_exceeded(self) -> None:
        with self._callback_lock:
            if not self._event.is_set():
                raise RuntimeError(
                    "cannot record cancellation grace before cancellation"
                )
            self._cancellation_grace_exceeded = True

    @contextmanager
    def uninterruptible(self) -> Iterator[None]:
        """Mark a backend call that can observe cancellation only at its boundary."""
        with self._callback_lock:
            self._uninterruptible_depth += 1
        try:
            yield
        finally:
            with self._callback_lock:
                self._uninterruptible_depth -= 1
                if (
                    self._event.is_set()
                    and self._cancellation_outcome == "cancel_requested"
                ):
                    self._cancellation_outcome = (
                        "cancel_requested_but_not_interruptible"
                    )

    def owner_context(self) -> "SynthesisExecutionContext":
        """Create a separate cancellation signal with the same fixed deadline."""
        return SynthesisExecutionContext(
            deadline_monotonic=self._deadline_monotonic,
            correlation_id=self._correlation_id,
            clock=self._clock,
        )


def _bounded_reason(value: object) -> str:
    text = " ".join(str(value).split())[:128]
    if not text or any(ord(character) < 32 for character in text):
        return "cancelled"
    return text
