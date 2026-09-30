from __future__ import annotations

import math
import threading

import pytest

from streammuse.application.rap.execution import (
    ExecutionCancelled,
    ExecutionDeadlineExceeded,
    SynthesisExecutionContext,
)


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_absolute_deadline_budget_decreases_and_expires_exactly() -> None:
    clock = _Clock()
    execution = SynthesisExecutionContext(
        deadline_monotonic=15.0,
        correlation_id="request-1",
        clock=clock,
    )

    assert execution.remaining_seconds() == 5.0
    clock.value = 14.25
    assert execution.remaining_seconds() == 0.75
    assert execution.deadline_outcome == "within_deadline"

    clock.value = 15.0
    assert execution.remaining_seconds() == 0.0
    assert execution.deadline_outcome == "deadline_exceeded"
    with pytest.raises(ExecutionDeadlineExceeded, match="deadline"):
        execution.checkpoint()


def test_cancellation_wins_when_deadline_and_cancel_are_both_observable() -> None:
    clock = _Clock()
    execution = SynthesisExecutionContext(
        deadline_monotonic=11.0,
        correlation_id="request-2",
        clock=clock,
    )
    execution.cancel("client disconnected")
    clock.value = 12.0

    with pytest.raises(ExecutionCancelled, match="cancelled"):
        execution.checkpoint()


def test_cancel_during_uninterruptible_region_is_reported_honestly() -> None:
    execution = SynthesisExecutionContext.from_timeout(
        10.0, correlation_id="request-3"
    )

    with execution.uninterruptible():
        assert execution.cancel("last waiter detached") is True
        assert (
            execution.cancellation_outcome
            == "cancel_requested_but_not_interruptible"
        )

    with pytest.raises(ExecutionCancelled):
        execution.checkpoint()


def test_cancel_callbacks_run_once_and_can_record_transport_outcome() -> None:
    execution = SynthesisExecutionContext.from_timeout(
        10.0, correlation_id="request-4"
    )
    calls: list[str] = []

    def close_transport() -> None:
        calls.append("closed")
        execution.record_cancellation_outcome(
            "transport_closed_abort_unconfirmed"
        )

    remove = execution.add_cancel_callback(close_transport)
    assert execution.cancel("disconnect") is True
    assert execution.cancel("duplicate") is False
    remove()

    assert calls == ["closed"]
    assert execution.cancel_reason == "disconnect"
    assert (
        execution.cancellation_outcome
        == "transport_closed_abort_unconfirmed"
    )


@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf, True, "12"))
def test_rejects_invalid_deadlines(value: object) -> None:
    with pytest.raises(ValueError, match="deadline"):
        SynthesisExecutionContext(
            deadline_monotonic=value,  # type: ignore[arg-type]
            correlation_id="request-5",
        )


@pytest.mark.parametrize(
    "correlation_id", ("", " contains-space", "x" * 129, "line\nbreak")
)
def test_rejects_invalid_correlation_ids(correlation_id: str) -> None:
    with pytest.raises(ValueError, match="correlation_id"):
        SynthesisExecutionContext.from_timeout(
            10.0, correlation_id=correlation_id
        )


def test_rejects_wrong_cancellation_signal_and_non_finite_clock() -> None:
    with pytest.raises(ValueError, match="threading.Event"):
        SynthesisExecutionContext(
            deadline_monotonic=20.0,
            correlation_id="request-6",
            cancellation_event=object(),  # type: ignore[arg-type]
            validation_time_monotonic=10.0,
        )
    with pytest.raises(ValueError, match="non-finite"):
        SynthesisExecutionContext.from_timeout(
            10.0,
            correlation_id="request-7",
            clock=lambda: math.nan,
        )


def test_owner_context_keeps_deadline_but_owns_a_distinct_signal() -> None:
    clock = _Clock()
    waiter = SynthesisExecutionContext(
        deadline_monotonic=20.0,
        correlation_id="request-8",
        cancellation_event=threading.Event(),
        clock=clock,
    )
    owner = waiter.owner_context()

    waiter.cancel("waiter detached")

    assert owner.deadline_monotonic == waiter.deadline_monotonic
    assert owner.correlation_id == waiter.correlation_id
    assert owner.cancelled is False


def test_cancellation_recovery_evidence_distinguishes_abort_and_restart() -> None:
    execution = SynthesisExecutionContext.from_timeout(
        10.0, correlation_id="cancel-evidence"
    )

    assert execution.upstream_abort_confirmed is False
    assert execution.cancellation_grace_exceeded is False
    assert execution.recovery_outcome == "not_required"

    execution.cancel("client_disconnected")
    assert execution.recovery_outcome == "abort_unconfirmed"

    execution.record_cancellation_outcome("upstream_abort_confirmed")
    assert execution.upstream_abort_confirmed is True
    assert execution.recovery_outcome == "upstream_released"

    execution.record_cancellation_grace_exceeded()
    assert execution.cancellation_grace_exceeded is True
    assert execution.recovery_outcome == "restart_required"


def test_cancellation_grace_cannot_be_recorded_before_cancellation() -> None:
    execution = SynthesisExecutionContext.from_timeout(
        10.0, correlation_id="premature-grace"
    )

    with pytest.raises(RuntimeError, match="before cancellation"):
        execution.record_cancellation_grace_exceeded()
