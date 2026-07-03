"""ProjectIsochron-style Zip-Zap-Zop task implemented as a StreamMUSE task."""

from __future__ import annotations

from streammuse.domain.tasks.models import (
    ChatMessage,
    InteractiveActor,
    InteractiveTurnRecord,
    TaskRefereeResult,
    TaskState,
    TaskTurn,
)


class ZipZapZopTask:
    name = "zip_zap_zop"

    def __init__(self, *, start_number: int = 1, max_history: int = 50, history_limit: int = 8) -> None:
        self.start_number = int(start_number)
        self.max_history = int(max_history)
        self.history_limit = max(0, int(history_limit))

    def initial_state(self) -> TaskState:
        return TaskState(
            task_name=self.name,
            turn_index=0,
            data={"current_number": self.start_number - 1, "start_number": self.start_number},
        )

    def build_turn(self, state: TaskState) -> TaskTurn:
        next_number = self.current_number(state)
        expected = self.expected_response(next_number)
        return TaskTurn(
            task_name=self.name,
            turn_id=state.turn_index,
            state=state,
            expected_output=expected,
            messages=[
                {"role": "system", "content": self.rules_prompt()},
                {"role": "user", "content": f"{next_number}:"},
            ],
            metadata={"number": next_number},
        )

    def build_human_prompt(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str:
        _ = transcript
        return f"{self.current_number(state)}:"

    def build_llm_messages(
        self,
        state: TaskState,
        transcript: list[InteractiveTurnRecord],
    ) -> list[ChatMessage]:
        number = self.current_number(state)
        if self.history_limit <= 0 or not transcript:
            user_content = f"{number}:"
        else:
            recent = transcript[-self.history_limit :]
            history_lines = []
            for record in recent:
                label = "Human" if record.actor == "human" else "Assistant"
                turn_number = record.number if record.number is not None else "?"
                history_lines.append(f"{label} at {turn_number}: {record.response}")
            user_content = "Recent turns:\n" + "\n".join(history_lines) + f"\nYour turn. {number}:"

        return [
            {"role": "system", "content": self.rules_prompt(with_human_context=True)},
            {"role": "user", "content": user_content},
        ]

    def build_hint(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str | None:
        _ = transcript
        number = self.current_number(state)
        divisors = [str(divisor) for divisor in (3, 4, 5) if number % divisor == 0]
        if not divisors:
            return f"{number} is not divisible by 3, 4, or 5."
        return f"Check divisibility by: {', '.join(divisors)}."

    def expected_for_state(self, state: TaskState, transcript: list[InteractiveTurnRecord]) -> str | None:
        _ = transcript
        return self.expected_response(self.current_number(state))

    def validate_response(
        self,
        state: TaskState,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskRefereeResult:
        _ = actor, transcript
        next_number = self.current_number(state)
        expected = self.expected_response(next_number)
        normalized = str(response_text or "").strip()
        is_valid = normalized.lower() == expected.lower()
        return TaskRefereeResult(
            is_valid=is_valid,
            expected_output=expected,
            failure_reason="NONE" if is_valid else "EXPECTED_MISMATCH",
            metadata={"number": next_number, "actual_output": normalized},
        )

    def advance_state(
        self,
        state: TaskState,
        referee_result: TaskRefereeResult,
        response_text: str,
        *,
        actor: InteractiveActor | None = None,
        transcript: list[InteractiveTurnRecord] | None = None,
    ) -> TaskState:
        _ = transcript
        next_number = self.current_number(state)
        role = "human" if actor == "human" else "assistant"
        history = list(state.history)
        history.append(
            {
                "role": role,
                "content": str(response_text or "").strip(),
                "valid": str(bool(referee_result.is_valid)),
                "expected": referee_result.expected_output or "",
                "number": str(next_number),
            }
        )
        if len(history) > self.max_history:
            history = history[-self.max_history :]
        return TaskState(
            task_name=self.name,
            turn_index=state.turn_index + 1,
            data={**state.data, "current_number": next_number},
            history=tuple(history),
        )

    def current_number(self, state: TaskState) -> int:
        return int(state.data.get("current_number", self.start_number - 1)) + 1

    @staticmethod
    def rules_prompt(*, with_human_context: bool = False) -> str:
        prefix = "You are playing Zip-Zap-Zop."
        if with_human_context:
            prefix = "You are playing Zip-Zap-Zop with a human."
        return (
            f"{prefix} Respond with only the next value. "
            "Multiples of 3 are Zip, multiples of 4 are Zap, multiples of 5 are Zop, "
            "and combined multiples concatenate those words. Otherwise respond with the number."
        )

    @staticmethod
    def expected_response(number: int) -> str:
        value = ""
        if number % 3 == 0:
            value += "Zip"
        if number % 4 == 0:
            value += "Zap"
        if number % 5 == 0:
            value += "Zop"
        return value or str(number)
