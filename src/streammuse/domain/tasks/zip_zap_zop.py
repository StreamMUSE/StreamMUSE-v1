"""ProjectIsochron-style Zip-Zap-Zop task implemented as a StreamMUSE task."""

from __future__ import annotations

from streammuse.domain.tasks.models import TaskRefereeResult, TaskState, TaskTurn


class ZipZapZopTask:
    name = "zip_zap_zop"

    def __init__(self, *, start_number: int = 1, max_history: int = 50) -> None:
        self.start_number = int(start_number)
        self.max_history = int(max_history)

    def initial_state(self) -> TaskState:
        return TaskState(
            task_name=self.name,
            turn_index=0,
            data={"current_number": self.start_number - 1, "start_number": self.start_number},
        )

    def build_turn(self, state: TaskState) -> TaskTurn:
        next_number = int(state.data.get("current_number", self.start_number - 1)) + 1
        expected = self.expected_response(next_number)
        system_prompt = (
            "You are playing Zip-Zap-Zop. Respond with only the next value. "
            "Multiples of 3 are Zip, multiples of 4 are Zap, multiples of 5 are Zop, "
            "and combined multiples concatenate those words. Otherwise respond with the number."
        )
        return TaskTurn(
            task_name=self.name,
            turn_id=state.turn_index,
            state=state,
            expected_output=expected,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{next_number}:"},
            ],
            metadata={"number": next_number},
        )

    def validate_response(self, state: TaskState, response_text: str) -> TaskRefereeResult:
        next_number = int(state.data.get("current_number", self.start_number - 1)) + 1
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
    ) -> TaskState:
        next_number = int(state.data.get("current_number", self.start_number - 1)) + 1
        history = list(state.history)
        history.append(
            {
                "role": "assistant",
                "content": str(response_text or "").strip(),
                "valid": str(bool(referee_result.is_valid)),
                "expected": referee_result.expected_output or "",
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
