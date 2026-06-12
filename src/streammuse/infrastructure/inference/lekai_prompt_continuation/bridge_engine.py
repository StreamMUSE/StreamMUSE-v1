"""Bridge-beat engine variant for Lekai prompt-continuation."""

from __future__ import annotations

import os
from typing import Optional

from streammuse.infrastructure.inference.lekai_prompt_continuation.bridge_scheduler import (
    LekaiPromptBridgeContinuationScheduler,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import (
    LekaiContinuationEngine,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.engine import (
    LekaiPromptContinuationEngine,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
    LekaiPromptEngine,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.scheduler import (
    TIMESTEPS_PER_BEAT,
)


class LekaiPromptBridgeContinuationEngine(LekaiPromptContinuationEngine):
    """Prompt-continuation engine that lets prompt output bridge into beat 9."""

    VARIANT = "bridge"

    def __init__(
        self,
        prompt_checkpoint_path: Optional[str] = None,
        continuation_checkpoint_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        prompt_engine: Optional[LekaiPromptEngine] = None,
        continuation_engine: Optional[LekaiContinuationEngine] = None,
        bridge_ticks: Optional[int] = None,
    ) -> None:
        super().__init__(
            prompt_checkpoint_path=prompt_checkpoint_path,
            continuation_checkpoint_path=continuation_checkpoint_path,
            checkpoint_path=checkpoint_path,
            prompt_engine=prompt_engine,
            continuation_engine=continuation_engine,
        )
        self._scheduler.shutdown()
        self._prompt_bridge_ticks = (
            int(bridge_ticks)
            if bridge_ticks is not None
            else self._env_bridge_ticks()
        )
        self._scheduler = LekaiPromptBridgeContinuationScheduler(
            prompt_engine=self._prompt_engine,
            continuation_engine=self._continuation_engine,
            bridge_ticks=self._prompt_bridge_ticks,
        )

    @staticmethod
    def _env_positive_int(name: str) -> Optional[int]:
        value = os.environ.get(name)
        if value is None or str(value).strip() == "":
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None

    @classmethod
    def _env_bridge_ticks(cls) -> int:
        bridge_ticks = cls._env_positive_int("LEKAI_PROMPT_CONTINUATION_BRIDGE_TICKS")
        if bridge_ticks is not None:
            return bridge_ticks
        bridge_beats = cls._env_positive_int("LEKAI_PROMPT_CONTINUATION_BRIDGE_BEATS")
        if bridge_beats is not None:
            return int(bridge_beats) * TIMESTEPS_PER_BEAT
        return TIMESTEPS_PER_BEAT

    def runtime_info(self) -> dict[str, str | float | bool | int | None]:
        info = dict(super().runtime_info())
        info["prompt_continuation_variant"] = self.VARIANT
        info["prompt_bridge_ticks"] = int(self._prompt_bridge_ticks)
        return info
