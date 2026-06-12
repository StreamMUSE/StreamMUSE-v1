"""Lekai prompt-continuation inference components.

Public classes are imported lazily so lightweight submodules such as
``catchup_state`` can be tested without importing model runtimes.
"""

from __future__ import annotations

from typing import Any


__all__ = [
    "LekaiContinuationEngine",
    "LekaiPromptContinuationBackend",
    "LekaiPromptContinuationEngine",
    "LekaiPromptContinuationScheduler",
    "LekaiPromptExtensionContinuationEngine",
    "LekaiPromptExtensionContinuationScheduler",
    "LekaiPromptEngine",
]


def __getattr__(name: str) -> Any:
    if name == "LekaiPromptContinuationBackend":
        from streammuse.infrastructure.inference.lekai_prompt_continuation.backend import (
            LekaiPromptContinuationBackend,
        )

        return LekaiPromptContinuationBackend
    if name == "LekaiContinuationEngine":
        from streammuse.infrastructure.inference.lekai_prompt_continuation.continuation_engine import (
            LekaiContinuationEngine,
        )

        return LekaiContinuationEngine
    if name == "LekaiPromptExtensionContinuationEngine":
        from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_extension_engine import (
            LekaiPromptExtensionContinuationEngine,
        )

        return LekaiPromptExtensionContinuationEngine
    if name == "LekaiPromptExtensionContinuationScheduler":
        from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_extension_scheduler import (
            LekaiPromptExtensionContinuationScheduler,
        )

        return LekaiPromptExtensionContinuationScheduler
    if name == "LekaiPromptContinuationEngine":
        from streammuse.infrastructure.inference.lekai_prompt_continuation.engine import (
            LekaiPromptContinuationEngine,
        )

        return LekaiPromptContinuationEngine
    if name == "LekaiPromptContinuationScheduler":
        from streammuse.infrastructure.inference.lekai_prompt_continuation.scheduler import (
            LekaiPromptContinuationScheduler,
        )

        return LekaiPromptContinuationScheduler
    if name == "LekaiPromptEngine":
        from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import (
            LekaiPromptEngine,
        )

        return LekaiPromptEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
