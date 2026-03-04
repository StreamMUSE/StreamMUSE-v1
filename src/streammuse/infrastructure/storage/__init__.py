"""Storage adapters (e.g. prompt repository). Optional for Step 4."""
from streammuse.infrastructure.storage.prompt_repository import (
    FileSystemPromptRepository,
    PromptRepositoryConfig,
)

__all__ = ["FileSystemPromptRepository", "PromptRepositoryConfig"]