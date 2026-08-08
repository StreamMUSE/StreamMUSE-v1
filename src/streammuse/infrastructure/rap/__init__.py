"""Infrastructure adapters for rap-alignment candidate generation."""

from streammuse.infrastructure.rap.generators import LocalChatCandidateGenerator, PhraseBankGenerator
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog, PrevalidatedFallbackLine

__all__ = [
    "LocalChatCandidateGenerator",
    "PhraseBankGenerator",
    "PrevalidatedFallbackCatalog",
    "PrevalidatedFallbackLine",
]
"""Infrastructure adapters for rap-alignment candidate generation."""

from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog, PrevalidatedFallbackLine
from streammuse.infrastructure.rap.generators import LocalChatCandidateGenerator, PhraseBankGenerator
from streammuse.infrastructure.rap.recorder import (
    RapSessionManifest,
    RapSessionRecorder,
    build_session_manifest,
    derive_bar_rows,
    derive_summary,
    validate_session_manifest,
)

__all__ = [
    "LocalChatCandidateGenerator",
    "PhraseBankGenerator",
    "PrevalidatedFallbackCatalog",
    "PrevalidatedFallbackLine",
    "RapSessionManifest",
    "RapSessionRecorder",
    "build_session_manifest",
    "derive_bar_rows",
    "derive_summary",
    "validate_session_manifest",
]
