"""Canonical MOSS generation settings shared by online and offline adapters."""

from __future__ import annotations

from types import MappingProxyType
from typing import Protocol

from streammuse.experiments.rap_audio_protocols.timing import moss_token_target


MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
LANGUAGE = "English"
RAP_INSTRUCTION = "clear, rhythmically spoken rap with restrained pitch"
GENERATION_MODE = "generation"
MAX_NEW_TOKENS = 256
DEFAULT_BASE_SEED = 20260816
SEED_POLICY_VERSION = "streammuse.moss_seed.v1"
GENERATION_KWARGS = MappingProxyType(
    {
        "max_new_tokens": MAX_NEW_TOKENS,
        "audio_temperature": 1.7,
        "audio_top_p": 0.8,
        "audio_top_k": 25,
        "audio_repetition_penalty": 1.0,
    }
)
STYLE_INSTRUCTION_CAVEAT = (
    "The shared MOSS processor accepts an instruction field, but MOSS-TTS-v1.5 does not document "
    "style-following as a guaranteed model capability."
)
DETERMINISTIC_SEED_CAVEAT = (
    "Official deterministic seeding is unsupported by the documented MOSS-TTS-v1.5 API; this backend "
    "uses best-effort PyTorch seeding on each attempt."
)


def producer_generation_settings(
    *, base_seed: int = DEFAULT_BASE_SEED
) -> dict[str, object]:
    """Return static output-affecting settings for producer fingerprinting."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("MOSS base seed must be a non-negative integer")
    return {
        "language": LANGUAGE,
        "instruction": RAP_INSTRUCTION,
        "generation_mode": GENERATION_MODE,
        "generation_kwargs": dict(GENERATION_KWARGS),
        "base_seed": base_seed,
    }


class _ChunkRequest(Protocol):
    chunk_index: int


def seed_for_attempt(*, base_seed: int, request: _ChunkRequest, attempt: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("MOSS base seed must be a non-negative integer")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("MOSS attempt must be a positive integer")
    chunk_index = request.chunk_index
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("MOSS chunk index must be a non-negative integer")
    return base_seed + chunk_index * 1000 + (attempt - 1)


def resolved_generation_settings(
    request: _ChunkRequest, *, base_seed: int, attempt: int = 1
) -> dict[str, object]:
    return {
        "language": LANGUAGE,
        "instruction": RAP_INSTRUCTION,
        "generation_mode": GENERATION_MODE,
        "generation_kwargs": dict(GENERATION_KWARGS),
        "token_target": moss_token_target(request),  # type: ignore[arg-type]
        "base_seed": base_seed,
        "attempt": attempt,
        "seed": seed_for_attempt(
            base_seed=base_seed,
            request=request,
            attempt=attempt,
        ),
    }
