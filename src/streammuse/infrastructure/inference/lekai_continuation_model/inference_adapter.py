"""Runtime adapter for the RT offline_model continuation checkpoint.

This package intentionally follows the tokenizer/model layout from
`RT-accompanimentV2/external/lekai_real_time/offline_model`, not the older
StreamMUSE `lekai_model` tokenizer. The prompt-continuation path needs this
because prompt-model accompaniment is re-encoded into the offline continuation
format before continuation generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import safetensors.torch
import torch
from transformers import LlamaConfig

from .config import ModelConfig
from .model import PianoLLaMA
from .my_tokenizer import PianoMusicTokenizer


def _llama_config(model_config: ModelConfig, *, use_cache: bool = True) -> LlamaConfig:
    return LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        pad_token_id=model_config.pad_token_id,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        rope_theta=model_config.rope_theta,
        attention_dropout=model_config.dropout,
        use_cache=use_cache,
        initializer_range=0.02,
    )


def _load_state_dict(path: str) -> dict[str, torch.Tensor]:
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        return safetensors.torch.load_file(path)
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        return payload
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value
    raise ValueError(f"Unsupported checkpoint structure: {path}")


def _strip_prefix_all(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _add_prefix_missing(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key if key.startswith(prefix) else f"{prefix}{key}": value for key, value in state_dict.items()}


def _load_strict_with_key_adapters(model: PianoLLaMA, state_dict: dict[str, torch.Tensor]) -> None:
    candidates = [
        state_dict,
        _strip_prefix_all(state_dict, "module."),
        _add_prefix_missing(_strip_prefix_all(state_dict, "module."), "model."),
        _add_prefix_missing(state_dict, "model."),
        _strip_prefix_all(state_dict, "model."),
    ]
    seen: set[tuple[str, ...]] = set()
    last_error: RuntimeError | None = None
    for candidate in candidates:
        key_tuple = tuple(candidate.keys())
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error


class PianoContinuationAdapter:
    """Small runtime wrapper around the RT offline continuation model."""

    BAR_TOKEN = 255

    def __init__(
        self,
        *,
        wrapper: PianoLLaMA,
        tokenizer: PianoMusicTokenizer,
        device: str,
        use_cache: bool,
    ) -> None:
        self.wrapper = wrapper
        # Expose the inner LlamaForCausalLM because the realtime backend manages
        # KV cache explicitly for one-beat generation.
        self.model = wrapper.model
        self.tokenizer = tokenizer
        self.device = device
        self.use_cache = use_cache

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        use_cache: bool = True,
    ) -> "PianoContinuationAdapter":
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model_config = ModelConfig()
        wrapper = PianoLLaMA(_llama_config(model_config, use_cache=use_cache))
        _load_strict_with_key_adapters(wrapper, _load_state_dict(checkpoint_path))

        if dtype is not None and dtype != torch.float32:
            wrapper = wrapper.to(dtype=dtype)
        wrapper = wrapper.to(device)
        wrapper.eval()

        return cls(
            wrapper=wrapper,
            tokenizer=PianoMusicTokenizer(config=model_config),
            device=device,
            use_cache=use_cache,
        )
