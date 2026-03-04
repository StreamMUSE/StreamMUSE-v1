"""InferenceEngine factory."""

from __future__ import annotations

from streammuse.application.config import ApplicationConfig
from streammuse.domain.interfaces import InferenceEngine
from streammuse.infrastructure.inference import (
    HttpInferenceClient,
    HttpInferenceClientConfig,
    StanleyInferenceConfig,
    StanleyInferenceEngine,
)


class InferenceEngineFactory:
    @staticmethod
    def create(app_config: ApplicationConfig) -> InferenceEngine:
        cfg = app_config.inference

        if cfg.type == "http":
            return HttpInferenceClient(
                HttpInferenceClientConfig(generate_url=cfg.server_generate_url, timeout_s=float(cfg.timeout_s))
            )

        if cfg.type == "stanley":
            if not cfg.checkpoint_path:
                raise ValueError("checkpoint_path is required for stanley inference engine")
            return StanleyInferenceEngine(
                config=StanleyInferenceConfig(
                    checkpoint_path=cfg.checkpoint_path,
                    model_size=cfg.model_size,
                    model_max_seq_len_frames=int(cfg.model_max_seq_len_frames),
                    generation_length_frames=int(cfg.generation_length_frames),
                )
            )

        raise ValueError(f"Unknown inference type: {cfg.type}")

