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
            if cfg.model_name == "lekai":
                if int(cfg.generation_interval_ticks) % 4 != 0:
                    raise ValueError("lekai requires generation_interval_ticks to be a multiple of 4")
                if int(cfg.generation_length_frames) % 4 != 0:
                    raise ValueError("lekai requires generation_length_frames to be a multiple of 4")
            return HttpInferenceClient(
                HttpInferenceClientConfig(
                    generate_url=cfg.server_generate_url,
                    timeout_s=float(cfg.timeout_s),
                    model_name=cfg.model_name,
                    inference_mode=cfg.inference_mode,
                    generation_interval_ticks=int(cfg.generation_interval_ticks),
                )
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

