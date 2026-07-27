"""Factories for creating infrastructure implementations from config."""

from __future__ import annotations

from typing import Any

__all__ = ["HumanInputFactory", "InferenceEngineFactory", "InputSourceFactory", "OutputSinkFactory"]


def __getattr__(name: str) -> Any:
    """Keep unrelated music infrastructure out of lightweight task commands."""

    if name == "HumanInputFactory":
        from streammuse.application.factories.human_input_factory import HumanInputFactory

        return HumanInputFactory
    if name == "InferenceEngineFactory":
        from streammuse.application.factories.inference_factory import InferenceEngineFactory

        return InferenceEngineFactory
    if name == "InputSourceFactory":
        from streammuse.application.factories.input_factory import InputSourceFactory

        return InputSourceFactory
    if name == "OutputSinkFactory":
        from streammuse.application.factories.output_factory import OutputSinkFactory

        return OutputSinkFactory
    raise AttributeError(name)
