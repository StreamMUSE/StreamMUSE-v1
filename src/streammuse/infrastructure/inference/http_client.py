"""HTTP inference adapter implementing the domain InferenceEngine protocol."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from streammuse.domain.interfaces import InferenceEngine, TimingInfo
from streammuse.domain.musical import MusicalEvent
from streammuse.infrastructure.inference.serialization import event_from_dict, event_to_dict, timing_info_from_dict


@dataclass(frozen=True)
class HttpInferenceClientConfig:
    generate_url: str
    timeout_s: float = 30.0
    model_name: str = "stanley"
    inference_mode: str = "sliding_window"
    generation_interval_ticks: int = 2
    checkpoint_path: Optional[str] = None
    bpm: Optional[int] = None


class HttpInferenceClient(InferenceEngine):
    """
    Adapter that talks to the legacy FastAPI inference server.

    Expects `generate_url` like `http://host:8000/generate_accompaniment` and
    derives the other endpoints by replacing the path segment.
    """

    def __init__(self, config: HttpInferenceClientConfig) -> None:
        self._config = config
        self._injection_offset_ticks = 0

    def _endpoint(self, replacement_path: str) -> str:
        # Legacy clients derive endpoints by replacing /generate_accompaniment.
        return self._config.generate_url.replace("/generate_accompaniment", replacement_path)

    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]:
        payload: Dict[str, Any] = {
            "melody_notes": [event_to_dict(e) for e in melody_events],
            "generation_start_tick": int(generation_start_tick),
            "client_request_send_time": time.time(),
            "generation_length_frames": int(generation_length_frames),
            "generation_interval_ticks": int(self._config.generation_interval_ticks),
            "model_name": str(self._config.model_name),
            "inference_mode": str(self._config.inference_mode),
            "checkpoint_path": self._config.checkpoint_path,
            "prompt_length_ticks": (int(prompt_length_ticks) if prompt_length_ticks is not None else None),
            "bpm": self._config.bpm,
        }
        # Drop nulls for cleaner wire format.
        payload = {k: v for k, v in payload.items() if v is not None}

        resp = requests.post(
            self._config.generate_url,
            json=payload,
            timeout=float(self._config.timeout_s),
        )
        resp.raise_for_status()
        data = resp.json()

        accompaniment = [event_from_dict(d) for d in data.get("accompaniment", [])]
        timings = timing_info_from_dict(data["timings"])
        return accompaniment, timings

    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None:
        url = self._endpoint("/inject_notes")
        payload: Dict[str, Any] = {
            "melody_notes": [event_to_dict(e) for e in melody_events],
            "accompaniment_notes": [event_to_dict(e) for e in accompaniment_events],
            "injection_length_ticks": int(injection_length_ticks),
        }
        resp = requests.post(url, json=payload, timeout=float(self._config.timeout_s))
        resp.raise_for_status()
        self._injection_offset_ticks = int(injection_length_ticks)

    def set_injection_offset(self, offset_ticks: int) -> None:
        # The legacy server exposes injection offset only via /inject_notes.
        # We track it client-side for callers that want to store it.
        self._injection_offset_ticks = int(offset_ticks)

    def clear_history(self) -> Dict[str, Any]:
        url = self._endpoint("/clear_history")
        resp = requests.post(url, timeout=float(self._config.timeout_s))
        resp.raise_for_status()
        data = resp.json()
        return {
            "success": bool(data.get("success", True)),
            "message": str(data.get("message", "History cleared")),
            "melody_history": list(data.get("melody_history", [])),
            "accompaniment_history": list(data.get("accompaniment_history", [])),
        }

    # Non-protocol helper
    def get_injection_status(self) -> Dict[str, Any]:
        url = self._endpoint("/injection_status")
        resp = requests.get(url, timeout=float(self._config.timeout_s))
        resp.raise_for_status()
        return resp.json()

