"""HTTP inference adapter implementing the domain InferenceEngine protocol."""

from __future__ import annotations

import copy
import time
import threading
import uuid
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
    input_file: Optional[str] = None  # 输入文件名，用于日志记录
    session_id: Optional[str] = None
    session_epoch: Optional[int] = None
    effective_seed: Optional[int] = None


class HttpInferenceClient(InferenceEngine):
    """
    Adapter that talks to the legacy FastAPI inference server.

    Expects `generate_url` like `http://host:8000/generate_accompaniment` and
    derives the other endpoints by replacing the path segment.
    """

    def __init__(self, config: HttpInferenceClientConfig) -> None:
        self._config = config
        self._injection_offset_ticks = 0
        self._session_id = config.session_id
        self._session_epoch = config.session_epoch
        self._effective_seed: Optional[int] = config.effective_seed
        self._request_counter = 0
        self._next_request_id: Optional[str] = None
        self._metadata_lock = threading.Lock()
        self._last_response_metadata: Dict[str, Any] = {}

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
        with self._metadata_lock:
            self._request_counter += 1
            request_id = self._next_request_id or (
                f"{self._session_id}-r{self._request_counter:06d}"
                if self._session_id
                else uuid.uuid4().hex
            )
            self._next_request_id = None
            session_id = self._session_id
            session_epoch = self._session_epoch
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
            "input_file": self._config.input_file,
            "session_id": session_id,
            "session_epoch": session_epoch,
            "request_id": request_id,
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
        metadata = dict(data.get("metadata", {}))
        metadata.setdefault("request_id", request_id)
        with self._metadata_lock:
            self._last_response_metadata = copy.deepcopy(metadata)
        return accompaniment, timings

    @property
    def last_response_metadata(self) -> Dict[str, Any]:
        with self._metadata_lock:
            return copy.deepcopy(self._last_response_metadata)

    def consume_last_response_metadata(self) -> Dict[str, Any]:
        with self._metadata_lock:
            metadata = copy.deepcopy(self._last_response_metadata)
            self._last_response_metadata = {}
            return metadata

    def set_next_request_id(self, request_id: str) -> None:
        """Bind the next HTTP request to its service lifecycle request id."""
        value = str(request_id).strip()
        if not value:
            raise ValueError("request_id must be non-empty")
        with self._metadata_lock:
            self._next_request_id = value

    def reset_session(self, seed: int) -> Dict[str, Any]:
        """Start a new, atomically seeded server session."""
        url = self._endpoint("/debug/reset_session")
        resp = requests.post(
            url,
            json={"seed": int(seed)},
            timeout=float(self._config.timeout_s),
        )
        resp.raise_for_status()
        data = dict(resp.json())
        with self._metadata_lock:
            self._session_id = str(data["session_id"])
            self._session_epoch = int(data["session_epoch"])
            self._effective_seed = int(data["effective_seed"])
            self._request_counter = 0
            self._next_request_id = None
            self._last_response_metadata = {}
        return data

    def get_runtime_info(self) -> Dict[str, Any]:
        url = self._endpoint("/runtime_info")
        resp = requests.get(url, timeout=float(self._config.timeout_s))
        resp.raise_for_status()
        return dict(resp.json())

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
