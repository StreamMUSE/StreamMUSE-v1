from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set


EventPayload = Dict[str, int | str]
TimingPayload = Dict[str, float]


@dataclass(frozen=True)
class BackendRuntimeConfig:
    model_name: str
    inference_mode: str
    generation_interval_ticks: int
    generation_length_frames: int
    prompt_length_ticks: Optional[int]
    checkpoint_path: Optional[str]


class LekaiHttpBackend:
    def __init__(self) -> None:
        self._melody_history: List[EventPayload] = []
        self._accompaniment_history: List[EventPayload] = []
        self._injection_length_ticks: int = 0
        self._active_model_pitches: Set[int] = set()
        self._runtime_config: Optional[BackendRuntimeConfig] = None

    def configure(self, config: BackendRuntimeConfig) -> None:
        self._runtime_config = config

    def generate(
        self,
        melody_events: List[EventPayload],
        generation_start_tick: int,
        generation_length_frames: int,
        generation_interval_ticks: int,
        prompt_length_ticks: Optional[int],
        inference_mode: str,
        model_name: str,
        checkpoint_path: Optional[str],
    ) -> tuple[List[EventPayload], TimingPayload]:
        request_arrival = time.perf_counter()
        preprocess_start = request_arrival

        self.configure(
            BackendRuntimeConfig(
                model_name=model_name,
                inference_mode=inference_mode,
                generation_interval_ticks=int(generation_interval_ticks),
                generation_length_frames=int(generation_length_frames),
                prompt_length_ticks=prompt_length_ticks,
                checkpoint_path=checkpoint_path,
            )
        )

        self._melody_history = list(melody_events)

        inference_start = time.perf_counter()
        accompaniment = self._generate_rule_based(
            generation_start_tick=int(generation_start_tick),
            generation_interval_ticks=int(generation_interval_ticks),
        )
        inference_end = time.perf_counter()

        self._accompaniment_history.extend(accompaniment)
        response_output = time.perf_counter()

        timings: TimingPayload = {
            "request_arrival_time": request_arrival,
            "response_output_time": response_output,
            "preprocess_start_time": preprocess_start,
            "inference_start_time": inference_start,
            "inference_end_time": inference_end,
            "postprocess_start_time": inference_end,
        }
        return accompaniment, timings

    def inject_history(
        self,
        melody_events: List[EventPayload],
        accompaniment_events: List[EventPayload],
        injection_length_ticks: int,
    ) -> Dict[str, int | bool | str]:
        self._melody_history = list(melody_events)
        self._accompaniment_history = list(accompaniment_events)
        self._injection_length_ticks = int(injection_length_ticks)
        self._active_model_pitches = self._build_active_pitches(accompaniment_events)

        return {
            "success": True,
            "message": "Notes injected successfully",
            "melody_notes_injected": len(melody_events),
            "accompaniment_notes_injected": len(accompaniment_events),
            "injection_length_ticks": int(injection_length_ticks),
        }

    def clear_history(self) -> Dict[str, bool | str]:
        self._melody_history = []
        self._accompaniment_history = []
        self._injection_length_ticks = 0
        self._active_model_pitches = set()
        return {"success": True, "message": "History cleared"}

    def injection_status(self) -> Dict[str, bool | int | str]:
        return {
            "is_injected": self._injection_length_ticks > 0,
            "injection_length_ticks": self._injection_length_ticks,
            "runtime_model_name": self._runtime_config.model_name if self._runtime_config else "",
            "runtime_inference_mode": self._runtime_config.inference_mode if self._runtime_config else "",
        }

    def _build_active_pitches(self, events: List[EventPayload]) -> Set[int]:
        active: Set[int] = set()
        sorted_events = sorted(
            events,
            key=lambda e: (int(e.get("tick", 0)), 0 if str(e.get("type", "")) == "note_off" else 1),
        )
        for event in sorted_events:
            event_type = str(event.get("type", ""))
            pitch = int(event.get("pitch", -1))
            if pitch < 0:
                continue
            if event_type == "note_on":
                active.add(pitch)
            elif event_type == "note_off":
                active.discard(pitch)
        return active

    def _generate_rule_based(self, generation_start_tick: int, generation_interval_ticks: int) -> List[EventPayload]:
        interval = max(1, int(generation_interval_ticks))
        window_start = generation_start_tick - interval * 2
        recent_on_pitches: List[int] = []

        for event in self._melody_history:
            event_type = str(event.get("type", ""))
            tick = int(event.get("tick", 0))
            pitch = int(event.get("pitch", -1))
            if pitch < 0:
                continue
            if event_type == "note_on" and window_start <= tick <= generation_start_tick:
                recent_on_pitches.append(pitch)

        unique: List[int] = []
        seen: Set[int] = set()
        for pitch in reversed(recent_on_pitches):
            if pitch in seen:
                continue
            seen.add(pitch)
            unique.append(pitch)
            if len(unique) >= 4:
                break

        if not unique and self._active_model_pitches:
            events: List[EventPayload] = []
            for pitch in sorted(self._active_model_pitches):
                events.append(
                    {
                        "type": "note_off",
                        "pitch": int(pitch),
                        "tick": int(generation_start_tick),
                    }
                )
            self._active_model_pitches = set()
            return events

        accompaniment: List[EventPayload] = []
        next_active: Set[int] = set()
        chord_root_tick = int(generation_start_tick)
        note_off_tick = int(generation_start_tick + interval)

        for pitch in unique:
            model_pitch = max(36, pitch - 12)
            accompaniment.append(
                {
                    "type": "note_on",
                    "pitch": int(model_pitch),
                    "tick": chord_root_tick,
                    "velocity": 80,
                }
            )
            accompaniment.append(
                {
                    "type": "note_off",
                    "pitch": int(model_pitch),
                    "tick": note_off_tick,
                }
            )
            next_active.add(model_pitch)

        self._active_model_pitches = next_active
        accompaniment.sort(key=lambda e: (int(e["tick"]), 0 if str(e["type"]) == "note_off" else 1))
        return accompaniment
