"""In-process Stanley inference adapter implementing InferenceEngine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from streammuse.domain.interfaces import InferenceEngine, TimingInfo
from streammuse.domain.musical import MusicalEvent, Note
from streammuse.domain.musical.converters import events_to_notes


class _LegacyStanleyLike(Protocol):
    generation_length_frames: int

    def generate_accompaniment(
        self,
        melody_notes: list[dict],
        generation_start_tick: int,
        acc_notes: Optional[list[dict]] = None,
        generation_length_frames: Optional[int] = None,
        prompt_length_ticks: Optional[int] = None,
    ) -> tuple[list[dict], float, float, float, float]:
        ...

    def clear_history(self) -> None: ...

    def set_injection_offset(self, offset_ticks: int) -> None: ...


@dataclass(frozen=True)
class StanleyInferenceConfig:
    checkpoint_path: str
    model_size: str = "0.12B"
    model_max_seq_len_frames: int = 96
    generation_length_frames: int = 20
    max_polyphony: int = 4


class StanleyInferenceEngine(InferenceEngine):
    """
    Adapter around the legacy Stanley duration-note engine.

    Domain interface is event-stream; the legacy engine is duration-note based.
    We convert events→notes using close-at-horizon policy before calling the legacy engine.
    """

    def __init__(self, *, config: StanleyInferenceConfig, legacy_engine: _LegacyStanleyLike | None = None) -> None:
        self._config = config
        if legacy_engine is None:
            from streammuse.infrastructure.inference.stanley_legacy import LegacyInferenceEngineStanley

            self._legacy: _LegacyStanleyLike = LegacyInferenceEngineStanley(
                checkpoint_path=config.checkpoint_path,
                model_size=config.model_size,
                max_polyphony=config.max_polyphony,
                model_max_seq_len_frames=config.model_max_seq_len_frames,
                generation_length_frames=config.generation_length_frames,
            )
        else:
            self._legacy = legacy_engine

    def generate_accompaniment(
        self,
        melody_events: List[MusicalEvent],
        generation_start_tick: int,
        generation_length_frames: int,
        prompt_length_ticks: int | None = None,
    ) -> tuple[List[MusicalEvent], TimingInfo]:
        # Convert event stream to duration notes for the legacy engine.
        melody_notes: list[Note] = events_to_notes(melody_events, horizon_tick=int(generation_start_tick))
        melody_dicts = [
            {"pitch": n.pitch, "tick": n.tick, "duration": n.duration, "velocity": n.velocity, "program": n.program}
            for n in melody_notes
        ]

        acc_notes, preprocess_start, inf_start, inf_end, post_start = self._legacy.generate_accompaniment(
            melody_dicts,
            generation_start_tick=int(generation_start_tick),
            acc_notes=None,
            generation_length_frames=int(generation_length_frames),
            prompt_length_ticks=(int(prompt_length_ticks) if prompt_length_ticks is not None else None),
        )

        # Convert duration notes -> event stream
        events: List[MusicalEvent] = []
        for n in acc_notes:
            note = Note(
                pitch=int(n["pitch"]),
                tick=int(n["tick"]),
                duration=int(n["duration"]),
                velocity=int(n.get("velocity", 100)),
                channel=int(n.get("channel", 0)),
                program=int(n.get("program", 0)),
                is_placeholder=bool(n.get("is_placeholder", False)),
            )
            events.extend(note.to_events())

        timing = TimingInfo(
            request_arrival_time=0.0,
            response_output_time=0.0,
            preprocess_start_time=float(preprocess_start),
            inference_start_time=float(inf_start),
            inference_end_time=float(inf_end),
            postprocess_start_time=float(post_start),
        )
        return events, timing

    def inject_history(
        self,
        melody_events: List[MusicalEvent],
        accompaniment_events: List[MusicalEvent],
        injection_length_ticks: int,
    ) -> None:
        self.clear_history()
        self.set_injection_offset(int(injection_length_ticks))

        mel_notes = events_to_notes(melody_events, horizon_tick=int(injection_length_ticks))
        acc_notes = events_to_notes(accompaniment_events, horizon_tick=int(injection_length_ticks))

        # Populate legacy histories directly (duration-based).
        if hasattr(self._legacy, "melody_history"):
            self._legacy.melody_history.extend(
                [{"pitch": n.pitch, "tick": n.tick, "duration": n.duration, "velocity": n.velocity} for n in mel_notes]
            )
        if hasattr(self._legacy, "accompaniment_history"):
            self._legacy.accompaniment_history.extend(
                [
                    {"pitch": n.pitch, "tick": n.tick, "duration": n.duration, "velocity": n.velocity, "program": 1}
                    for n in acc_notes
                ]
            )

    def set_injection_offset(self, offset_ticks: int) -> None:
        self._legacy.set_injection_offset(int(offset_ticks))

    def clear_history(self) -> Dict[str, Any]:
        self._legacy.clear_history()
        return {
            "success": True,
            "message": "History cleared",
            "melody_history": [],
            "accompaniment_history": [],
        }

