"""Realtime client-side orchestration for Lekai prompt-continuation mode."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from streammuse.application.services.input_timing import (
    build_input_quantization_trace_row,
    diagnose_input_quantization,
    stamp_user_input_event_at_tick,
)
from streammuse.domain.interfaces import InputSource, OutputSink
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import MusicalTime, PlaybackScheduler, Tempo


EventKey = tuple[int, int, str, int, int, int]
ModelNoteKey = tuple[int, int, int]
NoteSpanKey = tuple[int, int, int, int, int]


class PromptContinuationClient(Protocol):
    def clear_history(self) -> dict[str, Any]: ...

    def start(
        self,
        *,
        melody_events: list[MusicalEvent],
        prompt_length_ticks: int,
        generation_interval_ticks: int,
        observed_until_tick: int,
        bpm: int,
    ) -> dict[str, Any]: ...

    def append_melody(
        self,
        *,
        melody_events: list[MusicalEvent],
        observed_until_tick: int,
    ) -> dict[str, Any]: ...

    def status(self) -> dict[str, Any]: ...

    def playable(self) -> tuple[list[MusicalEvent], dict[str, Any]]: ...


@dataclass(frozen=True)
class PromptContinuationRuntime:
    session_start_time: float
    timeline_start_time: float


@dataclass(frozen=True)
class _ControlAction:
    kind: str
    melody_events: list[MusicalEvent]
    observed_until_tick: int


@dataclass(frozen=True)
class _PlayableBatch:
    accompaniment: list[MusicalEvent]
    status: dict[str, Any]
    arrival_time_s: float | None = None


class PromptContinuationRealtimeService:
    """Drive prompt-continuation endpoints from a realtime input source.

    Intended first use: frontend/client reads a melody MIDI file via `MidiFileInput`,
    stamps it as if it were user input, and talks to the special backend protocol:
    start prompt generation after the prompt window, keep appending melody progress,
    poll until catch-up is ready, then schedule playable accompaniment.
    """

    _INPUT_BUFFER_RATIO = 0.1

    def __init__(
        self,
        *,
        input_source: InputSource,
        prompt_client: PromptContinuationClient,
        output_sink: OutputSink,
        tempo: Tempo,
        scheduler: PlaybackScheduler,
        prompt_length_ticks: int = 32,
        generation_interval_ticks: int = 4,
        count_in_beats: int = 0,
        input_snap_forward_fraction: float = 0.0,
        input_quantization_trace_enabled: bool = False,
        source_tick_input: bool = False,
        model_condition_bpm: int | None = None,
        protocol_poll_interval_s: float = 0.05,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if int(prompt_length_ticks) <= 0:
            raise ValueError("prompt_length_ticks must be > 0")
        if int(generation_interval_ticks) <= 0:
            raise ValueError("generation_interval_ticks must be > 0")
        if int(count_in_beats) < 0:
            raise ValueError("count_in_beats must be >= 0")
        if tempo.ticks_per_beat != 4:
            raise ValueError(
                "Prompt+Continuation requires exactly 4 steps per beat; "
                f"got {tempo.ticks_per_beat}"
            )
        self._input = input_source
        self._client = prompt_client
        self._output = output_sink
        self._tempo = tempo
        self._scheduler = scheduler
        self._prompt_length_ticks = int(prompt_length_ticks)
        self._generation_interval_ticks = int(generation_interval_ticks)
        self._count_in_beats = int(count_in_beats)
        self._count_in_ticks = self._count_in_beats * int(self._tempo.ticks_per_beat)
        self._input_snap_forward_fraction = float(input_snap_forward_fraction)
        self._input_quantization_trace_enabled = bool(
            input_quantization_trace_enabled
        )
        self._source_tick_reader: Callable[[int], list[MusicalEvent]] | None = None
        self._source_tick_prepare: Callable[[], None] | None = None
        if source_tick_input:
            source_tick_reader = getattr(input_source, "read_events_at_tick", None)
            if not callable(source_tick_reader):
                raise ValueError(
                    "source_tick_input requires an input source with "
                    "read_events_at_tick(tick)"
                )
            self._source_tick_reader = source_tick_reader
            source_tick_prepare = getattr(
                input_source, "prepare_source_tick_replay", None
            )
            if callable(source_tick_prepare):
                self._source_tick_prepare = source_tick_prepare
        self._model_condition_bpm = (
            int(model_condition_bpm)
            if model_condition_bpm is not None
            else int(round(float(tempo.bpm)))
        )
        if self._model_condition_bpm <= 0:
            raise ValueError("model_condition_bpm must be > 0")
        self._protocol_poll_interval_s = float(protocol_poll_interval_s)
        self._now = now
        self._sleep = sleep

        self._running = False
        self._runtime: PromptContinuationRuntime | None = None
        self._input_thread: threading.Thread | None = None
        self._tick_thread: threading.Thread | None = None
        self._protocol_thread: threading.Thread | None = None

        self._event_q: queue.Queue[MusicalEvent] = queue.Queue()
        self._control_q: queue.Queue[_ControlAction] = queue.Queue()
        self._playable_q: queue.Queue[object] = queue.Queue()

        self._prompt_events: list[MusicalEvent] = []
        self._pending_append_events: list[MusicalEvent] = []
        self._start_enqueued = False
        self._last_append_observed_tick = 0
        self._handled_model_event_counts: Counter[EventKey] = Counter()
        self._played_model_event_counts: Counter[EventKey] = Counter()
        self._active_model_note_keys: set[ModelNoteKey] = set()
        self._pending_late_note_off_keys: set[ModelNoteKey] = set()
        self._scheduled_model_note_counts: Counter[NoteSpanKey] = Counter()
        self._rehydrated_model_note_span_counts: Counter[NoteSpanKey] = Counter()
        self._append_generation = 0
        self._last_playable_marker: tuple[int, int, int] | None = None
        self._protocol_started = False
        self._append_sent_after_prompt = False
        self._trace_path = os.environ.get("LEKAI_PROMPT_CONTINUATION_TRACE_PATH")
        self._trace_lock = threading.Lock()
        raw_scheduling_mode = os.environ.get(
            "LEKAI_PROMPT_CONTINUATION_SCHEDULING_MODE",
            "streaming_events",
        ).strip().lower()
        if raw_scheduling_mode not in {"streaming_events", "paired_future_only"}:
            raw_scheduling_mode = "streaming_events"
        self._scheduling_mode = raw_scheduling_mode
        self._recover_late_events = os.environ.get(
            "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_EVENTS",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self._bound_late_recovery_env = self._env_optional_bool(
            "LEKAI_PROMPT_CONTINUATION_BOUND_LATE_RECOVERY"
        )
        self._recover_late_max_ticks = self._env_optional_int(
            "LEKAI_PROMPT_CONTINUATION_RECOVER_LATE_MAX_TICKS"
        )
        self._bound_late_recovery = bool(self._recover_late_max_ticks is not None)
        if self._bound_late_recovery_env is not None:
            self._bound_late_recovery = self._bound_late_recovery_env
        if self._recover_late_events and self._bound_late_recovery and self._recover_late_max_ticks is None:
            self._recover_late_max_ticks = self._generation_interval_ticks
        self._rehydrate_active_notes = os.environ.get(
            "LEKAI_PROMPT_CONTINUATION_REHYDRATE_ACTIVE_NOTES",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        self._system_trace_frames: dict[int, dict[str, Any]] = {}
        self._system_trace_coverage_end_tick = 0
        self._system_trace_playable_request_counter = 0
        system_trace_logger = getattr(output_sink, "log_system_trace", None)
        self._system_trace_logger = (
            system_trace_logger if callable(system_trace_logger) else None
        )
        input_quantization_logger = getattr(
            output_sink, "log_input_quantization", None
        )
        self._input_quantization_logger = (
            input_quantization_logger
            if self._input_quantization_trace_enabled
            and callable(input_quantization_logger)
            else None
        )
        self._input_quantization_sequence = 0
        replay_request_logger = getattr(
            output_sink, "log_prompt_continuation_replay_request", None
        )
        self._replay_request_logger = (
            replay_request_logger if callable(replay_request_logger) else None
        )
        self._replay_request_sequence = 0

    @staticmethod
    def _serialize_replay_event(event: MusicalEvent) -> dict[str, Any]:
        """Serialize exactly the event fields sent by the HTTP client."""
        payload: dict[str, Any] = {
            "type": event.event_type.value,
            "pitch": int(event.pitch),
            "tick": int(event.tick),
            "velocity": int(event.velocity),
            "channel": int(event.channel),
            "program": int(event.program),
        }
        if event.is_placeholder:
            payload["is_placeholder"] = True
        return payload

    def _log_replay_request(
        self,
        *,
        action: _ControlAction,
        acknowledgement: dict[str, Any] | None,
        error: Exception | None = None,
    ) -> None:
        if self._replay_request_logger is None:
            return
        self._replay_request_sequence += 1
        request: dict[str, Any] = {
            "melody_events": [
                self._serialize_replay_event(event)
                for event in action.melody_events
            ],
            "observed_until_tick": int(action.observed_until_tick),
        }
        if action.kind == "start":
            request.update(
                {
                    "prompt_length_ticks": self._prompt_length_ticks,
                    "generation_interval_ticks": self._generation_interval_ticks,
                    "bpm": self._model_condition_bpm,
                }
            )
        row: dict[str, Any] = {
            "schema_version": 1,
            "sequence": self._replay_request_sequence,
            "operation": action.kind,
            "request": request,
            "protocol_context": {
                "prompt_length_ticks": self._prompt_length_ticks,
                "generation_interval_ticks": self._generation_interval_ticks,
                "bpm": self._model_condition_bpm,
            },
            "acknowledgement": acknowledgement,
        }
        if error is not None:
            row["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        try:
            self._replay_request_logger(row)
        except Exception:
            # Audit logging must never change the realtime protocol.
            return

    def _trace(self, kind: str, **payload: Any) -> None:
        if not self._trace_path:
            return
        row = {"timestamp": self._now(), "kind": kind, **payload}
        try:
            with self._trace_lock:
                with open(self._trace_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            # Tracing must never affect realtime playback.
            return

    @staticmethod
    def _env_optional_int(name: str) -> int | None:
        value = os.environ.get(name)
        if value is None or str(value).strip() == "":
            return None
        try:
            return max(0, int(value))
        except ValueError:
            return None

    @staticmethod
    def _env_optional_bool(name: str) -> bool | None:
        value = os.environ.get(name)
        if value is None or str(value).strip() == "":
            return None
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def effective_model_bpm(self) -> int:
        return self._model_condition_bpm

    def _sleep_until(self, target_time: float) -> None:
        delay = target_time - self._now()
        if delay > 0:
            self._sleep(delay)

    def _output_metronome_tick(self, tick: int, bar: int, beat: int) -> None:
        if hasattr(self._output, "output_metronome_tick"):
            self._output.output_metronome_tick(tick, bar, beat)  # type: ignore[union-attr]

    def _run_count_in(self) -> None:
        """Play metronome-only pre-roll before the formal timeline starts."""
        if self._count_in_ticks <= 0:
            return
        assert self._runtime is not None
        start = self._runtime.session_start_time
        ticks_per_beat = max(1, int(self._tempo.ticks_per_beat))
        beats_per_bar = max(1, int(self._tempo.beats_per_bar))

        self._output.output_status("count_in", f"{self._count_in_beats} beat(s)")
        for elapsed_tick in range(self._count_in_ticks):
            if not self._running:
                return
            target_time = start + self._tempo.tick_to_seconds(elapsed_tick)
            self._sleep_until(target_time)

            count_tick = elapsed_tick - self._count_in_ticks
            beat = (elapsed_tick // ticks_per_beat) % beats_per_bar
            bar = elapsed_tick // (ticks_per_beat * beats_per_bar)
            self._output_metronome_tick(tick=count_tick, bar=bar, beat=beat)

    def _input_worker(self) -> None:
        assert self._runtime is not None
        start = self._runtime.timeline_start_time
        self._sleep_until(start)
        if not self._running:
            return
        for ev in self._input.read_events():
            if not self._running:
                break
            received_time_s = self._now()
            result = diagnose_input_quantization(
                max(0.0, received_time_s - start),
                tempo=self._tempo,
                snap_forward_fraction=self._input_snap_forward_fraction,
            )
            stamped = stamp_user_input_event_at_tick(
                ev,
                tick=result.quantized_tick,
            )
            self._event_q.put(stamped)
            if self._input_quantization_logger is not None:
                self._input_quantization_sequence += 1
                try:
                    self._input_quantization_logger(
                        build_input_quantization_trace_row(
                            service="prompt_continuation",
                            event_sequence=self._input_quantization_sequence,
                            event=ev,
                            result=result,
                            received_time_s=received_time_s,
                            timeline_start_time_s=start,
                            clock_domain="service_now",
                            tempo=self._tempo,
                        )
                    )
                except Exception:
                    # Optional diagnostics must never interrupt realtime input.
                    pass

    def _protocol_worker(self) -> None:
        assert self._runtime is not None
        self._sleep_until(self._runtime.timeline_start_time)
        if not self._running:
            return
        try:
            self._client.clear_history()
        except Exception as exc:
            self._output.output_status("error", f"Prompt-continuation clear_history failed: {exc}")

        while self._running:
            try:
                action = self._control_q.get(timeout=self._protocol_poll_interval_s)
            except queue.Empty:
                action = None

            if action is not None:
                try:
                    if action.kind == "start":
                        self._trace(
                            "start_send",
                            observed_until_tick=action.observed_until_tick,
                            melody_event_count=len(action.melody_events),
                            prompt_length_ticks=self._prompt_length_ticks,
                            generation_interval_ticks=self._generation_interval_ticks,
                            effective_bpm=self._model_condition_bpm,
                        )
                        try:
                            acknowledgement = self._client.start(
                                melody_events=action.melody_events,
                                prompt_length_ticks=self._prompt_length_ticks,
                                generation_interval_ticks=self._generation_interval_ticks,
                                observed_until_tick=action.observed_until_tick,
                                bpm=self._model_condition_bpm,
                            )
                        except Exception as exc:
                            self._log_replay_request(
                                action=action,
                                acknowledgement=None,
                                error=exc,
                            )
                            raise
                        self._log_replay_request(
                            action=action,
                            acknowledgement=acknowledgement,
                        )
                        self._protocol_started = True
                        self._output.output_status("prompt_running", "Prompt-continuation start sent")
                    elif action.kind == "append":
                        self._trace(
                            "append_send",
                            observed_until_tick=action.observed_until_tick,
                            melody_event_count=len(action.melody_events),
                        )
                        try:
                            acknowledgement = self._client.append_melody(
                                melody_events=action.melody_events,
                                observed_until_tick=action.observed_until_tick,
                            )
                        except Exception as exc:
                            self._log_replay_request(
                                action=action,
                                acknowledgement=None,
                                error=exc,
                            )
                            raise
                        self._log_replay_request(
                            action=action,
                            acknowledgement=acknowledgement,
                        )
                        if int(action.observed_until_tick) > self._prompt_length_ticks:
                            self._append_sent_after_prompt = True
                            self._append_generation += 1
                    else:
                        self._output.output_status("error", f"Unknown prompt-continuation action: {action.kind}")
                except Exception as exc:
                    self._output.output_status("error", f"Prompt-continuation {action.kind} failed: {exc}")

            if self._protocol_started and self._append_sent_after_prompt:
                try:
                    status = self._client.status()
                    self._trace(
                        "status",
                        phase=status.get("phase"),
                        is_running=status.get("is_running"),
                        is_playback_ready=status.get("is_playback_ready"),
                        melody_history_beats=status.get("melody_history_beats"),
                        accompaniment_history_beats=status.get("accompaniment_history_beats"),
                        target_playable_accompaniment_beats=status.get("target_playable_accompaniment_beats"),
                        beats_needed_for_playback=status.get("beats_needed_for_playback"),
                        continuation_calls=status.get("continuation_calls"),
                        accompaniment_event_count=status.get("accompaniment_event_count"),
                        last_continuation_event_count=status.get("last_continuation_event_count"),
                        last_continuation_note_on_count=status.get("last_continuation_note_on_count"),
                        last_continuation_min_tick=status.get("last_continuation_min_tick"),
                        last_continuation_max_tick=status.get("last_continuation_max_tick"),
                        empty_continuation_output_streak=status.get("empty_continuation_output_streak"),
                    )
                    if status.get("is_failed"):
                        self._output.output_status("error", f"Prompt-continuation failed: {status.get('error')}")
                    elif status.get("is_playback_ready"):
                        marker = (
                            int(self._append_generation),
                            int(status.get("accompaniment_event_count", 0) or 0),
                            int(status.get("continuation_calls", 0) or 0),
                        )
                        if marker != self._last_playable_marker:
                            accompaniment, playable_status = self._client.playable()
                            arrival_time_s = None
                            if self._system_trace_logger is not None:
                                arrival_time_s = float(self._now())
                                self._record_playable_availability(
                                    playable_status,
                                    availability_time_s=arrival_time_s,
                                )
                            self._trace(
                                "playable_fetch",
                                accompaniment_event_count=len(accompaniment),
                                status_phase=playable_status.get("phase"),
                                status_melody_history_beats=playable_status.get("melody_history_beats"),
                                status_accompaniment_history_beats=playable_status.get("accompaniment_history_beats"),
                                status_beats_needed_for_playback=playable_status.get("beats_needed_for_playback"),
                                status_continuation_calls=playable_status.get("continuation_calls"),
                                status_last_continuation_event_count=playable_status.get(
                                    "last_continuation_event_count"
                                ),
                                status_last_continuation_note_on_count=playable_status.get(
                                    "last_continuation_note_on_count"
                                ),
                                status_empty_continuation_output_streak=playable_status.get(
                                    "empty_continuation_output_streak"
                                ),
                                playable_representation_match=playable_status.get(
                                    "playable_representation_match"
                                ),
                                server_playable_representation=playable_status.get(
                                    "server_playable_representation"
                                ),
                                client_playable_representation=playable_status.get(
                                    "client_playable_representation"
                                ),
                            )
                            self._playable_q.put(
                                _PlayableBatch(
                                    accompaniment=list(accompaniment),
                                    status=dict(playable_status),
                                    arrival_time_s=arrival_time_s,
                                )
                            )
                            self._last_playable_marker = marker
                            self._output.output_status("ready", "Prompt-continuation accompaniment is playable")
                except Exception as exc:
                    self._output.output_status("error", f"Prompt-continuation status/playable failed: {exc}")
                    self._sleep(self._protocol_poll_interval_s)

    def _drain_user_events(self) -> list[MusicalEvent]:
        drained: list[MusicalEvent] = []
        while True:
            try:
                event = self._event_q.get_nowait()
            except queue.Empty:
                break
            drained.append(event)
            self._output.output_event(event, source="user")
            if int(event.tick) < self._prompt_length_ticks:
                self._prompt_events.append(event)
            else:
                self._pending_append_events.append(event)
        return drained

    def _enqueue_source_tick_events(self, tick: int) -> None:
        if self._source_tick_reader is None:
            return
        for event in self._source_tick_reader(int(tick)):
            if int(event.tick) != int(tick):
                raise ValueError(
                    "source-tick input returned an event outside the requested "
                    f"tick: requested={tick}, event={event.tick}"
                )
            self._event_q.put(
                stamp_user_input_event_at_tick(event, tick=int(event.tick))
            )

    def _maybe_enqueue_start(self, observed_until_tick: int) -> None:
        if self._start_enqueued:
            return
        if int(observed_until_tick) < self._prompt_length_ticks:
            return
        self._control_q.put(
            _ControlAction(
                kind="start",
                melody_events=list(self._prompt_events),
                observed_until_tick=self._prompt_length_ticks,
            )
        )
        self._start_enqueued = True
        self._last_append_observed_tick = self._prompt_length_ticks

    def _maybe_enqueue_append(self, observed_until_tick: int) -> None:
        if not self._start_enqueued:
            return
        observed_until_tick = int(observed_until_tick)
        if observed_until_tick <= self._last_append_observed_tick:
            return
        if (observed_until_tick - self._prompt_length_ticks) % self._generation_interval_ticks != 0:
            return
        self._control_q.put(
            _ControlAction(
                kind="append",
                melody_events=list(self._pending_append_events),
                observed_until_tick=observed_until_tick,
            )
        )
        self._pending_append_events = []
        self._last_append_observed_tick = observed_until_tick

    def _schedule_playable(
        self,
        accompaniment: list[MusicalEvent],
        *,
        current_tick: int,
        arrival_time_s: float | None = None,
    ) -> None:
        if self._scheduling_mode == "paired_future_only":
            self._schedule_playable_paired_future_only(
                accompaniment,
                current_tick=int(current_tick),
                arrival_time_s=arrival_time_s,
            )
            return
        self._schedule_playable_streaming_events(
            accompaniment,
            current_tick=int(current_tick),
            arrival_time_s=arrival_time_s,
        )

    def _schedule_playable_paired_future_only(
        self,
        accompaniment: list[MusicalEvent],
        *,
        current_tick: int,
        arrival_time_s: float | None = None,
    ) -> None:
        input_tick_stats = self._event_tick_stats(accompaniment, current_tick=int(current_tick))
        scheduled = 0
        scheduled_notes = 0
        dropped_past = 0
        skipped_duplicate = 0
        clipped_sustains = 0
        skipped_unpaired = 0

        if self._system_trace_logger is not None:
            for event in accompaniment:
                if (event.is_placeholder or event.pitch == -1) and int(event.tick) >= int(
                    current_tick
                ):
                    self._record_system_trace_event(
                        self._to_model_event(event, current_tick=int(current_tick)),
                        scheduled_tick=int(event.tick),
                        logical_tick=int(event.tick),
                        arrival_time_s=arrival_time_s,
                        action="scheduled",
                        policy="future_placeholder",
                    )

        events_to_schedule, dropped_past, clipped_sustains = self._clip_playable_to_current_tick(
            accompaniment,
            current_tick=int(current_tick),
        )

        note_pairs, skipped_unpaired = self._pair_playable_events(events_to_schedule)
        seen_note_counts: Counter[NoteSpanKey] = Counter()
        for note_on, note_off in note_pairs:
            note_key = self._note_span_key(note_on, note_off)
            seen_note_counts[note_key] += 1
            occurrence = seen_note_counts[note_key]
            if self._scheduled_model_note_counts[note_key] >= occurrence:
                skipped_duplicate += 1
                continue
            scheduled_notes += 1
            for event in (note_on, note_off):
                model_event = self._to_model_event(event, current_tick=int(current_tick))
                self._scheduler.schedule(model_event, int(event.tick))
                self._record_system_trace_event(
                    model_event,
                    scheduled_tick=int(event.tick),
                    logical_tick=int(event.tick),
                    arrival_time_s=arrival_time_s,
                    action="scheduled",
                    policy="paired_future_only",
                )
                self._mark_event_scheduled(event)
                scheduled += 1
            self._ensure_count(self._scheduled_model_note_counts, note_key, occurrence)
        self._output.output_status(
            "ready",
            f"Scheduled {scheduled} playable accompaniment event(s); "
            f"dropped {dropped_past} past event(s); "
            f"clipped {clipped_sustains} sustaining note(s); "
            f"skipped {skipped_duplicate} duplicate note(s); "
            f"skipped {skipped_unpaired} unpaired event(s).",
        )
        self._trace(
            "schedule_playable",
            current_tick=int(current_tick),
            mode="paired_future_only",
            input_event_count=len(accompaniment),
            **input_tick_stats,
            pair_count=len(note_pairs),
            scheduled_note_count=scheduled_notes,
            scheduled_event_count=scheduled,
            dropped_past=dropped_past,
            clipped_sustains=clipped_sustains,
            skipped_duplicate=skipped_duplicate,
            skipped_unpaired=skipped_unpaired,
        )

    def _schedule_playable_streaming_events(
        self,
        accompaniment: list[MusicalEvent],
        *,
        current_tick: int,
        arrival_time_s: float | None = None,
    ) -> None:
        """Schedule playable history as a streaming event log."""
        current_tick = int(current_tick)
        input_tick_stats = self._event_tick_stats(accompaniment, current_tick=current_tick)
        scheduled = 0
        skipped_duplicate = 0
        late_event_count = 0
        recovered_late_event_count = 0
        dropped_past = 0
        dropped_too_late_note_on = 0
        closed_late_active_note_off = 0
        skipped_pending_late_note_off = 0
        dropped_orphan_late_note_off = 0
        rehydrated_note_count = 0
        placeholder_count = 0
        usable_events: list[MusicalEvent] = []
        for event in accompaniment:
            if event.is_placeholder or event.pitch == -1:
                placeholder_count += 1
                if self._system_trace_logger is not None and int(event.tick) >= current_tick:
                    self._record_system_trace_event(
                        self._to_model_event(event, current_tick=current_tick),
                        scheduled_tick=int(event.tick),
                        logical_tick=int(event.tick),
                        arrival_time_s=arrival_time_s,
                        action="scheduled",
                        policy="future_placeholder",
                    )
                continue
            usable_events.append(event)

        usable_events = sorted(
            usable_events,
            key=lambda ev: (
                int(ev.tick),
                0 if ev.event_type == EventType.NOTE_OFF else 1,
                int(ev.pitch),
                int(ev.channel),
                int(ev.program),
            ),
        )
        late_event_count = sum(1 for event in usable_events if int(event.tick) < current_tick)

        rehydrated_events: list[MusicalEvent] = []
        consumed_original_note_on_counts: Counter[EventKey] = Counter()
        if self._rehydrate_active_notes:
            rehydrated_events, consumed_original_note_on_counts = self._rehydrate_sustaining_notes(
                usable_events,
                current_tick=current_tick,
            )
            rehydrated_note_count = len(rehydrated_events)
            for key, count in consumed_original_note_on_counts.items():
                self._ensure_count(self._handled_model_event_counts, key, count)
                self._ensure_count(self._played_model_event_counts, key, count)

        seen_in_payload: Counter[EventKey] = Counter()
        for event in [*rehydrated_events, *usable_events]:
            event_key = self._event_key(event)
            seen_in_payload[event_key] += 1
            occurrence = seen_in_payload[event_key]
            if self._handled_model_event_counts[event_key] >= occurrence:
                skipped_duplicate += 1
                continue

            event_tick = int(event.tick)
            schedule_tick = event_tick
            event_to_schedule = event
            policy = "future_event"
            if event_tick < current_tick:
                if not self._recover_late_events:
                    if event.event_type == EventType.NOTE_OFF:
                        model_key = self._model_event_key(event)
                        if model_key in self._active_model_note_keys:
                            if model_key in self._pending_late_note_off_keys:
                                skipped_pending_late_note_off += 1
                                self._ensure_count(
                                    self._handled_model_event_counts,
                                    event_key,
                                    occurrence,
                                )
                                continue
                            self._pending_late_note_off_keys.add(model_key)
                            schedule_tick = current_tick
                            event_to_schedule = self._clone_event_at_tick(event, current_tick)
                            policy = "late_active_note_off"
                            closed_late_active_note_off += 1
                        else:
                            dropped_past += 1
                            dropped_orphan_late_note_off += 1
                            self._ensure_count(
                                self._handled_model_event_counts,
                                event_key,
                                occurrence,
                            )
                            continue
                    else:
                        dropped_past += 1
                        self._ensure_count(
                            self._handled_model_event_counts,
                            event_key,
                            occurrence,
                        )
                        continue
                elif self._would_drop_late_note_on(event, current_tick=current_tick):
                    dropped_too_late_note_on += 1
                    self._ensure_count(self._handled_model_event_counts, event_key, occurrence)
                    continue
                else:
                    schedule_tick = current_tick
                    policy = "recovered_late_event"
                    recovered_late_event_count += 1

            model_event = self._to_model_event(event_to_schedule, current_tick=current_tick)
            self._scheduler.schedule(model_event, schedule_tick)
            self._record_system_trace_event(
                model_event,
                scheduled_tick=schedule_tick,
                logical_tick=event_tick,
                arrival_time_s=arrival_time_s,
                action="scheduled",
                policy=policy,
            )
            self._ensure_count(self._handled_model_event_counts, event_key, occurrence)
            self._ensure_count(self._played_model_event_counts, event_key, occurrence)
            scheduled += 1

        self._output.output_status(
            "ready",
            f"Scheduled {scheduled} playable accompaniment event(s); "
            f"recovered {recovered_late_event_count} late event(s); "
            f"dropped {dropped_past} past event(s); "
            f"dropped {dropped_too_late_note_on} too-late note_on event(s); "
            f"closed {closed_late_active_note_off} active note(s) from late note_off; "
            f"skipped {skipped_pending_late_note_off} pending late note_off event(s); "
            f"dropped {dropped_orphan_late_note_off} orphan late note_off event(s); "
            f"rehydrated {rehydrated_note_count} active note(s); "
            f"skipped {skipped_duplicate} duplicate event(s); "
            f"skipped {placeholder_count} placeholder event(s).",
        )
        self._trace(
            "schedule_playable",
            current_tick=current_tick,
            mode="streaming_events",
            input_event_count=len(accompaniment),
            **input_tick_stats,
            scheduled_event_count=scheduled,
            late_event_count=late_event_count,
            recovered_late_event_count=recovered_late_event_count,
            dropped_past=dropped_past,
            dropped_too_late_note_on=dropped_too_late_note_on,
            rehydrated_note_count=rehydrated_note_count,
            rehydrate_active_notes=self._rehydrate_active_notes,
            bound_late_recovery=self._bound_late_recovery,
            recover_late_max_ticks=self._recover_late_max_ticks,
            skipped_duplicate=skipped_duplicate,
            placeholder_count=placeholder_count,
        )

    @staticmethod
    def _ensure_count(counter: Counter[Any], key: Any, count: int) -> None:
        if counter[key] < int(count):
            counter[key] = int(count)

    def _mark_event_scheduled(self, event: MusicalEvent) -> None:
        key = self._event_key(event)
        self._handled_model_event_counts[key] += 1
        self._played_model_event_counts[key] += 1

    @staticmethod
    def _is_model_note_on(event: MusicalEvent) -> bool:
        return (
            event.source == "model"
            and not event.is_placeholder
            and int(event.pitch) != -1
            and event.event_type == EventType.NOTE_ON
            and int(event.velocity) > 0
        )

    @staticmethod
    def _is_explicit_rest(event: MusicalEvent) -> bool:
        return event.source == "model" and (event.is_placeholder or int(event.pitch) == -1)

    def _record_system_trace_event(
        self,
        event: MusicalEvent,
        *,
        scheduled_tick: int,
        logical_tick: int,
        arrival_time_s: float | None,
        action: str,
        policy: str,
    ) -> None:
        if self._system_trace_logger is None:
            return
        if not self._is_model_note_on(event) and not self._is_explicit_rest(event):
            return

        frame = self._system_trace_frames.setdefault(
            int(scheduled_tick),
            {
                "emitted_model_note_on_count": 0,
                "explicit_rest": False,
                "note_provenance": None,
                "rest_provenance": None,
            },
        )
        provenance: dict[str, Any] = {
            "arrival_time_s": arrival_time_s,
            "logical_tick": int(logical_tick),
            "scheduled_tick": int(scheduled_tick),
            "generation_start_tick": None,
            "request_id": None,
            "action": str(action),
            "policy": str(policy),
        }
        if self._is_model_note_on(event):
            frame["emitted_model_note_on_count"] = int(frame["emitted_model_note_on_count"]) + 1
            if frame["note_provenance"] is None:
                frame["note_provenance"] = provenance
            return

        frame["explicit_rest"] = True
        if frame["rest_provenance"] is None:
            frame["rest_provenance"] = provenance

    def _log_system_trace_availability_span(
        self,
        *,
        start_tick: int,
        end_tick_exclusive: int,
        availability_time_s: float,
        generation_start_tick: int,
        request_id: str,
        source_stage: str,
    ) -> None:
        logger = self._system_trace_logger
        if logger is None:
            return
        start_tick = int(start_tick)
        end_tick_exclusive = int(end_tick_exclusive)
        if start_tick >= end_tick_exclusive:
            raise ValueError(
                "availability span must satisfy start_tick < end_tick_exclusive"
            )
        row: dict[str, Any] = {
            "schema_version": 2,
            "record_type": "availability_span",
            "mode": "realtime",
            "condition": "prompt_continuation",
            "clock_domain": "service_now",
            "start_tick": start_tick,
            "end_tick_exclusive": end_tick_exclusive,
            "availability_time_s": float(availability_time_s),
            "generation_start_tick": int(generation_start_tick),
            "request_id": str(request_id),
            "source_stage": str(source_stage),
        }
        try:
            logger(row)
        except Exception:
            return

    def _record_playable_availability(
        self,
        playable_status: dict[str, Any],
        *,
        availability_time_s: float,
    ) -> None:
        if self._system_trace_logger is None:
            return
        coverage_beats = playable_status.get("accompaniment_history_beats")
        if coverage_beats is None:
            return
        try:
            coverage_end_tick = int(coverage_beats) * int(self._tempo.ticks_per_beat)
        except (TypeError, ValueError):
            return

        coverage_start_tick = int(self._system_trace_coverage_end_tick)
        if coverage_end_tick <= coverage_start_tick:
            return

        self._system_trace_playable_request_counter += 1
        request_id_value = playable_status.get("request_id")
        request_id = (
            str(request_id_value)
            if request_id_value is not None and str(request_id_value).strip()
            else f"playable-{self._system_trace_playable_request_counter:04d}"
        )
        prompt_end_tick = int(self._prompt_length_ticks)
        segments = (
            (
                coverage_start_tick,
                min(coverage_end_tick, prompt_end_tick),
                "prompt",
            ),
            (
                max(coverage_start_tick, prompt_end_tick),
                coverage_end_tick,
                "continuation",
            ),
        )
        for start_tick, end_tick_exclusive, source_stage in segments:
            if start_tick >= end_tick_exclusive:
                continue
            self._log_system_trace_availability_span(
                start_tick=start_tick,
                end_tick_exclusive=end_tick_exclusive,
                availability_time_s=availability_time_s,
                generation_start_tick=start_tick,
                request_id=request_id,
                source_stage=source_stage,
            )
        self._system_trace_coverage_end_tick = coverage_end_tick

    def _emit_system_trace_frame(
        self,
        *,
        tick: int,
        nominal_tick_time_s: float,
        deadline_time_s: float,
        model_events: list[MusicalEvent],
    ) -> None:
        logger = self._system_trace_logger
        if logger is None:
            return
        frame = self._system_trace_frames.pop(
            int(tick),
            {
                "emitted_model_note_on_count": 0,
                "explicit_rest": False,
                "note_provenance": None,
                "rest_provenance": None,
            },
        )
        emitted_model_note_on_count = sum(
            1 for event in model_events if self._is_model_note_on(event)
        )
        explicit_rest = bool(frame["explicit_rest"]) or any(
            self._is_explicit_rest(event) for event in model_events
        )
        if emitted_model_note_on_count > 0:
            decision = "note"
            provenance = frame["note_provenance"]
        elif explicit_rest:
            decision = "rest"
            provenance = frame["rest_provenance"]
        else:
            decision = "missing"
            provenance = None

        if not isinstance(provenance, dict):
            provenance = {}
        arrival_time_s = provenance.get("arrival_time_s")
        row: dict[str, Any] = {
            "schema_version": 2,
            "record_type": "frame_deadline",
            "mode": "realtime",
            "condition": "prompt_continuation",
            "clock_domain": "service_now",
            "tick": int(tick),
            "nominal_tick_time_s": float(nominal_tick_time_s),
            "deadline_time_s": float(deadline_time_s),
            "decision": decision,
            "arrived_by_deadline": bool(
                arrival_time_s is not None and float(arrival_time_s) <= float(deadline_time_s)
            ),
            "arrival_time_s": (
                float(arrival_time_s) if arrival_time_s is not None else None
            ),
            "logical_tick": provenance.get("logical_tick"),
            "scheduled_tick": provenance.get("scheduled_tick"),
            "generation_start_tick": provenance.get("generation_start_tick"),
            "request_id": provenance.get("request_id"),
            "action": provenance.get("action"),
            "policy": provenance.get("policy"),
            "emitted_model_note_on_count": int(emitted_model_note_on_count),
            "explicit_rest": bool(explicit_rest),
            "observed_emit_time_s": float(self._now()),
        }
        try:
            logger(row)
        except Exception:
            return

    def _to_model_event(self, event: MusicalEvent, *, current_tick: int) -> MusicalEvent:
        return MusicalEvent(
            tick=event.tick,
            pitch=event.pitch,
            event_type=event.event_type,
            velocity=event.velocity,
            channel=event.channel,
            program=event.program,
            is_placeholder=event.is_placeholder,
            source="model",
            backup_level=max(0, int(event.tick) - int(current_tick)),
        )

    @staticmethod
    def _model_event_key(event: MusicalEvent) -> ModelNoteKey:
        return (int(event.pitch), int(event.channel), int(event.program))

    @staticmethod
    def _emission_order_key(event: MusicalEvent) -> tuple[int, int, int, int, int]:
        event_priority = 0 if event.event_type == EventType.NOTE_OFF else 1
        return (
            event_priority,
            int(event.pitch),
            int(event.channel),
            int(event.program),
            int(event.tick),
        )

    def _observe_model_output_event(self, event: MusicalEvent) -> None:
        if event.is_placeholder or int(event.pitch) == -1:
            return
        key = self._model_event_key(event)
        if event.event_type == EventType.NOTE_ON and int(event.velocity) > 0:
            self._active_model_note_keys.add(key)
        elif event.event_type == EventType.NOTE_OFF:
            self._active_model_note_keys.discard(key)
            self._pending_late_note_off_keys.discard(key)

    def _output_model_event(self, event: MusicalEvent) -> None:
        self._output.output_event(event, source=event.source)
        self._observe_model_output_event(event)

    def _would_drop_late_note_on(self, event: MusicalEvent, *, current_tick: int) -> bool:
        return (
            self._bound_late_recovery
            and self._recover_late_max_ticks is not None
            and int(event.tick) < int(current_tick)
            and int(current_tick) - int(event.tick) > self._recover_late_max_ticks
            and event.event_type == EventType.NOTE_ON
            and int(event.velocity) > 0
        )

    @staticmethod
    def _event_key(event: MusicalEvent) -> EventKey:
        return (
            int(event.tick),
            int(event.pitch),
            str(event.event_type.value),
            int(event.velocity),
            int(event.channel),
            int(event.program),
        )

    @staticmethod
    def _note_identity(event: MusicalEvent) -> tuple[int, int, int]:
        return (int(event.pitch), int(event.channel), int(event.program))

    @staticmethod
    def _note_span_key(
        note_on: MusicalEvent,
        note_off: MusicalEvent,
    ) -> NoteSpanKey:
        return (
            int(note_on.tick),
            int(note_off.tick),
            int(note_on.pitch),
            int(note_on.channel),
            int(note_on.program),
        )

    def _rehydrate_sustaining_notes(
        self,
        events: list[MusicalEvent],
        *,
        current_tick: int,
    ) -> tuple[list[MusicalEvent], Counter[EventKey]]:
        """Clone sustaining note_on events that would otherwise be dropped as too late."""
        current_tick = int(current_tick)
        active: dict[tuple[int, int, int], list[tuple[MusicalEvent, int]]] = {}
        note_on_occurrences: Counter[EventKey] = Counter()
        rehydrated: list[MusicalEvent] = []
        consumed_original_note_on_counts: Counter[EventKey] = Counter()
        seen_span_counts: Counter[NoteSpanKey] = Counter()

        for event in events:
            key = self._note_identity(event)
            event_key = self._event_key(event)
            event_tick = int(event.tick)
            if event.event_type == EventType.NOTE_ON and int(event.velocity) > 0:
                note_on_occurrences[event_key] += 1
                active.setdefault(key, []).append((event, note_on_occurrences[event_key]))
                continue
            if event.event_type != EventType.NOTE_OFF:
                continue
            if not active.get(key):
                continue
            note_on, note_on_occurrence = active[key].pop(0)
            if not active[key]:
                active.pop(key, None)
            if not (int(note_on.tick) < current_tick < event_tick):
                continue
            if not self._would_drop_late_note_on(note_on, current_tick=current_tick):
                continue
            span_key = self._note_span_key(note_on, event)
            seen_span_counts[span_key] += 1
            span_occurrence = seen_span_counts[span_key]
            if self._rehydrated_model_note_span_counts[span_key] >= span_occurrence:
                continue
            note_on_key = self._event_key(note_on)
            if self._played_model_event_counts[note_on_key] >= note_on_occurrence:
                continue
            rehydrated.append(self._clone_event_at_tick(note_on, current_tick))
            consumed_original_note_on_counts[note_on_key] = max(
                consumed_original_note_on_counts[note_on_key],
                note_on_occurrence,
            )
            self._ensure_count(self._rehydrated_model_note_span_counts, span_key, span_occurrence)

        return rehydrated, consumed_original_note_on_counts

    def _active_notes_at_current_tick(
        self,
        accompaniment: list[MusicalEvent],
        *,
        current_tick: int,
    ) -> list[tuple[MusicalEvent, MusicalEvent]]:
        current_tick = int(current_tick)
        sorted_events = sorted(
            accompaniment,
            key=lambda event: (
                int(event.tick),
                0 if event.event_type == EventType.NOTE_OFF else 1,
            ),
        )
        active: dict[tuple[int, int, int], list[MusicalEvent]] = {}
        rehydratable: list[tuple[MusicalEvent, MusicalEvent]] = []

        for event in sorted_events:
            if event.is_placeholder or event.pitch == -1:
                continue
            key = self._note_identity(event)
            event_tick = int(event.tick)
            if event.event_type == EventType.NOTE_ON and int(event.velocity) > 0:
                active.setdefault(key, []).append(event)
                continue
            if event.event_type != EventType.NOTE_OFF:
                continue
            if not active.get(key):
                continue
            note_on = active[key].pop(0)
            if not active[key]:
                active.pop(key, None)
            if int(note_on.tick) < current_tick < event_tick:
                rehydratable.append((note_on, event))

        return rehydratable

    @staticmethod
    def _event_tick_stats(events: list[MusicalEvent], *, current_tick: int) -> dict[str, int | None]:
        usable_events = [
            event
            for event in events
            if not event.is_placeholder and event.pitch != -1
        ]
        if not usable_events:
            return {
                "input_min_tick": None,
                "input_max_tick": None,
                "future_event_count": 0,
                "future_note_on_count": 0,
                "past_event_count": 0,
            }
        current_tick = int(current_tick)
        return {
            "input_min_tick": min(int(event.tick) for event in usable_events),
            "input_max_tick": max(int(event.tick) for event in usable_events),
            "future_event_count": sum(1 for event in usable_events if int(event.tick) >= current_tick),
            "future_note_on_count": sum(
                1
                for event in usable_events
                if int(event.tick) >= current_tick
                and event.event_type == EventType.NOTE_ON
                and int(event.velocity) > 0
            ),
            "past_event_count": sum(1 for event in usable_events if int(event.tick) < current_tick),
        }

    @staticmethod
    def _clone_event_at_tick(event: MusicalEvent, tick: int) -> MusicalEvent:
        return MusicalEvent(
            tick=int(tick),
            pitch=event.pitch,
            event_type=event.event_type,
            velocity=event.velocity,
            channel=event.channel,
            program=event.program,
            is_placeholder=event.is_placeholder,
            source=event.source,
            backup_level=event.backup_level,
        )

    def _clip_playable_to_current_tick(
        self,
        accompaniment: list[MusicalEvent],
        *,
        current_tick: int,
    ) -> tuple[list[MusicalEvent], int, int]:
        """Drop fully-past notes, but preserve notes sustaining into now.

        The backend returns the full generated accompaniment history. A realtime
        player cannot go back and play notes that ended in the past, but if a
        note started before `current_tick` and its note_off is still in the
        future, the audible output should retrigger it at `current_tick` instead
        of losing the whole sustaining note.
        """
        current_tick = int(current_tick)
        sorted_events = sorted(
            accompaniment,
            key=lambda event: (
                int(event.tick),
                0 if event.event_type == EventType.NOTE_OFF else 1,
            ),
        )
        active: dict[int, list[MusicalEvent]] = {}
        scheduled: list[MusicalEvent] = []
        dropped_past = 0
        clipped_sustains = 0

        for event in sorted_events:
            if event.is_placeholder or event.pitch == -1:
                continue

            pitch = int(event.pitch)
            event_tick = int(event.tick)
            if event.event_type == EventType.NOTE_ON and event.velocity > 0:
                active.setdefault(pitch, []).append(event)
                continue

            if event.event_type != EventType.NOTE_OFF:
                continue

            if not active.get(pitch):
                if event_tick < current_tick:
                    dropped_past += 1
                continue

            note_on = active[pitch].pop(0)
            if not active[pitch]:
                active.pop(pitch, None)

            note_on_tick = int(note_on.tick)
            note_off_tick = event_tick
            if note_off_tick <= current_tick:
                dropped_past += 2
                continue

            if note_on_tick < current_tick:
                scheduled.append(self._clone_event_at_tick(note_on, current_tick))
                scheduled.append(event)
                clipped_sustains += 1
                continue

            scheduled.append(note_on)
            scheduled.append(event)

        for pitch_events in active.values():
            for note_on in pitch_events:
                if int(note_on.tick) >= current_tick:
                    scheduled.append(note_on)
                else:
                    dropped_past += 1

        return scheduled, dropped_past, clipped_sustains

    def _pair_playable_events(
        self,
        events: list[MusicalEvent],
    ) -> tuple[list[tuple[MusicalEvent, MusicalEvent]], int]:
        sorted_events = sorted(
            events,
            key=lambda event: (
                int(event.tick),
                0 if event.event_type == EventType.NOTE_OFF else 1,
            ),
        )
        active: dict[int, list[MusicalEvent]] = {}
        pairs: list[tuple[MusicalEvent, MusicalEvent]] = []
        skipped_unpaired = 0
        for event in sorted_events:
            if event.is_placeholder or event.pitch == -1:
                continue
            pitch = int(event.pitch)
            if event.event_type == EventType.NOTE_ON and event.velocity > 0:
                active.setdefault(pitch, []).append(event)
                continue
            if event.event_type != EventType.NOTE_OFF:
                continue
            if not active.get(pitch):
                skipped_unpaired += 1
                continue
            note_on = active[pitch].pop(0)
            if not active[pitch]:
                active.pop(pitch, None)
            if int(event.tick) > int(note_on.tick):
                pairs.append((note_on, event))
            else:
                skipped_unpaired += 2
        skipped_unpaired += sum(len(pitch_events) for pitch_events in active.values())
        return pairs, skipped_unpaired

    @staticmethod
    def _normalize_playable_item(item: object) -> _PlayableBatch:
        if isinstance(item, _PlayableBatch):
            return item
        if isinstance(item, tuple) and len(item) == 2:
            accompaniment, status = item
            return _PlayableBatch(
                accompaniment=list(accompaniment),
                status=dict(status),
                arrival_time_s=None,
            )
        raise TypeError(f"Unsupported playable item: {type(item).__name__}")

    def _tick_loop(self, *, max_ticks: int | None) -> None:
        assert self._runtime is not None
        self._run_count_in()
        start = self._runtime.timeline_start_time
        tick = 0
        while self._running:
            if max_ticks is not None and tick >= max_ticks:
                self._running = False
                break

            target_time = start + self._tempo.tick_to_seconds(tick)
            deadline_time = target_time + self._tempo.seconds_per_tick * self._INPUT_BUFFER_RATIO
            delay = target_time - self._now()
            if delay > 0:
                self._sleep(delay)

            mt = MusicalTime.from_tick(tick, self._tempo)
            self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)
            self._sleep(self._tempo.seconds_per_tick * self._INPUT_BUFFER_RATIO)

            self._enqueue_source_tick_events(tick)
            self._drain_user_events()
            observed_until_tick = tick + 1
            # The prompt window [0, prompt_length_ticks) is not fully observed
            # until the wall clock reaches its exclusive end boundary.
            self._maybe_enqueue_start(tick)
            self._maybe_enqueue_append(observed_until_tick)

            while True:
                try:
                    playable = self._normalize_playable_item(self._playable_q.get_nowait())
                except queue.Empty:
                    break
                self._schedule_playable(
                    playable.accompaniment,
                    current_tick=tick,
                    arrival_time_s=playable.arrival_time_s,
                )

            self._output_metronome_tick(tick=tick, bar=mt.bar, beat=mt.beat)
            model_events_to_play = sorted(
                self._scheduler.get_events_at_tick(tick),
                key=self._emission_order_key,
            )
            for event in model_events_to_play:
                if event.source == "model":
                    self._output_model_event(event)
                else:
                    self._output.output_event(event, source=event.source)
            self._emit_system_trace_frame(
                tick=tick,
                nominal_tick_time_s=target_time,
                deadline_time_s=deadline_time,
                model_events=model_events_to_play,
            )

            tick += 1

    def start(self, *, max_ticks: int | None = None) -> None:
        if self._running:
            return
        if self._source_tick_prepare is not None:
            self._source_tick_prepare()
        self._running = True
        session_start_time = self._now()
        count_in_seconds = self._tempo.tick_to_seconds(self._count_in_ticks)
        self._runtime = PromptContinuationRuntime(
            session_start_time=session_start_time,
            timeline_start_time=session_start_time + count_in_seconds,
        )
        self._input_quantization_sequence = 0
        self._output.output_status("running", "prompt-continuation")

        self._input_thread = (
            None
            if self._source_tick_reader is not None
            else threading.Thread(target=self._input_worker, daemon=True)
        )
        self._tick_thread = threading.Thread(target=self._tick_loop, kwargs={"max_ticks": max_ticks}, daemon=True)
        self._protocol_thread = threading.Thread(target=self._protocol_worker, daemon=True)

        if self._input_thread is not None:
            self._input_thread.start()
        self._tick_thread.start()
        self._protocol_thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self._input.close()
        except Exception:
            pass
        try:
            self._output.output_status("stopped", "")
            self._output.close()
        except Exception:
            pass

        if self._protocol_thread is not None:
            self._protocol_thread.join(timeout=1.0)
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=1.0)
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
