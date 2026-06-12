"""Realtime client-side orchestration for Lekai prompt-continuation mode."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from streammuse.application.services.input_timing import stamp_user_input_event
from streammuse.domain.interfaces import InputSource, OutputSink
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import MusicalTime, PlaybackScheduler, Tempo


class PromptContinuationClient(Protocol):
    def clear_history(self) -> dict[str, Any]: ...

    def start(
        self,
        *,
        melody_events: list[MusicalEvent],
        prompt_length_ticks: int,
        generation_interval_ticks: int,
        observed_until_tick: int,
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


@dataclass(frozen=True)
class _ControlAction:
    kind: str
    melody_events: list[MusicalEvent]
    observed_until_tick: int


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
        protocol_poll_interval_s: float = 0.05,
        input_snap_forward_fraction: float = 0.0,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if int(prompt_length_ticks) <= 0:
            raise ValueError("prompt_length_ticks must be > 0")
        if int(generation_interval_ticks) <= 0:
            raise ValueError("generation_interval_ticks must be > 0")
        self._input = input_source
        self._client = prompt_client
        self._output = output_sink
        self._tempo = tempo
        self._scheduler = scheduler
        self._prompt_length_ticks = int(prompt_length_ticks)
        self._generation_interval_ticks = int(generation_interval_ticks)
        self._protocol_poll_interval_s = float(protocol_poll_interval_s)
        self._input_snap_forward_fraction = float(input_snap_forward_fraction)
        self._now = now
        self._sleep = sleep

        self._running = False
        self._runtime: PromptContinuationRuntime | None = None
        self._input_thread: threading.Thread | None = None
        self._tick_thread: threading.Thread | None = None
        self._protocol_thread: threading.Thread | None = None

        self._event_q: queue.Queue[MusicalEvent] = queue.Queue()
        self._control_q: queue.Queue[_ControlAction] = queue.Queue()
        self._playable_q: queue.Queue[tuple[list[MusicalEvent], dict[str, Any]]] = queue.Queue()

        self._prompt_events: list[MusicalEvent] = []
        self._pending_append_events: list[MusicalEvent] = []
        self._start_enqueued = False
        self._last_append_observed_tick = 0
        self._scheduled_model_event_keys: set[tuple[int, int, str, int]] = set()
        self._scheduled_model_note_keys: set[tuple[int, int, int, int, int]] = set()
        self._append_generation = 0
        self._last_playable_marker: tuple[int, int, int] | None = None
        self._protocol_started = False
        self._append_sent_after_prompt = False
        self._trace_path = os.environ.get("LEKAI_PROMPT_CONTINUATION_TRACE_PATH")
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

    def _trace(self, kind: str, **payload: Any) -> None:
        if not self._trace_path:
            return
        row = {"timestamp": self._now(), "kind": kind, **payload}
        try:
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

    def _input_worker(self) -> None:
        assert self._runtime is not None
        start = self._runtime.session_start_time
        for ev in self._input.read_events():
            if not self._running:
                break
            elapsed = self._now() - start
            stamped = stamp_user_input_event(
                ev,
                elapsed_seconds=elapsed,
                tempo=self._tempo,
                snap_forward_fraction=self._input_snap_forward_fraction,
            )
            self._event_q.put(stamped)

    def _protocol_worker(self) -> None:
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
                        )
                        self._client.start(
                            melody_events=action.melody_events,
                            prompt_length_ticks=self._prompt_length_ticks,
                            generation_interval_ticks=self._generation_interval_ticks,
                            observed_until_tick=action.observed_until_tick,
                        )
                        self._protocol_started = True
                        self._output.output_status("prompt_running", "Prompt-continuation start sent")
                    elif action.kind == "append":
                        self._trace(
                            "append_send",
                            observed_until_tick=action.observed_until_tick,
                            melody_event_count=len(action.melody_events),
                        )
                        self._client.append_melody(
                            melody_events=action.melody_events,
                            observed_until_tick=action.observed_until_tick,
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
                            self._trace(
                                "playable_fetch",
                                accompaniment_event_count=len(accompaniment),
                                status_phase=playable_status.get("phase"),
                                status_melody_history_beats=playable_status.get("melody_history_beats"),
                                status_accompaniment_history_beats=playable_status.get("accompaniment_history_beats"),
                                status_beats_needed_for_playback=playable_status.get("beats_needed_for_playback"),
                                status_continuation_calls=playable_status.get("continuation_calls"),
                            )
                            self._playable_q.put((accompaniment, playable_status))
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

    def _schedule_playable(self, accompaniment: list[MusicalEvent], *, current_tick: int) -> None:
        if self._recover_late_events:
            self._schedule_playable_recover_late(accompaniment, current_tick=int(current_tick))
            return

        scheduled = 0
        dropped_past = 0
        skipped_duplicate = 0
        clipped_sustains = 0
        skipped_unpaired = 0

        events_to_schedule, dropped_past, clipped_sustains = self._clip_playable_to_current_tick(
            accompaniment,
            current_tick=int(current_tick),
        )

        note_pairs, skipped_unpaired = self._pair_playable_events(events_to_schedule)
        for note_on, note_off in note_pairs:
            note_key = (
                int(note_on.tick),
                int(note_off.tick),
                int(note_on.pitch),
                int(note_on.channel),
                int(note_on.program),
            )
            if note_key in self._scheduled_model_note_keys:
                skipped_duplicate += 1
                continue
            for event in (note_on, note_off):
                event_key = (
                    int(event.tick),
                    int(event.pitch),
                    str(event.event_type.value),
                    int(event.velocity),
                )
                model_event = MusicalEvent(
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
                self._scheduler.schedule(model_event, int(event.tick))
                self._scheduled_model_event_keys.add(event_key)
                scheduled += 1
            self._scheduled_model_note_keys.add(note_key)
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
            scheduled_event_count=scheduled,
            dropped_past=dropped_past,
            clipped_sustains=clipped_sustains,
            skipped_duplicate=skipped_duplicate,
            skipped_unpaired=skipped_unpaired,
        )

    def _schedule_playable_recover_late(self, accompaniment: list[MusicalEvent], *, current_tick: int) -> None:
        """Schedule playable history event-by-event, recovering late events now."""
        scheduled = 0
        skipped_duplicate = 0
        late_event_count = 0
        dropped_too_late_note_on = 0
        placeholder_count = 0
        current_tick = int(current_tick)

        for event in sorted(accompaniment, key=lambda ev: (int(ev.tick), str(ev.event_type.value), int(ev.pitch))):
            if event.is_placeholder or event.pitch == -1:
                placeholder_count += 1
                continue
            event_key = (
                int(event.tick),
                int(event.pitch),
                str(event.event_type.value),
                int(event.velocity),
            )
            if event_key in self._scheduled_model_event_keys:
                skipped_duplicate += 1
                continue

            event_tick = int(event.tick)
            schedule_tick = event_tick if event_tick >= current_tick else current_tick
            if event_tick < current_tick:
                late_event_count += 1
                if (
                    self._bound_late_recovery
                    and self._recover_late_max_ticks is not None
                    and current_tick - event_tick > self._recover_late_max_ticks
                    and event.event_type == EventType.NOTE_ON
                    and int(event.velocity) > 0
                ):
                    dropped_too_late_note_on += 1
                    self._scheduled_model_event_keys.add(event_key)
                    continue

            model_event = MusicalEvent(
                tick=event.tick,
                pitch=event.pitch,
                event_type=event.event_type,
                velocity=event.velocity,
                channel=event.channel,
                program=event.program,
                is_placeholder=event.is_placeholder,
                source="model",
                backup_level=max(0, int(event.tick) - current_tick),
            )
            self._scheduler.schedule(model_event, schedule_tick)
            self._scheduled_model_event_keys.add(event_key)
            scheduled += 1

        self._output.output_status(
            "ready",
            f"Scheduled {scheduled} playable accompaniment event(s); "
            f"recovered {late_event_count} late event(s); "
            f"dropped {dropped_too_late_note_on} too-late note_on event(s); "
            f"skipped {skipped_duplicate} duplicate event(s); "
            f"skipped {placeholder_count} placeholder event(s).",
        )
        self._trace(
            "schedule_playable",
            current_tick=current_tick,
            mode="recover_late_events",
            input_event_count=len(accompaniment),
            scheduled_event_count=scheduled,
            late_event_count=late_event_count,
            dropped_too_late_note_on=dropped_too_late_note_on,
            bound_late_recovery=self._bound_late_recovery,
            recover_late_max_ticks=self._recover_late_max_ticks,
            skipped_duplicate=skipped_duplicate,
            placeholder_count=placeholder_count,
        )

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

    def _tick_loop(self, *, max_ticks: int | None) -> None:
        assert self._runtime is not None
        start = self._runtime.session_start_time
        tick = 0
        while self._running:
            if max_ticks is not None and tick >= max_ticks:
                self._running = False
                break

            target_time = start + self._tempo.tick_to_seconds(tick)
            delay = target_time - self._now()
            if delay > 0:
                self._sleep(delay)

            mt = MusicalTime.from_tick(tick, self._tempo)
            self._output.output_tick(tick=tick, bar=mt.bar, beat=mt.beat)
            self._sleep(self._tempo.seconds_per_tick * self._INPUT_BUFFER_RATIO)

            self._drain_user_events()
            observed_until_tick = tick + 1
            self._maybe_enqueue_start(observed_until_tick)
            self._maybe_enqueue_append(observed_until_tick)

            while True:
                try:
                    accompaniment, _status = self._playable_q.get_nowait()
                except queue.Empty:
                    break
                self._schedule_playable(accompaniment, current_tick=tick)

            for event in self._scheduler.get_events_at_tick(tick):
                self._output.output_event(event, source=event.source)

            tick += 1

    def start(self, *, max_ticks: int | None = None) -> None:
        if self._running:
            return
        self._running = True
        self._runtime = PromptContinuationRuntime(session_start_time=self._now())
        self._output.output_status("running", "prompt-continuation")

        self._input_thread = threading.Thread(target=self._input_worker, daemon=True)
        self._tick_thread = threading.Thread(target=self._tick_loop, kwargs={"max_ticks": max_ticks}, daemon=True)
        self._protocol_thread = threading.Thread(target=self._protocol_worker, daemon=True)

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
