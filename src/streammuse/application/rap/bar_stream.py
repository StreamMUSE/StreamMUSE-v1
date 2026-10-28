"""Client side of protocol v3: one bar at a time, from a server-written lyric stream.

The server writes each bar's lyric ahead of playback and pushes it here. The
controller keeps a copy, renders an eSpeak fallback that speaks the *same*
line, and fetches the bar's MOSS audio by lyric id. One tick before each bar
starts it commits the best audio it has, in this order:

1. remote audio of the latest lyric revision;
2. remote audio of an older revision (after a listener edit, before the new
   revision's audio arrived);
3. eSpeak speaking the latest streamed lyric;
4. eSpeak speaking a locally chosen line (only if no streamed lyric arrived).

Listener edits (theme / flow from a bar onward) are applied to the local live
plan and sent to the server, which rewrites every uncommitted bar.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from math import isfinite
from threading import RLock
import time
from typing import Protocol

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.audio_rendering import bar_frame_count
from streammuse.application.rap.audio_service import RapBarRenderer
from streammuse.application.rap.live_plan import LivePlan
from streammuse.application.rap.monitoring_payloads import (
    bounded_chunk_event_payload,
    flow_template_payload,
    scheduled_syllables_payload,
    prepared_bar_timing_payload,
)
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.application.rap.service import ProsodyAnalyzer
from streammuse.domain.rap import CandidateRequest, RapScenario, ScenarioSegment, materialize_flow
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapMixConfig
from streammuse.application.rap.chunk_audio import remix_prepared_bar
from streammuse.domain.rap.events import RapEventType
from streammuse.domain.rap.lyric_stream import (
    LYRIC_STREAM_SCHEMA_VERSION,
    BarAudioManifest,
    BarLyricRecord,
    PlanEdit,
    TempoUpdate,
    LYRIC_STREAM_CAPABILITIES,
    TEMPO_UPDATES_CAPABILITY,
)
from streammuse.domain.rap.timeline import RapTimeline, BarTimingSnapshot
from streammuse.domain.rap.remote_chunk import _flow_payload
from streammuse.domain.timing import Tempo

SOURCE_REMOTE_LATEST = "remote_latest"
SOURCE_REMOTE_OLDER = "remote_older"
SOURCE_ESPEAK_STREAM = "espeak_stream_lyric"
SOURCE_ESPEAK_LOCAL = "espeak_local_lyric"
# Last resort when nothing at all is ready at commit (should not happen while local fallbacks render).
SOURCE_SILENCE = "silence"
SOURCE_DRUMS_ONLY = "drums_only"
_RETRY_SLEEP_SECONDS = 0.05


@dataclass(frozen=True)
class BarAudioFetch:
    status: str
    lyric_id: str | None = None
    manifest: BarAudioManifest | None = None
    source_wav: bytes | None = None
    request_ms: float = 0.0


class LyricStreamPort(Protocol):
    def open(
        self, session_id: str, start: Mapping[str, object], on_message: Callable[[Mapping[str, object]], None]
    ) -> None: ...

    def send(self, message: Mapping[str, object]) -> None: ...

    def close(self) -> None: ...


class BarAudioSource(Protocol):
    def fetch(
        self, session_id: str, bar: int, lyric_id: str, *, wait_ms: int, accept_older: bool
    ) -> BarAudioFetch: ...

    def close(self) -> None: ...


class RemoteBarPreparer(Protocol):
    def prepare(self, record: BarLyricRecord, manifest: BarAudioManifest, source_wav: bytes) -> PreparedRapBar: ...

    def with_alignment_policy(self, policy: str) -> RemoteBarPreparer: ...


class FallbackLines(Protocol):
    def line_for(self, request: CandidateRequest): ...


class TemplateLookup(Protocol):
    def get(self, template_id: str): ...


@dataclass(frozen=True)
class _BarSource:
    record: BarLyricRecord
    fetch: BarAudioFetch
    timing: BarTimingSnapshot | None
    submitted: float


@dataclass
class _Slot:
    bar: int
    latest: BarLyricRecord | None = None
    records: dict[str, BarLyricRecord] = field(default_factory=dict)
    lyric_arrived: dict[str, float] = field(default_factory=dict)
    remote: dict[str, PreparedRapBar] = field(default_factory=dict)
    remote_ready_at: dict[str, float] = field(default_factory=dict)
    remote_timing: dict[str, dict[str, float]] = field(default_factory=dict)
    remote_continued: dict[str, bool] = field(default_factory=dict)
    sources: dict[str, _BarSource] = field(default_factory=dict)
    espeak: dict[str, PreparedRapBar] = field(default_factory=dict)
    local: PreparedRapBar | None = None
    local_plan: PlannedRapBar | None = None
    local_requested: bool = False
    fetching: set[str] = field(default_factory=set)
    espeak_requested: set[str] = field(default_factory=set)
    committed: PreparedRapBar | None = None
    committed_choice: str | None = None


class BarStreamController:
    """Protocol v3 controller with the same lifecycle as ``RollingRapChunkController``."""

    def __init__(
        self,
        *,
        tempo: Tempo,
        scenario: RapScenario,
        templates: TemplateLookup,
        fallback_catalog: FallbackLines,
        analyzer: ProsodyAnalyzer,
        fallback_renderer: RapBarRenderer,
        lyric_stream: LyricStreamPort,
        audio_source: BarAudioSource,
        preparer: RemoteBarPreparer,
        publisher,
        enqueue: Callable[[PreparedRapBar], None],
        session_id: str,
        seed: int,
        planning_bar_limit: int | None,
        startup_timeout_seconds: float,
        lyric_lead: int = 3,
        audio_lead: int = 2,
        fetch_window: int = 2,
        mac_margin_seconds: float = 0.25,
        fetch_executor: Executor | None = None,
        render_executor: Executor | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        audio_format: AudioFormat | None = None,
        timeline: RapTimeline | None = None,
        on_timeline_changed: Callable[[RapTimeline], None] | None = None,
        initial_epoch: int = 0,
        tempo_ack_timeout_seconds: float = 10.0,
    ) -> None:
        if tempo.ticks_per_beat != 4 or tempo.beats_per_bar != 4:
            raise ValueError("lyric stream bars require 4/4 with four ticks per beat")
        if not isfinite(startup_timeout_seconds) or startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        if isinstance(initial_epoch, bool) or not isinstance(initial_epoch, int) or initial_epoch < 0:
            raise ValueError("initial_epoch must be a non-negative integer")
        if not isfinite(tempo_ack_timeout_seconds) or tempo_ack_timeout_seconds <= 0:
            raise ValueError("tempo_ack_timeout_seconds must be positive and finite")
        self._tempo = tempo
        self._timeline = timeline or RapTimeline(tempo)
        self._initial_timeline = self._timeline
        self._timing_enabled = timeline is not None
        self._legacy_timing_before_bar = 0
        self._on_timeline_changed = on_timeline_changed
        self._tempo_edit_seq = 0
        self._last_tempo_edit: dict[str, object] | None = None
        self._tempo_uncertain = False
        self._tempo_ack_timeout = tempo_ack_timeout_seconds
        self._mix_changes: list[tuple[int, RapMixConfig]] = []
        self._mix_revision = 0
        self._scenario = scenario
        self._templates = templates
        self._fallback_catalog = fallback_catalog
        self._analyzer = analyzer
        self._fallback_renderer = fallback_renderer
        self._lyric_stream = lyric_stream
        self._audio_source = audio_source
        self._preparer = preparer
        self._alignment_revision = 0
        self._publisher = publisher
        self._enqueue = enqueue
        self._base_session_id = session_id
        self._seed = seed
        self._planning_bar_limit = planning_bar_limit
        self._startup_timeout = float(startup_timeout_seconds)
        self._lyric_lead = lyric_lead
        self._audio_lead = audio_lead
        self._fetch_window = fetch_window
        self._mac_margin = mac_margin_seconds
        self._fetch_executor = fetch_executor or ThreadPoolExecutor(max_workers=3, thread_name_prefix="rap-bar-fetch")
        self._render_executor = render_executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="rap-bar-espeak")
        self._clock = monotonic
        self._audio_format = audio_format or AudioFormat()
        self._lock = RLock()
        self._tempo_edit_lock = RLock()
        self._epoch = initial_epoch
        self._started = False
        self._stream_attached = False
        self._stopping = False
        self._stop_successor_bar: int | None = None
        self._closed = False
        self._clock_origin: float | None = None
        self._last_tick = -1
        self._playhead = -1
        self._slots: dict[int, _Slot] = {}
        self._history: list[tuple[int, str]] = []
        self._plan = self._base_plan()
        self._edit_seq = 0
        self._latest_plan_edit: dict[str, object] | None = None

    # -- public lifecycle (same surface as RollingRapChunkController) ------------------

    @property
    def scenario(self) -> RapScenario:
        return self._scenario

    @property
    def coordinator_epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def timeline(self) -> RapTimeline:
        with self._lock:
            return self._timeline

    @property
    def tempo_control_state(self) -> dict[str, object]:
        with self._lock:
            bar = max(0, self._last_tick // 16)
            return {"bpm": self._timeline.snapshot(bar, self._audio_format.sample_rate_hz).bpm,
                    "revision": self._timeline.revision, "last_edit": self._last_tempo_edit,
                    "requires_reset": self._tempo_uncertain,
                    "recovery": "local_fallback_until_reset" if self._tempo_uncertain else None,
                    "supported": TEMPO_UPDATES_CAPABILITY in getattr(self._lyric_stream, "capabilities", ())}

    def update_tempo(self, bpm: float, from_bar: int | None = None) -> dict[str, object]:
        """Negotiate timing before changing any local preparation or playback state."""
        with self._tempo_edit_lock:
            return self._update_tempo(bpm, from_bar)

    def _update_tempo(self, bpm: float, from_bar: int | None) -> dict[str, object]:
        with self._lock:
            requested = self._playhead + 1 if from_bar is None else from_bar
            TempoUpdate(self._tempo_edit_seq + 1, requested, bpm)
            if not self._started and not self._closed:
                if requested != 0:
                    raise ValueError("initial stopped tempo must start at bar zero")
                changed = RapTimeline(Tempo(bpm, 4, 4))
                if self._on_timeline_changed is not None:
                    self._on_timeline_changed(changed)
                self._tempo = changed.tempo
                self._timeline = self._initial_timeline = changed
                self._timing_enabled = True
                result = {"requested_from_bar": 0, "effective_from_bar": 0,
                          "revision": 0, "bpm": bpm, "coordinator_epoch": self._epoch}
                self._last_tempo_edit = result
                return result
            operation = getattr(self._lyric_stream, "request_tempo", None)
            if not callable(operation) or TEMPO_UPDATES_CAPABILITY not in getattr(self._lyric_stream, "capabilities", ()):
                raise ValueError("server did not negotiate tempo update capability")
            if self._closed or self._tempo_uncertain:
                raise RuntimeError("tempo changes require an active, synchronized session")
            effective = max(requested, self._playhead + 1)
            # Leave a bounded acknowledgement and local preparation interval.
            request_started = self._clock()
            while (self._clock_origin is not None
                   and self._commit_time_locked(effective) < request_started + self._tempo_ack_timeout + self._mac_margin):
                effective += 1
            update = TempoUpdate(self._tempo_edit_seq + 1, effective, bpm)
            epoch, prior = self._epoch, self._timeline
        try:
            ack = operation(update.to_payload(), timeout=self._tempo_ack_timeout)
        except (TimeoutError, ConnectionError):
            with self._lock:
                if epoch == self._epoch and not self._closed:
                    self._fail_tempo_locked("tempo acknowledgement lost; continuing local audio at confirmed timing; reset required")
            raise
        with self._lock:
            if epoch != self._epoch or self._closed:
                raise RuntimeError("tempo acknowledgement belongs to an obsolete session")
            accepted = ack.get("effective_from_bar")
            if (not isinstance(accepted, int) or isinstance(accepted, bool) or accepted < effective
                    or accepted <= self._playhead or self._timeline != prior
                    or ack.get("bpm") != bpm or ack.get("revision") != prior.revision + 1
                    or ack.get("edit_seq") != update.edit_seq):
                self._fail_tempo_locked("tempo acknowledgement is invalid or missed its boundary; reset required")
                raise ValueError("invalid or late tempo acknowledgement; reset required")
            changed = self._timeline.with_tempo(bpm, accepted)
            if self._on_timeline_changed is not None:
                try:
                    self._on_timeline_changed(changed)
                except Exception:
                    self._fail_tempo_locked("playback rejected acknowledged tempo; reset required")
                    raise
            if not self._timing_enabled:
                self._legacy_timing_before_bar = accepted
            self._timeline = changed
            self._timing_enabled = True
            self._tempo_edit_seq = update.edit_seq
            for bar, slot in self._slots.items():
                if bar >= accepted and slot.committed is None:
                    slot.remote.clear()
                    slot.remote_ready_at.clear()
                    slot.remote_timing.clear()
                    slot.sources.clear()
                    slot.espeak.clear()
                    slot.espeak_requested.clear()
                    slot.fetching.clear()
                    slot.local = None
                    slot.local_requested = False
            result = {**ack, "requested_from_bar": requested, "coordinator_epoch": self._epoch,
                      "ack_elapsed_ms": (self._clock() - request_started) * 1000.0,
                      "ack_timeout_seconds": self._tempo_ack_timeout}
            self._last_tempo_edit = result
            self._ensure_window_locked()
            return result

    def _fail_tempo_locked(self, message: str) -> None:
        self._tempo_uncertain = True
        # A lost reply cannot establish the server's timing. Keep the confirmed
        # local timeline audible, but quarantine remote audio until explicit reset.
        for slot in self._slots.values():
            if slot.committed is None:
                slot.remote.clear()
                slot.remote_ready_at.clear()
                slot.remote_timing.clear()
                slot.sources.clear()
        self._emit(RapEventType.PRESENTATION_ERROR, payload={
            "component": "tempo_update", "message": message, "requires_reset": True,
            "recovery": "local_fallback_until_reset", "ack_timeout_seconds": self._tempo_ack_timeout,
        })
        self._ensure_window_locked()

    def update_alignment_policy(self, policy: str) -> dict:
        """Re-warp retained speech locally from the first uncommitted bar."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot change alignment on a closed controller")
            switch = getattr(self._preparer, "with_alignment_policy", None)
            if not callable(switch):
                raise ValueError("audio preparer does not support alignment policy changes")
            preparer = switch(policy)
            effective = self._playhead + 1
            self._preparer = preparer
            self._alignment_revision += 1
            revision, epoch = self._alignment_revision, self._epoch
            pending = []
            for bar, slot in self._slots.items():
                if bar < effective or slot.committed is not None:
                    continue
                slot.remote.clear()
                slot.remote_ready_at.clear()
                slot.remote_timing.clear()
                pending.extend((bar, lyric_id) for lyric_id in slot.sources)
            for bar, lyric_id in pending:
                self._render_executor.submit(self._prepare_remote, epoch, bar, lyric_id, revision)
            return {"effective_from_bar": effective, "alignment_policy": policy,
                    "revision": revision, "coordinator_epoch": epoch}

    def update_mix(self, config: RapMixConfig, from_bar: int | None = None) -> dict[str, object]:
        """Recombine retained stems without another synthesis or warp job."""
        if not isinstance(config, RapMixConfig):
            raise ValueError("mix config must be a RapMixConfig")
        with self._lock:
            requested = self._playhead + 1 if from_bar is None else from_bar
            if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
                raise ValueError("from_bar must be a non-negative integer")
            effective = max(requested, self._playhead + 1)
            config = replace(config, revision=max(config.revision, self._mix_revision + 1))
            changes = []
            for bar, slot in self._slots.items():
                if bar < effective or slot.committed is not None:
                    continue
                remote = {key: remix_prepared_bar(audio, config) for key, audio in slot.remote.items()}
                espeak = {key: remix_prepared_bar(audio, config) for key, audio in slot.espeak.items()}
                local = remix_prepared_bar(slot.local, config) if slot.local is not None else None
                changes.append((slot, remote, espeak, local))
            self._mix_revision = config.revision
            self._mix_changes = [(bar, old) for bar, old in self._mix_changes if bar < effective] + [(effective, config)]
            for slot, remote, espeak, local in changes:
                slot.remote, slot.espeak, slot.local = remote, espeak, local
            return {"requested_from_bar": requested, "effective_from_bar": effective,
                    "revision": config.revision, "coordinator_epoch": self._epoch,
                    "vocal_gain": config.vocal_gain, "drum_gain": config.drum_gain,
                    "master_gain": config.master_gain, "peak_limit": config.peak_limit}

    def _apply_mix(self, prepared: PreparedRapBar) -> PreparedRapBar:
        config = next((config for bar, config in reversed(self._mix_changes) if bar <= prepared.bar), None)
        if config is not None and prepared.mix_config != config:
            return remix_prepared_bar(prepared, config)
        return prepared

    @property
    def terminal_bar_limit(self) -> int | None:
        return self._planning_bar_limit

    @property
    def next_editable_bar(self) -> int:
        """The first bar a listener edit can still change."""
        with self._lock:
            return self._playhead + 1

    @property
    def plan_control_state(self) -> dict[str, object]:
        """An atomic control snapshot, including the original target of the last edit."""
        with self._lock:
            return {
                "next_editable_bar": self._playhead + 1,
                "last_edit": dict(self._latest_plan_edit) if self._latest_plan_edit else None,
            }

    @property
    def session_id(self) -> str:
        return f"{self._base_session_id}-e{self._epoch}"

    def start(self) -> None:
        with self._lock:
            if self._closed or self._started:
                return
            self._started = True
            self._stopping = False
            self._stop_successor_bar = None
            epoch = self._epoch
            start = self._start_message(start_bar=0)
            session_id = self.session_id
        self._lyric_stream.open(session_id, start, lambda message: self._on_message(epoch, message))
        startup_deadline = self._clock() + self._startup_timeout
        with self._lock:
            self._ensure_window_locked(startup=True, deadline=startup_deadline)
        # Wait for bar 0: its streamed lyric and, within the startup budget, its remote audio.
        while self._clock() < startup_deadline:
            with self._lock:
                slot = self._slot(0)
                if slot.latest is not None and slot.latest.lyric_id in slot.remote:
                    break
                if epoch != self._epoch or self._closed or self._stopping:
                    return
            time.sleep(_RETRY_SLEEP_SECONDS)
        self._await_any_fallback(0, startup_deadline + 5.0, epoch=epoch)
        with self._lock:
            if epoch != self._epoch or self._closed or self._stopping:
                return
            prepared = self._commit_locked(0, tick=None)
            if epoch != self._epoch or self._closed or self._stopping:
                return
            if prepared is not None:
                self._enqueue(prepared)
            self._clock_origin = self._clock()
            self._ensure_window_locked()

    def on_tick(self, tick: int) -> None:
        deliver: PreparedRapBar | None = None
        with self._lock:
            if not self._started or self._closed:
                return
            epoch = self._epoch
            self._last_tick = tick
            bar = (tick + 1) // self._tempo.ticks_per_bar
            self._emit(RapEventType.TICK, bar=tick // self._tempo.ticks_per_bar, tick=tick, payload={
                "beat": (tick % self._tempo.ticks_per_bar) // self._tempo.ticks_per_beat,
                "tick_in_beat": tick % self._tempo.ticks_per_beat,
            })
            if self._stopping:
                return
            guard = bar * self._tempo.ticks_per_bar - 1
            if tick == guard and bar > self._playhead and self._within_limit(bar):
                deliver = self._commit_locked(bar, tick=tick)
                self._ensure_window_locked()
            if deliver is not None and not self._closed and not self._stopping and epoch == self._epoch:
                self._enqueue(deliver)

    def update_plan(self, *, from_bar: int, topic: str | None = None, template_id: str | None = None) -> PlanEdit:
        """A listener's live edit: theme and/or flow from ``from_bar`` onward."""
        with self._lock:
            self._edit_seq += 1
            edit = PlanEdit(
                edit_seq=self._edit_seq,
                from_bar=from_bar,
                topic=topic,
                flow_template=None if template_id is None else self._templates.get(template_id),
            )
            self._plan = self._plan.with_edit(edit)
            effective = max(from_bar, self._playhead + 1)
            for bar, slot in self._slots.items():
                if bar >= effective and slot.committed is None:
                    # Keep playable audio until its replacement is fully rendered.
                    slot.local_requested = False
            self._ensure_window_locked()
            self._latest_plan_edit = {
                "coordinator_epoch": self._epoch,
                "edit_seq": edit.edit_seq,
                "from_bar": from_bar,
                "effective_from_bar": effective,
                "topic": topic,
                "template_id": template_id,
            }
            self._emit(
                RapEventType.PLAN_EDITED, bar=effective,
                tick=self._last_tick if self._last_tick >= 0 else None,
                payload=dict(self._latest_plan_edit),
            )
        self._lyric_stream.send({"type": "update", "edit": edit.to_payload()})
        return edit

    def request_stop(self, *, successor_bar: int | None) -> None:
        with self._lock:
            if self._closed or not self._started or self._stopping:
                return
            self._stopping = True
            self._stop_successor_bar = successor_bar
            if successor_bar is not None and self._slot(successor_bar).committed is None:
                # Retained for resume_audio; nothing is enqueued while stopped.
                self._commit_locked(successor_bar, tick=self._last_tick)

    def resume_audio(self, bar: int) -> None:
        with self._lock:
            if not self._stopping or bar != self._stop_successor_bar:
                raise RuntimeError("audio controller is not stopped at this successor bar")
            prepared = self._slot(bar).committed
            if prepared is None:
                raise ValueError("audio resume requires the retained successor bar")
        self._enqueue(prepared)

    def resume_after_stop(self) -> None:
        with self._lock:
            if self._tempo_uncertain:
                raise RuntimeError("tempo acknowledgement is uncertain; reset required")
            if self._closed or not self._started or not self._stopping:
                raise RuntimeError("audio controller is not stopped for resume")
            successor = self._stop_successor_bar
            if successor is None:
                raise RuntimeError("terminal audio stop requires reset before resume")
            self._stopping = False
            self._stop_successor_bar = None
            successor_tick = successor * self._tempo.ticks_per_bar
            self._clock_origin = self._clock() - self._timeline.tick_to_seconds(successor_tick)
            self._ensure_window_locked()

    def reset(self) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot reset a closed controller")
            self._epoch += 1
            self._started = False
            self._stream_attached = False
            self._stopping = False
            self._stop_successor_bar = None
            self._clock_origin = None
            self._last_tick = -1
            self._playhead = -1
            self._slots.clear()
            self._history.clear()
            self._plan = self._base_plan()
            self._edit_seq = 0
            self._latest_plan_edit = None
            self._timeline = self._initial_timeline
            self._legacy_timing_before_bar = 0
            self._tempo_edit_seq = 0
            self._last_tempo_edit = None
            self._tempo_uncertain = False
            epoch = self._epoch
        self._close_stream()
        return epoch

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
            for slot in self._slots.values():
                slot.sources.clear()
        self._close_stream()
        self._fetch_executor.shutdown(wait=True, cancel_futures=True)
        self._render_executor.shutdown(wait=True, cancel_futures=True)
        self._audio_source.close()

    # -- lyric stream messages --------------------------------------------------------

    def _on_message(self, epoch: int, message: Mapping[str, object]) -> None:
        kind = message.get("type")
        with self._lock:
            if epoch != self._epoch or self._closed:
                return
            if kind == "started":
                if self._stream_attached and message.get("created") is True:
                    self._fail_tempo_locked(
                        "server recreated the lyric session; continuing local audio at confirmed timing; reset required")
                self._stream_attached = True
            elif kind == "lyric":
                record = BarLyricRecord.from_payload(message["record"])
                slot = self._slot(record.bar)
                if slot.committed is not None:
                    return
                slot.records[record.lyric_id] = record
                slot.lyric_arrived[record.lyric_id] = self._clock()
                if slot.latest is None or record.revision >= slot.latest.revision:
                    slot.latest = record
                self._emit(
                    RapEventType.BAR_RESERVED,
                    bar=record.bar,
                    tick=self._last_tick if self._last_tick >= 0 else None,
                    payload={
                        "source": "lyric_stream",
                        "text": record.text,
                        "topic": record.topic,
                        "template_id": record.flow_template.template_id,
                        "flow": flow_template_payload(record.flow_template),
                        "fallback": False,
                        "fallback_reason": None,
                        "revision": record.revision,
                        "lyric_id": record.lyric_id,
                        "lead_ms": self._lead_ms_locked(record.bar),
                    },
                )
                self._ensure_window_locked()
            elif kind == "superseded":
                for bar in message.get("bars") or ():
                    slot = self._slots.get(int(bar))
                    if slot is not None and slot.committed is None:
                        slot.latest = None
            elif kind == "error":
                self._emit(RapEventType.PRESENTATION_ERROR, bar=message.get("bar"), payload={
                    "component": "lyric_stream", "message": str(message.get("message", ""))[:512],
                })

    # -- window maintenance -----------------------------------------------------------

    def _ensure_window_locked(self, *, startup: bool = False, deadline: float | None = None) -> None:
        if self._stopping or self._closed:
            return
        first = self._playhead + 1
        for bar in range(first, first + self._lyric_lead):
            if not self._within_limit(bar):
                break
            slot = self._slot(bar)
            if slot.committed is not None:
                continue
            timing = self._timeline.snapshot(bar, self._audio_format.sample_rate_hz) if self._timing_enabled else None
            if slot.latest is not None and slot.latest.lyric_id not in slot.espeak_requested:
                slot.espeak_requested.add(slot.latest.lyric_id)
                self._render_executor.submit(self._render_stream_espeak, self._epoch, slot.latest, timing)
            if not slot.local_requested:
                slot.local_requested = True
                self._render_executor.submit(self._render_local_fallback, self._epoch, bar, timing)
            in_window = bar < first + self._fetch_window or startup
            latest = slot.latest
            if (
                in_window
                and not self._tempo_uncertain
                and latest is not None
                and latest.lyric_id not in slot.remote
                and latest.lyric_id not in slot.sources
                and latest.lyric_id not in slot.fetching
            ):
                slot.fetching.add(latest.lyric_id)
                commit_at = deadline if deadline is not None else self._commit_time_locked(bar)
                self._fetch_executor.submit(self._fetch_remote, self._epoch, latest, commit_at, timing)

    def _fetch_remote(self, epoch: int, record: BarLyricRecord, commit_at: float | None,
                      timing: BarTimingSnapshot | None = None) -> None:
        submitted = self._clock()
        try:
            with self._lock:
                if epoch != self._epoch or self._closed or self._tempo_uncertain or not self._timing_current(record.bar, timing):
                    return
                session_id = self.session_id
                # An older revision already here is the fallback; only the exact one is worth a fetch.
                accept_older = not self._slot(record.bar).remote
                self._emit(RapEventType.CHUNK_REQUEST_SUBMITTED, bar=record.bar, request_id=record.lyric_id,
                           payload=self._bar_payload(record, state="requested", decision="pending"))
            while True:
                remaining = (commit_at - self._clock() - self._mac_margin) if commit_at is not None else 8.0
                with self._lock:
                    if epoch != self._epoch or self._closed:
                        return
                    slot = self._slot(record.bar)
                    stale = (
                        epoch != self._epoch
                        or self._closed
                        or self._tempo_uncertain
                        or slot.committed is not None
                        or slot.latest is None
                        or slot.latest.lyric_id != record.lyric_id
                        or not self._timing_current(record.bar, timing)
                    )
                if stale or remaining <= 0:
                    return
                fetch = self._audio_source.fetch(
                    session_id,
                    record.bar,
                    record.lyric_id,
                    wait_ms=max(0, min(int(remaining * 1000), 9_000)),
                    accept_older=accept_older,
                )
                if fetch.manifest is not None and fetch.source_wav is not None and fetch.lyric_id is not None:
                    audio_record = self._record(record.bar, fetch.lyric_id)
                    if audio_record is not None:
                        if timing is not None and fetch.manifest.tempo_bpm != timing.bpm:
                            time.sleep(_RETRY_SLEEP_SECONDS)
                            continue
                        with self._lock:
                            if (epoch != self._epoch or self._closed or self._tempo_uncertain
                                    or not self._timing_current(record.bar, timing)):
                                return
                            slot = self._slot(record.bar)
                            if slot.committed is not None:
                                return
                            # Cache before releasing the lock: a policy edit during the
                            # warp must still find the source, even at a commit boundary.
                            slot.sources[audio_record.lyric_id] = _BarSource(audio_record, fetch, timing, submitted)
                            while len(slot.sources) > 2:
                                oldest = min(slot.sources, key=lambda key: slot.sources[key].record.revision)
                                slot.sources.pop(oldest)
                                slot.remote.pop(oldest, None)
                                slot.remote_ready_at.pop(oldest, None)
                                slot.remote_timing.pop(oldest, None)
                                slot.remote_continued.pop(oldest, None)
                            revision = self._alignment_revision
                        self._prepare_remote(epoch, record.bar, audio_record.lyric_id, revision)
                if fetch.status == "ready":
                    return
                if fetch.status == "older_version":
                    accept_older = False
                    continue
                if fetch.status == "pending":
                    if fetch.request_ms < 50.0:
                        time.sleep(_RETRY_SLEEP_SECONDS)  # do not spin if a pending came back early
                    continue
                return  # superseded, failed or unknown: the commit falls back
        except Exception as error:  # transport or warp failure: the commit falls back
            with self._lock:
                if epoch == self._epoch and not self._closed:
                    self._emit(RapEventType.CHUNK_REMOTE_REJECTED, bar=record.bar, request_id=record.lyric_id,
                               payload=self._bar_payload(record, state="rejected", decision="fallback",
                                                         failure=f"{type(error).__name__}: {error}"))
        finally:
            with self._lock:
                if epoch == self._epoch and self._timing_current(record.bar, timing):
                    slot = self._slots.get(record.bar)
                    if slot is not None:
                        slot.fetching.discard(record.lyric_id)

    def _prepare_remote(self, epoch: int, bar: int, lyric_id: str, revision: int) -> None:
        with self._lock:
            slot = self._slots.get(bar)
            if (epoch != self._epoch or self._closed or self._tempo_uncertain
                    or revision != self._alignment_revision or slot is None or slot.committed is not None):
                return
            source = slot.sources.get(lyric_id)
            if source is None or not self._timing_current(bar, source.timing):
                return
            preparer = self._preparer
        record, fetch, timing = source.record, source.fetch, source.timing
        try:
            started = self._clock()
            kwargs = {} if timing is None else {"timing": timing}
            prepared = preparer.prepare(record, fetch.manifest, fetch.source_wav, **kwargs)
            prepare_ms = (self._clock() - started) * 1000.0
            with self._lock:
                slot = self._slots.get(bar)
                if (epoch != self._epoch or self._closed or self._tempo_uncertain
                        or revision != self._alignment_revision or slot is None or slot.committed is not None
                        or slot.sources.get(lyric_id) is not source or not self._timing_current(bar, timing)
                        or not self._compatible(prepared, bar)):
                    return
                slot.remote[lyric_id] = self._apply_mix(prepared)
                slot.remote_ready_at[lyric_id] = self._clock()
                slot.remote_timing[lyric_id] = {
                    "transfer": fetch.request_ms, "mac": prepare_ms,
                    "total": (self._clock() - source.submitted) * 1000.0,
                    **{k: float(v) for k, v in fetch.manifest.stage_timings_ms.items()},
                }
                slot.remote_continued[lyric_id] = bool(
                    (fetch.manifest.diagnostics.get("continuation") or {}).get("used"))
                self._emit(RapEventType.CHUNK_REMOTE_COMPLETED, bar=bar, request_id=lyric_id,
                           payload=self._bar_payload(record, state="remote_ready",
                                                     decision="older" if fetch.status == "older_version" else "latest",
                                                     timing=slot.remote_timing[lyric_id]))
        except Exception as error:
            with self._lock:
                if epoch == self._epoch and not self._closed and revision == self._alignment_revision:
                    self._emit(RapEventType.CHUNK_REMOTE_REJECTED, bar=bar, request_id=lyric_id,
                               payload=self._bar_payload(record, state="rejected", decision="fallback",
                                                         failure=f"{type(error).__name__}: {error}"))

    def _render_stream_espeak(self, epoch: int, record: BarLyricRecord,
                              timing: BarTimingSnapshot | None = None) -> None:
        plan = PlannedRapBar(
            bar=record.bar,
            segment=ScenarioSegment(record.bar, 1, record.topic, record.flow_template.template_id, (record.text,)),
            template=record.flow_template,
            analysis=self._analyzer.analyze(record.text),
            scheduled=record.scheduled,
            text=record.text,
            source="lyric_stream_espeak",
            fallback_reason="remote_pending",
        )
        prepared = self._render_plan(epoch, plan, timing)
        if prepared is not None:
            with self._lock:
                slot = self._slots.get(record.bar)
                if (epoch == self._epoch and not self._closed and slot is not None
                        and slot.committed is None and self._timing_current(record.bar, timing)):
                    slot.espeak[record.lyric_id] = self._apply_mix(prepared)

    def _render_local_fallback(self, epoch: int, bar: int, timing: BarTimingSnapshot | None = None) -> None:
        try:
            self._render_local_fallback_unguarded(epoch, bar, timing)
        except Exception as error:  # an executor would swallow it; the commit then has no local line
            self._emit_for_epoch(epoch, RapEventType.PRESENTATION_ERROR, bar=bar, payload={
                "component": "local_fallback", "message": f"{type(error).__name__}: {error}"[:512],
            })

    def _render_local_fallback_unguarded(self, epoch: int, bar: int,
                                        timing: BarTimingSnapshot | None = None) -> None:
        with self._lock:
            if epoch != self._epoch or self._closed or not self._timing_current(bar, timing):
                return
            entry = self._plan.for_bar(bar)
            context = tuple(text for _, text in self._history[-4:])
        request = CandidateRequest(
            request_id=f"{self._base_session_id}-bar-{bar}-local",
            target_bar=bar,
            topic=entry.topic,
            flow_template=entry.flow_template,
            count=1,
            context_lines=context,
            seed=self._seed + bar,
        )
        topic, template = entry.topic, entry.flow_template
        try:
            line = self._fallback_catalog.line_for(request)
        except ValueError as error:
            # Live topics are open-ended; preserve the catalogue line's real topic
            # rather than label safety audio as a successful listener edit.
            lookup = getattr(self._fallback_catalog, "line_for_any", self._fallback_catalog.line_for)
            try:
                line = lookup(request)
                topic = getattr(line, "topic", None) or "prevalidated fallback"
            except ValueError:
                base = self._scenario.segment_for_bar(bar)
                base_template = self._templates.get(base.template_id)
                line = self._fallback_catalog.line_for(replace(
                    request, topic=base.topic, flow_template=base_template,
                ))
                topic = base.topic
                if len(line.analysis.syllables) != len(template.slots):
                    template = base_template
            self._emit_for_epoch(epoch, RapEventType.PRESENTATION_ERROR, bar=bar, payload={
                "component": "fallback_catalog", "recoverable": True,
                "message": (
                    f"{error}; safety audio uses a prevalidated {topic!r} line "
                    "until generated lyrics are ready."
                )[:512],
            })
        plan = PlannedRapBar(
            bar=bar,
            segment=ScenarioSegment(bar, 1, topic, template.template_id, ()),
            template=template,
            analysis=line.analysis,
            scheduled=align_exact(line.analysis, materialize_flow(template, bar)),
            text=line.text,
            source=line.source,
            fallback_reason="lyric_missing",
        )
        prepared = self._render_plan(epoch, plan, timing)
        if prepared is not None:
            with self._lock:
                slot = self._slots.get(bar)
                if (epoch == self._epoch and not self._closed and slot is not None
                        and slot.committed is None and self._plan.for_bar(bar) == entry
                        and self._timing_current(bar, timing)):
                    slot.local, slot.local_plan = self._apply_mix(prepared), plan

    def _render_plan(self, epoch: int, plan: PlannedRapBar,
                     timing: BarTimingSnapshot | None = None) -> PreparedRapBar | None:
        with self._lock:
            if epoch != self._epoch or self._closed or not self._timing_current(plan.bar, timing):
                return None
        self._emit_for_epoch(epoch, RapEventType.AUDIO_RENDER_STARTED, bar=plan.bar, payload={"source": plan.source})
        try:
            prepared = (self._fallback_renderer.render(plan, timing=timing)
                        if timing is not None else self._fallback_renderer.render(plan))
            with self._lock:
                if (epoch != self._epoch or self._closed or not self._timing_current(plan.bar, timing)
                        or not self._compatible(prepared, plan.bar)):
                    return None
                prepared = self._apply_mix(prepared)
        except Exception as error:
            self._emit_for_epoch(epoch, RapEventType.PRESENTATION_ERROR, bar=plan.bar, payload={
                "component": "fallback_renderer", "message": f"{type(error).__name__}: {error}"[:512],
            })
            return None
        self._emit_for_epoch(epoch, RapEventType.AUDIO_RENDER_COMPLETED, bar=plan.bar, payload=self._audio_payload(prepared))
        return prepared

    def _await_any_fallback(self, bar: int, deadline: float, *, epoch: int | None = None) -> None:
        while self._clock() < deadline:
            with self._lock:
                if self._closed or self._stopping or (epoch is not None and epoch != self._epoch):
                    return
                slot = self._slot(bar)
                if slot.remote or slot.espeak or slot.local is not None:
                    return
            time.sleep(_RETRY_SLEEP_SECONDS)

    # -- commit -----------------------------------------------------------------------

    def _commit_locked(self, bar: int, *, tick: int | None) -> PreparedRapBar | None:
        slot = self._slot(bar)
        if slot.committed is not None:
            return slot.committed
        slot.remote = {key: audio for key, audio in slot.remote.items()
                       if not self._tempo_uncertain and self._compatible(audio, bar)}
        slot.espeak = {key: audio for key, audio in slot.espeak.items() if self._compatible(audio, bar)}
        if slot.local is not None and not self._compatible(slot.local, bar):
            slot.local = None
        latest = slot.latest
        choice, prepared, record = None, None, None
        if latest is not None and latest.lyric_id in slot.remote:
            choice, prepared, record = SOURCE_REMOTE_LATEST, slot.remote[latest.lyric_id], latest
        elif slot.remote:
            best = max(slot.remote, key=lambda lyric_id: slot.records[lyric_id].revision)
            choice, prepared, record = SOURCE_REMOTE_OLDER, slot.remote[best], slot.records[best]
        elif latest is not None and latest.lyric_id in slot.espeak:
            choice, prepared, record = SOURCE_ESPEAK_STREAM, slot.espeak[latest.lyric_id], latest
        elif slot.espeak:
            best = max(slot.espeak, key=lambda lyric_id: slot.records[lyric_id].revision)
            choice, prepared, record = SOURCE_ESPEAK_STREAM, slot.espeak[best], slot.records[best]
        elif slot.local is not None:
            choice, prepared = SOURCE_ESPEAK_LOCAL, slot.local
        if prepared is None:
            prepared = self._emergency_bar(bar)
            choice = SOURCE_DRUMS_ONLY if prepared.source == SOURCE_DRUMS_ONLY else SOURCE_SILENCE
            self._emit(RapEventType.PRESENTATION_ERROR, bar=bar, payload={
                "component": "bar_stream", "message": f"no compatible vocal audio was ready at commit; playing {choice}",
            })
        fallback_reason = ("tempo_ack_uncertain" if self._tempo_uncertain else
                           None if choice == SOURCE_REMOTE_LATEST else choice)
        prepared = replace(prepared, fallback_reason=fallback_reason,
                           timing=self._timeline.snapshot(bar, self._audio_format.sample_rate_hz)
                           if self._timing_enabled else prepared.timing)
        slot.committed, slot.committed_choice = prepared, choice
        slot.sources.clear()
        self._playhead = max(self._playhead, bar)
        if prepared.text:
            self._history.append((bar, prepared.text))
        lyric_id = record.lyric_id if record is not None else None
        local_plan = slot.local_plan if choice == SOURCE_ESPEAK_LOCAL else None
        played_flow = (record.flow_template if record is not None else
                       local_plan.template if local_plan is not None else self._plan.for_bar(bar).flow_template)
        played_topic = (record.topic if record is not None else
                        local_plan.segment.topic if local_plan is not None else self._plan.for_bar(bar).topic)
        payload = {
            "text": prepared.text,
            "source": prepared.source,
            "fallback": choice != SOURCE_REMOTE_LATEST,
            "fallback_reason": fallback_reason,
            "choice": choice,
            "lyric_id": lyric_id,
            "revision": record.revision if record is not None else None,
            "latest_revision": latest.revision if latest is not None else None,
            "topic": played_topic,
            "template_id": played_flow.template_id,
            "flow": flow_template_payload(played_flow),
            "scheduled_syllables": scheduled_syllables_payload(
                record.scheduled if record is not None else prepared.scheduled, bar=bar,
            ),
            "remote_ready_lead_ms": (
                (self._commit_time_locked(bar) - slot.remote_ready_at[lyric_id]) * 1000.0
                if lyric_id is not None and lyric_id in slot.remote_ready_at and self._clock_origin is not None
                else None
            ),
            "continued": slot.remote_continued.get(lyric_id) if lyric_id is not None else None,
            "timing": prepared.timing.to_payload() if prepared.timing is not None else None,
        }
        self._emit(RapEventType.BAR_FROZEN, bar=bar, tick=tick, request_id=lyric_id, payload=payload)
        if choice != SOURCE_REMOTE_LATEST:
            self._emit(RapEventType.FALLBACK_ACTIVATED, bar=bar, tick=tick, request_id=lyric_id, payload=payload)
        self._emit(RapEventType.BAR_AUDIO_COMMITTED, bar=bar, tick=tick, request_id=lyric_id,
                   payload={**self._audio_payload(prepared), "coordinator_epoch": self._epoch, "choice": choice,
                            "timing_evidence": prepared_bar_timing_payload(prepared)})
        timing = slot.remote_timing.get(lyric_id or "", {})
        self._emit(RapEventType.CHUNK_COMMITTED, bar=bar, tick=tick, request_id=lyric_id,
                   payload=self._bar_payload(record, state="committed", decision=choice, timing=timing,
                                             text=prepared.text, bar=bar))
        try:
            self._lyric_stream.send({"type": "committed", "bar": bar, "lyric_id": lyric_id, "source": choice, "text": prepared.text})
        except Exception:  # the server learns the playhead again from the next commit
            pass
        for old in [item for item in self._slots if item < bar - 8]:
            self._slots.pop(old, None)
        return prepared

    # -- helpers ----------------------------------------------------------------------

    def _emergency_bar(self, bar: int) -> PreparedRapBar:
        render = getattr(self._fallback_renderer, "render_drums_only", None)
        if self._timing_enabled and callable(render):
            try:
                timing = self._timeline.snapshot(bar, self._audio_format.sample_rate_hz)
                prepared = render(bar=bar, template=self._plan.for_bar(bar).flow_template, timing=timing)
                if not self._compatible(prepared, bar):
                    raise ValueError("drums fallback returned incompatible timing")
                return self._apply_mix(prepared)
            except Exception as error:
                self._emit(RapEventType.PRESENTATION_ERROR, bar=bar, payload={
                    "component": "drums_only", "message": f"{type(error).__name__}: {error}"[:512]})
        return self._silent_bar(bar)

    def _silent_bar(self, bar: int) -> PreparedRapBar:
        timing = self._timeline.snapshot(bar, self._audio_format.sample_rate_hz) if self._timing_enabled else None
        frames = bar_frame_count(bar, timing or self._tempo, self._audio_format)
        width = self._audio_format.channels * self._audio_format.sample_width_bytes
        return PreparedRapBar(
            bar=bar, text="", source=SOURCE_SILENCE, fallback_reason=SOURCE_SILENCE, scheduled=(),
            audio=PcmAudio(self._audio_format, frames, bytes(frames * width)),
            diagnostics=(), warnings=(), render_latency_ms=0.0,
            timing=timing,
        )

    def _timing_current(self, bar: int, timing: BarTimingSnapshot | None) -> bool:
        if not self._timing_enabled:
            return timing is None
        if timing is None and bar < self._legacy_timing_before_bar:
            return (self._timeline.snapshot(bar, self._audio_format.sample_rate_hz)
                    == self._initial_timeline.snapshot(bar, self._audio_format.sample_rate_hz))
        return timing == self._timeline.snapshot(bar, self._audio_format.sample_rate_hz)

    def _compatible(self, prepared: PreparedRapBar, destination_bar: int | None = None) -> bool:
        if destination_bar is not None and prepared.bar != destination_bar:
            return False
        if not self._timing_enabled:
            return True
        timing = self._timeline.snapshot(prepared.bar, self._audio_format.sample_rate_hz)
        return ((prepared.timing == timing or prepared.timing is None and self._timing_current(prepared.bar, None))
                and prepared.audio.format == self._audio_format
                and prepared.audio.frame_count == timing.frame_count)

    def _base_plan(self) -> LivePlan:
        def base(bar: int):
            segment = self._scenario.segment_for_bar(bar)
            return segment.topic, self._templates.get(segment.template_id)

        return LivePlan(base)

    def _start_message(self, *, start_bar: int) -> dict[str, object]:
        segments = [
            {
                "start_bar": segment.start_bar,
                "bars": segment.bars,
                "topic": segment.topic,
                "flow_template": _flow_payload(self._templates.get(segment.template_id)),
            }
            for segment in self._scenario.segments
        ]
        return {
            "type": "start",
            "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
            "tempo_bpm": self._tempo.bpm,
            "start_bar": start_bar,
            "seed": self._seed,
            "loop": self._scenario.loop,
            "segments": segments,
            "edits": [edit.to_payload() for edit in self._plan.edits],
            "history": [{"bar": bar, "text": text} for bar, text in self._history[-4:]],
            "lyric_lead": self._lyric_lead,
            "audio_lead": self._audio_lead,
            "capabilities": list(LYRIC_STREAM_CAPABILITIES),
        }

    def _slot(self, bar: int) -> _Slot:
        slot = self._slots.get(bar)
        if slot is None:
            slot = _Slot(bar)
            self._slots[bar] = slot
        return slot

    def _record(self, bar: int, lyric_id: str) -> BarLyricRecord | None:
        with self._lock:
            slot = self._slots.get(bar)
            return slot.records.get(lyric_id) if slot is not None else None

    def _within_limit(self, bar: int) -> bool:
        return self._planning_bar_limit is None or bar < self._planning_bar_limit

    def _commit_time_locked(self, bar: int) -> float | None:
        if self._clock_origin is None:
            return None
        return self._clock_origin + self._timeline.tick_to_seconds(max(0, bar * self._tempo.ticks_per_bar - 1))

    def _lead_ms_locked(self, bar: int) -> float | None:
        commit_at = self._commit_time_locked(bar)
        return None if commit_at is None else (commit_at - self._clock()) * 1000.0

    def _close_stream(self) -> None:
        try:
            self._lyric_stream.send({"type": "stop"})
        except Exception:
            pass
        try:
            self._lyric_stream.close()
        except Exception:
            pass

    def _emit(self, event_type: RapEventType, **kwargs: object) -> None:
        if self._publisher is not None:
            self._publisher.emit(event_type, **kwargs)

    def _emit_for_epoch(self, epoch: int, event_type: RapEventType, **kwargs: object) -> None:
        with self._lock:
            if epoch == self._epoch and not self._closed:
                self._emit(event_type, **kwargs)

    @staticmethod
    def _audio_payload(prepared: PreparedRapBar) -> dict[str, object]:
        return {
            "source": prepared.source,
            "warnings": [warning.code.value for warning in prepared.warnings],
            "render_latency_ms": prepared.render_latency_ms,
            "frame_count": prepared.audio.frame_count,
        }

    def _bar_payload(
        self,
        record: BarLyricRecord | None,
        *,
        state: str,
        decision: str,
        timing: Mapping[str, float] | None = None,
        failure: str | None = None,
        text: str | None = None,
        bar: int | None = None,
    ) -> dict[str, object]:
        bar_index = record.bar if record is not None else bar
        raw = {
            "state": state,
            "renderer_decision": decision,
            "coordinator_epoch": self._epoch,
            "chunk_index": bar_index,
            "bars": [bar_index],
            "selected_lines": [text or (record.text if record is not None else "")],
            "flows": [flow_template_payload(record.flow_template)] if record is not None else [],
            "stage_timings_ms": dict(timing or {}),
            "failure_reason": failure,
            "hashes": {"lyric_id": record.lyric_id} if record is not None else {},
        }
        return bounded_chunk_event_payload(raw)
