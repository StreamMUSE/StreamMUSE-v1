"""Alternating keyboard-flow turns, separate from continuous lyric streaming.

The remote session's bars are logical response ordinals. Only generated lyric
deliveries commit that cursor; local drums and flow echoes consume physical
playback bars without changing the remote session's progress.
"""

from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from math import isfinite
from threading import Event, RLock
import time
from typing import Callable, Mapping

from streammuse.application.rap.audio_rendering import bar_frame_count
from streammuse.application.rap.chunk_audio import remix_prepared_bar
from streammuse.application.rap.live_plan import LivePlan
from streammuse.application.rap.monitoring_payloads import flow_template_payload, prepared_bar_timing_payload, scheduled_syllables_payload
from streammuse.application.rap.realtime import PlannedRapBar
from streammuse.domain.rap import ScenarioSegment, materialize_flow
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapMixConfig
from streammuse.domain.rap.events import RapEventType
from streammuse.domain.rap.interaction import CaptureWindow, InteractionPhase, TapRecord, TurnRequest, quantize_capture
from streammuse.domain.rap.lyric_stream import LYRIC_STREAM_SCHEMA_VERSION, BarLyricRecord, PlanEdit
from streammuse.domain.rap.remote_chunk import _flow_payload
from streammuse.domain.rap.timeline import RapTimeline, validate_live_bpm
from streammuse.domain.timing import Tempo


@dataclass(frozen=True)
class PreparedTurnBar:
    record: BarLyricRecord
    prepared: PreparedRapBar
    choice: str


def _relocate(prepared: PreparedRapBar, record: BarLyricRecord, physical_bar: int, *, frames: int, timing=None) -> PreparedRapBar:
    if prepared.text != record.text:
        raise ValueError("prepared audio must speak the matching generated lyric")
    if abs(prepared.audio.frame_count - frames) > 1:
        raise ValueError("prepared response duration does not match destination timing")
    width = prepared.audio.format.channels * prepared.audio.format.sample_width_bytes
    data = prepared.audio.data[:frames * width].ljust(frames * width, b"\0")
    slots = materialize_flow(record.flow_template, physical_bar)
    scheduled = tuple(replace(item, slot=slot) for item, slot in zip(record.scheduled, slots))
    stems = {}
    for name in ("vocal_stem", "drum_stem", "chord_stem"):
        stem = getattr(prepared, name)
        if stem is not None:
            if stem.format != prepared.audio.format or abs(stem.frame_count - frames) > 1:
                raise ValueError("prepared stem duration does not match destination timing")
            stems[name] = PcmAudio(stem.format, frames, stem.data[:frames * width].ljust(frames * width, b"\0"))
    relocated = replace(
        prepared, bar=physical_bar, scheduled=scheduled,
        audio=PcmAudio(prepared.audio.format, frames, data),
        diagnostics=tuple(replace(item, bar=physical_bar) for item in prepared.diagnostics),
        warnings=tuple(replace(item, bar=physical_bar) for item in prepared.warnings), timing=timing, **stems,
    )
    return remix_prepared_bar(relocated, prepared.mix_config) if prepared.mix_config is not None else relocated


class ProgressiveLyricTurnAdapter:
    """Bounded v3 lyrics, raw remote audio and matching local eSpeak preparation.

    Network operations and preparation never run on the playback tick thread.
    Call ``poll`` each tick for the next physical destination; equal relative
    timing can reuse prepared audio, but a changed tempo requires preparation.
    The assembly must preflight ``bounded_turns_v1`` before enabling this mode.
    """

    def __init__(self, *, lyric_stream, audio_source, preparer, fallback_renderer, analyzer, tempo, seed,
                 fetch_executor: Executor | None = None, render_executor: Executor | None = None,
                 stream_executor: Executor | None = None, monotonic: Callable[[], float] = time.monotonic,
                 tempo_ack_timeout_seconds: float = 10.0):
        if not isfinite(tempo_ack_timeout_seconds) or not 0 < tempo_ack_timeout_seconds <= 30:
            raise ValueError("tempo acknowledgement timeout must be positive and at most thirty seconds")
        self._tempo_ack_timeout = tempo_ack_timeout_seconds
        self._stream, self._source, self._preparer = lyric_stream, audio_source, preparer
        self._fallback, self._analyzer, self._tempo, self._seed = fallback_renderer, analyzer, tempo, seed
        self._clock = monotonic
        self._fetch_executor = fetch_executor or ThreadPoolExecutor(max_workers=3, thread_name_prefix="rap-turn-fetch")
        self._render_executor = render_executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="rap-turn-render")
        self._stream_executor = stream_executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix="rap-turn-stream")
        self._lock = RLock()
        self._token = 0
        self._cancelled = Event()
        self._turn: TurnRequest | None = None
        self._records: dict[int, BarLyricRecord] = {}
        self._raw: dict[str, object] = {}
        self._prepared: dict[tuple, PreparedRapBar] = {}
        self._requested: set[tuple] = set()
        self._committed: set[int] = set()
        self._commit_outbox: dict[int, dict] = {}
        self._commit_sent: set[int] = set()
        self._connection_revision = 0
        self._error: str | None = None
        self._fatal_error: str | None = None
        self._closed = False
        self._remote_timeline = RapTimeline(tempo)
        self._tempo_pending = None
        self._tempo_seq = self._audio_generation = 0
        self._capabilities = frozenset()
        self._stream_ready = Event()
        self._mix_config = None
        self._alignment_revision = 0

    @property
    def error(self):
        with self._lock:
            return self._error

    @property
    def fatal_error(self):
        with self._lock:
            return self._fatal_error

    def begin_turn(self, turn: TurnRequest) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("generation adapter is closed")
            self._cancelled.set()
            self._cancelled = Event()
            self._token += 1
            token = self._token
            self._turn = turn
            self._records.clear()
            self._raw.clear()
            self._prepared.clear()
            self._requested.clear()
            self._committed.clear()
            self._commit_outbox.clear()
            self._commit_sent.clear()
            self._connection_revision = 0
            self._error = self._fatal_error = None
            self._remote_timeline = RapTimeline(self._tempo)
            self._tempo_seq = self._audio_generation = 0
            self._tempo_pending = None
            self._stream_ready = Event()
        self._stream_executor.submit(self._open, token, turn)

    def _open(self, token: int, turn: TurnRequest) -> None:
        if not self._active(token):
            return
        try:
            self._close_stream()
            if not self._active(token):
                return
            capabilities = self._stream.get_capabilities()
            if "bounded_turns_v1" not in capabilities:
                raise ValueError("server does not support bounded_turns_v1")
            with self._lock:
                if not self._active(token):
                    return
                self._capabilities = capabilities
            self._stream.open(turn.turn_id, {
                "type": "start", "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
                "capabilities": ["bounded_turns_v1", "tempo_updates_v1"],
                "tempo_bpm": self._tempo.bpm, "start_bar": 0, "seed": self._seed,
                "loop": False, "turn_id": turn.turn_id, "end_bar_exclusive": turn.response_bars,
                "segments": [{"start_bar": 0, "bars": turn.response_bars, "topic": turn.topic,
                              "flow_template": _flow_payload(turn.template)}],
                "edits": [], "history": [{"bar": index - len(turn.history[-4:]), "text": text} for index, text in enumerate(turn.history[-4:])],
                "lyric_lead": min(3, turn.response_bars), "audio_lead": min(2, turn.response_bars),
            }, lambda message: self._on_message(token, message))
            if not self._active(token):
                self._close_stream()
        except Exception as error:
            self._set_error(token, error, fatal=True)

    def _on_message(self, token: int, message: Mapping[str, object]) -> None:
        with self._lock:
            if not self._active(token):
                return
            kind = message.get("type")
            if kind == "started":
                if "bounded_turns_v1" not in (message.get("capabilities") or ()):
                    self._fatal_error = "server does not support bounded_turns_v1"
                else:
                    self._stream_ready.set()
                    self._connection_revision += 1
                    self._commit_sent.clear()
                    self._stream_executor.submit(self._send_commits, token, self._connection_revision)
                return
            if kind == "error":
                self._error = str(message.get("message", "lyric stream error"))[:512]
                return
            if kind != "lyric":
                return
            try:
                record = BarLyricRecord.from_payload(message["record"])
                turn = self._turn
                if record.session_id != turn.turn_id or not 0 <= record.bar < turn.response_bars:
                    raise ValueError("lyric record belongs to another turn or is outside its response span")
                if record.flow_template != turn.template:
                    raise ValueError("generated lyric does not match the captured flow")
                previous = self._records.get(record.bar)
                if record.bar in self._committed or (previous is not None and previous.revision >= record.revision):
                    return
                if previous is not None:
                    self._discard_audio(previous.lyric_id)
                self._records[record.bar] = record
                cancellation = self._cancelled
                audio_generation = self._audio_generation
            except Exception as error:
                self._error = str(error)[:512]
                return
        self._fetch_executor.submit(self._fetch, token, cancellation, record, audio_generation)

    def _fetch(self, token: int, cancellation: Event, record: BarLyricRecord, audio_generation: int) -> None:
        try:
            while self._active_record(token, record):
                with self._lock:
                    if audio_generation != self._audio_generation:
                        return
                fetched = self._source.fetch(record.session_id, record.bar, record.lyric_id, wait_ms=1000, accept_older=False)
                if fetched.status == "ready":
                    if fetched.lyric_id != record.lyric_id or fetched.manifest is None or fetched.source_wav is None:
                        raise ValueError("remote audio identity does not match the generated lyric")
                    if fetched.manifest.lyric_id != record.lyric_id or fetched.manifest.bar != record.bar:
                        raise ValueError("remote manifest does not match logical response identity")
                    with self._lock:
                        if self._active_record(token, record) and audio_generation == self._audio_generation:
                            self._raw[record.lyric_id] = fetched
                    return
                if fetched.status != "pending":
                    self._set_error(token, ValueError(f"remote response audio: {fetched.status}"))
                    return
                if cancellation.wait(0.05):
                    return
        except Exception as error:
            self._set_error(token, error)

    def poll(self, logical_bar: int, *, physical_bar: int, allow_fallback: bool, timing=None) -> PreparedTurnBar | None:
        with self._lock:
            if self._closed or self._cancelled.is_set() or self._fatal_error:
                return None
            target_bpm = self._tempo.bpm if timing is None else timing.bpm
            self._ensure_remote_tempo(logical_bar, target_bpm)
            if self._tempo_pending is not None or self._fatal_error:
                return None
            record = self._records.get(logical_bar)
            if record is None or logical_bar in self._committed:
                return None
            frames = bar_frame_count(physical_bar, self._tempo, AudioFormat()) if timing is None else timing.frame_count
            relative_timing = (self._tempo.bpm, frames) if timing is None else (timing.bpm, frames, tuple(timing.tick_offsets))
            token = self._token
            for choice in ("remote_latest", "espeak_stream_lyric"):
                key = self._preparation_key(record, relative_timing, choice)
                # Only the current destination and policy are reusable. Late jobs
                # cannot repopulate these caches after their keys are invalidated.
                obsolete = {old for old in self._requested
                            if old[0] == record.lyric_id and old[-1] == choice and old != key}
                self._requested.difference_update(obsolete)
                for old in obsolete:
                    self._prepared.pop(old, None)
                raw = self._raw.get(record.lyric_id)
                remote_compatible = raw is not None and raw.manifest.tempo_bpm == target_bpm
                if key not in self._requested and (choice != "remote_latest" or remote_compatible):
                    self._requested.add(key)
                    self._render_executor.submit(self._prepare, token, record, key, physical_bar, timing)
            choices = ("remote_latest", "espeak_stream_lyric") if allow_fallback else ("remote_latest",)
            for choice in choices:
                cached = self._prepared.get(self._preparation_key(record, relative_timing, choice))
                if cached is not None:
                    try:
                        relocated = _relocate(cached, record, physical_bar, frames=frames, timing=timing)
                    except ValueError as error:
                        self._set_error(token, error)
                        continue
                    return PreparedTurnBar(record, replace(relocated, fallback_reason=None if choice == "remote_latest" else choice), choice)
        return None

    def _preparation_key(self, record, relative_timing, choice):
        version = (self._audio_generation, self._alignment_revision) if choice == "remote_latest" else None
        return (record.lyric_id, relative_timing, version, choice)

    def _prepare(self, token, record, key, physical_bar, timing) -> None:
        with self._lock:
            if not self._active_record(token, record) or key not in self._requested:
                return
            raw, preparer = self._raw.get(record.lyric_id), self._preparer
        try:
            kwargs = {} if timing is None else {"timing": timing}
            if key[-1] == "remote_latest":
                if raw.manifest.lyric_id != record.lyric_id or raw.manifest.bar != record.bar:
                    raise ValueError("source audio does not match logical lyric identity")
                slots = materialize_flow(record.flow_template, physical_bar)
                local_record = replace(record, bar=physical_bar,
                                       scheduled=tuple(replace(item, slot=slot) for item, slot in zip(record.scheduled, slots)))
                local_manifest = replace(raw.manifest, bar=physical_bar, lyric_id=local_record.lyric_id)
                prepared = preparer.prepare(local_record, local_manifest, raw.source_wav, **kwargs)
            else:
                slots = materialize_flow(record.flow_template, physical_bar)
                scheduled = tuple(replace(item, slot=slot) for item, slot in zip(record.scheduled, slots))
                plan = PlannedRapBar(
                    bar=physical_bar,
                    segment=ScenarioSegment(physical_bar, 1, record.topic, record.flow_template.template_id, (record.text,)),
                    template=record.flow_template, analysis=self._analyzer.analyze(record.text), scheduled=scheduled,
                    text=record.text, source="lyric_stream_espeak", fallback_reason="remote_pending",
                )
                prepared = self._fallback.render(plan, **kwargs)
            if prepared.text != record.text:
                raise ValueError("prepared audio does not match generated text")
            with self._lock:
                if self._active_record(token, record) and key in self._requested:
                    self._prepared[key] = prepared if self._mix_config is None else remix_prepared_bar(prepared, self._mix_config)
        except Exception as error:
            with self._lock:
                if key in self._requested:
                    self._set_error(token, error)

    def set_alignment_policy(self, policy: str) -> int:
        """Invalidate local renders, not in-flight fetches or retained speech."""
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot change alignment on a closed adapter")
            switch = getattr(self._preparer, "with_alignment_policy", None)
            if not callable(switch):
                raise ValueError("audio preparer does not support alignment policy changes")
            self._preparer = switch(policy)
            self._alignment_revision += 1
            self._prepared = {key: value for key, value in self._prepared.items() if key[-1] != "remote_latest"}
            self._requested = {key for key in self._requested if key[-1] != "remote_latest"}
            return self._alignment_revision

    def _discard_audio(self, lyric_id):
        self._raw.pop(lyric_id, None)
        self._prepared = {key: value for key, value in self._prepared.items() if key[0] != lyric_id}
        self._requested = {key for key in self._requested if key[0] != lyric_id}

    def require_tempo_support(self):
        capabilities = self._capabilities or self._stream.get_capabilities()
        if "tempo_updates_v1" not in capabilities:
            raise ValueError("server does not support tempo_updates_v1")

    def set_mix_config(self, config):
        if not isinstance(config, RapMixConfig):
            raise ValueError("mix configuration must be RapMixConfig")
        with self._lock:
            for renderer in (self._fallback, self._preparer):
                if not callable(getattr(renderer, "set_mix_config", None)):
                    raise ValueError("audio adapter does not support live mix changes")
            remixed = {key: remix_prepared_bar(prepared, config) for key, prepared in self._prepared.items()}
            self._fallback.set_mix_config(config)
            self._preparer.set_mix_config(config)
            self._prepared = remixed
            self._mix_config = config

    def set_default_tempo(self, bpm):
        validate_live_bpm(bpm)
        with self._lock:
            self._tempo = Tempo(bpm, 4, 4)

    def _ensure_remote_tempo(self, logical_bar, bpm):
        if self._remote_timeline.snapshot(logical_bar, 48000).bpm == bpm or self._tempo_pending is not None:
            return
        self._tempo_seq += 1
        self._audio_generation += 1
        self._tempo_pending = (self._tempo_seq, logical_bar, bpm)
        token, audio_generation = self._token, self._audio_generation
        self._raw.clear()
        for key in tuple(self._prepared):
            if key[-1] == "remote_latest":
                self._prepared.pop(key)
        self._requested = {key for key in self._requested if key[-1] != "remote_latest"}
        self._stream_executor.submit(self._negotiate_tempo, token, audio_generation, *self._tempo_pending)

    def _negotiate_tempo(self, token, audio_generation, edit_seq, logical_bar, bpm):
        if not self._active(token):
            return
        try:
            if not self._stream_ready.wait(self._tempo_ack_timeout) or not self._active(token):
                raise TimeoutError("lyric stream has not negotiated tempo support")
            ack = self._stream.request_tempo({"edit_seq": edit_seq, "from_bar": logical_bar, "bpm": bpm}, timeout=self._tempo_ack_timeout)
            if (ack.get("bpm") != bpm or ack.get("effective_from_bar") != logical_bar
                    or ack.get("edit_seq") != edit_seq or ack.get("revision") != self._remote_timeline.revision + 1):
                raise ValueError("remote tempo acknowledgement does not match the logical response")
            with self._lock:
                if not self._active(token) or audio_generation != self._audio_generation:
                    return
                self._remote_timeline = self._remote_timeline.with_tempo(bpm, logical_bar)
                self._tempo_pending = None
                records = tuple(record for bar, record in self._records.items() if bar >= logical_bar and bar not in self._committed)
                cancellation = self._cancelled
            for record in records:
                self._fetch_executor.submit(self._fetch, token, cancellation, record, audio_generation)
        except Exception as error:
            self._set_error(token, error, fatal=True)

    def commit(self, response: PreparedTurnBar, *, physical_bar: int) -> None:
        with self._lock:
            record = response.record
            if not self._active_record(self._token, record) or record.bar in self._committed:
                raise ValueError("cannot commit stale or duplicate response")
            self._committed.add(record.bar)
            self._discard_audio(record.lyric_id)
            self._commit_outbox[record.bar] = {"type": "committed", "bar": record.bar, "lyric_id": record.lyric_id,
                                               "source": response.choice, "text": record.text}
            token = self._token
            connection_revision = self._connection_revision
        self._stream_executor.submit(self._send_commits, token, connection_revision)

    def _send_commits(self, token, connection_revision):
        # Retain the bounded turn's identities: transport success is not a server ACK.
        # A reconnect replays all commits; matching duplicates are server-idempotent.
        with self._lock:
            pending = sorted(self._commit_outbox.items())
        for bar, message in pending:
            with self._lock:
                if not self._active(token) or connection_revision != self._connection_revision:
                    return
                if bar in self._commit_sent:
                    continue
            try:
                self._stream.send(message)
            except Exception as error:
                self._set_error(token, error)
                return
            with self._lock:
                if self._active(token) and connection_revision == self._connection_revision:
                    self._commit_sent.add(bar)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled.set()
            self._token += 1
            self._turn = None
            self._records.clear()
            self._raw.clear()
            self._prepared.clear()
            self._requested.clear()
            self._error = self._fatal_error = None
            self._commit_outbox.clear()
            self._commit_sent.clear()
        if not self._closed:
            self._stream_executor.submit(self._close_stream)

    def close(self) -> None:
        self.cancel()
        with self._lock:
            self._closed = True
        for executor in (self._stream_executor, self._fetch_executor, self._render_executor):
            executor.shutdown(wait=False, cancel_futures=False)
        self._source.close()

    def _close_stream(self):
        try:
            self._stream.send({"type": "stop"})
        except Exception:
            pass
        try:
            self._stream.close()
        except Exception:
            pass

    def _active(self, token):
        with self._lock:
            return token == self._token and not self._closed and not self._cancelled.is_set()

    def _active_record(self, token, record):
        with self._lock:
            latest = self._records.get(record.bar)
            return self._active(token) and record.bar not in self._committed and latest is not None and latest.lyric_id == record.lyric_id

    def _set_error(self, token, error, *, fatal=False):
        with self._lock:
            if self._active(token):
                self._error = f"{type(error).__name__}: {error}"[:512]
                if fatal:
                    self._fatal_error = self._error


class AlternatingRapController:
    """Bar-quantized capture/echo/response controller with the runtime lifecycle."""

    def __init__(self, *, tempo, scenario, templates, generation, echo_renderer, publisher, enqueue,
                 session_id, response_bars=2, initial_epoch=0, wait_timeout_seconds=30.0,
                 monotonic=time.monotonic, audio_format=None, timeline=None, capture_grace_seconds=0.1,
                 on_timeline_changed=None, planning_bar_limit=None):
        if tempo.ticks_per_bar != 16 or tempo.ticks_per_beat != 4:
            raise ValueError("alternating interaction requires sixteen-tick 4/4 bars")
        if not isfinite(wait_timeout_seconds) or not 0 < wait_timeout_seconds <= 30:
            raise ValueError("wait timeout must be positive and at most thirty seconds")
        if not isinstance(initial_epoch, int) or isinstance(initial_epoch, bool) or initial_epoch < 0:
            raise ValueError("initial_epoch must be a nonnegative integer")
        if not isfinite(capture_grace_seconds) or not 0 <= capture_grace_seconds <= 0.5:
            raise ValueError("capture_grace_seconds must be between zero and 0.5")
        if planning_bar_limit is not None and (not isinstance(planning_bar_limit, int)
                or isinstance(planning_bar_limit, bool) or planning_bar_limit <= 0):
            raise ValueError("planning_bar_limit must be a positive integer or None")
        limits = [value for value in (planning_bar_limit, None if scenario.loop else scenario.total_bars) if value is not None]
        self._planning_bar_limit = min(limits) if limits else None
        self._tempo, self._scenario, self._templates = tempo, scenario, templates
        self._generation, self._echo, self._publisher, self._enqueue = generation, echo_renderer, publisher, enqueue
        self._base_session_id, self._epoch, self._clock = session_id, initial_epoch, monotonic
        self._timeout, self._format = wait_timeout_seconds, audio_format or AudioFormat()
        self._timeline = timeline or RapTimeline(tempo)
        self._capture_grace = capture_grace_seconds
        self._on_timeline_changed = on_timeline_changed
        self._last_tempo_edit = None
        self._lock = RLock()
        self._started = self._stopping = self._closed = False
        self._stop_successor_bar = None
        self._origin = None
        self._last_tick = self._queued_bar = -1
        self._phase = InteractionPhase.DRUMS_IDLE
        self._window = self._capture = None
        self._taps = []
        self._event_ids = set()
        self._rejected_taps = {}
        self._turn_number = self._delivered = self._replays = 0
        self._response_cursor = 0
        self._pending_deliveries = []
        self._last_generation_error = None
        self._turn_id = self._last_error = None
        self._wait_started = self._completion_bar = self._intended_bar = None
        self._history = []
        self._plan = self._base_plan()
        self._edit_seq = 0
        self._last_edit = None
        self.set_response_bars(response_bars)

    @property
    def scenario(self):
        return self._scenario

    @property
    def coordinator_epoch(self):
        with self._lock:
            return self._epoch

    @property
    def timeline(self):
        with self._lock:
            return self._timeline

    @property
    def terminal_bar_limit(self):
        return self._planning_bar_limit

    @property
    def session_id(self):
        return f"{self._base_session_id}-e{self._epoch}"

    @property
    def captured_template(self):
        with self._lock:
            return None if self._capture is None else self._capture.template

    @property
    def next_editable_bar(self):
        with self._lock:
            return self._queued_bar + 1

    @property
    def plan_control_state(self):
        with self._lock:
            return {"next_editable_bar": self.next_editable_bar, "last_edit": self._last_edit}

    def set_response_bars(self, response_bars):
        with self._lock:
            if not isinstance(response_bars, int) or isinstance(response_bars, bool) or not 1 <= response_bars <= 8:
                raise ValueError("response_bars must be between one and eight")
            if self._started and not self._stopping:
                raise RuntimeError("response length can only change while stopped")
            self._response_bars = response_bars

    def start(self):
        with self._lock:
            if self._closed or self._started:
                return
            self._started = True
            self._origin = self._clock()
            self._enqueue_local(0)

    def arm_capture(self):
        with self._lock:
            if not self._started or self._stopping or self._closed:
                raise RuntimeError("capture requires running playback")
            if self._phase != InteractionPhase.DRUMS_IDLE:
                return self.interaction_state()
            bar = max(0, self._last_tick) // 16 + 1
            if not self._within_limit(bar + 2):
                self._last_error = "playback_limit_reached"
                return self.interaction_state()
            self._window = CaptureWindow(bar, self._bar_time(bar), self._bar_time(bar + 1))
            self._taps, self._event_ids = [], set()
            self._rejected_taps = {}
            self._capture = None
            self._last_error = None
            self._last_generation_error = None
            self._turn_id = None
            self._wait_started = self._completion_bar = self._intended_bar = None
            self._delivered = self._response_cursor = self._replays = 0
            self._pending_deliveries.clear()
            self._phase = InteractionPhase.ARMED
            return self.interaction_state()

    def record_tap(self, *, server_time_seconds, client_time_ms=None, uncertainty_ms=0.0, event_id=None, repeat=False):
        with self._lock:
            tap = TapRecord(server_time_seconds, self._clock(), client_time_ms, uncertainty_ms, event_id)
            reason = None
            if repeat:
                reason = "key_repeat"
            elif self._stopping or self._closed or self._phase not in (InteractionPhase.ARMED, InteractionPhase.CAPTURING):
                reason = "not_capturing"
            elif event_id is not None and event_id in self._event_ids:
                reason = "duplicate_event"
            elif not self._window.start_seconds <= server_time_seconds < self._window.end_seconds:
                reason = "outside_capture"
            elif self._clock() > self._window.end_seconds + self._capture_grace:
                reason = "capture_grace_expired"
            elif server_time_seconds > self._clock() + uncertainty_ms / 1000 + 0.05:
                reason = "future_timestamp"
            elif len(self._taps) >= 1024:
                reason = "capture_tap_limit"
            if reason is None:
                self._taps.append(tap)
                if event_id is not None:
                    self._event_ids.add(event_id)
            elif self._window is not None and self._phase in (InteractionPhase.ARMED, InteractionPhase.CAPTURING):
                self._rejected_taps[reason] = self._rejected_taps.get(reason, 0) + 1
            return {**self.interaction_state(), "accepted": reason is None, "reason": reason}

    def on_tick(self, tick):
        with self._lock:
            if not self._started or self._stopping or self._closed or tick <= self._last_tick:
                return
            if self._last_tick < 0:
                self._origin = self._clock() - self._tick_seconds(tick)
                if self._phase == InteractionPhase.ARMED:
                    self._window = CaptureWindow(self._window.bar, self._bar_time(self._window.bar), self._bar_time(self._window.bar + 1))
            self._last_tick = tick
            self._confirm_started(tick // 16)
            self._emit(RapEventType.TICK, bar=tick // 16, tick=tick, payload={"beat": tick % 16 // 4, "tick_in_beat": tick % 4})
            if self._phase == InteractionPhase.ARMED and tick >= self._window.bar * 16:
                self._phase = InteractionPhase.CAPTURING
            if (self._phase == InteractionPhase.CAPTURING and tick >= (self._window.bar + 1) * 16
                    and self._clock() >= self._window.end_seconds + self._capture_grace):
                self._finish_capture()
            if self._completion_bar is not None and tick >= self._completion_bar * 16:
                self._phase = InteractionPhase.DRUMS_IDLE
                self._completion_bar = None
                self._generation.cancel()
            physical_bar = tick // 16 + 1
            if not self._within_limit(physical_bar):
                if (tick % 16 == 15 or not self._within_limit(tick // 16)) and self._phase in (InteractionPhase.WAIT_REPLAY, InteractionPhase.MODEL_PLAY) and self._response_cursor < self._response_bars:
                    self._fail_turn("playback_limit_reached")
                return
            candidate = None
            active = self._phase in (InteractionPhase.WAIT_REPLAY, InteractionPhase.MODEL_PLAY) and self._response_cursor < self._response_bars
            if active:
                expired = self._wait_started is not None and self._clock() - self._wait_started >= self._timeout
                candidate = self._generation.poll(self._response_cursor, physical_bar=physical_bar,
                                                  timing=self._timing(physical_bar), allow_fallback=self._response_cursor > 0 or expired)
                if self._generation.fatal_error:
                    self._fail_turn(self._generation.fatal_error)
                elif expired and candidate is None:
                    self._fail_turn("response_timeout")
            if tick % 16 == 15 and physical_bar > self._queued_bar:
                if candidate is not None and self._phase in (InteractionPhase.WAIT_REPLAY, InteractionPhase.MODEL_PLAY):
                    self._deliver(candidate, physical_bar)
                elif self._phase in (InteractionPhase.WAIT_REPLAY, InteractionPhase.MODEL_PLAY) and self._response_cursor < self._response_bars:
                    self._phase = InteractionPhase.WAIT_REPLAY
                    self._replays += 1
                    self._enqueue_local(physical_bar, echo=True)
                else:
                    self._enqueue_local(physical_bar)

    def _finish_capture(self):
        self._turn_number += 1
        self._capture = quantize_capture(self._window, tuple(self._taps), template_id=f"{self.session_id}-capture-{self._turn_number}")
        self._turn_id = f"{self.session_id}-turn-{self._turn_number}"
        self._emit(RapEventType.CAPTURE_FINALIZED, bar=self._window.bar, payload={
            "turn_id": self._turn_id, "capture_window": asdict(self._window),
            "capture_grace_seconds": self._capture_grace, "finalized_time_seconds": self._clock(),
            "raw_taps": [asdict(tap) for tap in self._capture.raw_taps],
            "quantized_ticks": list(self._capture.quantized_ticks),
            "slot_ticks": [] if self._capture.template is None else [slot.tick_in_bar for slot in self._capture.template.slots],
            "collision_count": self._capture.collision_count,
            "rejected_count": sum(self._rejected_taps.values()),
            "rejected_reasons": [{"reason": reason, "count": count} for reason, count in sorted(self._rejected_taps.items())],
            "template_id": None if self._capture.template is None else self._capture.template.template_id,
            "flow": None if self._capture.template is None else flow_template_payload(self._capture.template),
            "error": "empty_capture" if self._capture.template is None else None,
        })
        if self._capture.template is None:
            self._phase = InteractionPhase.DRUMS_IDLE
            self._last_error = "empty_capture"
            return
        self._delivered = self._response_cursor = self._replays = 0
        self._pending_deliveries.clear()
        self._completion_bar = None
        self._intended_bar = self._window.bar + 1
        self._wait_started = self._clock()
        topic = self._plan.for_bar(self._window.bar).topic
        self._generation.set_default_tempo(self._timing(self._window.bar + 1).bpm)
        self._generation.begin_turn(TurnRequest(self._turn_id, self._epoch, self._capture.template, topic, self._response_bars, tuple(self._history[-4:])))
        self._phase = InteractionPhase.WAIT_REPLAY

    def _deliver(self, response, physical_bar):
        logical_bar = self._response_cursor
        self._enqueue(response.prepared)
        self._generation.commit(response, physical_bar=physical_bar)
        self._queued_bar = physical_bar
        self._phase = InteractionPhase.MODEL_PLAY
        self._response_cursor += 1
        self._pending_deliveries.append((physical_bar, response.record.text))
        payload = {
            **self._mapping_payload(physical_bar, logical_bar),
            "text": response.record.text, "source": response.prepared.source,
            "choice": response.choice, "fallback": response.choice != "remote_latest",
            "fallback_reason": response.prepared.fallback_reason, "lyric_id": response.record.lyric_id,
            "revision": response.record.revision, "topic": response.record.topic,
            "template_id": response.record.flow_template.template_id, "flow": flow_template_payload(response.record.flow_template),
            "scheduled_syllables": scheduled_syllables_payload(response.prepared.scheduled, bar=physical_bar),
            "timing_evidence": prepared_bar_timing_payload(response.prepared),
        }
        self._emit(RapEventType.BAR_FROZEN, bar=physical_bar, payload=payload)
        if response.choice != "remote_latest":
            self._emit(RapEventType.FALLBACK_ACTIVATED, bar=physical_bar, payload=payload)
        self._emit(RapEventType.BAR_AUDIO_COMMITTED, bar=physical_bar, payload={**payload, "frame_count": response.prepared.audio.frame_count})
        self._wait_started = self._bar_time(physical_bar + 1) - self._tick_duration(physical_bar)
        if self._response_cursor == self._response_bars:
            self._completion_bar = physical_bar + 1

    def _confirm_started(self, physical_bar):
        for bar, text in self._pending_deliveries:
            if bar <= physical_bar:
                self._delivered += 1
                self._history.append(text)
        self._history = self._history[-4:]
        self._pending_deliveries = [(bar, text) for bar, text in self._pending_deliveries if bar > physical_bar]

    def _within_limit(self, bar):
        return self._planning_bar_limit is None or bar < self._planning_bar_limit

    def _enqueue_local(self, bar, *, echo=False):
        if not self._within_limit(bar):
            return
        prepared = self._echo.render(bar=bar, tempo=self._tempo, audio_format=self._format,
                                     template=self.captured_template if echo else None, timing=self._timing(bar))
        if not echo and self._window is not None and bar == self._window.bar + 1 and self._phase == InteractionPhase.CAPTURING:
            prepared = replace(prepared, source="capture_transition")
        self._enqueue(prepared)
        self._queued_bar = bar
        payload = {**self._mapping_payload(bar, self._response_cursor), "text": "", "source": prepared.source,
                   "choice": prepared.source, "fallback": False, "replay_count": self._replays,
                   "frame_count": prepared.audio.frame_count,
                   "timing_evidence": prepared_bar_timing_payload(prepared)}
        if echo:
            payload["flow"] = flow_template_payload(self.captured_template)
        self._emit(RapEventType.BAR_FROZEN, bar=bar, payload=payload)
        self._emit(RapEventType.BAR_AUDIO_COMMITTED, bar=bar, payload=payload)

    def _mapping_payload(self, physical_bar, logical_bar):
        intended = None if self._intended_bar is None else self._intended_bar + logical_bar
        return {"turn_id": self._turn_id, "physical_bar": physical_bar, "logical_response_bar": logical_bar,
                "intended_physical_bar": intended, "deadline_missed": intended is not None and physical_bar > intended}

    def _fail_turn(self, reason):
        self._last_error = reason
        self._last_generation_error = self._generation.error
        self._phase = InteractionPhase.DRUMS_IDLE
        self._generation.cancel()
        self._emit(RapEventType.PRESENTATION_ERROR, bar=self._last_tick // 16,
                   payload={"component": "alternating_interaction", "message": reason,
                            "turn_id": self._turn_id, "delivered_response_bars": self._delivered, "incomplete": True})

    def interaction_state(self):
        with self._lock:
            now = self._clock()
            capture = self._capture
            if capture is None and self._window is not None:
                capture = quantize_capture(self._window, tuple(self._taps), template_id="capture-preview")
            clock = {"monotonic_seconds": now, "current_tick": self._last_tick,
                     "playback_seconds": None if self._origin is None else now - self._origin,
                     "clock_origin_seconds": self._origin, "tick_seconds": self._tick_duration(max(0, self._last_tick) // 16)}
            return {
                "supported": True, "mode": "alternating", "phase": self._phase.value, "coordinator_epoch": self._epoch,
                "turn_id": self._turn_id, "response_bars": self._response_bars,
                "delivered_response_bars": self._delivered, "logical_response_bar": self._response_cursor,
                "queued_response_bars": len(self._pending_deliveries), "terminal_bar_limit": self._planning_bar_limit,
                "physical_bar": max(0, self._last_tick) // 16, "replay_count": self._replays,
                "capture_bar": None if self._window is None else self._window.bar,
                "capture_start_seconds": None if self._window is None else self._window.start_seconds,
                "capture_end_seconds": None if self._window is None else self._window.end_seconds,
                "capture_grace_seconds": self._capture_grace,
                "capture_finalizes_at_seconds": None if self._window is None else self._window.end_seconds + self._capture_grace,
                "capture_finalizing": self._phase == InteractionPhase.CAPTURING and self._window is not None and now >= self._window.end_seconds,
                "transition_bar": None if self._window is None else self._window.bar + 1,
                "captured_ticks": [] if capture is None else list(capture.quantized_ticks),
                "raw_taps": [asdict(tap) for tap in self._taps],
                "collision_count": 0 if capture is None else capture.collision_count,
                "rejected_count": sum(self._rejected_taps.values()), "rejected_reasons": dict(self._rejected_taps),
                "flow": None if capture is None or capture.template is None else flow_template_payload(capture.template),
                "last_error": self._last_error, "generation_error": self._generation.error or self._last_generation_error,
                "wait_elapsed_seconds": (
                    max(0.0, now - self._wait_started)
                    if self._wait_started is not None and self._response_cursor < self._response_bars
                    and self._phase in (InteractionPhase.WAIT_REPLAY, InteractionPhase.MODEL_PLAY) else 0.0
                ),
                "monotonic_seconds": now, "current_tick": self._last_tick, "clock": clock,
                "running": self._started and not self._stopping and not self._closed,
            }

    def request_stop(self, *, successor_bar):
        with self._lock:
            if self._closed or not self._started or self._stopping:
                return
            self._stopping = True
            self._stop_successor_bar = successor_bar
            self._confirm_started(self._last_tick // 16 if successor_bar is None else successor_bar - 1)
            self._pending_deliveries.clear()
            self._response_cursor = self._delivered
            self._generation.cancel()
            self._phase = InteractionPhase.DRUMS_IDLE
            self._window = None
            self._completion_bar = None

    def resume_audio(self, bar):
        with self._lock:
            if not self._stopping or bar != self._stop_successor_bar:
                raise RuntimeError("controller is not stopped at this successor bar")
            self._enqueue_local(bar)

    def resume_after_stop(self):
        with self._lock:
            if self._closed or not self._stopping or self._stop_successor_bar is None:
                raise RuntimeError("controller is not stopped for resume")
            bar = self._stop_successor_bar
            self._origin = self._clock() - self._tick_seconds(bar * 16)
            self._last_tick = bar * 16 - 1
            self._stopping = False
            self._stop_successor_bar = None

    def reset(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot reset closed controller")
            self._generation.cancel()
            self._epoch += 1
            self._started = self._stopping = False
            self._stop_successor_bar = self._origin = None
            self._last_tick = self._queued_bar = -1
            self._phase = InteractionPhase.DRUMS_IDLE
            self._window = self._capture = None
            self._taps, self._event_ids = [], set()
            self._rejected_taps = {}
            self._turn_id = self._last_error = self._completion_bar = self._wait_started = self._intended_bar = None
            self._delivered = self._response_cursor = self._replays = 0
            self._pending_deliveries.clear()
            self._last_generation_error = None
            self._history.clear()
            self._plan = self._base_plan()
            self._edit_seq = 0
            self._last_edit = None
            return self._epoch

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
        self._generation.close()

    def update_plan(self, *, from_bar, topic=None, template_id=None):
        with self._lock:
            if template_id is not None:
                raise ValueError("alternating mode uses captured flow; template presets apply in continuous mode")
            self._edit_seq += 1
            edit = PlanEdit(self._edit_seq, from_bar, topic, None if template_id is None else self._templates.get(template_id))
            effective = max(from_bar, self.next_editable_bar)
            if self._window is not None and self._phase in (InteractionPhase.ARMED, InteractionPhase.CAPTURING):
                effective = max(effective, self._window.bar + 1)
            self._plan = self._plan.with_edit(replace(edit, from_bar=effective))
            self._last_edit = {"coordinator_epoch": self._epoch, "edit_seq": edit.edit_seq, "from_bar": from_bar,
                               "effective_from_bar": effective, "topic": topic,
                               "template_id": template_id, "application": "next_capture"}
            self._emit(RapEventType.PLAN_EDITED, bar=self.next_editable_bar, payload=self._last_edit)
            return edit

    def _base_plan(self):
        def base(bar):
            segment = self._scenario.segment_for_bar(bar)
            return segment.topic, self._templates.get(segment.template_id)

        return LivePlan(base)

    def set_timeline(self, timeline):
        with self._lock:
            if self._window is not None and self._phase in (InteractionPhase.ARMED, InteractionPhase.CAPTURING):
                old = self._timing(self._window.bar)
                new = timeline.snapshot(self._window.bar, self._format.sample_rate_hz)
                if old.start_frame != new.start_frame or old.end_frame != new.end_frame:
                    raise ValueError("cannot change timing of an armed or capturing bar")
            self._timeline = timeline

    @property
    def tempo_control_state(self):
        with self._lock:
            timing = self._timing(max(0, self._last_tick) // 16)
            return {"bpm": timing.bpm, "revision": self._timeline.revision, "last_edit": self._last_tempo_edit,
                    "requires_reset": False, "supported": True}

    def update_tempo(self, bpm, from_bar=None):
        validate_live_bpm(bpm)
        self._generation.require_tempo_support()
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot change timing on a closed controller")
            requested = self.next_editable_bar if from_bar is None else from_bar
            if not isinstance(requested, int) or isinstance(requested, bool) or requested < 0:
                raise ValueError("from_bar must be a nonnegative integer")
            effective = max(requested, self.next_editable_bar)
            if self._window is not None and self._phase in (InteractionPhase.ARMED, InteractionPhase.CAPTURING):
                effective = max(effective, self._window.bar + 1)
            timeline = self._timeline.with_tempo(bpm, effective)
            if self._on_timeline_changed is not None:
                self._on_timeline_changed(timeline)
            self._timeline = timeline
            self._last_tempo_edit = {
                "requested_from_bar": requested, "effective_from_bar": effective,
                "bpm": bpm, "revision": timeline.revision, "coordinator_epoch": self._epoch,
                "remote_application": "logical_response_destination",
            }
            return dict(self._last_tempo_edit)

    def update_alignment_policy(self, policy: str) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot change alignment on a closed controller")
            revision = self._generation.set_alignment_policy(policy)
            return {"effective_from_bar": self.next_editable_bar, "alignment_policy": policy,
                    "revision": revision, "coordinator_epoch": self._epoch}

    def update_mix(self, config):
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot change mix on a closed controller")
            self._generation.set_mix_config(config)
            self._echo.set_mix_config(config)
            return {"effective_from_bar": self.next_editable_bar, "mix_revision": config.revision,
                    "coordinator_epoch": self._epoch}

    def _timing(self, bar):
        return None if self._timeline is None else self._timeline.snapshot(bar, self._format.sample_rate_hz)

    def _tick_seconds(self, tick):
        if self._timeline is None:
            return self._tempo.tick_to_seconds(tick)
        timing = self._timing(tick // 16)
        return (timing.start_frame + timing.tick_offsets[tick % 16]) / timing.sample_rate_hz

    def _bar_time(self, bar):
        return self._origin + self._tick_seconds(bar * 16)

    def _tick_duration(self, bar):
        timing = self._timing(bar)
        return self._tempo.tick_to_seconds(1) if timing is None else (timing.end_frame - timing.start_frame) / timing.sample_rate_hz / 16

    def _emit(self, kind, *, bar=None, tick=None, payload=None):
        if self._publisher is not None:
            self._publisher.emit(kind, bar=bar, tick=self._last_tick if tick is None else tick,
                                 payload={**(payload or {}), "coordinator_epoch": self._epoch})
