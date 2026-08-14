"""Fake-clock acceptance coverage for the optional realtime rap audio path."""

from __future__ import annotations

import json
import struct
from collections import deque
from dataclasses import asdict
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event

from streammuse.application.rap.audio_coordination import BarAudioCoordinator
from streammuse.application.rap.audio_rendering import bar_start_frame
from streammuse.application.rap.bar_renderer import DeterministicRapBarRenderer
from streammuse.application.rap.monitoring import RapEventDispatcher, RapEventPublisher, RapStateProjector
from streammuse.application.rap.monitoring_payloads import flow_template_payload
from streammuse.application.rap.playback import RapPlaybackService
from streammuse.application.rap.realtime import RollingRapController
from streammuse.domain.rap import (
    AudioFormat,
    AudioPlaybackNotice,
    AudioPlaybackNoticeKind,
    AudioPlaybackSnapshot,
    CandidateBatch,
    CandidateRequest,
    PcmAudio,
    PlaybackState,
    PreparedRapBar,
    RenderedSyllable,
    RapEventType,
    ScoreWeights,
)
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.audio_output import CompositeAudioSink, Float32WavAudioSink
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.recorder import RapSessionRecorder, build_session_manifest
from streammuse.infrastructure.rap.scenarios import default_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


class ManualAudioSink:
    """Test-only sample clock; advancing it never opens a PortAudio device."""

    def __init__(self, audio_format: AudioFormat) -> None:
        self._format = audio_format
        self._queued: deque[PreparedRapBar] = deque()
        self._active: PreparedRapBar | None = None
        self._frame_in_bar = 0
        self._absolute_frame = 0
        self._state = PlaybackState.STOPPED
        self._stop_requested = False
        self._notices: SimpleQueue[AudioPlaybackNotice] = SimpleQueue()
        self.completed: list[PreparedRapBar] = []

    def start(self) -> None:
        if self._state == PlaybackState.CLOSED:
            raise RuntimeError("manual sink is closed")
        self._state = PlaybackState.RUNNING

    def enqueue(self, bar: PreparedRapBar) -> None:
        if bar.audio.format != self._format:
            raise ValueError("prepared bar format does not match manual sink")
        self._queued.append(bar)

    def request_stop_after_bar(self) -> None:
        if self._state not in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
            return
        if self._active is None:
            self._state = PlaybackState.STOPPED
            self._notice(AudioPlaybackNoticeKind.STOPPED, None, "playback stopped")
        else:
            self._stop_requested = True
            self._state = PlaybackState.STOP_REQUESTED

    def advance(self, frames: int) -> None:
        """Advance exact frames without gaps or wall-clock sleeps."""

        remaining = frames
        while remaining and self._state in (PlaybackState.RUNNING, PlaybackState.STOP_REQUESTED):
            if self._active is None:
                if not self._queued:
                    self._absolute_frame += remaining
                    return
                self._active = self._queued.popleft()
                self._frame_in_bar = 0
                self._notice(AudioPlaybackNoticeKind.BAR_STARTED, self._active.bar, "bar playback started")
            assert self._active is not None
            consumed = min(remaining, self._active.audio.frame_count - self._frame_in_bar)
            self._frame_in_bar += consumed
            self._absolute_frame += consumed
            remaining -= consumed
            if self._frame_in_bar != self._active.audio.frame_count:
                continue
            completed = self._active
            self.completed.append(completed)
            self._notice(AudioPlaybackNoticeKind.BAR_COMPLETED, completed.bar, "bar playback completed")
            self._active = None
            self._frame_in_bar = 0
            if self._stop_requested:
                self._stop_requested = False
                self._state = PlaybackState.STOPPED
                self._notice(AudioPlaybackNoticeKind.STOPPED, None, "playback stopped")

    def reset(self) -> None:
        self._queued.clear()
        self._active = None
        self._frame_in_bar = 0
        self._absolute_frame = 0
        self._stop_requested = False
        if self._state != PlaybackState.CLOSED:
            self._state = PlaybackState.STOPPED
        self._notices = SimpleQueue()

    def snapshot(self) -> AudioPlaybackSnapshot:
        return AudioPlaybackSnapshot(
            state=self._state,
            current_bar=self._active.bar if self._active is not None else None,
            frame_in_bar=self._frame_in_bar,
            absolute_frame=self._absolute_frame,
            queue_depth=len(self._queued),
            underrun_count=0,
        )

    def drain_notices(self) -> tuple[AudioPlaybackNotice, ...]:
        notices: list[AudioPlaybackNotice] = []
        while True:
            try:
                notices.append(self._notices.get_nowait())
            except Empty:
                return tuple(notices)

    def close(self) -> None:
        self._state = PlaybackState.CLOSED
        self._queued.clear()

    def _notice(self, kind: AudioPlaybackNoticeKind, bar: int | None, message: str) -> None:
        self._notices.put(
            AudioPlaybackNotice(
                kind=kind,
                bar=bar,
                absolute_frame=self._absolute_frame,
                queue_depth=len(self._queued),
                message=message,
            )
        )


class FakeSpeech:
    def synthesize(self, request) -> RenderedSyllable:
        audio = PcmAudio(AudioFormat(channels=1), 1, bytes(4))
        return RenderedSyllable(
            request=request,
            audio=audio,
            renderer_phonemes=(),
            pronunciation_source="fake",
            synthesis_latency_ms=0.0,
        )


class DelayedGenerator:
    """Keeps the real planner behind playback until teardown releases it."""

    def __init__(self) -> None:
        self._release = Event()

    def generate(self, request: CandidateRequest) -> CandidateBatch:
        self._release.wait(timeout=5.0)
        return CandidateBatch(
            request_id=request.request_id,
            candidates=(),
            source="delayed",
            prompt=(),
            raw_response="",
            latency_ms=0.0,
            error_type="delayed",
            error_message="generator was intentionally delayed",
        )

    def stop(self) -> None:
        self._release.set()


def _manifest(tempo: Tempo) -> dict[str, object]:
    scenario = default_scenario()
    templates = []
    for template_id in dict.fromkeys(segment.template_id for segment in scenario.segments):
        payload = flow_template_payload(BUILTIN_TEMPLATES.get(template_id))
        templates.append(
            {
                "template_id": payload["template_id"],
                "name": payload["name"],
                "definition": {
                    "ticks_per_beat": payload["ticks_per_beat"],
                    "beats_per_bar": payload["beats_per_bar"],
                    "slots": [
                        {
                            "tick_in_bar": slot["tick_in_bar"],
                            "duration_ticks": slot["duration_ticks"],
                            "target_stress": slot["target_stress"],
                            "boundary_strength": slot["boundary_strength"],
                            "rhyme_group": slot["rhyme_group"],
                        }
                        for slot in payload["slots"]
                    ],
                },
                "provenance": payload["provenance"],
            }
        )
    return build_session_manifest(
        scenario_id=scenario.scenario_id,
        scenario={
            "scenario_id": scenario.scenario_id,
            "tempo_bpm": tempo.bpm,
            "loop": scenario.loop,
            "segments": [
                {
                    "start_bar": segment.start_bar,
                    "bars": segment.bars,
                    "topic": segment.topic,
                    "template_id": segment.template_id,
                    "fallback_lines": list(segment.fallback_lines),
                }
                for segment in scenario.segments
            ],
        },
        seed=7,
        tempo={"bpm": tempo.bpm, "ticks_per_beat": 4, "beats_per_bar": 4},
        templates=templates,
        generator_config={"name": "delayed"},
        model_config={"name": "none"},
        score_weights=asdict(ScoreWeights()),
        minimum_score=0.55,
        timeout_seconds=5.0,
        lookahead_bars=3,
        python_version="test",
        platform="test",
        package_version="test",
        git_revision="test",
        git_dirty=False,
    )


def test_audio_reset_starts_a_new_recorder_and_wav_epoch(tmp_path: Path) -> None:
    tempo = Tempo(60.0, 4, 4)
    audio_format = AudioFormat()
    scenario = default_scenario()
    analyzer = CmuProsodyAnalyzer()
    recorder = RapSessionRecorder(tmp_path / "session", _manifest(tempo))
    publisher = RapEventPublisher("audio-reset")
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(recorder,))
    dispatcher.start()
    manual = ManualAudioSink(audio_format)
    output_path = tmp_path / "session.wav"
    sink = CompositeAudioSink(manual, Float32WavAudioSink(output_path, audio_format))
    coordinator = BarAudioCoordinator(
        DeterministicRapBarRenderer(
            tempo=tempo,
            audio_format=audio_format,
            synthesizer=FakeSpeech(),
            drums=ProceduralBoomBapRenderer(seed=7),
        ),
        publisher=publisher,
    )
    controller: RollingRapController
    playback = RapPlaybackService(
        tempo=tempo,
        sink=sink,
        publisher=publisher,
        on_tick=lambda tick: controller.on_tick(tick),
    )
    controller = RollingRapController(
        tempo=tempo,
        scenario=scenario,
        templates=BUILTIN_TEMPLATES,
        fallback_catalog=PrevalidatedFallbackCatalog.build(scenario, BUILTIN_TEMPLATES, analyzer),
        analyzer=analyzer,
        weights=ScoreWeights(),
        publisher=publisher,
        primary_generator=None,
        candidate_count=12,
        lookahead_bars=3,
        minimum_score=0.55,
        seed=7,
        audio_coordinator=coordinator,
        on_audio_committed=playback.enqueue,
    )
    bar_frames = bar_start_frame(1, tempo, audio_format)

    try:
        controller.start()
        playback.start()
        manual.advance(bar_frames)
        playback.poll()
        controller.request_stop(successor_bar=1)
        playback.request_stop()
        playback.poll()

        playback.reset()
        controller.reset()

        controller.start()
        playback.start()
        manual.advance(bar_frames)
        playback.poll()
        controller.request_stop(successor_bar=1)
        playback.request_stop()
        playback.poll()
    finally:
        playback.close()
        controller.close()
        dispatcher.flush_and_close()
        recorder.close()

    summary = json.loads((tmp_path / "session" / "summary.json").read_text(encoding="utf-8"))
    wav_bytes = output_path.read_bytes()

    assert summary["events"]["epoch"] == 1
    assert summary["audio"]["completed_bars"] == 1
    assert summary["audio"]["completed_frames"] == bar_frames
    assert struct.unpack("<I", wav_bytes[40:44])[0] == bar_frames * audio_format.channels * 4


def test_rolling_audio_runs_without_gap_when_generator_is_late(tmp_path: Path) -> None:
    tempo = Tempo(60.0, 4, 4)
    audio_format = AudioFormat()
    scenario = default_scenario()
    analyzer = CmuProsodyAnalyzer()
    fallback_catalog = PrevalidatedFallbackCatalog.build(scenario, BUILTIN_TEMPLATES, analyzer)
    delayed = DelayedGenerator()
    recorder = RapSessionRecorder(tmp_path / "session", _manifest(tempo))
    publisher = RapEventPublisher("audio-acceptance")
    projector = RapStateProjector()
    events = []
    dispatcher = RapEventDispatcher(publisher.queue, sinks=(recorder, projector, events.append))
    dispatcher.start()
    manual = ManualAudioSink(audio_format)
    output_path = tmp_path / "session.wav"
    sink = CompositeAudioSink(manual, Float32WavAudioSink(output_path, audio_format))
    renderer = DeterministicRapBarRenderer(
        tempo=tempo,
        audio_format=audio_format,
        synthesizer=FakeSpeech(),
        drums=ProceduralBoomBapRenderer(seed=7),
    )
    coordinator = BarAudioCoordinator(renderer, publisher=publisher)
    controller: RollingRapController
    playback = RapPlaybackService(
        tempo=tempo,
        sink=sink,
        publisher=publisher,
        on_tick=lambda tick: controller.on_tick(tick),
    )
    controller = RollingRapController(
        tempo=tempo,
        scenario=scenario,
        templates=BUILTIN_TEMPLATES,
        fallback_catalog=fallback_catalog,
        analyzer=analyzer,
        weights=ScoreWeights(),
        publisher=publisher,
        primary_generator=delayed,
        candidate_count=12,
        lookahead_bars=3,
        minimum_score=0.55,
        seed=7,
        planning_bar_limit=100,
        stop_primary=delayed.stop,
        close_primary=delayed.stop,
        audio_coordinator=coordinator,
        on_audio_committed=playback.enqueue,
    )

    try:
        controller.start()
        playback.start()
        while len(manual.completed) < 100:
            manual.advance(192_000)
            playback.poll()
        playback.request_stop()
        playback.poll()
    finally:
        playback.close()
        controller.close()
        dispatcher.flush_and_close()
        recorder.close()

    assert len(manual.completed) == 100
    assert manual.snapshot().absolute_frame == bar_start_frame(100, tempo, audio_format)
    assert not [event for event in events if event.event_type == RapEventType.AUDIO_UNDERRUN]
    assert any(event.event_type == RapEventType.FALLBACK_ACTIVATED for event in events)
    timing_errors = {
        event.payload["software_error_samples"]
        for event in events
        if event.event_type == RapEventType.SYLLABLE_EMITTED
    }
    assert timing_errors == {0}
    summary = json.loads((tmp_path / "session" / "summary.json").read_text(encoding="utf-8"))
    assert summary["bars"]["frozen"] == 100
    assert summary["audio"]["completed_bars"] == 100
    assert summary["audio"]["underruns"] == 0
    assert output_path.is_file()
