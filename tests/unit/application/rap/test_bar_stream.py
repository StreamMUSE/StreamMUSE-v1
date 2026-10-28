from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace
import pytest
import threading
import time

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.bar_stream import (
    SOURCE_ESPEAK_LOCAL,
    SOURCE_ESPEAK_STREAM,
    SOURCE_REMOTE_LATEST,
    SOURCE_REMOTE_OLDER,
    BarAudioFetch,
    BarStreamController,
)
from streammuse.domain.rap import ProsodyAnalysis, Syllable, materialize_flow, normalize_text
from streammuse.application.rap.audio_rendering import bar_frame_count
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapWarpDiagnostics
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
from streammuse.domain.rap.events import RapEventType
from streammuse.domain.timing import Tempo
from streammuse.domain.rap.timeline import RapTimeline
from streammuse.infrastructure.rap.fallback import PrevalidatedFallbackCatalog
from streammuse.infrastructure.rap.scenarios import default_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

_FORMAT = AudioFormat()


class _Inline(Executor):
    """Runs work synchronously so tests are deterministic."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


class _Pausable(_Inline):
    def __init__(self):
        self.paused = False
        self.pending = []

    def submit(self, fn, /, *args, **kwargs):
        if not self.paused:
            return super().submit(fn, *args, **kwargs)
        future = Future()
        self.pending.append((future, fn, args, kwargs))
        return future


class _Analyzer:
    def analyze(self, text: str) -> ProsodyAnalysis:
        words = normalize_text(text).split()
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalize_text(text),
            syllables=tuple(Syllable(word, 0, 1, 1, ("AH1",), "fake") for word in words),
            end_rhyme_tail=("AH1",),
            oov_words=(),
            heuristic_words=(),
            punctuation_boundary_after=(),
        )


def _audio(bar: int, source: str) -> PreparedRapBar:
    return PreparedRapBar(
        bar=bar, text=source, source=source, fallback_reason=None, scheduled=(),
        audio=PcmAudio(_FORMAT, 0, b""), diagnostics=(), warnings=(), render_latency_ms=0.0,
    )


class _Espeak:
    def render(self, plan):
        return _audio(plan.bar, f"espeak:{plan.text}")


@dataclass
class _Line:
    text: str
    analysis: ProsodyAnalysis
    source: str = "local_catalog"


class _Catalog:
    def line_for(self, request):
        text = " ".join(["local"] * len(request.flow_template.slots))
        return _Line(text, _Analyzer().analyze(text))


class _Stream:
    def __init__(self) -> None:
        self.sent = []
        self.handler = None

    def open(self, session_id, start, on_message):
        self.session_id, self.start, self.handler = session_id, start, on_message

    def send(self, message):
        self.sent.append(dict(message))

    def close(self):
        pass


class _Source:
    def __init__(self) -> None:
        self.ready: dict[str, BarAudioFetch] = {}
        self.calls = []

    def fetch(self, session_id, bar, lyric_id, *, wait_ms, accept_older):
        self.calls.append((bar, lyric_id, accept_older))
        return self.ready.get(lyric_id, BarAudioFetch("failed"))

    def close(self):
        pass


class _Preparer:
    def prepare(self, record, manifest, source_wav):
        return _audio(record.bar, f"remote:{record.revision}")


class _PolicyPreparer:
    def __init__(self, policy="gentle_sparse_r3", calls=None, during_prepare=None):
        self.policy = policy
        self.calls = [] if calls is None else calls
        self.during_prepare = during_prepare

    def with_alignment_policy(self, policy):
        if policy not in ("gentle_sparse_r3", "all_onsets_r3", "all_onsets_stress_r3"):
            raise ValueError("invalid alignment policy")
        return type(self)(policy, self.calls, self.during_prepare)

    def prepare(self, record, manifest, source_wav, **kwargs):
        self.calls.append((record.bar, self.policy, source_wav))
        if self.during_prepare is not None:
            self.during_prepare(record, self.policy)
        result = replace(_audio(record.bar, self.policy), warp_diagnostics=RapWarpDiagnostics(
            (0.1,), (0.1,), (0.1,), (1.0,), self.policy, 0.0))
        timing = kwargs.get("timing")
        if timing is not None:
            result = replace(result, timing=timing, audio=PcmAudio(
                _FORMAT, timing.frame_count, bytes(timing.frame_count * _FORMAT.channels * 4)))
        return result


def _record(bar: int, revision: int = 0, word: str = "flow") -> BarLyricRecord:
    template = BUILTIN_TEMPLATES.get(default_scenario().segment_for_bar(bar).template_id)
    text = " ".join([word] * len(template.slots))
    return BarLyricRecord(
        session_id="s-e0", bar=bar, revision=revision, topic="space", flow_template=template, text=text,
        scheduled=align_exact(_Analyzer().analyze(text), materialize_flow(template, bar)), score=0.5,
    )


def _fetch(record: BarLyricRecord, status: str = "ready") -> BarAudioFetch:
    return BarAudioFetch(
        status=status,
        lyric_id=record.lyric_id,
        manifest=BarAudioManifest(
            lyric_id=record.lyric_id, bar=record.bar, tempo_bpm=90.0, frame_count=64_000, source_sha256="b" * 64,
            source_onsets=(0.1,), onset_confidence=(0.9,), stage_timings_ms={"moss": 500.0},
        ),
        source_wav=b"wav",
        request_ms=12.0,
    )


class _Publisher:
    def __init__(self):
        self.events = []

    def emit(self, event_type, **kwargs):
        self.events.append((event_type, kwargs))


def _controller(stream, source, enqueued, publisher=None, clock=time.monotonic, **overrides):
    options = dict(
        tempo=Tempo(90.0, 4, 4),
        scenario=default_scenario(),
        templates=BUILTIN_TEMPLATES,
        fallback_catalog=_Catalog(),
        analyzer=_Analyzer(),
        fallback_renderer=_Espeak(),
        lyric_stream=stream,
        audio_source=source,
        preparer=_Preparer(),
        publisher=publisher,
        enqueue=enqueued.append,
        session_id="s",
        seed=1,
        planning_bar_limit=None,
        startup_timeout_seconds=0.3,
        fetch_executor=_Inline(),
        render_executor=_Inline(),
        monotonic=clock,
    )
    options.update(overrides)
    return BarStreamController(**options)


def _push(stream, record):
    stream.handler({"type": "lyric", "record": record.to_payload()})


def _start_with(stream, source, enqueued, *records, publisher=None, **options):
    controller = _controller(stream, source, enqueued, publisher, **options)
    thread = threading.Thread(target=controller.start)
    thread.start()
    deadline = time.monotonic() + 2
    while stream.handler is None and time.monotonic() < deadline:
        time.sleep(0.005)
    for record in records:
        _push(stream, record)
    thread.join(5)
    return controller


def test_alignment_reuses_cached_sources_and_leaves_committed_bar_unchanged():
    stream, source, enqueued = _Stream(), _Source(), []
    preparer = _PolicyPreparer()
    first, second = _record(0), _record(1)
    source.ready = {record.lyric_id: _fetch(record) for record in (first, second)}
    controller = _start_with(stream, source, enqueued, first, second, preparer=preparer)
    try:
        committed = enqueued[0]
        calls, messages, session = list(source.calls), list(stream.sent), controller.session_id
        result = controller.update_alignment_policy("all_onsets_r3")
        assert result == {"effective_from_bar": 1, "alignment_policy": "all_onsets_r3",
                          "revision": 1, "coordinator_epoch": 0}
        assert source.calls == calls and stream.sent == messages and controller.session_id == session
        assert enqueued[0] is committed and committed.source == "gentle_sparse_r3"
        assert preparer.calls[-1] == (1, "all_onsets_r3", b"wav")
        assert not controller._slots[0].sources
        controller.on_tick(15)
        assert enqueued[-1].source == "all_onsets_r3"
        assert not controller._slots[1].sources
        assert controller.update_alignment_policy("all_onsets_stress_r3")["effective_from_bar"] == 2
    finally:
        controller.close()


def test_alignment_change_during_render_rejects_stale_result_without_refetch():
    stream, source, enqueued = _Stream(), _Source(), []
    preparer = _PolicyPreparer()
    first = _record(0)
    source.ready[first.lyric_id] = _fetch(first)
    controller = _start_with(stream, source, enqueued, first, preparer=preparer)
    second = _record(1)
    source.ready[second.lyric_id] = _fetch(second)

    def switch(record, policy):
        if record.bar == 1 and policy == "gentle_sparse_r3":
            controller.update_alignment_policy("all_onsets_r3")

    preparer.during_prepare = switch
    try:
        _push(stream, second)
        assert source.calls.count((1, second.lyric_id, True)) == 1
        assert controller._slots[1].remote[second.lyric_id].source == "all_onsets_r3"
        controller.on_tick(15)
        assert enqueued[-1].source == "all_onsets_r3"
    finally:
        controller.close()


def test_alignment_change_while_fetching_preserves_arriving_source_and_uses_current_policy():
    stream, source, enqueued = _Stream(), _Source(), []
    controller = _controller(stream, source, enqueued, preparer=_PolicyPreparer())
    first = _record(0)
    source.ready[first.lyric_id] = _fetch(first)
    fetch = source.fetch

    def switch_during_fetch(*args, **kwargs):
        controller.update_alignment_policy("all_onsets_r3")
        return fetch(*args, **kwargs)

    source.fetch = switch_during_fetch
    try:
        controller._on_message(0, {"type": "lyric", "record": first.to_payload()})
        assert len(source.calls) == 1
        assert controller._slots[0].remote[first.lyric_id].source == "all_onsets_r3"
        assert first.lyric_id in controller._slots[0].sources
        controller.update_alignment_policy("all_onsets_stress_r3")
        assert len(source.calls) == 1
        assert controller._slots[0].remote[first.lyric_id].source == "all_onsets_stress_r3"
    finally:
        controller.close()


def test_alignment_render_racing_commit_does_not_overwrite_or_retain_committed_source():
    stream, source, enqueued = _Stream(), _Source(), []
    preparer = _PolicyPreparer()
    first, second, third = _record(0), _record(1), _record(2)
    source.ready = {record.lyric_id: _fetch(record) for record in (first, second, third)}
    controller = _start_with(stream, source, enqueued, first, second, third, preparer=preparer)
    commits = []

    def commit_during_warp(record, policy):
        if record.bar == 1 and policy == "all_onsets_r3":
            controller.on_tick(15)
            commits.append(enqueued[-1])

    preparer.during_prepare = commit_during_warp
    try:
        controller.update_alignment_policy("all_onsets_r3")
        assert commits and controller._slots[1].committed is commits[0]
        assert not controller._slots[1].sources
        ack = controller.update_alignment_policy("all_onsets_stress_r3")
        assert ack["effective_from_bar"] == 2
        assert enqueued[-1] is commits[0]
        assert controller._slots[2].remote[third.lyric_id].source == "all_onsets_stress_r3"
    finally:
        controller.close()


def test_alignment_sources_are_bounded_across_lyric_revisions_and_cleared_on_reset():
    stream, source = _Stream(), _Source()
    controller = _controller(stream, source, [], preparer=_PolicyPreparer())
    try:
        for revision in range(12):
            record = _record(0, revision, word=f"word{revision}")
            source.ready[record.lyric_id] = _fetch(record)
            controller._on_message(0, {"type": "lyric", "record": record.to_payload()})
        assert 0 < len(controller._slots[0].sources) <= 2
        controller.update_alignment_policy("all_onsets_r3")
        assert controller._slots[0].remote[record.lyric_id].source == "all_onsets_r3"
        calls = list(source.calls)
        controller.reset()
        assert not controller._slots
        assert controller.update_alignment_policy("all_onsets_stress_r3")["effective_from_bar"] == 0
        assert source.calls == calls
    finally:
        controller.close()


def test_alignment_running_worker_cannot_replace_newer_policy_from_another_thread():
    entered, release = threading.Event(), threading.Event()
    stream, source, enqueued = _Stream(), _Source(), []
    preparer = _PolicyPreparer()
    first, second = _record(0), _record(1)
    source.ready = {record.lyric_id: _fetch(record) for record in (first, second)}
    fetch_executor = ThreadPoolExecutor(max_workers=1)
    controller = _start_with(stream, source, enqueued, first, preparer=preparer, fetch_executor=fetch_executor)

    def block_old_policy(record, policy):
        if record.bar == 1 and policy == "gentle_sparse_r3":
            entered.set()
            assert release.wait(3)

    preparer.during_prepare = block_old_policy
    try:
        _push(stream, second)
        assert entered.wait(3)
        with ThreadPoolExecutor(max_workers=1) as control_worker:
            result = control_worker.submit(controller.update_alignment_policy, "all_onsets_r3").result(timeout=2)
        assert result["effective_from_bar"] == 1
        assert controller._slots[1].remote[second.lyric_id].source == "all_onsets_r3"
        release.set()
        fetch_executor.submit(lambda: None).result(timeout=3)
        assert controller._slots[1].remote[second.lyric_id].source == "all_onsets_r3"
        assert len(source.calls) == 2
    finally:
        release.set()
        controller.close()


def test_alignment_fetch_survives_predecessor_commit_and_remains_reusable():
    stream, source, enqueued = _Stream(), _Source(), []
    records = [_record(bar) for bar in range(3)]
    source.ready = {record.lyric_id: _fetch(record) for record in records}
    controller = _start_with(stream, source, enqueued, *records[:2], preparer=_PolicyPreparer())
    fetch = source.fetch

    def commit_during_fetch(*args, **kwargs):
        assert controller.update_alignment_policy("all_onsets_r3")["effective_from_bar"] == 1
        controller.on_tick(15)
        return fetch(*args, **kwargs)

    source.fetch = commit_during_fetch
    try:
        _push(stream, records[2])
        assert records[2].lyric_id in controller._slots[2].sources
        assert controller._slots[2].remote[records[2].lyric_id].source == "all_onsets_r3"
        calls = list(source.calls)
        assert controller.update_alignment_policy("all_onsets_stress_r3")["effective_from_bar"] == 2
        assert source.calls == calls
        assert controller._slots[2].remote[records[2].lyric_id].source == "all_onsets_stress_r3"
    finally:
        controller.close()


def test_alignment_coalesces_queued_policies_and_fences_reset_workers():
    stream, source, enqueued = _Stream(), _Source(), []
    records = [_record(bar) for bar in range(2)]
    source.ready = {record.lyric_id: _fetch(record) for record in records}
    executor, preparer = _Pausable(), _PolicyPreparer()
    controller = _start_with(stream, source, enqueued, *records, preparer=preparer, render_executor=executor)
    executor.paused = True
    try:
        controller.update_alignment_policy("all_onsets_r3")
        ack = controller.update_alignment_policy("all_onsets_stress_r3")
        assert ack["revision"] == 2
        with pytest.raises(ValueError):
            controller.update_alignment_policy("invalid")
        for future, fn, args, kwargs in executor.pending:
            future.set_result(fn(*args, **kwargs))
        executor.pending.clear()
        assert [policy for bar, policy, _ in preparer.calls if bar == 1] == ["gentle_sparse_r3", "all_onsets_stress_r3"]
        assert controller._slots[1].remote[records[1].lyric_id].source == "all_onsets_stress_r3"
        controller.update_alignment_policy("all_onsets_r3")
        controller.reset()
        before = list(preparer.calls)
        for future, fn, args, kwargs in executor.pending:
            future.set_result(fn(*args, **kwargs))
        assert preparer.calls == before and not controller._slots
    finally:
        controller.close()
    with pytest.raises(RuntimeError):
        controller.update_alignment_policy("all_onsets_r3")


class _TimedEspeak:
    def render(self, plan, *, timing=None):
        result = _audio(plan.bar, "timed-espeak")
        return replace(result, audio=PcmAudio(_FORMAT, timing.frame_count,
            bytes(timing.frame_count * _FORMAT.channels * 4)), timing=timing)


class _TempoStream(_Stream):
    def __init__(self):
        super().__init__()
        self.revision = 0
        self.capabilities = frozenset({"tempo_updates_v1"})

    def request_tempo(self, payload, **kwargs):
        self.sent.append(dict(payload))
        self.revision += 1
        return {"type": "tempo_updated", "edit_seq": payload["edit_seq"],
                "requested_from_bar": payload["from_bar"], "effective_from_bar": payload["from_bar"],
                "bpm": payload["bpm"], "revision": self.revision}


def test_tempo_update_invalidates_prepared_future_and_retains_past_snapshot():
    stream, source, enqueued, installed = _TempoStream(), _Source(), [], []
    clock = RapTimeline(Tempo(90, 4, 4))
    controller = _start_with(stream, source, enqueued, timeline=clock,
                             fallback_renderer=_TimedEspeak(), on_timeline_changed=installed.append,
                             tempo_ack_timeout_seconds=1.0)
    try:
        original = enqueued[0].timing
        ack = controller.update_tempo(120, from_bar=0)
        assert ack["requested_from_bar"] == 0
        assert ack["effective_from_bar"] == 1
        assert installed[-1] == controller.timeline
        controller.on_tick(15)
        assert enqueued[0].timing == original
        assert enqueued[1].timing.bpm == 120
        assert enqueued[1].audio.frame_count == 96000
        assert enqueued[1].timing.start_frame == original.end_frame
    finally:
        controller.close()


def test_legacy_tempo_rejection_does_not_change_local_timeline():
    controller = _controller(_Stream(), _Source(), [])
    controller._started = True
    before = controller.timeline
    try:
        with pytest.raises(ValueError, match="capability"):
            controller.update_tempo(120)
        assert controller.timeline == before
    finally:
        controller.close()


def test_stopped_initial_tempo_is_local_and_sent_in_start_contract():
    controller = _controller(_Stream(), _Source(), [])
    try:
        result = controller.update_tempo(120)
        assert result['effective_from_bar'] == 0
        assert controller.timeline.snapshot(0, 48000).bpm == 120
        assert controller._start_message(start_bar=0)['tempo_bpm'] == 120
        assert controller.reset() == 1
        assert controller.timeline.snapshot(0, 48000).bpm == 120
    finally:
        controller.close()


def test_stop_during_startup_returns_promptly_without_enqueue():
    stream, source, enqueued = _Stream(), _Source(), []
    controller = _controller(stream, source, enqueued, startup_timeout_seconds=30)
    thread = threading.Thread(target=controller.start)
    thread.start()
    try:
        deadline = time.monotonic() + 1
        while stream.handler is None and time.monotonic() < deadline:
            time.sleep(.005)
        controller.request_stop(successor_bar=None)
        thread.join(.5)
        assert not thread.is_alive()
        assert enqueued == []
    finally:
        controller.reset()
        thread.join(1)
        controller.close()


def test_initial_epoch_is_preserved_and_must_be_nonnegative():
    controller = _controller(_Stream(), _Source(), [], initial_epoch=7)
    try:
        assert controller.coordinator_epoch == 7
        assert controller.session_id == "s-e7"
        assert controller.reset() == 8
    finally:
        controller.close()
    with pytest.raises(ValueError):
        _controller(_Stream(), _Source(), [], initial_epoch=-1)


def test_late_old_timing_result_cannot_replace_reprepared_bar():
    class PausedRenderer(_TimedEspeak):
        def __init__(self):
            self.started, self.release = threading.Event(), threading.Event()
            self.finished = threading.Event()

        def render(self, plan, *, timing=None):
            if plan.bar == 1 and timing.revision == 0:
                self.started.set()
                assert self.release.wait(3)
            result = super().render(plan, timing=timing)
            if plan.bar == 1 and timing.revision == 0:
                self.finished.set()
            return result

    renderer, executor = PausedRenderer(), ThreadPoolExecutor(max_workers=3)
    stream, source, enqueued = _TempoStream(), _Source(), []
    controller = _start_with(stream, source, enqueued, timeline=RapTimeline(Tempo(90,4,4)),
                             fallback_renderer=renderer, render_executor=executor,
                             tempo_ack_timeout_seconds=1.0)
    try:
        assert renderer.started.wait(1)
        controller.update_tempo(120, from_bar=1)
        deadline = time.monotonic() + 1
        while controller._slot(1).local is None and time.monotonic() < deadline:
            time.sleep(.005)
        assert controller._slot(1).local.timing.bpm == 120
        renderer.release.set()
        assert renderer.finished.wait(1)
        controller.on_tick(15)
        assert enqueued[1].timing.bpm == 120
        assert enqueued[1].audio.frame_count == 96000
    finally:
        renderer.release.set()
        controller.close()


def test_wrong_duration_preparation_falls_back_to_exact_silent_bar():
    class WrongDuration(_TimedEspeak):
        def render(self, plan, *, timing=None):
            return replace(super().render(plan, timing=timing), audio=PcmAudio(_FORMAT, 0, b''))

    controller = _controller(_Stream(), _Source(), [], timeline=RapTimeline(Tempo(90,4,4)),
                              fallback_renderer=WrongDuration())
    try:
        with controller._lock:
            controller._ensure_window_locked()
            result = controller._commit_locked(0, tick=None)
        assert result.source == 'silence'
        assert result.audio.frame_count == result.timing.frame_count == 128000
    finally:
        controller.close()


def test_wrong_destination_bar_is_never_committed():
    controller = _controller(_Stream(), _Source(), [])
    try:
        controller._slot(0).local = _audio(3, 'wrong destination')
        result = controller._commit_locked(0, tick=None)
        assert result.bar == 0
        assert result.source == 'silence'
    finally:
        controller.close()


def test_timing_fallback_uses_drums_only_hook_without_waiting_for_voice():
    class DrumsOnly(_TimedEspeak):
        def render(self, plan, *, timing=None):
            raise RuntimeError('voice unavailable')

        def render_drums_only(self, *, bar, template, timing):
            return replace(_audio(bar, 'drums_only'), timing=timing,
                           audio=PcmAudio(_FORMAT, timing.frame_count,
                                          bytes(timing.frame_count * _FORMAT.channels * 4)))

    controller = _controller(_Stream(), _Source(), [], timeline=RapTimeline(Tempo(90,4,4)),
                              fallback_renderer=DrumsOnly())
    try:
        with controller._lock:
            controller._ensure_window_locked()
            result = controller._commit_locked(0, tick=None)
        assert result.source == 'drums_only'
        assert result.audio.frame_count == result.timing.frame_count == 128000
        assert result.fallback_reason == 'drums_only'
    finally:
        controller.close()


def test_live_mix_remixes_uncommitted_stems_without_rendering_again():
    import numpy as np
    from streammuse.domain.rap.audio import RapMixConfig

    class StemRenderer(_TimedEspeak):
        def __init__(self):
            self.calls = 0
            self.config = RapMixConfig()

        def set_mix_config(self, config):
            self.config = config

        def render(self, plan, *, timing=None):
            self.calls += 1
            result = super().render(plan, timing=timing)
            samples = np.full(timing.frame_count * _FORMAT.channels, .1, dtype=np.float32)
            stem = PcmAudio(_FORMAT, timing.frame_count, samples.tobytes())
            return replace(result, vocal_stem=stem, drum_stem=stem, mix_config=self.config)

    renderer = StemRenderer()
    controller = _controller(_Stream(), _Source(), [], timeline=RapTimeline(Tempo(90,4,4)),
                             fallback_renderer=renderer)
    try:
        with controller._lock:
            controller._ensure_window_locked()
            committed = controller._commit_locked(0, tick=None)
        calls = renderer.calls
        config = RapMixConfig(vocal_gain=.2, drum_gain=.4, revision=1)
        ack = controller.update_mix(config, from_bar=0)
        assert ack['effective_from_bar'] == 1
        assert renderer.calls == calls
        assert controller._slot(0).committed is committed
        assert controller._slot(1).local.mix_config == config
        assert np.frombuffer(controller._slot(1).local.audio.data, np.float32)[0] == pytest.approx(.06)
    finally:
        controller.close()


def test_tempo_ack_wait_does_not_block_tick_dispatch():
    class SlowStream(_TempoStream):
        def __init__(self):
            super().__init__()
            self.entered, self.release = threading.Event(), threading.Event()

        def request_tempo(self, payload, **kwargs):
            self.entered.set()
            assert self.release.wait(2)
            return super().request_tempo(payload, **kwargs)

    stream, enqueued = SlowStream(), []
    controller = _start_with(stream, _Source(), enqueued, timeline=RapTimeline(Tempo(90,4,4)),
                             fallback_renderer=_TimedEspeak(), tempo_ack_timeout_seconds=5.0)
    results = []
    thread = threading.Thread(target=lambda: results.append(controller.update_tempo(120, from_bar=3)))
    thread.start()
    try:
        assert stream.entered.wait(1)
        tick_done = threading.Event()
        tick_thread = threading.Thread(target=lambda: (controller.on_tick(15), tick_done.set()))
        tick_thread.start()
        assert tick_done.wait(.2)
        assert enqueued[-1].bar == 1
        assert enqueued[-1].timing.bpm == 90
    finally:
        stream.release.set()
        thread.join(2)
        tick_thread.join(2)
        controller.close()
    assert results[0]['effective_from_bar'] == 3


@pytest.mark.parametrize("change_policy_again", [False, True])
def test_alignment_ack_is_local_while_tempo_ack_and_old_timing_warp_are_pending(change_policy_again):
    ack_entered, ack_release = threading.Event(), threading.Event()
    warp_entered, warp_release = threading.Event(), threading.Event()

    class SlowStream(_TempoStream):
        def request_tempo(self, payload, **kwargs):
            ack_entered.set()
            assert ack_release.wait(5)
            return super().request_tempo(payload, **kwargs)

    class BlockingPreparer(_PolicyPreparer):
        def prepare(self, record, manifest, source_wav, **kwargs):
            if record.bar == 1 and self.policy == "all_onsets_r3" and manifest.tempo_bpm == 90:
                warp_entered.set()
                assert warp_release.wait(5)
            return super().prepare(record, manifest, source_wav, **kwargs)

    class RenderPool(ThreadPoolExecutor):
        submitted = []

        def submit(self, fn, /, *args, **kwargs):
            future = super().submit(fn, *args, **kwargs)
            self.submitted.append(future)
            return future

    stream, source, enqueued, publisher = SlowStream(), _Source(), [], _Publisher()
    first, second = _record(0), _record(1)
    source.ready = {record.lyric_id: _fetch(record) for record in (first, second)}
    renders = RenderPool(max_workers=3)
    controls = ThreadPoolExecutor(max_workers=2)
    controller = _start_with(stream, source, enqueued, first, second, preparer=BlockingPreparer(),
                             render_executor=renders, fallback_renderer=_TimedEspeak(), publisher=publisher,
                             timeline=RapTimeline(Tempo(90, 4, 4)), tempo_ack_timeout_seconds=1.0)
    try:
        committed = enqueued[0]
        pending_tempo = controls.submit(controller.update_tempo, 120, 1)
        assert ack_entered.wait(2)
        calls, messages = list(source.calls), list(stream.sent)
        ack = controls.submit(controller.update_alignment_policy, "all_onsets_r3").result(timeout=0.5)
        assert ack == {"effective_from_bar": 1, "alignment_policy": "all_onsets_r3",
                       "revision": 1, "coordinator_epoch": 0}
        assert warp_entered.wait(2)
        assert not pending_tempo.done()
        assert source.calls == calls and stream.sent == messages
        controls.submit(controller.on_tick, 1).result(timeout=0.5)
        policy = "all_onsets_stress_r3" if change_policy_again else "all_onsets_r3"
        if change_policy_again:
            controls.submit(controller.update_alignment_policy, policy).result(timeout=0.5)
        source.ready[second.lyric_id] = replace(_fetch(second), manifest=replace(_fetch(second).manifest, tempo_bpm=120))
        ack_release.set()
        assert pending_tempo.result(timeout=2)["effective_from_bar"] == 1
        assert controller._slots[1].remote[second.lyric_id].timing.bpm == 120
        warp_release.set()
        for future in tuple(renders.submitted):
            future.result(timeout=2)
        remote = controller._slots[1].remote[second.lyric_id]
        assert remote.source == policy and remote.timing.bpm == 120
        assert controller._slots[1].sources[second.lyric_id].fetch.manifest.tempo_bpm == 120
        controller.on_tick(15)
        assert enqueued[0] is committed
        assert enqueued[-1].warp_diagnostics.policy == policy
        event = [payload for kind, payload in publisher.events if kind == RapEventType.BAR_AUDIO_COMMITTED][-1]
        assert event["payload"]["timing_evidence"]["warp"]["policy"] == policy
    finally:
        ack_release.set()
        warp_release.set()
        controls.shutdown(wait=True)
        controller.close()


def test_tempo_boundary_reserves_full_ack_budget_before_commit():
    now = [100.0]
    stream = _TempoStream()
    controller = _controller(stream, _Source(), [], clock=lambda: now[0])
    controller._started = True
    controller._clock_origin = 100.0
    controller._playhead = 3
    now[0] = 110.0
    try:
        ack = controller.update_tempo(50)
        # The complete ten-second retry budget ends at 120s, before bar 8's guard.
        assert ack['ack_timeout_seconds'] == 10.0
        assert ack['effective_from_bar'] == 8
        assert controller.timeline.snapshot(7, 48_000).bpm == 90
        assert controller.timeline.snapshot(8, 48_000).bpm == 50
    finally:
        controller.close()


def test_uncertain_tempo_ack_keeps_local_audio_running_and_blocks_remote():
    class LostAck(_TempoStream):
        def request_tempo(self, payload, **kwargs):
            raise TimeoutError('lost response after server may have applied it')

    stream, source, enqueued, publisher = LostAck(), _Source(), [], _Publisher()
    controller = _start_with(stream, source, enqueued, publisher=publisher,
                             timeline=RapTimeline(Tempo(90, 4, 4)),
                             fallback_renderer=_TimedEspeak())
    try:
        before = controller.timeline
        with pytest.raises(TimeoutError):
            controller.update_tempo(50)
        assert controller.tempo_control_state['requires_reset'] is True
        controller.on_tick(15)
        controller.on_tick(31)
        assert [bar.bar for bar in enqueued] == [0, 1, 2]
        assert all(bar.timing.bpm == 90 and bar.audio.frame_count == 128_000 for bar in enqueued)
        assert enqueued[-1].source == 'timed-espeak'
        assert controller.timeline == before
        assert enqueued[-1].fallback_reason == 'tempo_ack_uncertain'
        _push(stream, _record(3))
        assert source.calls == []
        with pytest.raises(RuntimeError, match='synchronized'):
            controller.update_tempo(60)
        errors = [event['payload'] for kind, event in publisher.events if kind == RapEventType.PRESENTATION_ERROR]
        assert errors[-1]['recovery'] == 'local_fallback_until_reset'
        controller.request_stop(successor_bar=None)
        controller.on_tick(47)
        assert len(enqueued) == 3
    finally:
        controller.close()


def test_remote_preparation_in_flight_is_quarantined_after_lost_tempo_ack():
    class LostAck(_TempoStream):
        def request_tempo(self, payload, **kwargs):
            raise TimeoutError('lost acknowledgement')

    entered, release = threading.Event(), threading.Event()

    class DelayedPreparer:
        def prepare(self, record, manifest, wav, *, timing):
            entered.set()
            assert release.wait(2)
            return replace(_audio(record.bar, 'remote'), timing=timing,
                           audio=PcmAudio(_FORMAT, timing.frame_count,
                                          bytes(timing.frame_count * _FORMAT.channels * 4)))

    stream, source, enqueued, publisher = LostAck(), _Source(), [], _Publisher()
    controller = _start_with(stream, source, enqueued, publisher=publisher,
                             timeline=RapTimeline(Tempo(90, 4, 4)),
                             fallback_renderer=_TimedEspeak(), preparer=DelayedPreparer())
    record = _record(1)
    source.ready[record.lyric_id] = _fetch(record)
    # Fetch outside _on_message's inline test executor so the preparation races the edit.
    with controller._lock:
        slot = controller._slot(1)
        slot.latest = record
        slot.records[record.lyric_id] = record
    worker = threading.Thread(target=controller._fetch_remote,
                              args=(0, record, time.monotonic() + 5, controller.timeline.snapshot(1, 48_000)))
    worker.start()
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError):
            controller.update_tempo(50)
        release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert not controller._slot(1).remote
        controller.on_tick(15)
        assert enqueued[-1].source == 'timed-espeak'
        assert not any(kind == RapEventType.CHUNK_REMOTE_COMPLETED for kind, _ in publisher.events)
    finally:
        release.set()
        worker.join(2)
        controller.close()


@pytest.mark.parametrize('created', [False, True])
def test_reconnect_recreated_session_requires_recovery_without_losing_local_timing(created):
    stream, enqueued = _TempoStream(), []
    controller = _start_with(stream, _Source(), enqueued, timeline=RapTimeline(Tempo(90, 4, 4)),
                             fallback_renderer=_TimedEspeak())
    try:
        stream.handler({'type': 'started', 'created': True})
        controller.update_tempo(120, from_bar=3)
        before = controller.timeline
        stream.handler({'type': 'started', 'created': created})
        assert controller.tempo_control_state['requires_reset'] is created
        assert controller.timeline == before
        controller.on_tick(15)
        assert enqueued[-1].bar == 1 and enqueued[-1].timing.bpm == 90
        if created:
            with pytest.raises(RuntimeError, match='synchronized'):
                controller.update_tempo(60)
    finally:
        controller.close()


@pytest.mark.parametrize('in_flight', [False, True])
def test_first_tempo_edit_preserves_unchanged_legacy_vocals(in_flight):
    entered, release = threading.Event(), threading.Event()

    class LegacyTimedRenderer:
        def render(self, plan, *, timing=None):
            if in_flight and plan.bar == 1 and timing is None:
                entered.set()
                assert release.wait(2)
            frames = bar_frame_count(plan.bar, timing or Tempo(90, 4, 4), _FORMAT)
            return replace(_audio(plan.bar, 'legacy-voice'), timing=timing,
                           audio=PcmAudio(_FORMAT, frames, bytes(frames * _FORMAT.channels * 4)))

    stream, enqueued = _TempoStream(), []
    executor = ThreadPoolExecutor(max_workers=3)
    controller = _start_with(stream, _Source(), enqueued, fallback_renderer=LegacyTimedRenderer(),
                             render_executor=executor, tempo_ack_timeout_seconds=5.0)
    try:
        if in_flight:
            assert entered.wait(1)
        controller.update_tempo(120, from_bar=3)
        release.set()
        deadline = time.monotonic() + 1
        while controller._slot(1).local is None and time.monotonic() < deadline:
            time.sleep(.005)
        controller.on_tick(15)
        assert enqueued[-1].source == 'legacy-voice'
        assert enqueued[-1].timing == RapTimeline(Tempo(90, 4, 4)).snapshot(1, 48_000)
        assert enqueued[-1].audio.frame_count == 128_000
        controller.on_tick(31)
        controller.on_tick(47)
        assert enqueued[-1].timing.bpm == 120
        assert enqueued[-1].audio.frame_count == 96_000
    finally:
        release.set()
        controller.close()


def test_late_mix_job_is_remixed_at_publication_to_latest_config():
    import numpy as np
    from streammuse.domain.rap.audio import RapMixConfig

    controller = _controller(_Stream(), _Source(), [])
    entered, release = threading.Event(), threading.Event()
    samples = np.full(4, .1, dtype=np.float32)
    stem = PcmAudio(_FORMAT, 2, samples.tobytes())

    def delayed_plan(epoch, plan, timing=None):
        prepared = replace(_audio(plan.bar, 'old mix'), audio=stem, vocal_stem=stem,
                           drum_stem=stem, mix_config=RapMixConfig())
        entered.set()
        assert release.wait(2)
        return prepared

    controller._render_plan = delayed_plan
    controller._slot(1)
    thread = threading.Thread(target=controller._render_local_fallback, args=(0, 1))
    thread.start()
    try:
        assert entered.wait(1)
        config = RapMixConfig(vocal_gain=.2, drum_gain=.4, revision=1)
        controller.update_mix(config, from_bar=1)
        release.set()
        thread.join(1)
        assert controller._slot(1).local.mix_config == config
        assert np.frombuffer(controller._slot(1).local.audio.data, np.float32)[0] == pytest.approx(.06)
    finally:
        release.set()
        thread.join(1)
        controller.close()


def test_obsolete_fetch_does_not_recreate_slots_or_emit_current_epoch_events():
    publisher, source = _Publisher(), _Source()
    controller = _controller(_Stream(), source, [], publisher)
    record = _record(2)
    try:
        controller.reset()
        controller._fetch_remote(0, record, None)
        assert controller._slots == {}
        assert source.calls == []
        assert publisher.events == []
    finally:
        controller.close()


@pytest.mark.parametrize('startup', [False, True])
def test_stop_during_commit_publication_prevents_late_enqueue(startup):
    publisher, enqueued = _Publisher(), []
    controller = _controller(_Stream(), _Source(), enqueued, publisher, startup_timeout_seconds=.01)
    original_emit = publisher.emit

    def emit(event_type, **kwargs):
        original_emit(event_type, **kwargs)
        if event_type == RapEventType.BAR_AUDIO_COMMITTED:
            controller.request_stop(successor_bar=None)

    publisher.emit = emit
    try:
        if startup:
            controller.start()
        else:
            controller._started = True
            controller._playhead = 0
            controller._slot(1).local = _audio(1, 'local')
            controller.on_tick(15)
        assert enqueued == []
    finally:
        controller.close()


def test_old_fetch_failure_after_reset_does_not_publish_as_new_epoch():
    class FailingSource(_Source):
        def __init__(self):
            super().__init__()
            self.entered, self.release = threading.Event(), threading.Event()

        def fetch(self, *args, **kwargs):
            self.entered.set()
            assert self.release.wait(2)
            raise RuntimeError('old network failure')

    publisher, source = _Publisher(), FailingSource()
    controller = _controller(_Stream(), source, [], publisher)
    record = _record(2)
    controller._slot(2).latest = record
    thread = threading.Thread(target=controller._fetch_remote, args=(0, record, None))
    thread.start()
    try:
        assert source.entered.wait(1)
        controller.reset()
        count = len(publisher.events)
        source.release.set()
        thread.join(1)
        assert len(publisher.events) == count
        assert controller._slots == {}
    finally:
        source.release.set()
        thread.join(1)
        controller.close()


def test_startup_commits_bar_zero_with_remote_audio_of_the_streamed_line() -> None:
    stream, source, enqueued = _Stream(), _Source(), []
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)

    controller = _start_with(stream, source, enqueued, bar0)

    assert stream.start["schema_version"].endswith("v3") and stream.session_id == "s-e0"
    assert enqueued[0].source == "remote:0"
    assert stream.sent[-1] == {"type": "committed", "bar": 0, "lyric_id": bar0.lyric_id,
                               "source": SOURCE_REMOTE_LATEST, "text": "remote:0"}
    controller.close()


def test_commit_prefers_latest_remote_then_older_remote_then_streamed_espeak_then_local() -> None:
    stream, source, enqueued = _Stream(), _Source(), []
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0)
    choices = []
    try:
        # bar 1: latest remote is ready.
        bar1 = _record(1)
        source.ready[bar1.lyric_id] = _fetch(bar1)
        _push(stream, bar1)
        # bar 2: only revision 0 audio arrived; revision 1 is the latest line.
        bar2_old, bar2_new = _record(2, 0, "old"), _record(2, 1, "new")
        source.ready[bar2_old.lyric_id] = _fetch(bar2_old)
        _push(stream, bar2_old)
        _push(stream, bar2_new)
        # bar 3: a streamed line but no audio; bar 4: nothing streamed at all.
        _push(stream, _record(3))
        for bar in (1, 2, 3, 4):
            controller.on_tick(bar * 16 - 1)
            choices.append(stream.sent[-1]["source"])
        assert choices == [SOURCE_REMOTE_LATEST, SOURCE_REMOTE_OLDER, SOURCE_ESPEAK_STREAM, SOURCE_ESPEAK_LOCAL]
        assert [item.bar for item in enqueued] == [0, 1, 2, 3, 4]
        assert enqueued[2].source == "remote:0" and enqueued[2].fallback_reason == SOURCE_REMOTE_OLDER
        assert enqueued[3].source.startswith("espeak:flow")
        assert enqueued[4].source.startswith("espeak:local")
    finally:
        controller.close()


def test_older_version_keeps_fetching_the_exact_revision() -> None:
    stream, source, enqueued = _Stream(), _Source(), []
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0)
    try:
        bar1_old, bar1_new = _record(1, 0, "old"), _record(1, 1, "new")
        source.ready[bar1_old.lyric_id] = BarAudioFetch("superseded")
        _push(stream, bar1_old)
        older = _fetch(bar1_old, status="older_version")
        responses = iter([older, _fetch(bar1_new)])
        source.fetch = lambda *a, **k: (source.calls.append((a[1], a[2], k["accept_older"])) or next(responses))
        _push(stream, bar1_new)

        controller.on_tick(15)

        assert enqueued[-1].source == "remote:1"
        assert [call[2] for call in source.calls[-2:]] == [True, False]
    finally:
        controller.close()


def test_listener_edit_is_sent_and_resets_local_fallbacks_for_uncommitted_bars() -> None:
    stream, source, enqueued = _Stream(), _Source(), []
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0)
    try:
        edit = controller.update_plan(from_bar=2, topic="deep sea", template_id="baseline_straight_9")

        update = stream.sent[-1]
        assert update["type"] == "update" and update["edit"]["topic"] == "deep sea"
        assert update["edit"]["edit_seq"] == edit.edit_seq == 1
        assert update["edit"]["flow_template"]["template_id"] == "baseline_straight_9"
        assert controller.plan_control_state["last_edit"]["effective_from_bar"] == 2
        assert controller.plan_control_state["next_editable_bar"] == 1
        controller.reset()
        assert controller.plan_control_state["last_edit"] is None
    finally:
        controller.close()


def test_unknown_live_topic_keeps_local_safety_audio_for_new_bars() -> None:
    scenario = default_scenario()
    analysis = _Analyzer().analyze(" ".join(["local"] * 9))
    catalog = PrevalidatedFallbackCatalog({
        (segment.topic, segment.template_id): (analysis,) for segment in scenario.segments
    })
    stream, source, enqueued, publisher = _Stream(), _Source(), [], _Publisher()
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0, publisher=publisher, fallback_catalog=catalog)
    try:
        controller.update_plan(from_bar=1, topic="rainbows", template_id="baseline_straight_9")
        stream.handler({"type": "superseded", "bars": [1, 2, 3]})
        # No new lyric or remote audio arrives. Cross the original prepared window.
        for bar in range(1, 8):
            controller.on_tick(bar * 16 - 1)

        assert [audio.bar for audio in enqueued] == list(range(8))
        frozen = [data["payload"] for kind, data in publisher.events if kind == RapEventType.BAR_FROZEN]
        assert all(item["choice"] == SOURCE_ESPEAK_LOCAL for item in frozen[1:])
        assert all(item["topic"] != "rainbows" for item in frozen[1:])
        assert all(item["template_id"] == "baseline_straight_9" for item in frozen[1:])
        assert any(kind == RapEventType.PRESENTATION_ERROR and data["payload"]["component"] == "fallback_catalog"
                   for kind, data in publisher.events)
    finally:
        controller.close()


def test_live_edit_retains_ready_fallback_while_replacement_is_pending() -> None:
    stream, source, enqueued, publisher = _Stream(), _Source(), [], _Publisher()
    executor = _Pausable()
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0, publisher=publisher, render_executor=executor)
    try:
        executor.paused = True
        controller.update_plan(from_bar=1, topic="rainbows", template_id="baseline_straight_9")
        stream.handler({"type": "superseded", "bars": [1, 2, 3]})
        controller.on_tick(15)

        assert [audio.bar for audio in enqueued] == [0, 1]
        frozen = [data["payload"] for kind, data in publisher.events if kind == RapEventType.BAR_FROZEN][-1]
        assert frozen["choice"] == SOURCE_ESPEAK_LOCAL
        assert frozen["topic"] == "space"
        assert frozen["template_id"] == default_scenario().segment_for_bar(1).template_id
    finally:
        controller.close()


def test_committed_flow_is_the_played_revision_not_the_latest_reserved_flow() -> None:
    from dataclasses import replace
    from streammuse.domain.rap.events import RapEventType

    stream, source, enqueued, publisher = _Stream(), _Source(), [], _Publisher()
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0, publisher=publisher)
    try:
        old = _record(1)
        template = BUILTIN_TEMPLATES.get("baseline_staggered_9")
        new = replace(_record(1, 1), flow_template=template,
                      scheduled=align_exact(_Analyzer().analyze(old.text), materialize_flow(template, 1)))
        source.ready[old.lyric_id] = _fetch(old)
        _push(stream, old)
        _push(stream, new)
        controller.on_tick(15)
        frozen = [data for kind, data in publisher.events if kind == RapEventType.BAR_FROZEN][-1]["payload"]
        assert frozen["choice"] == SOURCE_REMOTE_OLDER
        assert frozen["flow"]["template_id"] == old.flow_template.template_id
        assert len(frozen["scheduled_syllables"]) == len(old.scheduled)
    finally:
        controller.close()


def test_reset_starts_a_new_server_session() -> None:
    stream, source, enqueued = _Stream(), _Source(), []
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0)
    try:
        epoch = controller.reset()
        assert epoch == 1 and controller.session_id == "s-e1"
        assert {"type": "stop"} in stream.sent
    finally:
        controller.close()


def test_an_older_revision_already_fetched_is_not_fetched_again() -> None:
    stream, source, enqueued = _Stream(), _Source(), []
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _start_with(stream, source, enqueued, bar0)
    try:
        bar1_old, bar1_new = _record(1, 0, "old"), _record(1, 1, "new")
        source.ready[bar1_old.lyric_id] = _fetch(bar1_old)
        _push(stream, bar1_old)
        source.calls.clear()
        source.ready[bar1_new.lyric_id] = _fetch(bar1_new)
        _push(stream, bar1_new)

        controller.on_tick(15)

        assert source.calls == [(1, bar1_new.lyric_id, False)]
        assert enqueued[-1].source == "remote:1"
    finally:
        controller.close()


class _FailingCatalog:
    def line_for(self, request):
        raise ValueError("no prevalidated fallback for this topic")


def test_nothing_ready_at_commit_plays_silence_and_keeps_moving() -> None:
    stream, source, enqueued, publisher = _Stream(), _Source(), [], _Publisher()
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _controller(stream, source, enqueued, publisher)
    controller._fallback_catalog = _FailingCatalog()
    thread = threading.Thread(target=controller.start)
    thread.start()
    deadline = time.monotonic() + 2
    while stream.handler is None and time.monotonic() < deadline:
        time.sleep(0.005)
    _push(stream, bar0)
    thread.join(5)
    try:
        # bar 1: no lyric, no remote audio and the local fallback fails.
        controller.on_tick(15)

        silent = enqueued[-1]
        assert silent.bar == 1 and silent.source == "silence" and not any(silent.audio.data)
        assert silent.audio.frame_count == bar_frame_count(1, Tempo(90.0, 4, 4), AudioFormat())
        assert stream.sent[-1] == {"type": "committed", "bar": 1, "lyric_id": None, "source": "silence", "text": ""}
        assert controller.next_editable_bar == 2  # the playhead moved on
        errors = [kwargs["payload"]["component"] for kind, kwargs in publisher.events if kind.value == "presentation_error"]
        assert "local_fallback" in errors and "bar_stream" in errors
    finally:
        controller.close()


def test_local_fallback_after_an_edit_to_a_new_topic_uses_any_matching_line() -> None:
    class _EditAwareCatalog(_Catalog):
        def line_for(self, request):
            if request.topic != "space":
                raise ValueError("no prevalidated fallback for this topic")
            return super().line_for(request)

        def line_for_any(self, request):
            return super().line_for(request)

    stream, source, enqueued = _Stream(), _Source(), []
    bar0 = _record(0)
    source.ready[bar0.lyric_id] = _fetch(bar0)
    controller = _controller(stream, source, enqueued)
    controller._fallback_catalog = _EditAwareCatalog()
    thread = threading.Thread(target=controller.start)
    thread.start()
    deadline = time.monotonic() + 2
    while stream.handler is None and time.monotonic() < deadline:
        time.sleep(0.005)
    _push(stream, bar0)
    thread.join(5)
    try:
        controller.update_plan(from_bar=1, topic="volcano")
        controller.on_tick(15)

        assert enqueued[-1].bar == 1 and enqueued[-1].source.startswith("espeak:local")
        assert stream.sent[-1]["source"] == SOURCE_ESPEAK_LOCAL
    finally:
        controller.close()
