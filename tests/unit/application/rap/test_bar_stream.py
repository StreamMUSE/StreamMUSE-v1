from __future__ import annotations

from concurrent.futures import Executor, Future
from dataclasses import dataclass
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
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
from streammuse.domain.rap.events import RapEventType
from streammuse.domain.timing import Tempo
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
        assert frozen["template_id"] == "baseline_syncopated_9"
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
