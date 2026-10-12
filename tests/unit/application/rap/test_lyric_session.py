from __future__ import annotations

import threading
import time

import pytest

from streammuse.application.rap.live_plan import LivePlan
from streammuse.application.rap.lyric_session import (
    FETCH_OLDER_VERSION,
    FETCH_PENDING,
    FETCH_READY,
    FETCH_SUPERSEDED,
    FETCH_UNKNOWN,
    LyricStreamSession,
    RenderedBarAudio,
)
from streammuse.application.rap.lyric_writer import LyricWriterPolicy, SequentialLyricWriter
from streammuse.domain.rap import (
    CandidateBatch,
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    ProsodyAnalysis,
    ScoreWeights,
    Syllable,
    normalize_text,
)
from streammuse.domain.rap.lyric_stream import BarAudioManifest, PlanEdit

_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")


def _flow(template_id: str = "four") -> FlowTemplate:
    return FlowTemplate(
        template_id=template_id,
        name=template_id,
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=tuple(FlowSlot(tick_in_bar=tick, duration_ticks=4, target_stress=1.0) for tick in (0, 4, 8, 12)),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


class _Analyzer:
    def analyze(self, text: str) -> ProsodyAnalysis:
        normalized = normalize_text(text)
        words = normalized.split()
        return ProsodyAnalysis(
            text=text,
            normalized_text=normalized,
            syllables=tuple(Syllable(word, 0, 1, 1, ("AH1",), "fake") for word in words),
            end_rhyme_tail=("AH1",),
            oov_words=(),
            heuristic_words=(),
            punctuation_boundary_after=(),
        )


class _Generator:
    """Four-word lines that encode topic, bar and revision so tests can read them back."""

    def __init__(self) -> None:
        self.requests = []
        self.lock = threading.Lock()

    def generate(self, request):
        with self.lock:
            self.requests.append(request)
            index = len(self.requests)
        topic = request.topic.replace(" ", "")
        lines = tuple(
            f"{topic} {_WORDS[request.target_bar % 10]} {_WORDS[index % 10]} {_WORDS[choice]}"
            for choice in range(request.count)
        )
        return CandidateBatch(
            request_id=request.request_id, candidates=lines, source="fake", prompt=(), raw_response="", latency_ms=0.0
        )


class _Renderer:
    def __init__(self) -> None:
        self.rendered = []
        self.gate = threading.Event()
        self.gate.set()
        self.lock = threading.Lock()

    def render(self, record, *, tempo_bpm):
        assert self.gate.wait(5)
        with self.lock:
            self.rendered.append((record.bar, record.revision))
        return RenderedBarAudio(
            lyric_id=record.lyric_id,
            bar=record.bar,
            revision=record.revision,
            package=f"audio:{record.bar}:{record.revision}".encode(),
            media_type="application/test",
            manifest=BarAudioManifest(
                lyric_id=record.lyric_id,
                bar=record.bar,
                tempo_bpm=tempo_bpm,
                frame_count=64_000,
                source_sha256="b" * 64,
                source_onsets=(0.1,),
                onset_confidence=(0.9,),
                stage_timings_ms={"moss": 1.0},
            ),
            render_ms=1.0,
        )


class _Listener:
    def __init__(self) -> None:
        self.lyrics = []
        self.superseded_bars = []
        self.ready = []
        self.errors = []

    def lyric(self, record):
        self.lyrics.append(record)

    def superseded(self, bars, edit_seq):
        self.superseded_bars.append((bars, edit_seq))

    def audio_ready(self, bar, lyric_id):
        self.ready.append((bar, lyric_id))

    def error(self, message, *, bar=None):
        self.errors.append((bar, message))


def _session(renderer=None, listener=None, **overrides):
    writer = SequentialLyricWriter(
        _Generator(),
        _Analyzer(),
        ScoreWeights(),
        LyricWriterPolicy(candidates=2, rescue_candidates=0, minimum_valid=1, rhyme_scheme="none"),
    )
    values = dict(
        session_id="s",
        tempo_bpm=90.0,
        plan=LivePlan(lambda bar: ("sea", _flow())),
        writer=writer,
        renderer=renderer or _Renderer(),
        listener=listener or _Listener(),
        start_bar=0,
        lyric_lead=3,
        audio_lead=2,
    )
    values.update(overrides)
    session = LyricStreamSession(**values)
    session.start()
    return session


def _wait(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    raise AssertionError("condition not reached")


def test_writes_lyrics_in_bar_order_up_to_the_lead_and_renders_up_to_the_audio_lead() -> None:
    renderer, listener = _Renderer(), _Listener()
    session = _session(renderer, listener)
    try:
        _wait(lambda: len(listener.lyrics) == 3 and len(renderer.rendered) == 2)
        time.sleep(0.05)
        assert [record.bar for record in listener.lyrics] == [0, 1, 2]
        assert renderer.rendered == [(0, 0), (1, 0)]
        # Each line follows the one before.
        assert listener.lyrics[2].provenance["seed"] != listener.lyrics[1].provenance["seed"]

        session.committed(0, listener.lyrics[0].lyric_id, listener.lyrics[0].text)
        _wait(lambda: len(listener.lyrics) == 4 and len(renderer.rendered) == 3)
        assert listener.lyrics[3].bar == 3
        assert renderer.rendered[-1] == (2, 0)
    finally:
        session.close()


def test_fetch_returns_ready_audio_at_once_and_unknown_ids_are_reported() -> None:
    listener = _Listener()
    session = _session(listener=listener)
    try:
        _wait(lambda: len(listener.ready) >= 1)
        record = listener.lyrics[0]

        result = session.fetch_audio(0, record.lyric_id, wait_seconds=0.0)

        assert result.status == FETCH_READY
        assert result.audio.package == b"audio:0:0"
        assert session.fetch_audio(0, "f" * 64, wait_seconds=0.0).status == FETCH_UNKNOWN
    finally:
        session.close()


def test_fetch_beyond_the_audio_lead_jumps_the_queue_and_waits() -> None:
    listener = _Listener()
    session = _session(listener=listener, lyric_lead=3, audio_lead=1)
    try:
        _wait(lambda: len(listener.lyrics) == 3 and len(listener.ready) == 1)
        bar_two = listener.lyrics[2]

        result = session.fetch_audio(2, bar_two.lyric_id, wait_seconds=5.0)

        assert result.status == FETCH_READY
        assert result.audio.bar == 2
    finally:
        session.close()


def test_edit_rewrites_uncommitted_bars_and_serves_the_old_version_until_the_new_one_is_ready() -> None:
    renderer, listener = _Renderer(), _Listener()
    session = _session(renderer, listener)
    try:
        _wait(lambda: len(listener.lyrics) == 3 and len(renderer.rendered) == 2)
        session.committed(0, listener.lyrics[0].lyric_id, listener.lyrics[0].text)
        _wait(lambda: len(renderer.rendered) == 3)
        old_bar_one = listener.lyrics[1]
        renderer.gate.clear()  # hold rendering so the new revision is not ready yet

        superseded = session.apply_edit(PlanEdit(edit_seq=1, from_bar=0, topic="space"))

        assert superseded[0] == 1  # bar 0 is committed and stays
        _wait(lambda: any(r.bar == 1 and r.revision == 1 for r in listener.lyrics))
        new_bar_one = next(r for r in listener.lyrics if r.bar == 1 and r.revision == 1)
        assert new_bar_one.topic == "space" and new_bar_one.text.startswith("space")

        # A client still holding the old id gets that ready audio back.
        old_fetch = session.fetch_audio(1, old_bar_one.lyric_id, wait_seconds=0.0)
        assert old_fetch.status == FETCH_READY and old_fetch.audio.revision == 0
        # Bar 3's old line was never rendered: asking for it does not start a render.
        old_bar_three = next(r for r in listener.lyrics if r.bar == 3 and r.revision == 0)
        assert session.fetch_audio(3, old_bar_three.lyric_id, wait_seconds=0.0).status == FETCH_SUPERSEDED
        new_fetch = session.fetch_audio(1, new_bar_one.lyric_id, wait_seconds=0.0)
        assert new_fetch.status == FETCH_OLDER_VERSION
        assert new_fetch.audio.revision == 0

        renderer.gate.set()
        ready = session.fetch_audio(1, new_bar_one.lyric_id, wait_seconds=5.0, accept_older=False)
        assert ready.status == FETCH_READY and ready.audio.revision == 1
    finally:
        renderer.gate.set()
        session.close()


def test_pending_when_nothing_is_ready_within_the_wait() -> None:
    renderer, listener = _Renderer(), _Listener()
    renderer.gate.clear()
    session = _session(renderer, listener)
    try:
        _wait(lambda: len(listener.lyrics) >= 1)

        result = session.fetch_audio(0, listener.lyrics[0].lyric_id, wait_seconds=0.05)

        assert result.status == FETCH_PENDING
    finally:
        renderer.gate.set()
        session.close()


def test_edit_seq_must_increase() -> None:
    session = _session()
    try:
        session.apply_edit(PlanEdit(edit_seq=2, from_bar=0, topic="space"))
        with pytest.raises(ValueError):
            session.apply_edit(PlanEdit(edit_seq=2, from_bar=0, topic="code"))
    finally:
        session.close()
