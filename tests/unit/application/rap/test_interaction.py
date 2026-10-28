from concurrent.futures import Executor, Future, ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import pytest

from streammuse.application.rap.alignment import align_exact
from streammuse.application.rap.bar_stream import BarAudioFetch
from streammuse.application.rap.interaction import AlternatingRapController, ProgressiveLyricTurnAdapter
from streammuse.domain.rap import ProsodyAnalysis, Syllable, materialize_flow, normalize_text
from streammuse.domain.rap.audio import AudioFormat, PcmAudio, PreparedRapBar, RapMixConfig, RapWarpDiagnostics
from streammuse.domain.rap.events import RapEventType
from streammuse.domain.rap.lyric_stream import BarAudioManifest, BarLyricRecord
from streammuse.domain.rap.timeline import RapTimeline
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.flow_echo import FlowEchoRenderer
from streammuse.infrastructure.rap.scenarios import default_scenario
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


class Inline(Executor):
    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


class Pausable(Inline):
    def __init__(self):
        self.pending = []

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        self.pending.append((future, fn, args, kwargs))
        return future

    def finish(self):
        for future, fn, args, kwargs in self.pending:
            future.set_result(fn(*args, **kwargs))
        self.pending.clear()


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class Stream:
    def __init__(self):
        self.opens = []
        self.sent = []
        self.handler = None

    def open(self, session_id, start, on_message):
        self.opens.append((session_id, start))
        self.handler = on_message
        on_message({"type": "started", "capabilities": ["bounded_turns_v1", "tempo_updates_v1"]})

    def get_capabilities(self):
        return frozenset(("bounded_turns_v1", "tempo_updates_v1"))

    def request_tempo(self, payload, **kwargs):
        self.sent.append({"type": "tempo_update", "update": payload})
        return {"effective_from_bar": payload["from_bar"], "bpm": payload["bpm"], "revision": payload["edit_seq"], "edit_seq": payload["edit_seq"]}

    def send(self, message):
        self.sent.append(message)

    def close(self):
        pass


class Analyzer:
    def analyze(self, text):
        return ProsodyAnalysis(
            text=text, normalized_text=normalize_text(text),
            syllables=tuple(Syllable(word, 0, 1, 1, ("AH1",), "fixture") for word in text.split()),
            end_rhyme_tail=("AH1",), oov_words=(), heuristic_words=(), punctuation_boundary_after=(),
        )


class Drums:
    def render(self, template, tempo, audio_format, bar):
        frames = round(tempo.tick_to_seconds((bar + 1) * 16) * 48000) - round(tempo.tick_to_seconds(bar * 16) * 48000)
        return PcmAudio(audio_format, frames, bytes(frames * 8))


def prepared(bar, text, scheduled, source):
    audio = PcmAudio(AudioFormat(), 192000, bytes(192000 * 8))
    return PreparedRapBar(bar, text, source, None, scheduled, audio, (), (), 0.0,
                          vocal_stem=audio, drum_stem=audio, mix_config=RapMixConfig())


class Fallback:
    def set_mix_config(self, config):
        self.mix_config = config

    def render(self, plan, **kwargs):
        result = prepared(plan.bar, plan.text, plan.scheduled, "lyric_stream_espeak")
        if kwargs.get("timing") is not None:
            timing = kwargs["timing"]
            assert timing.bar == plan.bar
            audio = PcmAudio(AudioFormat(), timing.frame_count, bytes(timing.frame_count * 8))
            result = replace(result, audio=audio, vocal_stem=audio, drum_stem=audio, timing=timing)
        return result


class Preparer:
    def set_mix_config(self, config):
        self.mix_config = config

    def prepare(self, record, manifest, source_wav, **kwargs):
        assert manifest.lyric_id == record.lyric_id
        if kwargs.get("timing") is not None:
            assert kwargs["timing"].bar == record.bar
            assert manifest.bar == record.bar
            assert manifest.tempo_bpm == kwargs["timing"].bpm
        return prepared(record.bar, record.text, record.scheduled, "moss_lyric_stream")


class PolicyPreparer(Preparer):
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
        result = replace(super().prepare(record, manifest, source_wav, **kwargs), source=self.policy,
                         warp_diagnostics=RapWarpDiagnostics((0.1,), (0.1,), (0.1,), (1.0,), self.policy, 0.0))
        timing = kwargs.get("timing")
        if timing is not None:
            audio = PcmAudio(AudioFormat(), timing.frame_count, bytes(timing.frame_count * 8))
            result = replace(result, audio=audio, vocal_stem=audio, drum_stem=audio, timing=timing)
        return result


class Source:
    def __init__(self):
        self.ready = {}
        self.calls = []

    def fetch(self, session_id, bar, lyric_id, **kwargs):
        self.calls.append((session_id, bar, lyric_id, kwargs))
        return self.ready.get(lyric_id, BarAudioFetch("failed"))

    def close(self):
        pass


class Publisher:
    def __init__(self):
        self.events = []

    def emit(self, kind, **kwargs):
        self.events.append((kind, kwargs))


class Harness:
    def __init__(self, *, initial_epoch=0, render_executor=None, capture_grace_seconds=0.0, stream=None, source=None, preparer=None, scenario=None, **controller_options):
        self.clock, self.stream, self.source, self.publisher = Clock(), stream or Stream(), source or Source(), Publisher()
        self.enqueued = []
        self.adapter = ProgressiveLyricTurnAdapter(
            lyric_stream=self.stream, audio_source=self.source, preparer=preparer or Preparer(), fallback_renderer=Fallback(),
            analyzer=Analyzer(), tempo=Tempo(60, 4, 4), seed=3, monotonic=self.clock,
            fetch_executor=Inline(), render_executor=render_executor or Inline(), stream_executor=Inline(),
        )
        scenario = scenario or default_scenario()
        self.controller = AlternatingRapController(
            tempo=Tempo(60, 4, 4), scenario=scenario, templates=BUILTIN_TEMPLATES,
            generation=self.adapter, echo_renderer=FlowEchoRenderer(
                drums=Drums(), idle_template=BUILTIN_TEMPLATES.get(scenario.segments[0].template_id),
            ), publisher=self.publisher, enqueue=self.enqueued.append, session_id="alternating", initial_epoch=initial_epoch,
            monotonic=self.clock,
            capture_grace_seconds=capture_grace_seconds,
            **controller_options,
        )

    def tick(self, tick):
        self.clock.now = 100.0 + tick / 4.0
        self.controller.on_tick(tick)

    def capture(self, taps=True):
        opens_before = len(self.stream.opens)
        self.controller.start()
        self.tick(2)
        self.controller.arm_capture()
        self.tick(15)
        self.tick(16)
        if taps:
            self.clock.now = 104.02
            assert self.controller.record_tap(server_time_seconds=104.0, event_id="a")["accepted"]
            self.clock.now = 105.02
            assert self.controller.record_tap(server_time_seconds=105.0, event_id="b")["accepted"]
        self.tick(31)
        assert len(self.stream.opens) == opens_before
        self.tick(32)

    def lyric(self, bar, *, remote=True, word="line"):
        session_id, start = self.stream.opens[-1]
        template = self.controller.captured_template
        text = " ".join([word] * len(template.slots))
        record = BarLyricRecord(
            session_id, bar, 0, "space", template, text,
            align_exact(Analyzer().analyze(text), materialize_flow(template, bar)), 0.5,
        )
        if remote:
            manifest = BarAudioManifest(record.lyric_id, bar, 60.0, 96000, "b" * 64,
                                        tuple(slot.tick_in_bar / 4 for slot in template.slots),
                                        tuple(1.0 for slot in template.slots), {})
            self.source.ready[record.lyric_id] = BarAudioFetch("ready", record.lyric_id, manifest, b"wave")
        self.stream.handler({"type": "lyric", "record": record.to_payload()})
        return record


def test_alignment_update_rewarps_cached_turn_without_remote_messages_or_committed_changes():
    preparer = PolicyPreparer()
    h = Harness(preparer=preparer, initial_epoch=9)
    h.capture()
    first, second = h.lyric(0), h.lyric(1)
    h.tick(46)
    h.tick(47)
    committed = h.enqueued[-1]
    assert committed.source == "gentle_sparse_r3"
    h.tick(48)
    calls, messages, opens = list(h.source.calls), list(h.stream.sent), list(h.stream.opens)
    ack = h.controller.update_alignment_policy("all_onsets_stress_r3")
    assert ack == {"effective_from_bar": 4, "alignment_policy": "all_onsets_stress_r3",
                   "revision": 1, "coordinator_epoch": 9}
    h.tick(62)
    assert h.source.calls == calls and h.stream.sent == messages and h.stream.opens == opens
    assert h.enqueued[-1] is committed
    assert preparer.calls[-1] == (4, "all_onsets_stress_r3", b"wave")
    assert first.lyric_id not in h.adapter._raw and second.lyric_id in h.adapter._raw
    h.tick(63)
    assert h.enqueued[-1].source == "all_onsets_stress_r3"
    assert not h.adapter._raw and not h.adapter._prepared and not h.adapter._requested
    h.controller.close()


def test_alignment_turn_rejects_running_old_policy_but_keeps_its_source():
    preparer = PolicyPreparer()
    h = Harness(preparer=preparer)
    h.capture()
    record = h.lyric(0)

    def switch(record, policy):
        if policy == "gentle_sparse_r3":
            h.controller.update_alignment_policy("all_onsets_r3")

    preparer.during_prepare = switch
    h.tick(45)
    assert record.lyric_id in h.adapter._raw
    h.tick(46)
    h.tick(47)
    assert len(h.source.calls) == 1
    assert h.enqueued[-1].source == "all_onsets_r3"
    h.controller.close()


def test_alignment_update_keeps_inflight_turn_fetch_and_latest_policy_wins():
    h = Harness(preparer=PolicyPreparer())
    h.capture()
    fetch = h.source.fetch

    def switch_during_fetch(*args, **kwargs):
        h.controller.update_alignment_policy("all_onsets_r3")
        return fetch(*args, **kwargs)

    h.source.fetch = switch_during_fetch
    record = h.lyric(0)
    assert record.lyric_id in h.adapter._raw
    h.controller.update_alignment_policy("all_onsets_stress_r3")
    h.tick(46)
    h.tick(47)
    assert len(h.source.calls) == 1
    assert h.enqueued[-1].source == "all_onsets_stress_r3"
    h.controller.close()


def test_alignment_turn_cache_prunes_superseded_records_and_cancelled_turns():
    h = Harness(preparer=PolicyPreparer())
    h.capture()
    original = h.lyric(0)
    for revision in range(1, 10):
        record = replace(original, revision=revision)
        raw = h.source.ready[original.lyric_id]
        h.source.ready[record.lyric_id] = replace(raw, lyric_id=record.lyric_id,
                                                 manifest=replace(raw.manifest, lyric_id=record.lyric_id))
        h.stream.handler({"type": "lyric", "record": record.to_payload()})
        h.tick(46)
        assert len(h.adapter._raw) == 1
        assert len(h.adapter._prepared) <= 2
    h.controller.update_alignment_policy("all_onsets_r3")
    h.controller.request_stop(successor_bar=3)
    assert not h.adapter._raw and not h.adapter._prepared and not h.adapter._requested
    h.controller.reset()
    assert h.controller.update_alignment_policy("all_onsets_stress_r3")["effective_from_bar"] == 0
    h.controller.close()


def test_alignment_turn_queued_renders_use_only_latest_policy_and_reset_fences_them():
    executor, preparer = Pausable(), PolicyPreparer()
    h = Harness(preparer=preparer, render_executor=executor)
    h.capture()
    h.lyric(0)
    h.tick(43)
    h.controller.update_alignment_policy("all_onsets_r3")
    h.tick(44)
    ack = h.controller.update_alignment_policy("all_onsets_stress_r3")
    assert ack["revision"] == 2 and ack["effective_from_bar"] == 3
    with pytest.raises(ValueError):
        h.controller.update_alignment_policy("invalid")
    h.tick(45)
    executor.finish()
    assert [policy for _, policy, _ in preparer.calls] == ["all_onsets_stress_r3"]
    h.tick(47)
    assert h.enqueued[-1].source == "all_onsets_stress_r3"
    h.lyric(1)
    h.tick(48)
    h.controller.reset()
    before = list(preparer.calls)
    executor.finish()
    assert preparer.calls == before and not h.adapter._raw and not h.adapter._prepared
    h.controller.close()
    with pytest.raises(RuntimeError):
        h.controller.update_alignment_policy("all_onsets_r3")


@pytest.mark.parametrize("change_policy_again", [False, True])
def test_alignment_turn_ack_bypasses_pending_tempo_and_rejects_old_timing_warp(change_policy_again):
    ack_entered, ack_release = threading.Event(), threading.Event()
    warp_entered, warp_release = threading.Event(), threading.Event()

    class SlowStream(Stream):
        def request_tempo(self, payload, **kwargs):
            ack_entered.set()
            assert ack_release.wait(5)
            return super().request_tempo(payload, **kwargs)

    class BlockingPreparer(PolicyPreparer):
        def prepare(self, record, manifest, source_wav, **kwargs):
            if manifest.tempo_bpm == 60:
                warp_entered.set()
                assert warp_release.wait(5)
            return super().prepare(record, manifest, source_wav, **kwargs)

    renders, stream_worker, controls = (ThreadPoolExecutor(max_workers=count) for count in (3, 1, 1))
    h = Harness(stream=SlowStream(), preparer=BlockingPreparer(), render_executor=renders)
    try:
        h.capture()
        h.adapter._stream_executor = stream_worker
        record = h.lyric(0)
        committed = h.enqueued[-1]
        h.controller.update_alignment_policy("all_onsets_r3")
        h.tick(40)
        assert warp_entered.wait(2)
        h.controller.update_tempo(120)
        h.tick(41)
        assert ack_entered.wait(2)
        calls, messages = list(h.source.calls), list(h.stream.sent)
        policy = "all_onsets_stress_r3" if change_policy_again else "all_onsets_r3"
        if change_policy_again:
            ack = controls.submit(h.controller.update_alignment_policy, policy).result(timeout=0.5)
            assert ack == {"effective_from_bar": 3, "alignment_policy": policy,
                           "revision": 2, "coordinator_epoch": 0}
        controls.submit(h.tick, 42).result(timeout=0.5)
        assert h.source.calls == calls and h.stream.sent == messages
        assert h.adapter._tempo_pending is not None
        fetched = h.source.ready[record.lyric_id]
        h.source.ready[record.lyric_id] = replace(fetched, manifest=replace(fetched.manifest, tempo_bpm=120))
        ack_release.set()
        stream_worker.submit(lambda: None).result(timeout=2)
        h.tick(43)
        deadline = time.monotonic() + 2
        while not any(key[-1] == "remote_latest" for key in h.adapter._prepared) and time.monotonic() < deadline:
            time.sleep(0.005)
        assert any(key[-1] == "remote_latest" for key in h.adapter._prepared)
        warp_release.set()
        renders.shutdown(wait=True)
        assert all(audio.timing.bpm == 120 for audio in h.adapter._prepared.values())
        assert h.adapter._raw[record.lyric_id].manifest.tempo_bpm == 120
        assert h.enqueued[-1] is committed
        h.tick(47)
        assert h.enqueued[-1].warp_diagnostics.policy == policy
        assert h.enqueued[-1].timing.bpm == 120
        event = [payload for kind, payload in h.publisher.events if kind == RapEventType.BAR_AUDIO_COMMITTED][-1]
        assert event["payload"]["timing_evidence"]["warp"]["policy"] == policy
    finally:
        ack_release.set()
        warp_release.set()
        h.controller.close()
        for executor in (renders, stream_worker, controls):
            executor.shutdown(wait=True)


def test_arm_next_full_bar_and_capture_submits_once_at_end_not_last_tick():
    h = Harness(initial_epoch=12)
    h.capture()
    assert h.controller.interaction_state()["phase"] == "WAIT_REPLAY"
    assert h.controller.interaction_state()["capture_bar"] == 1
    assert len(h.stream.opens) == 1
    session_id, start = h.stream.opens[0]
    assert "e12" in session_id
    assert start["end_bar_exclusive"] == 2
    assert start["turn_id"] == session_id
    assert start["start_bar"] == 0
    h.tick(32)
    assert len(h.stream.opens) == 1
    assert all(event[1]["payload"]["coordinator_epoch"] == 12 for event in h.publisher.events)


def test_two_progressive_responses_with_three_echoes_never_commit_echo_to_server():
    h = Harness()
    h.capture()
    h.tick(47)
    h.tick(63)
    first = h.lyric(0, word="first")
    h.tick(78)
    h.tick(79)
    assert h.enqueued[-1].bar == 5 and h.enqueued[-1].text == "first first"
    assert h.enqueued[-1].scheduled[0].slot.tick == 80
    assert h.enqueued[-1].scheduled[1].slot.tick == 84
    h.tick(95)
    second = h.lyric(1, word="second")
    h.tick(110)
    h.tick(111)
    h.tick(128)
    assert [(item.bar, item.source) for item in h.enqueued if item.source == "flow_echo"] == [(3, "flow_echo"), (4, "flow_echo"), (6, "flow_echo")]
    commits = [message for message in h.stream.sent if message["type"] == "committed"]
    assert [(item["bar"], item["lyric_id"]) for item in commits] == [(0, first.lyric_id), (1, second.lyric_id)]
    deliveries = [payload["payload"] for kind, payload in h.publisher.events if kind == RapEventType.BAR_FROZEN and payload["payload"].get("lyric_id")]
    assert [(item["logical_response_bar"], item["physical_bar"], item["intended_physical_bar"]) for item in deliveries] == [(0, 5, 2), (1, 7, 3)]
    assert h.controller.interaction_state()["phase"] == "DRUMS_IDLE"
    assert h.controller.interaction_state()["delivered_response_bars"] == 2
    assert h.controller.interaction_state()["wait_elapsed_seconds"] == 0.0


def test_empty_capture_and_duplicate_network_taps_do_not_submit_spurious_turns():
    h = Harness()
    h.capture(taps=False)
    assert h.stream.opens == []
    assert h.controller.interaction_state()["last_error"] == "empty_capture"
    assert h.controller.interaction_state()["phase"] == "DRUMS_IDLE"
    h.controller.arm_capture()
    h.tick(48)
    h.clock.now = 112.1
    assert h.controller.record_tap(server_time_seconds=112.0, event_id="same")["accepted"]
    assert not h.controller.record_tap(server_time_seconds=112.0, event_id="same")["accepted"]
    assert not h.controller.record_tap(server_time_seconds=112.1, repeat=True)["accepted"]
    assert not h.controller.record_tap(server_time_seconds=111.9)["accepted"]


def test_first_fallback_waits_thirty_seconds_and_matches_generated_lyric():
    h = Harness()
    h.capture()
    record = h.lyric(0, remote=False)
    h.tick(47)
    assert h.enqueued[-1].source == "flow_echo"
    h.tick(151)
    assert h.enqueued[-1].source == "flow_echo"
    h.tick(159)
    assert h.enqueued[-1].text == record.text
    assert h.enqueued[-1].fallback_reason == "espeak_stream_lyric"
    assert h.controller.interaction_state()["queued_response_bars"] == 1
    h.tick(160)
    assert h.controller.interaction_state()["delivered_response_bars"] == 1


def test_missing_first_and_late_bar_each_have_bounded_wait():
    h = Harness()
    h.capture()
    h.tick(152)
    assert h.controller.interaction_state()["last_error"] == "response_timeout"
    assert h.controller.interaction_state()["phase"] == "DRUMS_IDLE"
    h = Harness()
    h.capture()
    h.lyric(0)
    h.tick(46)
    h.tick(47)
    h.tick(63)
    assert h.controller.interaction_state()["phase"] == "WAIT_REPLAY"
    h.tick(183)
    assert h.controller.interaction_state()["last_error"] == "response_timeout"
    assert h.controller.interaction_state()["delivered_response_bars"] == 1


def test_stop_reset_fences_delayed_preparation_and_old_stream_callback():
    paused = Pausable()
    h = Harness(initial_epoch=8, render_executor=paused)
    h.capture()
    record = h.lyric(0)
    h.tick(46)
    old_callback = h.stream.handler
    h.controller.request_stop(successor_bar=3)
    assert h.controller.reset() == 9
    paused.finish()
    old_callback({"type": "lyric", "record": record.to_payload()})
    h.controller.start()
    h.tick(47)
    assert not any(item.text for item in h.enqueued)
    assert h.controller.interaction_state()["coordinator_epoch"] == 9
    assert h.controller.interaction_state()["phase"] == "DRUMS_IDLE"


def test_manual_stop_resumes_drums_without_restarting_cancelled_turn():
    h = Harness()
    h.capture()
    h.controller.request_stop(successor_bar=3)
    h.controller.resume_audio(3)
    h.controller.resume_after_stop()
    assert h.enqueued[-1].bar == 3 and h.enqueued[-1].source == "drums_idle"
    assert h.controller.interaction_state()["phase"] == "DRUMS_IDLE"


@pytest.mark.parametrize("count", [0, 9, True])
def test_invalid_response_length_rejected(count):
    h = Harness()
    with pytest.raises(ValueError):
        h.controller.set_response_bars(count)


def test_clock_mapping_exposes_event_clock_separately_from_receipt():
    h = Harness()
    h.controller.start()
    h.tick(0)
    h.controller.arm_capture()
    h.tick(16)
    h.clock.now = 104.3
    h.controller.record_tap(server_time_seconds=104.125, client_time_ms=550.0, uncertainty_ms=12.0)
    h.tick(32)
    state = h.controller.interaction_state()
    assert state["clock"]["monotonic_seconds"] == 108.0
    assert state["clock"]["current_tick"] == 32
    assert state["clock"]["playback_seconds"] == 8.0
    assert state["raw_taps"][0]["received_time_seconds"] == 104.3
    assert state["raw_taps"][0]["server_time_seconds"] == 104.125
    assert state["captured_ticks"] == [0]


def test_primary_preparation_rebinds_physical_destination_but_commits_original_identity():
    h = Harness()
    h.controller.set_timeline(RapTimeline(Tempo(60, 4, 4)))
    h.capture()
    record = h.lyric(0)
    h.tick(46)
    h.tick(47)
    assert h.enqueued[-1].text == record.text
    assert h.enqueued[-1].bar == 3
    assert h.enqueued[-1].timing.bar == 3
    commit = next(item for item in h.stream.sent if item["type"] == "committed")
    assert commit["lyric_id"] == record.lyric_id and commit["bar"] == 0


def test_legacy_server_rejected_before_bounded_start_and_capabilities_are_negotiated():
    h = Harness()
    h.stream.get_capabilities = lambda: frozenset()
    h.capture()
    assert h.stream.opens == []
    assert "bounded_turns_v1" in h.controller.interaction_state()["last_error"]
    supported = Harness()
    supported.capture()
    assert "bounded_turns_v1" in supported.stream.opens[0][1]["capabilities"]


def test_tempo_change_does_not_distort_capture_and_reprepares_matching_fallback():
    h = Harness()
    timeline = RapTimeline(Tempo(60, 4, 4))
    h.controller.set_timeline(timeline)
    h.controller.start()
    h.tick(2)
    h.controller.arm_capture()
    h.tick(16)
    h.clock.now = 105.01
    h.controller.record_tap(server_time_seconds=105.0)
    h.controller.set_timeline(timeline.with_tempo(120.0, from_bar=2))
    h.tick(31)
    h.tick(32)
    record = h.lyric(0)
    h.tick(46)
    h.tick(47)
    assert h.enqueued[-1].source == "flow_echo"
    assert h.enqueued[-1].audio.frame_count == 96000
    assert h.controller.interaction_state()["captured_ticks"] == [4]
    h.tick(159)
    assert h.enqueued[-1].text == record.text
    assert h.enqueued[-1].audio.frame_count == 96000
    assert h.enqueued[-1].fallback_reason == "espeak_stream_lyric"


def test_wrong_turn_wrong_flow_and_duplicate_lyrics_cannot_advance_response():
    h = Harness()
    h.capture()
    record = h.lyric(0)
    requests = len(h.source.calls)
    h.stream.handler({"type": "lyric", "record": record.to_payload()})
    assert len(h.source.calls) == requests
    h.stream.handler({"type": "lyric", "record": replace(record, session_id="another-turn").to_payload()})
    assert "another turn" in h.adapter.error
    h.tick(46)
    h.tick(47)
    assert h.enqueued[-1].text == record.text


def test_last_tick_15_8_tap_arriving_after_boundary_is_included_within_explicit_grace():
    h = Harness(capture_grace_seconds=0.1)
    h.controller.start()
    h.tick(2)
    h.controller.arm_capture()
    h.tick(16)
    h.tick(31)
    assert h.enqueued[-1].bar == 2
    assert h.enqueued[-1].source == "capture_transition"
    h.tick(32)
    assert h.stream.opens == []
    h.clock.now = 108.04
    result = h.controller.record_tap(server_time_seconds=107.95, client_time_ms=2500.0, uncertainty_ms=10)
    assert result["accepted"]
    h.clock.now = 108.11
    assert not h.controller.record_tap(server_time_seconds=107.99)["accepted"]
    h.tick(33)
    assert len(h.stream.opens) == 1
    assert h.controller.interaction_state()["captured_ticks"] == [15]
    assert h.controller.interaction_state()["transition_bar"] == 2
    assert h.controller.interaction_state()["raw_taps"][0]["server_time_seconds"] == 107.95


def test_new_turn_history_occupies_only_preceding_logical_bars():
    h = Harness()
    h.capture()
    h.lyric(0, word="first")
    h.tick(46)
    h.tick(47)
    h.lyric(1, word="second")
    h.tick(62)
    h.tick(63)
    h.tick(80)
    h.controller.arm_capture()
    h.tick(95)
    assert h.publisher.events[-1][1]["payload"]["turn_id"] is None
    assert h.publisher.events[-1][1]["payload"]["intended_physical_bar"] is None
    h.tick(96)
    h.clock.now = 124.05
    h.controller.record_tap(server_time_seconds=124.0)
    h.tick(111)
    h.tick(112)
    assert h.stream.opens[-1][1]["history"] == [{"bar": -2, "text": "first first"}, {"bar": -1, "text": "second second"}]


def test_live_tempo_applies_after_locked_capture_and_reports_physical_boundary():
    h = Harness()
    h.controller.start()
    h.tick(2)
    h.controller.arm_capture()
    h.tick(16)
    edit = h.controller.update_tempo(120.0, from_bar=1)
    assert edit["requested_from_bar"] == 1
    assert edit["effective_from_bar"] == 2
    assert h.controller.timeline.snapshot(1, 48000).bpm == 60
    assert h.controller.timeline.snapshot(2, 48000).bpm == 120
    assert h.controller.interaction_state()["capture_end_seconds"] == 108.0


def test_target_tempo_requests_remote_audio_again_without_rewriting_lyric_identity():
    h = Harness()
    h.capture()
    record = h.lyric(0)
    h.tick(40)
    h.controller.update_tempo(120.0)
    h.tick(46)
    updates = [item for item in h.stream.sent if item["type"] == "tempo_update"]
    assert updates[-1]["update"]["from_bar"] == 0
    assert updates[-1]["update"]["bpm"] == 120.0
    assert len([call for call in h.source.calls if call[2] == record.lyric_id]) >= 2


def test_live_mix_remixes_prepared_response_without_refetching_or_resynthesis():
    import numpy as np

    class StemPreparer(Preparer):
        calls = 0

        def prepare(self, record, manifest, source_wav, **kwargs):
            self.calls += 1
            result = super().prepare(record, manifest, source_wav, **kwargs)
            stem = PcmAudio(AudioFormat(), 192000, np.full((192000, 2), 0.2, dtype=np.float32).tobytes())
            return replace(result, audio=stem, vocal_stem=stem, drum_stem=stem, mix_config=RapMixConfig())

    source_preparer = StemPreparer()
    h = Harness(preparer=source_preparer)
    h.capture()
    h.lyric(0)
    h.tick(46)
    ack = h.controller.update_mix(RapMixConfig(vocal_gain=0, drum_gain=0, revision=2))
    h.tick(47)
    assert ack["effective_from_bar"] == 3
    assert h.enqueued[-1].text == "line line"
    assert not np.any(np.frombuffer(h.enqueued[-1].audio.data, dtype=np.float32))
    assert source_preparer.calls == 1
    assert len(h.source.calls) == 1


def test_template_edit_cannot_silently_replace_captured_flow():
    h = Harness()
    h.capture()
    flow = h.controller.captured_template
    with pytest.raises(ValueError, match="captured flow"):
        h.controller.update_plan(from_bar=3, template_id=default_scenario().segments[0].template_id)
    assert h.controller.captured_template == flow
    assert h.controller.interaction_state()["supported"] is True


def test_future_topic_edit_applies_only_to_eligible_capture_and_reset_restores_plan():
    h = Harness()
    h.controller.update_plan(from_bar=8, topic="rain")
    h.capture()
    assert h.stream.opens[0][1]["segments"][0]["topic"] == default_scenario().segment_for_bar(1).topic
    h.controller.request_stop(successor_bar=3)
    h.controller.reset()
    h.controller.update_plan(from_bar=0, topic="rain")
    h.capture()
    assert h.stream.opens[-1][1]["segments"][0]["topic"] == "rain"
    h.controller.request_stop(successor_bar=3)
    h.controller.reset()
    assert h.controller.plan_control_state["last_edit"] is None


@pytest.mark.parametrize("response_bars,lose_commit", [(2, False), (4, True), (8, True)])
def test_adapter_runs_progressively_against_real_bounded_lyric_session(response_bars, lose_commit):
    import threading
    import time

    from streammuse.application.rap.lyric_session import LyricStreamSession, RenderedBarAudio
    from streammuse.application.rap.lyric_writer import WrittenLyric
    from streammuse.presentation.rap_lyric_stream_server import plan_from_start

    second_line_gate = threading.Event()
    written_bars = []

    class Writer:
        def write(self, request):
            if request.bar == 1 and not lose_commit:
                assert second_line_gate.wait(3)
            written_bars.append(request.bar)
            text = " ".join(["first" if request.bar == 0 else "next"] * len(request.flow_template.slots))
            record = BarLyricRecord(
                request.session_id, request.bar, request.revision, request.topic, request.flow_template, text,
                align_exact(Analyzer().analyze(text), materialize_flow(request.flow_template, request.bar)), 0.5,
            )
            return WrittenLyric(record, (), 0.0, 0.0, 1)

    class Renderer:
        def render(self, record, *, tempo_bpm, previous=None):
            manifest = BarAudioManifest(record.lyric_id, record.bar, tempo_bpm, 96000, "b" * 64, (0.0, 1.0), (1.0, 1.0), {})
            return RenderedBarAudio(record.lyric_id, record.bar, record.revision, b"wave", "audio/wav", manifest, 0.0)

    class ServerStream(Stream):
        session = None
        dropped = False

        def open(self, session_id, start, on_message):
            super().open(session_id, start, on_message)

            class Listener:
                def lyric(self, record):
                    on_message({"type": "lyric", "record": record.to_payload()})

                def audio_ready(self, bar, lyric_id):
                    pass

                def superseded(self, bars, edit_seq):
                    on_message({"type": "superseded", "bars": bars})

                def error(self, message, *, bar=None):
                    on_message({"type": "error", "message": message, "bar": bar})

            self.session = LyricStreamSession(
                session_id=session_id, tempo_bpm=start["tempo_bpm"], plan=plan_from_start(start),
                writer=Writer(), renderer=Renderer(), listener=Listener(), start_bar=start["start_bar"],
                lyric_lead=start["lyric_lead"], audio_lead=start["audio_lead"],
                turn_id=start["turn_id"], end_bar_exclusive=start["end_bar_exclusive"],
            )
            self.session.start()

        def send(self, message):
            super().send(message)
            if self.session is not None and message["type"] == "committed":
                if lose_commit and not self.dropped:
                    self.dropped = True
                    raise ConnectionError("lost first delivery commit")
                self.session.committed(message["bar"], message["lyric_id"], message["text"])

        def close(self):
            if self.session is not None:
                self.session.close()

    stream = ServerStream()

    class SessionSource(Source):
        def fetch(self, session_id, bar, lyric_id, *, wait_ms, accept_older):
            result = stream.session.fetch_audio(bar, lyric_id, wait_ms / 1000, accept_older=accept_older)
            if result.status != "ready":
                return BarAudioFetch(result.status)
            return BarAudioFetch("ready", lyric_id, result.audio.manifest, result.audio.package)

    h = Harness(stream=stream, source=SessionSource(), response_bars=response_bars)

    def wait_ready(logical, physical):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if h.adapter.poll(logical, physical_bar=physical, allow_fallback=False, timing=h.controller.timeline.snapshot(physical, 48000)):
                return
            time.sleep(0.005)
        pytest.fail(f"logical response {logical} was not ready: {h.adapter.error}")

    try:
        h.capture()
        wait_ready(0, 3)
        h.tick(47)
        assert h.enqueued[-1].text == "first first"
        if lose_commit:
            assert stream.session.snapshot()["playhead"] == -1
            stream.handler({"type": "started", "capabilities": ["bounded_turns_v1", "tempo_updates_v1"]})
            for record in stream.session.latest_records():
                stream.handler({"type": "lyric", "record": record.to_payload()})
            assert stream.session.snapshot()["playhead"] == 0
            for logical in range(1, response_bars):
                physical = 3 + logical
                wait_ready(logical, physical)
                h.tick(physical * 16 - 1)
                assert stream.session.snapshot()["playhead"] == logical
            h.tick((3 + response_bars) * 16)
            assert h.controller.interaction_state()["delivered_response_bars"] == response_bars
            assert len([item for item in h.enqueued if item.text]) == response_bars
            assert written_bars == list(range(response_bars))
            return
        assert written_bars == [0]
        assert stream.session.snapshot()["playhead"] == 0
        h.tick(63)
        assert h.enqueued[-1].source == "flow_echo"
        assert stream.session.snapshot()["playhead"] == 0
        second_line_gate.set()
        wait_ready(1, 5)
        h.tick(79)
        assert h.enqueued[-1].text == "next next"
        assert stream.session.snapshot()["playhead"] == 1
        assert stream.session.snapshot()["end_bar_exclusive"] == 2
        assert written_bars == [0, 1]
    finally:
        second_line_gate.set()
        h.controller.close()


@pytest.mark.parametrize("limit", [1, 3])
def test_explicit_finite_bound_never_queues_or_captures_beyond_limit(limit):
    h = Harness(planning_bar_limit=limit)
    h.controller.start()
    for tick in range(limit * 16 + 32):
        h.tick(tick)
    assert h.controller.terminal_bar_limit == limit
    assert max(item.bar for item in h.enqueued) == limit - 1
    state = h.controller.arm_capture()
    assert state["last_error"] == "playback_limit_reached"
    assert state["phase"] == "DRUMS_IDLE"
    assert h.stream.opens == []


def test_finite_scenario_bound_and_partial_turn_terminate_without_index_error():
    scenario = replace(default_scenario(), loop=False)
    h = Harness(scenario=scenario, planning_bar_limit=30)
    assert h.controller.terminal_bar_limit == scenario.total_bars
    h.controller.start()
    h.tick(scenario.total_bars * 16)
    assert h.controller.arm_capture()["last_error"] == "playback_limit_reached"
    h = Harness(planning_bar_limit=4)
    h.capture()
    h.lyric(0)
    h.tick(47)
    h.tick(48)
    h.tick(63)
    assert h.controller.interaction_state()["last_error"] == "playback_limit_reached"
    assert h.controller.interaction_state()["delivered_response_bars"] == 1
    assert max(item.bar for item in h.enqueued) == 3


def test_past_topic_edit_during_capture_applies_at_acknowledged_next_capture():
    h = Harness()
    h.controller.start()
    h.tick(0)
    h.controller.arm_capture()
    h.tick(15)
    h.tick(16)
    h.controller.record_tap(server_time_seconds=104.0)
    h.controller.update_plan(from_bar=0, topic="changed early")
    assert h.controller.plan_control_state["last_edit"]["effective_from_bar"] == 2
    h.tick(32)
    assert h.stream.opens[-1][1]["segments"][0]["topic"] == default_scenario().segment_for_bar(1).topic


@pytest.mark.parametrize("started", [False, True])
def test_stop_between_guard_and_start_excludes_unheard_response_from_next_history(started):
    h = Harness()
    h.capture()
    h.lyric(0, word="heard" if started else "unheard")
    h.tick(47)
    assert h.controller.interaction_state()["delivered_response_bars"] == 0
    assert h.controller.interaction_state()["queued_response_bars"] == 1
    # Playback's successor is authoritative even if the tick callback lags.
    successor = 4 if started else 3
    h.controller.request_stop(successor_bar=successor)
    assert h.controller.interaction_state()["delivered_response_bars"] == int(started)
    h.controller.resume_audio(successor)
    h.clock.now = 100 + successor * 4
    h.controller.resume_after_stop()
    h.tick(successor * 16)
    h.controller.arm_capture()
    capture_bar = successor + 1
    h.tick(capture_bar * 16)
    h.controller.record_tap(server_time_seconds=h.clock.now)
    h.tick((capture_bar + 1) * 16)
    expected = [{"bar": -1, "text": "heard heard"}] if started else []
    assert h.stream.opens[-1][1]["history"] == expected


def test_reset_and_empty_capture_do_not_expose_previous_generation_error():
    h = Harness()
    h.capture()
    old_handler = h.stream.handler
    old_handler({"type": "error", "message": "no selectable candidate"})
    h.tick(152)
    assert h.controller.interaction_state()["generation_error"] == "no selectable candidate"
    h.controller.reset()
    old_handler({"type": "error", "message": "late old error"})
    h.capture(taps=False)
    state = h.controller.interaction_state()
    assert state["last_error"] == "empty_capture"
    assert state["generation_error"] is None


@pytest.mark.parametrize("slot_count", range(1, 17))
@pytest.mark.parametrize("fail_second", [False, True])
def test_one_to_sixteen_slot_rhythms_complete_or_fail_visibly_without_fabrication(slot_count, fail_second):
    h = Harness()
    h.controller.start()
    h.tick(0)
    h.controller.arm_capture()
    h.tick(16)
    ticks = sorted((index * 7) % 16 for index in range(slot_count))
    for tick in ticks:
        h.clock.now = 104 + tick / 4 + 0.01
        assert h.controller.record_tap(server_time_seconds=104 + tick / 4)["accepted"]
    h.tick(31)
    h.tick(32)
    assert h.controller.interaction_state()["captured_ticks"] == ticks
    first = h.lyric(0, word="first")
    h.tick(47)
    h.tick(48)
    if fail_second:
        h.stream.handler({"type": "error", "message": "no selectable candidate bar 1"})
        h.tick(183)
        assert h.controller.interaction_state()["last_error"] == "response_timeout"
        assert [item.text for item in h.enqueued if item.text] == [first.text]
        assert h.controller.interaction_state()["delivered_response_bars"] == 1
    else:
        second = h.lyric(1, word="next")
        h.tick(63)
        h.tick(80)
        assert [item.text for item in h.enqueued if item.text] == [first.text, second.text]
        assert h.controller.interaction_state()["delivered_response_bars"] == 2


@pytest.mark.parametrize("empty", [False, True])
def test_capture_finalized_event_records_calibrated_input_and_empty_failure_once(empty, tmp_path):
    import json

    from streammuse.application.rap.monitoring import RapEventPublisher
    from streammuse.infrastructure.rap.recorder import event_to_dict, read_events

    h = Harness(initial_epoch=7, capture_grace_seconds=0.1)
    h.controller.start()
    h.tick(0)
    h.controller.arm_capture()
    h.tick(16)
    if not empty:
        h.clock.now = 104.1
        assert h.controller.record_tap(server_time_seconds=104.02, client_time_ms=1000, uncertainty_ms=3, event_id="first")["accepted"]
        assert h.controller.record_tap(server_time_seconds=104.03, client_time_ms=1010, uncertainty_ms=3, event_id="collision")["accepted"]
        assert not h.controller.record_tap(server_time_seconds=104.03, event_id="collision")["accepted"]
        assert not h.controller.record_tap(server_time_seconds=104.03, repeat=True)["accepted"]
    h.tick(31)
    h.tick(32)
    if not empty:
        h.clock.now = 108.04
        assert h.controller.record_tap(server_time_seconds=107.95, event_id="last")["accepted"]
    h.tick(33)
    h.tick(34)
    events = [event for kind, event in h.publisher.events if kind.value == "capture_finalized"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["coordinator_epoch"] == 7
    assert payload["turn_id"] == h.controller.interaction_state()["turn_id"]
    assert payload["capture_window"] == {"bar": 1, "start_seconds": 104.0, "end_seconds": 108.0}
    assert payload["quantized_ticks"] == ([] if empty else [0, 0, 15])
    assert payload["slot_ticks"] == ([] if empty else [0, 15])
    assert payload["collision_count"] == (0 if empty else 1)
    assert payload["rejected_count"] == (0 if empty else 2)
    assert payload["rejected_reasons"] == ([] if empty else [{"reason": "duplicate_event", "count": 1}, {"reason": "key_repeat", "count": 1}])
    assert payload["error"] == ("empty_capture" if empty else None)
    assert (payload["flow"] is None) == empty
    if not empty:
        assert payload["raw_taps"][0]["received_time_seconds"] == 104.1
        assert payload["raw_taps"][-1]["server_time_seconds"] == 107.95
        assert payload["raw_taps"][-1]["received_time_seconds"] == 108.04
    event = RapEventPublisher("capture-evidence").emit(RapEventType.CAPTURE_FINALIZED, bar=1, tick=33, payload=payload)
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(event_to_dict(event), allow_nan=False) + "\n")
    restored = read_events(path)[0]
    assert restored.event_type == RapEventType.CAPTURE_FINALIZED
    assert restored.payload == payload


def test_tempo_ack_budget_tolerates_measured_tunnel_latency():
    class TunnelStream(Stream):
        budgets = []

        def request_tempo(self, payload, *, timeout):
            self.budgets.append(timeout)
            if timeout < 5.25:
                raise TimeoutError("tunnel round trip exceeds timeout")
            return super().request_tempo(payload)

    stream = TunnelStream()
    h = Harness(stream=stream)
    h.capture()
    h.lyric(0)
    h.controller.update_tempo(120.0)
    h.tick(46)
    assert stream.budgets == [10.0]
    assert h.adapter.fatal_error is None


def test_uncertain_tempo_ack_terminates_turn_and_never_plays_old_timing_audio():
    class TimeoutStream(Stream):
        def request_tempo(self, payload, **kwargs):
            raise TimeoutError("uncertain remote acknowledgement")

    h = Harness(stream=TimeoutStream())
    h.capture()
    h.lyric(0)
    h.controller.update_tempo(120.0)
    h.tick(47)
    assert "uncertain remote acknowledgement" in h.controller.interaction_state()["last_error"]
    assert h.enqueued[-1].source == "drums_idle"
    assert h.enqueued[-1].audio.frame_count == 96000
    assert not any(item.text for item in h.enqueued)
