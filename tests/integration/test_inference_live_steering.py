"""Steering through the long-session adapter against a real socket server (no models).

The second half drives the real InferenceService in real time, so the adapter's
playhead reports are derived from the service's actual request pattern.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from streammuse.application.rap.inference import InferenceService
from streammuse.domain.rap.inference import InferenceRequest

from tests.integration.lyric_stream_live import LiveClient, LiveLyricServer
from tests.unit.application.rap.test_lyric_session import _flow

TEMPLATES = {"four": _flow("four"), "four-b": _flow("four-b")}


def steering_request(request_id: str, start_bar: int = 0, version: int = 1, **changes) -> InferenceRequest:
    values = dict(session_id="live", epoch=1, request_id=request_id, input_version=version,
                  start_tick=start_bar * 16, duration_ticks=32, bpm=120, ticks_per_beat=4, beats_per_bar=4,
                  topic="deep sea", template_id="four", flow_ticks=(), history=(),
                  config={"mode": "steering", "seed": 5})
    return InferenceRequest(**{**values, **changes})


@pytest.fixture
def server(tmp_path):
    value = LiveLyricServer(tmp_path)
    yield value
    value.close()


@pytest.fixture
def client(server):
    value = LiveClient(server.url, templates=dict(TEMPLATES), steering_adapter="session")
    yield value
    value.close()


def sockets(server):
    return [item for item in server.requests() if item.kind == "websocket"]


def spans_of(client, request_id):
    return sorted((s for s in client.spans if s.request_id == request_id), key=lambda s: s.start_tick)


def test_requests_of_one_playback_share_one_server_session(server, client):
    client.submit(steering_request("r1"))
    client.finished("r1")
    client.submit(steering_request("r2", start_bar=2))
    client.finished("r2")
    client.submit(steering_request("r3", start_bar=4))
    client.finished("r3")

    assert len(sockets(server)) == 1
    remote = sockets(server)[0].path.rsplit("/", 1)[1]
    assert remote.startswith("inference-steer-")
    assert [s.start_tick for s in sorted(client.spans, key=lambda s: s.start_tick)] == [0, 16, 32, 48, 64, 80]
    # r3 covers bars [4, 6): it arrived "while bar 2 plays", so bars 0..2 are reported committed.
    assert server.service.session(remote).snapshot()["playhead"] == 2
    # Inside one session every bar but the very first continues the bar before it.
    used = [e["continuation"]["used"] for e in sorted(
        (e for e in client.events if e["type"] == "stage_timings"),
        key=lambda e: (int(e["request_id"][1:]), e["bar"]))]
    assert used == [False, True, True, True, True, True]


def test_edit_is_one_plan_update_and_new_bars_carry_the_new_topic_and_flow(server, client):
    client.submit(steering_request("r1"))
    client.finished("r1")
    client.submit(steering_request("r2", start_bar=2))
    client.finished("r2")
    before = server.mark()

    # What the service does on an edit made while bar 1 plays.
    client.backend.cancel("r1")
    client.backend.cancel("r2")
    client.submit(steering_request("r3", start_bar=2, version=2, topic="night market", template_id="four-b"))
    client.finished("r3")

    assert not [item for item in server.requests(since=before) if item.kind == "websocket"]
    assert not [item for item in server.requests(since=before) if item.path == "/v1/rap/capabilities"]
    old, new = spans_of(client, "r2"), spans_of(client, "r3")
    assert [s.start_tick for s in new] == [32, 48] and all(s.input_version == 2 for s in new)
    assert all(s.text.startswith("nightmarket ") for s in new) and all(s.revision >= 1 for s in new)
    assert all(s.text.startswith("deepsea ") for s in old)
    remote = sockets(server)[0].path.rsplit("/", 1)[1]
    snapshot = server.service.session(remote).snapshot()
    assert snapshot["edit_seq"] == 1 and snapshot["playhead"] == 1
    update = next(e for e in client.events if e["type"] == "plan_update")
    assert update["from_bar"] == 2 and update["request_id"] == "r3"


def test_stop_ends_the_server_session_and_the_next_playback_continues_its_voice(server, client):
    client.submit(steering_request("r1"))
    client.finished("r1")
    last_line = spans_of(client, "r1")[-1].text
    remote = sockets(server)[0].path.rsplit("/", 1)[1]
    client.backend.cancel("r1")  # Stop
    deadline = time.monotonic() + 5
    while server.service.session(remote) is not None:
        assert time.monotonic() < deadline, "the server session outlived Stop"
        time.sleep(0.02)

    client.submit(steering_request("r2", epoch=2, history=(last_line,)))
    client.finished("r2")
    assert len(sockets(server)) == 2
    first_bar = next(e for e in client.events if e["type"] == "stage_timings"
                     and e["request_id"] == "r2" and e["bar"] == 0)
    assert first_bar["continuation"]["used"] is True and first_bar["continuation"]["handoff"] is True


# -- the real service, in real time ---------------------------------------------------------------


class Playback:
    """Calls the service's audio callback at the rate a device would."""

    RATE = 100  # frames per second; one bar is one second at 240 BPM

    def __init__(self, service, server):
        self.service, self.server = service, server
        self.frames = 0
        self.started = None
        self.playhead_checks = []

    def run_until(self, predicate, timeout=20.0):
        if self.started is None:
            self.started = time.monotonic()
        deadline = time.monotonic() + timeout
        while True:
            due = int((time.monotonic() - self.started) * self.RATE) - self.frames
            if due > 0:
                self.service.audio_callback(due)
                self.frames += due
            snapshot = self.service.snapshot()
            self._check_reported_playhead(snapshot)
            if predicate(snapshot):
                return snapshot
            assert time.monotonic() < deadline, (snapshot["phase"], snapshot["current"]["tick"], snapshot["error"])
            time.sleep(0.004)

    def _check_reported_playhead(self, snapshot):
        if snapshot["phase"] != "MODEL":
            return
        for session in list(self.server.service._sessions.values()):
            reported = session.session.snapshot()["playhead"]
            self.playhead_checks.append((reported, math.floor(snapshot["current"]["tick"] / 16)))


@pytest.mark.parametrize("edit_window", [(2, 6), (10, 13)], ids=["first-half", "second-half"])
def test_real_service_steering_over_one_session_has_no_gap_and_edits_land_within_a_bar(tmp_path, edit_window):
    server = LiveLyricServer(tmp_path)
    client = LiveClient(server.url, templates=dict(TEMPLATES), steering_adapter="session")
    service = InferenceService(session_id="live", backend=client.backend, sample_rate=Playback.RATE,
                               channels=1, drums=np.zeros((100, 1), dtype=np.float32))
    playback = Playback(service, server)
    try:
        service.configure({"mode": "steering", "bpm": 240.0, "topic": "deep sea", "template_id": "four"})
        service.start()
        playback.run_until(lambda s: s["phase"] == "MODEL")
        low, high = edit_window
        playback.run_until(lambda s: s["current"]["tick"] >= 32 and low <= s["current"]["tick"] % 16 <= high)
        edited_at = service.snapshot()["current"]["tick"]
        accepted = service.steer({"topic": "night market", "template_id": "four-b"})
        target = accepted["current"]["requested_version"]
        changed = playback.run_until(lambda s: s["current"]["playing_version"] == target)
        final = playback.run_until(lambda s: s["current"]["tick"] >= changed["current"]["tick"] + 40)

        metrics = final["metrics"]
        assert final["error"] is None and metrics["error_count"] == 0
        assert metrics["missing_duration_ms"] == 0, "playback had a gap"
        # With no network delay the first rewritten bar is ready before the next bar starts.
        assert math.floor(changed["current"]["tick"] / 16) == math.floor(edited_at / 16) + 1
        assert metrics["steering_playback_latency_ms"] < 1000
        # "Stale" is the old version still sounding: here only the rest of the bar that was
        # already playing when the edit was made, never a whole bar of the old plan.
        rest_of_bar_ms = (16 - edited_at % 16) / 16 * 1000
        assert metrics["stale_duration_ms"] == pytest.approx(rest_of_bar_ms, abs=60)
        assert changed["current"]["text"].startswith("nightmarket ")

        # One server session for the whole playback, edit included.
        assert len(sockets(server)) == 1
        assert final["capabilities"]["steering_adapter"] == "session"
        # The reported playhead is never ahead of the bar actually playing (that would delay an edit),
        # and never so far behind that the server's audio window misses a requested bar.
        assert playback.playhead_checks and all(reported <= playing for reported, playing in playback.playhead_checks)
        assert max(playing - reported for reported, playing in playback.playhead_checks) <= 3

        service.stop()
        playback.run_until(lambda s: s["state"] == "stopped")
        deadline = time.monotonic() + 5
        while server.service._sessions:
            assert time.monotonic() < deadline, "the server session outlived Stop"
            time.sleep(0.02)
    finally:
        service.close()
        server.close()


def test_real_service_restart_reuses_nothing_stale_and_hands_the_voice_over(tmp_path):
    server = LiveLyricServer(tmp_path)
    client = LiveClient(server.url, templates=dict(TEMPLATES), steering_adapter="session")
    service = InferenceService(session_id="live", backend=client.backend, sample_rate=Playback.RATE,
                               channels=1, drums=np.zeros((100, 1), dtype=np.float32))
    try:
        service.configure({"mode": "steering", "bpm": 240.0, "topic": "deep sea", "template_id": "four"})
        for run in (1, 2):
            playback = Playback(service, server)
            service.start()
            playback.run_until(lambda s: s["phase"] == "MODEL" and s["current"]["tick"] >= 40)
            service.stop()
            playback.run_until(lambda s: s["state"] == "stopped")
            assert service.snapshot()["metrics"]["missing_duration_ms"] == 0
            assert len(sockets(server)) == run
        # The service receives the adapter's events; they are part of the exported trace.
        opened = [e["payload"] for e in service.export_trace()["events"]
                  if e["type"] == "backend" and e["payload"]["type"] == "steering_session"]
        assert len(opened) == 2 and opened[0]["continue_from"] is None and opened[1]["continue_from"]
        assert opened[0]["remote_session_id"] != opened[1]["remote_session_id"]
    finally:
        service.close()
        server.close()
