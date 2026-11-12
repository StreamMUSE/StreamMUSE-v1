"""The inference backend's real transports against a real socket server (no models).

These cover what unit tests with fake transports cannot: that the early
WebSocket handshake really carries the session, and that turns really reuse
TCP connections.
"""

from __future__ import annotations

import time

import pytest

from streammuse.domain.rap.inference import InferenceRequest

from tests.integration.lyric_stream_live import DelayProxy, LiveClient, LiveLyricServer


def cycle_request(request_id: str, **changes) -> InferenceRequest:
    values = dict(session_id="live", epoch=1, request_id=request_id, input_version=1,
                  start_tick=0, duration_ticks=32, bpm=120, ticks_per_beat=4, beats_per_bar=4,
                  topic="deep sea", template_id="unused",
                  flow_ticks=(2, 6, 10, 14, 18, 22, 26, 30), history=(),
                  config={"mode": "cycle", "seed": 5})
    return InferenceRequest(**{**values, **changes})


@pytest.fixture
def server(tmp_path):
    value = LiveLyricServer(tmp_path)
    yield value
    value.close()


@pytest.fixture
def client(server):
    value = LiveClient(server.url)
    yield value
    value.close()


def audio_paths(seen):
    return [item for item in seen if item.path.endswith("/audio")]


def test_prepared_turn_opens_no_new_connection_after_the_request_is_released(server, client):
    client.prepare()
    prepared = server.requests()
    sockets = [item for item in prepared if item.kind == "websocket"]
    assert len(sockets) == 1
    remote_session = sockets[0].path.rsplit("/", 1)[1]
    # Handshake only: the server holds the socket and has not created a session.
    assert server.service.session(remote_session) is None
    warm_peers = {item.peer for item in prepared if item.kind == "http"}
    assert len(warm_peers) >= 2

    mark = server.mark()
    client.submit(cycle_request("turn-1"))
    client.finished("turn-1")

    during = server.requests(since=mark)
    assert not [item for item in during if item.kind == "websocket"], "the turn made its own handshake"
    assert not [item for item in during if item.path == "/v1/rap/capabilities"], "capabilities were asked again"
    fetches = audio_paths(during)
    assert {item.path.split("/")[4] for item in fetches} == {remote_session}
    assert {item.peer for item in fetches} <= warm_peers, "a bar fetch opened a new connection"
    assert [span.start_tick for span in sorted(client.spans, key=lambda s: s.start_tick)] == [0, 16]
    cleanup = next(e for e in client.events if e["type"] == "session_cleanup")
    assert cleanup["remote_session_id"] == remote_session and cleanup["cleanup_errors"] == []


def test_later_turns_reuse_the_audio_connections_and_handshake_before_the_request(server, client):
    client.prepare()
    client.submit(cycle_request("turn-1"))
    client.finished("turn-1")
    peers_after_first = {item.peer for item in server.requests() if item.kind == "http"}

    for index in (2, 3):
        mark = server.mark()
        client.prepare()  # what the service does when the next USER phase begins
        before_release = server.requests(since=mark)
        assert len([item for item in before_release if item.kind == "websocket"]) == 1
        release = server.mark()
        client.submit(cycle_request(f"turn-{index}", history=("an earlier line",)))
        client.finished(f"turn-{index}")
        during = server.requests(since=release)
        assert not [item for item in during if item.kind == "websocket"]
        assert {item.peer for item in during} <= peers_after_first

    assert len(client.spans) == 6
    assert len(client.sources) == 1
    # One capability lookup by the backend; the other GETs on that path are connection warm-ups.
    assert not [e for e in client.events if e["type"] == "error"]


def test_unprepared_turn_still_works_and_shares_the_source_with_the_next(server, client):
    client.submit(cycle_request("turn-1"))
    client.finished("turn-1")
    first = server.requests()
    assert len([item for item in first if item.kind == "websocket"]) == 1
    assert len([item for item in first if item.path == "/v1/rap/capabilities"]) == 1
    first_fetch_peers = {item.peer for item in audio_paths(first)}

    mark = server.mark()
    client.submit(cycle_request("turn-2"))
    client.finished("turn-2")
    second = server.requests(since=mark)
    assert not [item for item in second if item.path == "/v1/rap/capabilities"]
    assert {item.peer for item in audio_paths(second)} <= first_fetch_peers


def test_early_handshake_that_died_is_replaced_by_a_normal_connection(server, client):
    client.prepare()
    early = next(item for item in server.requests() if item.kind == "websocket")
    spare_stream, spare_session, _ = client.backend._spare
    spare_stream._socket.close()  # the tunnel dropped it while the user was playing

    mark = server.mark()
    client.submit(cycle_request("turn-1"))
    client.finished("turn-1")
    during = server.requests(since=mark)
    sockets = [item for item in during if item.kind == "websocket"]
    # The connection notices on sending ``start`` and handshakes again for the same session.
    assert len(sockets) == 1 and sockets[0].peer != early.peer
    assert sockets[0].path.endswith(spare_session)
    assert len(client.spans) == 2
    assert not [e for e in client.events if e["type"] == "error"]


def test_stop_between_handshake_and_start_leaves_no_session_on_the_server(server, client):
    client.prepare()
    _, spare_session, _ = client.backend._spare
    client.backend.close()
    assert server.service.session(spare_session) is None
    assert not server.service._sessions


def test_short_server_keep_alive_only_costs_the_reuse_not_the_turn(tmp_path):
    server = LiveLyricServer(tmp_path, keep_alive_seconds=1)
    client = LiveClient(server.url)
    try:
        client.prepare()
        time.sleep(1.6)  # an old server has dropped the idle HTTP connections by now
        client.submit(cycle_request("turn-1"))
        client.finished("turn-1")
        assert len(client.spans) == 2
    finally:
        client.close()
        server.close()


def first_bar_latency(client, request) -> float:
    """Request release to the first playable bar: what WAIT measures on the page."""
    count = len(client.spans)
    released = time.monotonic()
    client.submit(request)
    client.wait(lambda: len(client.spans) > count)
    elapsed = time.monotonic() - released
    client.finished(request.request_id)
    return elapsed


def test_over_a_slow_link_a_prepared_turn_saves_the_connection_round_trips(server):
    one_way = 0.06
    proxy = DelayProxy(int(server.url.rsplit(":", 1)[1]), one_way_seconds=one_way)
    client = LiveClient(proxy.url)
    try:
        # No prepare: what every turn paid before (capabilities, handshake, new audio connections).
        cold = first_bar_latency(client, cycle_request("turn-1"))
        connections = proxy.connections
        client.prepare()
        prepared_connections = proxy.connections
        warm = first_bar_latency(client, cycle_request("turn-2"))
        assert proxy.connections == prepared_connections, "the prepared turn opened a connection"
        assert prepared_connections - connections == 1, "prepare should add only the next handshake"
        # Saved: capability lookup (2 round trips), handshake (2), one audio connection setup (1).
        # Require three of the five so scheduling noise cannot fail the test.
        assert cold - warm > 3 * 2 * one_way, (cold, warm)
        # What is left is two round trips: start -> lyric, fetch -> audio.
        assert warm < cold
    finally:
        client.close()
        proxy.close()
