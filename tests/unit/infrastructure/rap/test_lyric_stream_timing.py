import threading
import json
import queue
import hashlib
import socket

import httpx
import pytest

from streammuse.infrastructure.rap.lyric_stream_client import HttpBarAudioSource, LyricStreamConnection


def test_legacy_connection_rejects_tempo_before_sending():
    connection = LyricStreamConnection("http://test")
    with pytest.raises(ValueError, match="capability"):
        connection.request_tempo({"edit_seq": 1, "from_bar": 2, "bpm": 120})


def test_tempo_ack_is_delivered_without_waiting_for_blocked_lyric_callback():
    connection = LyricStreamConnection("http://test")
    connection._accept_control_message({"type": "started", "capabilities": ["tempo_updates_v1"]})
    sent = threading.Event()
    connection.send = lambda message: sent.set()
    results = []
    thread = threading.Thread(target=lambda: results.append(connection.request_tempo(
        {"edit_seq": 1, "from_bar": 2, "bpm": 120}, timeout=1)))
    thread.start()
    assert sent.wait(1)
    ack = {"type": "tempo_updated", "edit_seq": 1, "requested_from_bar": 2,
           "effective_from_bar": 2, "bpm": 120, "revision": 1}
    assert connection._accept_control_message(ack)
    thread.join(1)
    assert results == [ack]


def test_tempo_timeout_is_explicit_and_does_not_fabricate_ack():
    connection = LyricStreamConnection("http://test")
    connection._accept_control_message({"type": "started", "capabilities": ["tempo_updates_v1"]})
    connection.send = lambda message: None
    with pytest.raises(TimeoutError):
        connection.request_tempo({"edit_seq": 1, "from_bar": 2, "bpm": 120}, timeout=0.001)


@pytest.mark.parametrize('websocket_closed', [False, True])
def test_tempo_send_transport_failure_is_reported_as_uncertain_connection_error(websocket_closed):
    from websockets.exceptions import ConnectionClosedError

    class BrokenSocket:
        def send(self, text):
            if websocket_closed:
                raise ConnectionClosedError(None, None)
            raise OSError('transport failed after partial write')

    connection = LyricStreamConnection('http://test')
    connection._socket = BrokenSocket()
    connection._accept_control_message({'type': 'started', 'capabilities': ['tempo_updates_v1']})
    with pytest.raises(ConnectionError, match='tempo'):
        connection.request_tempo({'edit_seq': 1, 'from_bar': 2, 'bpm': 120})


def test_receive_loop_does_not_deadlock_ack_behind_blocked_callback():
    class Socket:
        def __init__(self):
            self.messages = queue.Queue()
            self.sent = threading.Event()

        def recv(self):
            result = self.messages.get(timeout=2)
            if result is None:
                raise ConnectionError()
            return json.dumps(result)

        def send(self, text):
            self.sent.set()
            self.messages.put(self.ack)

        def close(self):
            self.messages.put(None)

    connection, socket = LyricStreamConnection("http://test"), Socket()
    entered, release = threading.Event(), threading.Event()

    def handler(message):
        if message['type'] == 'lyric':
            entered.set()
            release.wait(2)

    connection._connect = lambda: setattr(connection, '_socket', socket)
    connection.open('s', {}, handler)
    socket.messages.put({'type': 'started', 'capabilities': ['tempo_updates_v1']})
    socket.messages.put({'type': 'lyric'})
    assert entered.wait(1)
    ack = {'type': 'tempo_updated', 'edit_seq': 1, 'effective_from_bar': 2, 'bpm': 120, 'revision': 1}
    socket.ack = ack
    try:
        assert connection.request_tempo({'edit_seq': 1, 'from_bar': 2, 'bpm': 120}, timeout=.5) == ack
    finally:
        release.set()
        connection.close()


@pytest.mark.parametrize('failure', [httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError])
def test_audio_get_retries_transport_failure_once_with_remaining_budget(failure):
    now, requests = [0.0], []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            now[0] = 2.0
            raise failure('Server disconnected without sending response')
        return httpx.Response(202, headers={'x-streammuse-fetch-status': 'pending'})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = HttpBarAudioSource('http://test', client=client, clock=lambda: now[0])
        result = source.fetch('s', 2, 'a' * 64, wait_ms=9000, accept_older=False)
    assert result.status == 'pending'
    assert result.request_ms == 2000
    assert len(requests) == 2
    assert requests[1].url.params['wait_ms'] == '7000'
    assert requests[1].url.params['lyric_id'] == 'a' * 64
    assert requests[1].extensions['timeout']['read'] == 17.0


@pytest.mark.parametrize('elapsed,expected_attempts', [(0.0, 2), (1.1, 2), (11.1, 1)])
def test_audio_transport_retry_is_bounded_by_attempts_and_bar_budget(elapsed, expected_attempts):
    now, requests = [0.0], []

    def handler(request):
        requests.append(request)
        now[0] = elapsed
        raise httpx.RemoteProtocolError('connection closed')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = HttpBarAudioSource('http://test', client=client, clock=lambda: now[0])
        with pytest.raises(httpx.RemoteProtocolError):
            source.fetch('s', 2, 'a' * 64, wait_ms=1000, accept_older=True)
    assert len(requests) == expected_attempts


def test_audio_retry_preserves_transport_allowance_when_rtt_exceeds_server_wait():
    now, requests = [0.0], []

    def handler(request):
        requests.append(request)
        now[0] += 1.3
        if len(requests) == 1:
            raise httpx.RemoteProtocolError('connection closed after request transit')
        assert request.url.params['wait_ms'] == '0'
        assert request.extensions['timeout']['read'] == pytest.approx(9.7)
        assert request.extensions['timeout']['read'] > 1.3
        return httpx.Response(202, headers={'x-streammuse-fetch-status': 'pending'})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = HttpBarAudioSource('http://test', client=client, clock=lambda: now[0])
        result = source.fetch('s', 2, 'a' * 64, wait_ms=1000, accept_older=True)
    assert result.status == 'pending'
    assert result.request_ms == pytest.approx(2600)
    assert len(requests) == 2


def test_audio_invalid_package_is_not_retried():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, content=b'not a zip package')

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source = HttpBarAudioSource('http://test', client=client)
        with pytest.raises(Exception, match='zip'):
            source.fetch('s', 2, 'a' * 64, wait_ms=1000, accept_older=True)
    assert len(requests) == 1


def test_tempo_request_replays_identical_update_once_when_first_ack_is_lost():
    connection = LyricStreamConnection('http://test')
    connection._accept_control_message({'type': 'started', 'capabilities': ['tempo_updates_v1']})
    messages = []
    ack = {'type': 'tempo_updated', 'edit_seq': 2, 'effective_from_bar': 74,
           'bpm': 90, 'revision': 2}

    def send(message):
        messages.append(message)
        if len(messages) == 2:
            connection._accept_control_message(ack)

    connection.send = send
    assert connection.request_tempo({'edit_seq': 2, 'from_bar': 74, 'bpm': 90}, timeout=.05) == ack
    assert len(messages) == 2
    assert messages[0] == messages[1]


def test_audio_get_recovers_real_http_disconnect_then_decodes_valid_audio():
    from tests.unit.presentation.test_rap_lyric_stream_server import _wav
    from tests.unit.domain.rap.test_lyric_stream import _manifest
    from streammuse.infrastructure.rap.bar_package import PcmBarPackageEncoder

    wav = _wav(66_000)
    manifest = _manifest(source_sha256=hashlib.sha256(wav).hexdigest())
    package = PcmBarPackageEncoder().encode(manifest, wav)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(2)
    listener.settimeout(3)
    requests, failures = [], []

    def serve():
        try:
            for attempt in range(2):
                connection, _ = listener.accept()
                with connection:
                    connection.settimeout(2)
                    request = b''
                    while b'\r\n\r\n' not in request:
                        request += connection.recv(4096)
                    requests.append(request.split(b'\r\n', 1)[0])
                    if attempt:
                        connection.sendall(
                            f'HTTP/1.1 200 OK\r\nContent-Length: {len(package)}\r\nConnection: close\r\n\r\n'.encode()
                            + package
                        )
                    # The first request is closed without any HTTP response,
                    # reproducing the live RemoteProtocolError exactly.
        except Exception as error:
            failures.append(error)

    thread = threading.Thread(target=serve)
    thread.start()
    source = HttpBarAudioSource(f'http://127.0.0.1:{listener.getsockname()[1]}')
    try:
        result = source.fetch('s', 5, manifest.lyric_id, wait_ms=2000, accept_older=False)
        assert result.manifest == manifest
        assert result.source_wav == wav
    finally:
        source.close()
        thread.join(4)
        listener.close()
    assert not thread.is_alive() and not failures
    assert len(requests) == 2


class _EarlySocket:
    """A socket whose handshake is done; records what is sent on it."""

    def __init__(self, broken=False):
        self.sent, self.broken, self.closed = [], broken, threading.Event()

    def send(self, raw):
        if self.broken:
            raise OSError("connection reset")
        self.sent.append(json.loads(raw))

    def recv(self):
        self.closed.wait(5)
        raise OSError("closed")

    def close(self):
        self.closed.set()


def _connection_with_handshakes(*sockets):
    connection = LyricStreamConnection("http://test")
    made = []
    pending = list(sockets)

    def handshake():
        made.append(connection._session_id)
        return pending.pop(0)

    connection._handshake = handshake
    return connection, made


def test_early_handshake_carries_the_start_without_connecting_again():
    socket_ = _EarlySocket()
    connection, made = _connection_with_handshakes(socket_)
    connection.connect("session-a")
    assert made == ["session-a"] and socket_.sent == []
    try:
        connection.open("session-a", {"type": "start", "seed": 1}, lambda message: None)
        assert made == ["session-a"]
        assert socket_.sent == [{"type": "start", "seed": 1}]
        connection.send({"type": "stop"})
        assert socket_.sent[-1] == {"type": "stop"}
    finally:
        connection.close()
    assert socket_.closed.is_set()


def test_early_handshake_for_another_session_is_closed_and_not_used():
    early, fresh = _EarlySocket(), _EarlySocket()
    connection, made = _connection_with_handshakes(early, fresh)
    connection.connect("session-a")
    try:
        connection.open("session-b", {"type": "start"}, lambda message: None)
        assert made == ["session-a", "session-b"]
        assert early.closed.is_set() and early.sent == []
        assert fresh.sent == [{"type": "start"}]
    finally:
        connection.close()


def test_stale_early_handshake_reconnects_and_sends_the_same_start():
    early, fresh = _EarlySocket(broken=True), _EarlySocket()
    connection, made = _connection_with_handshakes(early, fresh)
    # A start provider is only for reconnects of a running session, not for this fallback.
    connection._start_provider = lambda: {"type": "start", "from": "provider"}
    connection.connect("session-a")
    try:
        connection.open("session-a", {"type": "start", "seed": 4}, lambda message: None)
        assert made == ["session-a", "session-a"]
        assert early.closed.is_set()
        assert fresh.sent == [{"type": "start", "seed": 4}]
    finally:
        connection.close()


def test_closing_an_unused_early_handshake_closes_its_socket():
    socket_ = _EarlySocket()
    connection, _ = _connection_with_handshakes(socket_)
    connection.connect("session-a")
    connection.close()
    assert socket_.closed.is_set()
    with pytest.raises(ConnectionError):
        connection.send({"type": "stop"})


def test_open_without_early_handshake_connects_and_starts_as_before():
    socket_ = _EarlySocket()
    connection, made = _connection_with_handshakes(socket_)
    try:
        connection.open("session-a", {"type": "start"}, lambda message: None)
        assert made == ["session-a"] and socket_.sent == [{"type": "start"}]
    finally:
        connection.close()


def test_audio_source_keeps_the_httpx_idle_expiry_unless_asked():
    default = HttpBarAudioSource("http://test")
    longer = HttpBarAudioSource("http://test", keepalive_expiry_seconds=60.0)
    try:
        assert default._client._transport._pool._keepalive_expiry == 5.0
        assert longer._client._transport._pool._keepalive_expiry == 60.0
    finally:
        default.close()
        longer.close()


def test_warm_issues_concurrent_cheap_requests_and_tolerates_failures():
    barrier = threading.Barrier(2, timeout=3)
    paths = []

    def handler(request):
        paths.append(request.url.path)
        barrier.wait()  # both requests are in flight at once: each needs its own connection
        return httpx.Response(200, json={"capabilities": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert HttpBarAudioSource("http://test", client=client).warm(2) == 2
    assert paths == ["/v1/rap/capabilities"] * 2

    def refuse(request):
        raise httpx.ConnectError("refused")

    with httpx.Client(transport=httpx.MockTransport(refuse)) as client:
        assert HttpBarAudioSource("http://test", client=client).warm(2) == 0
