import json

import pytest

from streammuse.infrastructure.rap.lyric_stream_client import LyricStreamConnection


class Socket:
    def __init__(self):
        self.sent = []
    def send(self, message):
        self.sent.append(json.loads(message))


def test_session_bound_send_rejects_new_socket_but_ordinary_send_is_unchanged():
    connection = LyricStreamConnection("http://test")
    socket = Socket()
    connection._session_id, connection._socket = "new-session", socket
    with pytest.raises(ConnectionError, match="session"):
        connection.send_for_session("old-session", {"type": "update"})
    assert socket.sent == []
    connection.send_for_session("new-session", {"type": "update"})
    connection.send({"type": "committed"})
    assert socket.sent == [{"type": "update"}, {"type": "committed"}]


def test_socket_replacement_after_capture_never_redirects_old_edit():
    connection = LyricStreamConnection("http://test")
    newer = Socket()
    class Old(Socket):
        def send(self, message):
            with connection._lock:
                connection._session_id, connection._socket = "new", newer
            super().send(message)
    old = Old()
    connection._session_id, connection._socket = "old", old
    connection.send_for_session("old", {"type": "update"})
    assert old.sent == [{"type": "update"}] and newer.sent == []
