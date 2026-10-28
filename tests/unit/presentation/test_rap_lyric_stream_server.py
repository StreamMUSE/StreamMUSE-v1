from __future__ import annotations

import hashlib
import io
import wave
import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from streammuse.application.rap.bar_audio import MossBarAudioRenderer
from streammuse.application.rap.chunk_orchestration import PhraseSourceResult
from streammuse.application.rap.lyric_writer import LyricWriterPolicy, SequentialLyricWriter
from streammuse.domain.rap import ScoreWeights
from streammuse.domain.rap.lyric_stream import LYRIC_STREAM_SCHEMA_VERSION, BarLyricRecord, PlanEdit
from streammuse.domain.rap.remote_chunk import _flow_payload
from streammuse.infrastructure.rap.bar_package import (
    BAR_PACKAGE_MEDIA_TYPE,
    PcmBarPackageEncoder,
    decode_bar_package,
)
from streammuse.presentation.rap_lyric_stream_server import LyricStreamService, mount_lyric_stream_routes

from tests.unit.application.rap.test_lyric_session import _Analyzer, _flow, _Generator


def _wav(frames: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(b"\x10\x00" * frames)
    return buffer.getvalue()


class _SourceRenderer:
    def __init__(self) -> None:
        self.requests = []
        self.prefixes = []

    def render_source(self, request, workspace, *, prefix=None):
        self.requests.append(request)
        self.prefixes.append(prefix)
        source = _wav(60_000)
        return PhraseSourceResult(
            source_wav=source,
            alignment_diagnostics={
                "fallback_counts": {"word": 0},
                "source_onsets": [0.1 * (index + 1) for index in range(len(request.syllables))],
                "onset_confidence": [0.9] * len(request.syllables),
            },
            audio_diagnostics={"sample_rate_hz": 24_000, "frame_count": 60_000, "duration_seconds": 2.5, "peak": 0.1},
            model_tool_versions={"moss": "test", "aligner": "test"},
            warnings=(),
            stage_timings_ms={"moss": 500.0, "aligner": 20.0},
            monitoring_summary={},
        )


def _client(tmp_path):
    source = _SourceRenderer()

    def writer_factory():
        writer = SequentialLyricWriter(
            _Generator(),
            _Analyzer(),
            ScoreWeights(),
            LyricWriterPolicy(candidates=2, rescue_candidates=0, minimum_valid=1, rhyme_scheme="none"),
        )
        return writer, lambda: None

    service = LyricStreamService(
        writer_factory=writer_factory,
        renderer=MossBarAudioRenderer(source, PcmBarPackageEncoder(), workspace_root=tmp_path),
    )
    app = FastAPI()
    mount_lyric_stream_routes(app, service)
    return TestClient(app), service, source


def _start(session_id: str = "s1", **overrides) -> dict[str, object]:
    message = {
        "type": "start",
        "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
        "tempo_bpm": 90.0,
        "start_bar": 0,
        "seed": 3,
        "loop": True,
        "segments": [{"start_bar": 0, "bars": 4, "topic": "sea", "flow_template": _flow_payload(_flow())}],
        "edits": [],
        "history": [],
    }
    message.update(overrides)
    return message


def _receive_until(websocket, predicate, limit=50):
    seen = []
    for _ in range(limit):
        message = websocket.receive_json()
        seen.append(message)
        if predicate(message, seen):
            return seen
    raise AssertionError(f"predicate never matched: {seen}")


def test_capabilities_and_tempo_update_round_trip(tmp_path, caplog):
    caplog.set_level('INFO', logger='uvicorn.error')
    client, service, _ = _client(tmp_path)
    try:
        capabilities = client.get("/v1/rap/capabilities")
        assert capabilities.status_code == 200
        assert "bounded_turns_v1" in capabilities.json()["capabilities"]
        with client.websocket_connect("/v1/rap/sessions/tempo") as websocket:
            websocket.send_json(_start(capabilities=["tempo_updates_v1", "bounded_turns_v1"],
                                       turn_id="turn-a", end_bar_exclusive=2))
            started = websocket.receive_json()
            assert "tempo_updates_v1" in started["capabilities"]
            websocket.send_json({"type": "tempo_update", "update": {"edit_seq": 1, "from_bar": 1, "bpm": 120}})
            ack = _receive_until(websocket, lambda m, s: m["type"] == "tempo_updated")[-1]
            assert ack["effective_from_bar"] == 1 and ack["revision"] == 1
            assert ack['server_processing_ms'] >= 0
            assert ack['server_received_unix_ns'] > 0
            websocket.send_json({"type": "tempo_update", "update": {"edit_seq": 1, "from_bar": 1, "bpm": 120}})
            replay = _receive_until(websocket, lambda m, s: m["type"] == "tempo_updated")[-1]
            assert replay['revision'] == 1 and replay['effective_from_bar'] == 1
            assert service.session("tempo").snapshot()["end_bar_exclusive"] == 2
            websocket.send_json({"type": "stop"})
    finally:
        service.close()
    assert 'tempo_control_received session=tempo edit_seq=1' in caplog.text
    assert 'tempo_control_sent session=tempo edit_seq=1' in caplog.text


def test_unnegotiated_tempo_update_is_rejected_without_changing_session(tmp_path):
    client, service, _ = _client(tmp_path)
    try:
        with client.websocket_connect("/v1/rap/sessions/legacy") as websocket:
            websocket.send_json(_start())
            assert websocket.receive_json()["type"] == "started"
            websocket.send_json({"type": "tempo_update", "update": {"edit_seq": 1, "from_bar": 1, "bpm": 120}})
            error = _receive_until(websocket, lambda m, s: m["type"] == "error")[-1]
            assert "capability" in error["message"]
            assert service.session("legacy").snapshot()["timeline_revision"] == 0
            websocket.send_json({"type": "stop"})
    finally:
        service.close()


@pytest.mark.parametrize('ack_lost,transit_delay', [(False, 1.15), (True, 1.15), (False, 5.25)])
def test_controller_handles_delayed_ack_through_real_client_and_server(tmp_path, ack_lost, transit_delay):
    from streammuse.infrastructure.rap.lyric_stream_client import LyricStreamConnection
    from streammuse.domain.rap.timeline import RapTimeline
    from streammuse.domain.timing import Tempo
    from tests.unit.application.rap.test_bar_stream import _controller, _Source, _TimedEspeak

    client, service, _ = _client(tmp_path)
    connection = LyricStreamConnection('http://test')
    received_ack = threading.Event()
    release_ack = threading.Event()
    enqueued, results, failures = [], [], []
    controller = _controller(connection, _Source(), enqueued, timeline=RapTimeline(Tempo(90, 4, 4)),
                             fallback_renderer=_TimedEspeak(), startup_timeout_seconds=.01,
                             tempo_ack_timeout_seconds=.1 if ack_lost else (10.0 if transit_delay > 5 else 5.0))
    try:
        with client.websocket_connect('/v1/rap/sessions/s-e0') as websocket:
            class DelayedSocket:
                def send(self, raw):
                    websocket.send_text(raw)

                def recv(self):
                    raw = websocket.receive_text()
                    if json.loads(raw).get('type') == 'tempo_updated':
                        received_ack.set()
                        assert release_ack.wait(8)
                    return raw

                def close(self):
                    websocket.close()

            def connect():
                connection._socket = DelayedSocket()
                connection._socket.send(json.dumps(connection._start))

            connection._connect = connect
            controller.start()
            deadline = time.monotonic() + 2
            while not connection.capabilities and time.monotonic() < deadline:
                time.sleep(.005)

            def edit():
                try:
                    results.append(controller.update_tempo(50))
                except Exception as error:
                    failures.append(error)

            thread = threading.Thread(target=edit)
            thread.start()
            try:
                assert received_ack.wait(2)
                # Real request/route/session/receive correlation with >1 second transit delay.
                time.sleep(transit_delay)
                controller.on_tick(15)
                assert enqueued[-1].bar == 1 and enqueued[-1].timing.bpm == 90
                release_ack.set()
                thread.join(3)
                assert not thread.is_alive()
                assert service.session('s-e0').snapshot()['timeline_revision'] == 1
                controller.on_tick(31)
                controller.on_tick(47)
                if transit_delay > 5:
                    controller.on_tick(63)
                if ack_lost:
                    assert len(failures) == 1 and isinstance(failures[0], TimeoutError)
                    assert not results
                    assert controller.tempo_control_state['requires_reset'] is True
                    assert controller.timeline.revision == 0
                    assert enqueued[-1].timing.bpm == 90
                    assert enqueued[-1].audio.frame_count == 128_000
                    assert enqueued[-1].fallback_reason == 'tempo_ack_uncertain'
                else:
                    assert not failures
                    assert results[0]['effective_from_bar'] == (4 if transit_delay > 5 else 3)
                    assert results[0]['ack_elapsed_ms'] >= transit_delay * 1000
                    assert enqueued[-1].timing.bpm == 50
                    assert enqueued[-1].audio.frame_count == 230_400
            finally:
                release_ack.set()
                thread.join(4)
                controller.close()
    finally:
        controller.close()
        service.close()


def test_concurrent_reconnect_creates_exactly_one_bounded_session():
    barrier, closed = threading.Barrier(2), []

    def writer_factory():
        barrier.wait(timeout=2)
        return object(), lambda: closed.append(True)

    service = LyricStreamService(writer_factory=writer_factory, renderer=object())
    message = _start(capabilities=['bounded_turns_v1'], turn_id='same-turn', end_bar_exclusive=2)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(service.start, 'same', message) for _ in range(2)]
            results = [future.result(timeout=3) for future in futures]
        assert sum(created for _, created in results) == 1
        assert results[0][0] is results[1][0]
        assert len(closed) == 1
    finally:
        service.close()
    assert len(closed) == 2


def test_invalid_start_closes_its_writer_resource():
    closed = []
    service = LyricStreamService(writer_factory=lambda: (object(), lambda: closed.append(True)), renderer=object())
    with pytest.raises(ValueError):
        service.start('invalid', _start(tempo_bpm=float('nan')))
    assert closed == [True]


def test_session_streams_lyrics_and_serves_bar_audio_by_lyric_id(tmp_path) -> None:
    client, service, source = _client(tmp_path)
    try:
        with client.websocket_connect("/v1/rap/sessions/s1") as websocket:
            websocket.send_json(_start())
            seen = _receive_until(websocket, lambda m, s: sum(x["type"] == "lyric" for x in s) == 3)
            assert seen[0] == {"type": "started", "session_id": "s1", "created": True}
            records = [BarLyricRecord.from_payload(m["record"]) for m in seen if m["type"] == "lyric"]
            assert [record.bar for record in records] == [0, 1, 2]

            response = client.get(
                f"/v1/rap/sessions/s1/bars/0/audio?lyric_id={records[0].lyric_id}&wait_ms=5000"
            )
            assert response.status_code == 200
            assert response.headers["content-type"] == BAR_PACKAGE_MEDIA_TYPE
            assert response.headers["x-streammuse-fetch-status"] == "ready"
            decoded = decode_bar_package(response.content, expected_lyric_id=records[0].lyric_id)
            assert decoded.manifest.bar == 0
            assert decoded.manifest.source_sha256 == hashlib.sha256(decoded.source_wav).hexdigest()
            # The server rendered a one-bar request for that exact line.
            assert source.requests[0].bar_count == 1 and source.requests[0].text == records[0].text

            websocket.send_json({"type": "committed", "bar": 0, "lyric_id": records[0].lyric_id, "text": records[0].text})
            _receive_until(websocket, lambda m, s: m["type"] == "lyric" and m["record"]["bar"] == 3)

            edit = PlanEdit(edit_seq=1, from_bar=1, topic="space")
            websocket.send_json({"type": "update", "edit": edit.to_payload()})
            seen = _receive_until(websocket, lambda m, s: m["type"] == "lyric" and m["record"]["revision"] == 1)
            superseded = next(m for m in seen if m["type"] == "superseded")
            assert superseded["bars"][0] == 1 and superseded["edit_seq"] == 1
            assert seen[-1]["record"]["topic"] == "space"
            websocket.send_json({"type": "stop"})
    finally:
        service.close()


def test_fetch_errors_are_explicit(tmp_path) -> None:
    client, service, _ = _client(tmp_path)
    try:
        assert client.get(f"/v1/rap/sessions/missing/bars/0/audio?lyric_id={'a' * 64}").status_code == 404
        with client.websocket_connect("/v1/rap/sessions/s2") as websocket:
            websocket.send_json(_start("s2"))
            _receive_until(websocket, lambda m, s: m["type"] == "lyric")
            response = client.get(f"/v1/rap/sessions/s2/bars/0/audio?lyric_id={'a' * 64}")
            assert response.status_code == 404
            assert response.json()["error"]["code"] == "unknown_lyric"
            websocket.send_json({"type": "bogus"})
            error = _receive_until(websocket, lambda m, s: m["type"] == "error")[-1]
            assert "unknown message" in error["message"]
            websocket.send_json({"type": "stop"})
    finally:
        service.close()


def test_reconnect_reattaches_and_resends_live_lines(tmp_path) -> None:
    client, service, _ = _client(tmp_path)
    try:
        with client.websocket_connect("/v1/rap/sessions/s3") as websocket:
            websocket.send_json(_start("s3"))
            _receive_until(websocket, lambda m, s: sum(x["type"] == "lyric" for x in s) == 3)
        with client.websocket_connect("/v1/rap/sessions/s3") as websocket:
            websocket.send_json(_start("s3"))
            seen = _receive_until(websocket, lambda m, s: sum(x["type"] == "lyric" for x in s) == 3)
            assert seen[0]["created"] is False
            websocket.send_json({"type": "stop"})
    finally:
        service.close()


def test_bar_renderer_continues_the_previous_line_it_rendered(tmp_path) -> None:
    from streammuse.application.rap.bar_audio import MossBarAudioRenderer

    first = _record_for_continuation(0, "first line here now")
    second = _record_for_continuation(1, "second line here now")
    unknown = _record_for_continuation(1, "never rendered line")
    source = _SourceRenderer()
    renderer = MossBarAudioRenderer(source, PcmBarPackageEncoder(), workspace_root=tmp_path)

    renderer.render(first, tempo_bpm=90.0)
    rendered = renderer.render(second, tempo_bpm=90.0, previous=first)
    renderer.render(first, tempo_bpm=90.0, previous=unknown)

    assert source.prefixes[0] is None
    assert source.prefixes[1].text == first.text and source.prefixes[1].wav == _wav(60_000)
    assert rendered.manifest.diagnostics["continuation"] == {"used": True, "prefix_lyric_id": first.lyric_id}
    assert source.prefixes[2] is None

    disabled_source = _SourceRenderer()
    disabled = MossBarAudioRenderer(disabled_source, PcmBarPackageEncoder(), workspace_root=tmp_path, continuation=False)
    disabled.render(first, tempo_bpm=90.0)
    assert disabled.render(second, tempo_bpm=90.0, previous=first).manifest.diagnostics["continuation"]["used"] is False
    assert disabled_source.prefixes == [None, None]


def _record_for_continuation(bar: int, text: str):
    from streammuse.application.rap.alignment import align_exact
    from streammuse.domain.rap import ProsodyAnalysis, Syllable, materialize_flow, normalize_text
    from streammuse.domain.rap.lyric_stream import BarLyricRecord
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

    template = BUILTIN_TEMPLATES.get("baseline_straight_9")
    words = (normalize_text(text).split() * 3)[: len(template.slots)]
    line = " ".join(words)
    analysis = ProsodyAnalysis(
        text=line, normalized_text=line,
        syllables=tuple(Syllable(word, 0, 1, 1, ("AH1",), "fake") for word in words),
        end_rhyme_tail=("AH1",), oov_words=(), heuristic_words=(), punctuation_boundary_after=(),
    )
    return BarLyricRecord(
        session_id="s1", bar=bar, revision=0, topic="space", flow_template=template, text=line,
        scheduled=align_exact(analysis, materialize_flow(template, bar)), score=0.5,
    )
