from __future__ import annotations

import hashlib
import io
import wave

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

    def render_source(self, request, workspace):
        self.requests.append(request)
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
