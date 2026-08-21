from __future__ import annotations

import hashlib
import io
import json
import struct
import wave
from threading import Event, Lock, Thread, current_thread
from time import monotonic, sleep

import httpx
import pytest

from streammuse.domain.rap import (
    FlowProvenance,
    FlowSlot,
    FlowTemplate,
    REMOTE_CHUNK_SCHEMA_VERSION,
    RemoteCandidatePolicy,
    RemoteCandidateStats,
    RemoteRapBarRequest,
    RemoteRapChunkDiagnostics,
    RemoteRapChunkManifest,
    RemoteRapChunkRequest,
    RemoteSelectedBar,
    ScheduledSyllable,
    Syllable,
    materialize_flow,
)
from streammuse.infrastructure.rap.chunk_package import (
    RAP_CHUNK_PACKAGE_MEDIA_TYPE,
    RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE,
    encode_opus_chunk_package,
    encode_chunk_package,
)
from streammuse.infrastructure.rap.remote_chunk_client import (
    RemoteChunkCancelled,
    RemoteChunkClient,
    RemoteChunkClientError,
    RemoteChunkDeadlineExceeded,
    RemoteChunkProtocolError,
)


class _ChunkStream(httpx.SyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...], before_chunk=lambda index: None) -> None:
        self._chunks = chunks
        self._before_chunk = before_chunk

    def __iter__(self):
        for index, chunk in enumerate(self._chunks):
            self._before_chunk(index)
            yield chunk


class _TruncatedStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b'{"schema_version":'
        raise httpx.ReadError("truncated response")


def _flow() -> FlowTemplate:
    return FlowTemplate(
        template_id="test-flow",
        name="Test flow",
        ticks_per_beat=4,
        beats_per_bar=4,
        slots=(
            FlowSlot(tick_in_bar=0, duration_ticks=4, target_stress=1.0),
            FlowSlot(tick_in_bar=8, duration_ticks=4, target_stress=0.5),
        ),
        provenance=FlowProvenance(kind="test", source="unit-test"),
    )


def _request() -> RemoteRapChunkRequest:
    flow = _flow()
    return RemoteRapChunkRequest.create(
        session_id="session-1",
        chunk_index=0,
        bars=(RemoteRapBarRequest(0, "space", flow), RemoteRapBarRequest(1, "space", flow)),
        tempo_bpm=90.0,
        remaining_budget_ms=5_000,
        policy=RemoteCandidatePolicy.realtime_default(),
        context_lines=("previous line",),
        seed=7,
    )


def _package(request: RemoteRapChunkRequest) -> bytes:
    samples = struct.pack("<h", 1_000) * request.expected_frame_count
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes(samples)
    vocal_wav = wav_buffer.getvalue()
    selected = tuple(
        RemoteSelectedBar.create(
            bar_request,
            text="orbit orbit",
            scheduled=tuple(
                ScheduledSyllable(slot, Syllable("orbit", index // 2, 2, 1, ("AO1",), "test"))
                for index, slot in enumerate(materialize_flow(bar_request.flow_template, bar_request.bar))
            ),
            score=0.9,
        )
        for bar_request in request.bars
    )
    manifest = RemoteRapChunkManifest(
        request_id=request.request_id,
        chunk_index=request.chunk_index,
        tempo_bpm=request.tempo_bpm,
        output_sample_rate_hz=request.output_sample_rate_hz,
        expected_frame_count=request.expected_frame_count,
        selected_bars=selected,
        diagnostics=RemoteRapChunkDiagnostics(
            accepted_request_budget_ms=request.remaining_budget_ms,
            resolved_policy=request.policy,
            candidate_stats=RemoteCandidateStats(2, 2, 2, 2, (), ()),
            stage_timings_ms={
                "generation": 1.0,
                "evaluation": 1.0,
                "moss": 1.0,
                "aligner": 1.0,
                "warp": 1.0,
                "packaging": 1.0,
                "total": 6.0,
            },
            alignment_diagnostics={
                "fallback_counts": {"word": 0},
                "source_anchors": [0.0, 1.0, 2.0, 3.0],
                "target_anchors": [0.0, 1.0, 2.0, 3.0],
                "local_warp_ratios": [1.0],
            },
            audio_diagnostics={
                "sample_rate_hz": 24_000,
                "frame_count": request.expected_frame_count,
                "duration_seconds": request.expected_frame_count / 24_000,
                "peak": 0.5,
            },
            model_tool_versions={"moss": "test", "aligner": "test", "rubberband": "test"},
            warnings=(),
        ),
        vocal_sha256=hashlib.sha256(vocal_wav).hexdigest(),
    )
    return encode_chunk_package(manifest, vocal_wav)


def _client(
    handler,
    *,
    clock=lambda: 0.0,
    max_response_bytes=4 * 1024 * 1024,
    max_health_response_bytes=16 * 1024,
) -> RemoteChunkClient:
    return RemoteChunkClient(
        "http://render.test",
        transport=httpx.MockTransport(handler),
        clock=clock,
        max_response_bytes=max_response_bytes,
        max_health_response_bytes=max_health_response_bytes,
    )


def test_prepare_sends_canonical_binary_request_and_reports_transfer_timing() -> None:
    request = _request()
    package = _package(request)
    observed: list[httpx.Request] = []
    clock_values = iter((10.0, 10.0, 10.1, 10.3, 10.5, 10.5))

    def handler(http_request: httpx.Request) -> httpx.Response:
        observed.append(http_request)
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    result = _client(handler, clock=lambda: next(clock_values)).prepare(
        request, timeout_seconds=1.0, deadline_monotonic=12.0
    )

    assert observed[0].method == "POST"
    assert observed[0].url.path == "/v1/rap/chunks/render"
    assert observed[0].content == request.canonical_json_bytes()
    assert observed[0].headers["content-type"] == "application/json"
    assert observed[0].headers["idempotency-key"] == request.request_id
    assert result.package.manifest.request_id == request.request_id
    assert result.timing.request_ms == pytest.approx(100.0)
    assert result.timing.first_byte_ms == pytest.approx(200.0)
    assert result.timing.download_ms == pytest.approx(200.0)


def test_opus_client_prefers_opus_and_accepts_pcm_fallback_from_pcm_server() -> None:
    request = _request()
    package = _package(request)
    observed: list[httpx.Request] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        observed.append(http_request)
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    result = RemoteChunkClient(
        "http://render.test", transport=httpx.MockTransport(handler), clock=lambda: 0.0, audio_transport="opus"
    ).prepare(request, timeout_seconds=1.0, deadline_monotonic=2.0)

    assert RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE in observed[0].headers["accept"]
    assert RAP_CHUNK_PACKAGE_MEDIA_TYPE in observed[0].headers["accept"]
    assert result.package.transport_codec == "pcm"


def test_opus_client_decodes_negotiated_opus_response(monkeypatch) -> None:
    request = _request()
    canonical = _package(request)
    from streammuse.infrastructure.rap.chunk_package import decode_chunk_package

    decoded = decode_chunk_package(canonical, expected_request_id=request.request_id)

    class Codec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            return b"encoded-opus"

        def decode_to_pcm16_mono_24khz(
            self,
            encoded: bytes,
            *,
            expected_frame_count: int,
            timeout_seconds: float | None = None,
            cancelled=None,
        ) -> bytes:
            assert encoded == b"encoded-opus"
            assert timeout_seconds == pytest.approx(2.0)
            assert not cancelled()
            return decoded.vocal_wav[44:]

    monkeypatch.setattr("streammuse.infrastructure.rap.opus_codec.FFmpegOpusCodec", Codec)
    opus_package = encode_opus_chunk_package(decoded.manifest, decoded.vocal_wav, Codec())

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE}, content=opus_package)

    response = RemoteChunkClient(
        "http://render.test", transport=httpx.MockTransport(handler), clock=lambda: 0.0, audio_transport="opus"
    ).prepare(request, timeout_seconds=1.0, deadline_monotonic=2.0)

    assert response.package.transport_codec == "opus"


def test_abort_during_opus_decode_stops_decoder_and_reports_cancelled(monkeypatch) -> None:
    request = _request()
    canonical = _package(request)
    from streammuse.infrastructure.rap.chunk_package import decode_chunk_package

    decoded = decode_chunk_package(canonical, expected_request_id=request.request_id)
    entered = Event()
    stopped = Event()

    class PackageCodec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            return b"encoded-opus"

        def decode_to_pcm16_mono_24khz(self, encoded: bytes, *, expected_frame_count: int) -> bytes:
            return decoded.vocal_wav[44:]

    opus_package = encode_opus_chunk_package(decoded.manifest, decoded.vocal_wav, PackageCodec())

    class BlockingCodec:
        def decode_to_pcm16_mono_24khz(
            self, encoded: bytes, *, expected_frame_count: int, timeout_seconds: float, cancelled
        ) -> bytes:
            entered.set()
            while not cancelled():
                sleep(0.005)
            stopped.set()
            raise RuntimeError("decoder stopped")

    monkeypatch.setattr("streammuse.infrastructure.rap.opus_codec.FFmpegOpusCodec", BlockingCodec)

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE},
            content=opus_package,
        )

    client = RemoteChunkClient(
        "http://render.test",
        transport=httpx.MockTransport(handler),
        clock=monotonic,
        audio_transport="opus",
    )
    outcome: list[BaseException | object] = []
    worker = Thread(
        target=lambda: _capture_prepare(outcome, client, request, monotonic() + 2.0)
    )
    worker.start()
    assert entered.wait(1.0)
    client.abort()
    worker.join(0.5)

    assert not worker.is_alive()
    assert stopped.wait(0.5)
    assert len(outcome) == 1
    assert isinstance(outcome[0], RemoteChunkCancelled)
    client.close()


def test_attempt_timeout_cancels_opus_decode_without_retry_or_worker_leak(
    monkeypatch,
) -> None:
    request = _request()
    canonical = _package(request)
    from streammuse.infrastructure.rap.chunk_package import decode_chunk_package

    decoded = decode_chunk_package(canonical, expected_request_id=request.request_id)
    entered = Event()
    cancellation_observed = Event()
    release = Event()
    decoder_threads: list[Thread] = []
    decoder_calls = 0
    request_calls = 0

    class PackageCodec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            return b"encoded-opus"

        def decode_to_pcm16_mono_24khz(self, encoded: bytes, *, expected_frame_count: int) -> bytes:
            return decoded.vocal_wav[44:]

    opus_package = encode_opus_chunk_package(
        decoded.manifest, decoded.vocal_wav, PackageCodec()
    )

    class BlockingCodec:
        def decode_to_pcm16_mono_24khz(
            self,
            encoded: bytes,
            *,
            expected_frame_count: int,
            timeout_seconds: float,
            cancelled,
        ) -> bytes:
            nonlocal decoder_calls
            decoder_calls += 1
            decoder_threads.append(current_thread())
            entered.set()
            while not cancelled() and not release.is_set():
                sleep(0.005)
            if cancelled():
                cancellation_observed.set()
            raise RuntimeError("decoder stopped")

    monkeypatch.setattr(
        "streammuse.infrastructure.rap.opus_codec.FFmpegOpusCodec", BlockingCodec
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal request_calls
        request_calls += 1
        return httpx.Response(
            200,
            headers={"content-type": RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE},
            content=opus_package,
        )

    client = RemoteChunkClient(
        "http://render.test",
        transport=httpx.MockTransport(handler),
        clock=monotonic,
        audio_transport="opus",
    )
    try:
        with pytest.raises(RemoteChunkDeadlineExceeded, match="attempt timed out"):
            client.prepare(
                request,
                timeout_seconds=0.1,
                deadline_monotonic=monotonic() + 0.5,
            )
        assert entered.is_set()
        assert cancellation_observed.wait(0.5)
        assert request_calls == 1
        assert decoder_calls == 1
        assert all(not thread.is_alive() for thread in decoder_threads)
    finally:
        release.set()
        for thread in decoder_threads:
            thread.join(0.5)
        client.close()


def test_attempt_timeout_is_bounded_while_pre_header_worker_cleans_up() -> None:
    request = _request()
    package = _package(request)
    entered_transport = Event()
    release_headers = Event()
    worker_finished = Event()
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered_transport.set()
            release_headers.wait(2.0)
            worker_finished.set()
        return httpx.Response(
            200,
            headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE},
            content=package,
        )

    client = _client(handler, clock=monotonic)
    started = monotonic()
    try:
        with pytest.raises(RemoteChunkDeadlineExceeded, match="attempt timed out"):
            client.prepare(
                request,
                timeout_seconds=0.05,
                deadline_monotonic=monotonic() + 1.0,
            )
        assert entered_transport.is_set()
        assert monotonic() - started < 0.25

        with pytest.raises(RemoteChunkClientError, match="already has an active request"):
            client.prepare(request, timeout_seconds=1.0)

        release_headers.set()
        assert worker_finished.wait(0.5)
        deadline = monotonic() + 0.5
        while True:
            try:
                response = client.prepare(request, timeout_seconds=1.0)
                break
            except RemoteChunkClientError as error:
                if "already has an active request" not in str(error) or monotonic() >= deadline:
                    raise
                sleep(0.005)
        assert response.package.manifest.request_id == request.request_id
        assert calls == 2
    finally:
        release_headers.set()
        client.close()


def test_deadline_during_opus_decode_reports_deadline_instead_of_protocol_error(
    monkeypatch,
) -> None:
    request = _request()
    canonical = _package(request)
    from streammuse.infrastructure.rap.chunk_package import decode_chunk_package

    decoded = decode_chunk_package(canonical, expected_request_id=request.request_id)
    now = [0.0]

    class PackageCodec:
        encoder_identity = "ffmpeg test / libopus"

        def encode_pcm16_mono_24khz(self, pcm: bytes, *, expected_frame_count: int) -> bytes:
            return b"encoded-opus"

        def decode_to_pcm16_mono_24khz(self, encoded: bytes, *, expected_frame_count: int) -> bytes:
            return decoded.vocal_wav[44:]

    opus_package = encode_opus_chunk_package(decoded.manifest, decoded.vocal_wav, PackageCodec())

    class DeadlineCodec:
        def decode_to_pcm16_mono_24khz(
            self, encoded: bytes, *, expected_frame_count: int, timeout_seconds: float, cancelled
        ) -> bytes:
            assert timeout_seconds == pytest.approx(5.0)
            assert not cancelled()
            now[0] = 5.0
            raise RuntimeError("ffmpeg deadline")

    monkeypatch.setattr("streammuse.infrastructure.rap.opus_codec.FFmpegOpusCodec", DeadlineCodec)

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": RAP_CHUNK_OPUS_PACKAGE_MEDIA_TYPE},
            content=opus_package,
        )

    client = RemoteChunkClient(
        "http://render.test",
        transport=httpx.MockTransport(handler),
        clock=lambda: now[0],
        audio_transport="opus",
    )
    with pytest.raises(RemoteChunkDeadlineExceeded, match="decode"):
        client.prepare(request, timeout_seconds=5.0, deadline_monotonic=5.0)


def _capture_prepare(
    outcome: list[BaseException | object],
    client: RemoteChunkClient,
    request: RemoteRapChunkRequest,
    deadline: float,
) -> None:
    try:
        outcome.append(
            client.prepare(request, timeout_seconds=2.0, deadline_monotonic=deadline)
        )
    except BaseException as error:
        outcome.append(error)


def test_prepare_retries_the_identical_request_before_its_deadline() -> None:
    request = _request()
    package = _package(request)
    bodies: list[bytes] = []
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        bodies.append(http_request.content)
        if calls == 1:
            return httpx.Response(503, json={"detail": "private internal failure with token=secret"})
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    result = _client(handler).prepare(request, timeout_seconds=1.0, deadline_monotonic=2.0)

    assert result.package.manifest.request_id == request.request_id
    assert calls == 2
    assert bodies == [request.canonical_json_bytes(), request.canonical_json_bytes()]


def test_prepare_reduces_each_retry_timeout_to_the_remaining_deadline() -> None:
    request = _request()
    package = _package(request)
    now = [0.0]
    read_timeouts: list[float] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        read_timeouts.append(http_request.extensions["timeout"]["read"])
        if len(read_timeouts) == 1:
            now[0] = 7.0
            return httpx.Response(503)
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    result = _client(handler, clock=lambda: now[0]).prepare(
        request,
        timeout_seconds=8.0,
        deadline_monotonic=10.0,
    )

    assert result.timing.attempts == 2
    assert read_timeouts == [8.0, 3.0]


def test_prepare_does_not_retry_after_retryable_response_reaches_deadline() -> None:
    now = [0.0]
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        now[0] = 5.0
        return httpx.Response(503)

    with pytest.raises(RemoteChunkDeadlineExceeded, match="before retry"):
        _client(handler, clock=lambda: now[0]).prepare(
            _request(),
            timeout_seconds=5.0,
            deadline_monotonic=5.0,
        )
    assert calls == 1


def test_prepare_does_not_start_a_request_at_or_after_deadline() -> None:
    calls = 0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with pytest.raises(RemoteChunkDeadlineExceeded):
        _client(handler, clock=lambda: 5.0).prepare(_request(), timeout_seconds=1.0, deadline_monotonic=5.0)
    assert calls == 0


def test_prepare_does_not_start_when_deadline_expires_before_transport_entry() -> None:
    calls = 0
    clock_calls = 0

    def clock() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 5.0

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    with pytest.raises(RemoteChunkDeadlineExceeded):
        _client(handler, clock=clock).prepare(_request(), timeout_seconds=5.0, deadline_monotonic=5.0)
    assert calls == 0


def test_prepare_rejects_response_headers_arriving_at_the_deadline() -> None:
    request = _request()
    package = _package(request)
    now = [0.0]

    def handler(http_request: httpx.Request) -> httpx.Response:
        now[0] = 5.0
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    with pytest.raises(RemoteChunkDeadlineExceeded):
        _client(handler, clock=lambda: now[0]).prepare(
            request,
            timeout_seconds=5.0,
            deadline_monotonic=5.0,
        )


def test_prepare_rejects_deadline_expiry_during_streamed_download() -> None:
    request = _request()
    package = _package(request)
    now = [0.0]

    def before_chunk(index: int) -> None:
        if index == 1:
            now[0] = 5.0

    def handler(http_request: httpx.Request) -> httpx.Response:
        midpoint = len(package) // 2
        return httpx.Response(
            200,
            headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE},
            stream=_ChunkStream((package[:midpoint], package[midpoint:]), before_chunk),
        )

    with pytest.raises(RemoteChunkDeadlineExceeded):
        _client(handler, clock=lambda: now[0]).prepare(
            request,
            timeout_seconds=5.0,
            deadline_monotonic=5.0,
        )


def test_prepare_rejects_deadline_expiry_during_package_decode() -> None:
    request = _request()
    package = _package(request)
    clock_values = iter((0.0, 0.0, 0.0, 0.0, 0.0, 5.0))

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    with pytest.raises(RemoteChunkDeadlineExceeded, match="response preparation"):
        _client(handler, clock=lambda: next(clock_values)).prepare(
            request,
            timeout_seconds=5.0,
            deadline_monotonic=5.0,
        )


def test_prepare_rejects_oversized_and_non_binary_responses_with_sanitized_error() -> None:
    request = _request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"db password=secret and a very long remote trace")

    client = _client(handler)
    with pytest.raises(RemoteChunkProtocolError, match="remote render request failed: HTTP 500") as error:
        client.prepare(request, timeout_seconds=1.0)
    assert "secret" not in str(error.value)

    def oversized(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE, "content-length": "17"}, content=b"x" * 17)

    with pytest.raises(RemoteChunkProtocolError, match="response exceeds configured limit"):
        _client(oversized, max_response_bytes=16).prepare(request, timeout_seconds=1.0)


def test_prepare_rejects_wrong_media_type() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b"{}")

    with pytest.raises(RemoteChunkProtocolError, match="unsupported media type"):
        _client(handler).prepare(_request(), timeout_seconds=1.0)


def test_prepare_rejects_headerless_stream_overflow() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE},
            stream=_ChunkStream((b"12345678", b"9")),
        )

    with pytest.raises(RemoteChunkProtocolError, match="response exceeds configured limit"):
        _client(handler, max_response_bytes=8).prepare(_request(), timeout_seconds=1.0)


def test_prepare_rejects_calls_after_close() -> None:
    client = _client(lambda request: httpx.Response(500))
    client.close()

    with pytest.raises(RemoteChunkClientError, match="closed"):
        client.prepare(_request(), timeout_seconds=1.0)


def test_abort_is_reusable_and_close_is_idempotent() -> None:
    request = _request()
    package = _package(request)
    calls = 0
    client: RemoteChunkClient

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            client.abort()
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    client = _client(handler)
    with pytest.raises(RemoteChunkCancelled):
        client.prepare(request, timeout_seconds=1.0)
    assert client.prepare(request, timeout_seconds=1.0).package.manifest.request_id == request.request_id
    client.close()
    client.close()


def test_abort_unblocks_prepare_while_transport_waits_for_response_headers() -> None:
    request = _request()
    package = _package(request)
    entered_transport = Event()
    release_headers = Event()
    finished = Event()
    outcome: list[BaseException | object] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        entered_transport.set()
        release_headers.wait(2.0)
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    client = _client(handler, clock=monotonic)

    def prepare() -> None:
        try:
            outcome.append(client.prepare(request, timeout_seconds=2.0))
        except BaseException as error:
            outcome.append(error)
        finally:
            finished.set()

    worker = Thread(target=prepare)
    worker.start()
    assert entered_transport.wait(1.0)
    try:
        client.abort()
        assert finished.wait(0.25), "abort did not unblock the pre-header wait"
        assert len(outcome) == 1
        assert isinstance(outcome[0], RemoteChunkCancelled)
    finally:
        release_headers.set()
        worker.join(2.0)
        client.close()


def test_abort_token_is_not_cleared_when_next_prepare_starts() -> None:
    request = _request()
    package = _package(request)
    first_entered = Event()
    release_first = Event()
    first_finished = Event()
    call_lock = Lock()
    calls = 0
    outcome: list[BaseException | object] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        with call_lock:
            calls += 1
            call = calls
        if call == 1:
            first_entered.set()
            release_first.wait(2.0)
            first_finished.set()
        return httpx.Response(200, headers={"content-type": RAP_CHUNK_PACKAGE_MEDIA_TYPE}, content=package)

    client = _client(handler, clock=monotonic)

    def prepare_first() -> None:
        try:
            outcome.append(client.prepare(request, timeout_seconds=2.0))
        except BaseException as error:
            outcome.append(error)

    worker = Thread(target=prepare_first)
    worker.start()
    assert first_entered.wait(1.0)
    try:
        client.abort()
        worker.join(0.25)
        assert not worker.is_alive(), "abort did not release the first useful wait"
        assert len(outcome) == 1
        assert isinstance(outcome[0], RemoteChunkCancelled)

        second = client.prepare(request, timeout_seconds=2.0)
        assert second.package.manifest.request_id == request.request_id
        assert calls == 2
    finally:
        release_first.set()
        first_finished.wait(1.0)
        worker.join(2.0)
        client.close()


def test_close_cancels_pre_header_wait_without_starting_a_retry() -> None:
    request = _request()
    entered_transport = Event()
    release_headers = Event()
    finished = Event()
    calls = 0
    outcome: list[BaseException | object] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        entered_transport.set()
        release_headers.wait(2.0)
        return httpx.Response(503)

    client = _client(handler, clock=monotonic)

    def prepare() -> None:
        try:
            outcome.append(client.prepare(request, timeout_seconds=2.0))
        except BaseException as error:
            outcome.append(error)
        finally:
            finished.set()

    worker = Thread(target=prepare)
    worker.start()
    assert entered_transport.wait(1.0)
    try:
        client.close()
        assert finished.wait(0.25), "close did not cancel the useful pre-header wait"
        assert len(outcome) == 1
        assert isinstance(outcome[0], RemoteChunkCancelled)
        assert calls == 1
    finally:
        release_headers.set()
        worker.join(2.0)


def test_health_requires_the_remote_chunk_schema_version() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/health"
        return httpx.Response(200, json={"schema_version": "wrong", "ready": True})

    with pytest.raises(RemoteChunkProtocolError, match="health schema version is incompatible"):
        _client(handler).health(timeout_seconds=1.0)


def test_health_accepts_bounded_schema_compatible_json() -> None:
    payload = {
        "schema_version": REMOTE_CHUNK_SCHEMA_VERSION,
        "ready": True,
        "renderer": "moss",
    }

    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.url.path == "/health"
        return httpx.Response(200, json=payload)

    health = _client(handler).health(timeout_seconds=1.0)

    assert health.schema_version == REMOTE_CHUNK_SCHEMA_VERSION
    assert health.ready is True
    assert dict(health.details) == payload


def test_health_rejects_headerless_stream_overflow() -> None:
    payload = json.dumps(
        {
            "schema_version": REMOTE_CHUNK_SCHEMA_VERSION,
            "ready": True,
            "padding": "x" * 20_000,
        }
    ).encode("utf-8")

    def handler(http_request: httpx.Request) -> httpx.Response:
        midpoint = len(payload) // 2
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_ChunkStream((payload[:midpoint], payload[midpoint:])),
        )

    with pytest.raises(RemoteChunkProtocolError, match="health response exceeds configured limit"):
        _client(handler).health(timeout_seconds=1.0)


def test_health_rejects_declared_overflow_before_reading() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json", "content-length": "9"},
            stream=_ChunkStream((b"{}",)),
        )

    with pytest.raises(RemoteChunkProtocolError, match="health response exceeds configured limit"):
        _client(handler, max_health_response_bytes=8).health(timeout_seconds=1.0)


@pytest.mark.parametrize(
    "response",
    (
        httpx.Response(200, headers={"content-type": "application/json"}, content=b"{"),
        httpx.Response(200, headers={"content-type": "application/json"}, stream=_TruncatedStream()),
        httpx.Response(200, headers={"content-type": "text/plain"}, content=b"secret health trace"),
    ),
)
def test_health_rejects_malformed_truncated_and_non_json_content(response: httpx.Response) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return response

    with pytest.raises(RemoteChunkProtocolError) as error:
        _client(handler).health(timeout_seconds=1.0)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    (
        [],
        {"schema_version": REMOTE_CHUNK_SCHEMA_VERSION, "ready": "yes"},
    ),
)
def test_health_rejects_invalid_top_level_and_ready_types(payload: object) -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(RemoteChunkProtocolError, match="health response is invalid"):
        _client(handler).health(timeout_seconds=1.0)
