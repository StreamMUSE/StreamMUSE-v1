"""Comparison source fetching stays offline and accepts PCM and Opus packages."""

import hashlib
import io
import json
import subprocess
from types import SimpleNamespace
import wave
import zipfile

import httpx
import pytest

from scripts.experiments import rap_alignment_comparison as comparison
from streammuse.infrastructure.rap.lyric_stream_client import HttpBarAudioSource
from streammuse.infrastructure.rap.opus_codec import FFmpegOpusCodec
from tests.unit.domain.rap.test_lyric_stream import _manifest, _record


class _StopBeforeRendering(Exception):
    pass


@pytest.mark.parametrize("transport", ["opus", "pcm"])
def test_compare_preserves_decoded_bar_source_and_manifest(monkeypatch, tmp_path, transport):
    record = _record(bar=0)
    pcm = b"\x00\x00" * 48_000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(pcm)
    source_wav = buffer.getvalue()
    manifest = _manifest(
        lyric_id=record.lyric_id, bar=0, frame_count=48_000,
        source_sha256=hashlib.sha256(source_wav).hexdigest(),
    )
    message = {"type": "lyric", "record": record.to_payload()}
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("manifest.json", manifest.canonical_json_bytes())
        archive.writestr(
            "source.opus" if transport == "opus" else "source.wav",
            b"encoded-opus" if transport == "opus" else source_wav,
        )

    def run_ffmpeg(self, command, data, **kwargs):
        assert transport == "opus", "PCM packages must not invoke FFmpeg"
        assert data == b"encoded-opus"
        return subprocess.CompletedProcess(command, 0, stdout=pcm, stderr=b"")

    def stop_before_rendering():
        raise _StopBeforeRendering

    monkeypatch.setattr(FFmpegOpusCodec, "_run", run_ffmpeg)
    monkeypatch.setattr(comparison.BUILTIN_TEMPLATES, "get", lambda _: record.flow_template)
    monkeypatch.setattr(comparison, "LyricStreamConnection", lambda _: SimpleNamespace(
        open=lambda session, start, handler: handler(message), close=lambda: None,
    ))
    monkeypatch.setattr(comparison, "probe_rubberband_r3", stop_before_rendering)
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=package.getvalue()),
    )) as client:
        monkeypatch.setattr(comparison, "HttpBarAudioSource", lambda url, **kwargs: HttpBarAudioSource(
            url, client=client, **kwargs,
        ))
        args = SimpleNamespace(
            url="http://test", output=tmp_path / "comparison", template="two-slot",
            bpm=90.0, bars=1, topic="deep sea", name="sample",
        )
        with pytest.raises(_StopBeforeRendering):
            comparison.compare(args)

    assert (args.output / "sample_bar_00_source.wav").read_bytes() == source_wav
    assert json.loads((args.output / "sample_bar_00_manifest.json").read_text()) == manifest.to_payload()
    assert json.loads((args.output / "stream_messages.json").read_text()) == [message]
