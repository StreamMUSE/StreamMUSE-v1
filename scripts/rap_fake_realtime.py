"""Fake real-time rap demo runs: the same system with no time constraint.

Runs a version's own demo (``build_demo`` + ``run``) with two changes, so every
remote result arrives before it is needed and nothing falls back for timing:

- a virtual clock that stops while any remote call is in flight (v0.x chunk
  requests, v1.0 bar-audio fetches); the controller, its deadlines and the WAV
  playback clock all run on it, so remote work takes zero time on the session
  timeline;
- v0.x chunk requests carry a 60 s server budget (``remaining_budget_ms``) and
  a 60 s deadline instead of the time left before playback, so the server is
  never cut short either. The budget is set where the preparation strategy
  takes the request (``RemoteMossChunkPreparationStrategy.prepare``), because
  the strategy checks that the server accepted the budget it sent.

Fallbacks that are not about time (e.g. the server finds no valid lyric
candidate) still happen, as they would in that version.

Run it with the version's own interpreter so the version's code is imported:

    ../StreamMUSE-rap-v0.1/.venv/bin/python scripts/rap_fake_realtime.py -- \\
        --scenario demo_page/scenarios/S01.json --rap-audio-renderer moss_aligned_remote \\
        --rap-render-url http://127.0.0.1:18220 --audio-output wav --no-web ...

Plan: developing-logs/plans/2026-10-12-rap-version-demo-page-plan.md.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

_SERVER_BUDGET_MS = 60_000
_TRANSPORT_TIMEOUT_SECONDS = 60.0


class FreezableClock:
    """``time.monotonic`` minus every interval during which a remote call was in flight."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self._frozen_since = 0.0
        self._paused = 0.0

    def __call__(self) -> float:
        with self._lock:
            now = time.monotonic()
            paused = self._paused + (now - self._frozen_since if self._in_flight else 0.0)
            return now - paused

    @property
    def paused_seconds(self) -> float:
        with self._lock:
            return self._paused

    @contextmanager
    def frozen(self):
        with self._lock:
            if self._in_flight == 0:
                self._frozen_since = time.monotonic()
            self._in_flight += 1
        try:
            yield
        finally:
            with self._lock:
                self._in_flight -= 1
                if self._in_flight == 0:
                    self._paused += time.monotonic() - self._frozen_since


class _FrozenChunkClient:
    """v0.x ``RemoteChunkClient`` whose requests take no session time."""

    def __init__(self, inner, clock: FreezableClock) -> None:
        self._inner = inner
        self._clock = clock

    def prepare(self, *args, **kwargs):
        with self._clock.frozen():
            return self._inner.prepare(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _unbound_chunk_strategy(clock: FreezableClock) -> bool:
    """Give every v0.x chunk request a 60 s server budget and deadline; False if the version has none."""
    try:
        from streammuse.application.rap import chunk_audio
    except ImportError:
        return False
    strategy = getattr(chunk_audio, "RemoteMossChunkPreparationStrategy", None)
    if strategy is None:
        return False
    original = strategy.prepare

    def prepare(self, request, *, deadline_monotonic):
        return original(
            self,
            replace(request, remaining_budget_ms=_SERVER_BUDGET_MS),
            deadline_monotonic=clock() + _TRANSPORT_TIMEOUT_SECONDS,
        )

    strategy.prepare = prepare
    return True


class _FrozenFetchSource:
    """v1.0 bar-audio source whose fetches take no session time."""

    def __init__(self, inner, clock: FreezableClock) -> None:
        self._inner = inner
        self._clock = clock

    def fetch(self, *args, **kwargs):
        with self._clock.frozen():
            return self._inner.fetch(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _factories(cli, clock: FreezableClock):
    class FakeRealtimeFactories(cli.RapAudioFactories):
        def create_remote_client(self, *, base_url, clock, audio_transport):
            return _FrozenChunkClient(
                super().create_remote_client(base_url=base_url, clock=clock, audio_transport=audio_transport),
                virtual,
            )

        def create_bar_audio_source(self, *, base_url, opus_codec):
            return _FrozenFetchSource(super().create_bar_audio_source(base_url=base_url, opus_codec=opus_codec), virtual)

        def create_sink(self, *, output, audio_format, audio_file, audio_device):
            from streammuse.infrastructure.rap.audio_output import (
                CompositeAudioSink,
                Float32WavAudioSink,
                TimedAudioSink,
            )

            if output != "wav":
                raise ValueError("fake real-time runs record with --audio-output wav")
            return CompositeAudioSink(
                TimedAudioSink(audio_format=audio_format, clock=virtual),
                Float32WavAudioSink(audio_file, audio_format),
            )

    virtual = clock
    return FakeRealtimeFactories()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demo_args", nargs=argparse.REMAINDER, help="arguments for streammuse-rap-demo, after --")
    args = parser.parse_args()
    demo_argv = args.demo_args[1:] if args.demo_args[:1] == ["--"] else args.demo_args

    import streammuse
    import streammuse.presentation.rap_demo.cli as cli

    demo_args = cli.build_parser().parse_args(demo_argv)
    if not demo_args.no_web:
        raise SystemExit("fake real-time runs are headless: pass --no-web")
    before = set(demo_args.log_dir.glob("*")) if demo_args.log_dir.exists() else set()
    clock = FreezableClock()
    # Protocol v3 (v1.0) has no per-request server budget; v0.x chunks do.
    unbounded = getattr(demo_args, "rap_protocol", "v1") != "v3" and _unbound_chunk_strategy(clock)
    started = time.monotonic()
    demo = cli.build_demo(demo_args, clock=clock, audio_factories=_factories(cli, clock))
    # The same branch as the demo's own main() for --no-web.
    if getattr(demo, "autostart", True):
        demo.run(max_bars=demo_args.max_bars)
    else:
        try:
            demo.start()
        finally:
            demo.close()
    sessions = sorted(set(demo_args.log_dir.glob("*")) - before)
    record = {
        "mode": "fake_realtime",
        "streammuse_package": str(Path(streammuse.__file__).resolve().parent),
        "server_budget_ms": _SERVER_BUDGET_MS if unbounded else None,
        "transport_timeout_seconds": _TRANSPORT_TIMEOUT_SECONDS,
        "remote_wait_seconds": round(clock.paused_seconds, 3),
        "wall_seconds": round(time.monotonic() - started, 3),
        "argv": demo_argv,
    }
    for session in sessions:
        if session.is_dir():
            (session / "fake_realtime.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
