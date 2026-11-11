"""Decoded-span storage and atomic callback commitment; no inference or I/O."""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from streammuse.domain.rap.inference import AudioSpan


@dataclass
class Entry:
    span: AudioSpan
    audio: np.ndarray
    ready_ms: float
    state: str = "ready"
    committed_frame: int | None = None


class SpanBuffer:
    def __init__(self, *, sample_rate, channels, frame_at):
        self.sample_rate, self.channels, self.frame_at = sample_rate, channels, frame_at
        self.entries: list[Entry] = []
        self.identities = set()
        self.active: Entry | None = None
        self.frozen_until = 0
        self.played_until = 0

    def accept(self, span, ready_ms):
        if type(ready_ms) not in (int, float) or not isfinite(ready_ms):
            raise ValueError("readiness timestamp must be finite")
        if span.sample_rate != self.sample_rate or span.channels != self.channels:
            raise ValueError("incompatible audio format")
        expected = self.frame_at(span.end_tick) - self.frame_at(span.start_tick)
        if span.frame_count != expected:
            raise ValueError("PCM frame count does not match musical coverage")
        audio = np.frombuffer(span.pcm, dtype=np.float32).reshape(-1, self.channels)
        if not np.all(np.isfinite(audio)):
            raise ValueError("PCM contains non-finite values")
        identity = (span.output_id, span.revision)
        if identity in self.identities:
            return False
        for entry in self.entries:
            old = entry.span
            if old.output_id == span.output_id and (old.start_tick, old.end_tick) != (span.start_tick, span.end_tick):
                raise ValueError("output revision cannot change coverage")
            if (old.input_version == span.input_version and old.request_id == span.request_id
                    and (old.start_tick, old.end_tick) == (span.start_tick, span.end_tick)
                    and old.revision >= span.revision):
                return False
            if old.input_version == span.input_version and old.request_id == span.request_id and (
                old.start_tick < span.end_tick and span.start_tick < old.end_tick
            ) and (old.start_tick, old.end_tick) != (span.start_tick, span.end_tick):
                raise ValueError("conflicting revision coverage")
        self.identities.add(identity)
        self.entries.append(Entry(span, audio, ready_ms))
        return True

    def prefix_ticks(self, version):
        end = 0
        for entry in sorted(self.entries, key=lambda value: value.span.start_tick):
            span = entry.span
            if span.input_version == version and span.start_tick <= end:
                end = max(end, span.end_tick)
        return end

    def select(self, frame, version, quantum):
        # An accepted device callback cannot be changed retrospectively. Keep
        # the current choice to its next declared safe split/commit boundary.
        if frame < self.frozen_until:
            return self.active, self.frozen_until
        options = []
        for entry in self.entries:
            span = entry.span
            start, end = self.frame_at(span.start_tick), self.frame_at(span.end_tick)
            safe = {start, *(self.frame_at(span.start_tick + tick) for tick in span.safe_split_ticks)}
            if start <= frame < end and frame in safe:
                options.append(entry)
        entry = max(options, key=lambda item: (item.span.input_version, item.span.revision,
                                               item.ready_ms), default=None)
        if entry is not None:
            span = entry.span
            safe = [self.frame_at(span.start_tick + offset)
                    for offset in (*span.safe_split_ticks, span.duration_ticks)]
            minimum = self.frame_at(self._next_boundary(frame, quantum))
            # Quantum limits steering granularity, but never invents a safe cut
            # inside a phoneme merely because delivery packets are smaller.
            stop = next((value for value in safe if value > frame and value >= minimum),
                        self.frame_at(span.end_tick))
            entry.state = "stale" if span.input_version != version else "playing"
            entry.committed_frame = frame
        else:
            future = [self.frame_at(item.span.start_tick + offset) for item in self.entries
                      for offset in (0, *item.span.safe_split_ticks)
                      if self.frame_at(item.span.start_tick + offset) > frame]
            # Freeze missing coverage to the next musical tick. This bounds
            # callback work while allowing newly decoded future safe joins.
            next_tick = self._next_boundary(frame, 1)
            stop = min([self.frame_at(next_tick), *future])
        self.active, self.frozen_until = entry, stop
        return entry, stop

    def _next_boundary(self, frame, quantum):
        low, high = 0, 1
        while self.frame_at(high * quantum) <= frame:
            high *= 2
        while low + 1 < high:
            middle = (low + high) // 2
            if self.frame_at(middle * quantum) <= frame:
                low = middle
            else:
                high = middle
        return high * quantum

    def discard_before(self, frame):
        self.played_until = frame
        for entry in self.entries:
            if self.frame_at(entry.span.end_tick) <= frame:
                entry.state = "played" if entry.committed_frame is not None else "late"
        # Preserve bounded diagnostics without retaining unbounded PCM history.
        if len(self.entries) > 128:
            self.entries = [entry for entry in self.entries
                            if self.frame_at(entry.span.end_tick) > frame] + [
                                entry for entry in self.entries
                                if self.frame_at(entry.span.end_tick) <= frame][-64:]

    def snapshot(self):
        return [{"span_id": entry.span.output_id, "request_id": entry.span.request_id,
                 "start_tick": entry.span.start_tick, "duration_ticks": entry.span.duration_ticks,
                 "input_version": entry.span.input_version, "revision": entry.span.revision,
                 "state": entry.state, "text": entry.span.text, "ready_ms": entry.ready_ms,
                 "committed_frame": entry.committed_frame,
                 "declared_silence": entry.span.declared_silence}
                for entry in self.entries[-128:]]
