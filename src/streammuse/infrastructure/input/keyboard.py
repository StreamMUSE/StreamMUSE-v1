"""Computer keyboard input adapter implementing InputSource.

Uses the legacy StreamMUSE key→pitch mapping. Emits `MusicalEvent` with `tick=0`;
the application layer is responsible for timestamping and tick assignment.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from streammuse.domain.musical import EventType, MusicalEvent


DEFAULT_KEY_TO_PITCH: dict[str, int] = {
    # White keys (bottom row)
    "z": 60,
    "x": 62,
    "c": 64,
    "v": 65,
    "b": 67,
    "n": 69,
    "m": 71,
    ",": 72,
    ".": 74,
    "/": 76,
    # Black keys (top row)
    "s": 61,
    "d": 63,
    "g": 66,
    "h": 68,
    "j": 70,
    "l": 73,
    ";": 75,
}


@dataclass(frozen=True)
class KeyboardInputConfig:
    velocity: int = 100


class KeyboardInput:
    """
    InputSource that listens to computer keyboard events.

    Implementation note: this uses `pynput` when available. Unit tests should
    exercise `_handle_key_down/_handle_key_up` directly to avoid GUI dependency.
    """

    def __init__(
        self,
        *,
        key_to_pitch: Optional[dict[str, int]] = None,
        config: KeyboardInputConfig | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._key_to_pitch = dict(key_to_pitch or DEFAULT_KEY_TO_PITCH)
        self._config = config or KeyboardInputConfig()
        self._now = now

        self._events: queue.Queue[MusicalEvent | None] = queue.Queue()
        self._pressed: set[str] = set()
        self._listener = None
        self._thread: threading.Thread | None = None
        self._closed = False

    def _handle_key_down(self, char: str) -> None:
        if char not in self._key_to_pitch:
            return
        if char in self._pressed:
            return
        self._pressed.add(char)
        pitch = self._key_to_pitch[char]
        self._events.put(
            MusicalEvent(
                tick=0,
                pitch=pitch,
                event_type=EventType.NOTE_ON,
                velocity=int(self._config.velocity),
            )
        )

    def _handle_key_up(self, char: str) -> None:
        if char not in self._key_to_pitch:
            return
        if char not in self._pressed:
            return
        self._pressed.remove(char)
        pitch = self._key_to_pitch[char]
        self._events.put(
            MusicalEvent(
                tick=0,
                pitch=pitch,
                event_type=EventType.NOTE_OFF,
                velocity=0,
            )
        )

    def _run_listener(self) -> None:
        try:
            from pynput import keyboard  # type: ignore
        except Exception as e:  # pragma: no cover
            # We keep this adapter as optional, since pynput requires a graphical
            # environment. Fail fast but terminate the iterator cleanly.
            self._events.put(None)
            raise RuntimeError(f"Keyboard input unavailable (pynput import failed): {e}") from e

        def on_press(key) -> None:  # pragma: no cover
            if self._closed:
                return
            try:
                ch = key.char
            except AttributeError:
                return
            self._handle_key_down(ch)

        def on_release(key) -> bool | None:  # pragma: no cover
            if self._closed:
                return False
            try:
                ch = key.char
            except AttributeError:
                # ESC stops
                if key == keyboard.Key.esc:
                    self._events.put(None)
                    return False
                return None
            self._handle_key_up(ch)
            return None

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:  # pragma: no cover
            self._listener = listener
            listener.join()

    def read_events(self) -> Iterator[MusicalEvent]:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run_listener, daemon=True)
            self._thread.start()

        while not self._closed:
            ev = self._events.get()
            if ev is None:
                break
            yield ev

    def close(self) -> None:
        self._closed = True
        try:
            self._events.put_nowait(None)
        except Exception:
            pass
