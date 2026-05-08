"""Filesystem-backed prompt repository (loads prompts/metadata.json)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mido

from streammuse.domain.musical import MusicalEvent, Note


@dataclass(frozen=True)
class PromptRepositoryConfig:
    library_path: str = "prompts"
    ticks_per_beat: int = 4


class FileSystemPromptRepository:
    def __init__(self, config: PromptRepositoryConfig | None = None) -> None:
        self._config = config or PromptRepositoryConfig()
        self._library = Path(self._config.library_path)
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        path = self._library / "metadata.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def get_prompts_for_key(self, key: str) -> List[Dict[str, Any]]:
        return list(self._metadata.get(key, []))

    def select_prompt(self, key: str, strategy: str | int = "random") -> Optional[Dict[str, Any]]:
        prompts = self.get_prompts_for_key(key)
        if not prompts and key != "C_major":
            prompts = self.get_prompts_for_key("C_major")
        if not prompts:
            return None

        if strategy == "random":
            return random.choice(prompts)
        if strategy == "first":
            return prompts[0]
        if isinstance(strategy, int) and 0 <= strategy < len(prompts):
            return prompts[strategy]
        return prompts[0]

    def _resolve_prompt_path(self, key: str, p: str) -> Path:
        path = Path(p)
        if path.exists():
            return path
        # Common case: metadata contains absolute paths from another machine.
        # Try resolving as `<library>/<key>/<basename>`.
        candidate = self._library / key / path.name
        return candidate

    @staticmethod
    def _midi_to_notes(midi_path: Path, *, beat_div: int, max_tick: int) -> List[Dict[str, int]]:
        midi_file = mido.MidiFile(str(midi_path))
        resolution = int(midi_file.ticks_per_beat)
        ticks_per_output_tick = resolution / float(beat_div)

        notes: List[Dict[str, int]] = []
        active: Dict[Tuple[int, int], List[int]] = {}

        for track in midi_file.tracks:
            current_tick = 0
            for msg in track:
                current_tick += int(msg.time)
                if msg.type == "note_on" and int(msg.velocity) > 0:
                    output_tick = int(round(current_tick / ticks_per_output_tick))
                    active.setdefault((int(msg.channel), int(msg.note)), []).append(output_tick)
                elif msg.type == "note_off" or (msg.type == "note_on" and int(msg.velocity) == 0):
                    key = (int(msg.channel), int(msg.note))
                    if key not in active or not active[key]:
                        continue
                    start_tick = active[key].pop(0)
                    if not active[key]:
                        active.pop(key, None)
                    output_tick = int(round(current_tick / ticks_per_output_tick))
                    duration = max(1, output_tick - start_tick)
                    if start_tick < max_tick:
                        notes.append({"pitch": int(msg.note), "tick": int(start_tick), "duration": int(duration)})

        notes.sort(key=lambda n: (n["tick"], n["pitch"]))
        return [n for n in notes if n["tick"] < max_tick]

    def load_prompt_events(
        self,
        *,
        key: str,
        prompt: Dict[str, Any],
        max_ticks: int,
        load_melody: bool = True,
        load_accompaniment: bool = True,
    ) -> tuple[List[MusicalEvent], List[MusicalEvent]]:
        melody_events: List[MusicalEvent] = []
        accompaniment_events: List[MusicalEvent] = []

        if load_melody and "melody_path" in prompt:
            mel_path = self._resolve_prompt_path(key, str(prompt["melody_path"]))
            if mel_path.exists():
                mel_notes = self._midi_to_notes(
                    mel_path, beat_div=self._config.ticks_per_beat, max_tick=int(max_ticks)
                )
                for n in mel_notes:
                    melody_events.extend(
                        Note(pitch=n["pitch"], tick=n["tick"], duration=n["duration"], program=0).to_events()
                    )

        if load_accompaniment and "accompaniment_path" in prompt:
            acc_path = self._resolve_prompt_path(key, str(prompt["accompaniment_path"]))
            if acc_path.exists():
                acc_notes = self._midi_to_notes(
                    acc_path, beat_div=self._config.ticks_per_beat, max_tick=int(max_ticks)
                )
                for n in acc_notes:
                    accompaniment_events.extend(
                        Note(pitch=n["pitch"], tick=n["tick"], duration=n["duration"], program=1).to_events()
                    )

        return melody_events, accompaniment_events
