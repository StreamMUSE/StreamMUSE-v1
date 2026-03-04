import json

import mido

from streammuse.infrastructure.storage.prompt_repository import (
    FileSystemPromptRepository,
    PromptRepositoryConfig,
)


def _write_one_note_midi(path, note=60):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.Message("note_on", note=note, velocity=64, time=0))
    track.append(mido.Message("note_off", note=note, velocity=0, time=480))
    mid.save(str(path))


def test_prompt_repository_select_and_load(tmp_path):
    prompts_dir = tmp_path / "prompts"
    key_dir = prompts_dir / "C_major"
    key_dir.mkdir(parents=True)

    mel = key_dir / "mel.mid"
    acc = key_dir / "acc.mid"
    _write_one_note_midi(mel, note=60)
    _write_one_note_midi(acc, note=64)

    metadata = {
        "C_major": [
            {
                "name": "test",
                # Intentionally use absolute-looking paths to exercise resolver fallback.
                "melody_path": "/some/other/place/mel.mid",
                "accompaniment_path": "/some/other/place/acc.mid",
                "duration_ticks": 16,
            }
        ]
    }
    (prompts_dir / "metadata.json").write_text(json.dumps(metadata))

    repo = FileSystemPromptRepository(PromptRepositoryConfig(library_path=str(prompts_dir), ticks_per_beat=4))
    prompt = repo.select_prompt("C_major", strategy="first")
    assert prompt and prompt["name"] == "test"

    mel_events, acc_events = repo.load_prompt_events(key="C_major", prompt=prompt, max_ticks=32)
    assert any(e.pitch == 60 for e in mel_events)
    assert any(e.pitch == 64 for e in acc_events)

