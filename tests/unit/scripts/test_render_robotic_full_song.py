from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from streammuse.domain.rap import AudioFormat, PcmAudio, PreparedRapBar
from streammuse.experiments.rap_audio_protocols.artifacts import file_sha256


SONG_ID = "08_street_basketball"
TICKS = (0, 2, 3, 6, 7, 10, 11, 14, 15)


def _record(bar: int) -> dict[str, object]:
    words = ("street", "lights", "flick", "er", "as", "we", "hit", "the", "court")
    phonemes = (
        ("S", "T", "R", "IY1", "T"),
        ("L", "AY1", "T", "S"),
        ("F", "L", "IH1", "K"),
        ("ER0",),
        ("AE1", "Z"),
        ("W", "IY1"),
        ("HH", "IH1", "T"),
        ("DH", "AH0"),
        ("K", "AO1", "R", "T"),
    )
    schedule = []
    for slot_index, (tick, word, phones) in enumerate(zip(TICKS, words, phonemes, strict=True)):
        schedule.append(
            {
                "absolute_tick": bar * 16 + tick,
                "analysis_source": "cmudict_first_pronunciation",
                "lexical_stress": 0 if word == "er" else 1,
                "phonemes": list(phones),
                "slot_index": slot_index,
                "target_stress": slot_index / 10,
                "tick_in_bar": tick,
                "word": "flicker" if word in {"flick", "er"} else word,
            }
        )
    return {
        "bar": bar,
        "song_index": 8,
        "title": "Concrete Court",
        "topic": "street basketball",
        "template_id": "varied_crossbeat_push_9",
        "text": "Street lights flicker as we hit the court",
        "source": "h200_api_choices",
        "schedule": schedule,
    }


def _write_input(root: Path) -> Path:
    common = root / "common" / SONG_ID
    common.mkdir(parents=True)
    records = (_record(0), _record(1))
    (common / "chosen_lyrics.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (common / "requests.jsonl").write_text(
        json.dumps({"chunk_index": 0, "start_bar": 0, "end_bar": 2, "tempo_bpm": 90.0}) + "\n",
        encoding="utf-8",
    )
    wavfile.write(
        common / "drums.wav",
        48_000,
        np.full((256_000, 2), round(0.2 * 32767), dtype=np.int16),
    )
    return common


class _FakeRenderer:
    def render(self, plan) -> PreparedRapBar:
        frames = 128_000
        samples = np.full((frames, 2), 0.1 + plan.bar * 0.01, dtype=np.float32)
        return PreparedRapBar(
            bar=plan.bar,
            text=plan.text,
            source=plan.source,
            fallback_reason=None,
            scheduled=plan.scheduled,
            audio=PcmAudio(AudioFormat(), frames, samples.tobytes()),
            diagnostics=(),
            warnings=(),
            render_latency_ms=1.25,
        )


def test_build_planned_bar_replays_the_recorded_flow_schedule(load_script) -> None:
    module = load_script("render_robotic_full_song")

    plan = module.build_planned_bar(_record(3), total_bars=50)

    assert tuple(item.slot.tick for item in plan.scheduled) == tuple(48 + tick for tick in TICKS)
    assert tuple(item.slot.accent for item in plan.scheduled) == tuple(index / 10 for index in range(9))
    assert plan.scheduled[2].syllable.word == "flicker"
    assert plan.scheduled[2].syllable.index_in_word == 0
    assert plan.scheduled[2].syllable.syllable_count == 2
    assert plan.scheduled[3].syllable.index_in_word == 1
    assert plan.scheduled[3].syllable.phonemes == ("ER0",)


def test_render_song_writes_named_full_length_artifacts_with_supplied_drums(
    load_script,
    tmp_path: Path,
) -> None:
    module = load_script("render_robotic_full_song")
    input_dir = tmp_path / "input"
    common = _write_input(input_dir)
    output_dir = tmp_path / "output"

    stats = module.render_song(
        input_dir=input_dir,
        output_dir=output_dir,
        song_id=SONG_ID,
        expected_bars=2,
        renderer=_FakeRenderer(),
    )

    vocals_path = output_dir / "concrete_court_robotic_vocals.wav"
    mix_path = output_dir / "concrete_court_robotic_mix.wav"
    log_path = output_dir / "concrete_court_robotic_render_log.jsonl"
    stats_path = output_dir / "concrete_court_robotic_stats.json"
    vocal_rate, vocals = wavfile.read(vocals_path)
    mix_rate, mix = wavfile.read(mix_path)

    assert vocal_rate == mix_rate == 48_000
    assert vocals.shape == (256_000,)
    assert mix.shape == (256_000, 2)
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(stats_path.read_text(encoding="utf-8")) == stats
    assert stats["inputs"]["drums_sha256"] == file_sha256(common / "drums.wav")
    assert stats["artifacts"]["mix"] == str(mix_path.resolve())
    assert stats["frame_count"] == 256_000
    assert stats["bar_count"] == 2
