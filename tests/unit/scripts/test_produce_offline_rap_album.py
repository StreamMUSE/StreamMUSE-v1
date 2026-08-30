"""Tests for the resumable offline rap-album production command."""

from __future__ import annotations

import pytest


def test_default_campaign_is_ten_unique_fifty_bar_songs(load_script, tmp_path) -> None:
    script = load_script("produce_offline_rap_album")
    args = script.build_parser().parse_args(["--output-dir", str(tmp_path)])

    assert len(script.DEFAULT_SONGS) == 10
    assert len({song.topic for song in script.DEFAULT_SONGS}) == 10
    assert len({song.slug for song in script.DEFAULT_SONGS}) == 10
    assert args.bars == 50
    assert args.tempo == 90.0
    assert args.choices == 64
    assert args.minimum_stress == 0.60


def test_parse_choice_payload_deduplicates_normalized_one_line_choices(load_script) -> None:
    script = load_script("produce_offline_rap_album")
    body = {
        "choices": [
            {"index": 2, "message": {"content": "2. Bright stars move through night"}, "finish_reason": "stop"},
            {"index": 0, "message": {"content": "Bright stars move through night"}, "finish_reason": "stop"},
            {"index": 1, "message": {"content": "Cold moons glide beyond sight\nignored"}, "finish_reason": "length"},
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    parsed = script.parse_choice_payload(body)

    assert parsed.lines == (
        "Bright stars move through night",
        "Cold moons glide beyond sight",
    )
    assert parsed.raw_choice_count == 3
    assert parsed.finish_reasons == {"length": 1, "stop": 2}
    assert parsed.prompt_tokens == 10
    assert parsed.completion_tokens == 20


def test_validate_chosen_records_requires_one_contiguous_record_per_bar(load_script) -> None:
    script = load_script("produce_offline_rap_album")

    script.validate_chosen_records([{"bar": 0}, {"bar": 1}], expected_bars=2)
    with pytest.raises(ValueError, match="contiguous bars"):
        script.validate_chosen_records([{"bar": 0}, {"bar": 2}], expected_bars=2)


def test_fifty_bars_at_ninety_bpm_have_exact_expected_frame_count(load_script) -> None:
    script = load_script("produce_offline_rap_album")

    assert script.expected_song_frames(bars=50, tempo_bpm=90.0, sample_rate_hz=48_000) == 6_400_000


def test_offline_prompt_requests_fresh_structure_and_narrative_focus(load_script) -> None:
    script = load_script("produce_offline_rap_album")
    request = script.CandidateRequest(
        request_id="test",
        target_bar=20,
        topic="space exploration",
        flow_template=script.BUILTIN_TEMPLATES.get("baseline_straight_9"),
        count=1,
        context_lines=("space dreams ignite beyond stars",),
        seed=1,
    )

    messages = script.offline_messages(
        request,
        total_bars=50,
        blocked_openings=frozenset({"neon"}),
        blocked_closings=frozenset({"rise"}),
    )

    assert "fresh image and action" in messages[-1]["content"]
    assert "obstacle" in messages[-1]["content"]
    assert "Do not start with these overused words: neon" in messages[-1]["content"]
    assert "Do not end with these overused words: rise" in messages[-1]["content"]


def test_audio_stage_requires_rubberband_before_creating_artifacts(
    load_script,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    script = load_script("produce_offline_rap_album")
    output_dir = tmp_path / "album"
    monkeypatch.setattr(
        script.shutil,
        "which",
        lambda executable: None if executable == "rubberband" else f"/usr/bin/{executable}",
    )

    with pytest.raises(OSError, match="rubberband executable"):
        script.main(["--output-dir", str(output_dir), "--stage", "audio"])

    assert not output_dir.exists()
