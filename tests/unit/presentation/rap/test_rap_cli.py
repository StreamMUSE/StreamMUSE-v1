"""Tests for the rap-alignment prototype command surface."""

from __future__ import annotations

import json
from pathlib import Path

from streammuse.application.rap.service import RapPrototypeService
from streammuse.application.config import ApplicationConfig, RapConfig
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.generators import PhraseBankGenerator
from streammuse.presentation.cli.cli import _build_rap_controller
from streammuse.presentation.rap.cli import main, plan_to_dict, play_plan


def _plan():
    return RapPrototypeService(
        Tempo(bpm=120, ticks_per_beat=4, beats_per_bar=4),
        "boom_bap",
        PhraseBankGenerator(),
    ).build_plan("space travel", bars=1, candidate_count=8)


def test_default_cli_run_writes_an_inspectable_schedule_json(tmp_path: Path, capsys) -> None:
    output = tmp_path / "plan.json"

    assert main(["--topic", "space travel", "--bars", "1", "--output-json", str(output)]) == 0

    payload = json.loads(output.read_text())
    assert payload["pattern"] == "boom_bap"
    assert payload["lines"][0]["events"][0]["tick"] == 0
    assert payload["lines"][0]["events"][0]["label"]
    assert "Bar 1" in capsys.readouterr().out


def test_plan_to_dict_preserves_syllable_and_timing_metadata() -> None:
    payload = plan_to_dict(_plan())
    event = payload["lines"][0]["events"][0]

    assert event["bar"] == 0
    assert event["beat"] == 0
    assert event["tick_in_beat"] == 0
    assert event["seconds"] == 0.0
    assert event["word"]
    assert isinstance(event["stressed"], bool)


def test_play_plan_emits_the_existing_schedule_at_tick_offsets() -> None:
    writes: list[str] = []
    sleeps: list[float] = []

    play_plan(_plan(), write=writes.append, clock=lambda: 0.0, sleep=sleeps.append)

    assert writes[0].startswith("[B1 1.1]")
    assert len(writes) == len(_plan().events)
    assert sleeps
    assert all(delay > 0 for delay in sleeps)


def test_existing_main_cli_builds_the_scenario_aware_controller() -> None:
    controller = _build_rap_controller(
        ApplicationConfig(rap=RapConfig(topic="space", pattern="boom_bap")),
        Tempo(92.0, 4, 4),
    )

    assert controller is not None
    assert controller.scenario.segment_for_bar(0).topic == "space"
    assert controller.scenario.segment_for_bar(0).template_id == "baseline_syncopated_9"
    controller.start()
    controller.on_tick(0)
    controller.close()
