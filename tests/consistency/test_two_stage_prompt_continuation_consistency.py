"""Two-stage prompt+continuation consistency: realtime simulation == offline driver."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.consistency.conftest import LekaiServer, song_spec
from tests.consistency.two_stage_runners import (
    PROMPT_LENGTH_TICKS,
    continuation_counter,
    count_dropped_and_clipped,
    counter_summary,
    event_counter,
    load_event_history,
    run_two_stage_offline,
    run_two_stage_realtime,
)

pytestmark = pytest.mark.consistency

DEFAULT_SONGS = "4"
DEFAULT_TEMPOS = "15,120"


def _parse_int_list(env_name: str, default: str) -> list[int]:
    raw = os.environ.get(env_name, default)
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


def _songs() -> list[int]:
    return _parse_int_list("STREAMMUSE_TWO_STAGE_CONSISTENCY_SONGS", os.environ.get("STREAMMUSE_CONSISTENCY_SONGS", DEFAULT_SONGS))


def _tempos() -> list[int]:
    return _parse_int_list("STREAMMUSE_TWO_STAGE_CONSISTENCY_TEMPOS", os.environ.get("STREAMMUSE_CONSISTENCY_TEMPOS", DEFAULT_TEMPOS))


@pytest.mark.parametrize("song_number", _songs())
def test_two_stage_realtime_matches_offline(
    song_number: int,
    two_stage_lekai_server: LekaiServer,
    lekai_prompt_checkpoint: Path,
    lekai_continuation_checkpoint: Path,
    artifacts_dir: Path,
) -> None:
    song = song_spec(song_number)
    window = song.melody_last_beat
    out_dir = artifacts_dir / "two_stage"
    out_dir.mkdir(parents=True, exist_ok=True)

    offline = run_two_stage_offline(
        lekai_prompt_checkpoint,
        lekai_continuation_checkpoint,
        song,
        out_dir=out_dir,
    )
    offline_prompt_events = load_event_history(offline.song_dir / "stage1_prompt_history.json")
    offline_raw_events = load_event_history(offline.song_dir / "stage2_raw_history.json")
    comparison_end_tick = int(window) * 4

    offline_prompt_context = event_counter(offline_prompt_events, max_tick=PROMPT_LENGTH_TICKS)
    offline_continuation_context = continuation_counter(
        offline_raw_events,
        offline_prompt_events,
        max_tick=comparison_end_tick,
    )
    assert offline_prompt_context, (
        f"song {song_number} produced empty offline prompt context. summary={offline.summary_path}"
    )
    assert offline_continuation_context, (
        f"song {song_number} produced empty offline continuation context. summary={offline.summary_path}"
    )

    realtime_continuation_by_tempo = {}
    for tempo in _tempos():
        realtime = run_two_stage_realtime(
            two_stage_lekai_server,
            song,
            tempo=tempo,
            out_dir=out_dir,
        )
        schedule_counts = count_dropped_and_clipped(realtime.trace_path)
        status_payload = json.loads(realtime.history_status_path.read_text())
        raw_status = dict(status_payload.get("raw_status", {}))
        assert not raw_status.get("is_failed"), (
            f"server raw history reports failed scheduler at tempo {tempo}: {raw_status}. "
            f"session={realtime.session_dir}"
        )
        assert int(raw_status.get("accompaniment_history_beats", 0) or 0) >= int(window), (
            f"server raw history does not cover comparison window beat<{window} at tempo {tempo}: "
            f"{raw_status}. session={realtime.session_dir}"
        )

        realtime_prompt_events = load_event_history(realtime.prompt_history_path)
        realtime_raw_events = load_event_history(realtime.raw_history_path)
        realtime_prompt_context = event_counter(realtime_prompt_events, max_tick=PROMPT_LENGTH_TICKS)
        realtime_continuation_context = continuation_counter(
            realtime_raw_events,
            realtime_prompt_events,
            max_tick=comparison_end_tick,
        )

        assert realtime_prompt_context == offline_prompt_context, (
            f"song {song_number} tempo {tempo}: realtime prompt context != offline prompt context.\n"
            f"{counter_summary('realtime_prompt', realtime_prompt_context, 'offline_prompt', offline_prompt_context)}\n"
            f"realtime prompt history: {realtime.prompt_history_path}\n"
            f"offline prompt history: {offline.song_dir / 'stage1_prompt_history.json'}\n"
            f"schedule_counts (diagnostic only): {schedule_counts}"
        )
        assert realtime_continuation_context == offline_continuation_context, (
            f"song {song_number} tempo {tempo}: realtime continuation context != offline continuation context "
            f"within ticks [{PROMPT_LENGTH_TICKS}, {comparison_end_tick}).\n"
            f"{counter_summary('realtime_continuation', realtime_continuation_context, 'offline_continuation', offline_continuation_context)}\n"
            f"realtime raw history: {realtime.raw_history_path}\n"
            f"offline raw history: {offline.song_dir / 'stage2_raw_history.json'}\n"
            f"trace: {realtime.trace_path}\n"
            f"schedule_counts (diagnostic only): {schedule_counts}"
        )
        realtime_continuation_by_tempo[tempo] = realtime_continuation_context

    tempos = list(realtime_continuation_by_tempo)
    reference = realtime_continuation_by_tempo[tempos[0]]
    for tempo in tempos[1:]:
        assert realtime_continuation_by_tempo[tempo] == reference, (
            f"song {song_number}: two-stage realtime server continuation context differs between tempo "
            f"{tempos[0]} and {tempo}.\n"
            f"{counter_summary(str(tempos[0]), reference, str(tempo), realtime_continuation_by_tempo[tempo])}"
        )
