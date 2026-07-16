"""Golden end-to-end consistency test: realtime simulation == offline generation.

Premise (verified Phase 0, see the plan): with greedy sampling (top_k=1, temperature=0) and
the BPM token pinned identically on both sides, the real RealTimeMusicService driven by a
MIDI-file input produces the SAME accompaniment as the offline one-shot generator — at the
pianoroll level, within the song's melody window.

Two things make the comparison meaningful (all the hard-won Phase 0 lessons):
  * non-empty songs only — greedy collapses many songs to silence (songs 4 and 5 are defaults);
  * pianoroll (beat, pitch) comparison, not raw MIDI notes — realtime re-triggers sustained
    notes per beat while offline keeps them long;
  * truncate to the melody window — realtime runs a bit past the song end with no melody.

Tempo ladder separates "consistency regression" from "inference too slow":
  * tempo 15 is the gold rung — so slow that inference can never miss its schedule;
  * tempo 120 additionally checks the machine can keep up in real time.
A failure at tempo 15 is a real regression; tempo-15-green/tempo-120-red means the box was
too slow (also caught earlier as a dropped-request precondition failure), not a bug.

Opt-in (needs GPU + checkpoint); the default runs songs 4 and 5 at both tempo rungs::

    LEKAI_CHECKPOINT_PATH=models/ModelLekai/epoch_4_1104_1204/model.safetensors \
      uv run pytest tests/consistency/ -v

Scale up before a release::

    STREAMMUSE_CONSISTENCY_SONGS=4,5,2 STREAMMUSE_CONSISTENCY_TEMPOS=120,90,60,15 \
    LEKAI_CHECKPOINT_PATH=... uv run pytest tests/consistency/ -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.consistency.conftest import LekaiServer, song_spec
from tests.consistency.midi_pianoroll import accompaniment_active_cells, compare_accompaniment
from tests.consistency.runners import count_dropped_requests, run_offline, run_realtime

pytestmark = pytest.mark.consistency

DEFAULT_SONGS = "4,5"
DEFAULT_TEMPOS = "15,120"
LATE_SCHEDULE_POLICIES = {
    "clamped_partial_note",
    "clamped_partial_note_off",
    "clamped_open_note",
    "dropped_past_note",
    "dropped_past_placeholder",
    "late_isolated_note_off",
    "late_placeholder",
    "forced_note_off",
}


def _parse_int_list(env_name: str, default: str) -> list[int]:
    raw = os.environ.get(env_name, default)
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


def _songs() -> list[int]:
    return _parse_int_list("STREAMMUSE_CONSISTENCY_SONGS", DEFAULT_SONGS)


def _tempos() -> list[int]:
    return _parse_int_list("STREAMMUSE_CONSISTENCY_TEMPOS", DEFAULT_TEMPOS)


def _late_schedule_policy_rows(session_dir: Path) -> list[tuple[int, dict[str, object]]]:
    trace_path = session_dir / "model_schedule_trace.jsonl"
    assert trace_path.exists(), f"missing model schedule trace: {trace_path}"

    rows: list[tuple[int, dict[str, object]]] = []
    for line_no, line in enumerate(trace_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("policy") in LATE_SCHEDULE_POLICIES:
            rows.append((line_no, row))
    return rows


def _format_trace_rows(rows: list[tuple[int, dict[str, object]]], *, limit: int = 5) -> str:
    formatted = []
    for line_no, row in rows[:limit]:
        formatted.append(
            f"line {line_no}: policy={row.get('policy')} "
            f"logical={row.get('logical_tick')} scheduled={row.get('scheduled_tick')} "
            f"pitch={row.get('pitch')} action={row.get('action')}"
        )
    if len(rows) > limit:
        formatted.append(f"... {len(rows) - limit} more")
    return "\n".join(formatted)


@pytest.mark.parametrize("song_number", _songs())
def test_realtime_matches_offline(
    song_number: int,
    lekai_server: LekaiServer,
    lekai_checkpoint: Path,
    artifacts_dir: Path,
) -> None:
    song = song_spec(song_number)
    window = song.melody_last_beat  # truncate comparison to where melody (conditioning) exists

    # --- offline reference (one greedy pass) ---
    offline_midi = run_offline(lekai_checkpoint, song, out_dir=artifacts_dir)
    offline_cells = accompaniment_active_cells(offline_midi, max_beat=window)
    assert offline_cells, (
        f"song {song_number} produced an empty offline accompaniment under greedy — pick a "
        f"non-empty song (see plan appendix B; song 4/5/2 are non-empty)."
    )

    # --- realtime across the tempo ladder ---
    realtime_cells_by_tempo: dict[int, set[tuple[int, int]]] = {}
    for tempo in _tempos():
        session_dir = run_realtime(lekai_server, song, tempo=tempo, out_dir=artifacts_dir)

        dropped = count_dropped_requests(session_dir)
        assert dropped == 0, (
            f"run invalid: inference too slow at tempo {tempo} for song {song_number} "
            f"({dropped} stale request(s) dropped). This is a performance issue, not a "
            f"consistency regression — rerun on an idle GPU or at a slower tempo. "
            f"Session: {session_dir}"
        )

        late_rows = _late_schedule_policy_rows(session_dir)
        assert not late_rows, (
            f"run invalid: inference responses arrived after their target playback ticks at "
            f"tempo {tempo} for song {song_number}. This triggered late scheduling policy, "
            f"so this is a performance issue, not a consistency regression.\n"
            f"{_format_trace_rows(late_rows)}\n"
            f"Session: {session_dir}"
        )

        cmp = compare_accompaniment(
            session_dir / "combined.mid", offline_midi, max_beat=window
        )
        assert cmp.is_consistent, (
            f"song {song_number} tempo {tempo}: realtime != offline within beat<{window}.\n"
            f"{cmp.summary()}\n"
            f"realtime session: {session_dir}\noffline midi: {offline_midi}"
        )
        realtime_cells_by_tempo[tempo] = accompaniment_active_cells(
            session_dir / "combined.mid", max_beat=window
        )

    # --- cross-tempo: clock speed must not leak into generated content ---
    tempos = list(realtime_cells_by_tempo)
    reference = realtime_cells_by_tempo[tempos[0]]
    for tempo in tempos[1:]:
        assert realtime_cells_by_tempo[tempo] == reference, (
            f"song {song_number}: realtime output differs between tempo {tempos[0]} and "
            f"{tempo} — wall-clock speed leaked into generation."
        )
