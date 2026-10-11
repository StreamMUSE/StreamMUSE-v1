from __future__ import annotations

import json

import pytest

from scripts.midi_to_npz import midi_pair_to_npz
from scripts.compare_model_output_traces import compare_model_output_records, load_trace
from tests.consistency.conftest import LekaiServer, NPZ_DIR, song_spec
from tests.consistency.runners import run_offline_model_output_trace, run_realtime


pytestmark = pytest.mark.consistency


def test_streammuse_round_outputs_match_offline_greedy(
    lekai_server: LekaiServer,
    lekai_checkpoint,
    artifacts_dir,
):
    song = song_spec(4)
    streammuse_input_trace = lekai_server.model_input_trace_path
    streammuse_trace = lekai_server.model_output_trace_path
    offline_input_trace = artifacts_dir / "offline_model_inputs_song4.jsonl"
    offline_trace = artifacts_dir / "offline_model_outputs_song4.jsonl"
    npz_dir = NPZ_DIR
    if not (npz_dir / "4.npz").exists():
        npz_dir = artifacts_dir / "reconstructed_npz"
        npz_dir.mkdir(parents=True, exist_ok=True)
        assert midi_pair_to_npz(
            str(song.mel_path),
            str(song.mel_path.parent.parent / "acc" / song.mel_path.name),
            str(npz_dir / "4.npz"),
        )
    streammuse_trace.write_text("", encoding="utf-8")
    streammuse_input_trace.write_text("", encoding="utf-8")

    run_realtime(
        lekai_server,
        song,
        tempo=120,
        out_dir=artifacts_dir / "round_output_realtime",
    )
    run_offline_model_output_trace(
        lekai_checkpoint,
        npz_dir / "4.npz",
        trace_path=offline_trace,
        input_trace_path=offline_input_trace,
    )

    report = compare_model_output_records(
        load_trace(streammuse_trace),
        load_trace(offline_trace),
    )
    report_path = artifacts_dir / "round_model_output_comparison_song4.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert report["passed"], f"per-beat model outputs differ; report={report_path}"
