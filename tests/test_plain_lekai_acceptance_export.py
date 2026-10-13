"""The audit must replay the exported track without changing its timing."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import mido
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("plain_audit", SCRIPTS / "plain_lekai_device_replay_acceptance.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_extract_melody_retains_track_messages_and_tempo(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=220)
    midi.tracks = [
        mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=666667)]),
        mido.MidiTrack([mido.MetaMessage("track_name", name="Melody"),
                        mido.Message("note_on", note=65, velocity=71, time=55),
                        mido.Message("note_off", note=65, time=165)]),
        mido.MidiTrack([mido.MetaMessage("track_name", name="Accompaniment"),
                        mido.Message("note_on", note=48),
                        mido.Message("note_off", note=48, time=220)]),
    ]
    source, target = tmp_path / "combined.mid", tmp_path / "melody.mid"
    midi.save(source)
    audit.extract_melody(source, target)
    original, exported = mido.MidiFile(source), mido.MidiFile(target)
    assert exported.ticks_per_beat == original.ticks_per_beat
    assert exported.tracks == original.tracks[:2]
    assert audit.read(target.with_suffix(".json"))["source_sha256"] == audit.sha(source.read_bytes())


def test_no_guessing_when_melody_track_is_missing(tmp_path):
    source = tmp_path / "missing.mid"
    midi = mido.MidiFile()
    midi.tracks = [mido.MidiTrack()]
    midi.save(source)
    with pytest.raises(ValueError, match="Expected one Melody track"):
        audit.extract_melody(source, tmp_path / "out.mid")


def test_note_comparison_uses_exact_midi_positions(tmp_path):
    source = tmp_path / "music.mid"
    midi = mido.MidiFile(ticks_per_beat=220)
    midi.tracks = [mido.MidiTrack([
        mido.MetaMessage("track_name", name="Accompaniment"),
        mido.MetaMessage("set_tempo", tempo=666666),
        mido.Message("note_on", note=49, time=16 * 55, velocity=100),
        mido.Message("note_off", note=49, time=4 * 55),
    ])]
    midi.save(source)
    assert audit.midi_notes(source) == [(49, 16.0, 20.0, 100)]


def timed_record(sequence, elapsed, event_type):
    return {"event_sequence": sequence, "elapsed_seconds": elapsed,
            "application_received_time_s": 1000 + elapsed, "timeline_start_time_s": 1000,
            "clock_domain": "service_now", "bpm": 90, "ticks_per_beat": 4,
            "snap_forward_fraction": 0.4, "channel": 0, "event_type": event_type,
            "pitch": 65, "velocity": 80 if event_type == "note_on" else 0}


def test_timed_input_preserves_subtick_times_silence_and_order(tmp_path):
    trace = tmp_path / "trace.jsonl"
    records = [timed_record(1, 2.017, "note_on"), timed_record(2, 2.031, "note_off")]
    trace.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    original, events = audit.timed_messages(trace)
    assert original == records
    assert [seconds for seconds, _ in events] == [2.017, 2.031]
    assert [(m.type, m.velocity) for _, m in events] == [("note_on", 80), ("note_off", 0)]


def test_timed_input_rejects_missing_sequence(tmp_path):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(timed_record(2, 2.1, "note_on")), encoding="utf-8")
    with pytest.raises(AssertionError):
        audit.timed_messages(trace)


def test_sender_uses_formal_tick_zero_not_first_frame_observation(tmp_path):
    trace = {"record_type": "frame_deadline", "tick": 0, "clock_domain": "service_now",
             "nominal_tick_time_s": 1000.0, "observed_emit_time_s": 1000.03}
    (tmp_path / "system_trace.jsonl").write_text(json.dumps(trace), encoding="utf-8")
    assert audit.timeline_anchor(tmp_path) == 1000.0


def test_input_only_runs_selected_piece_without_export_replay(tmp_path, monkeypatch):
    import human6_timed_input_replay as cohort
    cases = []
    for piece in ("first", "second"):
        trace = tmp_path / f"{piece}.jsonl"
        trace.write_text("{}", encoding="utf-8")
        cases.append({"piece": piece, "trace": trace.name, "stop_tick": 40,
                      "source_sha256": {"input_quantization_trace.jsonl": audit.sha(trace.read_bytes())}})
    audit.save(tmp_path / "plan.json", {"cases": cases, "condition": "plain",
                                      "system_ref": "test", "seed": 123})
    called = []
    monkeypatch.setattr(cohort.audit, "identity", lambda ref: {"base": ref})
    monkeypatch.setattr(cohort.audit, "one_client", lambda args, label: called.append((args, label)))
    monkeypatch.setattr(cohort.audit, "clients", lambda args: pytest.fail("Unexpected MIDI replay"))
    cohort.run(SimpleNamespace(root=tmp_path, pieces=["second"], input_only=True,
                               server="http://127.0.0.1:18845"))
    assert len(called) == 1
    args, label = called[0]
    assert args.timed_input == tmp_path / "second.jsonl"
    assert args.seed == 123 and args.stop_tick == 40
    assert label == "A_virtual_device"
    assert audit.read(tmp_path / "run_contract.json")["pieces"] == ["second"]
    assert audit.read(tmp_path / "progress.json")["status"] == "completed"


def quantized_record(sequence, elapsed, event_type):
    from streammuse.application.services.input_timing import diagnose_input_quantization
    from streammuse.domain.timing import Tempo
    record = timed_record(sequence, elapsed, event_type)
    record["quantized_tick"] = diagnose_input_quantization(
        elapsed, Tempo(90, 4, 4), snap_forward_fraction=0.4).quantized_tick
    return record


def test_input_summary_reports_zero_length_transition_and_orphan_separately():
    from summarize_human_input_timing import compare
    old = [quantized_record(1, 0.01, "note_off"), quantized_record(2, 0.09844, "note_on"),
           quantized_record(3, 0.18, "note_off")]
    new = [quantized_record(1, 0.01, "note_off"), quantized_record(2, 0.10249, "note_on"),
           quantized_record(3, 0.182, "note_off")]
    summary, details = compare(old, new)
    assert summary["changed_note_on"] == 1 and summary["changed_note_off"] == 0
    assert summary["positive_to_zero_tick_notes"] == 1
    assert summary["paired_notes"] == 1 and summary["orphan_note_off_count"] == 1
    assert details["orphan_note_off_sequences"] == [1]


def test_input_summary_distinguishes_shifted_note_from_changed_duration():
    from summarize_human_input_timing import compare
    old = [quantized_record(1, 0.09844, "note_on"), quantized_record(2, 0.26511, "note_off")]
    new = [quantized_record(1, 0.10249, "note_on"), quantized_record(2, 0.26916, "note_off")]
    summary, _ = compare(old, new)
    assert summary["changed_events"] == 2
    assert summary["changed_note_intervals"] == 1
    assert summary["changed_durations"] == 0


def test_input_summary_rejects_pitch_mismatch():
    from summarize_human_input_timing import compare
    old = [quantized_record(1, 0.1, "note_on"), quantized_record(2, 0.2, "note_off")]
    new = [dict(r, pitch=66) for r in old]
    with pytest.raises(ValueError, match="sequence/content mismatch"):
        compare(old, new)
