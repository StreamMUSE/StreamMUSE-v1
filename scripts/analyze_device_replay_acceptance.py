"""Read-only analysis of the isolated device/file replay acceptance artifacts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def event_key(event):
    return event["type"], event["pitch"], event["tick"]


def first_difference(left, right):
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            return {"index_zero_based": index, "left": a, "right": b}
    return None


def analyze(root):
    # No checkpoint is supplied: instantiate the production encoder on CPU only.
    from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_engine import LekaiPromptEngine
    import mido

    client = root / "client_initialized"
    labels = {"A": "A_virtual_device", "B": "B_midi_replay", "C": "C_midi_replay_repeat"}
    sessions = {k: next((client / v / "logs").glob("*/*")) for k, v in labels.items()}
    events = {k: read(p / "prompt_continuation_replay_melody.json")["events"] for k, p in sessions.items()}
    traces = {k: read(p / "prompt_continuation_model_trace.json") for k, p in sessions.items()}
    prompts = {k: t["prompt_generation_log"]["prompt_tokens"] for k, t in traces.items()}
    output = {"session_names": {k: p.name for k, p in sessions.items()},
              "event_counts": {k: len(v) for k, v in events.items()}, "scope": "one automated source; not human acceptance"}
    output["trace_capture_complete"] = {k: t["trace_capture_complete"] for k, t in traces.items()}

    ca, cb = (Counter(map(event_key, events[k])) for k in ("A", "B"))
    missing = ca - cb
    output["A_only_events_ignoring_velocity"] = [{"event": k, "count": v} for k, v in missing.items()]
    output["B_only_events_ignoring_velocity"] = [{"event": k, "count": v} for k, v in (cb - ca).items()]
    filtered = []
    remaining = missing.copy()
    for e in events["A"]:
        if remaining[event_key(e)] > 0:
            remaining[event_key(e)] -= 1
        else:
            filtered.append(e)
    output["remove_missing_events_equals_B_order_ignoring_velocity"] = list(map(event_key, filtered)) == list(map(event_key, events["B"]))

    engine = LekaiPromptEngine()
    rebuilt = {k: engine._build_melody_prompt_tokens(v, 0, 32, bpm=80)[0].tolist() for k, v in events.items()}
    output["prompt_rebuild_matches_capture"] = {k: rebuilt[k] == prompts[k] for k in events}
    output["filtered_A_prompt_equals_B"] = engine._build_melody_prompt_tokens(filtered, 0, 32, bpm=80)[0].tolist() == prompts["B"]
    output["prompt_first_token_difference"] = first_difference(prompts["A"], prompts["B"])
    rolls = {k: engine._converter.events_to_pianoroll(v, 0, 32) for k, v in events.items()}
    output["prompt_pitch70_sustain_ticks"] = {k: rolls[k][0, 70 - 21].nonzero()[0].tolist() for k in rolls}
    output["prompt_pitch70_onset_ticks"] = {k: rolls[k][1, 70 - 21].nonzero()[0].tolist() for k in rolls}
    output["short_note_input_timing"] = [q for q in lines(sessions["A"] / "input_quantization_trace.jsonl")
                                         if (q["pitch"], q["quantized_tick"]) in {(70, 5), (61, 33)}]

    midi_events = []
    midi = mido.MidiFile(sessions["A"] / "prompt_continuation_replay_melody.mid")
    tick = 0
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type in ("note_on", "note_off"):
            midi_events.append({"type": "note_off" if message.type == "note_on" and message.velocity == 0 else message.type,
                                "pitch": message.note, "tick": tick * 4 / midi.ticks_per_beat})
    output["A_exported_midi_event_count"] = len(midi_events)
    output["A_exported_midi_matches_B_events_ignoring_velocity_and_order"] = Counter(map(event_key, midi_events)) == cb

    playback_notes = {}
    for k, session in sessions.items():
        midi = mido.MidiFile(session / "combined.mid")
        tracks = [t for t in midi.tracks if t.name == "Accompaniment"]
        assert len(tracks) == 1
        active, notes, tick = {}, [], 0
        for message in tracks[0]:
            tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                assert message.note not in active
                active[message.note] = (tick, message.velocity)
            elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
                start = active.pop(message.note, None)
                if start is not None:
                    notes.append((message.note, start[0] * 4 / midi.ticks_per_beat,
                                  tick * 4 / midi.ticks_per_beat, start[1]))
        assert not active
        playback_notes[k] = Counter(notes)
    output["combined_accompaniment_note_counts"] = {k: sum(v.values()) for k, v in playback_notes.items()}
    output["combined_B_only_notes"] = list((playback_notes["B"] - playback_notes["C"]).items())
    output["combined_C_only_notes"] = list((playback_notes["C"] - playback_notes["B"]).items())

    observed = defaultdict(list)
    for item in lines(root / "server" / "sampling_observer.jsonl"):
        observed[item["session_epoch"]].append(item)
    output["observer_epoch_call_counts"] = {k: len(v) for k, v in observed.items()}
    for left, right in (("A", "B"), ("B", "C")):
        full_a, full_b = [traces[k]["continuation_generations"] for k in (left, right)]
        by_tick = [{item["generation_start_tick"]: item for item in records} for records in (full_a, full_b)]
        assert len(by_tick[0]) == len(full_a) and len(by_tick[1]) == len(full_b)
        common_ticks = sorted(by_tick[0].keys() & by_tick[1].keys())
        a, b = [[records[t] for t in common_ticks] for records in by_tick]
        counts = {field: sum(x[field] == y[field] for x, y in zip(a, b))
                  for field in ("part0_tokens", "prompt_token_digest", "raw_tokens", "output_event_digest")}
        pair = {"left_calls": len(full_a), "right_calls": len(full_b), "equal_call_counts": counts,
                "compared_generation_ticks": common_ticks,
                "left_only_generation_ticks": sorted(by_tick[0].keys() - by_tick[1].keys()),
                "right_only_generation_ticks": sorted(by_tick[1].keys() - by_tick[0].keys()),
                "different_generation_ticks": {field: [t for t, u, v in zip(common_ticks, a, b) if u[field] != v[field]]
                    for field in counts},
                "prompt_input_exact": prompts[left] == prompts[right],
                "prompt_output_exact": traces[left]["prompt_generation_log"]["new_tokens"] == traces[right]["prompt_generation_log"]["new_tokens"]}
        epochs = [read(sessions[k] / "prompt_continuation_session_seed.json")["session_epoch"] for k in (left, right)]
        full_x, full_y = [observed[e] for e in epochs]
        assert len(full_x) == len(full_a) and len(full_y) == len(full_b)
        observed_by_tick = [dict(zip([v["generation_start_tick"] for v in records], samples))
                            for records, samples in ((full_a, full_x), (full_b, full_y))]
        x, y = [[samples[t] for t in common_ticks] for samples in observed_by_tick]
        pair["observer_equal_call_counts"] = {field: sum(u[field] == v[field] for u, v in zip(x, y))
                                               for field in ("prompt_tokens", "sampling", "returned_tokens")}
        for field in ("rng_before_sha256", "logits_sha256"):
            pair["observer_equal_call_counts"][field] = sum(
                [s[field] for s in u["samples"]] == [s[field] for s in v["samples"]] for u, v in zip(x, y))
        pair["observer_token_counts"] = [sum(len(v["samples"]) for v in run) for run in (x, y)]
        pair["observer_token_count_scope"] = "compared_generation_ticks"
        pair["observer_total_token_counts"] = [sum(len(v["samples"]) for v in run) for run in (full_x, full_y)]
        played = []
        for k in (left, right):
            played.append([(e["tick"], e["type"], e["data"]["pitch"], e["data"].get("velocity", 0))
                           for e in lines(sessions[k] / "events.jsonl")
                           if e["data"].get("source") == "model" and e["type"] in ("note_on", "note_off")])
        pair["playback_event_counts"] = list(map(len, played))
        pair["playback_events_exact"] = played[0] == played[1]
        pair["playback_first_difference"] = first_difference(*played)
        pair["left_only_playback_events"] = list((Counter(played[0]) - Counter(played[1])).items())
        pair["right_only_playback_events"] = list((Counter(played[1]) - Counter(played[0])).items())
        output[left + "_vs_" + right] = pair
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
