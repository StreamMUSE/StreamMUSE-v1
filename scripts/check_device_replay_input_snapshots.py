"""Reconstruct captured Melody arrays from protocol history, without inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from streammuse.infrastructure.inference.lekai_http_backend import LekaiHttpBackend
from streammuse.infrastructure.inference.lekai_model.MidiConverter import MidiConverter
from streammuse.infrastructure.inference.lekai_prompt_continuation.token_conversion import copy_events


def check(root):
    result = {"scope": "CPU reconstruction of actually captured input snapshots", "windows": {}}
    data = {}
    backend = LekaiHttpBackend()
    converter = MidiConverter(ticks_per_beat=4)
    for key, label in (("A", "A_virtual_device"), ("B", "B_midi_replay")):
        session = next((root / "client_initialized" / label / "logs").glob("*/*"))
        trace = json.loads((session / "prompt_continuation_model_trace.json").read_text())
        requests = [json.loads(line) for line in (session / "prompt_continuation_replay_requests.jsonl").read_text().splitlines()]
        history, prefixes = [], {}
        for request in requests:
            history.extend(copy_events(request["request"].get("melody_events", [])))
            digest = backend._canonical_sha256(history)
            entry = prefixes.setdefault(digest, {"events": list(history), "matching_prefixes": []})
            entry["matching_prefixes"].append({"sequence": request["sequence"],
                                               "observed_until_tick": request["request"].get("observed_until_tick")})
        data[key] = {"history": history, "prefixes": prefixes,
                     "calls": {v["generation_start_tick"]: v for v in trace["continuation_generations"]}}

    common = sorted(data["A"]["calls"].keys() & data["B"]["calls"].keys())
    bad_ticks = [t for t in common if data["A"]["calls"][t]["part0_tokens"] != data["B"]["calls"][t]["part0_tokens"]]
    for tick in bad_ticks:
        window, rolls = {}, {}
        for key in ("A", "B"):
            call = data[key]["calls"][tick]
            snapshot = data[key]["prefixes"][call["input_cumulative_digest"]]
            start, end = call["part0_roll_start_tick"], call["part0_roll_end_tick"]
            roll = converter.events_to_pianoroll(snapshot["events"], start, end,
                active_pitches=backend._active_pitches_before_tick(snapshot["events"], start))
            matches = hashlib.sha256(roll.tobytes()).hexdigest() == call["part0_roll_bytes_sha256"]
            assert matches
            rolls[key] = roll
            window[key] = {"roll_matches_capture": matches, "window_start_tick": start, "window_end_tick": end,
                           "matching_protocol_history_prefixes": snapshot["matching_prefixes"],
                           "snapshot_event_count": len(snapshot["events"]), "last_events": snapshot["events"][-6:]}
        assert window["A"]["window_start_tick"] == window["B"]["window_start_tick"]
        window["different_cells"] = [{"channel": "sustain" if c == 0 else "onset", "pitch": int(p + 21),
            "tick": int(t + window["A"]["window_start_tick"]), "A": int(rolls["A"][c, p, t]),
            "B": int(rolls["B"][c, p, t])} for c, p, t in np.argwhere(rolls["A"] != rolls["B"])]
        result["windows"][tick] = window

    full = {k: converter.events_to_pianoroll(v["history"], 0, 160) for k, v in data.items()}
    result["full_recorded_history_rolls_equal"] = bool(np.array_equal(full["A"], full["B"]))
    result["different_captured_melody_windows"] = bad_ticks
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}))
