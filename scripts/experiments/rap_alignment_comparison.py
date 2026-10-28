"""Fetch one real v3 turn, then compare local policies on identical source audio."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from queue import Queue, Empty
import time
import uuid

import numpy as np
from scipy.io import wavfile

from streammuse.application.rap.bar_audio import BarAudioPreparer
from streammuse.domain.rap.audio import AudioFormat
from streammuse.domain.rap.lyric_stream import BarLyricRecord, LYRIC_STREAM_SCHEMA_VERSION
from streammuse.domain.rap.remote_chunk import _flow_payload
from streammuse.domain.timing import Tempo
from streammuse.infrastructure.rap.drums import ProceduralBoomBapRenderer
from streammuse.infrastructure.rap.lyric_stream_client import HttpBarAudioSource, LyricStreamConnection
from streammuse.infrastructure.rap.phrase_warp import RubberBandPhraseWarper, probe_rubberband_r3
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


def compare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    template = BUILTIN_TEMPLATES.get(args.template)
    session = "alignment-" + uuid.uuid4().hex[:16]
    messages = Queue()
    connection = LyricStreamConnection(args.url)
    source = HttpBarAudioSource(args.url)
    records = {}
    captured = []
    started = time.monotonic()
    try:
        connection.open(session, {
            "type": "start", "schema_version": LYRIC_STREAM_SCHEMA_VERSION,
            "capabilities": ["bounded_turns_v1"], "tempo_bpm": args.bpm,
            "start_bar": 0, "seed": 20261028, "loop": False,
            "turn_id": session, "end_bar_exclusive": args.bars,
            "segments": [{"start_bar": 0, "bars": args.bars, "topic": args.topic,
                          "flow_template": _flow_payload(template)}],
            "edits": [], "history": [], "lyric_lead": args.bars, "audio_lead": args.bars,
        }, messages.put)
        while len(records) < args.bars:
            try:
                message = messages.get(timeout=180)
            except Empty as exc:
                raise TimeoutError("No complete lyric turn within 180 seconds") from exc
            captured.append(message)
            if message.get("type") == "error":
                raise RuntimeError(message)
            if message.get("type") == "lyric":
                record = BarLyricRecord.from_payload(message["record"])
                records[record.bar] = record
                print(f"LYRIC bar={record.bar} text={record.text!r}", flush=True)
        inputs = []
        for bar, record in sorted(records.items()):
            deadline = time.monotonic() + 180
            while True:
                fetched = source.fetch(session, bar, record.lyric_id, wait_ms=1000, accept_older=False)
                if fetched.status == "ready":
                    break
                if fetched.status != "pending" or time.monotonic() > deadline:
                    raise RuntimeError(f"Source unavailable: {fetched.status}")
            (args.output / f"{args.name}_bar_{bar:02}_source.wav").write_bytes(fetched.source_wav)
            (args.output / f"{args.name}_bar_{bar:02}_manifest.json").write_text(json.dumps(fetched.manifest.to_payload(), indent=2))
            inputs.append((record, fetched))
    finally:
        connection.close()
        source.close()
        (args.output / "stream_messages.json").write_text(json.dumps(captured, indent=2))
    source_elapsed = time.monotonic() - started
    version = probe_rubberband_r3()
    results = []
    for label, policy in (("A_relaxed", "gentle_sparse_r3"), ("B_strict", "all_onsets_r3"),
                          ("C_strict_stress", "all_onsets_stress_r3")):
        renderer = BarAudioPreparer(
            warper=RubberBandPhraseWarper(policy=policy, rubberband_version=version),
            drums=ProceduralBoomBapRenderer(seed=20261028), tempo=Tempo(args.bpm, 4, 4), audio_format=AudioFormat(),
        )
        mixed, vocals, evidence = [], [], []
        for record, fetched in inputs:
            prepared = renderer.prepare(record, fetched.manifest, fetched.source_wav)
            mixed.append(np.frombuffer(prepared.audio.data, dtype=np.float32).reshape(-1, 2))
            vocal = np.frombuffer(prepared.vocal_stem.data, dtype=np.float32).reshape(-1, 2)
            vocals.append(vocal)
            first_sample = prepared.onset_estimates[0].requested_sample
            prefix_peak = float(np.max(np.abs(vocal[:first_sample]))) if first_sample else None
            evidence.append({"bar": record.bar, "text": record.text, "render_ms": prepared.render_latency_ms,
                             "first_tick": record.flow_template.slots[0].tick_in_bar,
                             "leading_rest_peak": prefix_peak, "warp": asdict(prepared.warp_diagnostics),
                             "onsets": [asdict(item) for item in prepared.onset_estimates]})
        for stem, arrays in (("mix", mixed), ("vocals", vocals)):
            samples = np.concatenate(arrays)
            wavfile.write(args.output / f"{args.name}_{label}_{stem}.wav", 48000,
                          np.rint(np.clip(samples, -1, 1) * 32767).astype(np.int16))
        results.append({"label": label, "policy": policy, "bars": evidence})
        print(f"RENDER {label} complete", flush=True)
    report = {"topic": args.topic, "bpm": args.bpm, "template": _flow_payload(template),
              "source_generation_seconds": source_elapsed, "rubberband": version,
              "control": "Identical generated words, source PCM, source alignment, drums and mix gains; chords off",
              "caveat": "Onset errors are alignment/time-map estimates, not independently measured acoustic onsets",
              "results": results}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2))
    (args.output / f"{args.name}_lyrics.txt").write_text("\n".join(record.text for record, _ in inputs) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18720")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--name", default="night_train")
    parser.add_argument("--topic", default="a night train crossing a quiet city")
    parser.add_argument("--bpm", type=float, default=60)
    parser.add_argument("--bars", type=int, default=2)
    compare(parser.parse_args())
