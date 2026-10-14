"""Warp the phase 0a sources onto the beat the way the client does.

Reads ``compare_bar_unit_synthesis.py`` output and, for every item and round,
writes two 24 kHz PCM16 WAVs of the same lyrics and exactly two bars:

- ``phrase.wav``: the two-bar MOSS phrase warped with R3 in one piece;
- ``bars.wav``: each one-bar MOSS line warped to its bar, then concatenated.

It also prints the MOSS timing comparison. Quality metrics come from
``score_bar_unit_audio.py`` on these files.

    uv run python scripts/analyze_bar_unit_synthesis.py <compare-out-dir>
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import statistics
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]


def _request(payload):
    from streammuse.experiments.rap_audio_protocols.contracts import SyllableTarget, TwoBarRenderRequest

    values = dict(payload)
    values.pop("duration_seconds", None)
    values["syllables"] = tuple(
        SyllableTarget(**{**item, "phonemes": tuple(item["phonemes"])}) for item in values["syllables"]
    )
    return TwoBarRenderRequest(**values)


def _pcm(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as handle:
        return np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")


def _write(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(samples.astype("<i2").tobytes())


def main() -> None:
    from streammuse.domain.rap import RemoteRapChunkRequest
    from streammuse.infrastructure.rap.phrase_warp import RubberBandPhraseWarper, probe_rubberband_r3

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("compare_out", type=Path)
    parser.add_argument("--policy", default="gentle_sparse_r3")
    args = parser.parse_args()

    warper = RubberBandPhraseWarper(policy=args.policy, rubberband_version=probe_rubberband_r3())
    records = json.loads((args.compare_out / "records.json").read_text())
    by_key: dict[tuple[int, int], dict[str, dict]] = {}
    for record in records:
        by_key.setdefault((record["round"], record["item"]), {})[record["unit"]] = record

    out = args.compare_out / "warped"
    out.mkdir(exist_ok=True)
    timing = {"phrase_moss": [], "bars_moss_total": [], "bar0_moss": [], "bar1_moss": []}
    manifest = []
    for (round_index, item), units in sorted(by_key.items()):
        rendered = {}
        for unit, record in units.items():
            pieces = []
            for part in record["parts"]:
                request = _request(part["request"])
                source = (args.compare_out / part["workspace"] / "source.pcm16.wav").read_bytes()
                frames = RemoteRapChunkRequest.frame_count_for(request.tempo_bpm, bars=request.bar_count)
                warped = warper.warp(request, source, part["source_onsets"], target_frame_count=frames)
                pieces.append(_pcm(warped.vocal_wav))
            path = out / f"r{round_index}-{item:02d}-{unit}.wav"
            _write(path, np.concatenate(pieces))
            rendered[unit] = path.name
        reference = units.get("phrase") or units.get("bars") or units.get("continued")
        bar_unit = units.get("bars") or units.get("continued")
        if "phrase" in units:
            timing["phrase_moss"].append(units["phrase"]["parts"][0]["stage_timings_ms"]["moss"])
        if "bars" in units:
            bar_parts = units["bars"]["parts"]
            timing["bar0_moss"].append(bar_parts[0]["stage_timings_ms"]["moss"])
            timing["bar1_moss"].append(bar_parts[1]["stage_timings_ms"]["moss"])
            timing["bars_moss_total"].append(sum(part["stage_timings_ms"]["moss"] for part in bar_parts))
        if "continued" in units:
            timing.setdefault("continued_bar1_moss", []).append(units["continued"]["parts"][1]["stage_timings_ms"]["moss"])
        # Natural length of each bar's source against its slot: how far R3 has to stretch.
        for part in bar_unit["parts"]:
            target = _request(part["request"]).duration_seconds
            timing.setdefault("source_to_target_ratio", []).append(part["frame_count"] / 24_000 / target)
        manifest.append(
            {
                "round": round_index,
                "item": item,
                "text": " ".join(part["request"]["text"] for part in reference["parts"]),
                "bar_texts": [part["request"]["text"] for part in bar_unit["parts"]],
                **rendered,
            }
        )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                name: {"median": round(statistics.median(values), 3 if name.endswith("ratio") else 1),
                       "p95": round(sorted(values)[max(0, int(0.95 * len(values)) - 1)], 3 if name.endswith("ratio") else 1),
                       "max": round(max(values), 3 if name.endswith("ratio") else 1)}
                for name, values in timing.items() if values
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
