"""Compare MOSS two-bar phrases with per-bar synthesis (lyric-stream plan, phase 0a).

For every fixed two-bar corpus request, synthesize and MMS-align:

- ``phrase``: the two-bar request in one MOSS call (today's unit);
- ``bars``: the same lyrics as two independent one-bar MOSS calls;
- ``continued`` (phase 6): bar 0 alone, then bar 1 as the MOSS continuation of
  bar 0's audio (needs the continuation patch on the service).

Each source WAV, its syllable onsets and per-call timings are written under
``--out`` so warping and quality metrics (``analyze_bar_unit_synthesis.py``)
can run elsewhere. Runs next to a live SGLang-Omni MOSS service:

    python scripts/compare_bar_unit_synthesis.py --url http://127.0.0.1:8030 \\
        --config <moss_tts.yaml> --aligner-device cuda:0 --out <dir>
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]


def split_into_bars(request):
    """Return the two one-bar requests that carry the same lyrics as ``request``."""
    from streammuse.experiments.rap_audio_protocols.contracts import TwoBarRenderRequest

    first = [item for item in request.syllables if item.tick_in_chunk < 16]
    second = [item for item in request.syllables if item.tick_in_chunk >= 16]
    if second and second[0].index_in_word != 0:
        raise ValueError("a word straddles the bar line")
    tokens = request.text.split()
    first_words = sum(1 for item in first if item.index_in_word == 0)
    lines = (" ".join(tokens[:first_words]), " ".join(tokens[first_words:]))
    if not all(lines):
        raise ValueError("one bar has no words")
    bar_seconds = request.duration_seconds / 2
    halves = []
    for index, line in enumerate(lines):
        syllables = tuple(
            replace(
                item,
                tick_in_chunk=item.tick_in_chunk - 16 * index,
                target_seconds=item.target_seconds - bar_seconds * index,
            )
            for item in request.syllables
            if (item.tick_in_chunk >= 16) == bool(index)
        )
        halves.append(
            TwoBarRenderRequest(
                song_id=request.song_id,
                chunk_index=request.chunk_index * 2 + index,
                start_bar=request.start_bar + index,
                end_bar=request.start_bar + index + 1,
                text=line,
                syllables=syllables,
                tempo_bpm=request.tempo_bpm,
            )
        )
    return tuple(halves)


def main() -> None:
    import profile_sglang_moss as profile

    from streammuse.domain.rap.audio import SpeechPrefix
    from streammuse.infrastructure.rap.mms_forced_alignment import MmsForcedAligner
    from streammuse.infrastructure.rap.moss_aligned_phrase import MossAlignedPhraseRenderer

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8030")
    parser.add_argument("--bench", type=Path, default=profile.DEFAULT_BENCH)
    parser.add_argument("--config", type=Path, help="MOSS runtime config (required unless --tts-profile)")
    parser.add_argument("--tts-profile", help="model-upgrade candidates: a profile from scripts/tts_candidates.py")
    parser.add_argument("--tts-model-id", default="", help="the model name the candidate server expects")
    parser.add_argument("--aligner-device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--units", default="phrase,bars", help="comma-separated: phrase, bars, continued")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    units = [unit.strip() for unit in args.units.split(",") if unit.strip()]
    if not set(units) <= {"phrase", "bars", "continued"}:
        parser.error("--units takes phrase, bars and continued")
    args.out.mkdir(parents=True, exist_ok=False)

    if args.tts_profile:
        from tts_candidates import HttpTtsSynthesizer, Voice

        synthesizer = HttpTtsSynthesizer(
            profile=args.tts_profile, base_url=args.url, model_id=args.tts_model_id,
            voice=Voice(args.bench / "media" / "reference.wav", (args.bench / "reference.txt").read_text().strip()),
        )
        if "continued" in units and not synthesizer.supports_prefix:
            parser.error(f"{args.tts_profile} has no continuation; drop the continued unit")
    else:
        if args.config is None:
            parser.error("--config is required for the MOSS synthesizer")
        synthesizer = profile.make_synthesizer(args)
    renderer = MossAlignedPhraseRenderer(
        synthesizer=synthesizer,
        aligner=MmsForcedAligner.load(device=args.aligner_device),
    )
    requests = []
    for request in profile.load_requests(args.bench):
        try:
            split_into_bars(request)
        except ValueError as error:
            print(f"skip {request.song_id}:{request.chunk_index}: {error}", flush=True)
            continue
        requests.append(request)
    # Warm the service and the aligner once per unit before measuring.
    # Warm the service and the aligner before measuring.
    if "phrase" in units:
        renderer.render_source(requests[0], args.out / "warmup-phrase")
    for index in range(3):
        renderer.render_source(split_into_bars(requests[index])[0], args.out / f"warmup-bar-{index}")

    records = []
    for round_index in range(args.rounds):
        for index, request in enumerate(requests):
            for unit in units:
                parts = (request,) if unit == "phrase" else split_into_bars(request)
                outputs = []
                previous = None
                for part_index, part in enumerate(parts):
                    workspace = args.out / f"r{round_index}-{index:02d}-{unit}-{part_index}"
                    prefix = (
                        SpeechPrefix(parts[part_index - 1].text, previous.source_wav)
                        if unit == "continued" and previous is not None
                        else None
                    )
                    started = time.perf_counter()
                    result = (
                        renderer.render_source(part, workspace)
                        if prefix is None
                        else renderer.render_source(part, workspace, prefix=prefix)
                    )
                    previous = result
                    wall_ms = (time.perf_counter() - started) * 1000.0
                    (workspace / "source.pcm16.wav").write_bytes(result.source_wav)
                    outputs.append(
                        {
                            "workspace": workspace.name,
                            "request": part.to_payload(),
                            "source_onsets": list(result.alignment_diagnostics["source_onsets"]),
                            "onset_confidence": list(result.alignment_diagnostics["onset_confidence"]),
                            "frame_count": result.audio_diagnostics["frame_count"],
                            "stage_timings_ms": dict(result.stage_timings_ms),
                            "wall_ms": wall_ms,
                        }
                    )
                records.append({"round": round_index, "item": index, "unit": unit, "parts": outputs})
                print(
                    json.dumps(
                        {
                            "item": index,
                            "unit": unit,
                            "moss_ms": [round(part["stage_timings_ms"]["moss"]) for part in outputs],
                            "wall_ms": [round(part["wall_ms"]) for part in outputs],
                        }
                    ),
                    flush=True,
                )
    (args.out / "records.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
