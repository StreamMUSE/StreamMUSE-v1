# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "openai-whisper==20250625",
#   "jiwer==4.0.0",
#   "scipy>=1.13",
#   "numpy>=1.26",
# ]
# ///
"""ASR intelligibility diagnostic for client_simulation.py --save-audio runs.

For every accepted chunk it transcribes the server vocal (chunk-NNN-vocals.wav)
with Whisper and scores it against the two selected lyric lines. Settings
match scripts/check_moss_benchmark_audio.py (small.en, EnglishTextNormalizer,
temperature 0, beam 5) so numbers are comparable. This is an automatic
transcript-error diagnostic only, not human WER, MOS, or rhythm quality.

Runs in its own uv environment from the inline metadata above:

    uv run scripts/check_client_simulation_asr.py output/rap_mac_unconstrained_20260930 [more runs...]

Writes <run>/asr.jsonl and prints one summary line per run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from math import gcd
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import jiwer
    import numpy as np
    import whisper
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from whisper.normalizers import EnglishTextNormalizer

    model = whisper.load_model(args.model, device=args.device)
    normalize = EnglishTextNormalizer()
    for run in args.runs:
        rows = [json.loads(line) for line in (run / "records.jsonl").read_text().splitlines()]
        results = []
        with (run / "asr.jsonl").open("w") as ledger:
            for row in rows:
                vocals = (row.get("audio_files") or {}).get("vocals")
                if row["warmup"] or row["error"] or not vocals or not row.get("selected"):
                    continue
                path = run / vocals
                rate, audio = wavfile.read(path)
                scale = np.iinfo(audio.dtype).max if audio.dtype.kind in "iu" else 1
                audio = audio.astype(np.float32) / scale
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                factor = gcd(rate, 16_000)
                audio = resample_poly(audio, 16_000 // factor, rate // factor).astype(np.float32)
                hypothesis_raw = model.transcribe(
                    audio,
                    language="en",
                    temperature=0,
                    beam_size=5,
                    condition_on_previous_text=False,
                    fp16=args.device.startswith("cuda"),
                )["text"].strip()
                reference = normalize(" ".join(row["selected"]))
                hypothesis = normalize(hypothesis_raw)
                score = jiwer.process_words(reference, hypothesis)
                result = {
                    "index": row["index"],
                    "vocals_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "wer": score.wer,
                    "substitutions": score.substitutions,
                    "deletions": score.deletions,
                    "insertions": score.insertions,
                    "reference_words": len(reference.split()),
                }
                ledger.write(json.dumps(result) + "\n")
                results.append(result)
        if not results:
            print(f"{run}: no accepted chunks with saved vocals")
            continue
        errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in results)
        words = sum(r["reference_words"] for r in results)
        wers = [r["wer"] for r in results]
        print(
            json.dumps(
                {
                    "run": str(run),
                    "chunks": len(results),
                    "corpus_wer": round(errors / words, 4),
                    "chunk_wer_median": round(statistics.median(wers), 4),
                    "chunk_wer_max": round(max(wers), 4),
                    "chunks_with_errors": sum(1 for w in wers if w > 0),
                    "model": args.model,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
