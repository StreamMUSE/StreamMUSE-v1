"""Post-benchmark ASR diagnostic, not a perceptual quality acceptance test."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from math import gcd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import jiwer
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    import whisper
    from whisper.normalizers import EnglishTextNormalizer

    output = args.root / "audio-quality"
    output.mkdir(exist_ok=False)
    parameters = {
        "model": args.model, "device": args.device,
        "download_root": str(args.download_root.resolve()),
        "whisper_version": importlib.metadata.version("openai-whisper"),
        "jiwer_version": importlib.metadata.version("jiwer"),
        "normalizer": "whisper.normalizers.EnglishTextNormalizer",
        "language": "en", "temperature": 0, "beam_size": 5,
        "condition_on_previous_text": False,
        "caveat": "Automatic transcript error diagnostic only; not human WER, MOS, voice similarity, or rhythm quality.",
    }
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    model = whisper.load_model(args.model, device=args.device,
                               download_root=str(args.download_root))
    weight = args.download_root / f"{args.model}.pt"
    parameters["model_sha256"] = hashlib.sha256(weight.read_bytes()).hexdigest()
    (output / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n")
    normalize = EnglishTextNormalizer()
    transcripts = {}
    results = []
    with (output / "samples.jsonl").open("w", buffering=1) as ledger:
        for label in ("A1", "B1", "B2", "A2"):
            rows = [json.loads(line) for line in
                    (args.root / label / "records.jsonl").read_text().splitlines()]
            for row in rows:
                if row["warmup"] or not row["success"]:
                    continue
                for kind, name in (("source", "source.wav"), ("final", "vocal.wav")):
                    path = args.root / label / f"sample-{row['sample']:02}" / name
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if digest not in transcripts:
                        rate, audio = wavfile.read(path)
                        scale = np.iinfo(audio.dtype).max if audio.dtype.kind in "iu" else 1
                        audio = audio.astype(np.float32) / scale
                        if audio.ndim == 2:
                            audio = audio.mean(axis=1)
                        factor = gcd(rate, 16000)
                        audio = resample_poly(audio, 16000 // factor, rate // factor)
                        transcripts[digest] = model.transcribe(
                            audio.astype(np.float32), language="en", temperature=0,
                            beam_size=5, condition_on_previous_text=False,
                            fp16=args.device.startswith("cuda"),
                        )["text"].strip()
                    reference = normalize(row["text"])
                    hypothesis = normalize(transcripts[digest])
                    score = jiwer.process_words(reference, hypothesis)
                    result = {
                        "block": label, "backend": row["backend"], "sample": row["sample"],
                        "kind": kind, "sha256": digest, "reference": row["text"],
                        "transcript": transcripts[digest], "reference_normalized": reference,
                        "transcript_normalized": hypothesis, "wer": score.wer,
                        "substitutions": score.substitutions, "deletions": score.deletions,
                        "insertions": score.insertions, "reference_words": len(reference.split()),
                    }
                    ledger.write(json.dumps(result, sort_keys=True) + "\n")
                    results.append(result)
                    print(json.dumps({key: result[key] for key in
                                      ("block", "sample", "kind", "wer", "transcript")}), flush=True)
    summary = {"unique_audio_transcriptions": len(transcripts), "groups": {}}
    for backend in sorted({r["backend"] for r in results}):
        for kind in ("source", "final"):
            rows = [r for r in results if r["backend"] == backend and r["kind"] == kind]
            counts = {key: sum(r[key] for r in rows) for key in
                      ("substitutions", "deletions", "insertions", "reference_words")}
            summary["groups"][f"{backend}/{kind}"] = {
                "n": len(rows), **counts,
                "corpus_wer": sum(counts[k] for k in
                                  ("substitutions", "deletions", "insertions")) / counts["reference_words"],
                "zero_error_samples": sum(r["wer"] == 0 for r in rows),
            }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
