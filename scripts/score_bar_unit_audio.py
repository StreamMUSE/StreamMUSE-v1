# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "openai-whisper==20250625",
#   "jiwer==4.0.0",
#   "scipy>=1.13",
#   "numpy>=1.26",
#   "librosa==0.11.0",
#   "speechbrain==1.0.3",
#   "torchaudio>=2.5,<2.9",
#   "huggingface_hub<0.26",
# ]
# ///
"""Quality metrics for phase 0a: one two-bar MOSS phrase vs two one-bar lines.

Reads ``<compare-out>/warped/manifest.json`` written by
``analyze_bar_unit_synthesis.py``. For each item it scores ``phrase.wav`` and
``bars.wav`` (same lyrics, both warped to exactly two bars):

- WER: Whisper small.en against the lyrics (same settings as the other ASR checks);
- speaker similarity between the two bars: ECAPA cosine of bar 0 vs bar 1;
- seam loudness jump: |RMS dB of the last 500 ms of bar 0 - first 500 ms of bar 1|
  over voiced frames;
- seam pitch jump: |median F0 (semitones) of the same two windows|.

The phrase numbers are the natural within-phrase baseline: a seam metric for
``bars`` is only worse if it exceeds what one continuous phrase already does at
the same point. Automatic proxies only; blind listening decides.

    uv run scripts/score_bar_unit_audio.py <compare-out-dir>
"""

from __future__ import annotations

import argparse
import json
from math import gcd
from pathlib import Path
import statistics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("compare_out", type=Path)
    parser.add_argument("--reference", type=Path, help="reference voice WAV: adds each take's similarity to it")
    parser.add_argument("--whisper-model", default="small.en")
    args = parser.parse_args()

    import jiwer
    import librosa
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from speechbrain.inference.speaker import EncoderClassifier
    import torch
    import whisper
    from whisper.normalizers import EnglishTextNormalizer

    asr = whisper.load_model(args.whisper_model, device="cpu")
    normalize = EnglishTextNormalizer()
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})

    def load16k(path: Path) -> np.ndarray:
        rate, audio = wavfile.read(path)
        audio = audio.astype(np.float32) / 32768.0
        factor = gcd(rate, 16_000)
        return resample_poly(audio, 16_000 // factor, rate // factor).astype(np.float32)

    def embedding(audio: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            vector = ecapa.encode_batch(torch.from_numpy(audio).unsqueeze(0)).squeeze().numpy()
        return vector / np.linalg.norm(vector)

    def voiced_rms_db(audio: np.ndarray) -> float:
        frames = librosa.util.frame(audio, frame_length=400, hop_length=160)
        rms = np.sqrt(np.mean(frames**2, axis=0))
        active = rms[rms > np.percentile(rms, 30)] if rms.size else rms
        return float(20 * np.log10(np.mean(active) + 1e-9))

    def median_semitones(audio: np.ndarray) -> float | None:
        f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=16_000, frame_length=1024)
        values = f0[voiced & np.isfinite(f0)]
        return float(12 * np.log2(np.median(values) / 55.0)) if values.size else None

    reference_embedding = embedding(load16k(args.reference)) if args.reference else None
    manifest = json.loads((args.compare_out / "warped" / "manifest.json").read_text())
    rows = []
    for entry in manifest:
        row = {"round": entry["round"], "item": entry["item"]}
        reference = normalize(entry["text"])
        for unit in [unit for unit in ("phrase", "bars", "continued") if unit in entry]:
            audio = load16k(args.compare_out / "warped" / entry[unit])
            half = audio.size // 2
            hypothesis = normalize(
                asr.transcribe(audio, language="en", temperature=0, beam_size=5, condition_on_previous_text=False, fp16=False)["text"]
            )
            score = jiwer.process_words(reference, hypothesis)
            window = 8_000  # 500 ms at 16 kHz
            before, after = audio[half - window : half], audio[half : half + window]
            pitch_before, pitch_after = median_semitones(before), median_semitones(after)
            row[unit] = {
                "errors": score.substitutions + score.deletions + score.insertions,
                "words": len(reference.split()),
                "hypothesis": hypothesis,
                "speaker_similarity": float(np.dot(embedding(audio[:half]), embedding(audio[half:]))),
                "reference_similarity": (
                    None if reference_embedding is None else float(np.dot(embedding(audio), reference_embedding))
                ),
                "seam_loudness_jump_db": abs(voiced_rms_db(before) - voiced_rms_db(after)),
                "seam_pitch_jump_semitones": (
                    None if pitch_before is None or pitch_after is None else abs(pitch_before - pitch_after)
                ),
            }
        rows.append(row)
        print(json.dumps({"item": row["item"], "round": row["round"], **{u: {k: v for k, v in row[u].items() if k != "hypothesis"} for u in ("phrase", "bars", "continued") if u in row}}), flush=True)

    summary = {}
    for unit in [unit for unit in ("phrase", "bars", "continued") if all(unit in row for row in rows)]:
        values = [row[unit] for row in rows]
        pitch = [item["seam_pitch_jump_semitones"] for item in values if item["seam_pitch_jump_semitones"] is not None]
        summary[unit] = {
            "wer": sum(item["errors"] for item in values) / sum(item["words"] for item in values),
            "speaker_similarity_median": statistics.median(item["speaker_similarity"] for item in values),
            "speaker_similarity_min": min(item["speaker_similarity"] for item in values),
            "reference_similarity_median": (
                statistics.median(item["reference_similarity"] for item in values)
                if values[0]["reference_similarity"] is not None else None
            ),
            "seam_loudness_jump_db_median": statistics.median(item["seam_loudness_jump_db"] for item in values),
            "seam_loudness_jump_db_p90": sorted(item["seam_loudness_jump_db"] for item in values)[int(0.9 * len(values)) - 1],
            "seam_pitch_jump_semitones_median": statistics.median(pitch) if pitch else None,
            "seam_pitch_jump_semitones_p90": sorted(pitch)[int(0.9 * len(pitch)) - 1] if pitch else None,
        }
    (args.compare_out / "quality.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
