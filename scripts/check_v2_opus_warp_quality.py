"""Protocol v2 Opus check: how much does Opus on the raw phrase cost after the Mac warp?

For every v2 ``response.zip`` under a render server ``--artifact-root``, compare
against the PCM warp (the lossless reference):

- v2: R3(Opus(source)), the v2 Opus route;
- v1: Opus(R3(source)), the v1 Opus route, as the bar to meet;
- dither: R3(source + 1 LSB noise), an inaudible-change control.

R3 is a phase vocoder, so a tiny input change rotates the output phase: the
dither control already destroys waveform SNR/correlation. Only magnitude metrics
(STFT magnitude SNR, log-spectral distance) are meaningful downstream of R3.

    uv run python scripts/check_v2_opus_warp_quality.py <artifact-root> [--bitrate-kbps 48]
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]


def _samples(wav: bytes) -> np.ndarray:
    _, samples = wavfile.read(io.BytesIO(wav))
    return samples.astype(np.float64) / 32768.0


def _wav(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    wavfile.write(buffer, 24_000, samples.astype("<i2"))
    return buffer.getvalue()


def _opus_round_trip(wav: bytes, bitrate_kbps: int) -> bytes:
    _, pcm = wavfile.read(io.BytesIO(wav))
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    encoded = subprocess.run(
        [*common, "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0", "-c:a", "libopus",
         "-b:a", f"{bitrate_kbps}k", "-vbr", "on", "-application", "audio", "-compression_level", "10",
         "-f", "opus", "pipe:1"],
        input=pcm.astype("<i2").tobytes(), capture_output=True, check=True,
    ).stdout
    decoded = subprocess.run(
        [*common, "-i", "pipe:0", "-f", "s16le", "-ar", "24000", "-ac", "1", "pipe:1"],
        input=encoded, capture_output=True, check=True,
    ).stdout
    return _wav(np.frombuffer(decoded, "<i2")[: len(pcm)])


def _dithered(wav: bytes) -> bytes:
    _, pcm = wavfile.read(io.BytesIO(wav))
    noise = np.random.default_rng(0).integers(-1, 2, len(pcm))
    return _wav(np.clip(pcm.astype(np.int32) + noise, -32768, 32767))


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    n = min(len(reference), len(candidate))
    reference, candidate = reference[:n], candidate[:n]
    wave_snr = 10 * np.log10(np.sum(reference**2) / max(np.sum((reference - candidate) ** 2), 1e-20))
    ref_mag = np.abs(stft(reference, fs=24_000, nperseg=1024, noverlap=768)[2])
    cand_mag = np.abs(stft(candidate, fs=24_000, nperseg=1024, noverlap=768)[2])
    mag_snr = 10 * np.log10(np.sum(ref_mag**2) / max(np.sum((ref_mag - cand_mag) ** 2), 1e-20))
    floor = 1e-5
    log_ratio = 20 * np.log10(np.maximum(ref_mag, floor) / np.maximum(cand_mag, floor))
    return {
        "wave_snr_db": float(wave_snr),
        "wave_corr": float(np.corrcoef(reference, candidate)[0, 1]),
        "mag_snr_db": float(mag_snr),
        "lsd_db": float(np.mean(np.sqrt(np.mean(log_ratio**2, axis=0)))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--policy", choices=("gentle_sparse_r3", "all_onsets_r3"), default="gentle_sparse_r3")
    parser.add_argument("--bitrate-kbps", type=int, default=48, help="v2 route bitrate; v1 stays at 48")
    parser.add_argument("--out", type=Path, help="optional JSON file for per-chunk rows and medians")
    args = parser.parse_args()

    from streammuse.application.rap.chunk_orchestration import chunk_render_request
    from streammuse.domain.rap import REMOTE_CHUNK_SCHEMA_VERSION_V2, RemoteRapChunkRequest
    from streammuse.infrastructure.rap.chunk_package import decode_chunk_package
    from streammuse.infrastructure.rap.phrase_warp import RubberBandPhraseWarper, probe_rubberband_r3

    warper = RubberBandPhraseWarper(policy=args.policy, rubberband_version=probe_rubberband_r3())
    rows = []
    for workspace in sorted(path.parent for path in args.artifact_root.glob("*/*/response.zip")):
        request = RemoteRapChunkRequest.from_payload(json.loads((workspace / "request.json").read_text()))
        if request.schema_version != REMOTE_CHUNK_SCHEMA_VERSION_V2:
            continue
        package = decode_chunk_package(
            (workspace / "response.zip").read_bytes(), expected_request_id=request.request_id
        )
        render_request = chunk_render_request(request, package.manifest.selected_bars)
        onsets = package.manifest.diagnostics.alignment_diagnostics["source_onsets"]

        def warp(source_wav: bytes) -> bytes:
            return warper.warp(
                render_request, source_wav, onsets, target_frame_count=request.expected_frame_count
            ).vocal_wav

        reference_wav = warp(package.vocal_wav)
        reference = _samples(reference_wav)
        row = {
            "request_id": request.request_id,
            "v1": _metrics(reference, _samples(_opus_round_trip(reference_wav, 48))),
            "v2": _metrics(reference, _samples(warp(_opus_round_trip(package.vocal_wav, args.bitrate_kbps)))),
            "dither": _metrics(reference, _samples(warp(_dithered(package.vocal_wav)))),
        }
        rows.append(row)
        print(
            request.request_id[:8],
            " | ".join(
                f"{route} mag {row[route]['mag_snr_db']:5.1f} dB LSD {row[route]['lsd_db']:.2f} "
                f"corr {row[route]['wave_corr']:.3f}"
                for route in ("v1", "v2", "dither")
            ),
            flush=True,
        )
    if not rows:
        raise SystemExit(f"no v2 response.zip under {args.artifact_root}")
    medians = {
        route: {key: float(np.median([row[route][key] for row in rows])) for key in rows[0][route]}
        for route in ("v1", "v2", "dither")
    }
    summary = {"n": len(rows), "policy": args.policy, "v2_bitrate_kbps": args.bitrate_kbps, "median": medians}
    print(json.dumps(summary, indent=2))
    if args.out is not None:
        args.out.write_text(json.dumps({**summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
