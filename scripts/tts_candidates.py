"""Generic HTTP synthesizers for the TTS candidates of the model-upgrade plan (B1).

Each profile turns one render request (one bar of lyrics) into a request for
an SGLang-Omni ``/v1/audio/speech`` server (or a compatible one) running that
model, with our reference voice. The response is converted to 24 kHz mono
PCM16, which is what ``MossAlignedPhraseRenderer.render_source`` (MMS
alignment) expects, so every candidate goes through the same alignment, warp
and scoring as MOSS.

Duration: ``token_count`` profiles ask for the bar's length (MOSS, 12.5 tokens
per second); the others speak at their own pace and the R3 warp fits them.

Plan: developing-logs/plans/2026-10-14-rap-model-upgrade-plan.md.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
from math import gcd
from pathlib import Path
import time

import httpx
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

RAP_INSTRUCTION = "clear, rhythmically spoken rap with restrained pitch"


@dataclass(frozen=True)
class Voice:
    audio_path: Path
    text: str

    @property
    def uri(self) -> str:
        return self.audio_path.resolve().as_uri()


def _moss(request, voice: Voice, seed: int) -> dict:
    return {
        "input": request.text, "ref_audio": voice.uri, "ref_text": voice.text, "language": "English",
        "instructions": RAP_INSTRUCTION, "token_count": round(request.duration_seconds * 12.5),
        "max_new_tokens": 256, "seed": seed,
    }


def _references(request, voice: Voice, seed: int, **sampling) -> dict:
    return {"input": request.text, "references": [{"audio_path": str(voice.audio_path), "text": voice.text}],
            "seed": seed, **sampling}


PROFILES = {
    # name: (payload builder, supports our continuation prefix)
    "moss_local": (_moss, False),
    "higgs": (lambda r, v, s: _references(r, v, s, temperature=0.8, top_k=50, max_new_tokens=200), False),
    "fish_s2_pro": (lambda r, v, s: _references(r, v, s, temperature=0.8, top_p=0.8, top_k=30), False),
    "fish_drama3": (lambda r, v, s: _references(r, v, s, temperature=0.8, top_p=0.8, top_k=30), False),
    "qwen3_tts": (lambda r, v, s: {"input": r.text, "ref_audio": v.uri, "ref_text": v.text, "language": "English",
                                   "task_type": "Base", "seed": s}, False),
    "indextts": (lambda r, v, s: {"input": r.text, "ref_audio": str(v.audio_path), "seed": s}, False),
}


def to_pcm16_24k_mono(data: bytes) -> tuple[bytes, int, int]:
    """Any WAV response as 24 kHz mono PCM16: (wav bytes, the response's own rate, frames)."""
    rate, samples = wavfile.read(io.BytesIO(data))
    audio = samples.astype(np.float64)
    if samples.dtype == np.int16:
        audio /= 32768.0
    elif samples.dtype == np.int32:
        audio /= 2147483648.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 24_000:
        factor = gcd(rate, 24_000)
        audio = resample_poly(audio, 24_000 // factor, rate // factor)
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    out = io.BytesIO()
    wavfile.write(out, 24_000, pcm)
    return out.getvalue(), int(rate), int(pcm.size)


class HttpTtsSynthesizer:
    """``synthesize(request, output_wav, execution=...)`` for one candidate, as MossAlignedPhraseRenderer expects."""

    def __init__(self, *, profile: str, base_url: str, voice: Voice, model_id: str, base_seed: int = 20261014) -> None:
        self._build, self.supports_prefix = PROFILES[profile]
        self._profile = profile
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(180.0, connect=10.0))
        self._voice = voice
        self._model_id = model_id
        self._base_seed = base_seed
        self.native_sample_rates: set[int] = set()

    def synthesize(self, request, output_wav: Path, *, execution=None, prefix=None):
        from streammuse.infrastructure.rap.moss_tts import MossPhraseResult

        if prefix is not None:
            raise ValueError(f"{self._profile} does not support continuation")
        seed = self._base_seed + request.chunk_index * 7 + request.start_bar
        payload = {"model": self._model_id, "voice": "default", "response_format": "wav", "stream": False,
                   **self._build(request, self._voice, seed)}
        started = time.perf_counter()
        response = self._client.post("/v1/audio/speech", json=payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            raise RuntimeError(f"{self._profile} HTTP {response.status_code}: {response.text[:300]}")
        data = response.content
        if response.headers.get("content-type", "").startswith("application/json"):
            data = base64.b64decode(response.json()["audio"])
        pcm, native_rate, frames = to_pcm16_24k_mono(data)
        self.native_sample_rates.add(native_rate)
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(pcm)
        return MossPhraseResult(
            output_wav=output_wav, model_id=self._model_id, model_revision=self._profile,
            reference_voice_sha256=hashlib.sha256(self._voice.audio_path.read_bytes()).hexdigest(),
            source_wav_sha256=hashlib.sha256(pcm).hexdigest(), sample_rate_hz=24_000, frame_count=frames,
            generation_time_ms=elapsed_ms, resolved_generation_settings={"profile": self._profile, "seed": seed},
            warnings=(),
        )

    def warmup(self, *, execution=None):
        return {}

    def probe(self):
        return {}

    def close(self) -> None:
        self._client.close()
