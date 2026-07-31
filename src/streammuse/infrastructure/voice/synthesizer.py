"""Speech synthesizer adapters with delayed optional imports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from streammuse.application.tasks.speech_output import SpeechOutputConfig

from . import SpeechSynthesisError, VoiceDependencyError, _json_safe


@dataclass(frozen=True)
class SynthesizedAudio:
    samples: np.ndarray
    sample_rate_hz: int

    def __post_init__(self) -> None:
        samples = np.ascontiguousarray(
            np.asarray(self.samples, dtype=np.float32).reshape(-1)
        )
        if int(self.sample_rate_hz) <= 0:
            raise ValueError("sample_rate_hz must be positive")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_rate_hz", int(self.sample_rate_hz))

    @property
    def duration_ms(self) -> float:
        return self.samples.size * 1000.0 / self.sample_rate_hz

    @property
    def byte_count(self) -> int:
        return int(self.samples.nbytes)


class SpeechSynthesizer(Protocol):
    def start(self) -> None: ...

    @property
    def provenance(self) -> dict[str, Any]: ...

    def synthesize(self, text: str) -> SynthesizedAudio: ...

    def close(self) -> None: ...


class NullSpeechSynthesizer:
    def __init__(self, config: SpeechOutputConfig) -> None:
        self.config = config
        self._started = False

    def start(self) -> None:
        self._started = True

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "null",
            "sample_rate_hz": 22_050,
            "rate": float(self.config.rate),
            "native_rate": float(self.config.rate),
            "native_rate_unit": "multiplier",
        }

    def synthesize(self, text: str) -> SynthesizedAudio:
        if not self._started:
            raise SpeechSynthesisError("Null synthesizer has not been started")
        word_count = max(1, len(str(text).split()))
        frames = max(1, int(22_050 * 0.12 * word_count / self.config.rate))
        return SynthesizedAudio(np.zeros(frames, dtype=np.float32), 22_050)

    def close(self) -> None:
        self._started = False


class CommandSpeechSynthesizer:
    """macOS ``say`` or ``espeak-ng`` adapter."""

    def __init__(
        self,
        config: SpeechOutputConfig,
        *,
        backend: str,
        popen_factory: Any = subprocess.Popen,
    ) -> None:
        self.config = config
        self.backend = backend
        self._popen_factory = popen_factory
        self._executable: str | None = None
        self._provenance: dict[str, Any] = {}
        self._active_process: Any | None = None
        self._closed = False

    def start(self) -> None:
        command = "say" if self.backend == "say" else "espeak-ng"
        executable = shutil.which(command)
        if executable is None:
            raise VoiceDependencyError(
                f"Speech backend {command!r} is unavailable. Install it and ensure "
                "the executable is on PATH."
            )
        self._executable = str(Path(executable).resolve())
        native_rate = max(1, int(round(175.0 * float(self.config.rate))))
        provenance: dict[str, Any] = {
            "backend": "system" if self.backend == "say" else "espeak_ng",
            "executable": self._executable,
            "voice": self.config.voice,
            "rate": float(self.config.rate),
            "native_rate": native_rate,
            "native_rate_unit": "words_per_minute",
        }
        if self.backend == "say":
            if self.config.voice is not None:
                voices = self._run_capture([self._executable, "-v", "?"])
                available = {
                    line.split()[0]
                    for line in voices.splitlines()
                    if line.strip()
                }
                if self.config.voice not in available:
                    raise SpeechSynthesisError(
                        f"macOS voice {self.config.voice!r} is not installed"
                    )
            provenance["platform_version"] = self._sw_vers_value(
                "productVersion"
            ) or platform.mac_ver()[0]
            provenance["platform_build"] = self._sw_vers_value("buildVersion")
        else:
            provenance["version"] = self._run_capture(
                [self._executable, "--version"]
            ).splitlines()[0]
        self._provenance = _json_safe(provenance)
        self._closed = False

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def synthesize(self, text: str) -> SynthesizedAudio:
        if self._executable is None or self._closed:
            raise SpeechSynthesisError("Speech synthesizer has not been started")
        temporary_path: Path | None = None
        try:
            handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            handle.close()
            temporary_path = Path(handle.name)
            native_rate = str(max(1, int(round(175.0 * self.config.rate))))
            if self.backend == "say":
                command = [
                    self._executable,
                    "-f",
                    "-",
                    "--data-format=LEI16@22050",
                    "-r",
                    native_rate,
                    "-o",
                    str(temporary_path),
                ]
                if self.config.voice is not None:
                    command[1:1] = ["-v", self.config.voice]
            else:
                command = [
                    self._executable,
                    "--stdin",
                    "-s",
                    native_rate,
                    "-w",
                    str(temporary_path),
                ]
                if self.config.voice is not None:
                    command[1:1] = ["-v", self.config.voice]
            self._run(command, input_text=str(text))
            return _read_pcm_wav(temporary_path)
        except (KeyboardInterrupt, SystemExit):
            self._terminate_active_process()
            raise
        except (SpeechSynthesisError, VoiceDependencyError):
            raise
        except (OSError, ValueError, wave.Error) as exc:
            raise SpeechSynthesisError(f"Speech synthesis failed: {exc}") from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _run(self, command: list[str], *, input_text: str | None = None) -> bytes:
        try:
            process = self._popen_factory(
                command,
                shell=False,
                stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._active_process = process
            try:
                stdout, stderr = process.communicate(
                    None if input_text is None else input_text.encode("utf-8"),
                    timeout=float(self.config.synthesis_timeout_s),
                )
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.communicate()
                raise SpeechSynthesisError(
                    f"Speech synthesis timed out after "
                    f"{self.config.synthesis_timeout_s:g}s"
                ) from exc
            except BaseException:
                self._terminate_process(process)
                raise
            if process.returncode:
                detail = stderr.decode("utf-8", errors="replace").strip()
                raise SpeechSynthesisError(
                    f"Speech command exited with {process.returncode}: {detail}"
                )
            return bytes(stdout)
        except OSError as exc:
            raise SpeechSynthesisError(f"Could not run speech command: {exc}") from exc
        finally:
            self._active_process = None

    def _run_capture(self, command: list[str]) -> str:
        return self._run(command).decode("utf-8", errors="replace")

    def _sw_vers_value(self, key: str) -> str | None:
        executable = shutil.which("sw_vers")
        if executable is None:
            return None
        try:
            return self._run_capture([executable, f"-{key}"]).strip() or None
        except SpeechSynthesisError:
            return None

    @staticmethod
    def _terminate_process(process: Any) -> None:
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except BaseException:
            try:
                process.kill()
                process.wait(timeout=1.0)
            except BaseException:
                pass

    def _terminate_active_process(self) -> None:
        if self._active_process is not None:
            self._terminate_process(self._active_process)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._terminate_active_process()


class KokoroSpeechSynthesizer:
    """Optional Kokoro adapter using an explicitly resolved model snapshot."""

    def __init__(self, config: SpeechOutputConfig) -> None:
        self.config = config
        self._pipeline: Any | None = None
        self._voice_tensor: Any | None = None
        self._provenance: dict[str, Any] = {}

    def start(self) -> None:
        if self.config.model is None or self.config.model_revision is None:
            raise SpeechSynthesisError(
                "Kokoro requires an explicit model and model revision"
            )
        try:
            from huggingface_hub import snapshot_download
            from kokoro import KModel, KPipeline
            import torch
        except (ImportError, OSError) as exc:
            raise VoiceDependencyError(
                "Kokoro speech output requires the tts-kokoro extra"
            ) from exc
        model_path = Path(self.config.model).expanduser()
        if model_path.exists():
            snapshot = model_path.resolve()
        else:
            try:
                snapshot = Path(
                    snapshot_download(
                        self.config.model,
                        revision=self.config.model_revision,
                        cache_dir=self.config.model_cache,
                        local_files_only=self.config.local_files_only,
                    )
                ).resolve()
            except Exception as exc:
                raise SpeechSynthesisError(
                    f"Could not resolve Kokoro model snapshot: {exc}"
                ) from exc
        config_path = snapshot / "config.json"
        model_candidates = tuple(sorted(
            path
            for path in snapshot.glob("*.pth")
            if path.is_file()
        ))
        voice = self.config.voice or "af_heart"
        voice_asset = next(iter(snapshot.rglob(f"{voice}.pt")), None)
        if (
            not config_path.is_file()
            or not model_candidates
            or voice_asset is None
        ):
            raise SpeechSynthesisError(
                "Kokoro snapshot must contain config.json, model weights, "
                f"and voices/{voice}.pt"
            )
        preferred_model = snapshot / "kokoro-v1_0.pth"
        if preferred_model in model_candidates:
            model_path = preferred_model
        elif len(model_candidates) == 1:
            model_path = model_candidates[0]
        else:
            raise SpeechSynthesisError(
                "Kokoro snapshot contains ambiguous model weight files"
            )
        try:
            model = KModel(
                config=str(config_path),
                model=str(model_path),
            ).to("cpu").eval()
            self._pipeline = KPipeline(
                lang_code="a",
                repo_id=self.config.model,
                model=model,
            )
            self._voice_tensor = torch.load(
                voice_asset,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise SpeechSynthesisError(f"Could not start Kokoro: {exc}") from exc
        self._provenance = _json_safe(
            {
                "backend": "kokoro",
                "package_version": importlib.metadata.version("kokoro"),
                "model": self.config.model,
                "model_revision": self.config.model_revision,
                "model_snapshot": str(snapshot),
                "model_weights": str(model_path),
                "voice": voice,
                "voice_asset_sha256": _sha256_file(voice_asset),
                "rate": float(self.config.rate),
                "native_rate": float(self.config.rate),
                "native_rate_unit": "multiplier",
            }
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return dict(self._provenance)

    def synthesize(self, text: str) -> SynthesizedAudio:
        if self._pipeline is None or self._voice_tensor is None:
            raise SpeechSynthesisError("Kokoro has not been started")
        voice = self.config.voice or "af_heart"
        try:
            chunks = [
                np.asarray(audio, dtype=np.float32).reshape(-1)
                for _, _, audio in self._pipeline(
                    str(text),
                    voice=self._voice_tensor,
                    speed=float(self.config.rate),
                )
            ]
        except Exception as exc:
            raise SpeechSynthesisError(f"Kokoro synthesis failed: {exc}") from exc
        if not chunks:
            raise SpeechSynthesisError("Kokoro returned no audio")
        return SynthesizedAudio(np.concatenate(chunks), 24_000)

    def close(self) -> None:
        self._pipeline = None
        self._voice_tensor = None


def create_synthesizer(config: SpeechOutputConfig) -> SpeechSynthesizer:
    if config.backend == "null":
        return NullSpeechSynthesizer(config)
    if config.backend == "kokoro":
        return KokoroSpeechSynthesizer(config)
    if config.backend == "espeak_ng":
        return CommandSpeechSynthesizer(config, backend="espeak_ng")
    backend = "say" if platform.system() == "Darwin" else "espeak_ng"
    return CommandSpeechSynthesizer(config, backend=backend)


def _read_pcm_wav(path: Path) -> SynthesizedAudio:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise SpeechSynthesisError(
            f"Expected 16-bit PCM WAV, received {sample_width * 8}-bit audio"
        )
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return SynthesizedAudio(samples, sample_rate)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
