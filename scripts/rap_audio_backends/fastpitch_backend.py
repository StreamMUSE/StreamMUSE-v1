"""NeMo FastPitch explicit-phoneme backend for the rap audio protocol comparison."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.io import wavfile

from streammuse.experiments.rap_audio_protocols.artifacts import (
    append_chunk_record,
    chunk_record_is_complete,
    file_sha256,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.audio import validate_wav_metadata
from streammuse.experiments.rap_audio_protocols.contracts import (
    ChunkRenderRecord,
    ProtocolId,
    SyllableTarget,
    TwoBarRenderRequest,
    canonical_json_dumps,
)
from streammuse.experiments.rap_audio_protocols.timing import FastPitchPhonePlan, build_fastpitch_phone_plan


FASTPITCH_MODEL_ID = "tts_en_fastpitch"
HIFIGAN_MODEL_ID = "tts_en_hifigan"
FASTPITCH_SAMPLE_RATE_HZ = 22_050
DEFAULT_MAX_ATTEMPTS = 3
PROSODY_MODE_DURATION_ONLY = "duration-only"
PROSODY_MODE_SMOKE_VALIDATED = "smoke-validated"
PROSODY_MODES = (PROSODY_MODE_DURATION_ONLY, PROSODY_MODE_SMOKE_VALIDATED)
PROSODY_CONTROLS_ENABLED = "stress_pitch_energy"
PROSODY_CONTROLS_DURATION_ONLY = "duration_only_default"
PROSODY_CONTROLS_GUARDED = "duration_only_version_guard"
PROSODY_CONTROLS_NOT_STARTED = "not_started"


@dataclass(frozen=True)
class FastPitchBackendRuntime:
    fastpitch: Any
    hifigan: Any
    torch_module: Any
    device: str
    fastpitch_model_id: str
    hifigan_model_id: str

    @property
    def sample_rate_hz(self) -> int:
        return int(getattr(self.hifigan, "sample_rate", FASTPITCH_SAMPLE_RATE_HZ))


@dataclass(frozen=True)
class FastPitchRenderPlan:
    tokens: Any
    tokenizer_labels: tuple[str, ...]
    phone_plan: FastPitchPhonePlan
    duration_tensor: Any

    @property
    def duration_frames(self) -> tuple[int, ...]:
        return self.phone_plan.duration_frames

    @property
    def anchor_error_frames(self) -> tuple[int, ...]:
        return self.phone_plan.anchor_error_frames

    @property
    def syllable_phone_groups(self) -> tuple[tuple[str, ...], ...]:
        return self.phone_plan.syllable_phone_groups

    @property
    def compressed_consonant_regions(self) -> tuple[int, ...]:
        return self.phone_plan.compressed_consonant_regions

    @property
    def grapheme_fallback_words(self) -> tuple[str, ...]:
        return self.phone_plan.grapheme_fallback_words


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--requests-jsonl", required=True, type=Path)
    parser.add_argument("--records-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fastpitch-model", default=FASTPITCH_MODEL_ID)
    parser.add_argument("--hifigan-model", default=HIFIGAN_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--song", default=None)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--prosody-controls",
        choices=PROSODY_MODES,
        default=PROSODY_MODE_DURATION_ONLY,
        help="enable synthetic pitch/energy only after smoke validation on the installed NeMo build",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: Any | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if args.max_attempts <= 0:
        print("error: --max-attempts must be positive", file=sys.stderr)
        return 2

    requests = load_requests(args.requests_jsonl)
    if args.song:
        requests = tuple(request for request in requests if request.song_id == args.song)
    if not requests:
        print("error: no requests selected for rendering", file=sys.stderr)
        return 2

    factory = runtime_factory or create_runtime
    runtime = factory(
        fastpitch_model_id=args.fastpitch_model,
        hifigan_model_id=args.hifigan_model,
        device=args.device,
    )
    records = render_pending_requests(
        request_path=args.requests_jsonl,
        record_path=args.records_jsonl,
        output_dir=args.output_dir,
        runtime=runtime,
        max_attempts=args.max_attempts,
        selected_song_id=args.song,
        prosody_mode=args.prosody_controls,
    )
    return 1 if any(not record.success for record in records) else 0


def create_runtime(
    *,
    fastpitch_model_id: str = FASTPITCH_MODEL_ID,
    hifigan_model_id: str = HIFIGAN_MODEL_ID,
    device: str = "cuda:0",
) -> FastPitchBackendRuntime:
    torch_module = importlib.import_module("torch")
    nemo_tts = importlib.import_module("nemo.collections.tts")
    models = getattr(nemo_tts, "models")

    fastpitch = _load_nemo_model(getattr(models, "FastPitchModel"), fastpitch_model_id, device=device)
    hifigan = _load_nemo_model(getattr(models, "HifiGanModel"), hifigan_model_id, device=device)
    return FastPitchBackendRuntime(
        fastpitch=fastpitch,
        hifigan=hifigan,
        torch_module=torch_module,
        device=device,
        fastpitch_model_id=fastpitch_model_id,
        hifigan_model_id=hifigan_model_id,
    )


def _load_nemo_model(model_class: Any, model_ref: str, *, device: str) -> Any:
    model_path = Path(model_ref)
    if model_path.exists():
        model = model_class.restore_from(restore_path=str(model_path), map_location=device)
    else:
        model = model_class.from_pretrained(model_name=model_ref, map_location=device)
    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_requests(path: Path | str) -> tuple[TwoBarRenderRequest, ...]:
    requests: list[TwoBarRenderRequest] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            requests.append(_request_from_payload(payload))
    return tuple(requests)


def render_pending_requests(
    *,
    request_path: Path | str,
    record_path: Path | str,
    output_dir: Path | str,
    runtime: FastPitchBackendRuntime | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    selected_song_id: str | None = None,
    prosody_mode: str = PROSODY_MODE_DURATION_ONLY,
    progress_stream: Any = None,
) -> tuple[ChunkRenderRecord, ...]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if prosody_mode not in PROSODY_MODES:
        raise ValueError(f"unsupported prosody mode: {prosody_mode}")

    requests = load_requests(request_path)
    if selected_song_id is not None:
        requests = tuple(request for request in requests if request.song_id == selected_song_id)

    active_runtime = runtime if runtime is not None else create_runtime()
    existing = read_chunk_record_index(record_path)
    output_root = Path(output_dir)
    ledger_path = Path(record_path)
    stream = progress_stream if progress_stream is not None else sys.stdout
    records: list[ChunkRenderRecord] = []

    for request in requests:
        key = (ProtocolId.FASTPITCH_PHONEME, request.song_id, request.chunk_index)
        output_path = output_root / request.song_id / f"chunk-{request.chunk_index:03d}.wav"
        if chunk_record_is_complete(
            ledger_path,
            output_path,
            request=request,
            protocol_id=ProtocolId.FASTPITCH_PHONEME,
        ):
            continue
        record = _render_with_retries(
            request=request,
            output_path=output_path,
            record_path=ledger_path,
            runtime=active_runtime,
            max_attempts=max_attempts,
            prosody_mode=prosody_mode,
            replace_existing=key in existing,
            progress_stream=stream,
        )
        existing[key] = record
        records.append(record)
    return tuple(records)


def build_render_plan(*, request: TwoBarRenderRequest, runtime: FastPitchBackendRuntime) -> FastPitchRenderPlan:
    tokens = runtime.fastpitch.parse(request.text)
    if hasattr(tokens, "to"):
        tokens = tokens.to(runtime.device)
    token_ids = list(tokens[0].tolist())
    tokenizer_labels = _token_ids_to_labels(runtime.fastpitch.vocab, token_ids)
    phone_plan = build_fastpitch_phone_plan(request, tokenizer_labels)
    duration_tensor = runtime.torch_module.tensor(
        [list(phone_plan.duration_frames)],
        device=runtime.device,
        dtype=getattr(runtime.torch_module, "long", None),
    )
    if getattr(duration_tensor, "shape", None) != getattr(tokens, "shape", None):
        raise ValueError("duration tensor must match the FastPitch token shape")
    return FastPitchRenderPlan(
        tokens=tokens,
        tokenizer_labels=tokenizer_labels,
        phone_plan=phone_plan,
        duration_tensor=duration_tensor,
    )


def _token_ids_to_labels(vocab: Any, token_ids: Sequence[int]) -> tuple[str, ...]:
    ids_to_tokens = getattr(vocab, "ids_to_tokens", None)
    if callable(ids_to_tokens):
        return tuple(ids_to_tokens(token_ids))

    id_to_token = getattr(vocab, "_id2token", None)
    if id_to_token is not None:
        return tuple(id_to_token[token_id] for token_id in token_ids)

    tokens = getattr(vocab, "tokens", None)
    if tokens is not None:
        return tuple(tokens[token_id] for token_id in token_ids)

    raise AttributeError("FastPitch vocabulary does not expose token ID labels")


def render_request(
    *,
    request: TwoBarRenderRequest,
    output_path: Path | str,
    runtime: FastPitchBackendRuntime,
    prosody_mode: str = PROSODY_MODE_DURATION_ONLY,
) -> ChunkRenderRecord:
    if prosody_mode not in PROSODY_MODES:
        raise ValueError(f"unsupported prosody mode: {prosody_mode}")
    plan = build_render_plan(request=request, runtime=runtime)
    audio, _ = _synthesise_audio(request=request, runtime=runtime, plan=plan, prosody_mode=prosody_mode)
    _write_audio_wav(output_path, audio, sample_rate_hz=runtime.sample_rate_hz)
    metadata = validate_wav_metadata(output_path, expected_sample_rate_hz=runtime.sample_rate_hz, expected_channels=1)
    _write_timing_sidecar(output_path, plan)
    return ChunkRenderRecord(
        protocol_id=ProtocolId.FASTPITCH_PHONEME,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=True,
        output_path=str(Path(output_path)),
        output_sha256=file_sha256(output_path),
        source_chunk_sha256=None,
        sample_rate_hz=metadata.sample_rate_hz,
        attempts=1,
        error=None,
    )


def chunk_frame_count(
    request: TwoBarRenderRequest,
    *,
    sample_rate_hz: int = FASTPITCH_SAMPLE_RATE_HZ,
) -> int:
    return int(round(request.duration_seconds * sample_rate_hz))


def _render_with_retries(
    *,
    request: TwoBarRenderRequest,
    output_path: Path,
    record_path: Path,
    runtime: FastPitchBackendRuntime,
    max_attempts: int,
    prosody_mode: str,
    replace_existing: bool,
    progress_stream: Any,
) -> ChunkRenderRecord:
    plan: FastPitchRenderPlan | None = None
    last_error: Exception | None = None
    last_prosody_controls = PROSODY_CONTROLS_NOT_STARTED

    for attempt in range(1, max_attempts + 1):
        try:
            if plan is None:
                plan = build_render_plan(request=request, runtime=runtime)
            audio, last_prosody_controls = _synthesise_audio(
                request=request,
                runtime=runtime,
                plan=plan,
                prosody_mode=prosody_mode,
            )
            _write_audio_wav(output_path, audio, sample_rate_hz=runtime.sample_rate_hz)
            metadata = validate_wav_metadata(
                output_path,
                expected_sample_rate_hz=runtime.sample_rate_hz,
                expected_channels=1,
            )
            _write_timing_sidecar(output_path, plan)
            record = ChunkRenderRecord(
                protocol_id=ProtocolId.FASTPITCH_PHONEME,
                song_id=request.song_id,
                chunk_index=request.chunk_index,
                request_sha256=request.sha256,
                success=True,
                output_path=str(output_path),
                output_sha256=file_sha256(output_path),
                source_chunk_sha256=None,
                sample_rate_hz=metadata.sample_rate_hz,
                attempts=attempt,
                error=None,
            )
            stored = _store_chunk_record(record_path, record, replace_existing=replace_existing)
            print(_progress_line(stored, plan=plan, prosody_controls=last_prosody_controls), file=progress_stream)
            return stored
        except Exception as exc:
            last_error = exc

    _timing_sidecar_path(output_path).unlink(missing_ok=True)
    silence_frame_count = chunk_frame_count(request, sample_rate_hz=runtime.sample_rate_hz)
    _write_silence_wav(
        output_path,
        frame_count=silence_frame_count,
        sample_rate_hz=runtime.sample_rate_hz,
    )
    metadata = validate_wav_metadata(
        output_path,
        expected_sample_rate_hz=runtime.sample_rate_hz,
        expected_channels=1,
        expected_frame_count=silence_frame_count,
    )
    failure = ChunkRenderRecord(
        protocol_id=ProtocolId.FASTPITCH_PHONEME,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=False,
        output_path=str(output_path),
        output_sha256=file_sha256(output_path),
        source_chunk_sha256=None,
        sample_rate_hz=metadata.sample_rate_hz,
        attempts=max_attempts,
        error=_error_payload(last_error, attempts=max_attempts, plan=plan, prosody_controls=last_prosody_controls),
    )
    stored_failure = _store_chunk_record(record_path, failure, replace_existing=replace_existing)
    print(_progress_line(stored_failure, plan=plan, prosody_controls=last_prosody_controls), file=progress_stream)
    return stored_failure


def _synthesise_audio(
    *,
    request: TwoBarRenderRequest,
    runtime: FastPitchBackendRuntime,
    plan: FastPitchRenderPlan,
    prosody_mode: str,
) -> tuple[np.ndarray, str]:
    torch_module = runtime.torch_module
    no_grad = torch_module.no_grad if hasattr(torch_module, "no_grad") else nullcontext
    kwargs = {
        "text": plan.tokens,
        "durs": plan.duration_tensor,
        "pitch": None,
        "energy": None,
        "speaker": None,
        "pace": 1.0,
    }
    with no_grad():
        spect, *_ = runtime.fastpitch(**kwargs)
        prosody_controls = PROSODY_CONTROLS_DURATION_ONLY
        if prosody_mode == PROSODY_MODE_SMOKE_VALIDATED and _supports_pitch_energy_controls(runtime.fastpitch):
            pitch, energy = _build_prosody_controls(request=request, runtime=runtime, plan=plan)
            try:
                spect, *_ = runtime.fastpitch(
                    text=plan.tokens,
                    durs=plan.duration_tensor,
                    pitch=pitch,
                    energy=energy,
                    speaker=None,
                    pace=1.0,
                )
                prosody_controls = PROSODY_CONTROLS_ENABLED
            except Exception as exc:
                if not _is_version_guard_error(exc):
                    raise
                prosody_controls = PROSODY_CONTROLS_GUARDED
        elif prosody_mode == PROSODY_MODE_SMOKE_VALIDATED:
            prosody_controls = PROSODY_CONTROLS_GUARDED
        audio = runtime.hifigan.convert_spectrogram_to_audio(spec=spect)
    return _to_mono_float32(audio), prosody_controls


def _supports_pitch_energy_controls(fastpitch: Any) -> bool:
    try:
        parameters = inspect.signature(fastpitch.__call__).parameters.values()
    except (TypeError, ValueError):
        return True
    has_pitch = False
    has_energy = False
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        has_pitch = has_pitch or parameter.name == "pitch"
        has_energy = has_energy or parameter.name == "energy"
    return has_pitch and has_energy


def _build_prosody_controls(
    *,
    request: TwoBarRenderRequest,
    runtime: FastPitchBackendRuntime,
    plan: FastPitchRenderPlan,
) -> tuple[Any, Any]:
    total_frames = sum(plan.duration_frames)
    pitch = np.zeros((1, total_frames), dtype=np.float32)
    energy = np.ones((1, total_frames), dtype=np.float32)
    label_groups = _syllable_label_groups(plan.phone_plan)
    for syllable, label_indices in zip(request.syllables, label_groups):
        if not label_indices:
            continue
        start_frame = sum(plan.duration_frames[index] for index in range(label_indices[0]))
        end_frame = sum(plan.duration_frames[index] for index in range(label_indices[-1] + 1))
        if end_frame <= start_frame:
            continue
        pitch_value = 0.05 + 0.10 * float(syllable.target_stress)
        energy_value = 1.00 + 0.15 * float(syllable.target_stress)
        pitch[0, start_frame:end_frame] = pitch_value
        energy[0, start_frame:end_frame] = energy_value
    tensor = getattr(runtime.torch_module, "tensor")
    return (
        tensor(pitch.tolist(), device=runtime.device, dtype=getattr(runtime.torch_module, "float32", None)),
        tensor(energy.tolist(), device=runtime.device, dtype=getattr(runtime.torch_module, "float32", None)),
    )


def _syllable_label_groups(phone_plan: FastPitchPhonePlan) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    cursor = 0
    spoken = phone_plan.spoken_label_indices
    for phones in phone_plan.syllable_phone_groups:
        next_cursor = cursor + len(phones)
        groups.append(tuple(spoken[cursor:next_cursor]))
        cursor = next_cursor
    return tuple(groups)


def _is_version_guard_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(token in message for token in ("pitch", "energy", "shape", "size", "length", "tensor"))


def _to_mono_float32(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("audio output must contain at least one sample")
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        if array.shape[0] == 1:
            return array[0]
        if array.shape[1] == 1:
            return array[:, 0]
        return np.mean(array, axis=0, dtype=np.float32)
    raise ValueError("audio output must be one- or two-dimensional")


def _store_chunk_record(
    path: Path,
    record: ChunkRenderRecord,
    *,
    replace_existing: bool,
) -> ChunkRenderRecord:
    key = (record.protocol_id, record.song_id, record.chunk_index)
    if replace_existing:
        _replace_chunk_record(path, record)
    else:
        append_chunk_record(path, record)
    return read_chunk_record_index(path)[key]


def _replace_chunk_record(path: Path, record: ChunkRenderRecord) -> None:
    existing = read_chunk_record_index(path)
    key = (record.protocol_id, record.song_id, record.chunk_index)
    if key not in existing:
        raise ValueError(f"cannot replace missing chunk record for {key}")

    path.parent.mkdir(parents=True, exist_ok=True)
    original_mode = path.stat().st_mode & 0o7777
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for existing_key, existing_record in existing.items():
                if existing_key == key:
                    continue
                handle.write(canonical_json_dumps(existing_record.to_payload()))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        append_chunk_record(temporary_path, record)
        with temporary_path.open("rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary_path, original_mode)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_audio_wav(path: Path | str, audio: Any, *, sample_rate_hz: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, sample_rate_hz, _to_mono_float32(audio))


def _write_silence_wav(path: Path | str, *, frame_count: int, sample_rate_hz: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, sample_rate_hz, np.zeros(frame_count, dtype=np.int16))


def _timing_sidecar_path(output_path: Path | str) -> Path:
    return Path(output_path).with_suffix(".timing.json")


def _write_timing_sidecar(output_path: Path | str, plan: FastPitchRenderPlan) -> Path:
    destination = _timing_sidecar_path(output_path)
    payload = {
        "tokenizer_labels": list(plan.tokenizer_labels),
        "duration_frames": list(plan.duration_frames),
        "anchor_error_frames": list(plan.anchor_error_frames),
        "compressed_consonant_regions": list(plan.compressed_consonant_regions),
        "grapheme_fallback_words": list(plan.grapheme_fallback_words),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical_json_dumps(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return destination


def _error_payload(
    error: Exception | None,
    *,
    attempts: int,
    plan: FastPitchRenderPlan | None,
    prosody_controls: str,
) -> str:
    payload = {
        "message": str(error) if error is not None else "unknown FastPitch backend error",
        "attempts": attempts,
        "prosody_controls": prosody_controls,
        "tokenizer_labels": list(plan.tokenizer_labels) if plan is not None else [],
        "duration_frames": list(plan.duration_frames) if plan is not None else [],
        "anchor_error_frames": list(plan.anchor_error_frames) if plan is not None else [],
        "compressed_consonant_regions": list(plan.compressed_consonant_regions) if plan is not None else [],
        "grapheme_fallback_words": list(plan.grapheme_fallback_words) if plan is not None else [],
    }
    return canonical_json_dumps(payload)


def _progress_line(
    record: ChunkRenderRecord,
    *,
    plan: FastPitchRenderPlan | None,
    prosody_controls: str,
) -> str:
    duration_frames = sum(plan.duration_frames) if plan is not None else 0
    max_anchor_error = max((abs(value) for value in (plan.anchor_error_frames if plan is not None else ())), default=0)
    grapheme_fallback_words = ",".join(plan.grapheme_fallback_words) if plan is not None else ""
    return (
        f"protocol={record.protocol_id.value} song_id={record.song_id} chunk_index={record.chunk_index} "
        f"success={int(record.success)} attempts={record.attempts} sample_rate_hz={record.sample_rate_hz} "
        f"prosody_controls={prosody_controls} duration_frames={duration_frames} "
        f"tokenizer_labels={len(plan.tokenizer_labels) if plan is not None else 0} "
        f"max_anchor_error_frames={max_anchor_error} "
        f"grapheme_fallback_words={grapheme_fallback_words or 'none'} output_path={record.output_path}"
    )


def _request_from_payload(payload: dict[str, Any]) -> TwoBarRenderRequest:
    syllables = tuple(_syllable_from_payload(item) for item in payload["syllables"])
    return TwoBarRenderRequest(
        song_id=str(payload["song_id"]),
        chunk_index=int(payload["chunk_index"]),
        start_bar=int(payload["start_bar"]),
        end_bar=int(payload["end_bar"]),
        text=str(payload["text"]),
        syllables=syllables,
    )


def _syllable_from_payload(payload: dict[str, Any]) -> SyllableTarget:
    return SyllableTarget(
        word=str(payload["word"]),
        index_in_word=int(payload["index_in_word"]),
        phonemes=tuple(str(phone) for phone in payload["phonemes"]),
        lexical_stress=int(payload["lexical_stress"]),
        target_stress=float(payload["target_stress"]),
        boundary_strength=int(payload["boundary_strength"]),
        absolute_tick=int(payload["absolute_tick"]),
        tick_in_chunk=int(payload["tick_in_chunk"]),
        target_seconds=float(payload["target_seconds"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
