"""MOSS-TTS global-duration backend for the rap audio protocol comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
from streammuse.experiments.rap_audio_protocols.timing import moss_token_target


MODEL_ID = "OpenMOSS-Team/MOSS-TTS-v1.5"
LANGUAGE = "English"
RAP_INSTRUCTION = "clear, rhythmically spoken rap with restrained pitch"
GENERATION_MODE = "generation"
MAX_NEW_TOKENS = 256
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_SEED = 20260816
CAMPAIGN_MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CAMPAIGN_MANIFEST_NAME = "campaign_manifest.json"
GENERATION_KWARGS = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "audio_temperature": 1.7,
    "audio_top_p": 0.8,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.0,
}
STYLE_INSTRUCTION_CAVEAT = (
    "The shared MOSS processor accepts an instruction field, but MOSS-TTS-v1.5 does not document "
    "style-following as a guaranteed model capability."
)
DETERMINISTIC_SEED_CAVEAT = (
    "Official deterministic seeding is unsupported by the documented MOSS-TTS-v1.5 API; this backend "
    "uses best-effort PyTorch seeding on each attempt."
)


@dataclass(frozen=True)
class MossBackendRuntime:
    processor: Any
    model: Any
    torch_module: Any
    torchaudio_module: Any
    device: str
    model_id: str
    attn_implementation: str
    dtype: Any

    @property
    def sample_rate_hz(self) -> int:
        return int(self.processor.model_config.sampling_rate)


def create_runtime(*, model_id: str = MODEL_ID, device: str = "cuda:0") -> MossBackendRuntime:
    """Lazily load heavy runtime dependencies and construct the MOSS model stack."""
    torch_module = importlib.import_module("torch")
    torchaudio_module = importlib.import_module("torchaudio")
    transformers_module = importlib.import_module("transformers")
    auto_model = getattr(transformers_module, "AutoModel")
    auto_processor = getattr(transformers_module, "AutoProcessor")

    dtype = torch_module.bfloat16 if _is_cuda_device(device) else torch_module.float32
    attn_implementation = _resolve_attn_implementation(torch_module, device=device, dtype=dtype)
    _configure_torch_backends(torch_module)

    processor = auto_processor.from_pretrained(model_id, trust_remote_code=True)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None and hasattr(audio_tokenizer, "to"):
        processor.audio_tokenizer = audio_tokenizer.to(device)

    model = auto_model.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=dtype,
    ).to(device)
    if hasattr(model, "eval"):
        model.eval()

    return MossBackendRuntime(
        processor=processor,
        model=model,
        torch_module=torch_module,
        torchaudio_module=torchaudio_module,
        device=device,
        model_id=model_id,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )


def render_requests(
    *,
    requests: Sequence[TwoBarRenderRequest],
    output_dir: Path | str,
    record_path: Path | str,
    reference_wav: Path | str,
    runtime: MossBackendRuntime | None = None,
    campaign_manifest_path: Path | str | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_seed: int = DEFAULT_BASE_SEED,
    progress_stream: Any = None,
) -> tuple[ChunkRenderRecord, ...]:
    """Render all pending requests sequentially and append auditable JSONL records."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    output_root = Path(output_dir)
    ledger_path = Path(record_path)
    reference_path = Path(reference_wav)
    stream = progress_stream if progress_stream is not None else sys.stdout
    active_runtime = runtime if runtime is not None else create_runtime()

    _write_progress_line(stream, f"backend_note {DETERMINISTIC_SEED_CAVEAT}")
    _write_progress_line(stream, f"backend_note {STYLE_INSTRUCTION_CAVEAT}")

    existing = read_chunk_record_index(ledger_path)
    manifest_path = (
        Path(campaign_manifest_path)
        if campaign_manifest_path is not None
        else ledger_path.with_name(DEFAULT_CAMPAIGN_MANIFEST_NAME)
    )
    _ensure_campaign_manifest(
        manifest_path,
        runtime=active_runtime,
        reference_wav=reference_path,
        max_retries=max_retries,
        base_seed=base_seed,
        has_resumable_outputs=_has_resumable_outputs(existing, requests=requests, output_root=output_root),
    )

    rendered: list[ChunkRenderRecord] = []
    for request in requests:
        key = (ProtocolId.MOSS_GLOBAL, request.song_id, request.chunk_index)
        output_path = _chunk_output_path(output_root, request)
        if chunk_record_is_complete(
            ledger_path,
            output_path,
            request=request,
            protocol_id=ProtocolId.MOSS_GLOBAL,
        ):
            record = existing[key]
            rendered.append(record)
            _write_progress_line(
                stream,
                (
                    f"song={request.song_id} chunk={request.chunk_index:03d} status=skip "
                    f"attempts={record.attempts} output={record.output_path or '-'}"
                ),
            )
            continue
        record = _render_single_request(
            request=request,
            output_path=output_path,
            record_path=ledger_path,
            reference_wav=reference_path,
            runtime=active_runtime,
            max_retries=max_retries,
            base_seed=base_seed,
            progress_stream=stream,
            replace_existing=key in existing,
        )
        existing[key] = record
        rendered.append(record)
    return tuple(rendered)


def write_campaign_manifest(
    path: Path | str,
    *,
    runtime: MossBackendRuntime,
    reference_wav: Path | str,
    max_retries: int,
    base_seed: int,
) -> None:
    payload = campaign_manifest_payload(
        runtime=runtime,
        reference_wav=reference_wav,
        max_retries=max_retries,
        base_seed=base_seed,
    )
    _atomic_write_json(Path(path), payload)


def campaign_manifest_payload(
    *,
    runtime: MossBackendRuntime,
    reference_wav: Path | str,
    max_retries: int,
    base_seed: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": CAMPAIGN_MANIFEST_SCHEMA_VERSION,
        "protocol_id": ProtocolId.MOSS_GLOBAL.value,
        "model_id": runtime.model_id,
        "device": runtime.device,
        "sample_rate_hz": runtime.sample_rate_hz,
        "attn_implementation": runtime.attn_implementation,
        "dtype": _dtype_name(runtime.dtype),
        "generation_mode": GENERATION_MODE,
        "generation_kwargs": dict(GENERATION_KWARGS),
        "reference_wav": str(Path(reference_wav)),
        "style_instruction": RAP_INSTRUCTION,
        "style_instruction_caveat": STYLE_INSTRUCTION_CAVEAT,
        "deterministic_seed_caveat": DETERMINISTIC_SEED_CAVEAT,
        "max_retries": max_retries,
        "base_seed": base_seed,
    }
    payload["configuration_sha256"] = hashlib.sha256(
        canonical_json_dumps(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _ensure_campaign_manifest(
    path: Path,
    *,
    runtime: MossBackendRuntime,
    reference_wav: Path,
    max_retries: int,
    base_seed: int,
    has_resumable_outputs: bool,
) -> None:
    expected = campaign_manifest_payload(
        runtime=runtime,
        reference_wav=reference_wav,
        max_retries=max_retries,
        base_seed=base_seed,
    )
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            if has_resumable_outputs:
                raise ValueError(f"campaign manifest conflicts at {path}") from exc
        else:
            if existing == expected:
                return
            legacy_expected = {
                key: value
                for key, value in expected.items()
                if key not in {"schema_version", "configuration_sha256"}
            }
            if existing == legacy_expected:
                _atomic_write_json(path, expected)
                return
            if has_resumable_outputs:
                raise ValueError(f"campaign manifest conflicts at {path}")
    elif has_resumable_outputs:
        raise ValueError(f"campaign manifest conflicts at {path}: missing for resumable outputs")

    _atomic_write_json(path, expected)


def _has_resumable_outputs(
    existing: dict[tuple[ProtocolId, str, int], ChunkRenderRecord],
    *,
    requests: Sequence[TwoBarRenderRequest],
    output_root: Path,
) -> bool:
    selected = {
        (ProtocolId.MOSS_GLOBAL, request.song_id, request.chunk_index): request
        for request in requests
    }
    for record in existing.values():
        if not record.success or record.output_path is None or record.output_sha256 is None:
            continue
        output_path = Path(record.output_path)
        try:
            if not output_path.is_file() or file_sha256(output_path) != record.output_sha256:
                continue
        except OSError:
            continue
        key = (record.protocol_id, record.song_id, record.chunk_index)
        request = selected.get(key)
        if request is not None and (
            record.request_sha256 != request.sha256
            or output_path.resolve() != _chunk_output_path(output_root, request).resolve()
        ):
            continue
        return True
    return False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            handle.write(canonical_json_dumps(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_requests_jsonl(path: Path | str) -> tuple[TwoBarRenderRequest, ...]:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--record-path", required=True, type=Path)
    parser.add_argument("--reference-wav", required=True, type=Path)
    parser.add_argument("--campaign-manifest", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--song", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    requests = load_requests_jsonl(args.request_jsonl)
    if args.song:
        requests = tuple(request for request in requests if request.song_id == args.song)
    if not requests:
        raise ValueError("no requests selected for rendering")

    runtime = create_runtime(model_id=args.model_id, device=args.device)
    records = render_requests(
        requests=requests,
        output_dir=args.output_dir,
        record_path=args.record_path,
        reference_wav=args.reference_wav,
        runtime=runtime,
        campaign_manifest_path=args.campaign_manifest,
        max_retries=args.max_retries,
        base_seed=args.base_seed,
    )
    success_count = sum(1 for record in records if record.success)
    failure_count = len(records) - success_count
    _write_progress_line(
        sys.stdout,
        f"summary rendered={len(records)} success={success_count} failed={failure_count} device={args.device}",
    )
    return 0 if failure_count == 0 else 1


def _render_single_request(
    *,
    request: TwoBarRenderRequest,
    output_path: Path,
    record_path: Path,
    reference_wav: Path,
    runtime: MossBackendRuntime,
    max_retries: int,
    base_seed: int,
    progress_stream: Any,
    replace_existing: bool,
) -> ChunkRenderRecord:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        seed = _seed_for_attempt(base_seed=base_seed, request=request, attempt=attempt)
        _seed_torch_best_effort(runtime.torch_module, seed=seed)
        try:
            _generate_chunk(
                request=request,
                output_path=output_path,
                reference_wav=reference_wav,
                runtime=runtime,
            )
            metadata = validate_wav_metadata(
                output_path,
                expected_sample_rate_hz=runtime.sample_rate_hz,
                expected_channels=1,
            )
            record = ChunkRenderRecord(
                protocol_id=ProtocolId.MOSS_GLOBAL,
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
            _write_progress_line(
                progress_stream,
                (
                    f"song={request.song_id} chunk={request.chunk_index:03d} status=success "
                    f"attempt={attempt}/{max_retries} tokens={moss_token_target(request)} "
                    f"sample_rate_hz={runtime.sample_rate_hz} output={output_path}"
                ),
            )
            return stored
        except Exception as error:  # pragma: no cover - exercised via tests with fake exceptions
            last_error = error
            _write_progress_line(
                progress_stream,
                (
                    f"song={request.song_id} chunk={request.chunk_index:03d} status=retry "
                    f"attempt={attempt}/{max_retries} error={error}"
                ),
            )

    silence_frame_count = int(round(request.duration_seconds * runtime.sample_rate_hz))
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
        protocol_id=ProtocolId.MOSS_GLOBAL,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=False,
        output_path=str(output_path),
        output_sha256=file_sha256(output_path),
        source_chunk_sha256=None,
        sample_rate_hz=metadata.sample_rate_hz,
        attempts=max_retries,
        error=str(last_error) if last_error is not None else "unknown render failure",
    )
    stored_failure = _store_chunk_record(record_path, failure, replace_existing=replace_existing)
    _write_progress_line(
        progress_stream,
        (
            f"song={request.song_id} chunk={request.chunk_index:03d} status=failed "
            f"attempt={max_retries}/{max_retries} error={stored_failure.error}"
        ),
    )
    return stored_failure


def _write_silence_wav(path: Path, *, frame_count: int, sample_rate_hz: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(b"\x00\x00" * frame_count)


def _generate_chunk(
    *,
    request: TwoBarRenderRequest,
    output_path: Path,
    reference_wav: Path,
    runtime: MossBackendRuntime,
) -> None:
    user_message = runtime.processor.build_user_message(
        text=request.text,
        language=LANGUAGE,
        reference=[str(reference_wav)],
        tokens=moss_token_target(request),
        instruction=RAP_INSTRUCTION,
    )
    batch = runtime.processor([[user_message]], mode=GENERATION_MODE)
    outputs = runtime.model.generate(
        input_ids=batch["input_ids"].to(runtime.device),
        attention_mask=batch["attention_mask"].to(runtime.device),
        **GENERATION_KWARGS,
    )
    decoded_message = runtime.processor.decode(outputs)[0]
    audio = decoded_message.audio_codes_list[0]
    waveform = _normalise_audio_for_save(audio, torch_module=runtime.torch_module)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime.torchaudio_module.save(str(output_path), waveform, runtime.sample_rate_hz)


def _normalise_audio_for_save(audio: Any, *, torch_module: Any) -> Any:
    as_tensor = getattr(torch_module, "as_tensor", None)
    if not callable(as_tensor):
        raise TypeError("torch runtime must provide as_tensor for audio serialization")
    tensor = as_tensor(audio, dtype=torch_module.float32)
    tensor = tensor.detach().cpu().float()
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        shape = tuple(tensor.shape)
        if shape[0] == 1:
            return tensor
        if shape[1] == 1:
            return tensor.transpose(0, 1)
    raise ValueError(f"expected mono audio tensor, got shape {tuple(tensor.shape)}")


def _chunk_output_path(output_root: Path, request: TwoBarRenderRequest) -> Path:
    return output_root / request.song_id / f"chunk-{request.chunk_index:03d}.wav"


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


def _seed_for_attempt(*, base_seed: int, request: TwoBarRenderRequest, attempt: int) -> int:
    return base_seed + request.chunk_index * 1000 + (attempt - 1)


def _seed_torch_best_effort(torch_module: Any, *, seed: int) -> None:
    manual_seed = getattr(torch_module, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        manual_seed_all = getattr(cuda, "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)


def _resolve_attn_implementation(torch_module: Any, *, device: str, dtype: Any) -> str:
    if _is_cuda_device(device):
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
        if has_flash_attn and dtype in {getattr(torch_module, "float16", object()), getattr(torch_module, "bfloat16", object())}:
            get_device_capability = getattr(torch_module.cuda, "get_device_capability", None)
            capability = get_device_capability(device) if callable(get_device_capability) else (0, 0)
            if capability[0] >= 8:
                return "flash_attention_2"
        return "sdpa"
    return "eager"


def _configure_torch_backends(torch_module: Any) -> None:
    cuda_backends = getattr(getattr(torch_module, "backends", None), "cuda", None)
    if cuda_backends is None:
        return
    for method_name, value in (
        ("enable_cudnn_sdp", False),
        ("enable_flash_sdp", True),
        ("enable_mem_efficient_sdp", True),
        ("enable_math_sdp", True),
    ):
        method = getattr(cuda_backends, method_name, None)
        if callable(method):
            method(value)


def _is_cuda_device(device: str) -> bool:
    return device.startswith("cuda")


def _request_from_payload(payload: dict[str, Any]) -> TwoBarRenderRequest:
    return TwoBarRenderRequest(
        song_id=str(payload["song_id"]),
        chunk_index=int(payload["chunk_index"]),
        start_bar=int(payload["start_bar"]),
        end_bar=int(payload["end_bar"]),
        text=str(payload["text"]),
        syllables=tuple(_syllable_from_payload(item) for item in payload["syllables"]),
        tempo_bpm=float(payload.get("tempo_bpm", 90.0)),
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


def _dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype))


def _write_progress_line(stream: Any, message: str) -> None:
    print(message, file=stream, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
