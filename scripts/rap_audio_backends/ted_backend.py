"""TED-TTS local-duration backend for the rap audio protocol comparison."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

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
from streammuse.experiments.rap_audio_protocols.timing import TimedTextSegment, build_ted_segments


TED_SAMPLE_RATE_HZ = 22_050
TED_DURATION_TOKEN_SECONDS = 0.02
TED_INFERENCE_METHODS = ("hmm", "max_head")
DEFAULT_TED_INFERENCE_METHOD = "hmm"
TED_DURATION_MODE = "both"
TED_SEGMENT_DESCRIPTION = "clear, confident, rhythmic spoken rap with restrained melody"
TED_DETERMINISM_NOTE = "use_random=False does not guarantee determinism because TED still samples"
DEFAULT_MAX_ATTEMPTS = 3
TED_FIXED_GENERATION_SETTINGS = {
    "emo_alpha": 0,
    "use_emo_text": True,
    "verbose": True,
    "use_random": False,
    "do_sample": True,
    "top_p": 0.8,
    "top_k": 30,
    "temperature": 0.8,
    "num_beams": 3,
    "repetition_penalty": 10.0,
    "length_penalty": 0.0,
    "max_mel_tokens": 850,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--requests-jsonl", required=True, type=Path)
    parser.add_argument("--records-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference-wav", required=True, type=Path)
    parser.add_argument("--ted-checkout", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--cfg-path", required=True, type=Path)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument(
        "--inference-method",
        choices=TED_INFERENCE_METHODS,
        default=DEFAULT_TED_INFERENCE_METHOD,
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    ted_model_factory: Callable[..., Any] | None = None,
) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)
    if args.max_attempts <= 0:
        print("error: --max-attempts must be positive", file=sys.stderr)
        return 2

    factory = ted_model_factory or create_ted_model
    try:
        ted_model = factory(
            model_dir=args.model_dir,
            cfg_path=args.cfg_path,
            ted_checkout=args.ted_checkout,
        )
    except Exception as exc:  # pragma: no cover - exercised only with real TED installs
        print(f"error: unable to initialize TED backend: {exc}", file=sys.stderr)
        return 2

    try:
        records = render_pending_requests(
            request_path=args.requests_jsonl,
            record_path=args.records_jsonl,
            output_dir=args.output_dir,
            reference_wav_path=args.reference_wav,
            ted_model=ted_model,
            max_attempts=args.max_attempts,
            inference_method=args.inference_method,
        )
    except Exception as exc:
        print(f"error: TED backend failed: {exc}", file=sys.stderr)
        return 2

    return 1 if any(not record.success for record in records) else 0


def create_ted_model(*, model_dir: Path, cfg_path: Path, ted_checkout: Path) -> Any:
    index_tts2_class = _load_indextts2_class(ted_checkout)
    return index_tts2_class(
        model_dir=str(model_dir),
        cfg_path=str(cfg_path),
        is_fp16=True,
    )


def _load_indextts2_class(ted_checkout: Path) -> type[Any]:
    checkout = Path(ted_checkout).resolve()
    original_path = list(sys.path)
    cached_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "indextts" or name.startswith("indextts.")
    }
    try:
        for name in cached_modules:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(checkout))
        importlib.invalidate_caches()
        infer_module = importlib.import_module("indextts.infer_v2")
        module_path = getattr(infer_module, "__file__", None)
        if module_path is None or not _path_is_within(Path(module_path), checkout):
            raise ImportError(f"indextts.infer_v2 was not loaded from TED checkout: {checkout}")
        index_tts2_class = getattr(infer_module, "IndexTTS2")
    finally:
        for name in tuple(sys.modules):
            if name == "indextts" or name.startswith("indextts."):
                sys.modules.pop(name, None)
        sys.modules.update(cached_modules)
        sys.path[:] = original_path
        importlib.invalidate_caches()
    return index_tts2_class


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
    except ValueError:
        return False
    return True


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
    reference_wav_path: Path | str,
    ted_model: Any,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    inference_method: str = DEFAULT_TED_INFERENCE_METHOD,
    segment_builder: Callable[[TwoBarRenderRequest], tuple[TimedTextSegment, ...]] = build_ted_segments,
) -> tuple[ChunkRenderRecord, ...]:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    records: list[ChunkRenderRecord] = []
    output_root = Path(output_dir)
    reference = Path(reference_wav_path)
    ledger_path = Path(record_path)
    _ensure_inference_config(ledger_path, inference_method)
    requests = load_requests(request_path)

    for request in requests:
        output_path = output_root / request.song_id / f"chunk-{request.chunk_index:03d}.wav"
        if chunk_record_is_complete(
            ledger_path,
            output_path,
            request=request,
            protocol_id=ProtocolId.TED_LOCAL,
        ):
            continue
        record = _render_with_retries(
            request=request,
            ted_model=ted_model,
            reference_wav_path=reference,
            output_path=output_path,
            max_attempts=max_attempts,
            inference_method=inference_method,
            segment_builder=segment_builder,
        )
        _store_chunk_record(ledger_path, record)
        records.append(record)
        print(_progress_line(record, inference_method=inference_method))
    return tuple(records)


def _store_chunk_record(path: Path, record: ChunkRenderRecord) -> None:
    existing = read_chunk_record_index(path)
    key = (record.protocol_id, record.song_id, record.chunk_index)
    if key not in existing:
        append_chunk_record(path, record)
        return

    existing[key] = record
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
            for item in existing.values():
                handle.write(canonical_json_dumps(item.to_payload()))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def render_request(
    *,
    request: TwoBarRenderRequest,
    ted_model: Any,
    reference_wav_path: Path | str,
    output_path: Path | str,
    inference_method: str = DEFAULT_TED_INFERENCE_METHOD,
    segment_builder: Callable[[TwoBarRenderRequest], tuple[TimedTextSegment, ...]] = build_ted_segments,
) -> ChunkRenderRecord:
    infer_kwargs = build_infer_kwargs(
        request=request,
        reference_wav_path=reference_wav_path,
        output_path=output_path,
        inference_method=inference_method,
        segment_builder=segment_builder,
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ted_model.infer(**infer_kwargs)
    metadata = validate_wav_metadata(output_path, expected_sample_rate_hz=TED_SAMPLE_RATE_HZ, expected_channels=1)
    return ChunkRenderRecord(
        protocol_id=ProtocolId.TED_LOCAL,
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


def build_infer_kwargs(
    *,
    request: TwoBarRenderRequest,
    reference_wav_path: Path | str,
    output_path: Path | str,
    inference_method: str = DEFAULT_TED_INFERENCE_METHOD,
    segment_builder: Callable[[TwoBarRenderRequest], tuple[TimedTextSegment, ...]] = build_ted_segments,
) -> dict[str, Any]:
    _validate_inference_method(inference_method)
    segments = tuple(segment_builder(request))
    if not segments:
        raise ValueError("TED requests require at least one segment")
    if any(not segment.text_with_spacing for segment in segments):
        raise ValueError("TED segments must be non-empty")

    text_segments = [segment.text_with_spacing for segment in segments]
    emotion_segments = [TED_SEGMENT_DESCRIPTION for _ in segments]
    duration_tokens = [int(segment.target_seconds / TED_DURATION_TOKEN_SECONDS) for segment in segments]
    if any(token <= 0 for token in duration_tokens):
        raise ValueError("TED target duration tokens must be positive")

    text = "|".join(text_segments)
    emo_text = "|".join(emotion_segments)
    expected_count = len(segments)
    if len(text.split("|")) != expected_count or len(emo_text.split("|")) != expected_count or len(duration_tokens) != expected_count:
        raise ValueError("TED segment counts must match before invocation")

    return {
        "spk_audio_prompt": str(Path(reference_wav_path)),
        "text": text,
        "output_path": str(Path(output_path)),
        "emo_audio_prompt": None,
        "emo_text": emo_text,
        "target_duration_tokens": duration_tokens,
        "duration_mode": TED_DURATION_MODE,
        **TED_FIXED_GENERATION_SETTINGS,
        "method": inference_method,
    }


def chunk_frame_count(request: TwoBarRenderRequest) -> int:
    return int(round(request.duration_seconds * TED_SAMPLE_RATE_HZ))


def _render_with_retries(
    *,
    request: TwoBarRenderRequest,
    ted_model: Any,
    reference_wav_path: Path,
    output_path: Path,
    max_attempts: int,
    inference_method: str,
    segment_builder: Callable[[TwoBarRenderRequest], tuple[TimedTextSegment, ...]],
) -> ChunkRenderRecord:
    last_error: Exception | None = None
    last_infer_kwargs: dict[str, Any] | None = None
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_attempts + 1):
        try:
            infer_kwargs = build_infer_kwargs(
                request=request,
                reference_wav_path=reference_wav_path,
                output_path=output_path,
                inference_method=inference_method,
                segment_builder=segment_builder,
            )
            last_infer_kwargs = infer_kwargs
            ted_model.infer(**infer_kwargs)
            metadata = validate_wav_metadata(output_path, expected_sample_rate_hz=TED_SAMPLE_RATE_HZ, expected_channels=1)
            return ChunkRenderRecord(
                protocol_id=ProtocolId.TED_LOCAL,
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
        except Exception as exc:
            last_error = exc

    _write_silence_wav(output_path, frame_count=chunk_frame_count(request))
    return ChunkRenderRecord(
        protocol_id=ProtocolId.TED_LOCAL,
        song_id=request.song_id,
        chunk_index=request.chunk_index,
        request_sha256=request.sha256,
        success=False,
        output_path=str(output_path),
        output_sha256=file_sha256(output_path),
        source_chunk_sha256=None,
        sample_rate_hz=TED_SAMPLE_RATE_HZ,
        attempts=max_attempts,
        error=_error_payload(last_error, max_attempts=max_attempts, infer_kwargs=last_infer_kwargs),
    )


def _write_silence_wav(path: Path | str, *, frame_count: int) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(destination, TED_SAMPLE_RATE_HZ, np.zeros(frame_count, dtype=np.int16))


def _error_payload(
    error: Exception | None,
    *,
    max_attempts: int,
    infer_kwargs: dict[str, Any] | None,
) -> str:
    payload = {
        "message": str(error) if error is not None else "unknown TED backend error",
        "attempts": max_attempts,
        "determinism_note": TED_DETERMINISM_NOTE,
        "infer_kwargs": infer_kwargs,
    }
    return canonical_json_dumps(payload)


def _progress_line(record: ChunkRenderRecord, *, inference_method: str) -> str:
    return (
        f"protocol={record.protocol_id.value} song_id={record.song_id} chunk_index={record.chunk_index} "
        f"success={int(record.success)} attempts={record.attempts} sample_rate_hz={record.sample_rate_hz} "
        f"output_path={record.output_path} inference_method={inference_method} "
        f"determinism_note={TED_DETERMINISM_NOTE}"
    )


def _ensure_inference_config(ledger_path: Path, inference_method: str) -> None:
    payload = _inference_config_payload(inference_method)
    config_path = ledger_path.with_name("inference_config.json")
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"inference config conflicts at {config_path}") from exc
        if existing != payload:
            raise ValueError(f"inference config conflicts at {config_path}")
        return

    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(canonical_json_dumps(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, config_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _inference_config_payload(inference_method: str) -> dict[str, Any]:
    _validate_inference_method(inference_method)
    return {
        "method": inference_method,
        "duration_mode": TED_DURATION_MODE,
        "determinism_note": TED_DETERMINISM_NOTE,
        "generation_settings": dict(TED_FIXED_GENERATION_SETTINGS),
    }


def _validate_inference_method(inference_method: str) -> None:
    if inference_method not in TED_INFERENCE_METHODS:
        raise ValueError(f"unsupported TED inference method: {inference_method}")


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
