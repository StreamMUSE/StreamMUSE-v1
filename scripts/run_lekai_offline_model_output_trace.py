#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter import (
    PianoContinuationAdapter,
)
from streammuse.infrastructure.inference.lekai_continuation_model.Token2Midi import MidiConverter
from streammuse.infrastructure.inference.model_output_trace import ModelInputTracer, ModelOutputTracer
from streammuse.infrastructure.inference.runtime_device import resolve_device, resolve_dtype


def install_model_output_hook(
    wrapper: Any,
    tracer: ModelOutputTracer,
    *,
    input_file: str,
    input_tracer: ModelInputTracer | None = None,
) -> Callable[..., Any]:
    original = wrapper._generate_one_beat
    target_beat = 0

    def traced_generate_one_beat(self, generated, *args, **kwargs):
        nonlocal target_beat
        prompt_tokens = [int(token) for token in generated[0].detach().cpu().tolist()]
        if input_tracer is not None:
            input_tracer.record(
                input_tokens=prompt_tokens,
                target_beat=target_beat,
                input_file=input_file,
                metadata={
                    "slot_kind": "beat",
                    "attention_mask": None,
                    "position_ids": None,
                    "use_cache": True,
                },
            )
        output_tokens, updated = original(generated, *args, **kwargs)
        tracer.record(
            output_tokens=output_tokens,
            target_beat=target_beat,
            input_file=input_file,
            prompt_tokens=prompt_tokens,
            metadata={"slot_kind": "beat"},
        )
        target_beat += 1
        return output_tokens, updated

    wrapper._generate_one_beat = types.MethodType(traced_generate_one_beat, wrapper)
    return original


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace the current RT offline model output per beat")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npz-file", type=Path, required=True)
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--input-trace-path", type=Path)
    parser.add_argument("--output-midi", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16"])
    parser.add_argument("--gt-prefix-beats", type=int, default=0)
    parser.add_argument("--bpm", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    npz_file = args.npz_file.expanduser().resolve()
    trace_path = args.trace_path.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not npz_file.is_file():
        raise FileNotFoundError(npz_file)

    device = resolve_device(args.device)
    dtype = resolve_dtype(device, args.dtype)
    adapter = PianoContinuationAdapter.from_checkpoint(
        str(checkpoint),
        device=device,
        dtype=dtype,
        use_cache=True,
    )

    save_dict = np.load(npz_file, allow_pickle=True)
    metadata = dict(save_dict["metadata"].item())
    if args.bpm is not None:
        metadata["bpm"] = int(args.bpm)
    measures = [save_dict[f"measure_{index}"] for index in range(int(metadata["num_measures"]))]
    generation = adapter.tokenizer.build_generation_schedule(
        measures=measures,
        metadata=metadata,
        gt_prefix_beats=int(args.gt_prefix_beats),
    )

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.unlink(missing_ok=True)
    tracer = ModelOutputTracer(mode="offline", path=trace_path)
    input_tracer = None
    if args.input_trace_path is not None:
        input_trace_path = args.input_trace_path.expanduser().resolve()
        input_trace_path.parent.mkdir(parents=True, exist_ok=True)
        input_trace_path.unlink(missing_ok=True)
        input_tracer = ModelInputTracer(mode="offline", path=input_trace_path)
    install_model_output_hook(
        adapter.wrapper,
        tracer,
        input_file=str(npz_file),
        input_tracer=input_tracer,
    )

    output_beats, generated = adapter.wrapper.generate_accompaniment(
        initial_tokens=generation["initial_tokens"],
        schedule=generation["schedule"],
        vocab=adapter.tokenizer.vocab,
        device=device,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.0,
        verbose=False,
    )
    output_midi = args.output_midi.expanduser().resolve() if args.output_midi else None
    if output_midi is not None:
        MidiConverter(adapter.tokenizer).beats_to_midi(
            mel_beats=generation["mel_beats"],
            acc_beats=output_beats,
            tempo=int(metadata.get("bpm", 120) or 120),
            save_path=str(output_midi),
        )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "npz_file": str(npz_file),
                "trace_path": str(trace_path),
                "input_trace_path": (
                    str(args.input_trace_path.expanduser().resolve())
                    if args.input_trace_path is not None
                    else None
                ),
                "output_midi": str(output_midi) if output_midi is not None else None,
                "output_beat_count": len(output_beats),
                "generated_token_count": int(generated.numel()),
                "sampling": "greedy via top_k=1",
                "bpm": int(metadata.get("bpm", 120) or 120),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
