#!/usr/bin/env python
"""Run Prompt batch selection and continuation through the pure NPZ offline path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from streammuse.infrastructure.inference.lekai_continuation_model.config import (
    ModelConfig as ContinuationConfig,
)
from streammuse.infrastructure.inference.lekai_continuation_model.inference_adapter import (
    PianoContinuationAdapter,
)
from streammuse.infrastructure.inference.lekai_continuation_model.Token2Midi import (
    MidiConverter,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_batch_selector import (
    accompaniment_features_from_pianoroll,
    median_note_duration_from_pianoroll,
    score_prompt_batch_ppl,
    select_rule_s_candidate,
    select_rule_s_v2_candidate,
    trim_at_eos,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.config import (
    ModelConfig as PromptConfig,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.inference import (
    load_model as load_prompt_model,
)
from streammuse.infrastructure.inference.lekai_prompt_continuation.prompt_model.my_tokenizer import (
    PianoMusicTokenizer as PromptTokenizer,
)


TIMESTEPS_PER_BEAT = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Prompt Model candidates from one prepared NPZ, select "
            "candidate 1 and Rule-S, then run the direct offline continuation schedule."
        )
    )
    parser.add_argument("--npz-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-checkpoint", type=Path, required=True)
    parser.add_argument("--continuation-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--prompt-seed", type=int, default=20260813)
    parser.add_argument("--continuation-seeds", default="0")
    parser.add_argument("--prompt-prefix-beats", type=int, default=8)
    parser.add_argument(
        "--include-gt-accompaniment",
        action="store_true",
        help="Add a condition using the NPZ GT accompaniment for the Prompt prefix",
    )
    parser.add_argument("--prompt-num-bars", type=int, default=2)
    parser.add_argument("--prompt-max-new-tokens", type=int, default=1024)
    parser.add_argument("--prompt-temperature", type=float, default=0.8)
    parser.add_argument("--prompt-top-k", type=int, default=50)
    parser.add_argument("--prompt-top-p", type=float, default=0.95)
    parser.add_argument("--prompt-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--duration-weight", type=float, default=0.49)
    parser.add_argument("--duration-expected-log-ratio", type=float, default=0.0)
    parser.add_argument("--continuation-temperature", type=float, default=1.1)
    parser.add_argument("--continuation-top-k", type=int, default=0)
    parser.add_argument("--continuation-top-p", type=float, default=0.95)
    parser.add_argument("--continuation-repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--max-schedule-steps",
        type=int,
        default=0,
        help="Optional smoke-test cap; 0 runs the complete NPZ schedule",
    )
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("--continuation-seeds must contain unique integers")
    return seeds


def load_npz(path: Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=True) as payload:
        metadata = dict(payload["metadata"].item())
        measures = [
            np.asarray(payload[f"measure_{index}"])
            for index in range(int(metadata["num_measures"]))
        ]
    if not measures:
        raise ValueError(f"NPZ has no measures: {path}")
    if any(measure.shape[0] != 4 for measure in measures):
        raise ValueError("NPZ measures must use four channels: mel sustain/onset, acc sustain/onset")
    return measures, metadata


def build_prompt_tokens(
    tokenizer: PromptTokenizer,
    measures: list[np.ndarray],
    metadata: dict[str, Any],
    *,
    num_bars: int,
) -> torch.Tensor:
    if len(measures) < num_bars:
        raise ValueError(f"NPZ has {len(measures)} bars; Prompt requires {num_bars}")
    selected = measures[:num_bars]
    _, _, _, measure_beats = tokenizer._encode_measures(selected, metadata)
    parts = [
        torch.tensor(
            [
                tokenizer.vocab.bos_token_id,
                tokenizer.encode_time_sig(int(metadata["time_signature_idx"])),
                tokenizer.encode_bpm(float(metadata.get("bpm") or 120)),
            ],
            dtype=torch.long,
        )
    ]
    for beats in measure_beats:
        parts.append(torch.tensor([tokenizer.vocab.bar_token_id], dtype=torch.long))
        for melody, _accompaniment in beats:
            parts.append(torch.tensor([tokenizer.vocab.beat_marker], dtype=torch.long))
            parts.append(melody)
    return torch.cat(parts)


def convert_prompt_beat(prompt_tokenizer, continuation_tokenizer, beat: Any) -> torch.Tensor:
    pianoroll = prompt_tokenizer.decode_beats_to_pianoroll(
        [beat],
        track_marker_id=prompt_tokenizer.vocab.track_marker_acc,
    )
    beat_width = int(continuation_tokenizer.vocab.default_patch_w)
    if pianoroll.shape[2] < beat_width:
        pianoroll = np.pad(
            pianoroll,
            ((0, 0), (0, 0), (0, beat_width - pianoroll.shape[2])),
        )
    elif pianoroll.shape[2] > beat_width:
        pianoroll = pianoroll[:, :, :beat_width]
    tokens = continuation_tokenizer._codec.image_to_patch_tokens(
        pianoroll,
        strict_mode=True,
    )
    compressed = continuation_tokenizer.compress_tokens(
        tokens,
        track_marker=continuation_tokenizer.vocab.track_marker_acc,
    )
    return torch.tensor(compressed, dtype=torch.long)


def inject_prompt_prefix(schedule: list[Any], prefix: list[torch.Tensor]) -> list[Any]:
    output = []
    accompaniment_index = 0
    for step in schedule:
        if step.action in {"generate", "inject_gt"}:
            if accompaniment_index < len(prefix):
                output.append(replace(step, action="inject_gt", data=prefix[accompaniment_index]))
            else:
                output.append(replace(step, action="generate", data=None))
            accompaniment_index += 1
        else:
            output.append(step)
    if accompaniment_index < len(prefix):
        raise ValueError(
            f"schedule has only {accompaniment_index} accompaniment beats; "
            f"cannot inject {len(prefix)}"
        )
    return output


def candidate_from_sequence(
    sequence: torch.Tensor,
    *,
    candidate_number: int,
    prompt_tokenizer: PromptTokenizer,
    prefix_beats: int,
    ppl: dict[str, Any],
) -> tuple[dict[str, Any], list[Any]]:
    trimmed = trim_at_eos(sequence, int(prompt_tokenizer.vocab.eos_token_id))
    _melody, all_acc = prompt_tokenizer.parse_generated_sequence(trimmed)
    prefix = all_acc[:prefix_beats]
    if prefix:
        pianoroll = prompt_tokenizer.decode_beats_to_pianoroll(
            prefix,
            track_marker_id=prompt_tokenizer.vocab.track_marker_acc,
        )
    else:
        pianoroll = np.zeros((2, 88, prefix_beats * TIMESTEPS_PER_BEAT), dtype=np.uint8)
    features = accompaniment_features_from_pianoroll(
        pianoroll,
        length_ticks=prefix_beats * TIMESTEPS_PER_BEAT,
    )
    ppl_available = bool(ppl.get("available"))
    row = {
        "candidate_number": candidate_number,
        "generated_beats": len(all_acc),
        "required_beats": prefix_beats,
        "prompt_ppl_available": ppl_available,
        "prompt_ppl": float(ppl["ppl"]) if ppl_available else None,
        "prompt_ppl_scored_token_count": int(ppl.get("scored_token_count", 0)),
        "prompt_ppl_reason": ppl.get("reason"),
        "prompt_token_hash": hashlib.sha256(trimmed.numpy().tobytes()).hexdigest(),
        **features,
    }
    return row, prefix


def main() -> None:
    args = parse_args()
    npz_path = args.npz_file.expanduser().resolve()
    prompt_checkpoint = args.prompt_checkpoint.expanduser().resolve()
    continuation_checkpoint = args.continuation_checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path, label in (
        (npz_path, "NPZ"),
        (prompt_checkpoint, "Prompt checkpoint"),
        (continuation_checkpoint, "continuation checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.candidate_count < 2:
        raise ValueError("--candidate-count must be at least 2")
    output_dir.mkdir(parents=True, exist_ok=True)
    continuation_seeds = parse_seeds(args.continuation_seeds)
    measures, metadata = load_npz(npz_path)

    prompt_config = PromptConfig()
    prompt_tokenizer = PromptTokenizer(config=prompt_config)
    prompt_tokens = build_prompt_tokens(
        prompt_tokenizer,
        measures,
        metadata,
        num_bars=args.prompt_num_bars,
    )
    torch.manual_seed(args.prompt_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.prompt_seed)
    prompt_model = load_prompt_model(
        str(prompt_checkpoint),
        prompt_config,
        device=args.device,
        use_fp16=args.fp16,
    )
    prompt_started = time.perf_counter()
    generated = prompt_model.generate_music_batch(
        initial_tokens=prompt_tokens,
        batch_size=args.candidate_count,
        device=args.device,
        max_length=len(prompt_tokens) + args.prompt_max_new_tokens,
        temperature=args.prompt_temperature,
        top_k=args.prompt_top_k,
        top_p=args.prompt_top_p,
        repetition_penalty=args.prompt_repetition_penalty,
    )
    prompt_seconds = time.perf_counter() - prompt_started
    ppl_scores = score_prompt_batch_ppl(
        prompt_model,
        generated,
        prompt_token_count=len(prompt_tokens),
        device=args.device,
        tokenizer=prompt_tokenizer,
        max_acc_beats=args.prompt_prefix_beats,
    )

    candidates = []
    prompt_prefixes = []
    prefix_ticks = args.prompt_prefix_beats * TIMESTEPS_PER_BEAT
    melody_pianoroll = np.concatenate(
        [measure[:2] for measure in measures[: args.prompt_num_bars]],
        axis=2,
    )
    melody_median_duration = median_note_duration_from_pianoroll(
        melody_pianoroll,
        length_ticks=prefix_ticks,
    )
    for index in range(args.candidate_count):
        row, prefix = candidate_from_sequence(
            generated[index],
            candidate_number=index + 1,
            prompt_tokenizer=prompt_tokenizer,
            prefix_beats=args.prompt_prefix_beats,
            ppl=ppl_scores[index],
        )
        row["mel_median_note_duration_ticks"] = melody_median_duration
        candidates.append(row)
        prompt_prefixes.append(prefix)
    rule_s_v1 = select_rule_s_candidate(candidates)
    rule_s_v2 = select_rule_s_v2_candidate(
        rule_s_v1["candidates"],
        duration_weight=args.duration_weight,
        expected_log_duration_ratio=args.duration_expected_log_ratio,
    )

    del prompt_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    continuation_config = ContinuationConfig()
    continuation = PianoContinuationAdapter.from_checkpoint(
        str(continuation_checkpoint),
        device=args.device,
        dtype=torch.float16 if args.fp16 else torch.float32,
        use_cache=True,
    )
    tokenizer = continuation.tokenizer
    converter = MidiConverter(tokenizer)
    prepared = tokenizer.build_generation_schedule(
        measures,
        metadata,
        gt_prefix_beats=0,
        timesteps_per_beat=TIMESTEPS_PER_BEAT,
    )
    converter.gt_to_midi(str(npz_path), str(output_dir / "00_input_decoded.mid"))

    conditions: list[tuple[str, int | None, list[Any]]] = [
        ("batch_first", 0, prompt_prefixes[0]),
        (
            "rule_s_v1",
            int(rule_s_v1["selected_index"]),
            prompt_prefixes[int(rule_s_v1["selected_index"])],
        ),
        (
            "rule_s_v2",
            int(rule_s_v2["selected_index"]),
            prompt_prefixes[int(rule_s_v2["selected_index"])],
        ),
    ]
    if args.include_gt_accompaniment:
        conditions.append(
            (
                "gt_accompaniment",
                None,
                prepared["acc_beats_gt"][: args.prompt_prefix_beats],
            )
        )
    outputs = []
    for condition, candidate_index, raw_prefix in conditions:
        if len(raw_prefix) != args.prompt_prefix_beats:
            raise RuntimeError(
                f"{condition} has {len(raw_prefix)} complete Prompt beats; "
                f"expected {args.prompt_prefix_beats}"
            )
        if candidate_index is None:
            converted_prefix = raw_prefix
            prompt_source = "npz_gt_accompaniment"
        else:
            converted_prefix = [
                convert_prompt_beat(prompt_tokenizer, tokenizer, beat)
                for beat in raw_prefix
            ]
            prompt_source = "prompt_model_candidate"
        schedule = inject_prompt_prefix(prepared["schedule"], converted_prefix)
        if args.max_schedule_steps > 0:
            schedule = schedule[: args.max_schedule_steps]
        converter.beats_to_midi(
            prepared["mel_beats"][: args.prompt_prefix_beats],
            converted_prefix,
            tempo=float(metadata.get("bpm") or 120),
            save_path=str(output_dir / f"01_{condition}_prompt.mid"),
        )
        for seed in continuation_seeds:
            generator = torch.Generator(device=args.device)
            generator.manual_seed(seed)
            started = time.perf_counter()
            acc_beats, _tokens = continuation.wrapper.generate_accompaniment(
                initial_tokens=prepared["initial_tokens"],
                schedule=schedule,
                vocab=prepared.get("vocab", tokenizer.vocab),
                device=args.device,
                temperature=args.continuation_temperature,
                top_k=args.continuation_top_k,
                top_p=args.continuation_top_p,
                repetition_penalty=args.continuation_repetition_penalty,
                verbose=False,
                generator=generator,
            )
            elapsed = time.perf_counter() - started
            output_path = output_dir / f"02_{condition}_seed{seed}_continuation.mid"
            converter.beats_to_midi(
                prepared["mel_beats"],
                acc_beats,
                tempo=float(metadata.get("bpm") or 120),
                save_path=str(output_path),
            )
            outputs.append(
                {
                    "condition": condition,
                    "prompt_source": prompt_source,
                    "candidate_number": (
                        candidate_index + 1 if candidate_index is not None else None
                    ),
                    "continuation_seed": seed,
                    "continuation_seconds": elapsed,
                    "acc_beats": len(acc_beats),
                    "output_midi": output_path.name,
                }
            )

    write_json(
        output_dir / "run_summary.json",
        {
            "pipeline": "prepared_npz_prompt_batch_rule_s_direct_offline_schedule",
            "npz_file": str(npz_path),
            "metadata": metadata,
            "prompt_checkpoint": str(prompt_checkpoint),
            "continuation_checkpoint": str(continuation_checkpoint),
            "prompt_batch_seconds": prompt_seconds,
            "prompt_parameters": {
                "candidate_count": args.candidate_count,
                "seed": args.prompt_seed,
                "prefix_beats": args.prompt_prefix_beats,
                "temperature": args.prompt_temperature,
                "top_k": args.prompt_top_k,
                "top_p": args.prompt_top_p,
                "repetition_penalty": args.prompt_repetition_penalty,
                "include_gt_accompaniment": args.include_gt_accompaniment,
            },
            "continuation_parameters": {
                "seeds": continuation_seeds,
                "temperature": args.continuation_temperature,
                "top_k": args.continuation_top_k,
                "top_p": args.continuation_top_p,
                "repetition_penalty": args.continuation_repetition_penalty,
            },
            "candidates": rule_s_v2["candidates"],
            "rule_s_v1_decision": {
                key: value for key, value in rule_s_v1.items() if key != "candidates"
            },
            "rule_s_v2_decision": {
                key: value for key, value in rule_s_v2.items() if key != "candidates"
            },
            "outputs": outputs,
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
