#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"

RUN_NAME="${RUN_NAME:-formal_offline_npz_case_comparison_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/offline-formal/$RUN_NAME}"

SEEDS_TEXT="${SEEDS:-0 1 2 3 4}"
GPU_IDS_TEXT="${GPU_IDS:-0 1 2 3 6 7}"
DATASETS_TEXT="${DATASETS:-scored19 pop909}"
MODELS_TEXT="${MODELS:-base_epoch15 pop909_finetune_epoch9}"
CONDITIONS_TEXT="${CONDITIONS:-01_no_prompt 02_gt_melody_gt_accompaniment_prompt8_oracle 03_melody_only_prompt8 04_prompt_model_acc8_then_continuation}"
SCORED19_SET="${SCORED19_SET:-all19}"
read -r -a SEEDS <<< "$SEEDS_TEXT"
read -r -a GPU_IDS <<< "$GPU_IDS_TEXT"
read -r -a DATASETS <<< "$DATASETS_TEXT"
read -r -a MODELS <<< "$MODELS_TEXT"
read -r -a CONDITIONS <<< "$CONDITIONS_TEXT"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_IDS[@]}}"

BASE_CKPT="${BASE_CKPT:-$ROOT_DIR/checkpoints-resume/epoch_15_0307_1858/model.safetensors}"
FINETUNE_CKPT="${FINETUNE_CKPT:-/data/home/yuanxin/RT-accompanimentV2-sft-pop909-compatible/experiments/sft_pop909_compatible/checkpoints/pop909_compatible_from_epoch15_ep10_0710_1034/epoch_9_0710_1044/model.safetensors}"
PROMPT_CKPT="${PROMPT_CKPT:-$ROOT_DIR/external/lekai_real_time/prompt_model/checkpoints/best_model/model.safetensors}"
TWO_STAGE_RUNNER="${TWO_STAGE_RUNNER:-$ROOT_DIR/external/lekai_real_time/run_offline_teacher_inference.py}"

SCORED19_SOURCE_DIR="${SCORED19_SOURCE_DIR:-$ROOT_DIR/data/scored19_npz}"
SCORED19_TAR="${SCORED19_TAR:-/data/home/yuanxin/scored19_npz.tar.gz}"

POP909_SOURCE_DIR="${POP909_SOURCE_DIR:-/data/home/yuanxin/data/rt_accompaniment_sft/final_datasets/rt4ch_pop909_rebar_only/test}"
POP909_MIDI_DIR="${POP909_MIDI_DIR:-/data/home/yuanxin/data/rt_accompaniment_sft/trimmed_split_midi/pop909_main_rebar_1_4_to_4_4_for_beat}"
NURSERY_SOURCE_DIR="${NURSERY_SOURCE_DIR:-$ROOT_DIR/formal_pop909_sanity_20260710/inputs/nursery/mel}"

SCORED19_IDS=(
  6383897
  69528
  5967993
  4805644
  1948786
  5346633
  6107560
  4809425
  5889045
  105118
  5771100
  5817083
  332891
  43038
  5472153
  5713648
  3410421
  2581346
  6175749
)

SCORED19_TOP7_IDS=(
  5889045  # case09, score7
  5817083  # case12, score6
  332891   # case13, score6
  43038    # case14, score6q
  5472153  # case15, score7
  3410421  # case17, score7
  2581346  # case18, score7q
)

POP909_NPZ_NAMES=(
  pop909_main_rebar_1_4_to_4_4_for_beat__pop909_023__bars0000-0063_chunk001of002.npz
  pop909_main_rebar_1_4_to_4_4_for_beat__pop909_050__bars0000-0054_chunk001of001.npz
  pop909_main_rebar_1_4_to_4_4_for_beat__pop909_091__bars0000-0053_chunk001of001.npz
  pop909_main_rebar_1_4_to_4_4_for_beat__pop909_161__bars0000-0043_chunk001of001.npz
  pop909_main_rebar_1_4_to_4_4_for_beat__pop909_393__bars0000-0063_chunk001of002.npz
)

NURSERY_MIDI_NAMES=(
  twinkle_twinkle_little_star_c_major_melody_only.mid
  two_tigers_frere_jacques_c_major_melody_only.mid
)

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Required directory not found: $path" >&2
    exit 1
  fi
}

model_ckpt() {
  case "$1" in
    base_epoch15) echo "$BASE_CKPT" ;;
    pop909_finetune_epoch9) echo "$FINETUNE_CKPT" ;;
    *) echo "Unknown model: $1" >&2; return 1 ;;
  esac
}

gpu_for_job() {
  local job_idx="$1"
  local gpu_idx=$((job_idx % ${#GPU_IDS[@]}))
  echo "${GPU_IDS[$gpu_idx]}"
}

wait_for_slot() {
  while [[ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL" ]]; do
    sleep 2
  done
}

join_csv() {
  local IFS=,
  echo "$*"
}

write_list_file() {
  local out="$1"
  shift
  : > "$out"
  for item in "$@"; do
    printf '%s\n' "$item" >> "$out"
  done
}

prepare_scored19_source() {
  mkdir -p "$SCORED19_SOURCE_DIR"
  local -n selected_ids="$1"
  local missing=0
  for id in "${selected_ids[@]}"; do
    if [[ ! -f "$SCORED19_SOURCE_DIR/${id}.npz" ]]; then
      missing=1
      break
    fi
  done
  if [[ "$missing" == "1" ]]; then
    require_file "$SCORED19_TAR"
    rm -f "$SCORED19_SOURCE_DIR"/*.npz
    tar -xzf "$SCORED19_TAR" -C "$SCORED19_SOURCE_DIR"
  fi
}

prepare_dataset_cases() {
  local dataset="$1"
  local case_dir="$OUTPUT_ROOT/case_npz_dirs/$dataset"
  local ref_dir="$OUTPUT_ROOT/reference_midi/$dataset"
  local meta_dir="$OUTPUT_ROOT/metadata/$dataset"
  mkdir -p "$case_dir" "$ref_dir" "$meta_dir"
  rm -f "$case_dir"/*.npz "$meta_dir"/npz_names.txt "$meta_dir"/npz_ids.csv

  if [[ "$dataset" == "scored19" ]]; then
    local selected_ids_name="SCORED19_IDS"
    if [[ "$SCORED19_SET" == "top7" ]]; then
      selected_ids_name="SCORED19_TOP7_IDS"
    elif [[ "$SCORED19_SET" != "all19" ]]; then
      echo "Unknown SCORED19_SET: $SCORED19_SET" >&2
      exit 1
    fi
    local -n selected_ids="$selected_ids_name"
    prepare_scored19_source "$selected_ids_name"
    for id in "${selected_ids[@]}"; do
      require_file "$SCORED19_SOURCE_DIR/${id}.npz"
      ln -s "$SCORED19_SOURCE_DIR/${id}.npz" "$case_dir/${id}.npz"
    done
    write_list_file "$meta_dir/npz_names.txt" "${selected_ids[@]/%/.npz}"
    join_csv "${selected_ids[@]}" > "$meta_dir/npz_ids.csv"
  elif [[ "$dataset" == "pop909" ]]; then
    require_dir "$POP909_SOURCE_DIR"
    require_dir "$POP909_MIDI_DIR"
    for name in "${POP909_NPZ_NAMES[@]}"; do
      require_file "$POP909_SOURCE_DIR/$name"
      ln -s "$POP909_SOURCE_DIR/$name" "$case_dir/$name"
    done
    write_list_file "$meta_dir/npz_names.txt" "${POP909_NPZ_NAMES[@]}"
    join_csv "${POP909_NPZ_NAMES[@]}" > "$meta_dir/npz_ids.csv"
  elif [[ "$dataset" == "nursery" ]]; then
    require_dir "$NURSERY_SOURCE_DIR"
    mkdir -p "$OUTPUT_ROOT/prepared_midi/nursery" "$OUTPUT_ROOT/trimmed_midi/nursery"
    for name in "${NURSERY_MIDI_NAMES[@]}"; do
      require_file "$NURSERY_SOURCE_DIR/$name"
    done
    (
      cd "$ROOT_DIR"
      "$PYTHON_BIN" - "$NURSERY_SOURCE_DIR" "$OUTPUT_ROOT/prepared_midi/nursery" "$OUTPUT_ROOT/trimmed_midi/nursery" "$case_dir" "$meta_dir" <<'PY'
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pretty_midi

source_dir, prepared_dir, trimmed_dir, npz_dir, meta_dir = map(Path, sys.argv[1:])
prepared_dir.mkdir(parents=True, exist_ok=True)
trimmed_dir.mkdir(parents=True, exist_ok=True)
npz_dir.mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)

names = [
    "twinkle_twinkle_little_star_c_major_melody_only.mid",
    "two_tigers_frere_jacques_c_major_melody_only.mid",
]

rows = []
npz_names = []
for order, name in enumerate(names, start=1):
    src = source_dir / name
    prepared = prepared_dir / name
    shutil.copy2(src, prepared)

    midi = pretty_midi.PrettyMIDI(str(src))
    _, tempos = midi.get_tempo_changes()
    bpm = float(tempos[0]) if len(tempos) else 120.0
    if bpm <= 0:
        bpm = 120.0
    sec_per_step = 60.0 / bpm / 4.0

    notes = [
        note
        for inst in midi.instruments
        if not inst.is_drum
        for note in inst.notes
        if 21 <= note.pitch <= 108
    ]
    if not notes:
        raise RuntimeError(f"no usable notes in {src}")

    first = min(n.start for n in notes)
    last = max(n.end for n in notes)
    total_steps = max(1, int(math.ceil((last - first) / sec_per_step)))
    measure_width = 16
    num_measures = int(math.ceil(total_steps / measure_width))
    padded_steps = num_measures * measure_width

    roll = np.zeros((4, 88, padded_steps), dtype=np.uint8)
    trimmed_midi = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    melody_inst = pretty_midi.Instrument(program=0, name="Melody")

    for note in notes:
        start = max(0.0, note.start - first)
        end = max(start + sec_per_step, note.end - first)
        start_step = int(round(start / sec_per_step))
        end_step = int(round(end / sec_per_step))
        if end_step <= start_step:
            end_step = start_step + 1
        start_step = max(0, min(start_step, padded_steps - 1))
        end_step = max(start_step + 1, min(end_step, padded_steps))
        pitch_idx = note.pitch - 21
        roll[0, pitch_idx, start_step:end_step] = 1
        roll[1, pitch_idx, start_step] = 1
        melody_inst.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=note.pitch,
                start=start_step * sec_per_step,
                end=end_step * sec_per_step,
            )
        )

    trimmed_midi.instruments.append(melody_inst)
    trimmed = trimmed_dir / name
    trimmed_midi.write(str(trimmed))

    stem = Path(name).stem
    npz_name = f"nursery__{stem}.npz"
    payload = {
        f"measure_{i}": roll[:, :, i * measure_width : (i + 1) * measure_width]
        for i in range(num_measures)
    }
    payload["metadata"] = {
        "time_signature": "4/4",
        "time_signature_idx": 0,
        "key_signature_idx": None,
        "key_signature": None,
        "bpm": int(round(bpm)),
        "tempo_text": None,
        "num_measures": num_measures,
        "resolution": 16,
        "ticks_per_beat": 4,
        "total_length": padded_steps,
        "num_parts": 2,
        "num_channels": 4,
        "original_measures": num_measures,
        "valid_measures": list(range(num_measures)),
        "source_format": "nursery_melody_midi",
        "conversion": "melody_midi_to_rt_4ch_empty_accompaniment",
        "source_midi": str(src),
        "prepared_midi": str(prepared),
        "trimmed_midi": str(trimmed),
        "source_bar_start": 0,
        "source_bar_end_exclusive": num_measures,
        "chunk_index": 1,
        "num_chunks": 1,
        "original_file": stem,
        "split_info": "1/1",
        "is_continuation": False,
        "accompaniment_gt_semantics": "empty_acc_channels_not_real_gt",
    }
    np.savez_compressed(npz_dir / npz_name, **payload)
    npz_names.append(npz_name)
    rows.append((order, npz_name, str(src), str(prepared), str(trimmed), num_measures, int(round(bpm))))

(meta_dir / "npz_names.txt").write_text("\n".join(npz_names) + "\n", encoding="utf-8")
(meta_dir / "npz_ids.csv").write_text(",".join(npz_names) + "\n", encoding="utf-8")
(meta_dir / "nursery_source_manifest.csv").write_text(
    "order,npz_name,source_midi,prepared_midi,trimmed_midi,num_measures,bpm,acc_gt_semantics\n"
    + "\n".join(",".join(map(str, row + ("empty_acc_channels_not_real_gt",))) for row in rows)
    + "\n",
    encoding="utf-8",
)
PY
    )
  else
    echo "Unknown dataset: $dataset" >&2
    exit 1
  fi

  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" - "$dataset" "$case_dir" "$ref_dir" "$meta_dir" "$POP909_MIDI_DIR" <<'PY'
import csv
import json
import os
import shutil
import sys
from pathlib import Path

dataset, case_dir, ref_dir, meta_dir, pop909_midi_dir = map(Path, sys.argv[1:])
dataset_name = str(dataset)
case_dir = Path(case_dir)
ref_dir = Path(ref_dir)
meta_dir = Path(meta_dir)
pop909_midi_dir = Path(pop909_midi_dir)

sys.path.insert(0, os.getcwd())
from config import ModelConfig
from PianoDataset import PianoDataset
from Token2Midi import MidiConverter

names = [line.strip() for line in (meta_dir / "npz_names.txt").read_text().splitlines() if line.strip()]
ds = PianoDataset(
    str(case_dir),
    config=ModelConfig(),
    cache_lengths=False,
    mode="train",
    test_split_ratio=0.0,
    random_seed=42,
)
indices = []
for name in names:
    if name not in ds.data_files:
        raise RuntimeError(f"{name} not found after PianoDataset split; available={ds.data_files}")
    indices.append(ds.data_files.index(name))
(meta_dir / "indices_for_order.json").write_text(json.dumps(indices, indent=2), encoding="utf-8")

converter = MidiConverter(ds.tokenizer)
rows = []
for order, name in enumerate(names, start=1):
    npz_path = case_dir / name
    stem = npz_path.stem
    decoded_gt = ref_dir / f"{order:02d}_{stem}_decoded_from_npz_GT.mid"
    converter.gt_to_midi(str(npz_path), str(decoded_gt))
    source_midi = ""
    copied_source = ""
    if dataset_name == "pop909":
        raw_stem = stem.split("__", 1)[1] if "__" in stem else stem
        source = pop909_midi_dir / f"{raw_stem}.mid"
        if not source.exists():
            raise FileNotFoundError(source)
        copied = ref_dir / f"{order:02d}_{raw_stem}_trimmed_source.mid"
        shutil.copy2(source, copied)
        source_midi = str(source)
        copied_source = str(copied)
    rows.append(
        {
            "order": order,
            "dataset": dataset_name,
            "npz_name": name,
            "npz_path": str(npz_path),
            "pianodataset_index": indices[order - 1],
            "decoded_gt_midi": str(decoded_gt),
            "source_trimmed_midi": source_midi,
            "copied_source_trimmed_midi": copied_source,
        }
    )

with (meta_dir / "npz_midi_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
PY
  )
}

run_base_or_prompt_condition() {
  local dataset="$1"
  local model_name="$2"
  local condition="$3"
  local seed="$4"
  local gpu="$5"
  local ckpt="$6"
  local case_dir="$OUTPUT_ROOT/case_npz_dirs/$dataset"
  local meta_dir="$OUTPUT_ROOT/metadata/$dataset"
  local out_dir="$OUTPUT_ROOT/raw_runs/$dataset/$model_name/$condition/seed_$(printf '%02d' "$seed")"
  local log_file="$out_dir/_run.log"
  mkdir -p "$out_dir"

  local num_samples
  num_samples="$(wc -l < "$meta_dir/npz_names.txt" | tr -d ' ')"

  local -a cmd=()
  if [[ "$condition" == "01_no_prompt" ]]; then
    cmd=(
      "$PYTHON_BIN" "$ROOT_DIR/inference_new.py"
      --ckpt "$ckpt"
      --data-dir "$case_dir"
      --indices-file "$meta_dir/indices_for_order.json"
      --num-samples "$num_samples"
      --dataset-mode train
      --dataset-seed 42
      --test-split-ratio 0.0
      --gt-prefix-beats 0
      --seed "$seed"
      --device cuda:0
      --output-dir "$out_dir"
      --temperature 1.1
      --top-k 10
      --top-p 0.95
      --repetition-penalty 1.0
      --no-export-gt-midi
    )
  elif [[ "$condition" == "02_gt_melody_gt_accompaniment_prompt8_oracle" ]]; then
    cmd=(
      "$PYTHON_BIN" "$ROOT_DIR/inference_new.py"
      --ckpt "$ckpt"
      --data-dir "$case_dir"
      --indices-file "$meta_dir/indices_for_order.json"
      --num-samples "$num_samples"
      --dataset-mode train
      --dataset-seed 42
      --test-split-ratio 0.0
      --gt-prefix-beats 8
      --seed "$seed"
      --device cuda:0
      --output-dir "$out_dir"
      --temperature 1.1
      --top-k 10
      --top-p 0.95
      --repetition-penalty 1.0
      --no-export-gt-midi
    )
  elif [[ "$condition" == "03_melody_only_prompt8" ]]; then
    cmd=(
      "$PYTHON_BIN" "$ROOT_DIR/Inference/no_acc_prompt_inference.py"
      --ckpt "$ckpt"
      --data-dir "$case_dir"
      --indices-file "$meta_dir/indices_for_order.json"
      --num-samples "$num_samples"
      --dataset-mode train
      --dataset-seed 42
      --test-split-ratio 0.0
      --prompt-mode mel_only
      --prompt-beats 8
      --seed "$seed"
      --device cuda:0
      --output-dir "$out_dir"
      --temperature 1.1
      --top-k 10
      --top-p 0.95
      --repetition-penalty 1.0
      --no-export-gt-midi
    )
  else
    echo "Unknown base condition: $condition" >&2
    return 1
  fi

  echo "[$dataset $model_name $condition seed=$seed gpu=$gpu] ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1
}

run_two_stage_condition() {
  local dataset="$1"
  local model_name="$2"
  local seed="$3"
  local gpu="$4"
  local ckpt="$5"
  local condition="04_prompt_model_acc8_then_continuation"
  local case_dir="$OUTPUT_ROOT/case_npz_dirs/$dataset"
  local meta_dir="$OUTPUT_ROOT/metadata/$dataset"
  local out_dir="$OUTPUT_ROOT/raw_runs/$dataset/$model_name/$condition/seed_$(printf '%02d' "$seed")"
  local log_file="$out_dir/_run.log"
  mkdir -p "$out_dir"

  local npz_ids_csv
  npz_ids_csv="$(cat "$meta_dir/npz_ids.csv")"
  local -a cmd=(
    "$PYTHON_BIN" "$TWO_STAGE_RUNNER"
    --checkpoint "$ckpt"
    --prompt-checkpoint "$PROMPT_CKPT"
    --data-dir "$case_dir"
    --output-dir "$out_dir"
    --npz-ids "$npz_ids_csv"
    --variants-per-id 1
    --prompt-prefix-beats 8
    --prompt-num-bars 2
    --prompt-max-new-tokens 1024
    --device cuda:0
    --temperature 1.1
    --top-k 0
    --top-p 0.95
    --repetition-penalty 1.0
    --seed "$seed"
    --mode train
  )

  echo "[$dataset $model_name $condition seed=$seed gpu=$gpu] ${cmd[*]}"
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1
}

write_run_readme() {
  cat > "$OUTPUT_ROOT/README.md" <<EOF
# Formal Offline NPZ Case Comparison

Created: $(date -Is)

This run is offline NPZ/dataset driven. MIDI files under reference_midi are references
or decoded outputs from the same NPZ cases; they are not the direct model input.

## Datasets

- scored19: $SCORED19_SET scored NPZ cases from $SCORED19_SOURCE_DIR
- pop909: 5 POP909 test NPZ cases from $POP909_SOURCE_DIR
- nursery: two melody-only sanity MIDIs converted to 4-channel NPZ with empty accompaniment GT

## Conditions

Enabled in this run:

$(printf -- "- %s\n" "${CONDITIONS[@]}")

Definitions:

- 01_no_prompt: inference_new.py --gt-prefix-beats 0
- 02_gt_melody_gt_accompaniment_prompt8_oracle: inference_new.py --gt-prefix-beats 8
- 03_melody_only_prompt8: Inference/no_acc_prompt_inference.py --prompt-mode mel_only --prompt-beats 8
- 04_prompt_model_acc8_then_continuation: external/lekai_real_time/run_offline_teacher_inference.py --prompt-prefix-beats 8

## Models

- base_epoch15: $BASE_CKPT
- pop909_finetune_epoch9: $FINETUNE_CKPT
- prompt model: $PROMPT_CKPT

## Repro

SEEDS="$SEEDS_TEXT" GPU_IDS="$GPU_IDS_TEXT" DATASETS="$DATASETS_TEXT" MODELS="$MODELS_TEXT" CONDITIONS="$CONDITIONS_TEXT" SCORED19_SET="$SCORED19_SET" \\
  bash scripts/run_formal_offline_npz_case_comparison.sh
EOF
}

main() {
  require_file "$PYTHON_BIN"
  require_file "$BASE_CKPT"
  require_file "$FINETUNE_CKPT"
  require_file "$PROMPT_CKPT"
  require_file "$TWO_STAGE_RUNNER"

  mkdir -p "$OUTPUT_ROOT/metadata" "$OUTPUT_ROOT/reference_midi" "$OUTPUT_ROOT/case_npz_dirs" "$OUTPUT_ROOT/raw_runs"
  write_run_readme

  for dataset in "${DATASETS[@]}"; do
    prepare_dataset_cases "$dataset"
  done

  local job_idx=0
  for dataset in "${DATASETS[@]}"; do
    for model_name in "${MODELS[@]}"; do
      local ckpt
      ckpt="$(model_ckpt "$model_name")"
      for condition in "${CONDITIONS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          wait_for_slot
          local gpu
          gpu="$(gpu_for_job "$job_idx")"
          if [[ "$condition" == "04_prompt_model_acc8_then_continuation" ]]; then
            run_two_stage_condition "$dataset" "$model_name" "$seed" "$gpu" "$ckpt" &
          else
            run_base_or_prompt_condition "$dataset" "$model_name" "$condition" "$seed" "$gpu" "$ckpt" &
          fi
          job_idx=$((job_idx + 1))
        done
      done
    done
  done

  wait
  echo "All jobs finished: $OUTPUT_ROOT"
}

main "$@"
