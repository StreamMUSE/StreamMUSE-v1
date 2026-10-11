# Lekai Melody Timing Robustness

## Why the perturbation is defined on the model grid

StreamMUSE and the Lekai offline data path quantize MIDI by computing:

```text
model_tick = round(beat_position * 4)
```

One model tick is therefore `0.25 beat`. The previous independent `[-20, +20] ms`
onset-jitter preflight changed zero 4tpb note records across all nine checked inputs. It
measured quantizer invariance, not continuation-model robustness.

The effective experiment moves selected Melody onsets by exactly `-1` or `+1` model
tick after determining their baseline 4tpb position. Quantized note duration, pitch,
velocity, accompaniment, and metadata are preserved.

## Frozen first-round conditions

```text
Pieces: 43038, 3410421, 2581346
Perturbed track: Melody only
Perturbation: onset only, independent direction per selected note
Magnitude: exactly +/-1 model tick (+/-0.25 beat)
Selection rates: 10%, 25%, 50%
Perturbation seeds: 0, 1, 2
Continuation checkpoint: base epoch15
Continuation condition: 8-beat GT accompaniment prompt
Decoding: greedy (top_k=1)
```

There are three baseline inputs and 27 perturbed inputs. The baseline is generated once
per piece because greedy decoding is deterministic.

Every prepared sample records the effective post-quantization changed-note rate and
pianoroll difference. Preparation fails if the perturbation disappears after 4tpb
conversion.

## Run on MBZ

```bash
cd /data/home/yuanxin/StreamMUSE-v1
GPU=0 bash experiments/lekai_melody_timing_robustness/run_task1_offline_greedy.sh
```

This uses the existing `RT-accompanimentV2/inference_new.py`; it does not introduce a
new model inference implementation.

After inference finishes, analyze the generated `Accompaniment` tracks with:

```bash
python experiments/lekai_melody_timing_robustness/analyze_task1_outputs.py \
  --prepared-manifest "$RUN_ROOT/prepared/manifest.json" \
  --inference-dir "$RUN_ROOT/inference" \
  --output-dir "$RUN_ROOT/analysis" \
  --ignore-prefix-beats 8
```

The analysis maps randomized inference filenames through the official manifest, compares
each perturbed output with its piece baseline, and creates a flat
`analysis/listening_by_piece/` bundle. Accompaniment notes beginning inside the shared
8-beat GT prefix are excluded from metrics so the prompt cannot inflate similarity.

The no-prompt greedy pilot is retained as a failed control: all three baselines produced
empty accompaniment, so it cannot measure robustness.

## Frozen Task 1 v2: greedy versus nucleus sampling

Task 1 v2 uses four held-out scored19 pieces that were not used by the first-round
three-piece experiment:

```text
Pieces: 5889045, 5817083, 332891, 5472153
Input conditions: clean, 10% perturbed notes, 50% perturbed notes
Perturbed track: Melody only
Perturbation: onset only, independent +/-1 model tick direction
Perturbation seed: 0
Continuation checkpoint: base epoch15
Continuation condition: 8-beat GT accompaniment prompt

Greedy:
  temperature=1.0, top_k=1, top_p=1.0, seed=0

Nucleus sampling:
  temperature=1.1, top_k=0, top_p=0.95, seeds=0,1,2
```

This produces `4 pieces x 3 input conditions x (1 greedy + 3 sampling seeds) = 48`
generated MIDI files. Each piece/condition/decoder-seed combination is run as a
singleton official `inference_new.py` invocation. This makes clean and perturbed
conditions begin from the same run-level random state for a given sampling seed.

Run it on MBZ with four GPUs:

```bash
cd /data/home/yuanxin/StreamMUSE-v1
GPUS=4,5,6,7 \
  bash experiments/lekai_melody_timing_robustness/run_task1_v2_greedy_sampling.sh
```

The run creates:

```text
prepared/manifest.json        post-quantization perturbation audit
inference/                    official per-run MIDI and manifests
analysis/per_condition.csv    objective clean-versus-perturbed comparisons
analysis/summary.json         aggregate results by decoder and perturbation rate
analysis/listening/           blind MIDI bundle, 12 files per piece
analysis/ratings.csv          listening score sheet without condition labels
analysis/_technical/          private listening key and merged manifests
```

## Scope

This first round isolates onset robustness. Duration-only perturbation should be a
separate follow-up so onset and duration effects are not conflated.
