# Lekai Greedy Versus Sampling Experiment

## Question

Can stochastic decoding reduce monotonous accompaniment patterns compared with the
current `top_k=1` setting, without causing excessive instability or lower musical
quality?

## Frozen first round

```text
Pieces: 2581346, 3410421, 43038
Input: clean Melody with a fixed 8-beat GT accompaniment prompt
Checkpoint: base epoch15 continuation model

Greedy:
  temperature = 1.0
  top_k = 1
  top_p = 1.0
  one output per piece

Sampling:
  temperature = 1.1
  top_k = 0
  top_p = 0.95
  seeds = 0, 1, 2, 3, 4
```

All runs use the same explicit dataset indices. Only the decoding configuration and
sampling seed change. The official `RT-accompanimentV2/inference_new.py` remains the
only inference implementation.

## Metrics

Metrics exclude the common 8-beat GT accompaniment prompt.

- Note density: accompaniment note onsets per beat.
- Pitch-class entropy: Shannon entropy over the 12 accompaniment pitch classes.
- Dominant pitch-class rate: fraction of notes assigned to the most frequent pitch
  class; larger values indicate stronger pitch collapse.
- Median/P90 duration and long-note rate: duration distribution in beats, including the
  fraction exceeding 8 beats. This isolates hanging or abnormally sustained notes.
- Piece-end note rate: fraction of continuation notes closed exactly at the MIDI end;
  a large value indicates unmatched note-ons being closed by final export cleanup.
- Unique-bar ratio: unique exact 4-beat onset/pitch/duration signatures divided by all
  evaluated 4-beat segments.
- Adjacent exact-bar repeat rate: fraction of neighboring 4-beat signatures that are
  identical.
- Maximum identical-bar run: longest consecutive run of the same exact signature.
- Non-empty adjacent repeat and non-empty maximum-run variants exclude silence.
- Empty-bar rate and maximum empty-bar run report silence separately.
- Empty-output rate.
- Pairwise Jaccard similarity between sampling seeds.

Exact bar repetition is deliberately strict. It is a reproducible proxy for literal
repetition, not a complete measure of perceived monotony. Human ratings remain the
primary evidence for monotony and musical quality.

## Sampling policy

The original-author convention treats top-k and nucleus sampling as alternative
policies rather than combining them:

```text
Nucleus sampling: top_k=0,  top_p<1
Top-k sampling:   top_k>1, top_p=1
Greedy:           top_k=1, top_p=1
```

`inference_new.py` currently defaults to `temperature=1.1`, `top_k=10`, and
`top_p=0.95`, so its CLI defaults conflict with that convention. The runner explicitly
uses nucleus-only sampling (`top_k=0`, `top_p=0.95`) instead of relying on those
defaults.

## Run

```bash
cd /data/home/yuanxin/StreamMUSE-v1
GPU=7 bash experiments/lekai_sampling_monotony/run_task2_offline.sh
```
