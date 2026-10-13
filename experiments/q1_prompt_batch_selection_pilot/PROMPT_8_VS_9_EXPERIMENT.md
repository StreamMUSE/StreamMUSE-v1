# Prompt 8 Beats vs 9 Beats

## Question

Does retaining the Prompt Model's ninth accompaniment beat improve the
transition and downstream Continuation compared with the existing 8-beat
prefix?

## Paired design

Each piece produces one Prompt Model sample with at least 9 accompaniment
beats. The same sample is used by both conditions:

```text
prompt_08_beats:
  Prompt beats 1..8 -> Continuation starts at beat 9

prompt_09_beats:
  Prompt beats 1..9 -> Continuation starts at beat 10
```

The only acceptance criterion for Prompt generation is the presence of at
least 9 accompaniment beats. Every attempted seed and generated beat count is
recorded in `prompt_attempts.csv`.

## Scope

```text
Pieces: 5889045, 5817083, 332891, 43038, 5472153, 3410421, 2581346
Conditions per piece: 2
Continuation seeds per condition: 3
Final blind listening samples: 7 * 2 * 3 = 42
```

Prompt sampling:

```text
temperature = 0.8
top_p = 0.95
top_k = 50
repetition_penalty = 1.0
```

Run on GPU7:

```bash
DEVICE=7 bash q1_prompt_batch_selection_pilot/run_prompt_8_vs_9_seven.sh
```

Use `blind_listening/` with `blind_ratings.csv`. Do not open
`blind_mapping_private.csv` before rating.
