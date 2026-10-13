# Q1 Prompt Batch Selection Pilot

## Question

Can prompt-only features identify which accompaniment prompt is more likely to
produce good downstream continuation, without observing future continuation at
selection time?

## Fixed pilot design

- Pieces: scored19 cases 09, 15, and 17 (`5889045`, `5472153`, `3410421`).
- Prompt candidates per piece: 3 (`prompt seeds = 0,1,2`).
- Continuations per fixed prompt: 4 (`continuation seeds = 0,1,2,3`).
- Prompt prefix: 8 beats / 2 bars.
- Total listening samples: `3 x 3 x 4 = 36`.

The runner generates each prompt once and reuses its exact decoded/re-encoded
prefix for all four continuation seeds. Prompt and continuation randomness are
therefore not confounded.

## Run

From `external/lekai_real_time`:

```bash
DEVICE=2 bash q1_prompt_batch_selection_pilot/run_q1_pilot.sh
```

For the scored19 top7 expansion (`7 x 3 x 4 = 84` listening samples):

```bash
DEVICE=2 bash q1_prompt_batch_selection_pilot/run_q1_top7.sh
```

For a short smoke test:

```bash
DEVICE=2 \
NPZ_IDS=5889045 \
PROMPT_SEEDS=0 \
CONTINUATION_SEEDS=0 \
bash q1_prompt_batch_selection_pilot/run_q1_pilot.sh --max-schedule-steps 16
```

## Outputs

Each timestamped run contains:

- one directory per piece;
- one GT MIDI per piece;
- three prompt-only MIDIs per piece;
- twelve prompt+continuation MIDIs per piece;
- raw Prompt Model log probabilities and PPL for accompaniment content tokens
  in only the injected 8 beats (unused extra generated beats are excluded);
- `prompt_manifest.csv` and `continuation_manifest.csv`;
- randomized listening MIDIs under `blind_listening/`;
- public `blind_ratings.csv` and private `blind_mapping_private.csv`.

Do not expose the private mapping before ratings are complete.

To create three independent rating sheets after generation:

```bash
python q1_prompt_batch_selection_pilot/prepare_rater_templates.py \
  --run-dir /path/to/q1_top7_TIMESTAMP
```

## Human score aggregation

For each fixed prompt candidate, aggregate its four downstream continuation
ratings using mean score and good-result rate. The prompt-only metrics are then
correlated with those aggregate targets.

This 9-prompt pilot is for workflow validation and exploratory trends, not a
claim of statistical significance.

## Analyze completed ratings

Run from the repository root:

```bash
uv run --isolated --no-project \
  --with pretty-midi --with pandas --with scipy \
  python experiments/q1_prompt_batch_selection_pilot/analyze_ratings.py \
  --run-dir remote_results/q1_prompt_batch_selection_pilot_20260722
```

The analysis resolves numeric ranges to their midpoint, provisionally maps
`3.x` to `3.5`, aggregates four continuations per fixed prompt, extracts
prompt-only MIDI features, and reports both raw and within-piece correlations.

Validate a completed bundle before rating:

```bash
uv run --isolated --no-project --with pretty-midi \
  python experiments/q1_prompt_batch_selection_pilot/validate_run.py \
  --run-dir /path/to/q1_top7_TIMESTAMP
```

For one rater, select 24 held-out samples with paired continuation seeds across
the three Prompt candidates for each piece:

```bash
python experiments/q1_prompt_batch_selection_pilot/prepare_single_rater_subset.py \
  --run-dir remote_results/q1_prompt_batch_selection_top7_20260724 \
  --discovery-run-dir remote_results/q1_prompt_batch_selection_pilot_20260722
```

The prior 3-piece pilot is used only to discover the candidate rules. Its
ratings and pieces are excluded from the 4-piece held-out evaluation.

Freeze the selector's choices before held-out rating:

```bash
uv run --isolated --no-project --with pretty-midi \
  python experiments/q1_prompt_batch_selection_pilot/apply_frozen_selector.py \
  --run-dir remote_results/q1_prompt_batch_selection_top7_20260724 \
  --subset-dir single_rater_new4_rule_eval24
```

## Actual-Batch Rule-S Best-of-N

The production-oriented comparison uses one real Prompt Model batch and sends
only the ordinary single sample and the Rule-S-selected sample to the
Continuation Model. See `RULE_S_BEST_OF_N_EXPERIMENT.md`.

Old 3-piece sampling sanity check:

```bash
DEVICE=7 bash q1_prompt_batch_selection_pilot/run_rule_s_best_of_5_old3.sh
```

New 4-piece held-out comparison:

```bash
DEVICE=7 bash q1_prompt_batch_selection_pilot/run_rule_s_best_of_5_new4.sh
```

## Prompt 8 Beats vs 9 Beats

The paired seven-piece experiment reuses one Prompt Model sample for both
prefix lengths and produces three continuation samples per condition:

```bash
DEVICE=7 bash q1_prompt_batch_selection_pilot/run_prompt_8_vs_9_seven.sh
```

See `PROMPT_8_VS_9_EXPERIMENT.md`.
