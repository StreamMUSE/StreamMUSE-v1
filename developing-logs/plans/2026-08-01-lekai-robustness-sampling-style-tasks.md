# Lekai Follow-up Experiment Tasks

Date: 2026-08-01

These tasks are independent experiments. Input perturbation, decoding randomness, and
music style must not be changed in the same comparison unless explicitly used as a
follow-up interaction experiment.

## Task 1: Robustness to Melody Note Timing Perturbations

Status: suspended after post-hoc baseline validation. Two of three greedy baselines have
large-scale piece-end note closure, so listening-based perturbation claims are invalid.

### Question

How robust is the Lekai continuation model when note timing in the input Melody track is
slightly perturbed?

### Initial comparison

- Original Melody MIDI.
- Onset perturbation.
- Duration perturbation.
- Combined onset and duration perturbation.

Use deterministic greedy decoding (`top_k=1`) first so output differences can be
attributed to the input perturbation rather than sampling randomness.

The 2026-07-31 preflight established that independent `[-20, +20] ms` onset jitter
changes zero note records after Lekai's `4 steps/beat` quantization. It is retained only
as a quantizer-control result and must not be sent to model inference as a robustness
condition.

The frozen first model-level round shifts selected Melody onsets by exactly `-1` or `+1`
model tick (`0.25 beat`) while preserving quantized duration. Overall strength is
controlled by exact note-selection rates of `10%`, `25%`, and `50%`, with perturbation
seeds `0`, `1`, and `2`. Every condition must report a non-zero post-quantization note
and pianoroll difference before GPU inference.

The initial no-accompaniment-prompt greedy pilot collapsed to empty accompaniment for
all three baseline pieces. The formal first round therefore uses a fixed 8-beat GT
accompaniment prompt and evaluates only continuation notes beginning after that prefix.

The formal run produced non-empty continuation for all 27 perturbed inputs. Increasing
the selected-note rate reduced output overlap with the unperturbed baseline in the
aggregate. See
`developing-logs/reports/2026-08-01-lekai-melody-timing-robustness-task1-initial-results.md`.

### Evaluation

- Empty-output and inference-failure rate.
- Output note count and density.
- Pianoroll/output-token difference from the unperturbed control.
- Rhythm, pitch, and duration changes.
- Melody-accompaniment alignment.
- Human listening assessment.

## Task 2: Sampling Versus Greedy for Repetition/Monotony

Status: the nucleus-only `top_k=0`, `top_p=0.95` round is complete. Decoder selection
remains unresolved.

### Question

Can stochastic sampling reduce the monotonous patterns observed with `top_k=1` without
causing an unacceptable increase in unstable or poor outputs?

### Comparison

```text
Greedy baseline:
  top_k = 1

Sampling candidate:
  top_p = 0.95
  temperature = 1.1
  top_k = 0
```

`top_k=1` must not remain enabled in the sampling condition, because it would make
`top_p` ineffective. Generate multiple fixed seeds per piece for the sampling condition.

### Evaluation

- Empty-output and failure rate.
- Repetition rate and unique-bar ratio.
- Pitch-class entropy.
- Note density and duration distribution.
- Melody-accompaniment compatibility.
- Human ratings for variety, musical quality, Melody-accompaniment fit, and stability;
  every dimension uses higher = better.

The expected result is a diversity/stability trade-off, not an assumption that sampling
is automatically better.

The frozen first round uses the same three clean pieces and fixed 8-beat GT
accompaniment prompt as the valid Task 1 condition. Greedy produces one deterministic
output per piece. Sampling uses `top_k=0`, `top_p=0.95`, `temperature=1.1`, and seeds
`0..4`. Dataset indices are explicit and identical across all six runs.

The greedy seed-0 and sampling seed-4 preflight manifests both used selection
fingerprint `53da6d5619e23e7c082359237877874a800c6be1`.

All six formal manifests retained that fingerprint. Machine diagnostics show that the
sampling candidate reduces greedy pitch/duration/pattern collapse but increases empty
regions for some piece/seed pairs. See
`developing-logs/reports/2026-08-02-lekai-greedy-vs-sampling-task2-initial-results.md`.

## Task 3: Results Across Music Styles

Status: five-subset 20-piece metadata selection and official input/schedule preflight
completed on 2026-08-02. Source-arrangement review and decoder freeze remain before
continuation inference.

### Question

How does the selected Lekai inference setting behave on pieces from different musical
styles?

### Style subset construction

Use:

`/data/home/yuanxin/data/allxml_xml/xml_metadata.jsonl`

Candidate style subsets will be constructed from the available piece-title and
author/composer information:

- Titles from `credits`, especially entries with `type=title` or `type=subtitle`.
- Composer/author information from `credits` and `creators`.
- Piano and part-count constraints from `parts` and `n_parts`.

Title/composer classification is a candidate-selection method rather than a perfect
ground-truth style label. Ambiguous pieces must be excluded or manually checked before
the evaluation set is frozen.

The initial subsets should favor clearly identifiable categories and avoid claiming
fine-grained genre labels unsupported by the metadata. Candidate examples include
classical/period-specific composers, pop/film/game arrangements, lyrical pieces, and
rhythmically strong pieces when the title/composer evidence is sufficiently clear.

### Evaluation

- Accompaniment success and empty-output rate by subset.
- Melody-accompaniment compatibility.
- Repetition/monotony by subset.
- Style consistency and human listening ratings.
- Systematic differences between style subsets.

Where possible, control or report piece length, time signature, BPM, Melody density, and
Prompt length so these factors are not mistaken for style effects.

## Execution Order

```text
Task 1: deterministic timing-perturbation robustness
  -> Task 2: greedy versus stochastic sampling
  -> Task 3: cross-style evaluation using the selected inference setting
```

The StreamMUSE/offline per-beat output mismatch is tracked separately in:

`developing-logs/reports/2026-07-31-streammuse-offline-model-output-audit.md`
