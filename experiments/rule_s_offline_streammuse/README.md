# Rule-S Offline vs StreamMUSE Experiment

## Question

Using the same legacy human-recorded Melody input and the same Prompt Model
candidate batch, compare:

1. candidate 1 (`batch_first`) versus the frozen Rule-S selection (`rule_s`);
2. one-shot offline inference versus the actual StreamMUSE MIDI-file simulation
   and scheduler path.

The files under `prompts/old_input/mel` are treated as legacy human-recorded
Melody cases for this experiment. This is an operational label; their original
recording provenance is not independently documented in the repository.

## Frozen design

```text
10 Melody MIDI files
x 2 Prompt selection conditions
x 2 continuation seeds
x 2 execution paths (offline and StreamMUSE)
= 80 final MIDI outputs
```

Both selection conditions generate one real Prompt Model batch with `N=5`:

- `batch_first`: use candidate 1 from that batch;
- `rule_s`: use the highest-scoring eligible candidate from that batch.

The Prompt seed is fixed per piece, so both conditions receive the same five
candidates. Continuation seeds are `0,1`.

## Rule-S

Empty and incomplete 8-beat Prompt candidates are ineligible. Remaining
candidates are ranked within the current batch:

```text
S =
  -0.450 * rank(PPL)
  +0.477 * rank(Pitch Range)
  +0.467 * rank(Pitch-Class Entropy)
  +0.496 * rank(Average Voice Number)
```

The coefficients are the piece-centered Spearman correlations frozen from the
earlier discovery pilot. PPL is recomputed from raw Prompt Model logits.

## Frozen generation settings

Prompt Model:

```text
candidate_count = 5
prompt_prefix_beats = 8
temperature = 0.8
top_k = 50
top_p = 0.95
repetition_penalty = 1.0
```

Continuation Model:

```text
seeds = 0,1
temperature = 1.1
top_k = disabled (0)
top_p = 0.95
repetition_penalty = 1.0
```

Shared system settings:

```text
model condition bpm = 120
StreamMUSE playback tempo = 15 bpm
evaluation window = 32 beats total (8-beat Prompt + 24 beats continuation)
final catch-up grace = 3 ticks (below the 4-tick generation interval)
Melody leading-rest trimming = on (first retained note moves to tick 0)
ticks_per_beat = 4
generation_interval_ticks = 4
generation_length_frames = 4
prompt_context_beats = 32
history_max_ticks = 128
late recovery = off
active-note rehydration = off
strict representation loop = on
```

## H200 command

Choose an idle physical GPU explicitly:

```bash
cd /data/home/yuanxin/StreamMUSE-rule-s-stanley
GPU=2 bash scripts/run_rule_s_offline_streammuse_h200.sh
```

Run the formal trimmed offline set independently:

```bash
GPU=2 EXECUTION_PATHS=offline \
  bash scripts/run_rule_s_offline_streammuse_h200.sh
```

Keep realtime and slowed-clock consistency runs in separate output roots by
using `EXECUTION_PATHS=streammuse` with an explicit playback tempo.

The output root defaults to:

```text
/data/home/yuanxin/runs/rule_s_offline_streammuse_<timestamp>
```

Each case records the offline summary, StreamMUSE session artifacts, runtime
configuration, Prompt candidate metrics, selected candidate, stdout/stderr,
and completion status. The runner resumes completed cases by default.

## Validity checks

Before listening analysis:

1. verify both conditions have the same Prompt candidate hashes per piece;
2. verify `batch_first` selected candidate 1 and `rule_s` selected the logged
   Rule-S candidate;
3. verify both paths loaded real checkpoints and used the frozen settings;
4. inspect StreamMUSE traces for dropped or clipped events;
5. compare only the continuation window after the first 8 beats.

The slower playback tempo is a simulation control, not a model-conditioning
change. Both offline and StreamMUSE model calls remain conditioned on 120 BPM;
the StreamMUSE MIDI clock runs at 15 BPM so the H200 implementation can finish
catch-up before the 32-beat listening window ends. Offline output stops at tick
128. StreamMUSE stops at tick 131, giving the final tick-128 append request
three seconds to finish without crossing another generation boundary or
submitting another Melody beat.

Leading-rest trimming uses the existing offline and MIDI-file input options in
both paths. The evaluation end tick is computed after subtracting the same
quantized first-note offset. Use `--no-trim-leading-rest` only for an explicit
ablation; formal listening runs keep trimming enabled.
