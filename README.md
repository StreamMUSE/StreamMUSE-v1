# StreamMUSE

StreamMUSE is a real-time accompaniment generation system. It reads melody input from keyboard, MIDI devices, or MIDI files, sends incremental melody context to an inference backend, and plays or records generated accompaniment.

## Environment

```bash
uv sync
```

This creates a `.venv` environment in the repository.

## Real-time Application

The real-time system is a CLI client plus an optional inference server. The main code lives under `src/streammuse/` and follows a Presentation / Application / Domain / Infrastructure structure.

### Quick Start

```bash
# 1. Start the fake inference server (no model required)
uv run python scripts/fake_inference_server.py

# 2. In another terminal, run the CLI
uv run streammuse-cli --input-mode keyboard
```

### CLI Examples

```bash
# Input modes
uv run streammuse-cli --input-mode keyboard
uv run streammuse-cli --input-mode midi_device
uv run streammuse-cli --input-mode midi_file --midi-file-path path/to/song.mid

# Output types
uv run streammuse-cli --output-type console
uv run streammuse-cli --output-type audio --midi-out-port "My Synth"
uv run streammuse-cli --output-type composite --log-dir logs

# Metronome + count-in
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --output-type console \
    --enable-metronome \
    --count-in-beats 4

# Music injection with MIDI-file input
uv run streammuse-cli \
    --input-mode midi_file \
    --midi-file-path prompts/inputs_lekai/mel/1.mid \
    --injection-file prompts/inputs_lekai/mel/1.mid \
    --injection-length 16
```

### Lekai HTTP Server

```bash
LEKAI_CHECKPOINT_PATH=path/to/lekai_checkpoint.safetensors \
LEKAI_DEVICE=auto \
LEKAI_DTYPE=auto \
uv run python -m streammuse.infrastructure.inference.server_lekai
```

Then run the client:

```bash
uv run streammuse-cli \
    --input-mode keyboard \
    --inference-type http \
    --model-name lekai \
    --server-url http://127.0.0.1:8000/generate_accompaniment
```

### Session Logging

Using `--output-type composite --log-dir logs` creates a timestamped session directory:

```text
logs/YYYY-MM-DD/session_HHMMSS/
├── events.jsonl
├── inferences.json
├── performance.json
├── statistics.csv
├── session_config.json
├── session_summary.txt
├── melody_history.json
├── accompaniment_history.json
└── combined.mid
```

`combined.mid` contains `Melody` and `Accompaniment` tracks. With `--enable-metronome`, it also contains a `Metronome` drum track. With `--count-in-beats`, count-in clicks are recorded at the beginning of the MIDI file.

### Continuation tonal constraint

The real-model interleaved continuation path enables a strict tonal sampling
constraint by default. Set `LEKAI_CONTINUATION_TONAL_CONSTRAINT=0` on the inference
backend to restore unconstrained sampling; restart the backend after changing code
or its launch environment. Prompt selection and Prompt outputs are unchanged.

- Every two bars, estimate a major/natural-minor key from duration-weighted
  melody pitch classes. Pre-estimate **at the start of the final beat**, using
  only melody already available, and apply it at the next two-bar boundary.
  In 4/4, estimate at tick 60 for tick 64, then 92 for 96, etc. The first
  continuation key is bootstrapped from the available Prompt melody.
- Accumulate melody from the start of the session to the estimation time,
  without a rolling cutoff or recency decay. For example, estimates at ticks
  60 and 92 use [0,60) and [0,92), not [32,60) and [64,92).
  Full input history is retained independently of the model's sliding token
  context; the current request's melody is included exactly once.
  Each note onset contributes at most **four beats (16 ticks)** of duration
  evidence across its lifetime, protecting against missing NOTE_OFF events.
  Retriggers count as new notes. This cap does not shorten any played note.
  Only when fewer than three pitch classes have been cumulatively observed,
  mark the key as undetermined (`key_state="NaN"`, `key=null`) and disable
  the tonal mask. Never turn insufficient evidence into a ban on every pitch.
  Once at least three pitch classes are present, use the original highest-scoring
  major/minor template. There is no confidence or best-versus-second-score threshold;
  close scores do not disable the mask. Updates still take effect at the next
  two-bar boundary. The EMPTY-token guard remains independent.
- Mask relative PIT tokens according to their decoded absolute pitch class,
  before temperature/top-k/top-p sampling; also enforce legal PIT/PAT pairs.
  Both new onsets and sustains must be in key. Old out-of-key sustains end at
  the new key boundary. No out-of-key fallback is permitted.
- No extra tick wait, network request, or model pass is added. Estimation and
  masking do have a small computation cost. The continuation trace and generation
  diagnostics include `tonal_decisions`: effective boundary, estimation tick,
  evidence, inferred key, allowed pitch classes, and fallback reason.

“In key” means membership in the estimated seven-note scale, not a guarantee of
musical quality. Chromatic passing tones and harmonic-minor raised leading tones
are disallowed unless included in the estimated scale. Empty beats remain legal;
this is not an anti-silence mechanism. This constraint does not modify historical
Prompt accompaniment or apply to rule-based fallback/stub generation.

### Continuation EMPTY-token guard

The interleaved continuation sampler also enables `LEKAI_CONTINUATION_EMPTY_TOKEN_GUARD=1`
by default. Four consecutive raw beat outputs exactly equal to `EMPTY(169), ACC_END(170)`
trigger a mask on token 169 for the **next eight beats**, before temperature/top-k/top-p.
The fourth empty beat is unchanged; ordinary sampling resumes after the eight beats,
and a later streak can trigger the guard again. Set the variable to `0` to disable it.

Counting persists across continuation requests, and resets on session reset, history
clear/injection, or a discontinuous/repeated beat tick. Prompt sampling is unchanged.
Generation traces and diagnostics record `empty_token_guard`, including the trigger,
mask status, and remaining beats. No extra model pass or tick wait is introduced.

This is specifically an EMPTY-token mask, not an audible-silence detector. Sustains
and `ACC_END`-only output do not count toward the four-beat streak and remain legal;
the guard does not force note onsets or relax the tonal constraint. Postprocessing
may represent an `ACC_END`-only output as an empty beat, so this rule alone does not
guarantee audible accompaniment in every guarded beat.

## Documentation

Start from [`docs/index.md`](docs/index.md). Key pages:

- [`docs/getting-started/configuration.md`](docs/getting-started/configuration.md)
- [`docs/reference/cli-reference.md`](docs/reference/cli-reference.md)
- [`docs/user-guide/running-realtime.md`](docs/user-guide/running-realtime.md)
- [`docs/user-guide/music-injection.md`](docs/user-guide/music-injection.md)
- [`docs/architecture/application/service.md`](docs/architecture/application/service.md)

## Dataset / Training Notes

Legacy dataset preparation, preprocessing, training, and inference scripts remain in the repository. See the existing scripts and historical logs under `developing-logs/` for experiment-specific workflows.
