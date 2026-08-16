# Rap Audio Protocol Comparison

Task 8 adds the operator path for the offline rap audio protocol campaign:

- `scripts/run_rap_audio_protocol_comparison.py`
- `scripts/setup_rap_audio_protocols_h200.sh`
- `src/streammuse/experiments/rap_audio_protocols/evaluation.py`
- `src/streammuse/experiments/rap_audio_protocols/listening.py`

## Scope

The campaign fixes the corpus to songs `01_space_exploration`, `02_deep_ocean`,
and `03_artificial_intelligence` from
`output/rap_album_10x50_90bpm_20260816_v4/`. `prepare` writes exactly seventy
five canonical two-bar requests plus one deterministic drum stem per selected
song. `assemble` requires twenty five successful chunk records per
protocol/song and produces exact-length `vocals.wav`, `mix.wav`, and
`artifact_manifest.json`. `evaluate` runs an independent faster-whisper pass
lazily and writes `metrics.json`. `package` writes `listening.html`,
`blind_map.json`, `package_audit.json`, and `COMPARISON.md`.

## H200 Setup

Run the setup script on the H200 host. It never writes into the main
StreamMUSE project environment.

```bash
bash scripts/setup_rap_audio_protocols_h200.sh
```

By default it creates:

- `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/moss`
- `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/ted`
- `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/nemo`
- `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/align`

Override `ROOT_PREFIX`, `ENV_ROOT`, `CHECKOUT_ROOT`, `HF_HOME`, `MANIFEST_PATH`,
`PYTHON_BIN`, and the `*_REF` or `*_REPO_URL` variables when the host layout
differs. The script emits `environment_manifest.json` with the resolved checkout
commits and environment paths.

## Prepare

```bash
uv run python scripts/run_rap_audio_protocol_comparison.py \
  --source-album output/rap_album_10x50_90bpm_20260816_v4 \
  --output-dir output/rap_audio_protocol_comparison_20260816 \
  --stage prepare
```

Resume safety rules:

- the script never assumes a home-directory path;
- an existing `common/corpus_manifest.json` must match the regenerated
  manifest exactly;
- an existing `requests.jsonl` must match the canonical request payloads
  exactly.

## Assemble

After backend runs have populated `<protocol>/<song>/render_chunks.jsonl` and
the per-chunk WAVs:

```bash
uv run python scripts/run_rap_audio_protocol_comparison.py \
  --source-album output/rap_album_10x50_90bpm_20260816_v4 \
  --output-dir output/rap_audio_protocol_comparison_20260816 \
  --stage assemble \
  --song 01_space_exploration \
  --protocol moss_global
```

`assemble` fails closed if any chunk record is missing, unsuccessful, or the
request count is not exactly twenty five.

## Evaluate

The evaluation stage builds the faster-whisper transcriber only when the stage
is selected.

```bash
uv run python scripts/run_rap_audio_protocol_comparison.py \
  --source-album output/rap_album_10x50_90bpm_20260816_v4 \
  --output-dir output/rap_audio_protocol_comparison_20260816 \
  --stage evaluate \
  --whisper-model small \
  --whisper-device cpu \
  --whisper-compute-type int8
```

Metrics include:

- exact Levenshtein WER;
- chunk duration error statistics;
- clipped sample counts and silent chunk counts;
- estimated syllable timing error from independent word timestamps;
- stress-to-RMS correlation.

The syllable timing metric is derived proportionally inside word intervals and
is intentionally labeled as an estimate, not phone-level ground truth.

## Package

```bash
uv run python scripts/run_rap_audio_protocol_comparison.py \
  --source-album output/rap_album_10x50_90bpm_20260816_v4 \
  --output-dir output/rap_audio_protocol_comparison_20260816 \
  --stage package
```

The listening package copies each selected mix to neutral `blind/<song>/<A-D>.wav`
paths so the HTML page does not expose protocol names before unblinding.

## Download

Use explicit `rsync` source and destination paths when pulling results back
from the H200 host:

```bash
rsync -avP \
  h200:/data/home/Andrew.Yang/StreamMUSE/output/rap_audio_protocol_comparison_20260816/ \
  output/rap_audio_protocol_comparison_20260816/
```
