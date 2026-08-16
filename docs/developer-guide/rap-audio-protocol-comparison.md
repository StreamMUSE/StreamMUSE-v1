# Rap Audio Protocol Comparison

This runbook operates the fixed comparison of `moss_global`, `ted_local`,
`fastpitch_phoneme`, and `moss_aligned`. The corpus is exactly 25 two-bar
requests for each of `01_space_exploration`, `02_deep_ocean`, and
`03_artificial_intelligence`: 75 requests, 300 protocol/chunk records, three
shared drum stems, and twelve final mixes.

## H200 Setup

The observed H200 tools are:

```bash
export ROOT_PREFIX=/data/home/Andrew.Yang/StreamMUSE
export HF_HOME=/data/home/Andrew.Yang/.cache/huggingface
export UV_BIN=/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/uv
export PYTHON_BIN=/usr/bin/python3.12
export MOSS_PYTHON_BIN=/usr/bin/python3.12
export PYTHON310_BIN=/data/home/Andrew.Yang/StreamMUSE/envs/streammuse-isochron/bin/python
```

`PYTHON310_BIN` is required by TED because its pinned dependency set includes a
Numba release that requires Python below 3.12. The script discovers
`micromamba` first and then `conda`, or accepts an explicit `CONDA_BIN`.

Run from the checked-out StreamMUSE repository:

```bash
bash scripts/setup_rap_audio_protocols_h200.sh
```

The script is idempotent and does not use `sudo` or alter
`streammuse-isochron`. It creates:

| Component | Isolated path |
| --- | --- |
| MOSS | `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/moss` |
| TED | `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/ted` |
| NeMo | `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/nemo` |
| MFA/alignment | `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/align` |
| FFmpeg 7 runtime | `/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/ffmpeg7` |
| Rubber Band extraction | `/data/home/Andrew.Yang/StreamMUSE/assets/rap-audio-protocols/rubberband/rootfs` |

The default pins are:

| Dependency | Pin |
| --- | --- |
| MOSS-TTS repository | `58b20a0d5fcc6766658d50967a90a9d890009a46` |
| TED-TTS repository | `36ffc3e2de346156baa7d60a1749ca4a9365625b` |
| MOSS-TTS model snapshot | `cdd3b911b1585e3f2dbc7775ef10f9926f58850a` |
| IndexTTS-2 model snapshot | `740dcaff396282ffb241903d150ac011cd4b1ede` |
| NeMo source/package | `v2.7.3` / `nemo_toolkit[tts]==2.7.3` |
| MFA source/package | `v3.4.1` / `montreal-forced-aligner=3.4.1` |
| FFmpeg runtime | `ffmpeg=7.1.1` in an isolated conda prefix |
| Rubber Band CLI/runtime | Ubuntu `rubberband-cli=3.3.0+dfsg-2build1` and `librubberband2=3.3.0+dfsg-2build1` debs |
| PyTorch/Torchaudio | `2.9.1+cu128` |
| faster-whisper | `1.2.1` |
| pronouncing | `0.3.0` in MOSS, TED, NeMo, and alignment |

MOSS is installed with Python 3.12 using PyPI and the CUDA 12.8 wheel index,
`--index-strategy unsafe-best-match`, and the editable
`MOSS-TTS[torch-runtime]` checkout. TED uses Python 3.10 with
`UV_PROJECT_ENVIRONMENT` and `uv sync --all-extras`. NeMo uses Python 3.10,
CUDA 12.8 Torch/Torchaudio wheels, and `nemo_toolkit[tts]==2.7.3`. Alignment is
a conda-forge prefix containing MFA 3.4.1. That MFA solve constrains its own
FFmpeg to 2.8, which is not suitable for TorchCodec. The setup therefore
creates a separate conda-forge `ffmpeg7` prefix with only FFmpeg 7.1.1 for
TorchCodec. It pins `pronouncing==0.3.0` in the MOSS, TED, NeMo, and alignment
Python prefixes because each backend imports StreamMUSE through `PYTHONPATH`.

Rubber Band is not requested from conda or pip. The configured conda-forge
channels do not provide the required package under that name. Without `sudo`,
the setup runs `apt download` for the pinned Ubuntu `rubberband-cli` and
`librubberband2` debs, then extracts them under the asset root with
`dpkg-deb -x`. It executes the extracted `rubberband --version` with the local
alignment-prefix library directory first, followed by the extracted Rubber
Band and FFmpeg-prefix library directories, and records the actual binary path
and resolved version. It likewise executes the isolated `ffmpeg -version` and
records its result.

The setup reuses the shared Hugging Face cache. It requests and verifies the
exact MOSS and IndexTTS-2 revisions even when those snapshots are already
present, preloads `tts_en_fastpitch` and `tts_en_hifigan`, and downloads the
MFA `english_us_arpa` acoustic and dictionary models. It also copies:

```text
<TED checkout>/datasets/Ref/0011_000001.wav
```

to:

```text
/data/home/Andrew.Yang/StreamMUSE/assets/rap-audio-protocols/ted/0011_000001.wav
```

The source and copied SHA-256 values, all environment paths, resolved Git
commits, package versions, model revisions, and cache roots are recorded in:

```text
/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/environment_manifest.json
```

Inspect the manifest after every setup run:

```bash
export MANIFEST_PATH=/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols/environment_manifest.json
"$PYTHON_BIN" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1])), indent=2))' "$MANIFEST_PATH"
```

Override `ENV_ROOT`, `CHECKOUT_ROOT`, `ASSET_ROOT`, `NEMO_CACHE_DIR`,
`MFA_ROOT_DIR`, `MANIFEST_PATH`, any tool path, repository URL, or pin when a
host layout differs. The setup defaults are declared paths; no value is
derived from a login home directory.

## Operator Paths

Declare all campaign paths explicitly on the H200. Adjust `REPO_ROOT` to the
actual rsync target:

```bash
export REPO_ROOT=/data/home/Andrew.Yang/StreamMUSE/StreamMUSE-NewSystem
export SOURCE_ALBUM="$REPO_ROOT/output/rap_album_10x50_90bpm_20260816_v4"
export CAMPAIGN_ROOT="$REPO_ROOT/output/rap_audio_protocol_comparison_20260816"
export ENV_ROOT=/data/home/Andrew.Yang/StreamMUSE/envs/rap-audio-protocols
export CHECKOUT_ROOT=/data/home/Andrew.Yang/StreamMUSE/checkouts/rap-audio-protocols
export ASSET_ROOT=/data/home/Andrew.Yang/StreamMUSE/assets/rap-audio-protocols
export HF_HOME=/data/home/Andrew.Yang/.cache/huggingface
export MANIFEST_PATH="$ENV_ROOT/environment_manifest.json"
export MOSS_ENV="$ENV_ROOT/moss"
export TED_ENV="$ENV_ROOT/ted"
export NEMO_ENV="$ENV_ROOT/nemo"
export ALIGN_ENV="$ENV_ROOT/align"
export FFMPEG7_ENV="$ENV_ROOT/ffmpeg7"
export MFA_ROOT_DIR="$ASSET_ROOT/mfa"
export RUBBERBAND_ROOT="$ASSET_ROOT/rubberband/rootfs"
export RUBBERBAND_BIN="$RUBBERBAND_ROOT/usr/bin/rubberband"
export TED_REFERENCE="$ASSET_ROOT/ted/0011_000001.wav"
export SONGS="01_space_exploration 02_deep_ocean 03_artificial_intelligence"

export MOSS_SNAPSHOT="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"]["moss_tts"]["snapshot_path"])' "$MANIFEST_PATH")"
export INDEXTTS_SNAPSHOT="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["models"]["index_tts_2"]["snapshot_path"])' "$MANIFEST_PATH")"

cd "$REPO_ROOT"
```

Every campaign CLI command requires `--source-album` and `--output-dir`; it
does not infer either from a home directory.

## Synchronize Inputs

From the local machine, declare both endpoints and transfer code plus only the
fixed source album:

```bash
export LOCAL_REPO="/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE-NewSystem/.worktrees/real_rap"
export REMOTE_HOST=h200
export REMOTE_REPO=/data/home/Andrew.Yang/StreamMUSE/StreamMUSE-NewSystem

rsync -avP \
  --exclude .git \
  --exclude .venv \
  --exclude output \
  "$LOCAL_REPO/" \
  "$REMOTE_HOST:$REMOTE_REPO/"

rsync -avP \
  "$LOCAL_REPO/output/rap_album_10x50_90bpm_20260816_v4/" \
  "$REMOTE_HOST:$REMOTE_REPO/output/rap_album_10x50_90bpm_20260816_v4/"
```

Compare the three remote `chosen_lyrics.jsonl` hashes to the local files before
rendering.

## Prepare The 75 Requests

```bash
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" "$MOSS_ENV/bin/python" \
  "$REPO_ROOT/scripts/run_rap_audio_protocol_comparison.py" \
  --source-album "$SOURCE_ALBUM" \
  --output-dir "$CAMPAIGN_ROOT" \
  --stage prepare
```

`prepare` writes only songs 01-03, exactly 25 canonical request lines and one
common drum stem per song. Re-running is safe only when the existing canonical
request files and `common/corpus_manifest.json` match byte-for-byte. Each
selected chunk prints one line containing `stage`, `song`, `protocol`, `chunk`,
`count`, `status`, and its request hash.

Audit the fixed corpus:

```bash
"$MOSS_ENV/bin/python" - "$CAMPAIGN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = json.loads((root / "common" / "corpus_manifest.json").read_text())
assert manifest["song_count"] == 3
assert manifest["total_request_count"] == 75
assert [item["request_count"] for item in manifest["songs"]] == [25, 25, 25]
assert all((root / "common" / item["song_id"] / "drums.wav").is_file() for item in manifest["songs"])
print("prepared songs=3 requests=75 drums=3")
PY
```

Any campaign-stage exception is appended and fsynced immediately to
`$CAMPAIGN_ROOT/campaign_errors.jsonl` before the exception is re-raised.
Backend successes and failures are recorded immediately in each
`render_chunks.jsonl`.

## Backend Smoke

Choose unused GPUs using `nvidia-smi`, then create one shared two-bar request:

```bash
export SMOKE_ROOT="$CAMPAIGN_ROOT/smoke"
export SMOKE_REQUESTS="$SMOKE_ROOT/requests.jsonl"
mkdir -p "$SMOKE_ROOT"
sed -n '1p' "$CAMPAIGN_ROOT/common/01_space_exploration/requests.jsonl" > "$SMOKE_REQUESTS"
```

Run MOSS:

```bash
PATH="$FFMPEG7_ENV/bin:$PATH" \
LD_LIBRARY_PATH="$FFMPEG7_ENV/lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" HF_HOME="$HF_HOME" "$MOSS_ENV/bin/python" \
  "$REPO_ROOT/scripts/rap_audio_backends/moss_backend.py" \
  --request-jsonl "$SMOKE_REQUESTS" \
  --output-dir "$SMOKE_ROOT/moss_global" \
  --record-path "$SMOKE_ROOT/moss_global/render_chunks.jsonl" \
  --reference-wav "$TED_REFERENCE" \
  --campaign-manifest "$SMOKE_ROOT/moss_global/campaign_manifest.json" \
  --model-id "$MOSS_SNAPSHOT" \
  --device cuda:0
```

Run TED:

TED defaults to `hmm`. On the observed H200 FP16 run, HMM alignment produced
`cannot convert float NaN to integer`; `max_head` completed as a fallback. This
is an explicit alignment ablation, not an equivalent HMM condition, and its
selection is recorded beside each TED ledger in `inference_config.json`.

```bash
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" HF_HOME="$HF_HOME" "$TED_ENV/bin/python" \
  "$REPO_ROOT/scripts/rap_audio_backends/ted_backend.py" \
  --requests-jsonl "$SMOKE_REQUESTS" \
  --records-jsonl "$SMOKE_ROOT/ted_local/render_chunks.jsonl" \
  --output-dir "$SMOKE_ROOT/ted_local" \
  --reference-wav "$TED_REFERENCE" \
  --ted-checkout "$CHECKOUT_ROOT/ted" \
  --model-dir "$INDEXTTS_SNAPSHOT" \
  --cfg-path "$INDEXTTS_SNAPSHOT/config.yaml" \
  --inference-method max_head
```

Run NeMo FastPitch:

```bash
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" HF_HOME="$HF_HOME" NEMO_CACHE_DIR="$HF_HOME/nemo" \
  "$NEMO_ENV/bin/python" "$REPO_ROOT/scripts/rap_audio_backends/fastpitch_backend.py" \
  --requests-jsonl "$SMOKE_REQUESTS" \
  --records-jsonl "$SMOKE_ROOT/fastpitch_phoneme/render_chunks.jsonl" \
  --output-dir "$SMOKE_ROOT/fastpitch_phoneme" \
  --fastpitch-model tts_en_fastpitch \
  --hifigan-model tts_en_hifigan \
  --device cuda:0 \
  --prosody-controls duration-only
```

Run MFA plus the aligned-MOSS warp against the exact MOSS source hash:

```bash
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export PATH="$RUBBERBAND_ROOT/usr/bin:$FFMPEG7_ENV/bin:$ALIGN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ALIGN_ENV/lib:$RUBBERBAND_ROOT/usr/lib/x86_64-linux-gnu:$FFMPEG7_ENV/lib:${LD_LIBRARY_PATH:-}"
"$ALIGN_ENV/bin/python" <<'PY'
import os
from pathlib import Path

from scripts.rap_audio_backends.aligned_moss_backend import (
    PendingAlignedChunk,
    render_aligned_chunk,
    run_forced_alignment,
    stage_alignment_inputs,
)
from scripts.rap_audio_backends.moss_backend import load_requests_jsonl
from streammuse.experiments.rap_audio_protocols.artifacts import append_chunk_record, read_chunk_record_index
from streammuse.experiments.rap_audio_protocols.contracts import ProtocolId

root = Path(os.environ["SMOKE_ROOT"])
request = load_requests_jsonl(os.environ["SMOKE_REQUESTS"])[0]
source = read_chunk_record_index(root / "moss_global" / "render_chunks.jsonl")[
    (ProtocolId.MOSS_GLOBAL, request.song_id, request.chunk_index)
]
if not source.success or not source.output_path or not source.output_sha256:
    raise RuntimeError("MOSS smoke source is incomplete")

staged = stage_alignment_inputs(
    (
        PendingAlignedChunk(
            request=request,
            source_wav_path=Path(source.output_path),
            expected_source_sha256=source.output_sha256,
        ),
    ),
    root / "moss_aligned" / "mfa-corpus",
    output_dir=root / "moss_aligned" / "mfa-output",
)
run_forced_alignment(root / "moss_aligned" / "mfa-corpus", root / "moss_aligned" / "mfa-output")
result = render_aligned_chunk(
    request=request,
    source_wav_path=staged[0].source_wav_path,
    expected_source_sha256=staged[0].source_sha256,
    textgrid_path=staged[0].expected_textgrid_path,
    output_wav_path=root / "moss_aligned" / request.song_id / "chunk-000.wav",
)
append_chunk_record(root / "moss_aligned" / "render_chunks.jsonl", result.record)
print(f"stage=smoke song={request.song_id} protocol=moss_aligned chunk=000 status={'success' if result.record.success else 'failed'}")
PY
```

Before the full render, require process exit zero, one nonempty WAV and complete
record per method, plausible sample rate/duration, no silence or repeated loop,
and intelligible playback. Confirm that the aligned record's
`source_chunk_sha256` equals the MOSS record's `output_sha256`.

## Resumable Full Render

Run the same commands again after an interruption. Each backend skips only a
chunk whose request hash, WAV metadata/path, source hash where applicable, and
output SHA-256 still match its ledger. Prior failures are retried. Do not delete
or hand-edit a ledger to resume.

MOSS, all 75 requests:

```bash
for SONG in $SONGS; do
  PATH="$FFMPEG7_ENV/bin:$PATH" \
  LD_LIBRARY_PATH="$FFMPEG7_ENV/lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" HF_HOME="$HF_HOME" "$MOSS_ENV/bin/python" \
    "$REPO_ROOT/scripts/rap_audio_backends/moss_backend.py" \
    --request-jsonl "$CAMPAIGN_ROOT/common/$SONG/requests.jsonl" \
    --output-dir "$CAMPAIGN_ROOT/moss_global" \
    --record-path "$CAMPAIGN_ROOT/moss_global/$SONG/render_chunks.jsonl" \
    --reference-wav "$TED_REFERENCE" \
    --campaign-manifest "$CAMPAIGN_ROOT/moss_global/$SONG/campaign_manifest.json" \
    --model-id "$MOSS_SNAPSHOT" \
    --device cuda:0
done
```

TED, all 75 requests:

```bash
for SONG in $SONGS; do
  PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" HF_HOME="$HF_HOME" "$TED_ENV/bin/python" \
    "$REPO_ROOT/scripts/rap_audio_backends/ted_backend.py" \
    --requests-jsonl "$CAMPAIGN_ROOT/common/$SONG/requests.jsonl" \
    --records-jsonl "$CAMPAIGN_ROOT/ted_local/$SONG/render_chunks.jsonl" \
    --output-dir "$CAMPAIGN_ROOT/ted_local" \
    --reference-wav "$TED_REFERENCE" \
    --ted-checkout "$CHECKOUT_ROOT/ted" \
    --model-dir "$INDEXTTS_SNAPSHOT" \
    --cfg-path "$INDEXTTS_SNAPSHOT/config.yaml" \
    --inference-method max_head
done
```

NeMo FastPitch, all 75 requests:

```bash
for SONG in $SONGS; do
  PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" HF_HOME="$HF_HOME" NEMO_CACHE_DIR="$HF_HOME/nemo" \
    "$NEMO_ENV/bin/python" "$REPO_ROOT/scripts/rap_audio_backends/fastpitch_backend.py" \
    --requests-jsonl "$CAMPAIGN_ROOT/common/$SONG/requests.jsonl" \
    --records-jsonl "$CAMPAIGN_ROOT/fastpitch_phoneme/$SONG/render_chunks.jsonl" \
    --output-dir "$CAMPAIGN_ROOT/fastpitch_phoneme" \
    --fastpitch-model tts_en_fastpitch \
    --hifigan-model tts_en_hifigan \
    --device cuda:0 \
    --prosody-controls duration-only
done
```

All backend commands print one dense line per rendered, skipped, or failed
chunk. Save stdout/stderr with the H200 job logs.

## Full MFA Alignment And Warp

Run Protocol 4 only after all 75 MOSS records and WAV hashes are complete. This
batch stages `.wav`/`.lab` pairs, invokes the pinned `english_us_arpa` models,
verifies every MOSS source hash, writes each aligned result to its ledger
immediately, and safely skips already verified aligned outputs:

```bash
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"
export PATH="$RUBBERBAND_ROOT/usr/bin:$FFMPEG7_ENV/bin:$ALIGN_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ALIGN_ENV/lib:$RUBBERBAND_ROOT/usr/lib/x86_64-linux-gnu:$FFMPEG7_ENV/lib:${LD_LIBRARY_PATH:-}"
"$ALIGN_ENV/bin/python" <<'PY'
import os
import tempfile
from pathlib import Path

from scripts.rap_audio_backends.aligned_moss_backend import (
    PendingAlignedChunk,
    render_aligned_chunk,
    run_forced_alignment,
    stage_alignment_inputs,
)
from scripts.rap_audio_backends.moss_backend import load_requests_jsonl
from streammuse.experiments.rap_audio_protocols.artifacts import (
    chunk_record_is_complete,
    read_chunk_record_index,
)
from streammuse.experiments.rap_audio_protocols.contracts import ProtocolId, canonical_json_dumps


def write_ledger(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            for key in sorted(records, key=lambda item: (item[1], item[2], item[0].value)):
                handle.write(canonical_json_dumps(records[key].to_payload()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


root = Path(os.environ["CAMPAIGN_ROOT"])
for song_id in os.environ["SONGS"].split():
    requests = load_requests_jsonl(root / "common" / song_id / "requests.jsonl")
    moss_records = read_chunk_record_index(root / "moss_global" / song_id / "render_chunks.jsonl")
    aligned_root = root / "moss_aligned" / song_id
    ledger_path = aligned_root / "render_chunks.jsonl"
    aligned_records = read_chunk_record_index(ledger_path)
    pending = []
    for ordinal, request in enumerate(requests, start=1):
        output_path = aligned_root / f"chunk-{request.chunk_index:03d}.wav"
        if chunk_record_is_complete(
            ledger_path,
            output_path,
            request=request,
            protocol_id=ProtocolId.MOSS_ALIGNED,
        ):
            print(
                f"stage=align song={song_id} protocol=moss_aligned "
                f"chunk={request.chunk_index:03d} count={ordinal}/25 status=skip"
            )
            continue
        source = moss_records[(ProtocolId.MOSS_GLOBAL, song_id, request.chunk_index)]
        if not source.success or not source.output_path or not source.output_sha256:
            raise RuntimeError(f"incomplete MOSS source: {song_id}/{request.chunk_index:03d}")
        pending.append(
            PendingAlignedChunk(
                request=request,
                source_wav_path=Path(source.output_path),
                expected_source_sha256=source.output_sha256,
            )
        )
    if not pending:
        continue

    corpus_dir = aligned_root / "mfa-corpus"
    mfa_output_dir = aligned_root / "mfa-output"
    staged = stage_alignment_inputs(pending, corpus_dir, output_dir=mfa_output_dir)
    run_forced_alignment(corpus_dir, mfa_output_dir)
    for item in staged:
        request = item.request
        result = render_aligned_chunk(
            request=request,
            source_wav_path=item.source_wav_path,
            expected_source_sha256=item.source_sha256,
            textgrid_path=item.expected_textgrid_path,
            output_wav_path=aligned_root / f"chunk-{request.chunk_index:03d}.wav",
        )
        key = (ProtocolId.MOSS_ALIGNED, song_id, request.chunk_index)
        aligned_records[key] = result.record
        write_ledger(ledger_path, aligned_records)
        print(
            f"stage=align song={song_id} protocol=moss_aligned "
            f"chunk={request.chunk_index:03d} count={request.chunk_index + 1}/25 "
            f"status={'success' if result.record.success else 'failed'} "
            f"source_sha256={item.source_sha256}"
        )
PY
```

## Assemble, Evaluate, And Package

Assemble all twelve exact-length vocal stems and mixes:

```bash
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" "$MOSS_ENV/bin/python" \
  "$REPO_ROOT/scripts/run_rap_audio_protocol_comparison.py" \
  --source-album "$SOURCE_ALBUM" \
  --output-dir "$CAMPAIGN_ROOT" \
  --stage assemble
```

`assemble` fails closed unless each selected song/protocol has exactly 25
successful matching records and WAVs. It writes 6,400,000-frame `vocals.wav`
and `mix.wav` files plus `artifact_manifest.json`.

Run independent faster-whisper evaluation after synthesis processes have
released their GPUs:

```bash
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" HF_HOME="$HF_HOME" "$MOSS_ENV/bin/python" \
  "$REPO_ROOT/scripts/run_rap_audio_protocol_comparison.py" \
  --source-album "$SOURCE_ALBUM" \
  --output-dir "$CAMPAIGN_ROOT" \
  --stage evaluate \
  --whisper-model large-v3 \
  --whisper-device cuda \
  --whisper-compute-type float16
```

The faster-whisper import, model construction, and model download occur only
in `evaluate`. WER uses normalized exact edit distance. Syllable timing uses
only exact word matches from the edit alignment and valid ASR intervals, so an
ASR insertion, deletion, or substitution cannot shift later words. The result
is labeled `estimated_syllable_timing_error_ms`; it is not phone-level ground
truth. Metrics also include duration error, clipping/silence, and target
stress-to-RMS correlation.

Build the blinded A-D package:

```bash
PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT" "$MOSS_ENV/bin/python" \
  "$REPO_ROOT/scripts/run_rap_audio_protocol_comparison.py" \
  --source-album "$SOURCE_ALBUM" \
  --output-dir "$CAMPAIGN_ROOT" \
  --stage package
```

The package contains `blind/<song>/<A-D>.wav`, `listening.html`,
`blind_map.json`, `package_audit.json`, `experiment_metrics.json`, and
`COMPARISON.md`. Protocol names do not appear in the HTML or neutral audio
paths.

Run the final count/hash audit:

```bash
"$MOSS_ENV/bin/python" - "$CAMPAIGN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
protocols = ("moss_global", "ted_local", "fastpitch_phoneme", "moss_aligned")
songs = ("01_space_exploration", "02_deep_ocean", "03_artificial_intelligence")

corpus = json.loads((root / "common" / "corpus_manifest.json").read_text())
assert corpus["total_request_count"] == 75

record_count = 0
for protocol in protocols:
    for song in songs:
        ledger = root / protocol / song / "render_chunks.jsonl"
        records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        assert len(records) == 25
        assert all(item["success"] for item in records)
        artifact = json.loads((root / protocol / song / "artifact_manifest.json").read_text())
        assert len(artifact["source_chunks"]) == 25
        assert artifact["vocal_stem"]["frame_count"] == 6_400_000
        assert artifact["mix"]["frame_count"] == 6_400_000
        record_count += len(records)

metrics = json.loads((root / "experiment_metrics.json").read_text())
assert len(metrics["metrics"]) == 12
blind_map = json.loads((root / "blind_map.json").read_text())
assert set(blind_map) == set(songs)
assert all(set(mapping) == {"A", "B", "C", "D"} for mapping in blind_map.values())
package = json.loads((root / "package_audit.json").read_text())
assert package["audio_file_count"] == 12
assert all(item["source_sha256"] == item["blind_sha256"] for item in package["audio_files"])
print(f"audit songs=3 protocols=4 records={record_count} mixes=12 vocals=12 drums=3 blind_audio=12")
PY
```

Review `campaign_errors.jsonl` and every failed backend record even if a later
resume succeeds; they are the durable interruption/failure history.

## Download And Listen Locally

From the local machine:

```bash
export REMOTE_HOST=h200
export REMOTE_CAMPAIGN=/data/home/Andrew.Yang/StreamMUSE/StreamMUSE-NewSystem/output/rap_audio_protocol_comparison_20260816
export LOCAL_CAMPAIGN="/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE-NewSystem/.worktrees/real_rap/output/rap_audio_protocol_comparison_20260816"

mkdir -p "$LOCAL_CAMPAIGN"
rsync -av --partial --progress \
  "$REMOTE_HOST:$REMOTE_CAMPAIGN/" \
  "$LOCAL_CAMPAIGN/"

ssh "$REMOTE_HOST" sha256sum "$REMOTE_CAMPAIGN/package_audit.json"
shasum -a 256 "$LOCAL_CAMPAIGN/package_audit.json"
```

Serve the package over HTTP so relative audio URLs work consistently:

```bash
cd "$LOCAL_CAMPAIGN"
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/listening.html`. Keep `blind_map.json` closed until
the first A-D listening pass is complete.
