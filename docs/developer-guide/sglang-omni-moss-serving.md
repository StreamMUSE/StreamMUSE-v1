# SGLang-Omni MOSS Serving Runbook

## Status And Scope

`sglang-omni` is an explicit experimental RAP backend. The production default
remains `inprocess` until a frozen H200 experiment produces
`promotion-candidate`, the real Mac gate passes, and the canary observation
window is signed off. A local test run cannot satisfy those gates.

The first release uses a complete non-streaming WAV response:

```text
Mac -> SSH tunnel -> StreamMUSE render :8020
                         |-> Qwen vLLM :8001
                         `-> SGLang-Omni MOSS :8030
```

Only `:8020` is forwarded to the Mac. Qwen and SGLang-Omni stay loopback-bound
on the H200. MOSS synthesis still feeds the existing MMS alignment, Rubber Band
R3 warp, exact-frame validation, artifact packaging, Opus transport, and Mac
fallback path. Raw PCM streaming is Phase 9 work and is not enabled here.

## Required Pins

Use an isolated SGLang-Omni image or Python environment. Do not install it into
the StreamMUSE environment. Before qualification, record all of the following:

- immutable container digest or exact wheel hashes;
- exact SGLang-Omni and SGLang versions and source revisions;
- Python, PyTorch, CUDA, FlashAttention, FlashInfer, compiler, and driver versions;
- exact MOSS model snapshot revision and absolute snapshot path;
- vendored `moss_tts.yaml` bytes and SHA-256;
- reference WAV bytes, transcript UTF-8 bytes, and both SHA-256 values;
- physical GPU UUID and `CUDA_VISIBLE_DEVICES` mapping;
- StreamMUSE commit and dirty patch digest.

The historical `FileNotFoundError: ninja` during FlashInfer JIT is a hard
preflight failure. The candidate runtime must expose `sgl-omni`, `ninja`, a C++
compiler, and `nvcc` before model startup.

Start from a clean shell and replace every `/absolute/path/...` value:

```bash
export REPO_ROOT=/absolute/path/to/StreamMUSE-v1
export STREAMMUSE_ENV=/absolute/path/to/streammuse-env
export SGLANG_ENV=/absolute/path/to/sglang-omni-env
export EVIDENCE_ROOT=/absolute/path/to/sglang-moss-evidence
export RAP_ARTIFACT_ROOT=/absolute/path/to/rap-artifacts/sglang-candidate

export MOSS_MODEL_ID=OpenMOSS-Team/MOSS-TTS-v1.5
export MOSS_MODEL_PATH=/absolute/path/to/immutable/moss-snapshot
export MOSS_MODEL_REVISION=replace-with-exact-snapshot-revision
export MOSS_RUNTIME_CONFIG=/absolute/path/to/pinned/moss_tts.yaml
export MOSS_REFERENCE_WAV=/absolute/path/to/reference.wav
export MOSS_REFERENCE_TEXT=/absolute/path/to/reference.txt
export MOSS_SERVICE_MEDIA_ROOT=/absolute/path/to/service-media
export MOSS_SERVICE_REFERENCE=/absolute/path/to/service-media/reference.wav

export SGLANG_OMNI_VERSION=replace-with-exact-version
export SGLANG_OMNI_REVISION=replace-with-exact-revision
export SGLANG_VERSION=replace-with-exact-version
export SGLANG_REVISION=replace-with-exact-revision
export IMPLEMENTATION_ID=sglang-moss-http-v1
export QWEN_GPU=replace-with-unused-physical-index
export MOSS_GPU=replace-with-unused-physical-index
```

For a container, `/absolute/path/to/service-media` means the path visible to
the service. Mount that one directory read-only and run preflight in a namespace
that can read both host and service-visible files. Never allowlist `/`, a home
directory, or the model cache.

Create immutable environment evidence and calculate hashes:

```bash
mkdir -p "$EVIDENCE_ROOT" "$RAP_ARTIFACT_ROOT"
"$SGLANG_ENV/bin/python" -m pip freeze --all \
  | LC_ALL=C sort > "$EVIDENCE_ROOT/sglang-environment.lock"

export MOSS_CONFIG_SHA256="$(sha256sum "$MOSS_RUNTIME_CONFIG" | awk '{print $1}')"
export MOSS_REFERENCE_SHA256="$(sha256sum "$MOSS_REFERENCE_WAV" | awk '{print $1}')"
export MOSS_SERVICE_REFERENCE_SHA256="$(sha256sum "$MOSS_SERVICE_REFERENCE" | awk '{print $1}')"
export MOSS_REFERENCE_TEXT_SHA256="$(sha256sum "$MOSS_REFERENCE_TEXT" | awk '{print $1}')"
export MOSS_ENVIRONMENT_SHA256="$(sha256sum "$EVIDENCE_ROOT/sglang-environment.lock" | awk '{print $1}')"
export MOSS_SERVICE_REFERENCE_URI="$(
  "$STREAMMUSE_ENV/bin/python" -c \
    'from pathlib import Path; import os; print(Path(os.environ["MOSS_SERVICE_REFERENCE"]).as_uri())'
)"

test "$MOSS_REFERENCE_SHA256" = "$MOSS_SERVICE_REFERENCE_SHA256"
test -s "$MOSS_REFERENCE_TEXT"
```

The transcript must match the spoken reference exactly and have no surrounding
whitespace. The preflight validates a bounded mono WAV, byte equality between
the two reference paths, transcript encoding, local URI mapping, loopback bind,
and single-GPU selection. It computes and records the config and environment
hashes; the operator must compare those values with the frozen experiment pins.

## Preflight

First inspect the exact candidate CLI and preserve its output:

```bash
PATH="$SGLANG_ENV/bin:/usr/local/cuda/bin:$PATH" \
"$SGLANG_ENV/bin/sgl-omni" serve --help \
  > "$EVIDENCE_ROOT/sgl-omni-serve-help.txt"
```

Build one common argument array. The supplied version/revision values are
declarations backed by the environment lock and build records; they must not be
floating names such as `main`, `latest`, or `unknown`.

```bash
cd "$REPO_ROOT"
PREFLIGHT_ARGS=(
  --implementation-id "$IMPLEMENTATION_ID"
  --sgl-omni-bin "$SGLANG_ENV/bin/sgl-omni"
  --ninja-bin "$SGLANG_ENV/bin/ninja"
  --cxx-bin /usr/bin/c++
  --nvcc-bin /usr/local/cuda/bin/nvcc
  --model-id "$MOSS_MODEL_ID"
  --model-path "$MOSS_MODEL_PATH"
  --model-revision "$MOSS_MODEL_REVISION"
  --config "$MOSS_RUNTIME_CONFIG"
  --reference-wav "$MOSS_REFERENCE_WAV"
  --reference-text "$MOSS_REFERENCE_TEXT"
  --service-reference-file "$MOSS_SERVICE_REFERENCE"
  --service-reference-uri "$MOSS_SERVICE_REFERENCE_URI"
  --allowed-local-media-path "$MOSS_SERVICE_MEDIA_ROOT"
  --runtime-environment-file "$EVIDENCE_ROOT/sglang-environment.lock"
  --sglang-omni-version "$SGLANG_OMNI_VERSION"
  --sglang-omni-revision "$SGLANG_OMNI_REVISION"
  --sglang-version "$SGLANG_VERSION"
  --sglang-revision "$SGLANG_REVISION"
  --host 127.0.0.1
  --port 8030
  --gpu "$MOSS_GPU"
)

uv run python scripts/preflight_sglang_omni_moss.py \
  "${PREFLIGHT_ARGS[@]}" \
  --output-manifest "$EVIDENCE_ROOT/launch.dry-run.json" \
  --dry-run
```

### Optional: MOSS latency patch

`patches/sglang-omni-0.1.4-af3ab61-moss-latency.patch` saves about 77 ms per
two-bar MOSS phrase and removes an occasional 100 ms response delay; see
`patches/README.md` for what it changes and why the audio is unchanged. Apply it
to a copy of the installed package and let preflight verify and record it:

```bash
export MOSS_PATCH="$REPO_ROOT/patches/sglang-omni-0.1.4-af3ab61-moss-latency.patch"
export MOSS_PATCH_ROOT=/absolute/path/to/sglang-omni-patched   # new, empty directory
mkdir "$MOSS_PATCH_ROOT"
cp -a "$SGLANG_ENV/lib/python3.12/site-packages/sglang_omni" "$MOSS_PATCH_ROOT/"
patch -p1 -d "$MOSS_PATCH_ROOT" -i "$MOSS_PATCH"
export MOSS_PATCH_SHA256="$(sha256sum "$MOSS_PATCH" | cut -d' ' -f1)"
PREFLIGHT_ARGS+=(--runtime-patch-file "$MOSS_PATCH" --runtime-patch-root "$MOSS_PATCH_ROOT")
```

Then add `--moss-runtime-patch-sha256 "$MOSS_PATCH_SHA256"` to the render server
command below. Rollback: skip this section and drop that flag.

Evidence files are immutable. An identical replay is allowed; a replay with
different bytes fails. Use a new implementation/experiment id and a new output
path when any pin changes. Never reuse the dry-run path for a validated launch.

## Launch On H200

Terminals A, B, and C are separate shells. Shell arrays such as
`PREFLIGHT_ARGS` cannot be exported. Repeat the environment exports and the
array definition in each terminal that uses them, or source them from an
operator-owned, untracked file whose hash is retained with the launch evidence.

Check both selected GPUs immediately before launch. A process not named in the
experiment manifest invalidates the affected block.

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader
```

### Terminal A: Qwen

```bash
PATH="$STREAMMUSE_ENV/bin:$PATH" \
CUDA_VISIBLE_DEVICES="$QWEN_GPU" \
"$STREAMMUSE_ENV/bin/vllm" serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --port 8001 \
  --served-model-name qwen-rap \
  --max-model-len 2048 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.25
```

`--max-num-seqs 32`: the render server asks for 16 candidates per bar and, with
`--concurrent-bar-generation`, both bars at once, so one chunk is 32 sequences.
With 8, a single n=16 request decodes in two rounds (measured on H200: server
generation p50 317 ms at 8, 175 ms at 32 for one bar at a time). Each sequence
is about 400 tokens; confirm in the vLLM startup log that the reported KV cache
capacity holds at least 32 x 2048 tokens at `--gpu-memory-utilization 0.25`,
otherwise vLLM queues the extra sequences.

Wait until `curl --fail http://127.0.0.1:8001/v1/models` contains `qwen-rap`.

### Terminal B: SGLang-Omni

The preflight writes the validated launch manifest before replacing itself with
the exact `sgl-omni serve` argv. Keep this terminal and PID for targeted stop.

```bash
cd "$REPO_ROOT"
PATH="$SGLANG_ENV/bin:/usr/local/cuda/bin:$PATH" \
uv run python scripts/preflight_sglang_omni_moss.py \
  "${PREFLIGHT_ARGS[@]}" \
  --output-manifest "$EVIDENCE_ROOT/launch.validated.json" \
  --launch
```

In another H200 shell, verify both private probes before starting StreamMUSE:

```bash
curl --fail --silent --show-error http://127.0.0.1:8030/health
curl --fail --silent --show-error http://127.0.0.1:8030/v1/models \
  | python -m json.tool
```

### Terminal C: StreamMUSE Render Server

The render process shares the MOSS physical GPU only for the lightweight MMS
aligner. It probes Qwen, probes SGLang, performs a real MOSS WAV warmup, warms
MMS, verifies Rubber Band R3, verifies the producer namespace, and only then
binds a ready service.

```bash
cd "$REPO_ROOT"
PATH="$STREAMMUSE_ENV/bin:$PATH" \
CUDA_VISIBLE_DEVICES="$MOSS_GPU" \
"$STREAMMUSE_ENV/bin/streammuse-rap-render-server" \
  --host 127.0.0.1 \
  --port 8020 \
  --artifact-root "$RAP_ARTIFACT_ROOT" \
  --vllm-url http://127.0.0.1:8001/v1 \
  --vllm-model qwen-rap \
  --concurrent-bar-generation \
  --moss-model "$MOSS_MODEL_ID" \
  --moss-serving-backend sglang-omni \
  --moss-reference-wav "$MOSS_REFERENCE_WAV" \
  --moss-sglang-url http://127.0.0.1:8030 \
  --moss-reference-text-file "$MOSS_REFERENCE_TEXT" \
  --moss-sglang-reference-uri "$MOSS_SERVICE_REFERENCE_URI" \
  --moss-sglang-reference-sha256 "$MOSS_REFERENCE_SHA256" \
  --moss-request-timeout-s 120 \
  --moss-cancellation-grace-s 2 \
  --moss-model-revision "$MOSS_MODEL_REVISION" \
  --moss-runtime-version "$SGLANG_OMNI_VERSION" \
  --moss-runtime-revision "$SGLANG_OMNI_REVISION" \
  --moss-sglang-version "$SGLANG_VERSION" \
  --moss-sglang-revision "$SGLANG_REVISION" \
  --moss-runtime-environment-sha256 "$MOSS_ENVIRONMENT_SHA256" \
  --moss-runtime-config "$MOSS_RUNTIME_CONFIG" \
  --moss-runtime-config-sha256 "$MOSS_CONFIG_SHA256" \
  --aligner-device cuda:0 \
  --candidate-profile realtime \
  --moss-warp-policy gentle_sparse_r3 \
  --wire-audio-codec opus
```

With the latency patch, also pass `--moss-runtime-patch-sha256 "$MOSS_PATCH_SHA256"`.

If MMS or Rubber Band requires host-specific library paths, prepend the already
qualified paths to `PATH` and `LD_LIBRARY_PATH`; record those values in the
runtime evidence rather than adding personal paths to this document.

## Health Gate

```bash
curl --fail --silent --show-error http://127.0.0.1:8020/health \
  | tee "$EVIDENCE_ROOT/render-health.json" \
  | python -m json.tool
```

Do not send RAP work unless the response has `ready=true`, `state="ready"`,
`backend="sglang-omni"`, the expected 64-character `producer_fingerprint`, and
ready/warmed `vllm`, `moss`, `aligner`, `rubberband`, and `warmup` summaries.
The public health response is bounded and omits credentials, private paths, and
the transcript.

Artifacts now live at:

```text
<artifact-root>/<producer-fingerprint>/<request-id>/
```

The namespace contains immutable producer identity. A baseline, candidate, or
changed config gets a different fingerprint. Legacy flat artifacts remain
isolated and are never silently promoted into the candidate cache. The private
`internal/moss_synthesis.v1.json` sidecar stays on H200 and is never included in
the PCM or Opus ZIP returned to the Mac.

## Degraded Recovery

A final waiter disconnect or deadline asks the owner render to cancel. If the
active backend cannot prove release within `--moss-cancellation-grace-s`, health
switches to `state="degraded"`, `ready=false`, and `restart_required=true`.
New MOSS requests then fail fast. There is no hidden retry and no implicit load
of the in-process model.

Recovery requires both external SGLang and the render process to restart:

1. Stop or pause the Mac client so no new requests arrive.
2. Save render health, the failed request directory, sidecar, and both service logs.
3. Stop only Terminal C with `Ctrl-C` and verify its recorded PID exited.
4. Stop only Terminal B with `Ctrl-C` and verify `:8030` and its recorded PID exited.
5. Re-run the SGLang preflight/launch with the same validated pins. Use a new
   immutable launch evidence path for a genuinely new launch event.
6. Probe `:8030`, restart Terminal C, and wait for the full warmup gate.
7. Run a new short uncached recovery request. Do not report recovery until the
   upstream request is released and this request completes within normal warm p95.

Never use `killall`, never stop an unrecorded GPU PID, and never delete the
artifact root to make health look clean.

## Normal Stop And Explicit Rollback

Normal order is Mac demo, SSH tunnel, render server, SGLang-Omni, then Qwen.
The render server closes its HTTP client but intentionally does not terminate
the external SGLang process.

For rollback, stop Terminal C and Terminal B first so two MOSS model copies
cannot contend for the same GPU. Preserve candidate artifacts, switch to a
separate operational root, and start only the explicit in-process backend:

```bash
export ROLLBACK_ARTIFACT_ROOT=/absolute/path/to/rap-artifacts/inprocess-rollback

PATH="$STREAMMUSE_ENV/bin:$PATH" \
CUDA_VISIBLE_DEVICES="$MOSS_GPU" \
"$STREAMMUSE_ENV/bin/streammuse-rap-render-server" \
  --host 127.0.0.1 \
  --port 8020 \
  --artifact-root "$ROLLBACK_ARTIFACT_ROOT" \
  --vllm-url http://127.0.0.1:8001/v1 \
  --vllm-model qwen-rap \
  --concurrent-bar-generation \
  --moss-model "$MOSS_MODEL_PATH" \
  --moss-serving-backend inprocess \
  --moss-device cuda:0 \
  --moss-reference-wav "$MOSS_REFERENCE_WAV" \
  --aligner-device cuda:0 \
  --candidate-profile realtime \
  --moss-warp-policy gentle_sparse_r3 \
  --wire-audio-codec opus
```

Do not pass any `--moss-sglang-*` or SGLang runtime pin flag to `inprocess`.
The CLI rejects mixed configuration. Recheck health and one complete Mac
session before declaring rollback complete.

## Freeze And Evaluate H200 A/B

The scripts freeze design and evaluate evidence; they do not fabricate or
collect H200/Mac measurements. Produce raw rows from the real services and
retain the referenced artifacts and logs.

The corpus JSON is an array of at least 100 unique objects with exactly these
keys:

```json
{
  "sample_id": "sample-000",
  "request_id": "64-lowercase-hex-derived-by-the-production-request-contract",
  "request_payload_sha256": "64-lowercase-hex-of-the-canonical-request",
  "text": "two bars of frozen text",
  "flow_id": "frozen-flow-id",
  "tempo_bpm": 96.0,
  "token_count": 64,
  "seed": 2026090400,
  "expected_frame_count": 120000
}
```

The pins JSON has exact top-level keys `streammuse`, `gpu`, `model`,
`generation`, and `producers`. Producer pins are separated by cohort and then
by `baseline`/`candidate`. Each producer contains:

```text
producer_fingerprint, sidecar_backend, model_revision, config_sha256,
reference_audio_sha256, reference_text_sha256, runtime_environment_sha256,
launch_manifest_sha256, health_identity, health_version
```

Use `null` for `producers.serving-only-parity` unless the pinned candidate has
proved that omitted `ref_text` works. Its producer pins must use
`reference_text_sha256="unavailable"`. Cohort B is mandatory; its baseline text
hash is `unavailable`, while its candidate uses the verified transcript hash.
No token, transcript text, hostname, or personal path is accepted in pins.

Freeze before running candidate samples:

```bash
uv run python scripts/sglang_moss_ab.py freeze \
  --experiment-id sglang-moss-h200-v1 \
  --implementation-id "$IMPLEMENTATION_ID" \
  --created-at-utc 2026-09-04T00:00:00Z \
  --corpus "$EVIDENCE_ROOT/corpus.json" \
  --pins "$EVIDENCE_ROOT/pins.json" \
  --schedule-seed 20260904 \
  --bootstrap-seed 20260905 \
  --bootstrap-resamples 10000 \
  --cohort-a-evidence not-supported-by-qualified-build \
  --output "$EVIDENCE_ROOT/experiment.json"
```

Add `--cohort-a-enabled` only with retained capability evidence. Add
`--speaker-similarity-hard-gate` only when the tool/model and complete metric
collection were frozen before the run.

Run the manifest's 30-sample qualification and four counter-balanced final
blocks exactly. Before starting each backend/block, bind a fresh empty artifact
root to that exact experiment coordinate. For example:

```bash
AB_ROOT="$EVIDENCE_ROOT/artifacts/production-candidate/qualification/candidate"
uv run python scripts/sglang_moss_ab.py prepare-root \
  --manifest "$EVIDENCE_ROOT/experiment.json" \
  --root "$AB_ROOT" \
  --cohort production-candidate \
  --phase qualification \
  --block-id qualification \
  --backend candidate
```

Repeat for both qualification backends and every enabled cohort's four final
blocks and two backends. The command refuses a populated root, a root bound to
another coordinate, and a symlink root. Keep
`_streammuse_ab_artifact_root.v1.json` in place while rendering, and copy its
`root_manifest_sha256` into every row produced from that root. Build the
evaluator input after all runs:

```bash
find "$EVIDENCE_ROOT/artifacts" \
  -name _streammuse_ab_artifact_root.v1.json -type f -print0 \
  | sort -z \
  | xargs -0 cat > "$EVIDENCE_ROOT/artifact-roots.jsonl"
```

The row JSONL schema is `streammuse.sglang_moss_ab_row.v1`; exact nested keys
are documented by `_validate_row` in
`src/streammuse/experiments/sglang_moss_acceptance.py`. The evaluator requires:

- 60 Cohort B qualification rows, 30 per backend;
- 200 final rows per enabled cohort, 100 per backend;
- one candidate service call and zero cache hits for every uncached candidate row;
- six recovery rows, both backends crossed with timeout/disconnect/cancellation;
- both backend scores from at least two reviewers for every frozen blind sample;
- complete timing, ASR, MMS, R3, artifact, and provenance fields, including failures.
- complete root-manifest coverage and a matching
  `provenance.artifact_root_manifest_sha256` on every row.

Evaluate without Mac evidence first:

```bash
uv run python scripts/sglang_moss_ab.py evaluate \
  --manifest "$EVIDENCE_ROOT/experiment.json" \
  --rows "$EVIDENCE_ROOT/rows.jsonl" \
  --artifact-roots "$EVIDENCE_ROOT/artifact-roots.jsonl" \
  --recovery-probes "$EVIDENCE_ROOT/recovery-probes.jsonl" \
  --blind-scores "$EVIDENCE_ROOT/blind-scores.jsonl" \
  --output "$EVIDENCE_ROOT/h200-acceptance.json"
```

Exit `0` means `promotion-candidate`, exit `1` means `blocked` or
`experimental`, and exit `2` means malformed/untrusted evidence. Missing
quality data fails closed. A H200 `promotion-candidate` is only permission to
enter the Mac gate; it is not rollout approval.

## Mac Gate And Canary

Forward only the render server:

```bash
ssh -o ExitOnForwardFailure=yes -N \
  -L 8020:127.0.0.1:8020 \
  user@h200-host
```

Run the normal Opus demo command from `rap-demo-quickstart.md`. Record baseline
and candidate `deadline_misses`, `playback_underruns`, and
`post_cancel_gpu_busy_requests` with their producer fingerprints in a
`streammuse.sglang_moss_mac_e2e.v1` JSON object. Re-run the evaluator with
`--mac-evidence` and a new immutable report path.

Keep `--moss-serving-backend sglang-omni` explicit during canary. Observe at
least 20 complete games over two workdays unless the signed experiment defines
a stricter window. Any schema/cache/provenance error, OOM/hang, repeated
degraded state, or quality hard-gate regression triggers the rollback above.

Do not change the quickstart default until this gate and rollback rehearsal are
both signed off.

## Local Verification

```bash
uv run pytest \
  tests/unit/infrastructure/rap/test_sglang_moss_tts.py \
  tests/unit/infrastructure/rap/test_producer_manifest.py \
  tests/unit/presentation/test_rap_render_server.py \
  tests/unit/scripts/test_preflight_sglang_omni_moss.py \
  tests/unit/experiments/test_sglang_moss_acceptance.py -q
```

These are hermetic contract tests. They do not replace the H200 qualification,
real cancellation/queue recovery evidence, blind listening, Mac playback gate,
or canary observation window.
