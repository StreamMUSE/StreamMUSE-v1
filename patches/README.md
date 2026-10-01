# Runtime patches

## `sglang-omni-0.1.4-af3ab61-moss-latency.patch`

Applies to the pinned SGLang-Omni 0.1.4 (`af3ab61`) package as installed in
site-packages (paths `a/sglang_omni/...`, apply with `-p1`). SGLang 0.5.18.
It changes only the MOSS-TTS model and the speech HTTP endpoint:

| Change | File | Why it does not change the audio |
|---|---|---|
| One gather for the 32 audio embeddings in decode batches of at most 8 rows | `models/moss_tts/sglang_model.py`, `_prepare_multi_modal_inputs` | An embedding lookup copies rows, and the rows are added in the original channel order (text first), so the bf16 sum is bit-identical. It is used only when every audio table is an unquantized single-GPU `VocabParallelEmbedding`, which is exactly what the per-table path calls (`F.embedding`). The tables are re-pointed at slices of one stacked buffer at load time, as the fused audio heads already are, so there is no steady-state memory. A replaced table (pointer check) or a weight-share follower whose storage is not one block falls back to per-table lookups. |
| Text head computes only the two control-token logits in audio decode | `models/moss_tts/sglang_model.py`, `compute_channel_logits` | It mirrors `LogitsProcessor._get_logits` for a plain text head: the same lm-head dtype path, in-place `logit_scale`, and a float32 result. It multiplies the two control rows instead of the 155k-row vocabulary before `index_select`. Softcapping, LoRA, quantized heads and TP/DP gathers fall back to the original path. The GEMM shape differs, so bit equality is not guaranteed by construction; it is verified on the real service (below). |
| The disconnect watcher is cancelled without waiting | `serve/openai_api.py` | The watcher only polls `request.is_disconnected()`. Not waiting for its cancellation removes an occasional 100 ms delay after a finished response. Generation tasks keep the bounded wait. |

Measured with the experimental equivalent (2026-09-25 profiling report,
section 4.4): MOSS median −77 ms per two-bar phrase, and identical source WAVs
in 60 of 60 runs.

### Build the patched runtime

The patch is applied to a copy of the installed package. The pinned
environment stays untouched, and rollback means dropping the copy.

```bash
SITE="$SGLANG_ENV/lib/python3.12/site-packages"
PATCH="$REPO_ROOT/patches/sglang-omni-0.1.4-af3ab61-moss-latency.patch"
PATCH_ROOT=/absolute/path/to/sglang-omni-patched   # new, empty directory
mkdir "$PATCH_ROOT"
cp -a "$SITE/sglang_omni" "$PATCH_ROOT/"
patch -p1 -d "$PATCH_ROOT" -i "$PATCH"
sha256sum "$PATCH"
```

`scripts/preflight_sglang_omni_moss.py --runtime-patch-file "$PATCH"
--runtime-patch-root "$PATCH_ROOT"` does the following:

- checks that the patch is applied there (`git apply --reverse --check`);
- records `runtime.patch_sha256` in the launch manifest;
- puts `$PATCH_ROOT` first on the service's `PYTHONPATH`.

Pass the same hash to the render server with `--moss-runtime-patch-sha256`, so
patched output gets its own producer fingerprint and cache namespace. To roll
back, start the service without the two preflight flags and the render server
without the hash. The fingerprint then returns to the unpatched value.
