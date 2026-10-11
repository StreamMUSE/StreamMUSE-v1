# Fixed-bar history repair: opt-in integration

Branch: `codex/bar-quality-history-experiment`. This is an experimental backend
entry point, not a change to the default Web, scheduler, or playback policy.

## Scope

The first version repairs one explicitly selected 4/4 bar. A separately owned
model process resamples that bar from the observed Melody and the preceding
accompaniment. If its result is ready and the captured live state is unchanged,
the backend can replace that bar in the **future model context**. The next
ordinary generation rebuilds its prompt from this context. This is not replacement
of a persistent cross-request KV cache.

Original raw generation metadata and previously returned playback events are
not rewritten. The repair audit keeps the original ticket/history, candidate,
provenance, decision and application time separately. Persist both
`engine.history_repair_audit()` and `backend.history_repair_audit_snapshot()`
before resetting the session. Actual playback remains in the existing export.

No automatic musical quality decision or drop threshold is included. Applying
a fixed intervention is not evidence that the candidate is better.

## Explicit partial-repair experiment

`run_realtime_comparison.sh preserve_cross_boundary` selects a separate
experimental policy; the default remains `strict` whole-bar replacement.
The partial policy retains original notes crossing either endpoint, including
notes with note-off exactly at the right boundary. It discards sampled notes
crossing either endpoint and any interior candidate overlapping a retained
note of the same pitch. Only complete remaining interior note pairs replace
the original interior. No pitch, onset, duration, or velocity is rewritten and
no release is synthesized. A candidate with no compatible interior notes or
no event change is rejected. Final validation uses the unchanged transaction
validator and compare-and-swap check.

This is a mixture of original boundary notes and sampled interior notes, NOT
an unmodified model-generated bar. The original candidate and its token trace
are stored separately; projected events do not claim a matching sampled-token
trace. The main model rebuilds invalidated context tokens on its next prefill.
This may preserve an undesirable boundary note; it is a safety/conservatism
trade-off, not a guarantee of higher musical quality.

The three saved guarded candidates passed CPU structural checks with 5, 7, 5
selected interior notes. That check is not live application evidence. Before
the partial-policy full-song run, 164 remote tests passed (including eight
projection tests). Full-song baseline/partial conditions each use seeds 0/1/2.

The H200 GPU2 full-song run `bar_history_partial_repair_20261007/run_E64spx`
completed with 3/3 partial repairs applied (5/7/5 interior notes for seeds 0/1/2).
All 309 paired Melody-token and Melody-roll digests matched. Full-input tokens
first differed at tick 84 in all three pairs; raw tokens first differed at
84/84/92. Applied audits verify unchanged main EMPTY guard and restored worker
bar-start constraints. Candidate generation/projection took 0.249--0.306 seconds.
Original full post-join ISR_f was 0.9725 baseline vs 0.9612 partial repair.
At the immediately following beat [84,88), baseline was 4/4 frames for each seed;
repair was 3/4, 4/4, 4/4. Timing/content effects have not been disentangled.
No music-quality improvement is established. Actual playback still differs from
backend raw; onset/offset audits are included rather than asserting equality.
Local results: `F:/repos/StreamMUSE-v1/remote_results/00_bar_history_probe_20261007/realtime_partial`.

## Safety boundaries

- Both an enabled backend and explicit controller attachment are required.
  `LEKAI_HISTORY_REPAIR_ENABLED` defaults to false; setting it alone does not
  allocate a worker or schedule repair.
- The guarded integration captures the actual bar-start EMPTY-token guard and
  tonal tracker state for the private worker. Missing snapshots are rejected.
  The live guard remains tied to actual generation chronology: replacing model
  conditioning does not reset it or overwrite it with counterfactual guard state.
- Live tonal state and sampling RNG are preserved. Candidate resampling has its
  own explicit seed; it is not a clone of the live sampling generator state.
- The Melody encoding horizon and generated-history frontier are checked
  separately. For one-beat ACC-first requests, submission is attempted after
  the request ending at `bar_end + 4`, not merely after generating the bar.
  This checks available model-input history, not a new guarantee that every
  human event arrived on time.
- Candidate completion is polled before the next ordinary generation. There
  is no wait for it, no clock backdating, no late-note replay and no retry loop.
  If generation has advanced since the snapshot, the candidate is stale.
- Unsafe note boundaries or changed active-note carry are rejected. No note-off
  is fabricated to make a replacement fit.
- Reset, history injection, clearing, or configuration changes invalidate the
  controller. Attach a **new** controller after session initialization to re-arm.
  Reapplying identical runtime/sampling configuration does not retire it;
  retirement is explicitly recorded even when no job had been submitted.
- One prestarted worker process owns its model and environment. A second model
  on the same GPU can still contend with inference; process isolation does not
  imply performance isolation.
- The worker checks checkpoint, BPM, dtype, context, sampling and constraint
  configuration against the ticket before submitting. Its generation logs use
  a new independent audit directory, never the live backend's log directory.
  Completed/discarded candidates remain there even after a session reset.

## Explicit Python entry point

`LekaiContinuationEngine.attach_history_repair(controller)` is the experimental
entry point. No Web/API controls or automatic server worker are added here.
Initialize/reset/inject/configure the session first, then attach:

```python
from streammuse.infrastructure.inference.history_repair_controller import FixedBarHistoryRepair
from experiments.bar_quality_history.process_worker import ProcessRepairWorker

# Run under an if __name__ == "__main__" guard (multiprocessing spawn).
# Verify GPU permission/occupancy externally BEFORE creating the worker.
# verified_config names the real checkpoint and all explicit model parameters.
worker = ProcessRepairWorker(
    config=verified_config, gpu="2", seed=101,
    audit_dir="/path/to/new/repair_audit", allow_gpu=True,
)
# Wait for worker.ready() in setup, BEFORE starting the real-time clock.
# backend was constructed with history_repair_enabled=True.
controller = FixedBarHistoryRepair(start_tick=32, end_tick=48, worker=worker)
engine.attach_history_repair(controller)

# Run the existing input/inference/playback loop without modifying its timing.
# Save audits, detach, then close outside the real-time loop:
engine.attach_history_repair(None)
worker.close()
```

The sketch omits the existing application's session setup and loop; it is not
a standalone launch command. Do not attach before worker readiness. Worker
failure/unavailability keeps the original history.

For the normal P+C lifecycle, `configure_history_repair_factory(factory)` attaches
a fresh controller after each successful Prompt history injection. The default
factory is absent. `launch_repair_server.py` is an isolated experimental launcher
which uses this hook and the normal server. `run_realtime_comparison.py` uses the
existing production MIDI CLI, HTTP server, scheduler, and playback export.

The guarded full-song implementation passed 91 remote CPU tests before its
2026-10-07 GPU2 run (history transactions, EMPTY guard, and P+C backend).
This is a separate test run from the earlier unguarded test counts below.

The first full-song run (`run_YTjcGv`) rejected all three repairs before worker
submission: the existing Prompt/Continuation handoff included duplicate
note-offs at tick 32. The revised validator only tolerates a same-tick release
that already successfully closed that pitch, strictly before the repair bar.
It preserves those original events and still rejects orphan releases, repeated
onsets, zero-length notes, and duplicate releases in candidates. The revision
passed 97 remote pytest cases plus 59 harness/controller/worker unittest cases.
`run_h2znzt` is the separate full-song rerun, not a replacement of the first logs.

That rerun completed all six full-song trials. Three candidates finished in
0.232--0.299 seconds, but all were rejected for different active pitches at the
replacement endpoint (0/3 applied). Across the three pairs, all 309 Melody-token,
Melody-roll, full-input-token and raw-token digests matched. Original ISR_f over
ticks [32,444) was 0.9741 baseline and 0.9676 with the repair worker. One piece and
one run per seed cannot establish performance impact. This does NOT validate
successful guarded replacement in the full playback path or improved quality.
Local results: `F:/repos/StreamMUSE-v1/remote_results/00_bar_history_probe_20261007/realtime_guarded`.

Successful application records are also copied into the controller audit before
the normal Stop cleanup clears backend state. This keeps evidence available to
the experiment without retaining live context for the next session.

## Verification status

The earlier isolated H200 probe completed real generation and history splicing.
It did not test this live CAS/controller or concurrent real-time performance.
2026-10-07 verification: 215 pytest cases passed (backend transaction, standard
backend, P+C backend, tonal/empty guards, and session contract); 59 stdlib cases
passed (existing isolated harness plus controller/worker tests). CUDA was hidden
for backend tests, and no checkpoint/model workload was run locally.
`git diff --check` passed. One existing Starlette/httpx deprecation warning was
reported. A successful CPU test is not a claim of music quality improvement or
an ISR result. No changes have been deployed to the Spark session used by
performers.

Subsequent H200 GPU2 validation is complete in
`/data/home/yuanxin/experiments/bar_history_live_20261007/run_ANUq1P`.
One Melody, three continuation seeds, paired baseline/repair paths at a 90 BPM
request cadence (80 BPM conditioning), plus an explicitly labeled identity
control. Real regenerated candidates: one applied, one rejected as stale, one
rejected for mismatching end carry. Every paired Melody token/roll comparison
matched; rejected candidates left full inputs/raw outputs/returned events
unchanged. The accepted candidate changed context and output only from tick 52.
Identity replacement preserved the complete generation sequence. These are
paced backend results, NOT HTTP/playback ISR or music-quality evidence.

The integration now supports cross-start sustained notes by restoring the
immutable prefix carry; end carry must still match and the suffix is preserved.
The updated transaction tests passed 47/47 on H200. No local GPU was used.
Detailed local results and README:
`F:/repos/StreamMUSE-v1/remote_results/00_bar_history_probe_20261007/live_run_ANUq1P`.

Worker shutdown is outside the playback loop. After a five-second graceful
wait it terminates only its own process, with a further bounded terminate/kill
wait, and records forced termination in `shutdown.json`. Python 3.10 requires
using the executor's private process registry for this bounded teardown; that
compatibility point must be rechecked when changing Python versions.
