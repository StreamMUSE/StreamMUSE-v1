# Isochron Integration Design

## Goal

Add the Isochron two-stage prompt-continuation workflow to `new_system_stanley`
without changing the established standard, Voice, or Rap behavior. Complete the
previously missed composition-root work so CLI and web construction ultimately
flow through `RuntimeSessionBuilder`, with that migration isolated behind
characterization tests.

## Compatibility Contract

- The existing 15 failing pytest node IDs are an accepted baseline and are not
  part of this work.
- Every integration checkpoint must introduce zero new failing node IDs, zero
  new collection errors, and must not turn any previously passing test into a
  skip. Newly ported Isochron tests may retain their documented environment-gated
  skips.
- Standard inference remains the default. Prompt continuation is enabled only
  by explicit configuration.
- Existing deterministic session metadata, raw-token history, request lifecycle,
  artifact tiers, run horizons, Voice tools, and Rap tick observers are retained.
- Rap and prompt-continuation modes are independently available. Their combined
  use is rejected explicitly in this first integration rather than silently
  constructing an unverified hybrid runtime.

## Approach

The integration is a selective functional port, not a wholesale Git merge.
Isochron-only models, services, clients, scripts, and tests are introduced as
new modules first. Existing target files remain authoritative; the smallest
required configuration fields, routes, and selection branches are then added.

After both standard and continuation paths are independently verified,
`RuntimeSessionBuilder` is expanded to reproduce the existing standard runtime
and construct the continuation runtime. CLI and web are migrated one at a time.
This final composition-root change is a distinct commit and can be reverted
without removing the prompt-continuation implementation.

## Components

### Prompt-continuation core

The port includes `PromptContinuationRealtimeService`, `MusicalSequence`, the
continuation model package, the prompt-continuation backend/engines/schedulers,
and `PromptContinuationHttpClient`. These components do not alter standard
inference when unselected.

### Configuration and entry points

`ApplicationConfig` gains disabled-by-default continuation settings and optional
leading-rest trimming. CLI and web parse these values, while
`RuntimeSessionBuilder` selects either the existing `RealTimeMusicService` or
the new `PromptContinuationRealtimeService`.

### Inference server

The existing standard Lekai backend remains authoritative. Prompt-continuation
state and endpoints are added alongside it. Existing reset epochs, deterministic
seeds, fingerprints, request metadata, and raw-token history contracts are not
replaced by older Isochron implementations.

### MIDI timing

Target clock anchoring remains unchanged. Leading-rest trimming is added as an
opt-in preprocessing choice and must not affect the default MIDI-file schedule.

### Tooling

Two-stage offline/realtime consistency runners and focused tests are included.
Generated VitePress cache/dist files and generated `egg-info` files are excluded.
Large MIDI comparison and Spark benchmark artifacts are kept out of the runtime
integration and can be reviewed separately.

## Error Handling

- Invalid continuation configuration fails before worker threads start.
- Simultaneous Rap and continuation selection produces a clear configuration
  error in this initial integration.
- Prompt backend failures remain inside the continuation path and cannot change
  standard inference fallback behavior.
- Every service built by the composition root has one idempotent shutdown path.

## Verification

Each commit runs its focused tests followed by the full suite. Full-suite output
is compared by exact node ID against the frozen baseline from commit `1734ea9b`:
15 failures, 439 passes, and 2 skips. Completion requires no additional failure,
collection-error, or skip identities and targeted prompt-continuation tests must
pass.

## Non-goals

- Fixing the 15 accepted baseline failures
- Rewriting standard model or backend behavior
- Downgrading the newer task/debugger implementations
- Combining Rap and prompt continuation in one live session
- Importing generated documentation builds or generated package metadata
