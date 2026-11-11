# Research UI, 2026-11-05

## Delivery

Separate compact research page with four tabs: Directed Story, Microphone,
Characters, and Coordination. Uses the existing StreamMUSE color and typography
tokens, full-width bands, wrapping controls, bounded logs, and a 360px layout.
No new packages, Python edits, service starts, commits, audio policy changes,
automatic runtime starts, automatic resets, or actual microphone recordings by
this worker. Parent owns router/page mounting and browser acceptance.

Parent reports the page is mounted at `http://127.0.0.1:8019/research`; the old
8018 runtime is unchanged. Parent also reported its initial fake-microphone
browser test passed, with screenshots under
`output/research_day_20261105/browser`. Those browser results were not independently
executed by this worker. The final playback-strip addition requires the parent's
follow-up browser pass.

## Owned Paths

- `src/streammuse/presentation/rap_demo/static/research.html` (new)
- `src/streammuse/presentation/rap_demo/static/js/rap-research.js` (new)
- `src/streammuse/presentation/rap_demo/static/css/rap-research.css` (new)
- `src/streammuse/presentation/rap_demo/static/index.html` (one Research link)
- `tests/integration/rap-research.test.cjs` (new)
- `docs/reports/2026-11-05-research-ui.md` (this report)

The existing Rap Lab link was placed in the wrapping Live audio heading rather
than its crowded mobile runtime toolbar. The research page has a persistent
Rap Lab return link. Existing GUI JavaScript and CSS were not edited.

## Contracts

### Directed Story

- GET `/api/story`; POST `/api/story/events` with `{text,event_id}`.
- Implements the completed `story_api.py` schema, including nested acceptance
  `event`, `duplicate`, and `effective_from_bar`.
- Displays the 32-event and 512 cumulative UTF-8-byte limits; validates drafts
  without silently truncating transcripts. Server validation remains authoritative.
- Unchanged-text retries reuse their event ID, including a locally accepted event
  discovered by polling after an uncertain response.
- Direction ledger is labeled separately from playback continuity. Acceptance
  does not claim generation, playback, or semantic fulfillment.
- Current-playback strip polls GET `/api/state`: current bar, matching frozen
  lyric, topic, source/fallback, runtime state, and errors. Future-bar lyrics are
  not substituted into the current bar. Displayed bars are one-based, matching
  Rap Lab; API bar indices remain zero-based.
- Explicit Start/Stop use POST `/api/control/start` and `/api/control/stop`.
  Controls mirror existing runtime-state gating and serialize pending commands.
  There is no Reset call or automatic Start/Stop action.

### Microphone

- GET `/api/microphone`; POST `/api/microphone/transcribe` with raw RIFF/WAVE,
  `Content-Type: audio/wav`, and a fresh `X-Utterance-ID`.
- Explicit pointer or keyboard press-and-hold only. No device access on page
  mount, tab selection, polling, or destination changes.
- Web Audio requests a 16 kHz context, captures mono, serializes little-endian
  PCM16, and validates the backend's 8-48 kHz range. A ScriptProcessorNode feeds
  a zero-gain output connection; captured audio is never monitored audibly.
- Release automatically stops tracks and transcribes. Both wall-clock timeout
  and sample count cap capture at 15 seconds. Cancel, pointer cancellation,
  focus/window blur, hidden document, tab exit, and page exit stop capture.
  Late permission/context results cannot restart a cancelled capture.
- Shows returned transcript immediately, with rounded integer transcription ms.
  Transcript destination is the default. Explicit Story destination automatically
  posts nonempty text using the transcription event ID; there is no confirmation.
- Failed story submission retains the visible transcript. No automatic network
  retry or local audio/transcript persistence is added.

### Characters And Coordination

- Exact completed backend: GET `/api/characters/status`, POST
  `/api/characters/start`, POST `/api/characters/stop`, and GET
  `/api/characters/events?after=...&run_id=...`.
- Start body is `{mode,policy,turns:8}`; policy values are `fixed_turn`, `naive`,
  and `coordinated`. Dialogue locks to `fixed_turn`. Counting targets 1 through
  10 regardless of the dialogue turns parameter; labels say Start count.
- Characters displays backend personas and accepted conversation turns.
  Coordination displays proposal/event logs and the backend metrics table,
  including correctness source and GPU cancellation confirmation.
- Tabs have separate controls/views but share the backend's single active
  character experiment. Running-state controls apply to both views. No second
  experiment engine, fake character output, or invented coordination API exists.
- Cursor polling resets on run changes; event reduction deduplicates and caps
  visible logs at 200 entries. Model identity alone never implies readiness.
  Microphone uses its actual `transcriber_state`; absent character/model health
  is explicitly displayed as Status not reported.

## Verification

New tests use Node's built-in test runner plus the repository's existing
VM-loaded browser-global pattern. No test dependencies or Python tests were added.

```sh
node --test tests/integration/rap-research.test.cjs
node --check src/streammuse/presentation/rap_demo/static/js/rap-research.js
git diff --check
```

Node result: **23 passed**. Covers WAV header/PCM clipping/rate/duration,
event deduplication/bounds, microphone state and resource cleanup races,
permission rejection, empty capture, UTF-8 budgets/idempotent retries,
exact story and character contracts, persona labels, no startup microphone
access, current/future lyric projection, latency rounding, and explicit runtime
Start/Stop without Reset. Recorder tests use fake devices only.

Adjacent existing GUI regression command:

```sh
env PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/integration/test_rap_demo_browser_reducer.py \
  tests/integration/test_rap_interaction_controls.py \
  tests/integration/test_rap_plan_controls.py \
  tests/integration/test_rap_syllable_display_browser.py
```

Result: **88 passed**. Full repository suite not run; other workers own backend
verification. No live model, actual device, or audible playback checks were run
by this worker.

## Parent Acceptance

- Reload `/research` (research assets now use `v=20261105-2`).
- Confirm desktop and 360px controls, text wrapping, and playback strip.
- Verify directed lyrics while using the explicit Start/Stop controls; stopping
  must preserve the session story and must not trigger Reset.
- Recheck fake-WAV release, permission denial, cancellation, 15-second cutoff,
  integer latency, and explicit Story auto-routing.
- Verify character/counter run transitions, failures, actor labels, cursor logs,
  and host-referee metric attribution against the mounted backend.

Known browser compatibility limit: ScriptProcessorNode is deprecated but retained
by current desktop browsers. AudioWorklet migration can be isolated later; it was
not mixed into the deadline-bounded backend integration. Browser capture requires
a secure context or localhost. The playback strip is a polled projection, not
sample-accurate playback confirmation or proof that a story direction was fulfilled.

## Real Character Browser Acceptance Follow-Up

Added standalone `scripts/experiments/research_day/characters_browser.cjs` using
the bundled Playwright module and installed Chromium 1217 from
`browser_acceptance.cjs`. Executed against the existing 8019 runtime on
2026-11-05, 14:27:38-14:28:02 UTC. No response mocks, microphone permissions,
story writes, rap control calls, or runtime resets. Only character Start/Stop
mutations were observed. The configured model was `qwen-rap` at
`http://127.0.0.1:18711/v1`.

| Check | Actual result | Model calls |
| --- | --- | --- |
| Fixed-turn dialogue | Completed; 8 alternating accepted messages | 8 |
| Coordinated counting | Completed; exact alternating 1-10 accepted sequence | 20 |
| Naive counting | Expected failure: illegal_number; 1 accepted, 1 rejected | 2 |
| Stop during dialogue | Cancelled with user_stop event | 1 |
| Fresh dialogue after Stop | Completed; 8 accepted messages | 8 |

All five checks passed. Accepted text/persona labels and every rendered metric
were compared against actual API snapshots. Event IDs/sequences, model-response
provenance, terminal error rendering, and state transitions were checked.
No JavaScript page errors or character HTTP failures were observed. Both 1440px
and 360px views passed page overflow, control overlap, text-fit, and hover-size
checks. Inspected representative desktop and mobile screenshots visually.

Artifacts: `output/research_day_20261105/characters_browser/summary.json` and
18 PNG screenshots in that directory. JSON includes complete final snapshots,
event pages, mutation audit, model metrics, and layout measurements. The final
character run is completed, not running. Stop followed by a fresh run is tested;
there is no same-run resume API. GPU-side cancellation is not claimed.

Reproduce with `node scripts/experiments/research_day/characters_browser.cjs`.
Optional environment overrides match the adjacent script: `DEMO_URL`,
`PLAYWRIGHT_MODULE`, and `CHROMIUM_PATH`. Local-network/browser sandbox approval
may be required. The script refuses to replace an active pre-existing run and
only cleans up its own run on failure.

### Microphone Source Follow-Up

Automatic transcript-to-story submissions now include `source: "microphone"`
with the transcription event ID. Typed submissions omit source and retain the
backend's `typed` default. Added a Node integration regression covering capture
release, raw WAV submission, transcript display, and the exact story request body;
it failed on the missing source before the fix. All **24 Node tests pass**.
Research JavaScript cache version is now `20261105-3`. No additional browser,
GPU, or live-server tests were started during the parent's client restart.

### Session-Bound Story Submissions

P2 fix: every GUI story POST now includes the expected `session_id`. Typed
pending events capture it on the initial submit and preserve it with the same
event ID on retry. A newly observed session clears the old retry and retains
the draft with a warning. The unchanged stale draft is blocked until explicitly
edited; it is never silently rebound to the new session.

Microphone Story destination captures its session synchronously at record PRESS,
before permission, capture, or ASR. Story polling while that destination is
selected can detect Reset without rebinding the capture. Unknown or changed
sessions retain the transcript and warn instead of auto-submitting. If Reset
has not yet been observed, the POST still sends the original session for the
backend's atomic 409 rejection. Typed source still defaults server-side;
microphone source remains explicit.

Success requires the acknowledgement and a refreshed story snapshot to match
the bound session, with no intervening observed session change. A stale or
unconfirmed acknowledgement neither clears the typed draft nor reports the
transcript as accepted into the current story.

Verification: **32 Node tests passed**, including stable typed retry IDs/session,
Reset invalidation, late acknowledgements, missing session, Reset during capture,
Reset during ASR, retained transcripts, and guarded backend rejection for an
unobserved Reset. New regressions failed before the implementation. JavaScript
syntax and `git diff --check` pass. Research JavaScript cache version is now
`20261105-4`. No service operations or live API/GPU tests were performed for this
fix. Parent owns the API's optional field and the controller's atomic guard.

### Preserve Retry Identity On Unconfirmed Refresh

Separated known session changes from unconfirmed responses. A successful story
POST followed by a failed, missing-session, or superseded confirmation GET does
not report acceptance and does not discard the original pending event ID/session.
An unchanged retry reuses both. Only a known differing session or an observed
session transition invalidates the retry and marks the retained draft stale.
Missing session metadata is not treated as evidence of Reset.

Added three regressions: accepted POST then GET 503 and same-ID/session retry;
missing-session GET then recovery; and a confirmation GET superseded by a
concurrent microphone submission. All three failed before the correction and
now pass with the existing Reset safeguards. **35 Node tests pass**. JavaScript
cache version is `20261105-5`. No live API tests or service operations were run.

### Final Verification And UI Freeze

Reran the three retry-confirmation regressions successfully, then the complete
Node suite: **37 passed**. The parent reports actual Chromium Reset-during-ASR
acceptance passed with the real transcript retained, old session rejected, and
new ledger empty; artifacts are under
`output/research_day_20261105/microphone_reset_browser`.

Added compact Playing context / Latest accepted version readouts using existing
`frozen_bars[*].story_version` and `/api/story.version`. The current projector
already copies that event payload, so no Python or backend change was required.
Only the matching frozen current bar contributes the playing version; future or
draft versions are never substituted. A missing version or mismatched playback
and story session displays `--`, not a misleading cross-session comparison.
Version provenance indicates generation context, not semantic fulfillment.

Final JavaScript cache version: `20261105-6`. UI is frozen after local verification;
no additional browser/API/GPU tests or service operations were started.

### Live Provenance Diagnosis And Final Fix

The final live check exposed an identity mismatch missed by the original unit
fixtures: `/api/state.session_id` is the event publisher's base run ID, while
`/api/story.session_id` is the controller's base ID plus `-e<epoch>`. The state
snapshot exposes that epoch separately as `coordinator_epoch`. Exact string
equality therefore hid a valid frozen `story_version: 2` after Reset.

Narrow UI fix: retain `coordinator_epoch` with the playback snapshot and accept
either identical session IDs or the exact base-ID/epoch-derived controller ID.
No suffix stripping or base-only match is used. Wrong, missing, or unknown epochs
and missing story IDs remain unmatched. No Python, projector, backend, audio,
or timing policy changes were required. Research JS cache is `20261105-7`.

New standalone real-browser script:
`scripts/experiments/research_day/final_ui_browser.cjs`. Its `--diagnose` mode
reproduced the old display: Playing context `--`, Latest accepted `v2`, despite
positive frozen provenance. Verification mode then observed Playing context
`v2`, Latest accepted `v2`, and a displayed lyric identical to the current frozen
bar: "The key slides, clicks, unlocks space, holds case". This confirms UI
provenance rendering, not semantic story fulfillment or a timing-performance claim.

New evidence, without replacing prior experiment summaries:

- `output/research_day_20261105/final_ui/provenance-before.json`
- `output/research_day_20261105/final_ui/provenance-before-1440.png`
- `output/research_day_20261105/final_ui/provenance-before-360.png`
- `output/research_day_20261105/final_ui/provenance-after.json`
- `output/research_day_20261105/final_ui/provenance-after-1440.png`
- `output/research_day_20261105/final_ui/provenance-after-360.png`

Real Chromium before/after checks passed, with no page errors or horizontal
overflow. Representative mobile evidence was inspected by this worker; parent
independently inspected desktop/mobile and confirmed `v2/v2` with no overflow.
Focused regression first reproduced the base/epoch mismatch. Final **42 Node
tests pass**; parent also reran and confirmed **88 original GUI tests pass**.
Tests include wrong/missing epoch and missing-session negative cases.

Both playback checks stopped without Reset and preserved the two diagnostic
directions. Subsequently, on the parent's explicit final cleanup request, this
worker reset 8019 at **15:26:08 UTC** and read back: stopped, controller epoch
`e8`, story version `0`, zero story events, and zero frozen bars. Evidence is
`output/research_day_20261105/final_ui/cleanup.json`. Parent subsequently assumed
final reset/health ownership; no further API calls or mutations were made by
this worker. Existing timing experiment summaries were not modified.

UI work is finished and frozen. Parent owns final progress and health reporting.
