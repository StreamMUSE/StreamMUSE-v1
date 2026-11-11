# Interaction Backend Review: 2026-11-05

Current disposition: all three original findings are resolved by the focused
backend fixes reviewed below. See the final addendum for independent reproduction
results and the 77-test verification run. The original findings are preserved
as historical evidence, not as currently open issues.

Review scope: directed-story backend/API and character backend/API. Three
reproduced findings remain in the last inspected source. No implementation files
were changed by this review. Parent requested this handoff without further tests.

Workspace root:

`/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/MBZUAI/StreamMUSE-NewSystem/.worktrees/real_rap`

Line references below were verified in the source before this report was written.

## P1: Story Submission Holds the Playback Lock During Socket Send

Primary location:

`src/streammuse/application/rap/bar_stream.py:557`

Related locations:

- `bar_stream.py:572`: duplicate retry performs socket send inside the outer lock.
- `bar_stream.py:575`: new submission calls `update_plan()` inside the outer lock.
- `bar_stream.py:615`: `update_plan()` performs socket send after releasing only
  its own nested acquisition; the caller's outer RLock remains held.
- `src/streammuse/infrastructure/rap/lyric_stream_client.py:153`: synchronous
  `socket.send(...)`, which can block on transport/backpressure.
- `bar_stream.py:620` and `bar_stream.py:654`: Stop and Reset require the same lock;
  the tick path also uses it.

### Cause and Impact

`submit_story_event()` wraps its entire operation in `with self._lock`.
Calling `update_plan()` reenters this RLock. Although `update_plan()` places
network delivery outside its own `with` block, the outer acquisition is still
held. A slow send therefore blocks controller progress, including playback tick
handling, Stop, Reset, and story snapshots. Moving only the inner send is not
sufficient; the outer lock must be released too.

### Reproduced Locally

Used existing `_controller`, `_Stream`, and `_Source` fixtures from
`tests/unit/application/rap/test_bar_stream.py`. Set `stream.capabilities` to
`{"story_context_v1"}` and `controller._started = True`. Replaced `stream.send`
with a function that sets an `entered` threading event and waits on a `release`
event. Started `submit_story_event()` in a worker thread, waited for `entered`,
then attempted a snapshot from another thread and a nonblocking lock acquisition.

Observed:

```text
controller read blocked during send: True
playback lock available: False
```

The blocked send was released and all test threads/controller resources were
closed. No live service or GPU was involved.

### Minimal Fix Recommendation

Separate the atomic local mutation from transport delivery. Capture the event,
edit, epoch/session, and delivery intent under the controller lock, then release
that lock before sending. Preserve outbound edit order with a separate
serialization mechanism that does not block tick/Stop/Reset on network I/O.
After delivery, update pending state under the controller lock only if the
captured epoch/session is still current. Apply this to both new and duplicate
story submissions. Coordinate with finding 2; do not fix the lock by introducing
out-of-order plan sends or cross-reset sends.

### Required Regression Tests

- Block `stream.send` and prove a controller snapshot and tick operation remain
  responsive from another thread.
- Exercise Stop/Reset while delivery is blocked; verify no deadlock and no
  delivery result mutates the replacement session.
- Concurrent topic/story submissions retain renderer-compatible edit ordering.

## P1: Failed Story Retry Is Lost After a Later Plan Edit

Primary location:

`src/streammuse/application/rap/bar_stream.py:571`

Related locations:

- `bar_stream.py:569`: retry retrieves the original story edit.
- `bar_stream.py:572`: retry sends that original edit sequence.
- `bar_stream.py:573`: pending tracking is cleared on socket-send success.
- `src/streammuse/application/rap/live_plan.py:37`: renderer rejects an edit
  sequence not greater than its current sequence.
- `src/streammuse/application/rap/lyric_session.py:188`: exact-retry exemption
  applies only when the original edit is already in the renderer's plan.
- `src/streammuse/presentation/rap_lyric_stream_server.py:291`: receives/applies
  update; line 322 catches rejection and emits an asynchronous error, not an
  acknowledgement correlated with the HTTP story submission.

### Cause and Impact

A failed send leaves the story event and its edit in the local plan and marks
the edit pending. A later ordinary topic/flow edit can be delivered with a higher
sequence but without `story_context`. Retrying the story reuses the original,
now-stale sequence. The renderer rejects it because it never received that edit.
The local API nevertheless returns `accepted` and clears pending tracking as
soon as the websocket write succeeds. The local ledger contains the direction
while the renderer still has no story context.

### Reproduced Locally

1. Construct a controller using the existing fixtures, mark it started, and
   advertise `story_context_v1`.
2. Make `stream.send` raise `ConnectionError` before delivering anything.
3. Submit `text="courier loses map", event_id="first"`. This creates local
   story edit sequence 1, then raises the transport error.
4. Restore transport. Deliver `controller.update_plan(from_bar=0, topic="city")`
   to a separate renderer-side `LivePlan`, producing remote sequence 2.
5. Retry `submit_story_event("courier loses map", event_id="first")`.
6. Model renderer behavior by applying the received edit using
   `LivePlan.with_edit(PlanEdit.from_payload(message["edit"]))` and collecting
   its ValueError asynchronously rather than raising it from socket send.

Observed:

```text
retry: duplicate=True, status='accepted', effective_from_bar=0
remote rejection: ['plan edits must have increasing edit_seq']
remote story: None
pending delivery tracking: set()
```

This is different from an exact duplicate already received by the renderer;
the existing exact-retry test covers that successful-delivery case, not this
missing-old-edit case.

### Minimal Fix Recommendation

Do not resend an undelivered stale edit after higher sequences have escaped.
Either prevent later edits from overtaking pending delivery, or on retry create
a fresh edit sequence carrying the complete current immutable story context at
the current editable boundary. Keep the original StoryEvent ID/sequence/content
unchanged and return `duplicate=true`; a new transport edit is not a new story
event. Ensure the pending state for older failed story edits is superseded
consistently. A new acknowledgement protocol is not required for this minimal
ordering fix, but socket success must not be described as renderer acceptance.

### Required Regression Tests

- Failed story #1, delivered topic #2, retry #1: renderer actually receives and
  applies the direction; local event count stays one.
- Repeat with an intervening flow edit and with multiple pending story events.
- Repeated retries after successful recovery do not trigger repeated
  regeneration or duplicate ledger entries.
- Verify original event identity survives any fresh recovery edit sequence.

## P2: Delayed Story Submissions Cross Reset Into the New Session

Primary location:

`src/streammuse/presentation/rap_demo/story_api.py:11`

Related locations:

- `story_api.py:13`: request contains text, event ID, and source, but no expected
  session/epoch.
- `story_api.py:23`: resolves whichever controller is current at handling time.
- `story_api.py:35`: submits without a session precondition.
- `src/streammuse/application/rap/bar_stream.py:657`: Reset advances the epoch.
- `bar_stream.py:667`: Reset discards the old ledger, so old IDs are now novel.

### Cause and Impact

Resolving the current controller prevents retaining a stale controller object,
but does not identify the session intended by the caller. An HTTP retry or a
delayed microphone-to-story submission from the previous session is accepted
into the new session after Reset. The event ID alone cannot fence this because
the new session's idempotency ledger is empty.

### Reproduced Locally Through the Real HTTP Router

Mounted `create_story_router(SimpleNamespace(controller=controller))` in FastAPI
TestClient using the existing controller fixtures. Posted:

```json
{"text":"old session direction","event_id":"old-id"}
```

Called `controller.reset()`, then posted the identical request as a delayed
retry. Observed:

```text
before reset: s-e0 False
late retry after reset: s-e1 False accepted
```

The booleans are `duplicate`: the old event became a new event in the reset
session rather than being rejected as stale.

### Minimal Fix Recommendation

Include an expected `session_id` (or coordinator epoch) in story submissions and
check it inside the same controller lock used to mutate the ledger. Return 409
on mismatch without changing the plan. Have the UI capture this identity when
the user starts the relevant action, including microphone capture/transcription,
and preserve it across retries; do not fetch a fresh identity immediately before
submitting a stale result. If backward compatibility requires an optional field,
the current UI must still supply it; omitted fields remain unfenced.

### Required Regression Tests

- Accept an event in epoch 0, reset, retry with expected epoch 0: 409 and empty
  epoch-1 ledger.
- Delayed microphone submission after Reset is rejected without story mutation.
- A normal event with the current session succeeds, and an identical same-session
  retry remains idempotent.
- Race submission against Reset and verify the check/mutation is atomic.

## Verification and Non-Findings

Last focused review run before this report: **56 passed in 1.72 seconds**:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider \
  tests/unit/application/rap/test_story.py \
  tests/unit/domain/rap/test_story.py \
  tests/unit/presentation/test_story_api.py \
  tests/unit/application/rap/test_characters.py \
  tests/unit/presentation/test_characters_api.py -q
```

Earlier in the review, character cleanup tests failed because terminal status
was published before cleanup and an exception skipped the remaining client.
Another worker fixed those while the review was in progress. Latest inspected
`characters.py:256` closes each client independently and publishes terminal
status afterward. Those superseded failures are NOT outstanding findings.
The earlier cancellation assertion also raced its test handler's cancellation
observer; the final focused run passed. No additional concrete character
cancellation/stale-publication bug was established.

The full 2048-token story prompt limit is not enforced by the 512-byte story
ledger limit alone: topic/history/flow and completion allowance also contribute.
This was reported as a residual input-budget limitation, not a reproduced
normal-demo overflow. Parent subsequently added story prompt priority after a
live probe; this review does not claim that change has been revalidated here.

The three findings above are failure/race paths and do not contradict passing
normal live feature demos or the parent's broader passing regression suite.
No additional tests were run for this requested report handoff.

## Addendum: Independent Verification of the Backend Fixes

Disposition: **all three original findings are resolved in the reviewed backend
changes. No additional P1/P2 was established in this focused re-review.**

Only this report was edited during re-review. No application files were changed,
no live service was restarted, and no GPU/model service was called. The existing
broader commit-message send under the playback lock is pre-existing behavior;
it is not being relabeled as a defect introduced by the story mutex fix.

### Finding 1: Playback Lock During Story Send - Resolved

Current `bar_stream.py:561` uses `_plan_edit_lock` to serialize plan controls.
The controller lock is released before `_deliver_plan_edit()` at line 589;
ordinary plan updates use the same ordering lock at line 595. Story transport
delivery occurs at line 638 outside the controller lock. Tick/Stop/Reset do not
acquire the plan-edit lock. Epoch-aware pending-state cleanup at line 640 avoids
writing completion bookkeeping into a reset session.

Reran the original independent blocked-send reproduction using the existing
controller fixtures and threading events. While story delivery remained blocked,
a separate thread completed `story_snapshot()` and the controller lock was
successfully acquired nonblocking:

```text
LOCK FIX: snapshot responds while story send blocked; playback lock available
```

The focused suite additionally exercised blocked initial/retry sends while
snapshot/tick/Stop/Reset execute, old-session acknowledgement identity, and
concurrent topic delivery ordering. All passed.

### Finding 2: Lost Retry After a Later Plan Edit - Resolved

Current `bar_stream.py:580` detects a pending original edit older than the
current plan. Line 583 creates a fresh transport edit containing the complete
current story context without creating a new StoryEvent. Successful delivery
at lines 640-643 clears superseded pending entries. The plan-edit lock prevents
concurrent topic/flow controls from overtaking a currently blocked story send.

Reran the independent lost-first-send reproduction against a separate
renderer-side LivePlan: fail story sequence 1, deliver topic sequence 2, retry
the original event. The renderer accepted recovery sequence 3 and retained
the original direction. A subsequent duplicate caused no further send:

```text
RETRY FIX: delivered sequences [2, 3] remote story present;
repeated retry sends nothing; local event count 1
```

Focused tests also passed for intervening topic and flow edits, one and two
pending events, immutable original event identity, and repeated recovery retries.

### Finding 3: Stale Submission Across Reset - Backend Guard Resolved

Current `story_api.py:16` accepts `session_id` and lines 36-38 forward it as
`expected_session_id`. `bar_stream.py:563-565` checks that identity under the
same controller lock used to mutate the ledger.

Reran the original HTTP reproduction with the new session field: accept the
event in the old session, Reset, then retry the same payload with the captured
old `session_id`:

```text
SESSION FIX: late retry 409 new ledger empty
```

The real transport implementation in `lyric_stream_client.py:155-162` checks
the expected session and captures the socket together under its transport lock,
then sends only on that captured socket. Focused tests passed for Reset between
the controller check and socket capture, and for socket replacement after
capture; an old story edit was not redirected to the new session's socket.

Compatibility caveat: `session_id` remains optional. Old clients that omit it
still have no stale-request fence; this is not a failure of the guarded path.
Current GUI source inspection shows typed submissions preserving their original
session and `rap-research.js:663` capturing the story session at microphone press,
before permission/capture/ASR yields. Line 637 passes the captured session to story
submission. GUI browser acceptance was not rerun by this backend review and
remains owned by the parent/UI worker.

### Verification Command and Result

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  -p no:cacheprovider \
  tests/unit/application/rap/test_story.py \
  tests/unit/application/rap/test_story_delivery.py \
  tests/unit/domain/rap/test_story.py \
  tests/unit/presentation/test_story_api.py \
  tests/unit/presentation/rap_demo/test_story_session_guard.py \
  tests/unit/infrastructure/rap/test_story_transport.py \
  tests/unit/application/rap/test_characters.py \
  tests/unit/presentation/test_characters_api.py -q
```

Result: **77 passed in 1.86 seconds**. The three independent reproduction
commands also exited successfully with the outputs recorded above. No further
review expansion or background checks remain running.
