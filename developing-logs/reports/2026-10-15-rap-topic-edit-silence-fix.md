# Topic Edit Silence and 50 BPM Retest

Date: 2026-10-15

Branch: `feature/rap_optimization` (uncommitted work).

## Observed Failure

Mac session `rap-20261014T234322Z-dd848ef7` began playback on October 15 at
90 BPM. Its first four bars used MOSS, after which remote audio increasingly
missed deadlines. A topic edit to `rainbows` at 16:23:35 UTC was followed by
the first missing-audio commit at 16:23:39 UTC. Playback completed 13 bars,
then underrun, while the clock continued advancing and logging missing audio.

The original log remains at:

```text
logs/rap_gui_v3_20261014/rap-20261014T234322Z-dd848ef7/events.jsonl
logs/rap_gui_v3_20261014/client-ready.log
```

## Root Cause and Fix

The v3 controller discarded each uncommitted local fallback when a topic or
flow changed. Its replacement used the startup fallback catalogue, whose
lookup requires an exact topic/template pair. A free-form topic such as
`rainbows` has no catalogue entry. The exception happened before the renderer's
existing exception handler and was silently retained in an unobserved executor
Future. When remote audio and streamed lyrics were late, no safety audio remained.

Changes in `application/rap/bar_stream.py`:

- Keep already-prepared local audio until a replacement is completely ready.
- On an unknown topic/template lookup, use a startup-validated scenario line.
  Re-align it to the requested flow when its syllable count matches; otherwise
  retain its validated original flow. This is an explicitly off-topic safety
  fallback, not successful application of the requested topic.
- Publish a recoverable fallback-catalogue diagnostic, and report other local
  preparation exceptions instead of silently losing them in worker Futures.
- Keep the actual local plan with its prepared audio so monitoring reports
  what played rather than relabeling old audio as the requested topic/flow.
- Discard local render results that finish after another edit, reset, or commit.

Two regression tests failed before the patch and passed afterward: unknown
live topics must continue beyond the pre-rendered window, and an edit must not
delete playable audio while replacement rendering is pending.

Full project suite: **2,189 passed, 5 skipped**, 7 warnings, 95.18 seconds.

## Independent Connection Problem

The original session's complete HTTP-fetch latency rose from approximately
4.2 seconds to 18.3 seconds. MOSS itself remained about 2 seconds, alignment
about 15 ms, and Mac preparation about 70-180 ms. The `transfer` metric covers
the entire HTTP fetch, including any server-side wait; it is not a measurement
of network transport alone.

During the retest, a basic forwarded `/health` request received HTTP 200 headers
but then timed out without its body. A fresh SSH connection also timed out
during banner exchange. This independently establishes a connection/server
responsiveness problem; current evidence cannot distinguish the campus network,
VPN, and SSH host as its cause. No H200 service was restarted or modified.

## 50 BPM Status

The Mac client was changed to **50 BPM**, giving 4.8 seconds per bar. The same
v3 PCM transport, H200 GPU1 services, and Mac speakers were retained.

The attempted live session `rap-20261015T163019Z-70aa128a` could not establish
its WebSocket handshake while SSH was stalling. It was cancelled without
playback. **There is no successful 50 BPM live acceptance result yet.** The
code regressions are verified; end-to-end verification needs restored H200
connectivity. The GUI also remained visually in priming after the handshake
failure until restart, an existing startup-error reporting limitation.

Retest artifacts and current client log:

```text
logs/rap_gui_v3_50bpm_20261015/
```

Next live check: start at 50 BPM, apply a new topic and then a flow edit, run
past the initial three-bar window, and count actual completed bars, MOSS/eSpeak
choices, and underruns. Do not infer success from `/health` alone.

## October 21 Upstream Reconciliation

Pulled through `77787320`, including the independently developed controller
recovery in `0cde79df`. Retained upstream's syllable-compatible `line_for_any`
lookup, exact-length silent-bar recovery, and exclusion of empty lines from
rhyme history. Combined these with prepared-fallback retention, stale-render
guards, and GUI monitoring of the actual played revision.

Fallback catalogue results now carry their source topic so a different-topic
safety line is not relabeled as a successful user edit. The GUI also explicitly
reports silent recovery bars and does not treat them as applied edits. Both
integration cases have regression tests. The earlier live-test limitation
above remains; this reconciliation does not establish a new H200 live result.

Reconciled verification: **2,194 passed, 5 skipped**, 7 warnings, 56.76 seconds
with `.venv/bin/python -m pytest tests/ -q --tb=short`; `git diff --check` clean.
