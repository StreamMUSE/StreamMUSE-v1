# Real-Time Rap Scheduler Design

## Goal

Extend the existing beat-aligned rap-text prototype into the active
`streammuse-cli` process. When `--rap-topic` is supplied, the application must
emit aligned lyric syllables on the same wall-clock musical ticks used for MIDI
input, accompaniment generation, console ticks, and the metronome. The mode
must run continuously, scheduling bars ahead of playback rather than making a
blocking text request at a beat boundary.

## User-Facing Contract

```text
uv run streammuse-cli --input-mode keyboard --rap-topic "space travel" \
  --rap-pattern boom_bap --rap-generator phrase_bank --rap-lookahead-bars 2
```

`--rap-topic` is optional; omitting it leaves the current real-time music
application unchanged. The first integration exposes these additional flags:

- `--rap-pattern {boom_bap,straight_8,trap_sparse}` (default: `boom_bap`)
- `--rap-generator {phrase_bank,local_chat}` (default: `phrase_bank`)
- `--rap-lookahead-bars` (default: `2`, minimum: `1`)
- `--rap-candidate-count` (default: `12`, minimum: `1`)
- `--rap-model-url`, `--rap-model`, and `--rap-timeout-s` for `local_chat`

Console emission uses `[RAP B<bar> <beat>.<tick>] <label>*`, where `*` marks
the heuristic stressed syllable. The output is a live visibility mechanism,
not speech synthesis.

## Runtime Architecture

`RealTimeMusicService` gains an optional narrow tick observer. It is started
with the service, invoked directly after `OutputSink.output_tick` at each
timeline boundary, and closed when the service stops. It does not change the
existing `OutputSink` protocol or any of its adapters.

```
RealTimeMusicService._tick_loop
  -> output_tick(tick, bar, beat)
  -> RollingRapController.on_tick(tick)
  -> input buffer / MIDI input / accompaniment playback / inference
```

The controller owns a mapping from absolute bar number to `AlignedLine`.
`start()` synchronously fills the configured initial lookahead using the
offline phrase bank so bar zero is available before tick zero. At every tick it
then:

1. Drains any completed asynchronous local-chat candidate batch.
2. Ensures fallback lines exist through `current_bar + lookahead_bars - 1`.
3. Emits only events whose absolute tick equals the current tick.
4. Starts at most one local-chat request for a future bar when that option is
   enabled.

The local chat call runs in a one-worker executor. `on_tick()` never waits for
it. If the request fails or returns after its target bar starts, the ready
phrase-bank line remains in place. This creates a bounded no-gap behavior that
is suitable for a real-time prototype.

## Planning Rules

The existing `build_bar_slots` and `choose_best_line` remain the timing source
of truth. Each bar is selected from a candidate batch while avoiding reuse
until the available candidates are exhausted. No result with overflow or no
scheduled events is accepted. All bar slots use absolute StreamMUSE ticks, so
rolling lines naturally follow the `Tempo` instance owned by the music service.

For `phrase_bank`, all planning is local and synchronous. For `local_chat`, the
controller still creates fallback bars from the phrase bank, then accepts only
`source == "local_chat"` results for a not-yet-started target bar. The existing
adapter's fallback metadata is not allowed to overwrite the already scheduled
fallback line.

## Lifecycle and Failure Handling

- `start()` is idempotent and validates positive lookahead/candidate counts.
- `close()` shuts down the executor without waiting for a slow HTTP request and
  invokes an optional close callback for the chat client.
- A controller callback is presentation-owned. Exceptions from lyric rendering
  are swallowed by the controller so terminal I/O cannot terminate the musical
  tick loop.
- `RealTimeMusicService.stop()` closes the observer before closing output. If
  the tick loop ends due to `--max-ticks`, service cleanup still calls `stop()`
  from the CLI's `finally` block.

## Scope Boundaries

This increment proves continuous, tick-locked text scheduling. It does not
derive slots from actual keyboard drum hits, revise topic text during a run,
learn rap flow/rhyme, or synthesize a voice. Those are successive layers above
the stable shared-clock and rolling-lookahead mechanism.

## Test Strategy

Focused tests will demonstrate that the real-time service calls a tick observer
after `output_tick`, that the controller emits absolute-tick events while
filling future bars, that a slow primary generator cannot block a tick, and
that late candidate results cannot replace a started bar. Parser and CLI tests
will cover optional defaults and controller assembly. The final verification is
the focused rap/runtime suite followed by the entire repository test suite.
