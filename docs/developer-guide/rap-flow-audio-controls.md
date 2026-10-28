# Flow and audio controls

This update targets the existing **Mac client, protocol v3, H200 MOSS renderer**.
The server still generates/selects lyrics and returns source speech plus
alignment. The Mac applies timing/stress policy, mixes accompaniment, and plays
the scheduled bar. No extra model request is needed for chords or policy changes.

## Flow patterns

The live picker now exposes six curated MCFlow-extracted bars, not the previous
synthetic patterns. Names identify the source song and measure; the catalog
retains source hashes and extraction provenance, not source lyrics.
Old template IDs remain readable for saved scenarios but are hidden in the picker.

Two curated patterns have a leading rest. In exchange mode, the first tap also
need not occur at tick zero: the capture window remains the complete bar, and
empty ticks before the first tap remain empty vocal time. The arming key is not
recorded as a tap.

## Vocal alignment

In **Live audio**, select **Vocal alignment**, including during playback:

| Choice | Behavior | Trade-off |
| --- | --- | --- |
| Relaxed timing | Sparse anchors with bounded local stretching where feasible | More natural word motion; some internal syllables can drift |
| Strict syllable timing | Every estimated syllable onset maps to its requested flow slot | Better timing agreement; extreme stretches can sound unnatural |
| Strict timing + stress | Strict mapping plus smoothly varying loudness accents from template stress | Stronger audible accents; not a learned expressive rap voice |

All policies preserve the leading vocal rest. A policy change reuses cached
source speech and rerenders the earliest uncommitted bars locally. Already
committed/playing bars remain unchanged; no new model request or session reset
is required. This selector is available for the v3 MOSS client; it does not silently
change the eSpeak fallback into a different voice or alignment algorithm.

The diagnostic onset positions are estimates derived from source forced
alignment and the applied time map. They are **not** independent measurements of
what a listener hears. Strict placement cannot correct an erroneous source
alignment or guarantee perfect word intelligibility.

## Chords

Select **Chord loop** and adjust **Chords**. Local controls apply automatically.
Choices are Off (default),
Am-F-C-G, and C-G-Am-F, with one chord per bar. These are simple hand-authored
loops, **not MCFlow harmony annotations**. They run locally and continue through
exchange waiting/echo bars. Chord changes take effect at an eligible bar boundary;
the current committed bar is not rewritten. Drums and vocals retain independent
gains. New loop definitions can be added to the small chord catalog later.

## Independent settings groups

| Group | Controls | Activation |
| --- | --- | --- |
| Live audio | Drum, vocal, chord gain | Smoothed live adjustment on the Mac; no H200 request |
| Live audio | Alignment policy | Earliest uncommitted bar after local rendering |
| Live audio | Chord loop | Eligible bar boundary |
| Generation | Tempo, topic, flow | Server-coordinated, with separate acceptance and playback evidence |
| Session setup | Mode, response length, output device | While stopped |

Waiting for a generation update does not disable the local controls. Changing
volume deliberately affects the current sound; changing alignment or chord
structure never rewrites the current bar. Local settings still travel from the
browser to the Python client on the Mac, not to H200.

Each control shows **Immediate**, **Next eligible bar**, **Server-coordinated**,
or **Stop to change**. Disabled controls explain the current restriction, such
as waiting for playback, stopping, or an unsupported backend. Independent group
statuses distinguish sending, server acknowledgement, scheduled playback, and
observed application. **Applied in bar N** remains confirmed after Stop or a
later drums-only bar; it does not claim audio is still playing.

API clients can use `POST /api/control/settings/local`, `/generation`, or
`/session` with flat field patches. Gains/chord mix can be combined; alignment
is a separate local operation. Device selection is separate from mode/response
changes. `GET /api/control/settings` exposes the groups, activation rules, and
independent acknowledgements. The original endpoint remains available for
existing scripts, including supported stopped bulk updates.

## Syllable grid

The grid now shows approximate written fragments such as `whis-` and `-per`,
while retaining the complete lyric above it. These fragments are display labels:
they do not alter pronunciation, phonemes, scheduling, or candidate scoring.

## Controlled listening comparison

From the worktree with the existing Python environment and SSH forward running:

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/rap_alignment_comparison.py \
  --url http://127.0.0.1:18720 \
  --template mcflow_4fa0466611a111afb52f \
  --name night_train --bars 2 --bpm 60 \
  --output output/rap_alignment_20261028/night_train
```

This generates new lyrics and MOSS speech once, then renders A/B/C from identical
source audio, words, onsets, drums, and gain settings. Chords are off to avoid
masking vocal differences. WAV names include the song name and policy; the JSON
report records timing estimates, render time, and the leading-rest signal peak.
