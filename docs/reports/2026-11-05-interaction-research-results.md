# Real-Time LLM Interactions: Today's Evidence

## Research Framing

Rap is the timed-output testbed. The broader problem is deciding when generated
work remains useful, when it may become public, and how to respond when user input
changes while work is in flight. Today adds three separate interaction surfaces:
directed story, microphone transcription, and two characters with a counting
coordination experiment. Stress alignment and voice consistency are deferred.

The literature review is in `docs/research/2026-11-04-realtime-llm-close-reading.md`.
Serving schedulers, incremental/revisable systems, and asynchronous multi-agent
systems already cover parts of this space. This pilot does not establish novelty.
It supplies implementation and measurements for choosing a narrower question.

## Directed Story: Delivery Is Not Meaning

Each direction has an immutable ID and an ordered version. New generation includes
all retained directions; the newest event becomes the lyric subject. Topic-only
rap keeps its original prompt. An initial run showed that merely appending plot
instructions was insufficient: the model mostly stayed on the old topic. The
story-specific subject change made the selected lines follow the courier, lost
map, stranger, and package. Coherence remains uneven.

Two distinct delays matter:

`L_context = first playback using the new direction version - submission time`

`L_meaning = first playback expressing the requested action - submission time`

The first is measured from software event timestamps and chosen-lyric provenance.
The second needs an explicit semantic criterion; we do not yet have a validated
automated measure. In one illustrative trace, "A stranger offers help" first
reached new-context playback at 3.886 s as "A stranger sees the map alone". The
explicit offer appeared about 11.920 s after submission as "he offers hand, map
safely tucked". That manual reading is an example, not a semantic benchmark.

## Arrival Phase Changes The Deadline

For four beats per bar and tempo `B` in beats/minute:

`T_bar = 4 * 60 / B = 240 / B` seconds.

Let `phi` be the fraction of the current bar elapsed when an edit arrives. Its
remaining time to the next bar is approximately `(1 - phi) * T_bar`. The audio
commit guard occurs before the bar starts, so the usable preparation time is
smaller still. Already committed audio is immutable.

```text
Early edit: | EDIT -------- preparation -------- | new-context bar |
Late edit:  | ----------------- EDIT -- deadline | old-context bar | new context |
            current bar                         next bar          following bar
```

| Run | Played bars | Edit placement | First new-context delay, seconds | New context |
| --- | ---: | --- | --- | --- |
| 60 BPM trace A | 16 | Near bar start | 3.850, 3.886, 3.953 | Next bar, 3/3 |
| 60 BPM recorded demo | 16 | Near bar start | 3.832, 3.828, 3.612 | Next bar, 3/3 |
| 90 BPM | 16 | Near bar start | 2.406, 2.404, 2.187 | Next bar, 3/3 |
| 120 BPM | 16 | Near bar start | 1.916, 1.842, 1.726 | Next bar, 3/3 |
| 90 BPM late edit | 16 | Phase 0.752-0.778 | 3.248, 3.313, 3.339 | Following bar, 3/3 |

Every one of these 80 played bars used `remote_latest` MOSS audio. In the late-edit
run, only 592-661 ms remained until the next nominal bar boundary. The targeted
bars kept their previous context version; the new versions arrived one bar later.
Thus `remote_latest` means the latest available audio record, not necessarily the
latest user intent. Zero vocal fallback does not imply zero stale-content playback.

These are short shared-H200-load pilots, not reliability estimates or an upper BPM
bound. Same broad schedule, but generated lines and runtime load differ. Faster
tempo also shortens the wait to a bar boundary; the table does not show faster
model computation. No external microphone measured sound-output latency.
One speculative future lyric failure occurred beyond the 16 played bars in the
late-edit run; the 16/16 playback claim excludes that unplayed future work.

## Microphone

Browser push-to-talk produces PCM WAV on release. The Mac runs cached
faster-whisper `base.en` on CPU and displays the transcript. Transcript is the
default destination; Story submission is opt-in and automatic after recognition.
The microphone's captured session identity prevents a delayed result crossing
Reset. A browser test deliberately delivered a real transcription after Reset:
the transcript stayed visible, a warning appeared, and the new ledger stayed empty.

The first end-to-end prerecorded-browser test took 1.262 s from release to visible
transcript, including lazy model load. A separate warm backend smoke call took
0.417 s; these are different measurement boundaries, not comparable repetitions.
The final warm browser-to-Story run took 0.493 s from release to visible Story
acceptance, covering upload, local recognition, and routing. It recognized
"Raindrops tap on glassy rooftops at sunrise." The later playing lyric was
"Raindrops tap soft at the waking city"; its arrival is not included in 0.493 s.
Actual physical microphone quality, room echo, recognition across speakers, and
simultaneous speech/playback remain unverified. No voice cloning or full-duplex
speech interaction was implemented.

## Characters And Coordination

One model endpoint serves two actor-specific prompts and histories. Eight text
turns completed in 7.240 s. Persona differentiation is weak and one response was
repeated verbatim; the result validates context plumbing, not character quality.

Counting results: fixed turns completed 3/3 runs in 6.993-8.020 s with 10 calls;
version-checked concurrency completed 3/3 in 5.580-6.670 s with 20 calls. Naive
concurrency failed all three after a duplicate one. The host enforces legal
publication; this is not evidence of unaided model collaboration. Fixed trial
order, cache effects, actor eligibility, and shared load prevent a causal speedup
claim. Details: `2026-11-05-coordination-pilot.md`.

## Most Useful Next Experiment

Vary edit arrival phase while keeping tempo, flow, direction sequence, and serving
configuration fixed. Compare policies for retaining old high-quality audio,
switching to a faster renderer, or delaying the response. Measure separately:

- Deadline misses and stale-context bars.
- Delay until the requested action is actually expressed, using a defined rubric.
- Voice quality/source, canceled or wasted work, and compute/network cost.

This separates a promising scheduling question from engineering milestones.
The next multi-agent experiment should similarly control observation delay and
publication policy, with matched useful-work budgets and randomized trial order.

## Artifacts

- Preserved audio/lyrics: `output/research_day_20261105/demo/`.
- Full traces: `story_prioritized`, `story_demo_60bpm`, `story_90bpm`,
  `story_120bpm`, `story_late_90bpm` under that day's output directory.
- Browser and cancellation evidence: `browser`, `characters_browser`,
  `story_microphone_browser`, and `microphone_reset_browser`.
- User guide: `docs/developer-guide/interaction-research-demo.md`.
- Presentation: `docs/research/presentation-20261105/`.
