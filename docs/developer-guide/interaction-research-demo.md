# Interaction Research Demo

The new experiments are independent of the existing rap controls. Audio synthesis,
alignment, and voice selection are unchanged.

## Open

- Research controls: <http://127.0.0.1:8019/research>
- Full rap monitor on the same client: <http://127.0.0.1:8019/>
- Previous demo retained for rollback: <http://127.0.0.1:8018/>

The Mac runs the UI, playback, and CPU speech recognition. H200 runs the existing
Qwen and MOSS models plus an isolated updated renderer on port 18721. Other GPU
processes are left alone; timing measurements are under shared load.

## First Demo: Click-By-Click

Allow about 10 minutes. Test these independently first; they are not a combined
spoken multi-character rap system.

### 1. Prepare Sound And A Clean Session

1. Open Rap Lab at <http://127.0.0.1:8019/>. While stopped, select the output device
   you actually want to hear. The launch configuration uses ZoomAudioDevice, which
   may not play through your headphones or speakers.
2. Keep 60 BPM and Continuous mode for the first test. Press Reset for a fresh
   session. Reset clears story directions; Stop does not.
3. Open <http://127.0.0.1:8019/research>. Do not also start the old 8018 client.

### 2. Direct A Story While Rap Plays

1. In Directed Story, enter `A courier loses the map.` and press Add event.
2. Confirm Event history contains it and Latest accepted reads v1. Press Start.
   Initial preparation can take time before playback begins.
3. Listen for drums and vocals. Check the current lyric changes each bar and
   Playing context eventually reads v1.
4. During playback add `A stranger offers help.`. Latest accepted should become
   v2 immediately. The playing bar must remain unchanged; a later bar should
   show Playing context v2. It may take an extra bar if the replacement is late.
5. After a few bars, add `The stranger wants the package.`. Judge whether the
   lyrics actually express this action, not just mention a stranger or package.
6. Press Stop: playback should finish its current bar. Press Start again without
   Reset: playback should resume and the direction history should remain.

Pass: directions are accepted, frozen playing-context versions advance, and
playback continues. Semantic quality is a separate judgment: a matching version
only proves the direction was in the generation context.

### 3. Test Your Microphone

1. Stop rap first to avoid recording its vocals. Open Microphone and leave
   Destination set to Transcript.
2. Hold Hold to record, grant permission, and say `A red bicycle waits outside
   the library.` Release. If the permission dialog interrupted the first press,
   hold again after allowing access.
3. Check that your words appear under Transcript. Story history must not change.
4. Select Destination Story. Record `The courier escapes on a red bicycle.`.
   Release and look for `Transcript added to Story.`.
5. Return to Directed Story: confirm the event is marked microphone. Press Start
   and check that the new context eventually plays.

Pass: recording stops on release, transcription appears, and only the Story
destination changes the plot. Capture is limited to 15 seconds. This is not voice
cloning: the rap voice will not become your voice.

### 4. Test Two Characters

1. Stop rap. Open Characters, keep Dialogue, and press Start 8 turns.
2. Watch the two named actors produce text turns. Wait for completion.
3. Optionally start another run and press Stop to test cancellation.

Pass: labeled turns arrive and the run completes or stops without getting stuck.
There is no character audio. Persona distinctiveness and conversation quality
are things to evaluate, not guarantees of this prototype.

### 5. Compare Coordination Policies

Open Coordination. Run one policy at a time, waiting for each to finish:

1. Fixed turn: serial requests should produce a legal alternating count 1-10.
2. Naive: overlapping requests may duplicate a number and fail. A visible illegal
   count is an intended experimental result, not necessarily a broken UI.
3. Coordinated: the host checks proposals before publishing them. Look for a
   legal count to 10 plus visible rejected/stale work in events and metrics.

Characters and Coordination share one runner. Finish or stop it before starting
another experiment. They use the same model as rap, so keep rap stopped for a
clean comparison. These are small experiments, not a proven concurrency speedup.

### What To Report Back

- No sound: selected output device, and whether the lyric/bar display advances.
- Story issue: submitted event, Latest accepted, Playing context, and the lyric
  that you think did or did not follow it.
- Microphone issue: browser permission, visible capture state, transcript/error.
- Character/counting issue: policy, run state, and last visible event.

Keep an error visible or take a screenshot before Reset. Do not Reset a running
experiment as the first troubleshooting step.

## Directed Story Details

Start rap playback, then add a short event such as:

1. `A courier loses the map.`
2. `A stranger offers help.`
3. `The stranger wants the package.`

Directions accumulate for the session. The latest direction becomes the lyric
subject; earlier directions remain in the prompt. The current bar stays unchanged.
The existing scheduler attempts the earliest editable future bar, retaining old
audio if a replacement is late. Accepted means the direction was recorded, not
that the model has already expressed it or that it has been heard.
The live readout separates Latest accepted (the current direction ledger) from
Playing context (the version attached to the frozen playing bar). Matching
versions confirm the generation context, not that the lyric fulfilled the plot.

Stop finishes the current bar. Reset in Rap Lab starts a new story. Bounds are
32 events and 512 total UTF-8 bytes of direction text. These are deliberately
small for the deployed model's 2048-token context. There is no automatic story
summary or verified record of plot events already fulfilled.
Browser submissions carry the session identity captured when the action began.
A delayed request from before Reset is rejected, not moved into the new story.

## Microphone

Open Microphone and hold the record button. Allow browser microphone permission
when asked, speak a short English utterance, and release. The transcript appears
automatically. Capture is limited to 15 seconds and microphone tracks stop on
release, cancellation, or leaving the page.
On first use, the permission prompt may interrupt the held button. After granting
permission, hold it again if Capture still reads Idle.

Destination defaults to Transcript, which does not change the rap. Selecting
Story sends a nonempty transcript as a direction automatically. The audio is
transcribed locally with faster-whisper `base.en` on CPU; only the direction text
goes to H200. This is utterance transcription, not full-duplex listening or voice
cloning. Headphones reduce playback leaking into the microphone.
If Reset happens while transcription is pending, the transcript remains visible
with a warning and is not added to the replacement session.

## Characters And Coordination

Characters runs an eight-turn text dialogue between two fictional personas using
one Qwen model with separate actor histories. It does not synthesize their voices
or modify the ongoing rap.

Coordination counts from one to ten with two actors:

- Fixed turns: one request at a time with an assigned speaker.
- Naive: both request from the same current state and publish immediately; a
  duplicate or illegal turn fails the run rather than being silently repaired.
- Coordinated: requests may overlap, but the host checks the state version and
  counting rules before accepting a proposal. Rejected work remains visible.

Both tabs share one bounded experiment runner. Stop cancels local requests;
server-side GPU cancellation is not confirmed. Host-checked success is not proof
that the model independently learned coordination.
For tomorrow's demo, stop rap playback before running these text experiments.
They share the same model endpoint; simultaneous character/rap contention has not
been benchmarked.

## Local Terminal: SSH Forwarding

Use a separate terminal and leave it open if the existing tunnel is absent:

```sh
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 18711:127.0.0.1:18711 \
  -L 18721:127.0.0.1:18721 H200
```

Do not run a second tunnel on occupied local ports. Existing Qwen, MOSS, and the
research renderer must already be running on H200. No H200 microphone service is
needed.

## Local Terminal: Start A Client

Only run this if port 8019 is free:

```sh
cd "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/MBZUAI/StreamMUSE-NewSystem/.worktrees/real_rap"
export PYTHONPATH="$PWD/src"
export PATH="$PWD/.venv/bin:/opt/homebrew/bin:$PATH"
export STREAMMUSE_RESEARCH_MODEL_URL=http://127.0.0.1:18711/v1
.venv/bin/python -m streammuse.presentation.rap_demo.cli \
  --rap-audio-renderer moss_aligned_remote --rap-protocol v3 \
  --rap-audio-transport opus --rap-render-url http://127.0.0.1:18721 \
  --tempo 60 --audio-output composite --audio-device ZoomAudioDevice \
  --max-bars 0 --port 8019 --terminal-layout stream --terminal-detail summary \
  --interaction-mode continuous --response-bars 2 \
  --rap-warp-policy all_onsets_r3 --drum-gain 0.5 --vocal-gain 0.8 \
  --chord-loop minor_cycle --chord-gain 0.2 \
  --log-dir output/research_day_20261105/manual_sessions
```

The launch preserves the existing Zoom output choice. Select a listening device
in Rap Lab while stopped if needed. Do not mistake virtual-device playback for
confirmed audibility through headphones.

For a fresh Mac environment only, install and cache the optional ASR dependency:

```sh
uv pip install --python .venv/bin/python 'faster-whisper>=1.1,<2'
.venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('base.en', device='cpu', compute_type='int8')"
```

The running server loads that cached model lazily; its first utterance can be
slower. Missing models/dependencies produce a visible unavailable error.

## Evidence

Artifacts are under `output/research_day_20261105/`: baseline edits, chosen story
lyrics, browser screenshots, microphone timing, character histories, counting
proposals, accepted messages, and comparison summaries. Session audio is recorded
by the existing composite output in `client/sessions/.../mixed.wav`.

Research and implementation limits are tracked in
`docs/reports/2026-11-05-interaction-progress.md` and the component reports.

For a short presentation, use
`docs/research/presentation-20261105/output/final/research-progress.pptx`
or its PDF preview. A preserved 64-second story demo and its chosen lyrics are
in `output/research_day_20261105/demo/`. These recordings are examples, not a
replacement for the live demo.
