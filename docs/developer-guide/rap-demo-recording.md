# Recording The Rap Demo

## Prepared Instance: 2026-10-31

- Open `http://127.0.0.1:8018/` on the Mac. Playback is stopped and ready.
- Settings: 75 BPM, alternating mode, two response bars, strict syllable timing,
  Am-F-C-G chords, drum/vocal/chord gains 0.5/0.8/0.2.
- Output is MacBook Pro Speakers. Change output only while stopped.
- H200: Qwen and the render/alignment service on GPU0; optimized
  MOSS-Local/SGLang on GPU1 (current placement as of 2026-10-31).
- The localhost-only SSH forward maps Mac 18720 to H200 18720. It is already
  running; do not launch a duplicate.
- Mac logs, exact startup command, launch record, and smoke-test evidence:
  `output/rap_demo_20261031/` in the `real_rap` worktree.
- The exchange smoke test delivered two real MOSS bars, one waiting replay,
  zero underruns, and no samples at the limiter. This tests the application,
  not Zoom capture or acoustic quality.

## Zoom Recording

Zoom is a convenient way to capture the interface and shared computer audio.
It is not the reference-quality research audio; retain the demo's own WAV too.

1. Start a Zoom meeting as host. Share the browser window containing the demo.
2. Enable **Share Sound**, and select **Stereo (high fidelity)**. Start sharing.
3. Set Zoom's speaker to the physical speakers/headphones used for monitoring.
4. In the stopped rap UI, select **ZoomAudioDevice** when routing the Python
   output into Zoom. This app selects its output device explicitly, so do not
   assume Zoom captures a separately selected physical output. Select the
   virtual device only after Zoom audio sharing is active.
5. Use **Record -> Record on this Computer**. Record a 20-30 second test with
   actual generated vocals, then replay it to confirm both video and audio.
6. For the final take, Start the demo, arm capture, tap a full rhythm bar, and
   let the model respond. Mute the microphone when not narrating to avoid
   recording the speakers a second time. Shared computer sound is separate
   from microphone capture; microphone "Original sound" is not its substitute.
7. Stop the demo after the take, stop Zoom recording, and allow Zoom to finish
   conversion. Keep Zoom's recording folder on a local drive, not this
   OneDrive-synced repository.

Zoom has not been started or configured by this preparation. The user will
switch the output and perform the recording check.

References: [Zoom computer-audio sharing](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0063608),
[Zoom local recording](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0076922),
[Berklee's explicit audio-device routing workflow](https://online.berklee.edu/help/en_US/zoom/how-to-share-audio-in-zoom).

## Preserve The WAV

After Stop, copy the current WAV **before Reset or changing interaction mode**;
those operations can reset the current recorder. The recording path is shown
under Audio playback evidence. This command saves a timestamped copy:

```bash
cd "/Users/andrewyang/Library/CloudStorage/OneDrive-UCSanDiego Real/Stuff/UGRIP/StreamMUSE-NewSystem/.worktrees/real_rap"
.venv/bin/python - <<'PY'
from datetime import datetime
import json
from pathlib import Path
import shutil
import urllib.request

state = json.load(urllib.request.urlopen("http://127.0.0.1:8018/api/state"))
assert state["audio"]["state"] == "stopped", "Stop playback first"
source = Path(state["audio"]["recording_path"])
assert source.stat().st_size > 44, "No completed audio recorded yet"
destination = Path("output/rap_demo_20261031/takes")
destination.mkdir(parents=True, exist_ok=True)
target = destination / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_rap_demo_mix.wav")
shutil.copy2(source, target)
print(target.resolve())
PY
```

This backup contains completed playback-callback PCM, not microphone capture,
Zoom encoding, or any artifacts introduced later by Bluetooth/output hardware.
