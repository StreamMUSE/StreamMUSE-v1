# Voice Input Migration Implementation Report

Date: 2026-07-17

Target branch: `feature/voice`

## Outcome

The interactive Zip-Zap-Zop voice-input implementation from the damaged
`StreamMUSE-new-sys` working tree has been migrated onto the current repository.
The committed source and target trees shared the same base commit/tree, so the
original working-tree implementation could be restored without translating it
across divergent code.

The migrated feature remains STT-only and opt-in on `streammuse-task play`.
Terminal input remains the default. The music `InputSource`, realtime music
service, MIDI/audio output, and web viewer were not changed.

## Recovery and migration

- Recovered the four OneDrive `compressed,dataless` files from the successful
  chronological Codex patch history: the voice package initializer, persistent
  faster-whisper adapter, qualification tool, and original implementation report.
- Verified their final byte counts against the OneDrive file metadata.
- Compared the other 36 feature files byte-for-byte with the readable source
  working tree before adding migration-specific hardening.
- Regenerated `uv.lock` from `pyproject.toml`; its SHA-256 is
  `08155ed0a91102efd9506fb6031c42f2772c417f53581bd9f333a490f326c2e4`,
  identical to the recovered source lock.
- Rebuilt tracked editable-package metadata after adding the new package files.

## Additional hardening

The post-report hardware logs contained one pathological tiny.en result in which
`Zop` became 112 repetitions of `Zap` with a segment compression ratio around
29.421. The migrated recognizer now applies a bounded, auditable transcript
quality gate:

- reject output longer than 512 characters;
- reject a run of the same token repeated at least 16 times;
- reject any segment compression ratio greater than 10.
- reject non-finite or otherwise invalid segment compression-ratio metadata.

Rejected model output is not silently rewritten or passed to the task parser.
It receives `rejected_transcript` status, remains available as the raw transcript,
and records reasons, measurements, and thresholds in ASR diagnostics. Offline
qualification records the rejected sample, forces its canonical prediction to
null, and summarizes total/positive/negative rejection rates plus a reason
histogram.

## Verification

- Voice-focused unit/integration suite: `232 passed, 2 skipped`.
- Transcript quality-gate focused tests: `28 passed`.
- Full unit suite: `600 passed, 14 failed`; all 14 failures are the pre-existing
  melody-robustness/perturbation-campaign failures recorded by the source report.
- Full integration suite: `4 passed, 1 failed, 2 skipped`; the failure is the
  pre-existing Lekai session-contract failure recorded by the source report.
- Consistency suite: `2 skipped`.
- Fixed offline `Systran/faster-whisper-tiny.en` snapshot
  `0d3d19a32d3338f10357c0889762bd8d64bbdeba`: model resolution, load, warm-up,
  and transcription smoke passed with `local_files_only=True`.
- Host microphone enumeration found the iPhone, MacBook Pro, Teams, and Zoom
  inputs. A no-persistence MacBook microphone preflight selected 16 kHz and
  successfully opened and immediately closed the PortAudio stream; it did not
  capture a gameplay utterance.
- `uv lock --check`, Python compilation, terminal lazy-import checks, CLI help,
  and `git diff --check` passed.

## Qualification status

Implementation and automated migration verification are complete. Formal product
qualification is still external work because this repository does not contain a
consented held-out speech corpus and this run did not record a human utterance.
The feature must not be described as fully qualified until the documented
multi-speaker accuracy, false-command, onset, overflow, 20-turn microphone, and
latency gates pass on the target hardware.
