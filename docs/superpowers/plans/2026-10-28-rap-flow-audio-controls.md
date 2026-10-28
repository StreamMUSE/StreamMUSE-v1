# MCFlow and audio controls implementation

Approved scope: six curated MCFlow patterns instead of synthetic picker choices;
simple optional chord loops; listening comparisons and alignment-policy selection;
preserve user-entered leading rests; approximate written syllable labels in the grid.

## Compatibility decisions

- Keep legacy template IDs readable for saved scenarios, but hide synthetic choices
  from the live picker. Record provenance without copying source lyrics.
- Chords are hand-authored accompaniment, not annotations from MCFlow. Default off.
- Alignment-policy changes may require a stopped transport; never silently relabel
  already prepared audio as belonging to another policy.
- Strict timing means mapping estimated source syllable onsets to requested slots,
  not claiming independently measured acoustic onset accuracy.
- Stress accents change amplitude, not the timing grid or pronunciation text.
- Syllable hyphenation is display-only and approximate.

## Tasks and verification

- [x] Curate six structurally different MCFlow patterns, including delayed entrances,
  with provenance and fallback coverage. Test catalog and scenario compatibility.
- [x] Add display-only syllable fragments to scheduled and emitted diagnostics;
  preserve full text and pronunciation. Test metadata and browser rendering.
- [x] Reproduce the exchange-mode leading-rest bug and fix its actual cause;
  test positive first onsets through capture, request, and audio preparation.
- [x] Add strict timing / strict timing plus stress beside the existing relaxed
  policy, with truthful diagnostics and controlled listening comparisons.
- [x] Add a small optional chord-loop catalog and exact-length local rendering;
  integrate independent gain and safe mixing across normal/echo/fallback bars.
- [x] Wire settings to the current GUI without removing existing diagnostics,
  transport, device selection, tempo, topic, flow, or exchange features.
- [x] Run focused and regression tests. Check GPU 0/1 availability before any H200
  service launch; test real remote model response with the local client.
- [x] Exercise browser controls and exchange capture, export listening samples,
  update the guide, and report evidence and remaining limitations.

## Execution log

- 2026-10-28: inspected the current dirty feature/rap_optimization worktree.
  Existing changes are preserved. The domain flow model supports positive first
  ticks; the complete capture-to-audio path still needs a regression test.
- Implementation: shared warp now preserves an explicit silent prefix, including
  single-slot captures; final output conversion zeros resampling pre-ringing.
  Chord stems are regenerated against the physical playback bar, not the logical
  response ordinal. Six curated patterns have 8-12 slots and two leading rests.
- Verification in progress: 1,539 broad rap tests (excluding the then-in-progress
  warp changes) passed with loopback enabled; 108 protocol/audio integrations
  passed. The Mac GUI policy/chord selectors were exercised through the browser.
- Final verification: 2,767 tests passed, 5 skipped. Real Mac/H200 acceptance:
  20 continuous MOSS bars and six exchange MOSS responses, no fallback vocals or
  underruns. Browser ASAP topic/flow edits, chord changes, policy controls,
  actual syllable fragments, and Stop/Start were exercised. Two controlled A/B/C
  listening sets exported at 60 and 90 BPM; leading-rest samples are zero.
- Read-only review found two settings acknowledgement bugs; both fixed with
  regressions. Session-start metadata now reflects selected audio settings.
  Guides and delivery/deployment reports are saved. No commit or push requested.
