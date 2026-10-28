# Separate Feature Diagnostic: Four Neutral Slots

2026-10-22 05:55 PDT. This is feature debugging, excluded from the primary
scheduling research figures, registry measurements, and deadline analyses.
Only research GPU0/18601 was used; GPU1/GPU2 remained reserved for live tests.

One two-line request: topic `space`, ticks `[0,4,8,12]`, duration four ticks
per slot, stress 0.5, boundary strength zero, no rhyme labels. Unmodified
`SequentialLyricWriter` and `IndependentChoiceCandidateGenerator`, default
32 initial + three 16-choice rescue waves, minimum valid three, couplet policy
(no rhyme target because the template has no rhyme group). No API seed was
added; writer seed is recorded but the unchanged generator does not send it as
an API seed. Bar1 includes the selected bar0 in history and `previous_line`.

| Bar | Selected text | Requests | Valid | Writer wall ms |
|---|---|---:|---:|---:|
| 0 | vast black expanse | 80 | 2 | 3385 |
| 1 | vast unknown space | 80 | 2 | 3026 |

Both succeeded after exhausting all four waves. The desired minimum of three
valid candidates is a rescue stopping criterion, not a mandatory success gate.
This single pair did not reproduce an all-attempt failure or the live timeout.

Bar1 wave0 returned only three unique parsed candidates from 32 API choices:
`endless silent void`, `endless cosmic dance`, `endless void above`.
All were rejected as `syllable_count:5!=4`. Across all bar1 waves:

- 30 syllable-count rejections: 12 with five, nine with six, seven with seven,
  and two with eight syllables instead of four.
- Six `repeats_recent_end_word`, eight `duplicate_normalized_text`, and two
  `non_lyric_characters` rejections; two valid/selected candidates.
- Generator uniqueness warnings: 32 requested/3 unique in wave0, then 16/14,
  16/16, 16/15. These are deduplication counts, not API transport failures.
- No `generation_error`, rhyme-label rejection, or rhyme-target hard gate.

Mechanism supported by the ledger: four-syllable yield and independent-choice
diversity are low. History further narrows the second line. Rescue waves drop
prompt history, but the writer still rejects prior end words, so those six
rejects concern an acceptance constraint not explicitly communicated in the
rescue prompt. The no-rhyme first-wave prompt also does not explicitly forbid
the prior final word. This is a concrete prompt/acceptance mismatch to review,
not proof of a unique root cause or a reason to weaken the gates.

Parent owns any prompt/generator changes. No product source was edited here.
Raw script, exact policy/flow/source hashes, every prompt/raw response and
candidate ledger: `logs/scheduling_evidence_20261022/feature-diagnostic/`.
Reproduce from `real_rap` with `PYTHONPATH=src .venv/bin/python
logs/scheduling_evidence_20261022/feature-diagnostic/four-slot-writer.py` after
choosing a new output folder; the current script writes beside itself, so do
not overwrite this diagnostic's evidence.
