# Feature Diagnostic: Six-Slot Turn 13

Status 07:04 PDT: bounded diagnosis complete; no product/source or launch changes.
This is separate from the completed scheduling research report and all its data.

## Finding

**Prompt/novelty-gate mismatch, not an observed six-slot parser defect.**
For `rap-20261022T134417Z-af6e1677`, epoch 3, turn 13, capture bar 94,
seven native presses produced six distinct ticks `[1,2,3,4,5,6]`. The first
delivered line was `Bright paper kites float high`; the next line failed before
ranking with `lyric writer produced no candidates`.

The unchanged direct-writer replay of bar 1 reproduced that exact error:

| Existing policy | Observed result |
|---|---|
| 32 + 16 + 16 + 16 choices | 80 requested; 36 parsed candidates across waves |
| Pre-ranking rejections | 29 repeated opening; 1 repeated final word; 6 cross-wave duplicates |
| Survived to ranking | 0; no API errors in any wave |
| Six-syllable candidates discarded | 8, all for repeated opening |

Examples rejected despite matching six slots: `Bright paper kites fly free`,
`Bright paper kites sail far`, `Bright paper kites touch sky`.

`lyric_writer.py:315` rejects any candidate sharing the previous first three
words. The generator asks for the topic `bright paper kites` but does not state
this three-word exclusion. Rescue waves hide history (`lyric_writer.py:248`)
while the writer still enforces its original history exclusions. This combines
a deterministic rejection rule with strongly concentrated stochastic outputs;
it is not evidence that every six-slot request must fail.

No hardcoded nine-slot example or nine-syllable instruction is in this deployed
path. The reconstructed prompt explicitly says **exactly 6 spoken syllables**,
lists the six ticks, and has no rhyme group. The generic line parser does not
branch on slot count. A fixed 32-token cap is used, but parsed outputs here are
normal complete lines; no evidence identifies truncation as this failure's cause.

## Bounded Prompt Probe

One additional 32-choice request appended the existing forbidden three-word
openings and final words explicitly. Product generator, parser, flow and writer
gates were unchanged. It returned 28 distinct candidates, three valid, selecting
`Bright skies whisper secrets`; the other 25 failed syllable count.

This supports aligning prompt instructions with existing gates, **not** a
release-ready global prompt change. Topic specificity and listening quality were
not validated; the selected probe line is less explicit about paper kites.
Preferred minimal repair to evaluate: carry compact forbidden openings/endings
to every wave, including rescue, independently of omitted prose history.
Do not remove syllable validation or globally disable novelty checks on this evidence.

## Original Trace and Limits

Local GUI summary and event log confirm first-bar delivery and second-bar error.
The isolated renderer log has the turn-13 WebSocket and successful bar-0 fetch;
no bar-1 fetch. Original failed candidate ledgers/prompts were not persisted by
the session exception path (`lyric_session.py:325` forwards only the message).
The replay therefore identifies a reproducible mechanism but cannot recover
the exact original sampled strings or exclude additional original failure causes.

Replay uses the captured flow, delivered first line, reconstructed last-four
history, and first-attempt writer seed `20261807`. API sampling seed is not sent
by the unchanged generator. Deployed writer/generator/prompt hashes match local
files; current launch is 32 initial choices with default 16x3 rescue and zero
minimum score. Existing GPU1/18611 only; shared live load means diagnostic
latencies are not isolated measurements. No renderer pipeline request or GPU
launch/stop was performed.

Raw prompts, waves, rejection ledger, config and hashes:
`logs/scheduling_evidence_20261022/feature-diagnostic/feature-diagnostic-six-slot-turn13/`.
Remote originals:
`/data/home/Andrew.Yang/StreamMUSE/deploy/rap_research_20261022/feature-diagnostic-six-slot-turn13/`.
Standalone diagnostic scripts are in the parent local `feature-diagnostic/` folder.

## Budget-Only Check

Eight direct one-wave requests, two matched writer seeds per flow, alternating
32/64 order; no rescue in this comparison. Seeds are prompt variation seeds,
not API-level reproducibility guarantees. Six-slot input is the turn-13
reconstruction. Four-slot input is the earlier `space` fixture with previous
line `vast black expanse`, ticks `[0,4,8,12]`, neutral stress and no rhyme.

| Flow | Initial budget | Selectable attempts | Parsed candidates, seed 1 / 2 | Rejection mechanism |
|---|---:|---:|---:|---|
| Six-slot turn 13 | 32 | 0/2 | 13 / 14 | All repeated opening or end word |
| Six-slot turn 13 | 64 | 0/2 | 17 / 18 | All repeated opening or end word |
| Four-slot space | 32 | 0/2 | 4 / 4 | Every parsed line had five syllables |
| Four-slot space | 64 | 0/2 | 3 / 3 | Every parsed line had five syllables |

**Do not recommend a budget-only launch bump as the fix for these failures.**
More requested choices did not escape the collapsed/invalid output families in
these eight attempts. This is not a general claim that 64 is ineffective: the
earlier nine-slot study had a different workload and did show higher valid yield.
Two seeds do not estimate a reliable short-flow failure rate or prove all short
flows fail. They do justify documenting a short-flow robustness limitation
instead of applying an untested global prompt or budget change.

There were 13 direct diagnostic API requests total: four baseline rescue waves,
one prompt-only probe, and eight budget comparisons. No server-pipeline request
was made. Raw budget results are `short-slot-budget-probe.json` in the same
separate diagnostic folder. Original turn artifact directory contains bar 0
only; failed bar-1 prompts/ledgers are not recoverable there.
