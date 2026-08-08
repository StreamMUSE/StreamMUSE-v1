# Real-Time Rap Demo Website Design

## Objective

Replace the current rap monitor presentation with a complete browser-based
equivalent of the full terminal dashboard. The site is a read-only live research
instrument: it must show what is performing, what the model saw, what it
returned, why a candidate won or failed, and whether the real-time contract is
healthy.

The existing FastAPI snapshot, session, and WebSocket contracts remain
unchanged. This keeps the UI usable with phrase-bank, scripted-failure, and
local-chat generators without introducing a second runtime path.

## Audience and Job

Primary audience: a researcher demonstrating the prototype to a professor.

Single job: make the current delivery and its complete decision trace legible
within seconds, while preserving enough detail for a technical audit.

## Visual Direction

The site follows the existing StreamMUSE GUI rather than the previous green
terminal aesthetic.

- `canvas` `#f5f6f8`: quiet application background.
- `surface` `#ffffff`: primary work areas.
- `ink` `#20242b`: text and strong rules.
- `brand` `#e91e63`: StreamMUSE identity and active research accents.
- `live` `#17835b`: connected, generated, and accepted states.
- `playhead` `#2468b4`: the currently sounding tick.
- `warning` `#a96308`: pending and degraded states.
- `danger` `#b23b3b`: fallback and error states.

Typography uses the existing GUI's system sans stack for interface text,
Arial Narrow/Roboto Condensed for compact headings, and Monaco/Consolas for
data. Letter spacing remains zero. Panels use square or 4px corners, thin
rules, and no decorative gradients, floating cards, or illustration.

The memorable element is the sixteen-step syllable rail. Each tick shows beat
position, scheduled/emitted syllable, target stress, current playhead, phrase
boundary, and rhyme group without changing size as values update.

## Information Architecture

### Persistent Header

- StreamMUSE wordmark and `Rap Lab` product label.
- Live/reconnecting/stopped connection state.
- Session ID, generator, model, tempo, bar, tick, topic, and template.
- Compact `Follow live` toggle. Disabling it keeps rendering data but stops
  automatic event-console scrolling so a researcher can inspect history.

### Live Delivery Band

- Frozen lyric, current syllable, source, selected score, fallback reason.
- Generation latency, deadline slack, and model state.
- Next reserved bar or planning request so the no-gap strategy is visible.
- Last generation/presentation error, when present.

### Beat-Aligned Flow Workbench

- Stable sixteen-tick rail grouped into four beats.
- Scheduled/emitted labels, target stress, active tick, final boundary, rhyme.
- Exact arrays for ticks, durations, stress, boundaries, and rhyme groups.
- Template name/ID, meter, and provenance.

### Generation Audit

- Request ID, target bar, topic, required syllables, candidate count, seed.
- Recent frozen context lines.
- Exact system/user prompt with roles preserved.
- Raw model response without truncation.
- Response source, count, latency, deadline state, token usage, warning, and
  error fields.

### Candidate Gate and Ranking

- Selected, valid, and rejected candidates ordered consistently.
- Raw text, syllable count versus required count, total score, OOV words, and
  rejection reasons.
- Every component's value, weight, and contribution.
- A selected-score breakdown that remains visible even in a dense candidate
  round.
- Sort controls with accessible button names and visible selected direction.

### Cumulative and Historical Evidence

- Candidate yield, fallback rate, deadline miss rate, generator error rate,
  pronunciation fallback rate, generation latency p50/p95, and jitter p95.
- Frozen-bar history with source, score, and fallback decision.
- Canonical event console with sequence, event type, bar/tick, request, and a
  concise payload detail.

## Responsive Behavior

- At 1280px and wider, the delivery/flow workbench occupies the left two thirds
  and generation audit occupies the right third. Candidate ranking spans the
  page below.
- Between 760px and 1279px, workbench and audit stack while metric and history
  grids remain two-column where possible.
- Below 760px, all content is one column. The fixed-format flow rail scrolls
  horizontally inside its own band; no text overlaps or changes the rail's
  dimensions. Candidate records become stacked rows rather than forcing the
  full desktop table into the viewport.

## Data and Error Handling

- Initial state uses explicit waiting copy and never invents values.
- WebSocket snapshots remain authoritative; event refreshes stay de-duplicated
  by sequence.
- Missing optional payload fields render as `--` or `none` according to meaning.
- All model/session content is assigned with `textContent`; no `innerHTML`.
- Reconnect remains automatic. The header distinguishes connecting, live,
  offline/retrying, and stopped states.
- The UI never sends model or runtime control messages.

## Testing and Acceptance

- Server contract tests assert every major terminal-equivalent section and the
  safe text-rendering contract.
- JavaScript contract tests assert exact flow arrays, raw response, selected
  score contributions, generator errors, and live-follow behavior are rendered.
- Full Python suite must pass.
- Browser QA runs against a live phrase-bank session at desktop and mobile
  viewports. It verifies no horizontal page overflow, nonblank flow content,
  live tick movement, candidate sorting, follow-live toggling, reconnect state,
  and absence of console errors.
- The finished local server remains running and its URL is provided to the user.

## Scope Boundaries

This task does not add runtime controls, live drum input, speech synthesis,
model configuration forms, authentication, or persistence beyond existing
session artifacts. It is a complete observer for the symbolic prototype.
