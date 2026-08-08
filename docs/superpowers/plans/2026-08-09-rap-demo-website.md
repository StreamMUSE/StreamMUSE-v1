# Real-Time Rap Demo Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a responsive StreamMUSE-styled website with full terminal-dashboard feature parity over the existing rap monitor API.

**Architecture:** Keep the FastAPI server, projector, REST endpoints, and WebSocket protocol unchanged. Replace the three static assets as a cohesive presentation layer, using semantic HTML, CSS grid, and dependency-free JavaScript render functions that consume the existing snapshot.

**Tech Stack:** FastAPI static serving, semantic HTML5, CSS Grid/Flexbox, vanilla JavaScript, pytest, in-app browser Playwright.

## Global Constraints

- Preserve `/api/state`, `/api/session`, and `/ws` contracts.
- Render untrusted runtime/model content only through `textContent`.
- Preserve all full-detail terminal evidence; do not truncate prompts or raw responses.
- Match the existing StreamMUSE white/magenta interface language.
- Support desktop, tablet, and 320px mobile without page-level horizontal overflow.
- Do not add frontend dependencies or a build step.

---

### Task 1: Terminal-Parity Static Contract

**Files:**
- Modify: `tests/unit/presentation/rap_demo/test_server.py`
- Modify: `src/streammuse/presentation/rap_demo/static/index.html`

**Interfaces:**
- Consumes: unchanged static routes in `create_app()`.
- Produces: stable IDs for delivery, flow, queue, health, prompt, raw response,
  score detail, candidate ranking, metrics, history, and event trace renderers.

- [ ] Add failing static-route assertions for every terminal-equivalent region,
  `follow-live`, `raw-response`, `selected-score`, `last-error`, and safe text use.
- [ ] Run the focused server test and confirm the new assertions fail.
- [ ] Replace the HTML with the semantic control-room structure.
- [ ] Run the focused test and confirm it passes.

### Task 2: StreamMUSE Visual System and Responsive Layout

**Files:**
- Modify: `src/streammuse/presentation/rap_demo/static/css/rap-demo.css`
- Test: `tests/unit/presentation/rap_demo/test_server.py`

**Interfaces:**
- Consumes: Task 1 IDs/classes.
- Produces: stable desktop/mobile dimensions and state selectors used by JS.

- [ ] Add failing assertions for brand tokens, responsive breakpoints, flow rail,
  mobile candidate layout, focus states, and reduced-motion handling.
- [ ] Run the focused test and confirm failure.
- [ ] Implement the white/magenta research-workbench layout and responsive rules.
- [ ] Run the focused test and confirm it passes.

### Task 3: Full Live Renderer

**Files:**
- Modify: `src/streammuse/presentation/rap_demo/static/js/rap-demo.js`
- Test: `tests/unit/presentation/rap_demo/test_server.py`

**Interfaces:**
- Consumes: JSON-safe projector snapshots and canonical events.
- Produces: live DOM rendering and candidate/follow interactions.

- [ ] Add failing script assertions for queue, error, exact arrays, history
  context, raw response, response status, value/weight/contribution scores,
  generator-error metrics, current syllable, sort direction, and follow state.
- [ ] Run the focused test and confirm failure.
- [ ] Implement small pure formatting helpers and focused render functions.
- [ ] Wire snapshot, WebSocket, sorting, follow toggle, and connection states.
- [ ] Run focused server and rap-demo tests.

### Task 4: Live Browser Acceptance

**Files:**
- Modify only if browser QA exposes a defect in the three static assets.

**Interfaces:**
- Consumes: live phrase-bank `streammuse-rap-demo` on localhost.
- Produces: visual and interaction acceptance evidence.

- [ ] Start a continuous local phrase-bank session with web monitoring.
- [ ] Verify desktop layout, tick updates, sorting, follow toggle, and no console
  errors at 1440x1000.
- [ ] Verify one-column layout, contained flow scrolling, readable candidates,
  and no page overflow at 390x844.
- [ ] Iterate on defects and repeat both viewport checks.
- [ ] Run the complete Python test suite and `git diff --check`.
- [ ] Review the final diff, commit, push `feature/real_rap`, and leave the local
  demo server running for handoff.
