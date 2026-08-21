"""Browser-state reducer regressions for the realtime rap monitor."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REDUCER = ROOT / "src/streammuse/presentation/rap_demo/static/js/rap-demo-state.js"
MAIN_SCRIPT = ROOT / "src/streammuse/presentation/rap_demo/static/js/rap-demo.js"


def test_browser_event_cache_rebases_on_session_reset() -> None:
    program = r"""
const fs = require("fs");
const vm = require("vm");
global.window = global;
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), { filename: process.argv[1] });
const reduce = global.StreamMuseRapState.reduceEventCache;
let events = [
  { sequence: 1, event_type: "bar_frozen", bar: 0 },
  { sequence: 2, event_type: "tick", bar: 0, tick: 15 },
];
events = reduce(events, { sequence: 3, event_type: "session_reset" }, 240);
events = reduce(events, { sequence: 4, event_type: "tick", bar: 0, tick: 0 }, 240);
events = reduce(events, { sequence: 4, event_type: "tick", bar: 0, tick: 0 }, 240);
process.stdout.write(JSON.stringify(events));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(REDUCER)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == [
        {"sequence": 3, "event_type": "session_reset"},
        {"sequence": 4, "event_type": "tick", "bar": 0, "tick": 0},
    ]


def test_browser_chunk_projection_orders_by_session_epoch_and_strict_sequence() -> None:
    program = r"""
const fs = require("fs");
const vm = require("vm");
global.window = global;
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), { filename: process.argv[1] });
const api = global.StreamMuseRapState;
const event = (session, sequence, epoch, state) => ({
  session_id: session,
  sequence,
  event_type: "chunk_committed",
  request_id: `${session}-${state}`,
  payload: { coordinator_epoch: epoch, state },
});
let projected = api.projectRemoteChunk(null, event("session-a", 100, 2, "old"));
projected = api.projectRemoteChunk(projected, event("session-a", 100, 2, "equal-rejected"));
projected = api.projectRemoteChunk(projected, event("session-a", 99, 2, "older-rejected"));
const afterOlder = projected;
projected = api.projectRemoteChunk(projected, event("session-a", 1, 3, "new-epoch"));
const afterEpoch = projected;
projected = api.projectRemoteChunk(projected, event("session-a", 200, 2, "old-epoch-rejected"));
const afterOldEpoch = projected;
projected = api.projectRemoteChunk(projected, event("session-b", 1, 1, "new-session"));
const afterSession = projected;
const afterSessionStart = api.projectRemoteChunk(projected, {
  session_id: "session-c", sequence: 1, event_type: "session_started", payload: {},
});
const fractionalEvents = api.reduceEventCache([], event("session-a", 1.5, 2, "fractional"), 240);
process.stdout.write(JSON.stringify({
  afterOlder, afterEpoch, afterOldEpoch, afterSession, afterSessionStart, fractionalEvents,
}));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(REDUCER)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["afterOlder"]["state"] == "old"
    assert result["afterEpoch"]["state"] == "new-epoch"
    assert result["afterOldEpoch"]["state"] == "new-epoch"
    assert result["afterSession"]["session_id"] == "session-b"
    assert result["afterSession"]["state"] == "new-session"
    assert result["afterSessionStart"] is None
    assert result["fractionalEvents"][0]["sequence"] is None


def test_browser_authoritative_snapshots_clear_and_replace_remote_projection() -> None:
    program = r"""
const fs = require("fs");
const vm = require("vm");
global.window = global;
global.document = {
  getElementById() { return null; },
  addEventListener() {},
  readyState: "loading",
};
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), { filename: process.argv[1] });
const api = global.StreamMuseRapState;
api.reduceEventCache([], {
  session_id: "session-a", sequence: 9, event_type: "chunk_committed", request_id: "old",
  payload: { coordinator_epoch: 1, state: "old" },
}, 240);
const before = api.currentRemoteChunk();
const cleared = api.applyRemoteSnapshot({
  session_id: "session-a", last_sequence: 10, coordinator_epoch: 2, remote_chunk: null,
});
const staleEvents = api.reduceEventCache([], {
  session_id: "session-a", sequence: 9, event_type: "chunk_committed", request_id: "stale",
  payload: { coordinator_epoch: 1, state: "stale" },
}, 240);
const afterStale = api.currentRemoteChunk();
const replaced = api.applyRemoteSnapshot({
  session_id: "session-b",
  last_sequence: 1,
  coordinator_epoch: 1,
  remote_chunk: {
    session_id: "session-b", sequence: 1, coordinator_epoch: 1,
    event_type: "chunk_committed", request_id: "new", state: "snapshot",
  },
});
process.stdout.write(JSON.stringify({ before, cleared, staleEvents, afterStale, replaced }));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(REDUCER)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["before"]["request_id"] == "old"
    assert result["cleared"] is None
    assert result["afterStale"] is None
    assert result["replaced"]["session_id"] == "session-b"
    assert result["replaced"]["request_id"] == "new"


def test_websocket_snapshot_path_updates_remote_panel_directly() -> None:
    source = MAIN_SCRIPT.read_text(encoding="utf-8")

    snapshot_branch = source[source.index('message.type === "snapshot"') : source.index('message.type === "event"')]
    assert "StreamMuseRapState.applyRemoteSnapshot(message.payload)" in snapshot_branch


def test_browser_chunk_projection_and_event_cache_drop_unbounded_artifact_bodies() -> None:
    program = r"""
const fs = require("fs");
const vm = require("vm");
global.window = global;
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), { filename: process.argv[1] });
const api = global.StreamMuseRapState;
const event = {
  sequence: 12,
  event_type: "chunk_committed",
  bar: 4,
  request_id: "request-2",
  payload: {
    state: "committed",
    renderer_decision: "moss_aligned_remote",
    chunk_index: 2,
    bars: [4, 5],
    selected_lines: ["First remote line", "Second remote line"],
    flows: [
      { template_id: "flow-a", slot_stress_schedule: "t0@1.00, t2@0.25", slots: [] },
      { template_id: "flow-b", slot_stress_schedule: "t1@0.80", slots: [] },
    ],
    candidate_counts: { requested: 32, parseable: 30, valid: 8, selectable: 4 },
    selected_scores: [
      { bar: 4, total: 0.91, component_scores: { stress_alignment: 0.88 } },
      { bar: 5, total: 0.87, component_scores: { continuity: 0.82 } },
    ],
    prompt_summary: "system: clean rap | user: both exact schedules",
    context_lines: ["Prior committed line"],
    stage_timings_ms: {
      generation: 1000, evaluation: 80, moss: 2000, aligner: 40, r3: 120,
      package: 10, transfer: 20, mac: 6, total: 3276,
    },
    request_budget_ms: 5000,
    elapsed_ms: 3300,
    deadline_slack_ms: 1700,
    alignment: {
      method: "mms_forced_alignment",
      confidence: 0.94,
      fallback_counts: { transcript_proportional: 1 },
      character_spans: [{ start: 0, end: 4 }],
    },
    stretch_warnings: ["stretch_ratio_high:1.22"],
    warnings: ["alignment_fallback:one_word"],
    hashes: { vocal_sha256: "vocal-hash" },
    artifact_refs: { manifest: "/h200/request-2/manifest.json" },
    transfer_bytes: 262144,
    failure_reason: null,
    raw_wav: "RIFF-forbidden",
    candidate_ledger: [{ text: "forbidden full ledger" }],
    character_spans: [{ start: 0, end: 4 }],
  },
};
const projected = api.projectRemoteChunk(null, event);
const cached = api.reduceEventCache([], event, 240);
process.stdout.write(JSON.stringify({ projected, cached }));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(REDUCER)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    projected = result["projected"]
    assert projected["event_type"] == "chunk_committed"
    assert projected["request_id"] == "request-2"
    assert projected["selected_lines"] == ["First remote line", "Second remote line"]
    assert [flow["template_id"] for flow in projected["flows"]] == ["flow-a", "flow-b"]
    assert projected["candidate_counts"] == {
        "requested": 32,
        "parseable": 30,
        "valid": 8,
        "selectable": 4,
    }
    assert projected["stage_timings_ms"]["mac"] == 6
    assert projected["alignment"] == {
        "method": "mms_forced_alignment",
        "confidence": 0.94,
        "fallback_counts": {"transcript_proportional": 1},
    }
    encoded = json.dumps(result)
    assert "RIFF-forbidden" not in encoded
    assert "forbidden full ledger" not in encoded
    assert "character_spans" not in encoded


def test_browser_chunk_sanitizer_enforces_numeric_collection_and_byte_bounds() -> None:
    program = r"""
const fs = require("fs");
const vm = require("vm");
global.window = global;
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), { filename: process.argv[1] });
const longText = "x".repeat(10000);
const largeMap = {};
for (let index = 0; index < 1000; index += 1) largeMap[`${longText}-${index}`] = longText;
let barChecks = 0;
const boundedBars = new Proxy(
  [Number.MAX_SAFE_INTEGER + 1, ...Array(999).fill(2)],
  {
    has(target, key) {
      if (/^\d+$/.test(String(key))) {
        barChecks += 1;
        if (barChecks > 4) throw new Error("bars scanned past the bounded prefix");
      }
      return Reflect.has(target, key);
    },
  },
);
const bounded = global.StreamMuseRapState.sanitizeChunkPayload({
  chunk_index: Number.MAX_SAFE_INTEGER + 1,
  bars: boundedBars,
  request_budget_ms: 0,
  elapsed_ms: -1,
  stage_timings_ms: { total: -1 },
  selected_lines: Array(1000).fill(longText),
  flows: Array(1000).fill({
    template_id: longText,
    slots: Array.from({ length: 1000 }, (_, index) => ({
      tick_in_bar: index, target_stress: index === 0 ? 2 : 1, rhyme_group: longText,
    })),
    selected_syllable_schedule: longText,
  }),
  prompt_summary: longText,
  context_lines: Array(1000).fill(longText),
  warnings: Array(1000).fill(longText),
  hashes: largeMap,
  artifact_refs: largeMap,
  alignment: { confidence: 2 },
});
process.stdout.write(JSON.stringify({
  bounded,
  barChecks,
  bytes: new TextEncoder().encode(JSON.stringify(bounded)).length,
  hashCount: Object.keys(bounded.hashes).length,
}));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(REDUCER)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["bounded"]["chunk_index"] is None
    assert result["bounded"]["bars"] == [2]
    assert result["barChecks"] <= 4
    assert result["bounded"]["flows"][0]["slots"][0]["target_stress"] is None
    assert result["bounded"]["alignment"]["confidence"] is None
    assert result["bounded"]["request_budget_ms"] == 0
    assert result["bounded"]["elapsed_ms"] is None
    assert result["bounded"]["stage_timings_ms"] == {}
    assert result["hashCount"] <= 8
    assert result["bytes"] <= 24_000


def test_browser_remote_chunk_renderer_populates_every_research_group() -> None:
    program = r"""
const fs = require("fs");
const vm = require("vm");
global.window = global;
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"), { filename: process.argv[1] });

class FakeNode {
  constructor(id = "") { this.id = id; this.hidden = false; this.children = []; this._text = ""; }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ""; this.children = [...children]; }
}
const ids = [
  "remote-chunk-panel", "remote-request-id", "remote-renderer", "remote-state",
  "remote-chunk-index", "remote-bars", "remote-lines", "remote-flows",
  "remote-candidate-counts", "remote-scores", "remote-prompt-summary",
  "remote-context-lines", "remote-stage-timings", "remote-budget", "remote-elapsed",
  "remote-deadline-slack", "remote-alignment", "remote-stretch-warnings",
  "remote-warnings", "remote-hashes", "remote-artifacts", "remote-transfer-bytes",
  "remote-failure-reason",
];
const nodes = Object.fromEntries(ids.map((id) => [id, new FakeNode(id)]));
const document = {
  getElementById(id) { return nodes[id] || null; },
  createElement() { return new FakeNode(); },
};
global.StreamMuseRapState.renderRemoteChunk({
  event_type: "chunk_committed",
  request_id: "request-2",
  state: "committed",
  renderer_decision: "moss_aligned_remote",
  chunk_index: 2,
  bars: [4, 5],
  selected_lines: ["First remote line", "Second remote line"],
  flows: [
    { template_id: "flow-a", slot_stress_schedule: "t0@1.00, t2@0.25", selected_syllable_schedule: "t0:first/stress1" },
    { template_id: "flow-b", slot_stress_schedule: "t1@0.80", selected_syllable_schedule: "t1:second/stress1" },
  ],
  candidate_counts: { requested: 32, parseable: 30, valid: 8, selectable: 4 },
  selected_scores: [
    { bar: 4, total: 0.91, component_scores: { stress_alignment: 0.88 } },
    { bar: 5, total: 0.87, component_scores: { continuity: 0.82 } },
  ],
  prompt_summary: "system: clean rap | user: both exact schedules",
  context_lines: ["Prior committed line"],
  stage_timings_ms: {
    generation: 1000, evaluation: 80, moss: 2000, aligner: 40, r3: 120,
    package: 10, transfer: 20, mac: 6, total: 3276,
  },
  request_budget_ms: 5000,
  elapsed_ms: 3300,
  deadline_slack_ms: 1700,
  alignment: { method: "mms_forced_alignment", confidence: 0.94, fallback_counts: { transcript_proportional: 1 } },
  stretch_warnings: ["stretch_ratio_high:1.22"],
  warnings: ["alignment_fallback:one_word"],
  hashes: { vocal_sha256: "vocal-hash" },
  artifact_refs: { manifest: "/h200/request-2/manifest.json" },
  transfer_bytes: 262144,
  failure_reason: null,
}, document);
const populatedValues = Object.fromEntries(ids.slice(1).map((id) => [id, nodes[id].textContent]));
global.StreamMuseRapState.renderRemoteChunk({
  event_type: "chunk_requested",
  request_id: "request-3",
  state: "requested",
}, document);
process.stdout.write(JSON.stringify({
  hidden: nodes["remote-chunk-panel"].hidden,
  populatedValues,
  missingCounts: nodes["remote-candidate-counts"].textContent,
}));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(REDUCER)],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    values = result["populatedValues"]
    assert result["hidden"] is False
    assert result["missingCounts"] == "-- / -- / -- / --"
    assert values["remote-request-id"] == "request-2"
    assert values["remote-renderer"] == "moss_aligned_remote"
    assert values["remote-lines"] == "Bar 5: First remote lineBar 6: Second remote line"
    assert values["remote-flows"] == (
        "flow-a | targets t0@1.00, t2@0.25 | selected t0:first/stress1"
        "flow-b | targets t1@0.80 | selected t1:second/stress1"
    )
    assert values["remote-candidate-counts"] == "32 / 30 / 8 / 4"
    assert "stress_alignment=0.880" in values["remote-scores"]
    assert "continuity=0.820" in values["remote-scores"]
    assert values["remote-prompt-summary"] == "system: clean rap | user: both exact schedules"
    assert values["remote-context-lines"] == "Prior committed line"
    for timing in ("generation=1000.0", "evaluation=80.0", "moss=2000.0", "aligner=40.0", "r3=120.0", "package=10.0", "transfer=20.0", "mac=6.0", "total=3276.0"):
        assert timing in values["remote-stage-timings"]
    assert "mms_forced_alignment" in values["remote-alignment"]
    assert "confidence=0.940" in values["remote-alignment"]
    assert "transcript_proportional=1" in values["remote-alignment"]
    assert values["remote-stretch-warnings"] == "stretch_ratio_high:1.22"
    assert values["remote-hashes"] == "vocal_sha256=vocal-hash"
    assert values["remote-artifacts"] == "manifest=/h200/request-2/manifest.json"
    assert values["remote-transfer-bytes"] == "262144 bytes"
    assert values["remote-failure-reason"] == "none"
