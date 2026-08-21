"""Browser-state reducer regressions for the realtime rap monitor."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
REDUCER = ROOT / "src/streammuse/presentation/rap_demo/static/js/rap-demo-state.js"


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
    { template_id: "flow-a", slot_stress_schedule: "t0@1.00, t2@0.25" },
    { template_id: "flow-b", slot_stress_schedule: "t1@0.80" },
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
    assert values["remote-flows"] == "flow-a | t0@1.00, t2@0.25flow-b | t1@0.80"
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
