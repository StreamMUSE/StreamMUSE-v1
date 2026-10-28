"""Run the real monitor renderer against a small text-only DOM."""

import json
from pathlib import Path
import subprocess

import pytest


MAIN_SCRIPT = Path(__file__).resolve().parents[2] / "src/streammuse/presentation/rap_demo/static/js/rap-demo.js"


@pytest.mark.parametrize("emitted_labels", [False, True])
def test_tick_cells_show_written_fragments_safely_without_losing_full_line(emitted_labels):
    program = r"""
const fs = require("fs");
const vm = require("vm");
class FakeNode {
  constructor() {
    this.children = []; this.dataset = {}; this.attributes = {}; this.options = [];
    this.classList = { add() {}, remove() {}, toggle() {} };
    this.style = { setProperty() {} }; this.parentElement = this;
    this.clientWidth = 500; this.offsetLeft = 0; this.offsetWidth = 30;
    this._text = "";
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map((child) => child.textContent).join(""); }
  set innerHTML(value) { throw new Error("Labels must not use innerHTML"); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this._text = ""; this.children = [...children]; }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener() {}
}
global.window = global;
global.Node = FakeNode;
const nodes = new Map();
global.document = {
  getElementById(id) {
    if (!nodes.has(id)) nodes.set(id, new FakeNode());
    return nodes.get(id);
  },
  createElement() { return new FakeNode(); },
  querySelectorAll() { return []; },
};
global.addEventListener = () => {};
global.fetch = () => new Promise(() => {});
global.location = { protocol: "http:", host: "localhost" };
global.WebSocket = class {};
vm.runInThisContext(fs.readFileSync(process.argv[1], "utf8"));
const maliciousLabel = "<img src=x onerror=alert(1)>";
const scheduled = [
  { tick_in_bar: 0, slot_index: 0, label: "whisper", word: "whisper", display_label: "whis-", display_index: 0, display_count: 2 },
  { tick_in_bar: 2, slot_index: 1, label: ".", word: "whisper", display_label: "-per", display_index: 1, display_count: 2 },
  { tick_in_bar: 4, slot_index: 2, label: "banana", word: "banana", display_label: "ba-", display_index: 0, display_count: 3 },
  { tick_in_bar: 6, slot_index: 3, label: ".", word: "banana", display_label: "-na-", display_index: 1, display_count: 3 },
  { tick_in_bar: 8, slot_index: 4, label: ".", word: "banana", display_label: "-na", display_index: 2, display_count: 3 },
  { tick_in_bar: 10, slot_index: 5, label: maliciousLabel, word: maliciousLabel, display_label: maliciousLabel, display_index: 0, display_count: 1 },
];
const emitted = scheduled.map((item) => {
  const event = { ...item, bar: 0, tick: item.tick_in_bar, beat: 0, subdivision: 2, stress: 0, observation_delay_ms: 12.5 };
  if (process.argv[2] === "false") delete event.display_label;
  return event;
});
const decision = { bar: 0, text: "whisper banana " + maliciousLabel, scheduled_syllables: scheduled,
  flow: { ticks_per_beat: 4, beats_per_bar: 4, slots: scheduled.map((item) => ({ tick_in_bar: item.tick_in_bar, target_stress: 0.8 })) },
};
const snapshot = { bars: [decision], emitted_syllables: emitted, current_syllable: emitted[1] };
const current = { bar: 0, tick: 2, segment: {} };
renderDecision(snapshot, current, decision);
const cells = nodes.get("flow-track").children;
const result = {
  labels: scheduled.map((item) => cells[item.tick_in_bar].children[1].textContent),
  labelChildCount: cells[10].children[1].children.length,
  cellCount: cells.length, fullLine: nodes.get("selected-line").textContent,
  currentLabel: nodes.get("current-syllable").textContent,
  detail: nodes.get("current-syllable-detail").textContent,
  ariaLabel: cells[2].attributes["aria-label"],
};
// The new emitted contract also works when the scheduled bar has aged out.
renderFlow({ emitted_syllables: [emitted[1]] }, current, decision.flow, {});
result.emittedOnly = nodes.get("flow-track").children[2].children[1].textContent;
renderFlow({ emitted_syllables: [{ bar: 0, tick: 2, word: "legacy" }] }, current, decision.flow,
  { scheduled_syllables: [{ tick_in_bar: 2, label: ".", word: "old" }] });
result.legacyWord = nodes.get("flow-track").children[2].children[1].textContent;
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", program, str(MAIN_SCRIPT), str(emitted_labels).lower()],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    markup = "<img src=x onerror=alert(1)>"
    assert result["labels"] == ["whis-", "-per", "ba-", "-na-", "-na", markup]
    assert result["cellCount"] == 16
    assert result["labelChildCount"] == 0
    assert result["fullLine"] == "whisper banana " + markup
    assert result["currentLabel"] == "-per"
    assert "observation delay 12.50 ms" in result["detail"]
    assert "target stress 0.80" in result["ariaLabel"]
    assert result["legacyWord"] == "legacy"
    if emitted_labels:
        assert result["emittedOnly"] == "-per"
