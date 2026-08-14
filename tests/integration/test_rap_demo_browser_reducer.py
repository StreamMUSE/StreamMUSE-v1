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
