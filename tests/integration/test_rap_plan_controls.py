"""Browser control logic uses the ASAP API and reports playback, not just acceptance."""

import json
from pathlib import Path
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/streammuse/presentation/rap_demo/static/js/rap-plan.js"


def run_js(code):
    result = subprocess.run(
        ["node", "-e", "global.window=global; require('vm').runInThisContext(require('fs').readFileSync(process.argv[1], 'utf8')); const api=StreamMuseRapPlan; " + code, str(SCRIPT)],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def run_controls_js(code):
    return run_js('''
function createNode() {
  return {
    value: "", textContent: "", disabled: false, dataset: {}, children: [], handlers: {},
    style: { setProperty() {} }, classList: { add() {} },
    addEventListener(type, handler) { this.handlers[type] = handler; },
    replaceChildren(...children) { this.children = children; },
    append(child) { this.children.push(child); },
    setAttribute() {},
  };
}
const flush = () => new Promise((resolve) => setImmediate(resolve));
const discovery = {
  supported: true, templates: [], next_editable_bar: 1, last_edit: null, state: "running",
};
function snapshot(epoch, state = "running", sequence = 1) {
  return {
    session_id: "test-session", coordinator_epoch: epoch, last_sequence: sequence,
    audio: { state, current_bar: state === "running" ? 0 : null },
  };
}
function event(sequence, type, payload) {
  return { session_id: "test-session", sequence, event_type: type, payload };
}
function harness() {
  const nodes = new Map();
  const document = {
    getElementById(id) {
      if (!nodes.has(id)) nodes.set(id, createNode());
      return nodes.get(id);
    },
    createElement: createNode,
  };
  const requests = [];
  global.fetch = (url, options = {}) => {
    if (url !== "/api/control/plan" || (options.method && options.method !== "GET")) {
      throw new Error(`Unexpected request: ${options.method || "GET"} ${url}`);
    }
    return new Promise((resolve, reject) => requests.push({ resolve, reject }));
  };
  return {
    controls: api.mount(document), get: document.getElementById, requests,
    async resolveDiscovery(info = discovery) {
      // Finish the pending response and any automatic replacement after invalidation.
      for (let attempt = 0; attempt < 10; attempt += 1) {
        await flush();
        if (!requests.length) return;
        requests.shift().resolve({ ok: true, json: async () => info });
      }
      throw new Error("Discovery did not settle");
    },
  };
}
(async () => {
''' + code + '''
})().catch((error) => { console.error(error); process.exitCode = 1; });
''')


def test_asap_payload_omits_target_bar_and_unchanged_fields():
    assert run_js('process.stdout.write(JSON.stringify(api.makePayload("  deep sea  ", "")));') == {"topic": "deep sea"}
    assert run_js('process.stdout.write(JSON.stringify(api.makePayload("", "baseline_syncopated_9")));') == {"template_id": "baseline_syncopated_9"}
    assert run_js('try { api.makePayload(" ", ""); } catch(e) { process.stdout.write(JSON.stringify(e.message)); }') == "Enter a topic or choose a flow."


def test_edit_is_not_applied_until_matching_audio_starts_playing():
    result = run_js('''
const edit={effective_from_bar:3,topic:"deep sea",template_id:"sync"};
const bar={bar:3,frozen:true,topic:"deep sea",template_id:"sync",choice:"remote_latest"};
process.stdout.write(JSON.stringify([
 api.editStatus(edit,[],2), api.editStatus(edit,[bar],2), api.editStatus(edit,[bar],3),
 api.editStatus(edit,[{...bar,topic:"space",choice:"remote_older"}],3),
 api.editStatus(edit,[{...bar,choice:"espeak_stream_lyric"}],3),
 api.editStatus(edit,[{...bar,choice:"espeak_local_lyric"}],3)
]));''')
    assert [item["state"] for item in result] == ["pending", "ready", "applied", "delayed", "applied", "delayed"]
    assert "eSpeak" in result[4]["text"]
    assert "older" in result[3]["text"]
    assert "fallback" in result[5]["text"]


def test_unfrozen_or_pre_target_bars_never_confirm_an_edit():
    assert run_js('''process.stdout.write(JSON.stringify(api.editStatus(
 {effective_from_bar:3,topic:"sea"},
 [{bar:2,frozen:true,topic:"sea"},{bar:3,frozen:false,topic:"sea"}],3
).state));''') == "pending"


def test_silent_recovery_bar_does_not_confirm_a_topic_edit():
    result = run_js('''process.stdout.write(JSON.stringify(api.editStatus(
 {effective_from_bar:3,topic:"sea"},
 [{bar:3,frozen:true,topic:"sea",choice:"silence"}],3
)));''')
    assert result["state"] == "delayed"
    assert "silence" in result["text"]


def test_discovery_recovers_when_initial_snapshot_changes_epoch_during_fetch():
    result = run_controls_js('''
const h = harness();
h.controls.update({}, [], { bar: null }, []);
h.controls.connection(true);
h.controls.update(snapshot(0), [], { bar: 0 }, []);
await h.resolveDiscovery();
h.get("plan-topic").value = "deep sea";
h.get("plan-topic").handlers.input();
process.stdout.write(JSON.stringify({
  topic_disabled: h.get("plan-topic").disabled,
  apply_disabled: h.get("plan-apply").disabled,
  mode: h.get("plan-mode").textContent,
}));
''')
    assert result == {"topic_disabled": False, "apply_disabled": False, "mode": "ASAP"}


@pytest.mark.parametrize("include_epoch", [True, False], ids=["payload-epoch", "legacy-sequence"])
def test_reset_rejects_cached_edits_and_accepts_new_epoch_lower_edit_seq(include_epoch):
    result = run_controls_js("const includeEpoch = " + json.dumps(include_epoch) + ";\n" + '''
const h = harness();
const oldEdit = event(100, "plan_edited", {
  edit_seq: 8, from_bar: 7, effective_from_bar: 7, topic: "old topic", template_id: null,
  ...(includeEpoch ? { coordinator_epoch: 0 } : {}),
});
h.controls.update(snapshot(0, "running", 100), [oldEdit], { bar: 6 }, []);
h.controls.connection(true);
await h.resolveDiscovery({ ...discovery, next_editable_bar: 7, last_edit: oldEdit.payload });
const initialRequest = h.get("plan-request").textContent;
const stopped = event(101, "session_stopped", { coordinator_epoch: 0 });
const stoppedSnapshot = snapshot(0, "stopped", 101);
h.controls.update(stoppedSnapshot, [oldEdit, stopped], { bar: 6 }, []);
h.controls.reset();
// The reset HTTP response renders before the websocket cache receives session_reset.
h.controls.update(stoppedSnapshot, [oldEdit, stopped], { bar: 6 }, []);
const afterReset = h.get("plan-request").textContent;
// /api/state may expose the new epoch while monitor.events still holds the old cache.
h.controls.update(snapshot(1, "stopped", 102), [oldEdit, stopped], { bar: null }, []);
const afterEpoch = h.get("plan-request").textContent;
await h.resolveDiscovery({ ...discovery, state: "stopped", next_editable_bar: 0 });
const reset = event(102, "session_reset", { coordinator_epoch: 1 });
const started = event(103, "session_started", { coordinator_epoch: 1 });
const newEdit = event(104, "plan_edited", {
  edit_seq: 1, from_bar: 1, effective_from_bar: 1, topic: "new topic", template_id: null,
  ...(includeEpoch ? { coordinator_epoch: 1 } : {}),
});
h.controls.update(snapshot(1, "running", 104), [reset, started, newEdit], { bar: 0 }, []);
process.stdout.write(JSON.stringify({
  initial_request: initialRequest, after_reset: afterReset, after_epoch: afterEpoch,
  current_request: h.get("plan-request").textContent,
  current_status: h.get("plan-status").dataset.state,
}));
''')
    assert "old topic" in result["initial_request"]
    assert result["after_reset"] == "", result
    assert result["after_epoch"] == "", result
    assert "new topic" in result["current_request"], result
    assert "old topic" not in result["current_request"], result
    assert result["current_status"] == "pending"


def test_successful_reconnect_discovery_clears_previous_discovery_error():
    result = run_controls_js('''
const h = harness();
h.controls.update(snapshot(0), [], { bar: 0 }, []);
h.controls.connection(true);
h.requests.shift().reject(new Error("temporary network failure"));
await flush();
const failed = {
  state: h.get("plan-status").dataset.state,
  text: h.get("plan-status").textContent,
};
h.controls.connection(false);
h.controls.connection(true);
await h.resolveDiscovery({
  ...discovery,
  last_edit: { edit_seq: 1, from_bar: 1, effective_from_bar: 1, topic: "deep sea" },
});
process.stdout.write(JSON.stringify({
  failed,
  recovered_state: h.get("plan-status").dataset.state,
  recovered_text: h.get("plan-status").textContent,
  current_request: h.get("plan-request").textContent,
  topic_disabled: h.get("plan-topic").disabled,
}));
''')
    assert result["failed"] == {"state": "error", "text": "temporary network failure"}
    assert result["topic_disabled"] is False
    assert "deep sea" in result["current_request"]
    assert result["recovered_state"] == "pending", result
    assert "temporary network failure" not in result["recovered_text"]
