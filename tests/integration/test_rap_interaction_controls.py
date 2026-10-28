"""Browser-owned settings, clock mapping, capture filtering and timing evidence."""

import json
from pathlib import Path
import subprocess

import pytest

STATIC = Path(__file__).resolve().parents[2] / "src/streammuse/presentation/rap_demo/static/js"


def run_js(module, code):
    program = """
const fs = require('fs'), vm = require('vm');
global.window = global;
if (fs.existsSync(process.argv[1])) vm.runInThisContext(fs.readFileSync(process.argv[1], 'utf8'));
""" + code
    result = subprocess.run(["node", "-e", program, str(STATIC / module)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


DOM_HARNESS = """
const nodes = new Map();
function node(id='') {
 return {id,tagName:'DIV',value:'',checked:false,textContent:'',disabled:false,hidden:false,dataset:{},children:[],handlers:{},options:[],
 style:{setProperty(){}},classList:{add(){},remove(){},toggle(){}},
 addEventListener(name,fn){this.handlers[name]=fn;},setAttribute(name,value){this[name]=value;},
 replaceChildren(...items){this.children=items;},append(...items){this.children.push(...items);},closest(){return null;}};
}
const document = {hidden:false,activeElement:null,getElementById(id){if(!nodes.has(id))nodes.set(id,node(id));return nodes.get(id);},createElement:()=>node(),addEventListener(){}};
const get = document.getElementById;
const timers = [];
global.setTimeout=(fn,delay)=>{timers.push({fn,delay});return timers.length;};
global.clearTimeout=id=>{if(timers[id-1])timers[id-1].cancelled=true;};
const debounce=async()=>{for(const timer of [...timers])if(!timer.cancelled&&timer.delay<1000){timer.cancelled=true;timer.fn();}await flush();};
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const requests=[];
global.fetch=(url,options={})=>new Promise((resolve,reject)=>requests.push({url,options,resolve,reject}));
const reply=(body,ok=true)=>requests.shift().resolve({ok,status:ok?200:422,json:async()=>body});
"""


SETTINGS_HARNESS = DOM_HARNESS + """
const groupFields={local:['drum_gain','vocal_gain','chord_gain','chord_loop','alignment_policy'],generation:['tempo_bpm'],session:['mode','response_bars','output_device']};
const baseline={tempo_bpm:90,mode:'continuous',response_bars:2,output_device:null,drum_gain:1,vocal_gain:1,chord_gain:0.2,chord_loop:'off',alignment_policy:'gentle_sparse_r3'};
const settings=(values={},metadata={},state='running')=>({
 values:{...baseline,...values},state,supported:true,devices:[],
 capabilities:Object.fromEntries(Object.keys(baseline).map(key=>[key,true])),
 application:{drum_gain:'immediate',vocal_gain:'immediate',chord_gain:'immediate',chord_loop:'bar-boundary',alignment_policy:'next-uncommitted-bar',tempo_bpm:'bar-boundary',mode:'stopped-only',response_bars:'stopped-only',output_device:'stopped-only'},
 groups:Object.fromEntries(Object.entries(groupFields).map(([group,fields])=>[group,{fields,...metadata[group]}])),
});
const accepted=(group,values,ack)=>({...settings(values,{[group]:ack}),group,ack});
const change=(field,value)=>{const control=get('setting-'+field.replaceAll('_','-'));control.value=value;return (control.handlers.input||control.handlers.change)();};
const submitTempo=()=>get('settings-form').handlers.submit({preventDefault(){}});
"""


def test_applied_policy_stays_confirmed_after_idle_stop_and_settings_refresh():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const info=settings({alignment_policy:'all_onsets_stress_r3'},{local:{changed_fields:['alignment_policy'],revision:1,effective_from_bar:5}});
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 controls.update({session_id:'s',coordinator_epoch:0,audio:{state:'running',current_bar:5,timing_evidence:{bar:5,warp:{policy:'all_onsets_stress_r3'}}}});
 const played=get('settings-local-status').textContent;
 controls.update({session_id:'s',coordinator_epoch:0,audio:{state:'stopped',current_bar:7,timing_evidence:{bar:7,warp:null}}});
 const stopped=get('settings-local-status').textContent;
 const poll=controls.load();reply({...info,state:'stopped'});await poll;
 const refreshed=get('settings-local-status').textContent;
 controls.connection(false);controls.connection(true);reply({...info,state:'stopped'});await flush();
 process.stdout.write(JSON.stringify({played,stopped,refreshed,reconnected:get('settings-local-status').textContent}));
})();
""")
    assert set(result.values()) == {"Applied in bar 6."}


def test_reload_confirms_only_frozen_history_that_has_actually_played_in_this_epoch():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const results=[];
 const frozen={bar:5,frozen:true,session_id:'s',coordinator_epoch:2,timing_evidence:{bar:5,warp:{policy:'all_onsets_stress_r3'}}};
 for(const [current,bar] of [[7,frozen],[4,frozen],[null,frozen],[7,{...frozen,coordinator_epoch:1}],[7,{...frozen,session_id:'other'}],[7,{...frozen,timing_evidence:{...frozen.timing_evidence,bar:3}}]]) {
  const controls=StreamMuseRapSettings.mount(document);
  controls.update({session_id:'s',coordinator_epoch:2,current_tick:160,audio:{state:'stopped',current_bar:current,timing_evidence:{bar:7,warp:null}},frozen_bars:{5:bar}});
  controls.connection(true);reply(settings({alignment_policy:'all_onsets_stress_r3'},{local:{changed_fields:['alignment_policy'],revision:1,effective_from_bar:5}},'stopped'));await flush();
  results.push({state:get('settings-local-status').dataset.state,text:get('settings-local-status').textContent});controls.connection(false);
 }
 process.stdout.write(JSON.stringify(results));
})();
""")
    assert result[0] == {"state": "applied", "text": "Applied in bar 6."}
    assert result[1:] == [{"state": "saved", "text": "Saved for next playback."}] * 5


def test_new_policy_request_and_epoch_clear_sticky_application_evidence():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings({alignment_policy:'all_onsets_r3'},{local:{changed_fields:['alignment_policy'],revision:1,effective_from_bar:5}}));await flush();
 controls.update({session_id:'s',coordinator_epoch:0,audio:{state:'running',current_bar:5,timing_evidence:{bar:5,warp:{policy:'all_onsets_r3'}}}});
 change('alignment_policy','all_onsets_stress_r3');
 const applying=get('settings-local-status').textContent;
 reply(accepted('local',{alignment_policy:'all_onsets_stress_r3'},{changed_fields:['alignment_policy'],revision:2,effective_from_bar:6}));await flush();
 const pending=get('settings-local-status').textContent;
 controls.update({session_id:'s',coordinator_epoch:0,audio:{state:'running',current_bar:6,timing_evidence:{bar:6,warp:{policy:'all_onsets_stress_r3'}}}});
 controls.update({session_id:'s',coordinator_epoch:1,audio:{state:'stopped',current_bar:null}});
 reply(settings({alignment_policy:'all_onsets_stress_r3'},{},'stopped'));await flush();
 process.stdout.write(JSON.stringify({applying,pending,reset:get('settings-local-status').dataset.state}));
})();
""")
    assert result["applying"] == "Applying locally..."
    assert result["pending"] == "Scheduled for bar 7. Awaiting playback."
    assert result["reset"] == "idle"


def test_availability_hints_distinguish_connection_backend_and_playback_state():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);
 const hints=()=>['drum-gain','alignment-policy','chord-loop','tempo-bpm','mode'].map(field=>get('setting-'+field+'-hint').textContent);
 const disconnected=hints();controls.connection(true);reply(settings());await flush();
 const running=hints();controls.update({audio:{state:'stopped'}});const stopped=hints();
 controls.update({audio:{state:'priming'}});const priming=hints();
 controls.update({audio:{state:'stop_requested'}});const stopping=hints();
 controls.update({audio:{state:'running'}});
 const poll=controls.load();reply({...settings(),capabilities:{}});await poll;
 process.stdout.write(JSON.stringify({disconnected,running,stopped,priming,stopping,unsupported:hints()}));
})();
""")
    assert result == {
        "disconnected": ["Connecting"] * 5,
        "running": ["Immediate", "Next eligible bar", "Next eligible bar", "Server-coordinated", "Stop to change"],
        "stopped": ["Immediate", "Next eligible bar", "Next eligible bar", "Server-coordinated", "Ready while stopped"],
        "priming": ["Wait for playback"] * 5,
        "stopping": ["Wait for stop"] * 5,
        "unsupported": ["Unavailable for this backend"] * 5,
    }


def test_group_status_distinguishes_local_work_server_wait_and_scheduled_tempo():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings());await flush();
 change('tempo_bpm','120');const tempo=submitTempo();const remote=requests.shift();
 change('drum_gain','0.7');await debounce();
 const waiting={local:get('settings-local-status').textContent,generation:get('settings-status').textContent,hint:get('setting-tempo-bpm-hint').textContent};
 reply(accepted('local',{drum_gain:0.7},{changed_fields:['drum_gain'],mix_revision:1}));await flush();
 remote.resolve({ok:true,json:async()=>accepted('generation',{tempo_bpm:120},{changed_fields:['tempo_bpm'],revision:1,effective_from_bar:10})});await tempo;
 process.stdout.write(JSON.stringify({waiting,scheduled:get('settings-status').textContent}));
})();
""")
    assert result["waiting"] == {"local": "Applying locally...", "generation": "Waiting for server...", "hint": "Waiting for server"}
    assert result["scheduled"] == "Scheduled for bar 11. Awaiting playback."


def test_missing_vocal_stem_is_a_local_playback_wait_not_a_server_wait():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 const info=settings({vocal_gain:0},{local:{changed_fields:['vocal_gain'],mix_revision:2,live_mix:{revision:2}}});
 reply({...info,audio_output:{live_mix:{revision:2,applied_revision:2,supported:true,supported_gains:['drum_gain'],applied:{drum_gain:1,vocal_gain:null}}}});await flush();
 const waiting=get('settings-local-status').textContent;
 controls.update({audio:{state:'running',current_bar:5},audio_output:{live_mix:{bar:5,revision:2,applied_revision:2,supported:true,supported_gains:['vocal_gain'],applied:{vocal_gain:0}}}});
 controls.update({audio:{state:'stopped',current_bar:6},audio_output:{live_mix:{applied:null}}});
 process.stdout.write(JSON.stringify({waiting,applied:get('settings-local-status').textContent}));
})();
""")
    assert result == {"waiting": "Saved; awaiting vocal playback.", "applied": "Applied in bar 6."}


def test_plan_controls_explain_mode_and_lifecycle_locks_and_remote_wait():
    result = run_js("rap-plan.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapPlan.mount(document);controls.settings({values:{mode:'continuous'}});controls.connection(true);
 reply({supported:true,templates:[],next_editable_bar:3});await flush();
 const update=state=>controls.update({session_id:'s',coordinator_epoch:0,audio:{state,current_bar:2}},[],{bar:2},[]);
 update('stopped');const stopped=get('plan-topic-hint').textContent;
 update('priming');const priming=get('plan-topic-hint').textContent;
 update('running');controls.settings({values:{mode:'alternating'}});const flow=get('plan-template-hint').textContent;
 get('plan-topic').value='sea';get('plan-topic').handlers.input();const submit=get('plan-form').handlers.submit({preventDefault(){}});
 const waiting=get('plan-status').textContent;
 reply({edit_seq:1,effective_from_bar:3,topic:'sea'});await submit;
 controls.connection(false);
 process.stdout.write(JSON.stringify({stopped,priming,flow,waiting,disconnected:get('plan-topic-hint').textContent}));
})();
""")
    assert result == {"stopped": "Start playback to change", "priming": "Wait for playback",
                      "flow": "Captured in alternating mode", "waiting": "Waiting for server...", "disconnected": "Connecting"}


def test_local_coalesces_gains_while_generation_is_pending_and_ignores_stale_full_snapshot():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings());await flush();
 change('tempo_bpm','120');const tempo=submitTempo();
 const generation=requests.shift();
 const enabled=!get('setting-drum-gain').disabled&&!get('setting-alignment-policy').disabled;
 change('drum_gain','0.6');change('drum_gain','0.7');change('vocal_gain','0.8');
 const before=requests.length;await debounce();
 const local=requests.shift();
 const pending=get('settings-local-status').dataset.state;
 local?.resolve({ok:true,json:async()=>accepted('local',{drum_gain:0.7,vocal_gain:0.8},{changed_fields:['drum_gain','vocal_gain'],revision:1,effective_from_bar:0})});await flush();
 const localStatus=get('settings-local-status').textContent;
 const generationStatus=get('settings-status').textContent;
 generation.resolve({ok:true,json:async()=>accepted('generation',{tempo_bpm:120},{changed_fields:['tempo_bpm'],revision:1,effective_from_bar:4})});await tempo;
 process.stdout.write(JSON.stringify({enabled,before,pending,localStatus,generationStatus,
  local:local&&{url:local.url,body:JSON.parse(local.options.body)},generation:{url:generation.url,body:JSON.parse(generation.options.body)},
  gain:Number(get('setting-drum-gain').value),hints:['drum-gain','chord-loop','alignment-policy'].map(field=>get('setting-'+field+'-hint').textContent)}));
})();
""")
    assert result["enabled"] is True
    assert result["before"] == 0
    assert result["pending"] == "pending"
    assert result["local"] == {"url": "/api/control/settings/local", "body": {"drum_gain": 0.7, "vocal_gain": 0.8}}
    assert result["generation"] == {"url": "/api/control/settings/generation", "body": {"tempo_bpm": 120}}
    assert "Accepted" in result["localStatus"]
    assert result["generationStatus"] == "Waiting for server..."
    assert result["gain"] == 0.7
    assert result["hints"] == ["Immediate", "Next eligible bar", "Next eligible bar"]


def test_local_edits_during_request_send_only_latest_followup():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings());await flush();
 change('drum_gain','0.6');await debounce();
 const first=requests.shift();
 change('drum_gain','0.3');change('drum_gain','0.4');await debounce();
 const whileBusy=requests.length;
 first?.resolve({ok:true,json:async()=>accepted('local',{drum_gain:0.6},{changed_fields:['drum_gain'],revision:1})});await flush();await debounce();
 process.stdout.write(JSON.stringify({first:first&&JSON.parse(first.options.body),whileBusy,next:requests.map(r=>({url:r.url,body:JSON.parse(r.options.body)})),value:Number(get('setting-drum-gain').value)}));
})();
""")
    assert result == {"first": {"drum_gain": 0.6}, "whileBusy": 0,
                      "next": [{"url": "/api/control/settings/local", "body": {"drum_gain": 0.4}}], "value": 0.4}


def test_alignment_and_queued_mix_are_separate_operations_with_independent_revisions():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings());await flush();
 change('drum_gain','0.6');change('alignment_policy','all_onsets_r3');
 const first=requests.shift();
 first?.resolve({ok:true,json:async()=>accepted('local',{alignment_policy:'all_onsets_r3'},{changed_fields:['alignment_policy'],revision:5,effective_from_bar:4})});await flush();await debounce();
 const second=requests.shift();
 second?.resolve({ok:true,json:async()=>({...accepted('local',{alignment_policy:'all_onsets_r3',drum_gain:0.6},{changed_fields:['drum_gain'],revision:1,mix_revision:1,effective_from_bar:2,live_mix:{revision:1}}),audio_output:{live_mix:{applied:{drum_gain:0.6},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}}})});await flush();
 const pending=get('settings-local-status').textContent;
 controls.update({audio:{state:'running',timing_evidence:{bar:4,warp:{policy:'all_onsets_r3'}}}});
 process.stdout.write(JSON.stringify({first:first&&JSON.parse(first.options.body),second:second&&JSON.parse(second.options.body),value:Number(get('setting-drum-gain').value),pending,played:get('settings-local-status').dataset.state}));
})();
""")
    assert result["first"] == {"alignment_policy": "all_onsets_r3"}
    assert result["second"] == {"drum_gain": 0.6}
    assert result["value"] == 0.6
    assert result["pending"] == "Scheduled for bar 5. Awaiting playback."
    assert result["played"] == "applied"


def test_local_evidence_requires_live_gain_and_played_alignment_not_requested_values():
    result = run_js("rap-settings.js", """
const api=StreamMuseRapSettings;
const ack={revision:2,effective_from_bar:4,changed_fields:['drum_gain','alignment_policy'],values:{drum_gain:0.7,alignment_policy:'all_onsets_r3'}};
const played={bar:4,mix:{drum_gain:0.7},warp:{policy:'all_onsets_r3'}};
process.stdout.write(JSON.stringify([
 api.settingsStatus(ack,played).state,
 api.settingsStatus(ack,{...played,bar:3},{live_mix:{applied:{drum_gain:0.7},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}}).state,
 api.settingsStatus(ack,{...played,warp:{policy:'gentle_sparse_r3'}},{live_mix:{applied:{drum_gain:0.7},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}}).state,
 api.settingsStatus(ack,played,{live_mix:{applied:{drum_gain:0.7},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}}).state,
 api.settingsStatus({...ack,changed_fields:['drum_gain']},null,{live_mix:{applied:{drum_gain:0.7},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}}).state,
]));
""")
    assert result == ["pending", "pending", "pending", "applied", "applied"]


def test_live_gain_confirmation_requires_finished_callback_revision_and_supported_stem():
    result = run_js("rap-settings.js", """
const ack={changed_fields:['drum_gain'],values:{drum_gain:0.7},live_mix:{revision:3},application:{drum_gain:'immediate'}};
const live={requested:{drum_gain:0.7},applied:{drum_gain:0.7},revision:3,applied_revision:3,supported:true,supported_gains:['drum_gain']};
process.stdout.write(JSON.stringify([
 {applied:null}, {applied_revision:null}, {applied_revision:2}, {supported_gains:['vocal_gain']}, {supported:false}, {revision:4}, {},
].map(patch=>StreamMuseRapSettings.settingsStatus(ack,null,{live_mix:{...live,...patch}}).state)));
""")
    assert result == ["pending"] * 6 + ["applied"]


def test_nonlive_sink_uses_declared_bar_boundary_gain_evidence():
    result = run_js("rap-settings.js", """
const ack={changed_fields:['drum_gain'],values:{drum_gain:0.7},effective_from_bar:4,application:{drum_gain:'bar-boundary'}};
process.stdout.write(JSON.stringify([3,4].map(bar=>StreamMuseRapSettings.settingsStatus(ack,{bar,mix:{drum_gain:0.7}}).state)));
""")
    assert result == ["pending", "applied"]


def test_local_timeout_checks_outcome_before_sending_a_newer_queued_value():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings());await flush();
 change('drum_gain','0.5');await debounce();
 requests.shift().reject(Object.assign(new Error('timeout'),{name:'TimeoutError'}));await flush();
 const unknown=get('settings-local-status').textContent;
 change('drum_gain','0.6');await debounce();const posts=requests.length;
 let poll=controls.load();reply(settings());await poll;
 const unresolved=get('settings-local-status').textContent;
 poll=controls.load();reply(settings({drum_gain:0.5},{local:{changed_fields:['drum_gain'],revision:1}}));await poll;
 process.stdout.write(JSON.stringify({unknown,unresolved,posts,next:requests.map(r=>({url:r.url,body:JSON.parse(r.options.body)})),value:Number(get('setting-drum-gain').value)}));
})();
""")
    assert "outcome is unknown" in result["unknown"]
    assert result["unresolved"] == result["unknown"]
    assert result["posts"] == 0
    assert result["next"] == [{"url": "/api/control/settings/local", "body": {"drum_gain": 0.6}}]
    assert result["value"] == 0.6


def test_old_epoch_reply_cannot_clear_new_local_request_or_overwrite_values():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings());await flush();
 controls.update({session_id:'session',coordinator_epoch:0,audio:{state:'running'}});
 change('alignment_policy','all_onsets_r3');const obsolete=requests.shift();
 controls.update({session_id:'session',coordinator_epoch:1,audio:{state:'running'}});
 reply(settings());await flush();
 change('alignment_policy','all_onsets_stress_r3');const current=requests.shift();
 obsolete.resolve({ok:true,json:async()=>accepted('local',{alignment_policy:'all_onsets_r3'},{changed_fields:['alignment_policy'],revision:1,effective_from_bar:4})});await flush();
 const sending=get('settings-local-status').textContent;
 current.resolve({ok:true,json:async()=>accepted('local',{alignment_policy:'all_onsets_stress_r3'},{changed_fields:['alignment_policy'],revision:2,effective_from_bar:5})});await flush();
 process.stdout.write(JSON.stringify({sending,value:get('setting-alignment-policy').value,status:get('settings-local-status').textContent}));
})();
""")
    assert result["sending"] == "Applying locally..."
    assert result["value"] == "all_onsets_stress_r3"
    assert result["status"] == "Scheduled for bar 6. Awaiting playback."


def test_stopped_tempo_rebuild_keeps_its_ack_but_reset_discards_it():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings({}, {}, 'stopped'));await flush();
 controls.update({session_id:'session',coordinator_epoch:0,audio:{state:'stopped'}});
 change('tempo_bpm','120');const submit=submitTempo();
 controls.update({session_id:'session',coordinator_epoch:1,audio:{state:'stopped'}});
 reply({...accepted('generation',{tempo_bpm:120},{changed_fields:['tempo_bpm'],revision:1,effective_from_bar:0}),state:'stopped'});await submit;
 const acceptedStatus=get('settings-status').textContent;
 reply(settings({tempo_bpm:120},{generation:{changed_fields:['tempo_bpm'],revision:1,effective_from_bar:0}},'stopped'));await flush();
 change('tempo_bpm','100');const stale=submitTempo();const request=requests.shift();
 controls.reset();reply(settings({tempo_bpm:120},{},'stopped'));await flush();
 request.resolve({ok:true,json:async()=>accepted('generation',{tempo_bpm:100},{changed_fields:['tempo_bpm'],revision:2,effective_from_bar:0})});await stale;
 process.stdout.write(JSON.stringify({acceptedStatus,reset:get('settings-status').dataset.state,value:Number(get('setting-tempo-bpm').value)}));
})();
""")
    assert result["acceptedStatus"] == "Saved for next playback."
    assert result["reset"] == "idle"
    assert result["value"] == 120


def test_stale_poll_does_not_revert_a_local_ack_and_capabilities_refresh():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 reply({...settings(),capabilities:{...settings().capabilities,tempo_bpm:false}});await flush();
 const unavailable=get('setting-tempo-bpm').disabled;
 const poll=controls.load();const oldPoll=requests.shift();
 change('chord_loop','minor_cycle');reply(accepted('local',{chord_loop:'minor_cycle'},{changed_fields:['chord_loop'],revision:1,effective_from_bar:2}));await flush();
 oldPoll.resolve({ok:true,json:async()=>settings()});await poll;
 process.stdout.write(JSON.stringify({unavailable,enabled:!get('setting-tempo-bpm').disabled,chord:get('setting-chord-loop').value,status:get('settings-local-status').textContent,timers:timers.filter(t=>!t.cancelled).map(t=>t.delay)}));
})();
""")
    assert result["unavailable"] is True
    assert result["enabled"] is True
    assert result["chord"] == "minor_cycle"
    assert result["status"] == "Scheduled for bar 3. Awaiting playback."
    assert 5000 in result["timers"]


@pytest.mark.parametrize("group,field,value,playback_state", [
    ("local", "chord_loop", "minor_cycle", "running"),
    ("generation", "tempo_bpm", 120, "running"),
    ("session", "response_bars", 4, "stopped"),
])
def test_poll_started_during_post_preserves_ack_and_refreshes_other_groups(group, field, value, playback_state):
    scenario = json.dumps({"group": group, "field": field, "value": value, "state": playback_state})
    result = run_js("rap-settings.js", SETTINGS_HARNESS + "const scenario = " + scenario + ";\n" + """
(async()=>{
 const {group,field,value,state}=scenario;
 const otherField=group==='local'?'tempo_bpm':'drum_gain';
 const initial=settings({}, {}, state);initial.capabilities[otherField]=false;
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(initial);await flush();
 const other=get('setting-'+otherField.replaceAll('_','-'));
 const initiallyDisabled=other.disabled;
 change(field,String(value));
 if(group==='generation') submitTempo();
 if(group==='session') get('settings-session-form').handlers.submit({preventDefault(){}});
 const post=requests.shift();
 const poll=controls.load();const stale=requests.shift();
 const current={...accepted(group,{[field]:value},{changed_fields:[field],revision:2,effective_from_bar:4}),state};
 post.resolve({ok:true,json:async()=>current});await flush();
 const status=get(group==='generation'?'settings-status':`settings-${group}-status`);
 const view=()=>({value:String(get('setting-'+field.replaceAll('_','-')).value),state:status.dataset.state,text:status.textContent,title:status.title});
 const before=view();
 stale.resolve({ok:true,json:async()=>settings({}, {[group]:{changed_fields:[field],revision:1,effective_from_bar:0}}, state)});await poll;
 const after=view();const otherEnabled=!other.disabled;
 const fresh=controls.load();reply({...current,groups:{...current.groups,[group]:{...current.groups[group],revision:3,effective_from_bar:6}}});await fresh;
 process.stdout.write(JSON.stringify({initiallyDisabled,before,after,otherEnabled,fresh:view()}));
})();
""")
    assert result["before"] == {
        "value": str(value),
        "state": "saved" if group == "session" else "pending",
        "text": "Saved for next playback." if group == "session" else "Scheduled for bar 5. Awaiting playback.",
        "title": "Accepted revision 2",
    }
    assert result["after"] == result["before"]
    assert result["initiallyDisabled"] is True
    assert result["otherEnabled"] is True
    assert result["fresh"]["title"] == "Accepted revision 3"
    assert result["fresh"]["value"] == str(value)


def test_device_auto_apply_is_separate_from_mode_and_response_submit():
    result = run_js("rap-settings.js", SETTINGS_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(settings({}, {}, 'stopped'));await flush();
 get('setting-mode-alternating').checked=true;get('setting-mode-continuous').checked=false;get('setting-mode-alternating').handlers.change();
 change('response_bars','4');change('output_device','device-2');await flush();
 const device=requests.shift();
 device?.resolve({ok:true,json:async()=>({...accepted('session',{output_device:'device-2'},{changed_fields:['output_device'],revision:1}),state:'stopped'})});await flush();
 const mode=get('setting-mode-alternating').checked;
 const submit=get('settings-session-form').handlers.submit?.({preventDefault(){}});
 const setup=requests.shift();
 process.stdout.write(JSON.stringify({device:device&&{url:device.url,body:JSON.parse(device.options.body)},mode,setup:setup&&{url:setup.url,body:JSON.parse(setup.options.body)}}));
})();
""")
    assert result == {"device": {"url": "/api/control/settings/session", "body": {"output_device": "device-2"}},
                      "mode": True, "setup": {"url": "/api/control/settings/session", "body": {"mode": "alternating", "response_bars": 4}}}


def test_alignment_and_chord_controls_validate_and_confirm_playback_evidence():
    result = run_js("rap-settings.js", """
const api=StreamMuseRapSettings;
const values=api.makePayload({alignment_policy:'all_onsets_stress_r3',chord_loop:'minor_cycle',chord_gain:'0.3'});
const info={state:'running',capabilities:{alignment_policy:true,chord_loop:true}};
const ack={changed_fields:['chord_loop','chord_gain'],values,effective_from_bar:2};
process.stdout.write(JSON.stringify({values,alignmentEnabled:api.fieldEnabled('alignment_policy',info,true),chordEnabled:api.fieldEnabled('chord_loop',info,true),
pending:api.settingsStatus(ack,{bar:2,mix:{chord_loop:'off'}},{live_mix:{applied:{chord_gain:0.3},revision:1,applied_revision:1,supported:true,supported_gains:['chord_gain']}}).state,
applied:api.settingsStatus(ack,{bar:2,mix:{chord_loop:'minor_cycle'}},{live_mix:{applied:{chord_gain:0.3},revision:1,applied_revision:1,supported:true,supported_gains:['chord_gain']}}).state}));
""")
    assert result["values"] == {"alignment_policy": "all_onsets_stress_r3", "chord_loop": "minor_cycle", "chord_gain": 0.3}
    assert result["alignmentEnabled"] is True
    assert result["chordEnabled"] is True
    assert result["pending"] == "pending"
    assert result["applied"] == "applied"


def test_settings_mount_preserves_edits_and_surfaces_rejected_changes():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const api=global.StreamMuseRapSettings;
 if(!api?.mount){process.stdout.write(JSON.stringify(null));return;}
 const values={tempo_bpm:90,mode:'continuous',response_bars:2,output_device:null,drum_gain:1,vocal_gain:1};
 const info={values,capabilities:Object.fromEntries(Object.keys(values).map(k=>[k,true])),state:'stopped',application:{},devices:[]};
 const controls=api.mount(document);
 controls.connection(true); reply(info); await flush();
 get('setting-tempo-bpm').value='120';get('setting-tempo-bpm').handlers.input();
 const pending=controls.load();reply(info);await pending;
 const preserved=get('setting-tempo-bpm').value;
 const submit=get('settings-form').handlers.submit({preventDefault(){}});
 const request=JSON.parse(requests[0].options.body);reply({detail:'Tempo unavailable on this backend'},false);await submit;
 const error=get('settings-status').textContent;
 const state=get('settings-status').dataset.state;
 controls.connection(false);
 process.stdout.write(JSON.stringify({preserved,request,error,state,disabled:get('setting-tempo-bpm').disabled}));
})().catch(e=>{console.error(e);process.exitCode=1;});
""")
    assert result is not None
    assert result["preserved"] == "120"
    assert result["request"] == {"tempo_bpm": 120}
    assert result["error"] == "Tempo unavailable on this backend"
    assert result["state"] == "error"
    assert result["disabled"] is True


def test_session_rebuild_epoch_preserves_rejection_and_unsaved_selection():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const info={values:{response_bars:2},capabilities:{response_bars:true},state:'stopped',devices:[]};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 controls.update({session_id:'session',coordinator_epoch:0,audio:{state:'stopped'}});
 get('setting-response-bars').value='4';get('setting-response-bars').handlers.input();
 const submit=get('settings-session-form').handlers.submit({preventDefault(){}});
 controls.update({session_id:'session',coordinator_epoch:1,audio:{state:'stopped'}});
 reply({detail:'Session rebuild failed'},false);await submit;
 const rejected={state:get('settings-session-status').dataset.state,error:get('settings-session-status').textContent,disabled:get('settings-session-apply').disabled};
 const poll=controls.load();reply({...info,coordinator_epoch:1});await poll;await flush();
 process.stdout.write(JSON.stringify({rejected,refreshed:{state:get('settings-session-status').dataset.state,error:get('settings-session-status').textContent,disabled:get('settings-session-apply').disabled},selected:get('setting-response-bars').value}));
})();
""")
    expected = {"state": "error", "error": "Session rebuild failed", "disabled": False}
    assert result["rejected"] == expected
    assert result["refreshed"] == expected
    assert result["selected"] == "4"


def test_session_rebuild_epoch_preserves_successful_acknowledgement():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const info={values:{response_bars:2},capabilities:{response_bars:true},state:'stopped',devices:[]};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 controls.update({session_id:'session',coordinator_epoch:0,audio:{state:'stopped'}});
 get('setting-response-bars').value='4';get('setting-response-bars').handlers.input();
 const submit=get('settings-session-form').handlers.submit({preventDefault(){}});
 controls.update({session_id:'session',coordinator_epoch:1,audio:{state:'stopped'}});
 reply({...info,values:{response_bars:4},changed_fields:['response_bars'],effective_from_bar:0,session_restarted:true});await submit;
 process.stdout.write(JSON.stringify({state:get('settings-session-status').dataset.state,status:get('settings-session-status').textContent,disabled:get('settings-session-apply').disabled,selected:Number(get('setting-response-bars').value)}));
})();
""")
    assert result == {"state": "saved", "status": "Saved for next playback.", "disabled": True,
                      "selected": 4}


def test_alternating_chord_acknowledgement_survives_reload_until_matching_playback():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const values={tempo_bpm:60,mode:'alternating',response_bars:2,output_device:null,drum_gain:0.78,vocal_gain:0.8,chord_loop:'minor_cycle',chord_gain:0.3,alignment_policy:'gentle_sparse_r3'};
 const info={values,capabilities:{chord_loop:true,chord_gain:true},state:'running',devices:[],changed_fields:['chord_gain','chord_loop'],effective_from_bar:4,mix_revision:1,coordinator_epoch:0};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 controls.update({session_id:'session',coordinator_epoch:0,audio:{state:'running',timing_evidence:{bar:3,mix:{chord_loop:'off',chord_gain:0.2,vocal_gain:1}}}});
 const pending={state:get('settings-local-status').dataset.state,status:get('settings-local-status').textContent};
 controls.update({audio_output:{live_mix:{applied:{chord_gain:0.3},revision:1,applied_revision:1,supported:true,supported_gains:['chord_gain']}},audio:{state:'running',timing_evidence:{bar:4,mix:{chord_loop:'minor_cycle',chord_gain:0.3,vocal_gain:1}}}});
 const applied=get('settings-local-status').dataset.state;
 const poll=controls.load();reply({...info,values:{...values,chord_gain:0.4},changed_fields:['chord_gain'],mix_revision:2,effective_from_bar:5});await poll;
 process.stdout.write(JSON.stringify({pending,applied,next:{state:get('settings-local-status').dataset.state,status:get('settings-local-status').textContent}}));
})();
""")
    assert result["pending"]["state"] == "pending"
    assert result["pending"]["status"] == "Scheduled for bar 5. Awaiting playback."
    assert result["applied"] == "applied"
    assert result["next"]["state"] == "pending"
    assert result["next"]["status"] == "Accepted locally; awaiting audio output."


def test_different_session_still_discards_an_inflight_settings_reply():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const info={values:{alignment_policy:'gentle_sparse_r3'},capabilities:{alignment_policy:true},state:'stopped',devices:[]};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 controls.update({session_id:'old',coordinator_epoch:0,audio:{state:'stopped'}});
 get('setting-alignment-policy').value='all_onsets_stress_r3';get('setting-alignment-policy').handlers.change();
 controls.update({session_id:'new',coordinator_epoch:0,audio:{state:'stopped'}});
 reply({detail:'obsolete session rejection'},false);await flush();
 const poll=controls.load();reply(info);await poll;await flush();
 process.stdout.write(JSON.stringify({state:get('settings-local-status').dataset.state,status:get('settings-local-status').textContent,selected:get('setting-alignment-policy').value,disabled:get('settings-local-retry').disabled}));
})();
""")
    assert result == {"state": "idle", "status": "No pending settings change.",
                      "selected": "gentle_sparse_r3", "disabled": True}


def test_mix_post_displays_its_revision_instead_of_a_previous_tempo_revision():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const info={values:{tempo_bpm:90,chord_loop:'off'},capabilities:{chord_loop:true},state:'running',devices:[],revision:7,effective_from_bar:8,changed_fields:['tempo_bpm']};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 get('setting-chord-loop').value='minor_cycle';get('setting-chord-loop').handlers.change();
 reply({values:{tempo_bpm:90,chord_loop:'minor_cycle'},mix_revision:2,effective_from_bar:4,changed_fields:['chord_loop']});await flush();
 process.stdout.write(JSON.stringify({state:get('settings-local-status').dataset.state,status:get('settings-local-status').textContent,revision:get('settings-local-status').title}));
})();
""")
    assert result["state"] == "pending"
    assert result["status"] == "Scheduled for bar 5. Awaiting playback."
    assert result["revision"] == "Accepted revision 2"


def test_mix_acknowledgement_identity_is_distinct_from_equal_tempo_revision():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const values={tempo_bpm:90,chord_loop:'off',chord_gain:0.2};
 const info={values,capabilities:{},state:'running',devices:[],coordinator_epoch:0};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 reply({...info,revision:1,effective_from_bar:8,changed_fields:['tempo_bpm']});await flush();
 const poll=controls.load();reply({...info,values:{...values,chord_loop:'minor_cycle'},mix_revision:1,effective_from_bar:4,changed_fields:['chord_loop']});await poll;
 controls.update({audio:{state:'running',timing_evidence:{bar:4,timing:{bpm:60},mix:{chord_loop:'minor_cycle'}}}});
 process.stdout.write(JSON.stringify({state:get('settings-local-status').dataset.state,status:get('settings-local-status').textContent,generation:get('settings-status').dataset.state}));
})();
""")
    assert result == {"state": "applied", "status": "Applied in bar 5.", "generation": "pending"}


def test_device_menu_uses_stable_identity_and_exposes_unavailable_error():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 const id=JSON.stringify({host_api:'Core Audio',name:'DAC'});
 reply({values:{output_device:id},capabilities:{output_device:true},state:'stopped',devices:[{id,label:'Core Audio / DAC',index:4,name:'DAC',host_api:'Core Audio',available:false,error:'Unsupported sample rate'}]});await flush();
 const option=get('setting-output-device').children[1];
 process.stdout.write(JSON.stringify({value:option.value,disabled:option.disabled,label:option.textContent,error:get('settings-device-error').textContent}));
})();
""")
    assert result["value"] == '{"host_api":"Core Audio","name":"DAC"}'
    assert result["disabled"] is True
    assert "Unsupported sample rate" in result["label"]
    assert "Unsupported sample rate" in result["error"]


def test_saved_device_settings_describe_next_playback_without_revision_noise():
    result = run_js("rap-settings.js", """
const ack={changed_fields:['output_device'],values:{output_device:null}};
process.stdout.write(JSON.stringify([StreamMuseRapSettings.settingsStatus(ack,null),StreamMuseRapSettings.settingsStatus({...ack,revision:0},null)]));
""")
    assert result == [
        {"state": "saved", "text": "Saved for next playback."},
        {"state": "saved", "text": "Saved for next playback."},
    ]


def test_settings_timeout_has_ack_margin_and_reconciles_without_retrying_post():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 global.AbortSignal={timeout:ms=>({timeoutMs:ms})};
 const values={tempo_bpm:60,mode:'continuous',response_bars:2,output_device:null,drum_gain:0.78,vocal_gain:0.8};
 const info={values,capabilities:{tempo_bpm:true},state:'running',devices:[]};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 get('setting-tempo-bpm').value='90';get('setting-tempo-bpm').handlers.input();
 const submit=get('settings-form').handlers.submit({preventDefault(){}});
 const request=requests.shift(),timeoutMs=request.options.signal.timeoutMs;
 request.reject(Object.assign(new Error('signal timed out'),{name:'TimeoutError'}));await submit;
 const unknown=get('settings-status').textContent;
 const unchanged=controls.load();reply(info);await unchanged;
 const unresolved=get('settings-status').textContent;
 const poll=controls.load();reply({...info,values:{...values,tempo_bpm:90},revision:1,effective_from_bar:3});await poll;
 const pending=get('settings-status').dataset.state;
 controls.update({audio:{state:'running',timing_evidence:{bar:3,timing:{bpm:90,revision:1}}}});
 process.stdout.write(JSON.stringify({timeoutMs,unknown,unresolved,pending,state:get('settings-status').dataset.state,status:get('settings-status').textContent,disabled:get('settings-apply').disabled,requests:requests.length}));
})();
""")
    assert result["timeoutMs"] == 35000
    assert "outcome is unknown" in result["unknown"]
    assert "before retrying" in result["unknown"]
    assert result["unresolved"] == result["unknown"]
    assert result["pending"] == "pending"
    assert result["state"] == "applied"
    assert result["status"] == "Applied in bar 4."
    assert result["disabled"] is True
    assert result["requests"] == 0


def test_explicit_settings_rejection_is_not_erased_by_matching_discovery():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const info={values:{tempo_bpm:60},capabilities:{tempo_bpm:true},state:'running',devices:[]};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);reply(info);await flush();
 get('setting-tempo-bpm').value='90';get('setting-tempo-bpm').handlers.input();
 const submit=get('settings-form').handlers.submit({preventDefault(){}});
 reply({detail:'Tempo unavailable on this backend'},false);await submit;
 const poll=controls.load();reply({...info,values:{tempo_bpm:90},revision:1,effective_from_bar:3});await poll;
 controls.update({audio:{state:'running',timing_evidence:{bar:3,timing:{bpm:90}}}});
 process.stdout.write(JSON.stringify({state:get('settings-status').dataset.state,error:get('settings-status').textContent}));
})();
""")
    assert result == {"state": "error", "error": "Tempo unavailable on this backend"}


def test_rounded_gain_display_is_not_submitted_with_another_setting():
    html = (STATIC.parent / "index.html").read_text()
    for field in ("drum", "vocal"):
        assert f'id="setting-{field}-gain" type="range" min="0" max="2" step="0.01"' in html
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const values={tempo_bpm:90,mode:'continuous',response_bars:2,output_device:null,drum_gain:0.776895,vocal_gain:0.8};
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 reply({values,capabilities:Object.fromEntries(Object.keys(values).map(k=>[k,true])),state:'stopped',devices:[]});await flush();
 // Native range inputs resolve a raw value to their supported step.
 get('setting-drum-gain').value='0.78';
 get('setting-tempo-bpm').value='120';get('setting-tempo-bpm').handlers.input();
 const display=get('setting-drum-gain-value').textContent;
 const submit=get('settings-form').handlers.submit({preventDefault(){}});
 const payload=JSON.parse(requests[0].options.body);
 reply({values:{...values,tempo_bpm:120},revision:2,effective_from_bar:0});await submit;
 process.stdout.write(JSON.stringify({payload,display,raw:values.drum_gain}));
})();
""")
    assert result == {"payload": {"tempo_bpm": 120}, "display": "0.78x", "raw": 0.776895}


def test_running_software_sink_is_not_reported_as_audible_output():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 reply({values:{tempo_bpm:90},state:'running',capabilities:{tempo_bpm:true},audio_output:{backend:'timed',audible:false,error:null},devices:[]});await flush();
 const silent=get('settings-audio-status').textContent;
 const next=controls.load();reply({values:{tempo_bpm:90},state:'running',capabilities:{tempo_bpm:true},audio_output:{backend:'sounddevice',audible:false,error:'Device disconnected'},devices:[]});await next;
 process.stdout.write(JSON.stringify({silent,error:get('settings-audio-status').textContent}));
})();
""")
    assert "no audible output" in result["silent"].lower()
    assert "timed" in result["silent"]
    assert "Device disconnected" in result["error"]


def test_runtime_error_is_visible_even_without_generation_error_events():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
get('last-error').classList.toggle=()=>{};
vm.runInThisContext(fs.readFileSync(require('path').join(require('path').dirname(process.argv[1]),'rap-demo.js'),'utf8'));
renderQueueAndHealth({runtime_error:'Output device disconnected',audio:{state:'priming'}},{bar:null},{});
const top=get('last-error').textContent;
renderQueueAndHealth({audio:{state:'priming',runtime_error:'Audio startup failed'}},{bar:null},{});
process.stdout.write(JSON.stringify({top,nested:get('last-error').textContent,health:get('model-health').textContent}));
""")
    assert result == {"top": "Output device disconnected", "nested": "Audio startup failed", "health": "ERROR"}


def test_device_recovery_requires_reset_remains_actionable_without_an_error_string():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 const info={values:{tempo_bpm:60},state:'running',capabilities:{},devices:[]};
 reply({...info,audio_output:{backend:'timed',audible:false,error:'Device disconnected',recovery_requires_reset:true}});await flush();
 const failed=get('settings-audio-status').textContent;
 const next=controls.load();reply({...info,state:'stopped',audio_output:{backend:'sounddevice',audible:false,error:null,recovery_requires_reset:true}});await next;
 const stopped=get('settings-audio-status').textContent;
 const recovered=controls.load();reply({...info,audio_output:{backend:'sounddevice',audible:true,error:null,recovery_requires_reset:false}});await recovered;
 process.stdout.write(JSON.stringify({failed,stopped,recovered:get('settings-audio-status').textContent}));
})();
""")
    assert "Device disconnected" in result["failed"]
    for field in ("failed", "stopped"):
        assert "Stop, wait for stopped, Reset, then Start" in result[field]
    assert result["recovered"] == ""


def test_running_state_does_not_reuse_stopped_output_audibility():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 const info={values:{tempo_bpm:60},state:'stopped',capabilities:{},devices:[]};
 reply({...info,audio_output:{backend:'sounddevice',audible:false,error:null}});await flush();
 controls.update({audio:{state:'running'}});
 const checking=get('settings-audio-status').textContent;
 const live=controls.load();reply({...info,state:'running',audio_output:{backend:'sounddevice',audible:true,error:null}});await live;
 const audible=get('settings-audio-status').textContent;
 const silent=controls.load();reply({...info,state:'running',audio_output:{backend:'timed',audible:false,error:null}});await silent;
 const confirmedSilent=get('settings-audio-status').textContent;
 const failed=controls.load();reply({...info,audio_output:{backend:'sounddevice',audible:false,error:'Device disconnected',recovery_requires_reset:true}});await failed;
 controls.update({audio:{state:'running'}});
 process.stdout.write(JSON.stringify({checking,audible,confirmedSilent,error:get('settings-audio-status').textContent}));
})();
""")
    assert result["checking"] == "Checking audio output..."
    assert result["audible"] == ""
    assert "No audible output" in result["confirmedSilent"]
    assert "Device disconnected" in result["error"]
    assert "Stop, wait for stopped, Reset, then Start" in result["error"]


def test_header_tempo_prefers_played_evidence_over_initial_or_requested_values():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
vm.runInThisContext(fs.readFileSync(require('path').join(require('path').dirname(process.argv[1]),'rap-demo.js'),'utf8'));
monitor.session={metadata:{tempo_bpm:60}};
const snapshot={session_metadata:{tempo_bpm:60},settings:{values:{tempo_bpm:120}},audio:{timing_evidence:{bar:8,frame_count:230400,timing:{bpm:50}}}};
renderHeader(snapshot,{bar:8,tick:128,segment:{}});
const played=get('tempo').textContent;
renderHeader({session_metadata:{tempo_bpm:60},settings:snapshot.settings},{bar:null,tick:null,segment:{}});
const initial=get('tempo').textContent;
renderHeader({audio:{timing_evidence:{timing:{bpm:NaN}}}},{bar:null,tick:null,segment:{}});
const invalid=get('tempo').textContent;
monitor.session=null;
renderHeader({},{bar:null,tick:null,segment:{}});
process.stdout.write(JSON.stringify({played,initial,invalid,unknown:get('tempo').textContent}));
""")
    assert result == {"played": "50 BPM", "initial": "60 BPM", "invalid": "60 BPM", "unknown": "--"}


def test_settings_mode_is_forwarded_to_plan_controls_in_the_real_app():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
const directory=require('path').dirname(process.argv[1]);
for(const script of ['rap-plan.js','rap-settings.js','rap-demo.js']) vm.runInThisContext(fs.readFileSync(require('path').join(directory,script),'utf8'));
const replyControl=(url,body)=>{
 const index=requests.findIndex(request=>request.url===url);
 if(index<0) throw new Error(`No request for ${url}`);
 requests.splice(index,1)[0].resolve({ok:true,json:async()=>body});
};
(async()=>{
 planControls.update({audio:{state:'running'}},[],{bar:0},[]);
 const discovery=planControls.load();replyControl('/api/control/plan',{supported:true,templates:[],next_editable_bar:1});await discovery;
 planControls.connection(true);replyControl('/api/control/plan',{supported:true,templates:[],next_editable_bar:1});await flush();
 settingsControls.connection(true);replyControl('/api/control/settings',{values:{mode:'alternating'},state:'running',capabilities:{},devices:[]});await flush();
 process.stdout.write(JSON.stringify({disabled:get('plan-template').disabled,topicDisabled:get('plan-topic').disabled,target:get('plan-target').textContent}));
})();
""")
    assert result == {"disabled": True, "topicDisabled": False, "target": "Topic applies to next capture"}


def test_recoverable_warning_clears_generator_error_after_new_remote_commit():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
vm.runInThisContext(fs.readFileSync(require('path').join(require('path').dirname(process.argv[1]),'rap-demo.js'),'utf8'));
const warning={sequence:20,event_type:'presentation_error',payload:{component:'fallback_catalog',recoverable:true,message:'No prevalidated topic fallback; safety audio uses code.'}};
const remote={sequence:21,state:'committed',renderer_decision:'remote_latest'};
const snapshots=[
 {last_error:warning},
 {last_error:warning,remote_chunk:remote},
 {last_error:warning,remote_chunk:{...remote,sequence:19}},
 {last_error:{...warning,payload:{message:'Writer failed'}},remote_chunk:remote},
 {last_error:{...warning,payload:{...warning.payload,requires_reset:true}},remote_chunk:remote},
 {last_error:warning,remote_chunk:remote,runtime_error:'Device disconnected'},
];
process.stdout.write(JSON.stringify(snapshots.map(snapshot=>{
 renderQueueAndHealth(snapshot,{bar:8},{});
 return {health:get('model-health').textContent,generator:get('generator-state').textContent,message:get('last-error').textContent};
})));
""")
    assert [item["health"] for item in result] == ["WARNING", "READY", "WARNING", "ERROR", "ERROR", "ERROR"]
    assert [item["generator"] for item in result] == ["warning", "ready", "warning", "error", "error", "error"]
    assert "No prevalidated topic fallback" in result[1]["message"]
    assert result[3]["message"] == "Writer failed"
    assert result[-1]["message"] == "Device disconnected"


def test_alternating_success_recovers_health_without_erasing_historical_error():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
vm.runInThisContext(fs.readFileSync(require('path').join(require('path').dirname(process.argv[1]),'rap-demo.js'),'utf8'));
const failure={session_id:'session',sequence:20,event_type:'presentation_error',payload:{component:'alternating_interaction',coordinator_epoch:2,turn_id:'turn-1',message:'response_timeout',incomplete:true}};
const success={session_id:'session',sequence:30,bar:8,frozen:true,coordinator_epoch:2,turn_id:'turn-2',logical_response_bar:0,choice:'remote_latest',source:'moss_lyric_stream',fallback:false};
const base={session_id:'session',coordinator_epoch:2,last_error:failure};
const snapshots=[
 {...base,frozen_bars:{8:success}},
 {...base,last_error:{...failure,event_type:'generation_failed',payload:{coordinator_epoch:2,error_type:'WriterError',error_message:'four-slot writer failed'}},frozen_bars:{8:success}},
 {...base,frozen_bars:{8:{...success,sequence:19}}},
 {...base,frozen_bars:{8:{...success,choice:'flow_echo',source:'flow_echo'}}},
 {...base,frozen_bars:{8:{...success,choice:'drums_idle',source:'drums_idle'}}},
 {...base,frozen_bars:{8:{...success,coordinator_epoch:1}}},
 {...base,frozen_bars:{8:{...success,session_id:'other-session'}}},
 {...base,frozen_bars:{8:{...success,turn_id:'turn-1'}}},
 {...base,frozen_bars:{8:{...success,sequence:null}}},
 {...base,frozen_bars:{8:success},last_error:{...failure,payload:{...failure.payload,requires_reset:true}},latest_batch:{sequence:40}},
 {...base,frozen_bars:{8:success},runtime_error:'Device disconnected'},
 {...base,frozen_bars:{8:success},audio:{runtime_error:'Output failed'}},
];
process.stdout.write(JSON.stringify(snapshots.map(snapshot=>{
 renderQueueAndHealth(snapshot,{bar:9},{});
 return {health:get('model-health').textContent,generator:get('generator-state').textContent,message:get('last-error').textContent,last_error:snapshot.last_error};
})));
""")
    assert [item["health"] for item in result] == ["READY", "READY"] + ["ERROR"] * 10
    assert [item["generator"] for item in result] == ["ready", "ready"] + ["error"] * 10
    assert result[0]["message"] == "response_timeout"
    assert result[1]["message"] == "WriterError: four-slot writer failed"
    assert result[0]["last_error"]["payload"]["message"] == "response_timeout"
    assert result[-2]["message"] == "Device disconnected"
    assert result[-1]["message"] == "Output failed"


def test_alternating_live_commit_recovers_only_before_a_newer_turn_failure():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
vm.runInThisContext(fs.readFileSync(require('path').join(require('path').dirname(process.argv[1]),'rap-demo.js'),'utf8'));
const failure={sequence:20,event_type:'presentation_error',payload:{component:'alternating_interaction',turn_id:'turn-1',message:'response_timeout'}};
monitor.events=[{sequence:30,event_type:'bar_audio_committed',bar:8,payload:{turn_id:'turn-2',choice:'remote_latest',source:'moss_lyric_stream',fallback:false}}];
renderQueueAndHealth({last_error:failure},{bar:8},{});
const recovered=get('model-health').textContent;
monitor.events.push({sequence:40,event_type:'presentation_error',payload:{component:'alternating_interaction',turn_id:'turn-3',message:'new response_timeout'}});
renderQueueAndHealth({last_error:failure},{bar:9},{});
process.stdout.write(JSON.stringify({recovered,latest:get('model-health').textContent,message:get('last-error').textContent}));
""")
    assert result == {"recovered": "READY", "latest": "ERROR", "message": "new response_timeout"}


def test_live_response_recovery_preserves_session_and_epoch_identity():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
vm.runInThisContext(fs.readFileSync(require('path').join(require('path').dirname(process.argv[1]),'rap-demo.js'),'utf8'));
const snapshot={session_id:'session',coordinator_epoch:2,last_error:{session_id:'session',sequence:20,event_type:'generation_failed',payload:{error_message:'Writer failed'}}};
const results=[];
for(const event_type of ['bar_frozen','bar_audio_committed']) {
 for(const [session_id,coordinator_epoch] of [['session',2],['other',2],['session',1]]) {
  monitor.events=[{session_id,sequence:30,event_type,bar:8,payload:{choice:'remote_latest',fallback:false,coordinator_epoch,turn_id:'turn-2'}}];
  renderQueueAndHealth(snapshot,{bar:8},{});
  results.push(get('model-health').textContent);
 }
}
process.stdout.write(JSON.stringify(results));
""")
    assert result == ["READY", "ERROR", "ERROR", "READY", "ERROR", "ERROR"]


def test_echo_and_drums_are_not_labeled_as_generated_lyrics():
    result = run_js("rap-flow.js", DOM_HARNESS + """
document.querySelectorAll=()=>[];
global.document=document;global.location={protocol:'http:',host:'localhost'};
global.addEventListener=()=>{};global.WebSocket=class{};
vm.runInThisContext(fs.readFileSync(require('path').join(require('path').dirname(process.argv[1]),'rap-demo.js'),'utf8'));
const result=['flow_echo','drums_idle','capture_transition','silence'].map(source=>{
 renderDecision({}, {bar:0,tick:null,segment:{}}, {source,choice:source,frozen:true});
 return {badge:get('source-badge').textContent,text:get('selected-line').textContent};
});
process.stdout.write(JSON.stringify(result));
""")
    assert [item["badge"] for item in result] == ["FLOW REPLAY", "DRUMS ONLY", "DRUMS ONLY", "SILENCE"]
    assert all("Waiting for the first" not in item["text"] for item in result)


def test_playback_tick_advances_past_boundary_only_cursor_snapshot():
    result = run_js("rap-flow.js", """
process.stdout.write(JSON.stringify(StreamMuseRapFlow.timingEvidence({slots:[]},{timing_evidence:{frame_count:1600,playback_cursor:{sample:0,kind:'device_or_software_clock'}}},{current_tick:19})));
""")
    assert result["cursor"] == 3


def test_interaction_arm_is_single_flight_and_does_not_record_the_arming_press():
    result = run_js("rap-interaction.js", DOM_HARNESS + """
(async()=>{
 const api=global.StreamMuseRapInteraction;
 if(!api?.mount){process.stdout.write(JSON.stringify(null));return;}
 const controls=api.mount(document);
 controls.settings({values:{mode:'alternating'},state:'running',capabilities:{mode:true}});
 controls.connection(true);
 reply({phase:'DRUMS_IDLE',monotonic_seconds:80,current_tick:2,supported:true});await flush();
 const pad=get('capture-pad');document.activeElement=pad;
 const event={code:'Space',target:pad,preventDefault(){this.defaultPrevented=true;}};
 pad.handlers.keydown({...event});pad.handlers.keydown({...event});
 const urls=requests.map(r=>r.url);
 reply({phase:'ARMED',monotonic_seconds:80,current_tick:2});await flush();
 controls.connection(false);
 process.stdout.write(JSON.stringify({urls,timers:timers.map(t=>t.delay)}));
})().catch(e=>{console.error(e);process.exitCode=1;});
""")
    assert result is not None
    assert result["urls"] == ["/api/control/interaction/arm"]
    assert all(delay >= 250 for delay in result["timers"])


def test_new_epoch_discards_late_arm_acknowledgement():
    result = run_js("rap-interaction.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapInteraction.mount(document);
 if(!controls.update){process.stdout.write(JSON.stringify(null));return;}
 controls.update({session_id:'session',coordinator_epoch:1});
 controls.settings({values:{mode:'alternating'},state:'running'});controls.connection(true);
 reply({phase:'DRUMS_IDLE',monotonic_seconds:80,current_tick:2});await flush();
 const pad=get('capture-pad');document.activeElement=pad;
 pad.handlers.keydown({code:'Space',target:pad,preventDefault(){}});
 controls.update({session_id:'session',coordinator_epoch:2});
 reply({phase:'ARMED',capture_bar:1});await flush();
 process.stdout.write(JSON.stringify({phase:get('interaction-phase').textContent,disabled:pad['aria-disabled']}));
})();
""")
    assert result == {"phase": "DRUMS IDLE", "disabled": "true"}


def test_capture_backpressure_is_visible_without_unbounded_posts():
    result = run_js("rap-interaction.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapInteraction.mount(document);
 controls.settings({values:{mode:'alternating'},state:'running'});controls.connection(true);
 reply({phase:'CAPTURING',monotonic_seconds:80,current_tick:16});await flush();
 const pad=get('capture-pad');document.activeElement=pad;
 for(let i=0;i<17;i++)pad.handlers.keydown({code:'Space',target:pad,preventDefault(){}});
 process.stdout.write(JSON.stringify({count:requests.length,error:get('interaction-status').textContent}));
})();
""")
    assert result["count"] == 16
    assert result["error"] == "Tap not sent: previous taps are still sending."


def test_frequent_monitor_updates_do_not_postpone_interaction_polling():
    result = run_js("rap-interaction.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapInteraction.mount(document);
 const info={values:{mode:'alternating'},state:'running',capabilities:{mode:true}};
 controls.settings(info);controls.connection(true);
 reply({phase:'DRUMS_IDLE',monotonic_seconds:80,current_tick:2});await flush();
 const before=timers.length;
 for(let i=0;i<20;i++) controls.settings({...info});
 process.stdout.write(JSON.stringify({before,after:timers.length}));
})();
""")
    assert result["after"] == result["before"]


def test_server_rejected_tap_remains_visible_and_turn_counts_use_coordinator_fields():
    result = run_js("rap-interaction.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapInteraction.mount(document);
 controls.settings({values:{mode:'alternating',response_bars:2},state:'running',capabilities:{mode:true}});
 controls.connection(true);
 reply({phase:'CAPTURING',monotonic_seconds:80,current_tick:16,delivered_response_bars:1,response_bars:2,capture_bar:1,last_error:null});await flush();
 const pad=get('capture-pad');document.activeElement=pad;
 pad.handlers.keydown({code:'Space',target:pad,preventDefault(){}});
 const request=JSON.parse(requests[0].options.body);
 reply({accepted:false,reason:'outside_capture'});await flush();
 process.stdout.write(JSON.stringify({error:get('interaction-status').textContent,response:get('interaction-response').textContent,request}));
})();
""")
    assert "outside_capture" in result["error"]
    assert result["response"] == "1 / 2"
    assert set(result["request"]) == {"client_time_ms", "server_time_seconds", "uncertainty_ms"}


def test_settings_validate_bounds_and_send_only_changed_fields():
    result = run_js("rap-settings.js", """
const api = global.StreamMuseRapSettings;
const original = {tempo_bpm:90,mode:'continuous',response_bars:2,output_device:null,drum_gain:1,vocal_gain:1};
const errors = [];
for (const change of [{tempo_bpm:29},{tempo_bpm:181},{tempo_bpm:''},{response_bars:1.5},{response_bars:9},{mode:'unknown'},{drum_gain:-1}]) {
  try { api?.makePayload({...original,...change},original); errors.push(false); } catch (_) { errors.push(true); }
}
process.stdout.write(JSON.stringify({errors,payload:api?.makePayload({...original,tempo_bpm:'120',response_bars:'3'},original) || null}));
""")
    assert result["errors"] == [True] * 7
    assert result["payload"] == {"tempo_bpm": 120, "response_bars": 3}


def test_settings_stopped_only_and_unsupported_fields_are_locked():
    result = run_js("rap-settings.js", """
const api = global.StreamMuseRapSettings;
const caps = {tempo_bpm:true,mode:true,output_device:false};
process.stdout.write(JSON.stringify(['stopped','running','priming','stopping'].map(state =>
 ['tempo_bpm','mode','output_device','response_bars'].map(field => api?.fieldEnabled(field,{state,capabilities:caps,application:{tempo_bpm:'bar-boundary',mode:'stopped-only'}},true) ?? null))));
""")
    assert result == [[True, True, False, False], [True, False, False, False], [False] * 4, [False] * 4]


def test_settings_acknowledgement_waits_for_matching_played_evidence():
    result = run_js("rap-settings.js", """
const ack={revision:3,effective_from_bar:4,changed_fields:['tempo_bpm','drum_gain'],values:{tempo_bpm:120,drum_gain:0.7}};
const played={bar:4,timing:{bpm:120,revision:3},mix:{drum_gain:0.7,revision:2}};
const api=StreamMuseRapSettings;
process.stdout.write(JSON.stringify([
api.settingsStatus?.(ack,null)?.state ?? null,
api.settingsStatus?.(ack,{...played,bar:3},{live_mix:{applied:{drum_gain:0.7},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}})?.state ?? null,
api.settingsStatus?.(ack,played,{live_mix:{applied:{drum_gain:0.55},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}})?.state ?? null,
api.settingsStatus?.(ack,played,{live_mix:{applied:{drum_gain:0.7},revision:1,applied_revision:1,supported:true,supported_gains:['drum_gain']}})?.state ?? null,
api.settingsStatus?.({...ack,changed_fields:['mode']},null)?.state ?? null,
]));
""")
    assert result == ["pending", "pending", "pending", "applied", "saved"]


def test_space_requires_capture_focus_and_excludes_repeat_editable_and_modifiers():
    result = run_js("rap-interaction.js", """
const api = global.StreamMuseRapInteraction;
const surface = {tagName:'DIV',isContentEditable:false,closest:()=>null};
const editable = {tagName:'INPUT',isContentEditable:false,closest:()=>null};
const base = {code:'Space',target:surface,repeat:false};
const cases = [base,{...base,repeat:true},{...base,target:editable},{...base,ctrlKey:true},{...base,isComposing:true},{...base,code:'KeyA'},{...base,target:{...surface,isContentEditable:true}}];
process.stdout.write(JSON.stringify({actions:cases.map(e=>api?.captureAction(e,surface,'CAPTURING',true) ?? null),arm:api?.captureAction(base,surface,'DRUMS_IDLE',true) ?? null,wait:api?.captureAction(base,surface,'WAIT_REPLAY',true) ?? null,blurred:api?.captureAction(base,editable,'CAPTURING',true) ?? null}));
""")
    assert result == {"actions": ["tap", None, None, None, None, None, None], "arm": "arm", "wait": None, "blurred": None}


def test_capture_surface_supports_enter_activation_without_key_repeats():
    result = run_js("rap-interaction.js", """
const target={tagName:'DIV',closest:()=>null};
const key={code:'Enter',target};
process.stdout.write(JSON.stringify([StreamMuseRapInteraction.captureAction(key,target,'DRUMS_IDLE',true),StreamMuseRapInteraction.captureAction({...key,repeat:true},target,'DRUMS_IDLE',true)]));
""")
    assert result == ["arm", None]


def test_clock_mapping_uses_midpoint_uncertainty_and_expires():
    result = run_js("rap-interaction.js", """
const api = global.StreamMuseRapInteraction;
const clock = api?.calibrateClock(1000,1040,80);
process.stdout.write(JSON.stringify({clock:clock || null,tap:api?.tapPayload(1100,clock) || null,expired:api?.tapPayload(20000,clock) || null}));
""")
    assert result["clock"] == {"offset_seconds": 78.98, "uncertainty_ms": 20, "sampled_at_ms": 1040}
    assert result["tap"] == {"client_time_ms": 1100, "server_time_seconds": 80.08, "uncertainty_ms": 20}
    assert result["expired"] is None


def test_capture_boundary_uses_calibrated_event_time_between_server_polls():
    result = run_js("rap-interaction.js", """
const api=StreamMuseRapInteraction;
const clock=api.calibrateClock(1000,1040,80);
const info={phase:'ARMED',capture_start_seconds:80.1,capture_end_seconds:82.1};
process.stdout.write(JSON.stringify({
 before:api.phaseAt?.(info,clock,1100) ?? null,
 inside:api.phaseAt?.(info,clock,1130) ?? null,
 after:api.phaseAt?.({...info,phase:'CAPTURING'},clock,3200) ?? null,
 stamp:api.eventTime?.({timeStamp:1125},1130) ?? null,
 epochStamp:api.eventTime?.({timeStamp:1700000000000},1130) ?? null,
}));
""")
    assert result == {"before": "ARMED", "inside": "CAPTURING", "after": "WAIT_REPLAY", "stamp": 1125, "epochStamp": 1130}


def test_settings_recovery_clears_only_transport_errors():
    result = run_js("rap-settings.js", DOM_HARNESS + """
(async()=>{
 const controls=StreamMuseRapSettings.mount(document);controls.connection(true);
 reply({detail:'Temporarily unavailable'},false);await flush();
 const failed=get('settings-status').textContent;
 const next=controls.load();reply({supported:true,values:{tempo_bpm:90},state:'stopped',capabilities:{tempo_bpm:true},devices:[]});await next;
 process.stdout.write(JSON.stringify({failed,recovered:get('settings-status').textContent}));
})();
""")
    assert result["failed"] == "Temporarily unavailable"
    assert "unavailable" not in result["recovered"].lower()


def test_requested_grid_never_substitutes_for_missing_rendered_or_acoustic_evidence():
    result = run_js("rap-flow.js", """
const api = global.StreamMuseRapFlow;
const flow = {slots:[{tick_in_bar:0,target_stress:1},{tick_in_bar:4,target_stress:0.5}]};
process.stdout.write(JSON.stringify(api?.timingEvidence(flow,{}, {current_tick:5}) || null));
""")
    assert result is not None
    assert result["requested"] == [0, 4]
    assert result["estimated"] is None
    assert result["cursor"] == 5
    assert result["acoustic"] is None


def test_rendered_estimates_preserve_sparse_anchor_drift_and_own_identity():
    result = run_js("rap-flow.js", """
const api = global.StreamMuseRapFlow;
const evidence = {template_id:'older',timing_evidence:{frame_count:1600,timing:{revision:2,tick_offsets:Array.from({length:17},(_,i)=>i*100)},rendered_onset_estimates:[{estimated_sample:0,confidence:0.8,method:'source_alignment_warp_estimate'},{estimated_sample:475,confidence:0.7,method:'source_alignment_warp_estimate'}],rendered_estimate_status:'available',playback_cursor:{sample:100,kind:'device_or_software_clock'}}};
process.stdout.write(JSON.stringify(api?.timingEvidence({slots:[{tick_in_bar:0},{tick_in_bar:4}]},evidence,{current_tick:17}) || null));
""")
    assert result is not None
    assert result["estimated"] == [0, 4.75]
    assert result["cursor"] == 1
    assert result["template_id"] == "older"
    assert result["timeline_revision"] == 2
    assert result["confidence"] == 0.7


def test_partial_timing_evidence_does_not_invent_missing_onsets():
    result = run_js("rap-flow.js", """
const evidence={timing_evidence:{frame_count:1600,rendered_estimate_status:'partial',rendered_onset_estimates:[{estimated_sample:125,confidence:0.6,drift_ms:12.5},{estimated_sample:null,unavailable_reason:'no alignment'}]}};
process.stdout.write(JSON.stringify(StreamMuseRapFlow.timingEvidence({slots:[{tick_in_bar:1},{tick_in_bar:4}]},evidence,{})));
""")
    assert result["estimated"] == [1.25]
    assert result["estimate_status"] == "partial"
    assert result["cursor"] is None
    assert result["acoustic"] is None


def test_stress_strip_exposes_weak_medium_strong_and_untested_status():
    result = run_js("rap-flow.js", DOM_HARNESS + """
const flow={name:'UNTESTED rising',template_id:'untested_straight_rising_9',slots:[{tick_in_bar:0,target_stress:0.1},{tick_in_bar:4,target_stress:0.5},{tick_in_bar:8,target_stress:1}]};
global.StreamMuseRapFlow?.renderStress?.(get('strip'),flow,document);
process.stdout.write(JSON.stringify(get('strip').children.map(n=>({label:n['aria-label'],stress:n.dataset.stress}))));
""")
    assert len(result) == 16
    assert [result[index]["stress"] for index in (0, 4, 8)] == ["weak", "medium", "strong"]
    assert "0.10" in result[0]["label"]
    assert "rest" in result[1]["label"]
