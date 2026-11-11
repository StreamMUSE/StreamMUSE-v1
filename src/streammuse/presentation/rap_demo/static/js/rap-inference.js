(function attachInference(root) {
  'use strict';

  const finite = (value) => typeof value === 'number' && Number.isFinite(value);
  const list = (value) => Array.isArray(value) ? value : [];
  const valueOf = (value) => value && typeof value === 'object' ? value.value : value;
  const numberText = (value) => finite(value) ? String(Math.round(value * 100) / 100) : '--';
  const version = (value) => Number.isSafeInteger(value) && value >= 0 ? `v${value}` : '--';
  const textValue = (value) => typeof value === 'string' || finite(value) ? String(value) : '--';
  function formatMetric(metric) {
    const value = valueOf(metric);
    return !finite(value) ? '--' : Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
  }
  function phaseLabel(snapshot) {
    const current = snapshot.current || {}, config = snapshot.config || {};
    if (['WAIT', 'PREPARING'].includes(snapshot.phase) || (snapshot.active_mode === 'steering' && snapshot.phase === 'MODEL')) {
      return `${finite(current.elapsed_ms) && current.elapsed_ms >= 0 ? `${(current.elapsed_ms / 1000).toFixed(2)} s` : '--'} elapsed`;
    }
    if (['USER', 'MODEL'].includes(snapshot.phase)) {
      return `${finite(current.remaining_ticks) && current.remaining_ticks >= 0 && finite(config.ticks_per_beat) && config.ticks_per_beat > 0 ? numberText(current.remaining_ticks / config.ticks_per_beat) : '--'} beats remaining`;
    }
    return '--';
  }
  function layoutSpans(spans, start, end) {
    if (!finite(start) || !finite(end) || end <= start) return [];
    return list(spans).filter((span) => span && finite(span.start_tick) && finite(span.duration_ticks) && span.duration_ticks > 0
      && finite(span.start_tick + span.duration_ticks) && span.start_tick < end && span.start_tick + span.duration_ticks > start)
      .map((span) => ({ ...span, left: 100 * (Math.max(start, span.start_tick) - start) / (end - start),
        width: 100 * (Math.min(end, span.start_tick + span.duration_ticks) - Math.max(start, span.start_tick)) / (end - start) }));
  }
  function positive(value, label) {
    if ((typeof value !== 'string' && typeof value !== 'number') || String(value).trim() === '' || !Number.isFinite(Number(value)) || Number(value) <= 0) {
      throw new Error(`${label} must be a finite positive number.`);
    }
    return Number(value);
  }
  function readConfig(values, config) {
    if (!['cycle', 'steering'].includes(values.mode)) throw new Error('Choose a supported mode.');
    const payload = { mode: values.mode, bpm: positive(values.bpm, 'BPM') };
    if (payload.bpm > 400) throw new Error('BPM must be at most 400.');
    if (values.mode === 'cycle') {
      const multiplier = values.unit === 'bars' ? positive(config.beats_per_bar, 'Reported meter') : 1;
      payload.user_beats = positive(values.user, 'USER duration') * multiplier;
      payload.model_beats = positive(values.model, 'MODEL duration') * multiplier;
      if (!finite(payload.user_beats) || !finite(payload.model_beats)) throw new Error('Duration is too large.');
    }
    return payload;
  }

  function createOwner(env, changed) {
    const uuid = () => env.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const key = 'streammuse-inference-owner';
    let id, channel, timer, releaseLock, ready = false, closed = false;
    try { id = env.sessionStorage?.getItem(key); } catch (_) { /* Storage may be blocked. */ }
    if (!id) id = uuid();
    const instance = uuid();
    const save = () => { try { env.sessionStorage?.setItem(key, id); } catch (_) { /* In-memory ownership remains valid. */ } };
    if (env.navigator?.locks?.request) {
      const acquire = () => env.navigator.locks.request(`streammuse-capture:${id}`, { ifAvailable: true }, (lock) => {
        if (closed) return;
        if (!lock) { id = uuid(); save(); return acquire(); }
        ready = true; changed();
        return new Promise((resolve) => { releaseLock = resolve; });
      }).catch(() => {
        if (!closed) { id = uuid(); save(); ready = true; changed(); }
      });
      save(); acquire();
      return { id: () => id, ready: () => ready && !closed, close() { closed = true; releaseLock?.(); } };
    }
    // A duplicated tab inherits sessionStorage. Existing tabs reject its probe before capture is enabled.
    try {
      if (env.BroadcastChannel) {
        channel = new env.BroadcastChannel('streammuse-inference-owner');
        channel.onmessage = ({ data }) => {
          if (!data || data.owner !== id || data.instance === instance) return;
          if (data.kind === 'probe') channel.postMessage({ kind: 'occupied', owner: id, instance, target: data.instance });
          if (data.kind === 'occupied' && data.target === instance) { id = uuid(); save(); changed(); }
        };
        channel.postMessage({ kind: 'probe', owner: id, instance });
        timer = env.setTimeout(() => { ready = true; changed(); }, 150);
      } else {
        // Without cross-tab arbitration a stored token cannot safely be reclaimed.
        if (env.sessionStorage?.getItem(key)) id = uuid();
        ready = true;
      }
    } catch (_) { id = uuid(); ready = true; }
    save();
    return { id: () => id, ready: () => ready && !closed, close() { closed = true; env.clearTimeout(timer); channel?.close(); } };
  }

  function mount(document, env = root) {
    const get = (id) => document.getElementById(id);
    if (!get('inference-root')) return null;
    let snapshot = null, connected = false, disposed = false, resetPending = false, generation = 0, requestSequence = 0, appliedSequence = 0;
    let pollTimer, heartbeatTimer, latestAt = 0, latestRtt = 0, error = '', connectionError = '', feedback = '', templates = [], audio = null;
    let requestSignature = '', configKey = '', tapCount = 0, captureKey = '', audioBusy = false, audioError = '';
    const dirty = new Set(), busy = new Set(), controllers = new Set();
    const configDirty = () => [...dirty].some((id) => !['topic', 'template'].includes(id));
    const now = () => env.performance.now();
    const set = (id, value) => { get(id).textContent = String(value); };
    const node = (tag, text, className) => { const element = document.createElement(tag); if (text !== undefined) element.textContent = String(text); if (className) element.className = className; return element; };
    const owner = createOwner(env, () => { if (!disposed) render(); });
    const stopped = () => snapshot?.state === 'stopped';
    const inactive = () => ['stopped', 'error'].includes(snapshot?.state)
      && !snapshot?.legacy_active && snapshot?.active_mode !== 'legacy';
    const supported = () => snapshot?.supported === true;
    const selectedMode = () => get('mode-steering').checked ? 'steering' : 'cycle';
    const experimental = () => ['cycle', 'steering'].includes(snapshot?.active_mode);
    const canCapture = () => connected && supported() && experimental() && snapshot.active_mode === 'cycle'
      && snapshot.state === 'running' && snapshot.phase === 'USER' && owner.ready() && snapshot.owner_id === owner.id()
      && snapshot.session_id && Number.isSafeInteger(snapshot.epoch) && snapshot.cycle_id !== null && snapshot.cycle_id !== undefined
      && finite(snapshot.current?.elapsed_ms) && now() - latestAt <= 1500 && !busy.has('stop') && !busy.has('reset');

    async function request(url, body) {
      const controller = env.AbortController ? new env.AbortController() : null;
      if (controller) controllers.add(controller);
      const timeout = controller ? env.setTimeout(() => controller.abort(), 5000) : null;
      try {
        const response = await env.fetch(url, { cache: 'no-store', ...(body === undefined ? {} : {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
        }), ...(controller ? { signal: controller.signal } : {}) });
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : typeof result.error === 'string' ? result.error : `Request rejected (${response.status}).`);
        return result;
      } finally { env.clearTimeout(timeout); controllers.delete(controller); }
    }
    function fillConfig(force = false) {
      const config = snapshot?.config || {};
      const key = JSON.stringify(config);
      if (!force && key === configKey) return;
      configKey = key;
      const fill = (id, value) => { if (force || !dirty.has(id)) get(id).value = value === undefined || value === null ? '' : String(value); };
      if (force || !dirty.has('mode')) {
        get('mode-cycle').checked = config.mode !== 'steering'; get('mode-steering').checked = config.mode === 'steering';
      }
      fill('bpm', config.bpm);
      if (!get('duration-unit').value) get('duration-unit').value = 'bars';
      if (!finite(config.beats_per_bar) || config.beats_per_bar <= 0) get('duration-unit').value = 'beats';
      const unit = get('duration-unit').value === 'bars' ? config.beats_per_bar : 1;
      fill('user-duration', finite(config.user_beats) ? config.user_beats / unit : null);
      fill('model-duration', finite(config.model_beats) ? config.model_beats / unit : null);
      fill('candidate-budget', config.candidate_budget); fill('concurrency', config.concurrency);
      fill('topic', config.topic); fill('template', config.template_id);
    }
    function detailList(target, entries) {
      target.replaceChildren(...entries.map(([label, value, reason]) => {
        const item = node('div'); const term = node('dt', label); const definition = node('dd', value);
        if (reason) definition.title = reason;
        item.append(term, definition); return item;
      }));
    }
    function renderGrid() {
      const current = snapshot?.current || {}, config = snapshot?.config || {};
      const syllables = list(current.syllables).filter(Boolean).slice(0, 256);
      const flow = list(current.flow).filter((point) => point !== null && point !== undefined).slice(0, 256);
      const tpb = config.ticks_per_beat;
      const duration = finite(current.duration_ticks) && current.duration_ticks > 0 ? current.duration_ticks
        : finite(tpb) && tpb > 0 ? (snapshot?.phase === 'USER' ? config.user_beats : config.model_beats) * tpb : null;
      const target = get('performance-grid'); target.replaceChildren();
      if (!finite(duration) || duration <= 0 || !finite(tpb) || tpb <= 0) { target.append(node('span', 'Timing unavailable', 'empty')); return; }
      const explicitStart = finite(current.start_tick) ? current.start_tick
        : snapshot?.active_mode === 'steering' && finite(current.tick) ? Math.floor(current.tick / duration) * duration : 0;
      const end = explicitStart + duration;
      const stride = Math.max(1, Math.ceil(duration / 256));
      for (let tick = explicitStart; tick < end; tick += stride) {
        const beat = (tick - explicitStart) % tpb === 0;
        const marker = node('span', beat ? numberText((tick - explicitStart) / tpb + 1) : '', `grid-tick${beat ? ' beat' : ''}`);
        marker.style.left = `${100 * (tick - explicitStart) / duration}%`; target.append(marker);
      }
      for (const syllable of layoutSpans(syllables, explicitStart, end)) {
        const element = node('span', textValue(syllable.text), 'syllable'); element.style.left = `${syllable.left}%`; element.style.width = `${syllable.width}%`;
        element.title = `${textValue(syllable.text)} / tick ${syllable.start_tick} / ${syllable.duration_ticks} ticks`; target.append(element);
      }
      for (const point of flow) {
        const tick = typeof point === 'number' ? point : point.tick;
        if (!finite(tick) || tick < explicitStart || tick >= end) continue;
        const element = node('i', undefined, 'flow-mark'); element.style.left = `${100 * (tick - explicitStart) / duration}%`; element.title = `Flow onset: tick ${tick}`; target.append(element);
      }
      if (finite(current.tick) && current.tick >= explicitStart && current.tick <= end) {
        const marker = node('i', undefined, 'playhead'); marker.style.left = `${100 * (current.tick - explicitStart) / duration}%`; target.append(marker);
      }
      target.setAttribute('aria-label', `Syllable and flow positions, ticks ${explicitStart} to ${end}`);
    }
    function renderSpans() {
      const spans = list(snapshot?.spans).filter((s) => s && finite(s.start_tick) && finite(s.duration_ticks) && s.duration_ticks > 0 && finite(s.start_tick + s.duration_ticks)).slice(-48);
      const target = get('span-timeline'); target.replaceChildren();
      if (!spans.length) { target.style.height = '48px'; target.append(node('span', 'No prepared spans', 'empty')); set('span-range', 'Ticks --'); set('span-count', ''); return; }
      const start = Math.min(...spans.map((s) => s.start_tick)), end = Math.max(...spans.map((s) => s.start_tick + s.duration_ticks));
      const lanes = [];
      for (const span of layoutSpans(spans, start, end).sort((a, b) => a.start_tick - b.start_tick)) {
        let lane = lanes.findIndex((until) => until <= span.start_tick);
        if (lane < 0) lane = lanes.length;
        lanes[lane] = span.start_tick + span.duration_ticks;
        const state = textValue(span.state);
        const element = node('span', `${span.start_tick} +${span.duration_ticks}`, 'prepared-span');
        element.dataset.state = state; element.style.left = `${span.left}%`; element.style.width = `${span.width}%`; element.style.top = `${lane * 28}px`;
        element.title = `${textValue(span.span_id || span.output_id)} / ${state} / ticks ${span.start_tick} to ${span.start_tick + span.duration_ticks} / ${version(span.input_version)}${finite(span.deadline_remaining_ms) ? ` / deadline in ${formatMetric(span.deadline_remaining_ms)}` : ''}`;
        target.append(element);
      }
      target.style.height = `${Math.max(48, lanes.length * 28)}px`;
      set('span-range', `Ticks ${numberText(start)} to ${numberText(end)}`); set('span-count', `${spans.length} spans`);
    }
    const metricFields = [
      ['metric-response', 'response_start_latency_ms'], ['metric-steering', 'steering_playback_latency_ms'],
      ['metric-prefix', 'playable_prefix_ready_ms'], ['metric-ttft', 'llm_ttft_ms'],
      ['metric-missing', 'missing_duration_ms'], ['metric-stale', 'stale_duration_ms'],
    ];
    function renderMetrics() {
      const metrics = snapshot?.metrics || {};
      for (const [id, field] of metricFields) {
        set(id, formatMetric(metrics[field]));
        get(id).title = metrics[field]?.reason || metrics[field]?.source || (field === 'llm_ttft_ms' && metrics.ttft_unavailable_reason) || (finite(valueOf(metrics[field])) ? 'Runtime-reported measurement' : 'Not measured or not reported');
      }
      const details = [
        ['First valid candidate', 'first_valid_candidate_ms'], ['First vocal onset', 'first_vocal_onset_ms'], ['Ready-to-start overhead', 'ready_to_start_overhead_ms'],
        ['Eligible scheduled duration', 'eligible_duration_ms'],
      ].map(([label, field]) => [label, formatMetric(metrics[field]), metrics[field]?.reason]);
      const latency = metrics.response_latency_summary || {}, coverage = metrics.coverage || {};
      details.push(['Response median', formatMetric(latency.p50_ms)], ['Response p95', formatMetric(latency.p95_ms)],
        ['Response maximum', formatMetric(latency.max_ms)], ['Samples', numberText(latency.count)],
        ['Fallback duration', formatMetric(coverage.fallback_ms)],
        ['On-time audio seconds / wall second', numberText(coverage.on_time_musical_s_per_wall_s), coverage.on_time_scope]);
      for (const [label, field] of [['Missing coverage', 'missing_rate'], ['Stale coverage', 'stale_rate'], ['Fallback coverage', 'fallback_rate']]) {
        const value = coverage[field]; details.push([label, finite(value) && value >= 0 && value <= 1 ? `${numberText(value * 100)}%` : '--']);
      }
      for (const [label, field] of [['Responses started', 'responses'], ['Errors', 'error_count'], ['Canceled', 'canceled_count'], ['Timed out', 'timed_out_count'], ['No valid result', 'no_valid_result_count'], ['Valid candidates / decision', 'mean_valid_candidates']]) details.push([label, numberText(valueOf(metrics[field]))]);
      details.push(['TTFT source', textValue(metrics.llm_ttft_ms?.source || metrics.ttft_source)], ['Metric group', textValue(metrics.group)], ['Warm / cold', textValue(metrics.warm_state)]);
      detailList(get('metric-details'), details);
    }
    function renderRequests() {
      const rows = list(snapshot?.requests).filter(Boolean).slice(-30).reverse();
      const signature = JSON.stringify([snapshot?.session_id, snapshot?.epoch, rows]);
      if (signature === requestSignature) return;
      requestSignature = signature;
      const target = get('requests-body');
      const expanded = new Set();
      for (const row of target.children) if (row.children[0]?.children[0]?.open) expanded.add(row.children[0].children[0].dataset.request);
      target.replaceChildren(); set('request-count', `${rows.length} recent`);
      if (!rows.length) { const row = node('tr'), cell = node('td', 'No requests'); cell.colSpan = 6; row.append(cell); target.append(row); return; }
      for (const request of rows) {
        const row = node('tr'), cell = node('td'), details = node('details'), summary = node('summary', textValue(request.request_id));
        details.dataset.request = String(request.request_id); details.open = expanded.has(details.dataset.request);
        details.append(summary, node('pre', JSON.stringify(request, null, 2))); cell.append(details); row.append(cell);
        const count = (value) => Number.isSafeInteger(value) && value >= 0;
        let yieldText = '--';
        if (count(request.valid_candidates)) {
          const denominator = count(request.evaluated_candidates) ? `${request.evaluated_candidates} evaluated`
            : count(request.requested_candidates) ? `${request.requested_candidates} requested` : '';
          yieldText = `${request.valid_candidates} valid${denominator ? ` / ${denominator}` : ''}`;
        }
        for (const value of [textValue(request.state), version(request.input_version), formatMetric(request.latency_ms), formatMetric(request.ttft_ms), yieldText]) row.append(node('td', value));
        target.append(row);
      }
    }
    function preview() {
      const flow = templates.find((entry) => entry.template_id === get('template').value), target = get('flow-preview');
      target.replaceChildren();
      const duration = flow?.beats_per_bar * flow?.ticks_per_beat;
      if (!finite(duration) || duration <= 0) return;
      for (const slot of list(flow.slots).slice(0, 256)) {
        if (!finite(slot.tick_in_bar) || slot.tick_in_bar < 0 || slot.tick_in_bar >= duration) continue;
        const mark = node('i'); mark.style.left = `${slot.tick_in_bar / duration * 100}%`;
        mark.style.height = `${finite(slot.target_stress) ? 8 + Math.max(0, Math.min(1, slot.target_stress)) * 22 : 20}px`; target.append(mark);
      }
    }
    function renderAudio() {
      const shared = snapshot?.capabilities?.audio_controls_supported === true;
      const reason = !shared ? snapshot?.capabilities?.audio_controls_reason || 'Audio settings unavailable for this renderer' : !audio ? 'Audio settings not loaded' : '';
      for (const [id, field] of [['drum-gain', 'drum_gain'], ['vocal-gain', 'vocal_gain'], ['output-device', 'output_device']]) {
        const enabled = connected && shared && audio?.supported === true && audio?.capabilities?.[field] === true
          && !audioBusy && (field !== 'output_device' || stopped());
        get(id).disabled = !enabled; get(id).title = reason || (!stopped() && field === 'output_device' ? 'Stop to change output device' : !enabled ? 'Unsupported by this audio output' : '');
      }
      set('audio-status', audioError || audio?.device_error || reason);
    }
    function render() {
      if (disposed) return;
      const config = snapshot?.config || {}, capabilities = snapshot?.capabilities || {}, current = snapshot?.current || {};
      const enabled = connected && supported(), editing = enabled && inactive() && busy.size === 0;
      const configurable = list(capabilities.configurable);
      const reason = !connected ? 'Connection unavailable' : !supported() ? capabilities.unsupported_reason || 'Unavailable for this runtime' : !inactive() ? 'Stop to change settings' : '';
      set('session-id', textValue(snapshot?.session_id)); set('active-mode', `Active: ${textValue(snapshot?.active_mode)}`);
      set('connection', connected ? 'Connected' : 'Disconnected'); get('connection').dataset.state = connected ? 'connected' : 'disconnected';
      set('renderer', snapshot?.active_mode === 'legacy' ? 'Rap Lab output' : textValue(capabilities.renderer)); set('backend-name', textValue(capabilities.backend));
      for (const [id, field] of [['mode-cycle', 'mode'], ['mode-steering', 'mode'], ['bpm', 'bpm'], ['user-duration', 'user_beats'], ['model-duration', 'model_beats'], ['candidate-budget', 'candidate_budget'], ['concurrency', 'concurrency']]) {
        get(id).disabled = !editing || !configurable.includes(field); get(id).title = reason || (!configurable.includes(field) ? 'Not configurable by this backend' : '');
      }
      get('duration-unit').disabled = !editing;
      get('save-config').disabled = !editing || !configDirty();
      get('start').disabled = !enabled || !inactive() || busy.size > 0 || !owner.ready() || dirty.size > 0;
      get('start').title = dirty.size ? 'Apply settings before starting' : 'Start a fresh playback epoch';
      get('stop').disabled = busy.has('stop'); get('reset').disabled = busy.has('reset');
      get('cycle-inputs').hidden = selectedMode() !== 'cycle'; get('steering-form').hidden = selectedMode() !== 'steering';
      get('capture-section').hidden = selectedMode() !== 'cycle';
      const steerable = enabled && config.mode === 'steering' && (inactive() || snapshot?.active_mode === 'steering') && !busy.has('steer') && !busy.has('reset');
      get('topic').disabled = !steerable; get('template').disabled = !steerable || !templates.length;
      get('template').title = templates.length ? '' : 'Flow inventory unavailable';
      get('apply-steering').disabled = !steerable || (!get('topic').value.trim() && !get('template').value);
      set('config-status', reason || (configDirty() ? 'Unapplied settings' : ''));
      set('meter', `${numberText(config.beats_per_bar)} beats / bar`);
      set('phase', textValue(snapshot?.phase)); get('phase').dataset.phase = snapshot?.phase || '';
      set('phase-time', snapshot ? phaseLabel(snapshot) : '--'); set('runtime-state', textValue(snapshot?.state));
      set('tick', numberText(current.tick)); set('beat', finite(current.tick) && finite(config.ticks_per_beat) && config.ticks_per_beat > 0 ? numberText(Math.floor(current.tick / config.ticks_per_beat) + 1) : '--');
      set('requested-version', version(current.requested_version)); set('playing-version', version(current.playing_version));
      set('current-text', typeof current.text === 'string' && current.text ? current.text : '--');
      set('scheduled-start', finite(current.scheduled_start_ms) ? `${numberText(current.scheduled_start_ms)} ms` : '--');
      set('observed-start', finite(current.observed_start_ms) ? `${numberText(current.observed_start_ms)} ms` : '--');
      set('timing-domain', `Clock domain ${textValue(current.clock_domain)}`);
      const key = `${snapshot?.session_id}:${snapshot?.epoch}:${snapshot?.cycle_id}`;
      if (key !== captureKey) { tapCount = 0; captureKey = key; }
      get('capture-surface').setAttribute('aria-disabled', String(!canCapture()));
      set('capture-status', !owner.ready() ? 'Checking capture owner' : snapshot?.owner_id && snapshot.owner_id !== owner.id() ? 'Monitor only: another tab owns capture' : canCapture() ? 'Capturing' : 'Not active');
      set('capture-count', finite(current.tap_count) ? `${current.tap_count} taps` : `${tapCount} taps sent`);
      set('phase-status', textValue(current.status) === '--' ? '' : current.status);
      const serverError = typeof snapshot?.error === 'string' ? snapshot.error : snapshot?.error?.message;
      set('error', connectionError || error || serverError || ''); get('error').hidden = !connectionError && !error && !serverError;
      set('feedback', feedback); get('feedback').hidden = !feedback;
      detailList(get('capabilities'), [
        ['Generation unit', textValue(capabilities.generation_unit)], ['Delivery unit', textValue(capabilities.delivery_unit)],
        ['Commit quantum', finite(capabilities.commit_quantum_ticks) ? `${numberText(capabilities.commit_quantum_ticks)} ticks` : '--'],
        ['Startup buffer', textValue(capabilities.startup_buffer_policy)], ['Guard time', formatMetric(capabilities.guard_ms)],
        ['Immutable lookahead', finite(capabilities.max_immutable_lookahead_ticks) ? `${numberText(capabilities.max_immutable_lookahead_ticks)} ticks` : '--'],
        ['Minimum prefix', finite(capabilities.min_prefix_ticks) ? `${numberText(capabilities.min_prefix_ticks)} ticks` : '--'],
      ]);
      set('optimization-reason', ['candidate_budget', 'concurrency'].every((field) => configurable.includes(field)) ? reason : capabilities.unsupported_reason || 'Unsupported controls are fixed by the backend');
      renderGrid(); renderSpans(); renderMetrics(); renderRequests(); renderAudio(); preview();
    }
    async function refresh() {
      if (resetPending || disposed) return;
      const fence = generation, sequence = ++requestSequence, sent = now();
      try {
        let next = await request('/api/inference/state');
        if (disposed || fence !== generation || sequence < appliedSequence) return;
        if (!next || typeof next !== 'object' || !('state' in next)) throw new Error('Invalid inference state response.');
        if (snapshot?.session_id === next.session_id && finite(snapshot?.epoch) && finite(next.epoch) && next.epoch < snapshot.epoch) return;
        const replaced = snapshot && (snapshot.session_id !== next.session_id || snapshot.epoch !== next.epoch
          || snapshot.audio_owner !== next.audio_owner || snapshot.active_mode !== next.active_mode);
        if (replaced) { generation += 1; dirty.clear(); requestSignature = ''; feedback = ''; error = ''; }
        if (next.legacy_active || next.active_mode === 'legacy') {
          next = { ...next, active_mode: 'legacy', phase: 'LEGACY', current: { status: 'Rap Lab owns playback' }, spans: [], metrics: {}, error: null };
        }
        snapshot = next; connected = true; connectionError = ''; appliedSequence = sequence; latestRtt = now() - sent; latestAt = now();
        fillConfig(Boolean(replaced)); render();
      } catch (failure) {
        if (!disposed && fence === generation && sequence >= appliedSequence) { connected = false; connectionError = `Connection lost: ${failure.message}`; render(); }
      }
    }
    async function command(name, body) {
      if (busy.has(name) || disposed) return;
      const fence = ++generation; busy.add(name); error = ''; feedback = '';
      if (name === 'reset') resetPending = true;
      if (name === 'reset' && snapshot) { snapshot = { ...snapshot, current: {}, spans: [], requests: [], metrics: {} }; requestSignature = ''; dirty.clear(); }
      render();
      try {
        await request(`/api/inference/${name}`, body);
        if (name === 'reset') resetPending = false;
        if (disposed || fence !== generation) return;
        if (name === 'config') { for (const id of [...dirty]) if (!['topic', 'template'].includes(id)) dirty.delete(id); configKey = ''; }
        if (name === 'steer') { dirty.delete('topic'); dirty.delete('template'); }
        feedback = name === 'steer' ? 'Steering accepted; playback version pending' : '';
        await refresh();
        if (name === 'start' && snapshot?.active_mode === 'cycle' && snapshot?.owner_id === owner.id()) get('capture-surface').focus();
      } catch (failure) {
        if (name === 'reset') resetPending = false;
        if (!disposed && fence === generation) { error = `${failure.message}${failure.name === 'AbortError' ? ' Outcome unknown; checking shared state.' : ''}`; await refresh(); }
      } finally { busy.delete(name); render(); }
    }
    function configPayload() {
      const payload = readConfig({ mode: selectedMode(), bpm: get('bpm').value, user: get('user-duration').value, model: get('model-duration').value, unit: get('duration-unit').value }, snapshot?.config || {});
      for (const [id, field] of [['candidate-budget', 'candidate_budget'], ['concurrency', 'concurrency']]) if (dirty.has(id) && !get(id).disabled) {
        const value = positive(get(id).value, field); if (!Number.isSafeInteger(value)) throw new Error(`${field} must be an integer.`); payload[field] = value;
      }
      return payload;
    }
    async function tap(event) {
      if (event.repeat || event.isComposing || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey || event.target !== get('capture-surface') || event.code !== 'Space' || !canCapture()) return;
      event.preventDefault();
      const fence = generation, activeCycle = snapshot.cycle_id;
      const elapsed = snapshot.current.elapsed_ms + now() - latestAt + latestRtt / 2;
      const body = { owner_id: owner.id(), session_id: snapshot.session_id, epoch: snapshot.epoch, cycle_id: activeCycle,
        event_id: env.crypto?.randomUUID?.() || `${owner.id()}-${now()}-${++tapCount}`, client_time_ms: now(),
        phase_elapsed_ms: elapsed, clock_uncertainty_ms: latestRtt / 2, key: event.code };
      tapCount += 1; set('capture-count', `${tapCount} taps sent`);
      try { await request('/api/inference/tap', body); }
      catch (failure) { if (!disposed && fence === generation && snapshot.cycle_id === activeCycle) { error = failure.message; render(); } }
    }
    async function heartbeat() {
      if (disposed) return;
      if (connected && owner.ready() && snapshot?.owner_id === owner.id() && snapshot?.active_mode === 'cycle' && snapshot?.state !== 'stopped') {
        const fence = generation;
        try { await request('/api/inference/heartbeat', { owner_id: owner.id() }); }
        catch (failure) { if (fence === generation && !disposed) { connected = false; error = `Capture lease: ${failure.message}`; render(); } }
      }
      if (!disposed) heartbeatTimer = env.setTimeout(heartbeat, 2000);
    }
    async function poll() { await refresh(); if (!disposed) pollTimer = env.setTimeout(poll, document.hidden ? 1000 : 250); }
    async function loadTemplates() {
      try {
        const result = await request('/api/control/plan'); if (disposed) return;
        templates = list(result.templates).filter((entry) => entry && typeof entry.template_id === 'string').slice(0, 256);
        const previous = get('template').value; get('template').replaceChildren(node('option', 'Keep current flow')); get('template').children[0].value = '';
        for (const flow of templates) { const option = node('option', flow.name || flow.template_id); option.value = flow.template_id; get('template').append(option); }
        get('template').value = previous; render();
      } catch (_) { if (!disposed) { templates = []; render(); } }
    }
    async function loadAudio() {
      try {
        const result = await request('/api/control/settings'); if (disposed) return;
        audio = result; audioError = '';
        for (const [id, field] of [['drum-gain', 'drum_gain'], ['vocal-gain', 'vocal_gain']]) {
          const value = result.values?.[field]; get(id).value = finite(value) ? String(value) : ''; set(`${id}-value`, finite(value) ? `${value.toFixed(2)}x` : '--');
        }
        const select = get('output-device'); select.replaceChildren(node('option', 'System default')); select.children[0].value = '';
        for (const device of list(result.devices).filter(Boolean)) { const option = node('option', device.name || device.id); option.value = device.id; option.disabled = device.available === false; select.append(option); }
        const selected = result.values?.output_device;
        const match = list(result.devices).find((device) => device.id === selected || device.name === selected);
        if (selected && !match) { const option = node('option', `${selected} (unavailable)`); option.value = selected; option.disabled = true; select.append(option); }
        select.value = match?.id || selected || ''; renderAudio();
      } catch (failure) { if (!disposed) { audio = null; audioError = failure.message; renderAudio(); } }
    }
    async function updateAudio(id, field) {
      if (get(id).disabled) return;
      let value;
      try {
        value = field === 'output_device' ? get(id).value || null : Number(get(id).value);
        if (field !== 'output_device' && (get(id).value.trim() === '' || !finite(value) || value < 0 || value > 2)) throw new Error('Gain must be between 0 and 2.');
        audioBusy = true; audioError = ''; renderAudio();
        await request(`/api/control/settings/${field === 'output_device' ? 'session' : 'local'}`, { [field]: value });
        await loadAudio();
      } catch (failure) { audioError = failure.message; }
      finally { audioBusy = false; renderAudio(); }
    }

    get('start').addEventListener('click', () => !get('start').disabled && command('start', { owner_id: owner.id() }));
    get('stop').addEventListener('click', () => command('stop', {}));
    get('reset').addEventListener('click', () => command('reset', {}));
    get('save-config').addEventListener('click', async () => { if (get('save-config').disabled) return; try { await command('config', configPayload()); } catch (failure) { error = failure.message; render(); } });
    for (const id of ['bpm', 'user-duration', 'model-duration', 'candidate-budget', 'concurrency']) get(id).addEventListener('input', () => { dirty.add(id); render(); });
    for (const id of ['mode-cycle', 'mode-steering']) get(id).addEventListener('change', () => { dirty.add('mode'); render(); });
    get('duration-unit').addEventListener('change', () => { dirty.delete('user-duration'); dirty.delete('model-duration'); configKey = ''; fillConfig(); render(); });
    get('topic').addEventListener('input', () => { dirty.add('topic'); render(); });
    get('template').addEventListener('change', () => { dirty.add('template'); render(); });
    get('steering-form').addEventListener('submit', (event) => {
      event.preventDefault(); if (get('apply-steering').disabled) return;
      const payload = {};
      if (get('topic').value.trim()) payload.topic = get('topic').value.trim();
      if (get('template').value) payload.template_id = get('template').value;
      return command('steer', payload);
    });
    get('capture-surface').addEventListener('keydown', tap);
    get('audio-settings').addEventListener('toggle', () => { if (get('audio-settings').open) return loadAudio(); });
    for (const [id, field] of [['drum-gain', 'drum_gain'], ['vocal-gain', 'vocal_gain'], ['output-device', 'output_device']]) get(id).addEventListener('change', () => updateAudio(id, field));
    function close() {
      disposed = true; generation += 1; env.clearTimeout(pollTimer); env.clearTimeout(heartbeatTimer); owner.close();
      for (const controller of controllers) controller.abort();
    }
    env.addEventListener('pagehide', close);
    env.addEventListener('pageshow', (event) => { if (event.persisted) env.location?.reload(); });
    poll(); loadTemplates(); heartbeatTimer = env.setTimeout(heartbeat, 2000);
    return { refresh };
  }
  root.StreamMuseInference = { finite, formatMetric, phaseLabel, layoutSpans, readConfig, createOwner, mount };
  if (root.document) mount(root.document, root);
})(typeof window !== 'undefined' ? window : globalThis);
