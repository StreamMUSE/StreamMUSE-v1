(function () {
    let trace = [];
    let comparison = null;
    let manifest = null;
    let stageRows = [];
    let selectedIndex = 0;
    let activeTab = 'meaning';
    let activeFilter = 'all';
    const artifactCache = new Map();

    const timeline = document.getElementById('timeline-list');
    const summary = document.getElementById('run-summary');
    const pageTitle = document.getElementById('page-title');
    const resultTitle = document.getElementById('result-title');
    const resultCopy = document.getElementById('result-copy');
    const runnerCount = document.getElementById('runner-count');
    const stageProgress = document.getElementById('stage-progress');
    const runFactTitle = document.getElementById('run-fact-title');
    const runFacts = document.getElementById('run-facts');
    const pipelineMap = document.getElementById('pipeline-map');
    const selectedStageTitle = document.getElementById('selected-stage-title');
    const selectedStageStatus = document.getElementById('selected-stage-status');
    const selectedStageExplanation = document.getElementById('selected-stage-explanation');
    const offlineHash = document.getElementById('offline-hash');
    const simHash = document.getElementById('sim-hash');
    const offlineSummary = document.getElementById('offline-summary');
    const simSummary = document.getElementById('sim-summary');
    const stageData = document.getElementById('stage-data');

    const STAGE_HELP = {
        input_midi_loaded: 'The MIDI file was read from disk. If this differs, the two paths are not even starting with the same source.',
        normalized_melody_events: 'MIDI notes were converted into the canonical event stream used by StreamMUSE.',
        prompt_window_selected: 'The initial melody window was selected for the prompt model.',
        append_chunks_selected: 'The melody after the prompt window was prepared as continuation input.',
        continuation_context_built: 'The Lekai prompt-continuation backend was created and its runtime mode was recorded.',
        prompt_start_submitted: 'The prompt stage was submitted to the backend.',
        prompt_tokenization: 'Melody prompt events became model tokens.',
        prompt_model_generate: 'The prompt model generated initial accompaniment tokens or fallback output.',
        prompt_decode_events: 'Prompt output was decoded back into symbolic accompaniment events.',
        continuation_tokenization: 'Continuation context became model tokens for later generation.',
        continuation_model_generate: 'The continuation model generated follow-up accompaniment output.',
        continuation_decode_events: 'Continuation output was decoded into symbolic events.',
        scheduler_status: 'The scheduler catch-up state was sampled: phase, readiness, and generated history size.',
        playable_history: 'The backend returned accompaniment that is ready to be scheduled.',
        audible_scheduling: 'Playable model events were transformed into events that would be scheduled for playback.',
        output_midi_render: 'The final combined MIDI artifact was rendered.'
    };

    const STAGE_META = {
        input_midi_loaded: {group: 'Input', type: 'summary', cause: 'input parsing', compares: 'MIDI load summary'},
        normalized_melody_events: {group: 'Input', type: 'events', cause: 'MIDI-to-event conversion', compares: 'canonical melody events'},
        prompt_window_selected: {group: 'Prompt setup', type: 'events', cause: 'prompt windowing', compares: 'prompt melody events'},
        append_chunks_selected: {group: 'Prompt setup', type: 'events', cause: 'continuation chunking', compares: 'append melody events'},
        continuation_context_built: {group: 'Prompt setup', type: 'summary', cause: 'runtime setup', compares: 'backend runtime info'},
        prompt_start_submitted: {group: 'Prompt setup', type: 'summary', cause: 'prompt request contract', compares: 'prompt start request/status'},
        prompt_tokenization: {group: 'Model generation', type: 'tokens', cause: 'tokenizer or prompt context', compares: 'prompt token payload'},
        prompt_model_generate: {group: 'Model generation', type: 'tokens', cause: 'model, fallback, RNG, or runtime', compares: 'prompt model output'},
        prompt_decode_events: {group: 'Model generation', type: 'events', cause: 'token-to-event decoding', compares: 'decoded prompt accompaniment events'},
        continuation_tokenization: {group: 'Continuation', type: 'tokens', cause: 'continuation tokenizer/context', compares: 'continuation token payload'},
        continuation_model_generate: {group: 'Continuation', type: 'tokens', cause: 'continuation model, fallback, RNG, or runtime', compares: 'continuation model output'},
        continuation_decode_events: {group: 'Continuation', type: 'events', cause: 'continuation decoding', compares: 'decoded continuation events'},
        scheduler_status: {group: 'Scheduling', type: 'scheduler', cause: 'catch-up or readiness scheduling', compares: 'scheduler state'},
        playable_history: {group: 'Scheduling', type: 'events', cause: 'playable history selection', compares: 'playable accompaniment events'},
        audible_scheduling: {group: 'Scheduling', type: 'events', cause: 'local playback scheduling', compares: 'scheduled audible events'},
        output_midi_render: {group: 'Output', type: 'midi', cause: 'MIDI rendering', compares: 'rendered MIDI artifact'}
    };
    const PIPELINE_GROUP_ORDER = ['Input', 'Prompt setup', 'Model generation', 'Continuation', 'Scheduling', 'Output', 'Other'];

    function json(value) {
        return JSON.stringify(value, null, 2);
    }

    function shortHash(value) {
        if (!value) return '--';
        return String(value).slice(0, 10);
    }

    function titleize(stage) {
        return String(stage || '').replace(/[_-]/g, ' ');
    }

    async function loadTrace() {
        try {
            const response = await fetch('/api/trace');
            if (!response.ok) {
                throw new Error(await response.text());
            }
            const payload = await response.json();
            trace = payload.trace || [];
            comparison = payload.comparison;
            manifest = payload.manifest || {};
            stageRows = buildStageRows();
            render();
        } catch (error) {
            pageTitle.textContent = 'Trace failed to load';
            summary.textContent = 'The viewer could not read this replay directory.';
            stageData.innerHTML = `<div class="empty-state">${escapeHtml(String(error))}</div>`;
        }
    }

    function buildStageRows() {
        if (comparison && Array.isArray(comparison.stages)) {
            return comparison.stages.map((stage) => ({
                stage: stage.stage,
                match: Boolean(stage.match),
                leftHash: stage.left_hash,
                rightHash: stage.right_hash,
                leftSummary: stage.left_summary || {},
                rightSummary: stage.right_summary || {},
                leftRefs: stage.left_refs || [],
                rightRefs: stage.right_refs || [],
                offlineEvent: findEvent(stage.stage, 'offline_direct'),
                simEvent: findEvent(stage.stage, 'realtime_sim')
            }));
        }
        const seen = [];
        trace.forEach((event) => {
            if (!seen.includes(event.stage)) seen.push(event.stage);
        });
        return seen.map((stage) => ({
            stage,
            match: null,
            leftHash: null,
            rightHash: null,
            leftSummary: {},
            rightSummary: {},
            leftRefs: [],
            rightRefs: [],
            offlineEvent: findEvent(stage, 'offline_direct') || trace.find((event) => event.stage === stage),
            simEvent: findEvent(stage, 'realtime_sim')
        }));
    }

    function findEvent(stage, runner) {
        const matches = trace.filter((event) => event.stage === stage && event.runner_kind === runner);
        return matches[matches.length - 1] || null;
    }

    function render() {
        const scenario = manifest.scenario || 'StreamMUSE replay';
        pageTitle.textContent = 'Did realtime playback match the offline reference?';
        summary.textContent = `${titleize(scenario)}; ${trace.length} trace events loaded`;
        renderOverview();
        renderPipeline();
        renderTimeline();
        const mismatchIndex = comparison && comparison.first_mismatch_stage
            ? stageRows.findIndex((row) => row.stage === comparison.first_mismatch_stage)
            : 0;
        selectStage(Math.max(0, mismatchIndex));
    }

    function renderOverview() {
        const mismatch = comparison && comparison.first_mismatch_stage;
        const matched = comparison ? comparison.matching_stage_count : 0;
        const count = comparison ? comparison.stage_count : stageRows.length;
        resultTitle.textContent = mismatch ? 'First difference found' : 'Replay contract holds';
        resultCopy.textContent = mismatch
            ? `First mismatch at ${titleize(mismatch)}. Start there.`
            : `The offline reference path and realtime-style simulation matched at all ${count} comparable checkpoints for this controlled run.`;
        document.querySelector('.status-card.primary').classList.toggle('bad', Boolean(mismatch));
        runnerCount.textContent = '2 paths';
        stageProgress.textContent = `${matched}/${count}`;
        runFactTitle.textContent = `${trace.length} events`;
        runFacts.textContent = buildRunFacts(count);
        const mismatchButton = document.getElementById('first-mismatch');
        mismatchButton.textContent = mismatch ? 'First difference' : 'No difference found';
        mismatchButton.disabled = !mismatch;
    }

    function buildRunFacts(count) {
        const inputEvent = findEvent('input_midi_loaded', 'offline_direct') || findEvent('input_midi_loaded', 'realtime_sim');
        const inputName = inputEvent && inputEvent.output_refs && inputEvent.output_refs[0]
            ? 'input artifact recorded'
            : 'input recorded';
        return `${inputName}; ${count} checkpoints; offline reference vs realtime simulation.`;
    }

    function renderPipeline() {
        const groups = pipelineGroups();
        pipelineMap.innerHTML = groups.map((group, index) => {
            const status = group.hasDiff ? 'diff' : (group.hasUnknown ? 'recorded' : 'match');
            const targetIndex = group.firstDiffIndex >= 0 ? group.firstDiffIndex : group.firstIndex;
            return `
                <button class="pipeline-node ${status}${group.containsSelected ? ' selected' : ''}" type="button" data-index="${targetIndex}">
                    <span>${escapeHtml(group.name)}</span>
                    <strong>${escapeHtml(group.hasDiff ? 'Needs inspection' : `${group.matchCount}/${group.rows.length} matched`)}</strong>
                    <small>${escapeHtml(group.summary)}</small>
                </button>
                ${index < groups.length - 1 ? '<span class="flow-arrow" aria-hidden="true">&rarr;</span>' : ''}
            `;
        }).join('');
        pipelineMap.querySelectorAll('.pipeline-node').forEach((button) => {
            button.addEventListener('click', () => selectStage(Number(button.dataset.index)));
        });
    }

    function pipelineGroups() {
        const groups = [];
        stageRows.forEach((row, index) => {
            const meta = stageMeta(row.stage);
            let group = groups.find((candidate) => candidate.name === meta.group);
            if (!group) {
                group = {
                    name: meta.group,
                    rows: [],
                    firstIndex: index,
                    firstDiffIndex: -1,
                    matchCount: 0,
                    hasDiff: false,
                    hasUnknown: false,
                    containsSelected: false,
                    summary: ''
                };
                groups.push(group);
            }
            group.rows.push(row);
            group.matchCount += row.match === true ? 1 : 0;
            group.hasDiff = group.hasDiff || row.match === false;
            group.hasUnknown = group.hasUnknown || row.match === null;
            group.containsSelected = group.containsSelected || index === selectedIndex;
            if (row.match === false && group.firstDiffIndex < 0) group.firstDiffIndex = index;
        });
        groups.forEach((group) => {
            const stages = group.rows.map((row) => titleize(row.stage));
            group.summary = stages.length > 2 ? `${stages[0]}, ${stages[1]}, ...` : stages.join(', ');
        });
        return groups.sort((left, right) => (
            PIPELINE_GROUP_ORDER.indexOf(left.name) - PIPELINE_GROUP_ORDER.indexOf(right.name)
        ));
    }

    function renderTimeline() {
        timeline.innerHTML = '';
        let lastGroup = '';
        visibleStageRows().forEach(({row, index}) => {
            const meta = stageMeta(row.stage);
            if (meta.group !== lastGroup) {
                const heading = document.createElement('li');
                heading.className = 'stage-group';
                heading.textContent = meta.group;
                timeline.appendChild(heading);
                lastGroup = meta.group;
            }
            const item = document.createElement('li');
            item.className = `timeline-item${index === selectedIndex ? ' selected' : ''}`;
            const status = row.match === null ? 'recorded' : (row.match ? 'match' : 'diff');
            item.innerHTML = `
                <button class="stage-button" type="button" ${index === selectedIndex ? 'aria-current="step"' : ''}>
                    <span class="stage-title">${String(index + 1).padStart(2, '0')} / ${stageRows.length} ${titleize(row.stage)}</span>
                    <span class="stage-meta">
                        <span class="dot ${status}"></span>
                        ${status === 'diff' ? 'different' : status}
                        <span class="type-chip">${meta.type}</span>
                    </span>
                    ${index === selectedIndex ? '<span class="selected-label">Selected</span>' : ''}
                </button>
            `;
            item.querySelector('button').addEventListener('click', () => selectStage(index));
            timeline.appendChild(item);
        });
    }

    function visibleStageRows() {
        return stageRows
            .map((row, index) => ({row, index}))
            .filter(({row}) => {
                const meta = stageMeta(row.stage);
                if (activeFilter === 'all') return true;
                if (activeFilter === 'diff') return row.match === false;
                return meta.type === activeFilter;
            });
    }

    function selectStage(index) {
        selectedIndex = Math.max(0, Math.min(index, stageRows.length - 1));
        document.querySelectorAll('.timeline-item').forEach((item, itemIndex) => {
            item.classList.toggle('selected', itemIndex === selectedIndex);
        });
        const row = stageRows[selectedIndex];
        if (!row) return;

        selectedStageTitle.textContent = titleize(row.stage);
        selectedStageExplanation.textContent = STAGE_HELP[row.stage] || 'Recorded checkpoint from the replay pipeline.';
        selectedStageStatus.textContent = row.match === null ? 'Recorded' : (row.match ? 'Match' : 'Different');
        selectedStageStatus.className = `badge ${row.match === false ? 'bad' : 'ok'}`;
        document.getElementById('prev-event').disabled = selectedIndex === 0;
        document.getElementById('next-event').disabled = selectedIndex >= stageRows.length - 1;
        offlineHash.textContent = shortHash(row.leftHash || (row.offlineEvent && row.offlineEvent.output_hash));
        simHash.textContent = shortHash(row.rightHash || (row.simEvent && row.simEvent.output_hash));
        offlineSummary.innerHTML = renderSummary(row.leftSummary, row.leftRefs, row.offlineEvent);
        simSummary.innerHTML = renderSummary(row.rightSummary, row.rightRefs, row.simEvent);
        document.querySelectorAll('.timeline-item').forEach((item) => item.classList.remove('selected'));
        renderPipeline();
        renderTimeline();
        renderDiagnosis(row);
        setTab(defaultTabForStage(row.stage));
        renderDetail(row);
    }

    function setTab(tabName) {
        activeTab = tabName;
        document.querySelectorAll('.tab').forEach((tab) => {
            const selected = tab.dataset.tab === activeTab;
            tab.classList.toggle('active', selected);
            tab.setAttribute('aria-selected', String(selected));
            tab.tabIndex = selected ? 0 : -1;
        });
        stageData.setAttribute('aria-labelledby', `tab-${activeTab}`);
    }

    function defaultTabForStage(stage) {
        const type = stageMeta(stage).type;
        if (type === 'events') return 'events';
        if (type === 'tokens') return 'tokens';
        if (type === 'scheduler') return 'scheduler';
        return 'meaning';
    }

    function renderDiagnosis(row) {
        const meta = stageMeta(row.stage);
        const previous = previousMatchedStage(selectedIndex);
        const diffSummary = row.diff_summary || {};
        const statusText = row.match === false
            ? `First inspect ${meta.cause}.`
            : `This stage is unlikely to explain an offline/realtime difference.`;
        document.getElementById('diagnosis-panel').innerHTML = `
            <div class="diagnosis-grid">
                <div><span>Compared value</span><strong>${escapeHtml(meta.compares)}</strong></div>
                <div><span>Likely subsystem</span><strong>${escapeHtml(meta.cause)}</strong></div>
                <div><span>Last matched boundary</span><strong>${escapeHtml(previous ? titleize(previous.stage) : 'Start of replay')}</strong></div>
                <div><span>Next action</span><strong>${escapeHtml(statusText)}</strong></div>
            </div>
            ${renderDiffMessage(diffSummary)}
            <p class="hash-help">${row.match === false ? 'Different hashes mean this stage produced different canonical output.' : 'Same hash means both paths produced the same canonical output.'}</p>
        `;
    }

    function renderDiffMessage(diffSummary) {
        if (!diffSummary || !diffSummary.message) return '';
        return `
            <div class="diff-message ${diffSummary.status === 'different' ? 'bad' : ''}">
                <span>Diff summary</span>
                <strong>${escapeHtml(diffSummary.message)}</strong>
            </div>
        `;
    }

    function previousMatchedStage(index) {
        for (let i = index - 1; i >= 0; i -= 1) {
            if (stageRows[i].match === true) return stageRows[i];
        }
        return null;
    }

    function renderSummary(summaryPayload, refs, event) {
        const summary = summaryPayload && Object.keys(summaryPayload).length ? summaryPayload : ((event && event.summary) || {});
        const rows = [];
        if (summary.event_count !== undefined) rows.push(['Events', summary.event_count]);
        if (summary.first_mismatch !== undefined) rows.push(['First diff', summary.first_mismatch || 'none']);
        if (summary.kind !== undefined) rows.push(['Payload', summary.kind]);
        if (event && event.logical_tick !== undefined && event.logical_tick !== null) rows.push(['Tick', event.logical_tick]);
        if (event && event.status) rows.push(['Status', event.status]);
        if (!rows.length) rows.push(['Summary', 'Recorded']);
        const refHtml = (refs && refs.length)
            ? `<div class="artifact-list">${refs.map((ref) => `<a href="/artifact/${encodeURIComponent(ref.path).replaceAll('%2F', '/')}" target="_blank">${escapeHtml(ref.kind)}</a>`).join('')}</div>`
            : '';
        return `
            <dl class="summary-list" aria-label="Runner summary">
                ${rows.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(String(value))}</dd></div>`).join('')}
            </dl>
            ${refHtml}
        `;
    }

    async function renderDetail(row) {
        if (activeTab === 'meaning') {
            renderMeaning(row);
        } else if (activeTab === 'events') {
            await renderEvents(row);
        } else if (activeTab === 'tokens') {
            await renderTokens(row);
        } else if (activeTab === 'scheduler') {
            renderScheduler(row);
        } else {
            renderRaw(row);
        }
    }

    function renderMeaning(row) {
        const mismatchText = row.match === false
            ? 'This is the first place to inspect if it is the earliest red stage. Compare the summaries and artifacts on each side.'
            : 'This checkpoint matched, so the two paths agree at this boundary.';
        stageData.innerHTML = `
            <div class="meaning-card">
                <h3>${escapeHtml(titleize(row.stage))}</h3>
                <p>${escapeHtml(STAGE_HELP[row.stage] || 'Recorded checkpoint from the replay pipeline.')}</p>
                <p>${escapeHtml(mismatchText)}</p>
                <p><strong>Usual cause:</strong> ${escapeHtml(stageMeta(row.stage).cause)}.</p>
            </div>
        `;
    }

    async function renderEvents(row) {
        const left = await loadFirstArtifact(row.leftRefs, 'events');
        const right = await loadFirstArtifact(row.rightRefs, 'events');
        if (!left && !right) {
            stageData.innerHTML = '<div class="empty-state">No event table for this stage. This checkpoint records summary data instead; use Stage guide, Scheduler, or Raw.</div>';
            return;
        }
        stageData.innerHTML = `
            <div class="table-grid">
                ${renderEventTable('Offline reference', left)}
                ${renderEventTable('Realtime simulation', right)}
            </div>
        `;
    }

    function renderEventTable(title, events) {
        const rows = Array.isArray(events) ? events.slice(0, 24) : [];
        if (!rows.length) {
            return `<section><h3>${escapeHtml(title)}</h3><p class="empty-state">No events recorded.</p></section>`;
        }
        return `
            <section>
                <h3>${escapeHtml(title)}</h3>
                <table>
                    <thead><tr><th>Tick</th><th>Pitch</th><th>Type</th><th>Velocity</th></tr></thead>
                    <tbody>
                        ${rows.map((event) => `
                            <tr>
                                <td>${escapeHtml(String(event.tick))}</td>
                                <td>${escapeHtml(String(event.pitch))}</td>
                                <td>${escapeHtml(String(event.type))}</td>
                                <td>${escapeHtml(String(event.velocity))}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
                <p class="table-note">Showing ${rows.length} of ${Array.isArray(events) ? events.length : 0} events.</p>
            </section>
        `;
    }

    async function renderTokens(row) {
        const left = await loadFirstArtifact(row.leftRefs, 'summary');
        const right = await loadFirstArtifact(row.rightRefs, 'summary');
        const tokenLike = row.stage.includes('token') || row.stage.includes('generate');
        if (!tokenLike) {
            stageData.innerHTML = '<div class="empty-state">This is not a token/model stage. Use Stage guide or Events for this checkpoint.</div>';
            return;
        }
        stageData.innerHTML = `
            <div class="table-grid">
                <section><h3>Offline reference</h3>${renderTokenBlock(left)}</section>
                <section><h3>Realtime simulation</h3>${renderTokenBlock(right)}</section>
            </div>
        `;
    }

    function renderTokenBlock(payload) {
        const compact = compactTokenPayload(payload);
        return `<pre>${escapeHtml(json(compact))}</pre>`;
    }

    function compactTokenPayload(payload) {
        if (!payload) return {available: false};
        if (Array.isArray(payload)) return {items: payload.length, first: payload.slice(0, 2)};
        const compact = {};
        ['available', 'prompt_token_ids', 'generated_token_ids', 'new_token_ids', 'prompt_tokens', 'diagnostics', 'reason'].forEach((key) => {
            if (payload[key] !== undefined) compact[key] = Array.isArray(payload[key]) ? payload[key].slice(0, 48) : payload[key];
        });
        return Object.keys(compact).length ? compact : payload;
    }

    async function renderScheduler(row) {
        const left = await loadFirstArtifact(row.leftRefs, 'summary');
        const right = await loadFirstArtifact(row.rightRefs, 'summary');
        stageData.innerHTML = `
            <div class="table-grid">
                <section><h3>Offline reference</h3><pre>${escapeHtml(json(left || row.leftSummary || {}))}</pre></section>
                <section><h3>Realtime simulation</h3><pre>${escapeHtml(json(right || {}))}</pre></section>
            </div>
        `;
    }

    function renderRaw(row) {
        stageData.innerHTML = `
            <details open>
                <summary>Comparison row</summary>
                <pre>${escapeHtml(json({
                    row,
                    offline_event: row.offlineEvent,
                    realtime_event: row.simEvent,
                    manifest
                }))}</pre>
            </details>
        `;
    }

    async function loadFirstArtifact(refs, kind) {
        const ref = (refs || []).find((candidate) => candidate.kind === kind) || (refs || [])[0];
        if (!ref || !ref.path) return null;
        if (artifactCache.has(ref.path)) return artifactCache.get(ref.path);
        const response = await fetch(`/artifact/${ref.path}`);
        if (!response.ok) return null;
        const payload = await response.json();
        artifactCache.set(ref.path, payload);
        return payload;
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;')
            .replaceAll('"', '&quot;')
            .replaceAll("'", '&#039;');
    }

    document.getElementById('prev-event').addEventListener('click', () => {
        selectStage(selectedIndex - 1);
    });

    document.getElementById('next-event').addEventListener('click', () => {
        selectStage(selectedIndex + 1);
    });

    document.getElementById('first-mismatch').addEventListener('click', () => {
        if (!comparison || !comparison.first_mismatch_stage) {
            return;
        }
        const index = stageRows.findIndex((row) => row.stage === comparison.first_mismatch_stage);
        if (index >= 0) selectStage(index);
    });

    document.querySelectorAll('.tab').forEach((button) => {
        button.setAttribute('role', 'tab');
        button.id = `tab-${button.dataset.tab}`;
        button.setAttribute('aria-controls', 'stage-data');
        button.addEventListener('click', () => {
            setTab(button.dataset.tab);
            renderDetail(stageRows[selectedIndex]);
        });
        button.addEventListener('keydown', (event) => {
            const tabs = Array.from(document.querySelectorAll('.tab'));
            const current = tabs.indexOf(button);
            let next = current;
            if (event.key === 'ArrowRight') next = (current + 1) % tabs.length;
            else if (event.key === 'ArrowLeft') next = (current - 1 + tabs.length) % tabs.length;
            else if (event.key === 'Home') next = 0;
            else if (event.key === 'End') next = tabs.length - 1;
            else return;
            event.preventDefault();
            tabs[next].focus();
            tabs[next].click();
        });
    });

    document.querySelectorAll('.filter').forEach((button) => {
        button.addEventListener('click', () => {
            activeFilter = button.dataset.filter;
            document.querySelectorAll('.filter').forEach((filter) => filter.classList.toggle('active', filter === button));
            renderTimeline();
        });
    });

    stageData.setAttribute('role', 'tabpanel');
    stageData.setAttribute('aria-live', 'polite');

    function stageMeta(stage) {
        return STAGE_META[stage] || {group: 'Other', type: 'summary', cause: 'recorded checkpoint', compares: 'summary payload'};
    }

    loadTrace();
})();
