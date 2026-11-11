(function attachInferenceStatus(root) {
  'use strict';

  function mount(document, env = root) {
    const status = document.getElementById('inference-mode-status');
    if (!status) return null;
    let disposed = false, timer, controller, runtimeKey = null;
    async function poll() {
      controller = env.AbortController ? new env.AbortController() : null;
      const timeout = controller ? env.setTimeout(() => controller?.abort(), 5000) : null;
      try {
        const response = await env.fetch('/api/inference/state', {
          cache: 'no-store', ...(controller ? { signal: controller.signal } : {}),
        });
        if (!response.ok) throw new Error(`Status unavailable (${response.status})`);
        const snapshot = await response.json();
        if (disposed) return;
        if (!snapshot || typeof snapshot.state !== 'string') throw new Error('Invalid inference status');
        const names = { cycle: 'Cyclic exchange', steering: 'Continuous steering' };
        const active = snapshot.supported === true && !snapshot.legacy_active
          && snapshot.audio_owner !== 'legacy' && names[snapshot.active_mode]
          && ['running', 'stopping'].includes(snapshot.state);
        status.textContent = active
          ? `Active: ${names[snapshot.active_mode]} / ${typeof snapshot.phase === 'string' ? snapshot.phase : '--'}`
          : 'Inference';
        status.dataset.active = String(Boolean(active));
        status.title = active ? 'Shared audio is owned by inference' : 'Open inference';
        const nextKey = JSON.stringify([snapshot.session_id, snapshot.epoch, snapshot.audio_owner,
          snapshot.active_mode, snapshot.state, snapshot.phase, snapshot.legacy_active]);
        if (snapshot.supported === true && nextKey !== runtimeKey) {
          // The legacy monitor remains authoritative for its transport and lyric projection.
          env.scheduleSnapshotRefresh?.();
          runtimeKey = nextKey;
        }
      } catch (error) {
        if (!disposed) { runtimeKey = null; status.textContent = 'Inference status unavailable'; status.dataset.active = 'unknown'; status.title = error.message; }
      } finally {
        env.clearTimeout(timeout);
        if (!disposed) timer = env.setTimeout(poll, document.hidden ? 2000 : 1000);
      }
    }
    env.addEventListener('pagehide', () => { disposed = true; env.clearTimeout(timer); controller?.abort(); });
    env.addEventListener('pageshow', (event) => { if (event.persisted && disposed) { disposed = false; poll(); } });
    poll();
    return {};
  }

  root.StreamMuseInferenceStatus = { mount };
  if (root.document) mount(root.document, root);
})(typeof window !== 'undefined' ? window : globalThis);
