"""Exclusive routing for the existing runtime, not an independent app session."""

import copy

from streammuse.domain.rap import PlaybackState


class RuntimeInferenceBridge:
    def __init__(self, runtime, service):
        self.runtime, self.service = runtime, service
        self.owns_output = False
        self.starting = False
        service._lock = runtime._lifecycle_lock
        service._claim = self._claim

    @property
    def active(self):
        return self.starting or self.owns_output and self.service.active

    @property
    def control_state(self):
        if self.starting:
            return PlaybackState.PRIMING
        if self.service.state == "stopping":
            return PlaybackState.STOP_REQUESTED
        if not self.service.active:
            return PlaybackState.STOPPED
        return PlaybackState.PRIMING if self.service.phase in ("WAIT", "PREPARING") else PlaybackState.RUNNING

    def _guard_legacy(self):
        if self.runtime._closed:
            raise RuntimeError("rap audio runtime is closed")
        if self.runtime._starting or self.runtime._settings_updating:
            raise RuntimeError("wait for legacy startup or settings changes")
        if self.runtime.playback.state != PlaybackState.STOPPED:
            raise RuntimeError("stop legacy playback before selecting inference modes")

    def _claim(self):
        self._guard_legacy()
        if not self.owns_output:
            self.runtime._reset_locked(self.runtime._settings_factory())
        self.owns_output = True

    def release_for_legacy(self):
        # Caller has reserved runtime startup. Never join a native callback
        # while holding the shared lifecycle lock that callback must acquire.
        if self.service.output is not None:
            self.service.output.suspend()
        with self.runtime._lifecycle_lock:
            self.owns_output = False

    def snapshot(self):
        with self.runtime._lifecycle_lock:
            result = self.service._snapshot_data()
            legacy_active = not self.owns_output and self.runtime.control_state != PlaybackState.STOPPED
            settings = self.runtime.settings_service
            mode = settings.values.mode if settings is not None else "continuous"
            result.update(legacy_mode=mode, legacy_active=legacy_active,
                          audio_owner="inference" if self.owns_output else "legacy")
            audio_editable = (self.runtime.control_state == PlaybackState.STOPPED
                              and not self.starting and not self.runtime._settings_updating)
            result["capabilities"].update(audio_controls_supported=audio_editable,
                audio_controls_reason="Device and mix changes apply at the next Start; live inference changes are unavailable")
            if legacy_active:
                result["active_mode"] = "legacy"
                result["state"] = ("stopping" if self.runtime.control_state == PlaybackState.STOP_REQUESTED else "running")
        return copy.deepcopy(result)

    def configure(self, payload):
        with self.runtime._lifecycle_lock:
            if self.starting:
                raise RuntimeError("inference audio is still starting")
            self._guard_legacy()
            self.service.configure(payload, _snapshot=False)
        return self.snapshot()

    def start(self, owner_id=None):
        with self.runtime._lifecycle_lock:
            if self.starting:
                raise RuntimeError("inference audio is still starting")
            self.starting = True
        try:
            self.service.start(owner_id, _snapshot=False)
        finally:
            with self.runtime._lifecycle_lock:
                self.starting = False
        return self.snapshot()

    def stop(self):
        if self.owns_output:
            self.service.stop(_snapshot=False)
        else:
            self.runtime.request_stop()
        return self.snapshot()

    def reset(self):
        if self.owns_output:
            self.service.reset(_snapshot=False)
        else:
            self.runtime.reset()
            self.service.reset(_snapshot=False)
        return self.snapshot()

    def steer(self, payload):
        with self.runtime._lifecycle_lock:
            self._guard_legacy()
            self.service.steer(payload, _snapshot=False)
        return self.snapshot()

    def tap(self, payload):
        self.service.tap(payload, _snapshot=False)
        return self.snapshot()

    def heartbeat(self, payload):
        self.service.heartbeat(payload, _snapshot=False)
        return self.snapshot()

    def export_trace(self):
        result = self.service.export_trace()
        result["snapshot"] = self.snapshot()
        return result

    def export_metrics_csv(self):
        return self.service.export_metrics_csv()

    def close(self):
        self.service.close()


def install_inference(runtime, service):
    if getattr(runtime, "inference", None) is not None:
        raise RuntimeError("inference is already installed")
    runtime.inference = RuntimeInferenceBridge(runtime, service)
    return runtime.inference
