"""Bind typed settings to replaceable rap controllers and audio sinks."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from threading import Lock, Thread
from typing import Callable, Any

from streammuse.application.rap.settings import RapSettings, RapSettingsService
from streammuse.domain.rap.audio import RapMixConfig
from streammuse.domain.rap.lyric_stream import BOUNDED_TURNS_CAPABILITY, TEMPO_UPDATES_CAPABILITY


def install_audio_settings(
    runtime,
    initial: RapSettings,
    *,
    sink,
    controller_factory: Callable[[RapSettings, int], Any] | None = None,
    capabilities_probe: Callable[[], set[str] | frozenset[str]] | None = None,
) -> RapSettingsService:
    """Install settings without starting playback, models, or a capture turn."""
    cached_capabilities: set[str] = set()
    probed_at = float("-inf")
    mix_revision = 0
    probe_lock = Lock()
    refresh_lock = Lock()
    refresh_thread: Thread | None = None

    def remote_capabilities() -> set[str]:
        nonlocal cached_capabilities, probed_at
        with probe_lock:
            if capabilities_probe is not None and time.monotonic() - probed_at >= 5.0:
                try:
                    cached_capabilities = set(capabilities_probe())
                except Exception:
                    cached_capabilities = set()
                probed_at = time.monotonic()
        return cached_capabilities

    def refresh_capabilities() -> None:
        nonlocal refresh_thread
        if capabilities_probe is None or time.monotonic() - probed_at < 5.0:
            return
        with refresh_lock:
            if refresh_thread is None or not refresh_thread.is_alive():
                refresh_thread = Thread(target=remote_capabilities, name="rap-settings-discovery", daemon=True)
                refresh_thread.start()

    def supports_devices() -> bool:
        if not callable(getattr(sink, "set_output_device", None)):
            return False
        status = getattr(sink, "output_device_status", None)
        return not callable(status) or bool(status().get("supports_device_selection", False))

    def capabilities(*, include_devices: bool = True) -> dict[str, bool]:
        remote = cached_capabilities
        mode = controller_factory is not None and BOUNDED_TURNS_CAPABILITY in remote
        mix = callable(getattr(runtime.controller, "update_mix", None))
        return {
            "mode": mode,
            "response_bars": mode,
            "tempo_bpm": TEMPO_UPDATES_CAPABILITY in remote and callable(getattr(runtime.controller, "update_tempo", None)),
            "drum_gain": mix,
            "vocal_gain": mix,
            "chord_loop": mix,
            "chord_gain": mix,
            "alignment_policy": callable(getattr(runtime.controller, "update_alignment_policy", None)),
            "output_device": supports_devices() if include_devices else False,
        }

    def update_capabilities(group: str, changed: frozenset[str]) -> dict[str, bool]:
        if changed & {"tempo_bpm", "mode", "response_bars"}:
            remote_capabilities()
        return capabilities(include_devices=group != "local")

    def applications() -> dict[str, str]:
        instant = callable(getattr(sink, "set_live_gains", None))
        status = getattr(sink, "output_device_status", None)
        if instant and callable(status):
            try:
                instant = status().get("supports_live_gains") is True
            except (OSError, RuntimeError):
                instant = False
        return {name: "immediate" if instant else "bar-boundary"
                for name in ("drum_gain", "vocal_gain", "chord_gain")}

    def devices() -> list[dict[str, Any]]:
        if not supports_devices():
            return []
        result = []
        for device in sink.list_output_devices():
            value = device.to_payload()
            identity = {"host_api": value["host_api"], "name": value["name"]}
            result.append({**value, "id": json.dumps(identity, sort_keys=True),
                           "label": f"{value['host_api']} / {value['name']}"})
        return result

    def apply(current: RapSettings, updated: RapSettings, changed: frozenset[str]) -> dict[str, Any]:
        nonlocal mix_revision
        ack: dict[str, Any] = {}
        rebuild = controller_factory is not None and (
            bool(changed & {"mode", "response_bars"})
            or ("tempo_bpm" in changed and runtime.control_state.value == "stopped")
        )
        operations = int("output_device" in changed) + int(rebuild)
        if not rebuild:
            operations += int("tempo_bpm" in changed) + int(bool(changed & {"drum_gain", "vocal_gain", "chord_loop", "chord_gain"}))
            operations += int("alignment_policy" in changed)
        if operations > 1:
            raise ValueError("apply output device, tempo, and mix changes separately; combined operations are not atomic")
        selected = None
        if "output_device" in changed:
            selected = updated.output_device
            if isinstance(selected, str) and selected.startswith("{"):
                selected = json.loads(selected)
            selected = sink.validate_output_device(selected)
        if rebuild:
            # A concurrent local edit may have completed during remote discovery.
            # Read its accepted values inside the lifecycle-fenced replacement.
            runtime.replace_controller(lambda epoch: controller_factory(
                replace(service.values, **{name: getattr(updated, name) for name in changed}), epoch,
            ))
            ack.update(effective_from_bar=0, session_restarted=True)
        elif "tempo_bpm" in changed:
            ack.update(runtime.controller.update_tempo(updated.tempo_bpm))
        if "alignment_policy" in changed and not rebuild:
            ack.update(runtime.controller.update_alignment_policy(updated.alignment_policy))
        if changed & {"drum_gain", "vocal_gain", "chord_loop", "chord_gain"} and not rebuild:
            next_revision = mix_revision + 1
            ack.update(runtime.controller.update_mix(RapMixConfig(
                drum_gain=updated.drum_gain, vocal_gain=updated.vocal_gain, revision=next_revision,
                chord_loop=updated.chord_loop, chord_gain=updated.chord_gain,
            )))
            mix_revision = next_revision
            ack["mix_revision"] = next_revision
        # Controller replacement retains the device sink and its live gain targets.
        live = getattr(sink, "set_live_gains", None)
        if callable(live) and changed & {"drum_gain", "vocal_gain", "chord_gain"}:
            ack["live_mix"] = live(
                drum_gain=updated.drum_gain, vocal_gain=updated.vocal_gain,
                chord_gain=updated.chord_gain,
            )
        if "output_device" in changed:
            resolved = sink.set_output_device(selected)
            record = getattr(runtime, "record_audio_device", None)
            if callable(record):
                name = getattr(resolved, "name", None) or getattr(selected, "name", None)
                if name is None and isinstance(selected, dict):
                    name = selected.get("name")
                record(str(name or updated.output_device or "System default"))
        return ack

    service = RapSettingsService(
        initial, state=lambda: runtime.control_state.value,
        capabilities=capabilities, apply=apply, devices=devices,
        output_status=getattr(sink, "output_device_status", None),
        update_capabilities=update_capabilities, applications=applications,
        refresh_capabilities=refresh_capabilities,
    )
    runtime.settings_service = service
    runtime.controller_factory = controller_factory
    return service
