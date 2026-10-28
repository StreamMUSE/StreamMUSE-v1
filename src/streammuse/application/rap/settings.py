"""Validated settings shared by the rap runtime and its control API."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields, replace
from math import isfinite
from threading import RLock
from typing import Callable, Any


@dataclass(frozen=True)
class RapSettings:
    tempo_bpm: float = 60.0
    mode: str = "continuous"
    response_bars: int = 2
    output_device: str | None = None
    drum_gain: float = 0.55
    vocal_gain: float = 0.80
    chord_loop: str = "off"
    chord_gain: float = 0.20
    alignment_policy: str = "gentle_sparse_r3"


SETTINGS_GROUPS = {
    "local": frozenset({"drum_gain", "vocal_gain", "chord_gain", "chord_loop", "alignment_policy"}),
    "generation": frozenset({"tempo_bpm"}),
    "session": frozenset({"mode", "response_bars", "output_device"}),
}
STOPPED_ONLY_SETTINGS = SETTINGS_GROUPS["session"]
SETTINGS_APPLICATION = {
    **{name: "stopped-only" for name in STOPPED_ONLY_SETTINGS},
    "tempo_bpm": "bar-boundary", "chord_loop": "bar-boundary",
    "alignment_policy": "next-uncommitted-bar",
    **{name: "immediate" for name in ("drum_gain", "vocal_gain", "chord_gain")},
}


def settings_group(patch: object, group: str | None = None, *, allow_combined: bool = False) -> str:
    if not isinstance(patch, Mapping) or not patch:
        raise ValueError("settings update must be a non-empty JSON object")
    if group is not None:
        if group not in SETTINGS_GROUPS:
            raise ValueError("unknown settings group")
        if set(patch) - SETTINGS_GROUPS[group]:
            raise ValueError(f"settings do not belong to the {group} group")
        return group
    groups = [name for name, names in SETTINGS_GROUPS.items() if set(patch) & names]
    if len(groups) > 1 and allow_combined:
        return "session"
    if len(groups) != 1:
        raise ValueError("apply settings groups separately")
    return groups[0]


class RapSettingsService:
    """Validate user changes and publish only successfully applied settings."""

    def __init__(
        self,
        values: RapSettings,
        *,
        state: Callable[[], str],
        capabilities: Callable[[], Mapping[str, bool]],
        apply: Callable[[RapSettings, RapSettings, frozenset[str]], Mapping[str, Any]],
        devices: Callable[[], list[dict[str, Any]]] | None = None,
        output_status: Callable[[], Mapping[str, Any]] | None = None,
        update_capabilities: Callable[[str, frozenset[str]], Mapping[str, bool]] | None = None,
        applications: Callable[[], Mapping[str, str]] | None = None,
        refresh_capabilities: Callable[[], None] | None = None,
    ) -> None:
        self._values = values
        self._state = state
        self._capabilities = capabilities
        self._apply = apply
        self._devices = devices
        self._output_status = output_status
        self._update_capabilities = update_capabilities
        self._applications = applications
        self._refresh_capabilities = refresh_capabilities
        self._lock = RLock()
        self._group_locks = {name: RLock() for name in SETTINGS_GROUPS}
        self._ack_lock = RLock()
        self._ack: dict[str, Any] = {}
        self._group_acks: dict[str, dict[str, Any]] = {}
        self._devices_cache: list[dict[str, Any]] = []

    @property
    def values(self) -> RapSettings:
        # The immutable object is replaced atomically; lifecycle reads must not
        # wait for slow device discovery or a remote capability request.
        return self._values

    def reset_acceptance(self, coordinator_epoch: int, *, applied: bool = True) -> None:
        """Discard prior-epoch acknowledgements without waiting for discovery I/O."""
        ack = {"coordinator_epoch": coordinator_epoch}
        if applied:
            ack.update(effective_from_bar=0, session_restarted=True)
        with self._ack_lock:
            self._ack = ack
            self._group_acks = {}

    def snapshot(self, *, discover_devices: bool = True, group: str | None = None) -> dict[str, Any]:
        if discover_devices and self._refresh_capabilities is not None:
            self._refresh_capabilities()
        result = {
            "supported": True, "capabilities": {}, "state": self._state(),
            "devices": list(self._devices_cache), "application": dict(SETTINGS_APPLICATION),
        }
        try:
            result["capabilities"] = dict(self._capabilities())
        except (OSError, RuntimeError) as error:
            result["device_error"] = str(error)
        if self._applications is not None:
            result["application"].update(self._applications())
        if self._devices is not None and discover_devices:
            try:
                result["devices"] = self._devices_cache = self._devices()
            except (OSError, RuntimeError) as error:
                result["device_error"] = str(error)
        if self._output_status is not None:
            try:
                result["audio_output"] = dict(self._output_status())
            except (OSError, RuntimeError) as error:
                result["device_error"] = str(error)
        # Discovery and remote acknowledgements never own the values lock.
        with self._lock, self._ack_lock:
            values = asdict(self._values)
            result["values"] = values
            application = result["application"]
            result.update(self._ack if group is None else self._group_acks.get(group, {}))
            result["application"] = application
            result["groups"] = {
                name: {
                    **self._group_acks.get(name, {}),
                    "fields": sorted(names),
                    "values": {key: values[key] for key in names},
                    "application": {key: result["application"][key] for key in names},
                } for name, names in SETTINGS_GROUPS.items()
            }
        if group is not None:
            result["group"] = group
        return result

    def update(self, patch: object, *, group: str | None = None) -> dict[str, Any]:
        validate_settings_patch(self._values, patch)
        selected_group = settings_group(patch, group, allow_combined=self._state() == "stopped")
        affected_groups = {name for name, names in SETTINGS_GROUPS.items() if set(patch) & names}
        with ExitStack() as stack:
            for name in sorted(affected_groups):
                stack.enter_context(self._group_locks[name])
            current = self._values
            updated = validate_settings_patch(current, patch)
            changed = frozenset(
                item.name for item in fields(RapSettings)
                if getattr(updated, item.name) != getattr(current, item.name)
            )
            if changed & STOPPED_ONLY_SETTINGS and self._state() != "stopped":
                raise RuntimeError("mode, response length, and output device changes require stopped playback")
            capabilities = (self._update_capabilities(selected_group, changed)
                            if self._update_capabilities is not None else self._capabilities())
            unsupported = [name for name in sorted(changed) if not capabilities.get(name, False)]
            if unsupported:
                raise NotImplementedError(f"settings unavailable for the current backend: {', '.join(unsupported)}")
            if changed:
                ack = self._apply(current, updated, changed)
                with self._lock, self._ack_lock:
                    self._values = replace(self._values, **{name: getattr(updated, name) for name in changed})
                    self._ack = {**ack, "changed_fields": sorted(changed)}
                    for name in affected_groups:
                        group_changed = changed & SETTINGS_GROUPS[name]
                        if group_changed:
                            self._group_acks[name] = {**ack, "changed_fields": sorted(group_changed)}
            return self.snapshot(discover_devices=selected_group == "session", group=selected_group)


def validate_settings_patch(current: RapSettings, patch: object) -> RapSettings:
    """Validate the complete update before any device or scheduling side effect."""
    if not isinstance(patch, Mapping) or not patch:
        raise ValueError("settings update must be a non-empty JSON object")
    unknown = set(patch) - {item.name for item in fields(RapSettings)}
    if unknown:
        raise ValueError(f"unknown settings: {', '.join(sorted(str(key) for key in unknown))}")
    values = dict(patch)
    for name, lower, upper in (("tempo_bpm", 30.0, 180.0), ("drum_gain", 0.0, 2.0), ("vocal_gain", 0.0, 2.0), ("chord_gain", 0.0, 2.0)):
        if name not in values:
            continue
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower:g} and {upper:g}")
        if not isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        values[name] = float(value)
    if "response_bars" in values:
        value = values["response_bars"]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 8:
            raise ValueError("response_bars must be an integer between 1 and 8")
    if "mode" in values and values["mode"] not in ("continuous", "alternating"):
        raise ValueError("mode must be continuous or alternating")
    if "alignment_policy" in values and values["alignment_policy"] not in (
        "gentle_sparse_r3", "all_onsets_r3", "all_onsets_stress_r3",
    ):
        raise ValueError("alignment_policy must be gentle_sparse_r3, all_onsets_r3, or all_onsets_stress_r3")
    if "chord_loop" in values and values["chord_loop"] not in ("off", "minor_cycle", "major_cycle"):
        raise ValueError("chord_loop must be off, minor_cycle, or major_cycle")
    if "output_device" in values:
        value = values["output_device"]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("output_device must be a device identifier or null for the default")
    return replace(current, **values)
