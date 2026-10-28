from dataclasses import asdict

import pytest

from streammuse.application.rap.settings import RapSettings, RapSettingsService, validate_settings_patch


def test_settings_validate_all_fields_without_mutating_current():
    current = RapSettings(tempo_bpm=60.0)
    updated = validate_settings_patch(current, {"tempo_bpm": 90, "mode": "alternating", "response_bars": 3})
    assert updated.tempo_bpm == 90
    assert updated.mode == "alternating"
    assert updated.response_bars == 3
    assert current.tempo_bpm == 60


@pytest.mark.parametrize("patch", [
    {"tempo_bpm": 29}, {"tempo_bpm": 181}, {"tempo_bpm": float("nan")},
    {"tempo_bpm": True}, {"response_bars": 0}, {"response_bars": 9},
    {"response_bars": 1.5}, {"response_bars": True}, {"mode": "combined"},
    {"drum_gain": -0.1}, {"vocal_gain": float("inf")}, {"output_device": 3},
    {"unknown": 12}, {"output_device": ""}, {"tempo_bpm": 10 ** 400},
])
def test_invalid_settings_are_rejected(patch):
    with pytest.raises(ValueError):
        validate_settings_patch(RapSettings(), patch)


def test_settings_reject_non_object_and_empty_updates():
    for patch in (None, [], {}, "tempo"):
        with pytest.raises(ValueError):
            validate_settings_patch(RapSettings(), patch)


def test_settings_default_is_existing_continuous_mode():
    values = asdict(RapSettings())
    assert values["mode"] == "continuous"
    assert values["response_bars"] == 2
    assert values["output_device"] is None


def test_live_settings_reject_stopped_only_change_before_side_effect():
    calls = []
    service = RapSettingsService(
        RapSettings(), state=lambda: "running", capabilities=lambda: {"mode": True},
        apply=lambda current, updated, changed: calls.append(changed) or {},
    )
    with pytest.raises(RuntimeError, match="stopped"):
        service.update({"mode": "alternating"})
    assert calls == []
    assert service.snapshot()["values"]["mode"] == "continuous"


def test_settings_apply_ack_and_unsupported_change():
    service = RapSettingsService(
        RapSettings(), state=lambda: "running", capabilities=lambda: {"tempo_bpm": True},
        apply=lambda current, updated, changed: {"effective_from_bar": 4, "revision": 2},
    )
    result = service.update({"tempo_bpm": 90})
    assert result["values"]["tempo_bpm"] == 90
    assert result["effective_from_bar"] == 4
    assert result["revision"] == 2
    with pytest.raises(NotImplementedError):
        service.update({"vocal_gain": 0.7})


def test_settings_discovery_preserves_exact_changed_fields_after_noop_and_rejection():
    def apply(current, updated, changed):
        if updated.chord_loop == "major_cycle":
            raise RuntimeError("mix rejected")
        return {"effective_from_bar": 4, "mix_revision": 1, "coordinator_epoch": 0}

    service = RapSettingsService(
        RapSettings(), state=lambda: "running",
        capabilities=lambda: {"chord_loop": True, "chord_gain": True}, apply=apply,
    )
    result = service.update({"chord_loop": "minor_cycle", "chord_gain": 0.3, "vocal_gain": 0.8})
    assert result["changed_fields"] == ["chord_gain", "chord_loop"]
    assert service.snapshot()["changed_fields"] == ["chord_gain", "chord_loop"]
    assert service.update({"chord_gain": 0.3})["changed_fields"] == ["chord_gain", "chord_loop"]
    with pytest.raises(RuntimeError, match="mix rejected"):
        service.update({"chord_loop": "major_cycle"})
    snapshot = service.snapshot()
    assert snapshot["changed_fields"] == ["chord_gain", "chord_loop"]
    assert snapshot["values"]["chord_loop"] == "minor_cycle"
    assert snapshot["mix_revision"] == 1
    assert snapshot["effective_from_bar"] == 4


def test_settings_reset_discards_previous_changed_fields():
    service = RapSettingsService(
        RapSettings(), state=lambda: "running", capabilities=lambda: {"chord_loop": True},
        apply=lambda *args: {"effective_from_bar": 4, "mix_revision": 1},
    )
    assert service.update({"chord_loop": "minor_cycle"})["changed_fields"] == ["chord_loop"]
    service.reset_acceptance(2)
    snapshot = service.snapshot()
    assert "changed_fields" not in snapshot
    assert "mix_revision" not in snapshot
    assert snapshot["coordinator_epoch"] == 2
    assert snapshot["effective_from_bar"] == 0


def test_device_enumeration_failure_is_visible_not_a_broken_settings_page():
    def unavailable():
        raise OSError("audio subsystem unavailable")
    service = RapSettingsService(
        RapSettings(), state=lambda: "stopped", capabilities=lambda: {},
        apply=lambda *args: {}, devices=unavailable,
    )
    snapshot = service.snapshot()
    assert snapshot["devices"] == []
    assert "unavailable" in snapshot["device_error"]


def test_failed_apply_does_not_claim_new_settings():
    def reject(*args):
        raise RuntimeError("server rejected tempo revision")
    service = RapSettingsService(
        RapSettings(), state=lambda: "running", capabilities=lambda: {"tempo_bpm": True}, apply=reject,
    )
    with pytest.raises(RuntimeError):
        service.update({"tempo_bpm": 100})
    assert service.snapshot()["values"]["tempo_bpm"] == 60


def test_output_status_failure_does_not_hide_successful_settings_application():
    def unavailable():
        raise OSError("device status unavailable")

    service = RapSettingsService(
        RapSettings(), state=lambda: "stopped", capabilities=lambda: {"tempo_bpm": True},
        apply=lambda *args: {"effective_from_bar": 0}, output_status=unavailable,
    )
    result = service.update({"tempo_bpm": 90})
    assert result["values"]["tempo_bpm"] == 90
    assert "device status unavailable" in result["device_error"]


def test_local_group_completes_while_generation_waits_and_keeps_both_values():
    from threading import Event, Thread

    entered, release, local_done = Event(), Event(), Event()
    errors = []

    def apply(current, updated, changed):
        if "tempo_bpm" in changed:
            entered.set()
            assert release.wait(2)
            return {"effective_from_bar": 5, "revision": 1}
        return {"application": "immediate", "revision": 2}

    service = RapSettingsService(
        RapSettings(), state=lambda: "running",
        capabilities=lambda: {"tempo_bpm": True, "drum_gain": True}, apply=apply,
    )

    def change(patch, group, completed=None):
        try:
            service.update(patch, group=group)
        except BaseException as error:
            errors.append(error)
        finally:
            if completed:
                completed.set()

    remote = Thread(target=change, args=({"tempo_bpm": 80}, "generation"))
    remote.start()
    local = None
    try:
        assert entered.wait(0.5)
        local = Thread(target=change, args=({"drum_gain": 0.3}, "local", local_done))
        local.start()
        assert local_done.wait(0.5), "local control blocked on the server update"
    finally:
        release.set()
        remote.join(2)
        if local:
            local.join(2)
    assert errors == []
    snapshot = service.snapshot()
    assert snapshot["values"]["tempo_bpm"] == 80
    assert snapshot["values"]["drum_gain"] == 0.3
    assert snapshot["groups"]["local"]["changed_fields"] == ["drum_gain"]
    assert snapshot["groups"]["generation"]["effective_from_bar"] == 5


def test_group_rejects_misrouted_fields_before_apply():
    applied = []
    service = RapSettingsService(
        RapSettings(), state=lambda: "running", capabilities=lambda: {"tempo_bpm": True},
        apply=lambda *args: applied.append(args) or {},
    )
    with pytest.raises(ValueError, match="group"):
        service.update({"tempo_bpm": 80}, group="local")
    assert applied == []


def test_alignment_is_live_local_and_settings_describe_activation():
    service = RapSettingsService(
        RapSettings(), state=lambda: "running", capabilities=lambda: {"alignment_policy": True},
        apply=lambda *args: {"effective_from_bar": 3, "alignment_policy": "all_onsets_r3"},
    )
    snapshot = service.update({"alignment_policy": "all_onsets_r3"}, group="local")
    assert snapshot["group"] == "local"
    assert snapshot["application"]["alignment_policy"] == "next-uncommitted-bar"
    assert snapshot["groups"]["local"]["effective_from_bar"] == 3
    assert snapshot["groups"]["session"]["fields"] == ["mode", "output_device", "response_bars"]


def test_legacy_stopped_bulk_settings_keep_grouped_acceptances():
    applied = []
    service = RapSettingsService(
        RapSettings(), state=lambda: "stopped",
        capabilities=lambda: {"mode": True, "tempo_bpm": True, "alignment_policy": True},
        apply=lambda current, updated, changed: applied.append(updated) or {"session_restarted": True},
    )
    result = service.update({"mode": "alternating", "tempo_bpm": 50, "alignment_policy": "all_onsets_r3"})
    assert len(applied) == 1
    assert result["groups"]["local"]["changed_fields"] == ["alignment_policy"]
    assert result["groups"]["generation"]["changed_fields"] == ["tempo_bpm"]
    assert result["groups"]["session"]["changed_fields"] == ["mode"]


def test_untouched_group_never_inherits_another_groups_ack():
    service = RapSettingsService(
        RapSettings(), state=lambda: "running", capabilities=lambda: {"tempo_bpm": True},
        apply=lambda *args: {"effective_from_bar": 4, "revision": 9},
    )
    service.update({"tempo_bpm": 80}, group="generation")
    local = service.snapshot(group="local", discover_devices=False)
    assert "changed_fields" not in local
    assert "effective_from_bar" not in local
    assert "revision" not in local
    assert local["groups"]["generation"]["revision"] == 9
