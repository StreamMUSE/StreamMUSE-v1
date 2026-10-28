"""Research accounting must preserve failures, pairing, and causal boundaries."""

import pytest

from scripts.experiments import rap_scheduling_evidence as evidence


def test_summary_keeps_failed_rounds_in_success_denominator():
    rows = [
        {"condition": "n8", "pair_id": "a", "success": True, "elapsed_ms": 100, "valid_count": 2},
        {"condition": "n8", "pair_id": "b", "success": False, "elapsed_ms": 300, "valid_count": 0},
    ]
    summary = evidence.summarize(rows)["n8"]
    assert summary["rounds"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["elapsed_ms_median"] == 200
    assert summary["valid_count_mean"] == 1


def test_summary_exposes_actual_observation_counts_without_imputing_missing_values():
    rows = [{"condition": "a", "success": True, "elapsed_ms": 20, "valid_count": 2},
            {"condition": "a", "success": False}]
    result = evidence.summarize(rows)["a"]
    assert result["rounds"] == 2
    assert result["elapsed_ms_observed_n"] == result["valid_count_observed_n"] == 1
    assert result["elapsed_ms_median"] == 20


def test_pair_deltas_match_identity_not_row_order_and_retain_failures():
    rows = [
        {"condition": "large", "pair_id": "b", "elapsed_ms": 400, "success": False},
        {"condition": "small", "pair_id": "a", "elapsed_ms": 100, "success": True},
        {"condition": "large", "pair_id": "a", "elapsed_ms": 150, "success": True},
        {"condition": "small", "pair_id": "b", "elapsed_ms": 200, "success": True},
    ]
    assert evidence.paired_deltas(rows, "small", "large", "elapsed_ms") == [50, 200]


def test_pair_deltas_reject_duplicate_identity():
    row = {"condition": "a", "pair_id": "x", "elapsed_ms": 10}
    with pytest.raises(ValueError, match="duplicate"):
        evidence.paired_deltas([row, row], "a", "b", "elapsed_ms")


def test_deadline_sensitivity_uses_guard_and_counts_failed_jobs():
    rows = [{"success": True, "elapsed_ms": 2300}, {"success": False, "elapsed_ms": 100}]
    result = evidence.deadline_sensitivity(rows, bpm=100, lookahead_bars=1, guard_ticks=1)
    assert result["deadline_ms"] == 2250
    assert result["misses"] == 2
    assert result["evidence_kind"] == "trace_counterfactual"


def test_pipeline_trace_enforces_dependency_and_resource_capacity():
    result = evidence.replay_pipeline([(100, 300), (100, 300)], render_slots=1)
    assert result == [
        {"writer_start_ms": 0, "writer_finish_ms": 100, "render_start_ms": 100, "finish_ms": 400},
        {"writer_start_ms": 100, "writer_finish_ms": 200, "render_start_ms": 400, "finish_ms": 700},
    ]
    assert evidence.replay_pipeline([(100, 300), (100, 300)], render_slots=2)[1]["finish_ms"] == 500


def test_revision_counterfactual_never_counts_stale_ready_as_fresh():
    result = evidence.revision_outcome(start_ms=0, finish_ms=100, deadline_ms=200, edit_ms=50)
    assert result == {"ready": True, "latest_revision": False, "discarded_ms": 100}
    assert evidence.revision_outcome(start_ms=0, finish_ms=100, deadline_ms=200, edit_ms=201)["latest_revision"]


def test_isolated_launch_plan_does_not_reuse_old_ports_gpu_or_artifacts():
    argv, env, port, gpu = evidence.isolated_service_plan(
        "render", ["python", "--port", "8620", "--artifact-root", "/old/artifacts", "--vllm-url", "http://127.0.0.1:8601/v1"],
        {"CUDA_VISIBLE_DEVICES": "1", "PYTHONPATH": "/old/deps:/old/repo/src:/old/repo", "HF_MODULES_CACHE": "/old/cache"},
        root="/new", old_root="/old", repo="/new/repo",
    )
    assert port == 18620
    assert gpu == 2
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert argv[argv.index("--artifact-root") + 1] == "/new/render_artifacts"
    assert argv[argv.index("--vllm-url") + 1] == "http://127.0.0.1:18611/v1"
    assert env["PYTHONPATH"] == "/old/deps:/new/repo/src:/new/repo"
    assert env["HF_MODULES_CACHE"] == "/new/hf-modules"


def test_live_llm_is_separate_from_research_llm():
    argv, env, port, gpu = evidence.isolated_service_plan(
        "vllm-live", ["vllm", "--port", "8601"], {},
        root="/new", old_root="/old", repo="/new/repo")
    assert (port, gpu) == (18611, 1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_failed_real_writer_round_preserves_generation_errors():
    from streammuse.application.rap.lyric_writer import LyricWriteRequest, LyricWriterPolicy
    from streammuse.infrastructure.rap.generators import ScriptedFailureGenerator
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

    request = LyricWriteRequest("research", 0, 0, "night", BUILTIN_TEMPLATES.get("baseline_straight_9"), (), 3)
    row = evidence.measure_writer_round(ScriptedFailureGenerator(), request, LyricWriterPolicy(candidates=8, rescue_waves=0))
    assert row["success"] is False
    assert row["valid_count"] == 0
    assert row["requested_count"] == 8
    assert row["raw_waves"][0]["batch"]["error_type"] == "generation_error"
    assert row["error_type"] == "LyricWriteFailed"
    assert row["evaluation_ms"] is None


def test_writer_round_preserves_ledger_and_exact_selected_record():
    from streammuse.application.rap.lyric_writer import LyricWriteRequest, LyricWriterPolicy
    from streammuse.infrastructure.rap.generators import PhraseBankGenerator
    from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES

    request = LyricWriteRequest("research", 0, 0, "night", BUILTIN_TEMPLATES.get("baseline_straight_9"), (), 3)
    row = evidence.measure_writer_round(PhraseBankGenerator(), request, LyricWriterPolicy(candidates=8, rescue_waves=0))
    assert row["success"]
    assert row["selected"]["text"]
    assert row["valid_count"] >= 1
    assert sum(item["status"] == "selected" for item in row["ledger"]) == 1
    assert row["elapsed_ms"] >= row["generation_ms"]


def test_start_payload_limits_initial_audio_without_claiming_playback():
    payload = evidence.make_start_payload(topic="city", seed=7, template_id="baseline_straight_9", bars=2, bpm=90)
    assert payload["lyric_lead"] == 2
    assert payload["audio_lead"] == 2
    assert payload["start_bar"] == 0
    assert payload["segments"][0]["bars"] == 2
    assert payload["loop"] is False


def test_raw_batch_preserves_frozen_prompt_mappings_without_deepcopy():
    from streammuse.domain.rap import CandidateBatch
    batch = CandidateBatch(request_id="a", candidates=("city lights",), source="local_chat_independent",
                           prompt=({"role": "user", "content": "write"},), raw_response="city lights", latency_ms=3)
    payload = evidence.json_value(batch)
    assert payload["prompt"] == [{"role": "user", "content": "write"}]
    assert payload["candidates"] == ["city lights"]


def test_concurrent_jobs_actually_overlap_and_preserve_result_order():
    from threading import Barrier
    gate = Barrier(2)

    def job(value):
        gate.wait(timeout=2)
        return value

    result = evidence.execute_jobs([lambda: job("first"), lambda: job("second")], workers=2)
    assert result["results"] == ["first", "second"]
    assert result["elapsed_ms"] >= 0


def test_edit_schedule_is_causal_and_version_monotone():
    edits = evidence.make_edit_schedule([0.25, 0.5, 0.75])
    assert [item["at_ms"] for item in edits] == [250, 500, 750]
    assert [item["message"]["edit"]["edit_seq"] for item in edits] == [1, 2, 3]
    assert all(item["message"]["edit"]["from_bar"] == 0 for item in edits)
    with pytest.raises(ValueError):
        evidence.make_edit_schedule([0.5, 0.25])


def test_pcm_validation_rejects_silent_model_output():
    import io
    import wave
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 2400)
    with pytest.raises(ValueError, match="silent"):
        evidence.inspect_pcm(output.getvalue())


def test_pcm_validation_reports_source_duration_not_target_bar_duration():
    import io
    import struct
    import wave
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(struct.pack("<hh", -300, 400) * 1200)
    assert evidence.inspect_pcm(output.getvalue()) == {
        "frames": 2400, "sample_rate_hz": 24000, "source_duration_s": 0.1,
        "peak_pcm16": 400, "rms_pcm16": pytest.approx(353.5533905932738)}


def test_latest_content_revision_is_not_the_plan_edit_sequence():
    lyric = {"lyric_id": "current", "revision": 2, "topic": "street drums"}
    assert evidence.matches_latest(lyric, latest_id="current", topic="street drums")
    assert not evidence.matches_latest(lyric, latest_id="other", topic="street drums")
    assert not evidence.matches_latest({**lyric, "revision": 99, "topic": "old"}, latest_id="current", topic="street drums")


def test_stop_authorization_requires_exact_process_identity_and_owner():
    identity = {"pid": 12, "pgid": 12, "uid": 4, "start_ticks": "20"}
    record = {"pid": 12, "identity": identity}
    assert evidence.can_stop(record, identity, uid=4)
    assert not evidence.can_stop(record, {**identity, "start_ticks": "21"}, uid=4)
    assert not evidence.can_stop(record, identity, uid=5)
    assert not evidence.can_stop({"pid": 12, "identity": {**identity, "pgid": 1}}, {**identity, "pgid": 1}, uid=4)


def test_deadlines_keep_original_future_bar_obligations():
    rows = [{"bar": 0, "success": True, "elapsed_ms": 2501},
            {"bar": 1, "success": True, "elapsed_ms": 5000}]
    result = evidence.deadline_sensitivity(rows, bpm=90, lookahead_bars=1, guard_ticks=1)
    assert result["misses"] == 1
    assert result["outcomes"][1]["deadline_ms"] == pytest.approx(5166.6666667)


@pytest.mark.parametrize("bpm", [0, -1, float("nan"), float("inf")])
def test_deadlines_reject_nonphysical_tempo(bpm):
    with pytest.raises(ValueError):
        evidence.deadline_sensitivity([], bpm=bpm, lookahead_bars=1, guard_ticks=1)


def test_deadline_boundary_is_inclusive_but_stale_ready_is_not_fresh_success():
    rows = [{"success": True, "elapsed_ms": 2250},
            {"success": True, "elapsed_ms": 100, "latest_revision": False}]
    result = evidence.deadline_sensitivity(rows, bpm=100, lookahead_bars=1, guard_ticks=1)
    assert result["misses"] == 1


@pytest.mark.parametrize("bar", [float("nan"), float("inf"), -1, 0.5, True, "0"])
def test_deadline_counterfactual_rejects_invalid_bar_ordinals(bar):
    with pytest.raises(ValueError):
        evidence.deadline_sensitivity([{"bar": bar, "success": True, "elapsed_ms": 100}],
                                      bpm=90, lookahead_bars=2, guard_ticks=1)


@pytest.mark.parametrize("slots", [1, 2, 3])
def test_pipeline_conserves_work_and_never_exceeds_resource_capacity(slots):
    import random
    rng = random.Random(3)
    costs = [(rng.randrange(0, 100), rng.randrange(0, 100)) for _ in range(100)]
    rows = evidence.replay_pipeline(costs, render_slots=slots)
    for index, (row, (writer, render)) in enumerate(zip(rows, costs)):
        assert row["writer_finish_ms"] - row["writer_start_ms"] == writer
        assert row["finish_ms"] - row["render_start_ms"] == render
        assert row["render_start_ms"] >= row["writer_finish_ms"]
        if index:
            assert row["writer_start_ms"] >= rows[index - 1]["writer_finish_ms"]
        active = sum(other["render_start_ms"] <= row["render_start_ms"] < other["finish_ms"] for other in rows)
        assert active <= slots


def test_cache_eligibility_requires_only_declared_dependencies_but_none_may_be_missing():
    artifact = {"text": "a", "voice": "v", "timing": 1}
    assert evidence.cache_compatible(artifact, {"text": "a", "voice": "v", "timing": 2}, dependencies=("text", "voice"))
    assert not evidence.cache_compatible(artifact, {"text": "b", "voice": "v"}, dependencies=("text", "voice"))
    assert not evidence.cache_compatible(artifact, {"text": "a"}, dependencies=("text", "voice"))
    assert not evidence.cache_compatible(artifact, {"text": "a", "voice": "v", "timing": 2}, dependencies=("text", "voice", "timing"))


def test_prepared_audio_validation_rejects_wrong_duration_and_nonfinite_samples():
    import numpy as np
    from streammuse.domain.rap import AudioFormat, PcmAudio
    fmt = AudioFormat()
    audio = PcmAudio(fmt, 4, np.zeros((4, 2), dtype=np.float32).tobytes())
    with pytest.raises(ValueError, match="frame"):
        evidence.inspect_prepared(audio, expected_frames=5)
    bad = PcmAudio(fmt, 4, np.full((4, 2), np.nan, dtype=np.float32).tobytes())
    with pytest.raises(ValueError, match="finite"):
        evidence.inspect_prepared(bad, expected_frames=4)


def test_prepared_audio_validation_reports_actual_peak_and_frame_count():
    import numpy as np
    from streammuse.domain.rap import AudioFormat, PcmAudio
    audio = PcmAudio(AudioFormat(), 4, np.full((4, 2), 0.5, dtype=np.float32).tobytes())
    result = evidence.inspect_prepared(audio, expected_frames=4)
    assert result["frame_count"] == 4
    assert result["peak"] == 0.5
    assert result["rms"] == 0.5


def test_research_preparation_audit_rejects_silent_output():
    import numpy as np
    from streammuse.domain.rap import AudioFormat, PcmAudio
    audio = PcmAudio(AudioFormat(), 4, np.zeros((4, 2), dtype=np.float32).tobytes())
    with pytest.raises(ValueError, match="silent"):
        evidence.inspect_prepared(audio, expected_frames=4)


def test_pipeline_cannot_use_input_before_its_release_watermark():
    rows = evidence.replay_pipeline([(10, 20), (10, 20)], render_slots=2,
                                    release_times_ms=[100, 500])
    assert [row["writer_start_ms"] for row in rows] == [100, 500]
    assert [row["finish_ms"] for row in rows] == [130, 530]


@pytest.mark.parametrize("releases", [[0], [0, -1], [0, float("nan")]])
def test_pipeline_rejects_missing_or_nonphysical_release_watermarks(releases):
    with pytest.raises(ValueError):
        evidence.replay_pipeline([(10, 20), (10, 20)], release_times_ms=releases)


@pytest.mark.parametrize("kwargs", [
    {"start_ms": -1}, {"finish_ms": float("inf")},
    {"deadline_ms": float("nan")}, {"edit_ms": -1},
])
def test_revision_counterfactual_rejects_nonphysical_timestamps(kwargs):
    values = {"start_ms": 0, "finish_ms": 100, "deadline_ms": 200, "edit_ms": 50}
    with pytest.raises(ValueError):
        evidence.revision_outcome(**{**values, **kwargs})


def test_revision_at_start_is_in_snapshot_but_before_deadline_invalidates_it():
    assert evidence.revision_outcome(start_ms=50, finish_ms=100, deadline_ms=200, edit_ms=50)["latest_revision"]
    assert not evidence.revision_outcome(start_ms=50, finish_ms=100, deadline_ms=200, edit_ms=150)["latest_revision"]


def test_stage_summary_does_not_turn_missing_observation_into_zero_cost():
    rows = [{"condition": "a", "success": True, "elapsed_ms": 100,
             "stage_timings_ms": {"moss": 70}},
            {"condition": "a", "success": False, "elapsed_ms": 200}]
    result = evidence.stage_summary(rows)["a"]
    assert result["rounds"] == 2
    assert result["stages"]["moss"]["observed_n"] == 1
    assert "render_residual" not in result["stages"]


def test_reduced_registry_has_typed_evidence_and_resolvable_local_references():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    registry = json.loads((root / "docs/research/2026-10-22-reduced-scheduling-registry.json").read_text())
    assert 5 <= len(registry["groups"]) <= 8
    assert len({group["id"] for group in registry["groups"]}) == len(registry["groups"])
    for group in registry["groups"]:
        assert group["merge_rationale"]
        assert group["parameters"]
        for parameter in group["parameters"]:
            for required in ("name", "definition", "type_domain", "unit", "scope", "source", "changes_at", "dependencies", "evidence_refs"):
                assert required in parameter
            assert "current_value" in parameter
            assert parameter["class"] in registry["class_enum"]
            assert parameter["status"] in registry["status_enum"]
            assert set(parameter["evidence_refs"]) <= registry["evidence_index"].keys()
            assert group["observation_method"]
            assert group["clock_and_uncertainty"]
    assert len(registry["theoretical_granularity_vector"]) == 18
    for reference in registry["evidence_index"].values():
        # Raw records are intentionally ignored; tracked references must resolve in a clean checkout.
        if not reference.startswith("logs/"):
            assert (root / reference.split("#", 1)[0]).is_file()


def test_reduced_registry_measurements_match_published_analysis():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[3] / "docs/research"
    registry = json.loads((root / "2026-10-22-reduced-scheduling-registry.json").read_text())
    summary = json.loads((root / "2026-10-22-scheduling-evidence.json").read_text())
    parameters = {item["name"]: item["current_value"] for group in registry["groups"] for item in group["parameters"]}
    for route in ("h200", "mac"):
        rows = [summary[f"budget_{route}"][f"n{budget}"] for budget in (16, 32, 64)]
        assert parameters["writer_latency"][f"{route}_median_n16_n32_n64"] == [round(row["elapsed_ms_median"], 2) for row in rows]
        assert parameters["usable_yield"][f"{route}_selectable_attempts"] == [round(row["success_rate"] * row["rounds"]) for row in rows]
    assert parameters["mac_warp_mix_cost"]["median"] == round(summary["preparation"]["mac_warp_mix"]["elapsed_ms_median"], 2)


def test_stage_summary_does_not_impute_missing_stage_times_as_zero():
    rows = [{"condition": "a", "success": True, "elapsed_ms": 25,
             "stage_timings_ms": {"moss": 20, "aligner": 2, "render_total": 25}},
            {"condition": "a", "success": False, "elapsed_ms": 40}]
    result = evidence.stage_summary(rows)["a"]
    assert result["success_rate"] == 0.5
    assert result["stages"]["moss"] == {"observed_n": 1, "median_ms": 20, "min_ms": 20, "max_ms": 20}
    assert result["stages"]["render_residual"]["median_ms"] == 3
