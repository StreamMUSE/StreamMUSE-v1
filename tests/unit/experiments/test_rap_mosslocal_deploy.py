"""Offline launcher contracts; process/GPU boundaries are simulated, never operated."""

from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def deploy(tmp_path, monkeypatch):
    path = Path(__file__).parents[3] / "scripts/experiments/rap_mosslocal_20261028_deploy.py"
    assert path.is_file(), "The scoped deployment launcher must exist"
    spec = importlib.util.spec_from_file_location("mosslocal_deploy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path / "new")
    monkeypatch.setattr(module, "OLDROOT", tmp_path / "old")
    monkeypatch.setattr(module, "HF_HOME", tmp_path / "home/.cache/huggingface")
    module.safety.ROOT = module.ROOT
    module.old_safety.ROOT = module.OLDROOT
    module.ROOT.mkdir()
    module.OLDROOT.mkdir()
    for relative, content in {
        "sglang-omni/examples/configs/moss_tts_local_non_streaming.yaml": "model: local\n",
        "moss-local-continuation.patch": "continuation patch\n",
        "environment.lock.txt": "sglang==0.5.21\nsglang-omni==0.1.8\n",
        "media/reference.wav": "wave fixture",
        "media/reference.txt": "reference words",
        "repo/src/feature.py": "CURRENT_FEATURE = True\n",
        "env/bin/sgl-omni": "fixture",
    }.items():
        target = module.ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    cached = module.HF_HOME / "hub/models--OpenMOSS-Team--MOSS-TTS-Local-Transformer-v1.5/snapshots" / module.MODEL_REVISION
    cached.mkdir(parents=True)
    model = module.ROOT / "model/be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
    model.mkdir(parents=True)
    (model / "config.json").write_text('{}\n')
    (model / "configuration_moss_tts.py").write_text('# materialized configuration\n')
    (model / "model.safetensors").write_bytes(b'weights fixture')
    codec = module.ROOT / "codec/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    codec.mkdir(parents=True)
    (codec / "config.json").write_text('{}\n')
    (codec / "codec.py").write_text('# materialized codec\n')
    (codec / "model.safetensors").write_bytes(b'codec weights fixture')
    return module


def baseline(deploy):
    return {
        "service": "render", "root": str(deploy.OLDROOT), "port": 18720,
        "gpu_uuid": deploy.GPU_UUID,
        "argv": ["/old/env/bin/python", "-u", "-m", "streammuse.presentation.rap_render_server",
                 "--port", "18720", "--vllm-url", "http://127.0.0.1:18711/v1",
                 "--vllm-model", "qwen-rap", "--moss-model", "old-model",
                 "--moss-device=cuda:0", "--moss-serving-backend", "inprocess",
                 "--artifact-root", "/old/artifacts", "--moss-reference-wav", "/old/ref.wav",
                 "--wire-audio-codec", "pcm", "--lyric-candidates=32",
                 "--vllm-extra-payload", '{"preserve":true}', "--concurrent-bar-generation"],
        "environment": {"PYTHONPATH": "/old/deps:/old/repo/src", "LD_LIBRARY_PATH": "/old/lib",
                        "KEEP_THIS": "unchanged", "STREAMMUSE_DEPLOYMENT_TOKEN": "old-token"},
    }


def save_baseline(deploy):
    path = deploy.OLDROOT / "render.pid.json"
    path.write_text(json.dumps(baseline(deploy)))
    return path


def option(argv, name):
    return argv[argv.index(name) + 1]


def test_render_plan_preserves_baseline_features_and_replaces_only_runtime(deploy):
    original = baseline(deploy)
    saved = deepcopy(original)
    plan = deploy.build_plan("render", original)
    argv, env = plan["argv"], plan["environment"]
    assert original == saved
    assert argv[:4] == original["argv"][:4]
    assert option(argv, "--vllm-model") == "qwen-rap"
    assert option(argv, "--vllm-url") == "http://127.0.0.1:18711/v1"
    assert option(argv, "--vllm-extra-payload") == '{"preserve":true}'
    assert "--concurrent-bar-generation" in argv
    assert "--moss-device=cuda:0" not in argv
    assert option(argv, "--moss-serving-backend") == "sglang-omni"
    assert option(argv, "--moss-model") == "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
    assert option(argv, "--moss-model-revision") == "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
    assert option(argv, "--moss-runtime-revision") == "870d77446b3fae6e4f4f2a6964f7eae2f75e663a"
    assert option(argv, "--moss-runtime-version") == "0.1.8"
    assert option(argv, "--moss-sglang-version") == "0.5.21"
    assert option(argv, "--wire-audio-codec") == "pcm"
    assert option(argv, "--lyric-candidates") == "32"
    assert {"--lyric-stream", "--lyric-continuation", "--moss-sglang-resample-output"} <= set(argv)
    assert env["KEEP_THIS"] == "unchanged"
    assert env["LD_LIBRARY_PATH"] == "/old/lib"
    assert env["PYTHONPATH"] == f"{deploy.ROOT}/repo/src:{deploy.ROOT}/repo:/old/deps"
    assert "STREAMMUSE_DEPLOYMENT_TOKEN" not in env
    for flag, relative in [
        ("--moss-runtime-environment-sha256", "environment.lock.txt"),
        ("--moss-runtime-config-sha256", "sglang-omni/examples/configs/moss_tts_local_non_streaming.yaml"),
        ("--moss-runtime-patch-sha256", "moss-local-continuation.patch"),
        ("--moss-sglang-reference-sha256", "media/reference.wav"),
    ]:
        assert option(argv, flag) == hashlib.sha256((deploy.ROOT / relative).read_bytes()).hexdigest()
    assert plan["evidence"]["reference_text_sha256"] == hashlib.sha256(b"reference words").hexdigest()


def test_tts_uses_pinned_local_snapshot_nonstreaming_and_both_memory_limits(deploy):
    plan = deploy.build_plan("tts")
    argv = plan["argv"]
    assert argv[:2] == [str(deploy.ROOT / "env/bin/sgl-omni"), "serve"]
    model = deploy.ROOT / "model/be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
    assert option(argv, "--model-path") == str(model)
    assert plan["evidence"]["resolved_model_path"] == str(model.resolve())
    assert option(argv, "--port") == "18730"
    assert option(argv, "--tts_engine.gpu_memory_fraction") == "0.3"
    assert option(argv, "--tts_engine.engine.mem_fraction_static") == "0.3"
    assert option(argv, "--host") == "127.0.0.1"
    assert plan["environment"]["HF_HOME"] == str(deploy.HF_HOME)
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == deploy.GPU_UUID


def test_tts_served_model_name_matches_renderer_id_without_changing_local_weights(deploy):
    tts = deploy.build_plan("tts")
    render = deploy.build_plan("render", baseline(deploy))
    assert "--model-name" in tts["argv"]
    assert option(tts["argv"], "--model-name") == "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
    assert option(tts["argv"], "--model-name") == option(render["argv"], "--moss-model")
    assert option(tts["argv"], "--model-path") == str(
        deploy.ROOT / "model/be7766a6735b98bd793f7c79fb720b4d0f5d13b8"
    )


def test_missing_materialized_model_never_falls_back_to_existing_hf_cache(deploy):
    model = deploy.ROOT / "model" / deploy.MODEL_REVISION
    model.rename(model.with_name("saved-model"))
    with pytest.raises(RuntimeError, match="materialized"):
        deploy.build_plan("tts")


@pytest.mark.parametrize("filename", ["configuration_moss_tts.py", "config.json", "model.safetensors"])
def test_materialized_model_rejects_symlinked_code_config_or_weights(deploy, filename):
    model = deploy.ROOT / "model" / deploy.MODEL_REVISION
    target = model / filename
    blob = deploy.HF_HOME / "blob"
    target.rename(blob)
    target.symlink_to(blob)
    with pytest.raises(RuntimeError, match="symlink"):
        deploy.build_plan("tts")


@pytest.mark.parametrize("filename", ["configuration_moss_tts.py", "config.json"])
def test_materialized_model_requires_sibling_configuration_files(deploy, filename):
    (deploy.ROOT / "model" / deploy.MODEL_REVISION / filename).unlink()
    with pytest.raises(RuntimeError, match="materialized"):
        deploy.build_plan("tts")


def test_materialized_model_accepts_hardlinked_weights(deploy):
    model = deploy.ROOT / "model" / deploy.MODEL_REVISION
    target = model / "model.safetensors"
    blob = deploy.HF_HOME / "blob"
    target.rename(blob)
    os.link(blob, target)
    plan = deploy.build_plan("tts")
    assert option(plan["argv"], "--model-path") == str(model)
    assert plan["evidence"]["model_revision"] == "be7766a6735b98bd793f7c79fb720b4d0f5d13b8"


def test_materialized_model_directory_cannot_be_a_symlink(deploy):
    model = deploy.ROOT / "model" / deploy.MODEL_REVISION
    saved = model.with_name("saved-model")
    model.rename(saved)
    model.symlink_to(saved, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        deploy.build_plan("tts")


def test_tts_explicitly_disables_mps_and_pins_eight_cpu_threads(deploy):
    plan = deploy.build_plan("tts")
    # Contract supplied from the pinned upstream schema; runtime is not installed here.
    parser = argparse.ArgumentParser()
    parser.add_argument("--mps", choices=("off", "on", "auto"))
    parsed, _ = parser.parse_known_args(plan["argv"][2:])
    assert parsed.mps == "off"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert plan["environment"][key] == "8"
    assert plan["evidence"]["mps"] == "off"
    assert plan["evidence"]["cpu_threads"] == 8


@pytest.mark.parametrize("source", ["parent", "plan"])
def test_start_refuses_external_mps_pipe_without_spawning(deploy, monkeypatch, source):
    plan = deploy.build_plan("tts")
    if source == "parent":
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/unowned/mps/pipe")
    else:
        plan["environment"]["CUDA_MPS_PIPE_DIRECTORY"] = "/unowned/mps/pipe"
    monkeypatch.setattr(deploy, "verify_runtime", lambda: {})
    monkeypatch.setattr(deploy, "preflight", lambda name: {})
    monkeypatch.setattr(deploy.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError, match="MPS"):
        deploy.start("tts", plan)
    assert not (deploy.ROOT / "tts.pid.json").exists()


def test_both_codec_factories_use_same_pinned_materialized_path_without_config_edits(deploy):
    codec = deploy.ROOT / "codec/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    config = deploy.ROOT / "model" / deploy.MODEL_REVISION / "config.json"
    original = config.read_bytes()
    plan = deploy.build_plan("tts")
    for flag in ("--preprocessing.factory.codec_model_path", "--vocoder.factory.codec_model_path"):
        assert option(plan["argv"], flag) == str(codec)
    assert plan["evidence"]["codec_revision"] == "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    assert plan["evidence"]["resolved_codec_model_path"] == str(codec.resolve())
    assert config.read_bytes() == original


def test_missing_materialized_codec_refuses_launch_plan(deploy):
    codec = deploy.ROOT / "codec/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    codec.rename(codec.with_name("saved-codec"))
    with pytest.raises(RuntimeError, match="materialized codec"):
        deploy.build_plan("tts")


@pytest.mark.parametrize("filename", ["codec.py", "config.json", "model.safetensors"])
def test_materialized_codec_rejects_symlinked_files(deploy, filename):
    codec = deploy.ROOT / "codec/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    target = codec / filename
    blob = deploy.HF_HOME / "codec-blob"
    target.rename(blob)
    target.symlink_to(blob)
    with pytest.raises(RuntimeError, match="symlink"):
        deploy.build_plan("tts")


def test_materialized_codec_requires_config(deploy):
    (deploy.ROOT / "codec/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169/config.json").unlink()
    with pytest.raises(RuntimeError, match="materialized codec"):
        deploy.build_plan("tts")


def test_materialized_codec_accepts_hardlinked_weights(deploy):
    codec = deploy.ROOT / "codec/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    target = codec / "model.safetensors"
    blob = deploy.HF_HOME / "codec-blob"
    target.rename(blob)
    os.link(blob, target)
    plan = deploy.build_plan("tts")
    assert option(plan["argv"], "--vocoder.factory.codec_model_path") == str(codec)


def test_materialized_codec_directory_cannot_be_a_symlink(deploy):
    codec = deploy.ROOT / "codec/f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169"
    saved = codec.with_name("saved-codec")
    codec.rename(saved)
    codec.symlink_to(saved, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        deploy.build_plan("tts")


def test_baseline_survives_old_helper_archival_and_never_overwrites(deploy):
    path = save_baseline(deploy)
    first = deploy.load_baseline()
    path.rename(path.with_name("render.stopped-123.pid.json"))
    assert deploy.load_baseline() == first
    assert (deploy.ROOT / "render.baseline.json").is_file()
    different = baseline(deploy)
    different["argv"] += ["--candidate-profile", "balanced"]
    path.write_text(json.dumps(different))
    with pytest.raises(RuntimeError, match="baseline"):
        deploy.load_baseline(path)


@pytest.mark.parametrize("field,value", [("root", "/unrelated"), ("gpu_uuid", "GPU-other"), ("port", 9999)])
def test_unrelated_baseline_refused(deploy, field, value):
    record = baseline(deploy)
    record[field] = value
    with pytest.raises(RuntimeError):
        deploy.build_plan("render", record)


def test_environment_hash_is_computed_and_supplied_hash_must_match(deploy):
    with pytest.raises(RuntimeError, match="environment"):
        deploy.build_plan("tts", environment_sha256="0" * 64)
    (deploy.ROOT / "environment.lock.txt").unlink()
    with pytest.raises(RuntimeError, match="environment"):
        deploy.build_plan("tts")


def gpu(deploy, memory=0):
    return {"index": 1, "uuid": deploy.GPU_UUID, "memory_mib": memory}


@pytest.mark.parametrize("memory,processes", [(17, []), (0, [101])])
def test_tts_refuses_any_gpu_occupant(deploy, memory, processes):
    with pytest.raises(RuntimeError, match="occupied"):
        deploy.validate_gpu("tts", gpu(deploy, memory), [
            {"gpu_uuid": deploy.GPU_UUID, "pid": pid} for pid in processes
        ])


def test_renderer_only_allows_verified_tts_members_not_an_unrelated_pid(deploy):
    processes = [{"gpu_uuid": deploy.GPU_UUID, "pid": 101}]
    deploy.validate_gpu("render", gpu(deploy, 45000), processes, {101, 102})
    with pytest.raises(RuntimeError, match="unrelated"):
        deploy.validate_gpu("render", gpu(deploy, 45000), processes, {102})
    with pytest.raises(RuntimeError):
        deploy.validate_gpu("render", gpu(deploy, 45000), [], set())
    with pytest.raises(RuntimeError):
        deploy.validate_gpu("tts", {**gpu(deploy), "uuid": "other"}, [])
    deploy.validate_gpu("tts", gpu(deploy), [{"gpu_uuid": "GPU0", "pid": 99}])


@pytest.mark.parametrize("field,value", [("start_ticks", "999"), ("boot_id", "other"),
                                        ("uid", -1), ("pgid", 999), ("sid", 999)])
def test_reused_identity_cannot_stop_new_service(deploy, field, value):
    identity = {"pid": 101, "pgid": 101, "sid": 101, "uid": os.getuid(),
                "start_ticks": "123", "boot_id": "boot"}
    record = {"pid": 101, "identity": identity, "root": str(deploy.ROOT),
              "gpu_uuid": deploy.GPU_UUID, "token": "owned"}
    env = {"STREAMMUSE_DEPLOYMENT_TOKEN": "owned", "STREAMMUSE_DEPLOYMENT_ROOT": str(deploy.ROOT),
           "CUDA_VISIBLE_DEVICES": deploy.GPU_UUID}
    assert deploy.safety.can_stop(record, identity, env, os.getuid())
    assert not deploy.safety.can_stop(record, {**identity, field: value}, env, os.getuid())
    assert not deploy.old_safety.can_stop(record, identity, env, os.getuid())


def test_start_refuses_existing_record_before_launch(deploy, monkeypatch):
    (deploy.ROOT / "tts.pid.json").write_text("{}")
    monkeypatch.setattr(deploy.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn"))
    with pytest.raises(RuntimeError, match="record"):
        deploy.start("tts", deploy.build_plan("tts"))


def test_start_records_exact_launch_identity_and_runtime_evidence(deploy, monkeypatch):
    plan = deploy.build_plan("tts")
    identity = {"pid": 101, "pgid": 101, "sid": 101, "uid": os.getuid(),
                "start_ticks": "123", "boot_id": "boot"}
    monkeypatch.setattr(deploy, "verify_runtime", lambda: {"verified": True})
    monkeypatch.setattr(deploy, "preflight", lambda name: {"gpus": [gpu(deploy)]})
    monkeypatch.setattr(deploy.safety, "proc_identity", lambda pid: identity)
    launches = []
    def spawn(argv, **kwargs):
        launches.append((argv, kwargs))
        return SimpleNamespace(pid=101)
    monkeypatch.setattr(deploy.subprocess, "Popen", spawn)
    record = deploy.start("tts", plan)
    assert record == json.loads((deploy.ROOT / "tts.pid.json").read_text())
    assert record["identity"] == identity
    assert record["runtime_verification"] == {"verified": True}
    assert record["token"] == launches[0][1]["env"]["STREAMMUSE_DEPLOYMENT_TOKEN"]
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][0] == plan["argv"]
    assert record["evidence"] == plan["evidence"]


def test_renderer_dependency_refusal_prevents_gpu_sharing(deploy, monkeypatch):
    save_baseline(deploy)
    (deploy.ROOT / "tts.pid.json").write_text("{}")
    monkeypatch.setattr(deploy.safety, "alive", lambda record: False)
    with pytest.raises(RuntimeError, match="TTS"):
        deploy.preflight("render")


@pytest.mark.parametrize("device", [["--moss-device", "cuda:0"], ["--moss-device=cuda:0"]])
def test_generated_command_passes_real_renderer_parser_and_config_validation(deploy, device):
    from streammuse.presentation.rap_render_server import build_parser, _server_config_from_args

    original = baseline(deploy)
    original["argv"].remove("--moss-device=cuda:0")
    original["argv"] += device + ["--lyric-rhyme-scheme", "chain", "--moss-request-timeout-s", "180"]
    plan = deploy.build_plan("render", original)
    argv = plan["argv"][4:]
    config = _server_config_from_args(build_parser().parse_args(argv))
    assert config.moss_serving_backend == "sglang-omni"
    assert config.moss_device == "external"
    assert config.lyric_continuation and config.moss_sglang_resample_output
    assert config.lyric_rhyme_scheme == "chain"
    assert config.moss_request_timeout_s == 180
    assert config.vllm_model == "qwen-rap"
    assert config.wire_audio_codec == "pcm"
    assert "--trust-remote-code" not in argv
    for name in ("--moss-device", "--moss-mlx-url", "--mlx-version", "--mlx-audio-version",
                 "--mlx-audio-commit", "--mlx-moss-quantization", "--mlx-moss-weights-sha256",
                 "--mlx-moss-audio-tokenizer-revision"):
        action = next(action for action in build_parser()._actions if name in action.option_strings)
        assert action.nargs is None, f"Removal expects one value for {name}"


@pytest.mark.parametrize("requested,expected", [(None, "pcm"), ("pcm", "pcm"), ("opus", "opus")])
def test_render_transport_defaults_pcm_or_explicit_opus_level_five(deploy, requested, expected):
    from streammuse.presentation.rap_render_server import build_parser, _server_config_from_args

    original = baseline(deploy)
    original["argv"] += ["--opus-compression-level=9"]
    saved = deepcopy(original)
    kwargs = {} if requested is None else {"wire_audio_codec": requested}
    plan = deploy.build_plan("render", original, **kwargs)
    config = _server_config_from_args(build_parser().parse_args(plan["argv"][4:]))
    assert config.wire_audio_codec == expected
    assert config.opus_compression_level == 5
    assert ("--opus-compression-level" in plan["argv"]) is (expected == "opus")
    assert "--opus-compression-level=9" not in plan["argv"]
    assert original == saved
    assert plan["evidence"] == deploy.runtime_evidence()


def test_build_plan_rejects_unsupported_transport(deploy):
    with pytest.raises(ValueError, match="codec"):
        deploy.build_plan("render", baseline(deploy), wire_audio_codec="wav")


def test_build_plan_rejects_opus_for_tts(deploy):
    with pytest.raises(ValueError, match="render"):
        deploy.build_plan("tts", wire_audio_codec="opus")


@pytest.mark.parametrize("action", ["plan", "start"])
@pytest.mark.parametrize("codec", [None, "pcm", "opus"])
def test_cli_passes_render_transport_to_plan_and_start(deploy, monkeypatch, capsys, action, codec):
    path = save_baseline(deploy)
    monkeypatch.setattr(deploy, "start", lambda name, plan: plan)
    args = [action, "render", "--baseline", str(path)]
    if codec is not None:
        args += ["--wire-audio-codec", codec]
    assert deploy.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert option(result["argv"], "--wire-audio-codec") == (codec or "pcm")


@pytest.mark.parametrize("action,service", [("plan", "tts"), ("start", "tts"),
                                           ("status", "render"), ("stop", "render")])
def test_cli_transport_option_only_applies_to_plan_start_render(deploy, capsys, action, service):
    with pytest.raises(SystemExit) as error:
        deploy.main([action, service, "--wire-audio-codec", "opus"])
    assert error.value.code == 2
    assert "plan/start render" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [["--moss-device"], ["--moss-device", "--concurrent-bar-generation"]])
def test_missing_option_value_is_rejected_not_consumed_as_another_flag(deploy, suffix):
    original = baseline(deploy)
    original["argv"] += suffix
    with pytest.raises(RuntimeError, match="value"):
        deploy.build_plan("render", original)


def test_json_lock_hashes_exact_bytes_without_assuming_a_schema(deploy):
    content = b'{"python":"3.12.10","packages":{"sglang":"0.5.21"}}\n'
    (deploy.ROOT / "environment.lock.json").write_bytes(content)
    evidence = deploy.runtime_evidence()
    assert evidence["environment_path"].endswith("environment.lock.json")
    assert evidence["environment_sha256"] == hashlib.sha256(content).hexdigest()


def test_stop_uses_pinned_leader_and_child_fds_and_archives_manifest(deploy, monkeypatch):
    import signal

    helper = deploy.safety
    leader = {"pid": 101, "pgid": 101, "sid": 101, "uid": os.getuid(),
              "start_ticks": "123", "boot_id": "boot"}
    child = {**leader, "pid": 102, "start_ticks": "124"}
    record = {"pid": 101, "identity": leader, "root": str(deploy.ROOT),
              "gpu_uuid": deploy.GPU_UUID, "token": "owned"}
    (deploy.ROOT / "tts.pid.json").write_text(json.dumps(record))
    sent = []
    def identity(pid):
        if len(sent) == 2:
            raise ProcessLookupError(pid)
        return leader if pid == 101 else child
    monkeypatch.setattr(helper, "proc_identity", identity)
    monkeypatch.setattr(helper, "proc_environment", lambda pid: {
        "STREAMMUSE_DEPLOYMENT_TOKEN": "owned", "STREAMMUSE_DEPLOYMENT_ROOT": str(deploy.ROOT),
        "CUDA_VISIBLE_DEVICES": deploy.GPU_UUID})
    monkeypatch.setattr(helper, "group_members", lambda record: [leader, child])
    monkeypatch.setattr(helper, "Path", lambda path: SimpleNamespace(iterdir=lambda: []) if path == "/proc" else Path(path))
    monkeypatch.setattr(helper.os, "pidfd_open", lambda pid: pid + 1000, raising=False)
    monkeypatch.setattr(helper.signal, "pidfd_send_signal", lambda fd, sig: sent.append((fd, sig)), raising=False)
    monkeypatch.setattr(helper.os, "close", lambda fd: None)
    assert deploy.main(["stop", "tts"]) == 0
    assert sent == [(1101, signal.SIGTERM), (1102, signal.SIGTERM)]
    assert not (deploy.ROOT / "tts.pid.json").exists()
    archives = list(deploy.ROOT.glob("tts.stopped-*.pid.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == record


def test_runtime_verification_checks_revision_patch_and_package_versions(deploy, monkeypatch):
    commands = []
    def output(argv, **kwargs):
        commands.append(argv)
        if argv[0] == "git":
            return "870d77446b3fae6e4f4f2a6964f7eae2f75e663a\n"
        return '{"sglang-omni":"0.1.8","sglang":"0.5.21"}'
    monkeypatch.setattr(deploy.subprocess, "check_output", output)
    monkeypatch.setattr(deploy.subprocess, "run", lambda argv, **kwargs: commands.append(argv))
    assert deploy.verify_runtime()["patch_reverse_check"] is True
    assert commands[1][-4:] == ["apply", "--reverse", "--check", str(deploy.ROOT / "moss-local-continuation.patch")]
    assert commands[2][0] == str(deploy.ROOT / "env/bin/python")
