from __future__ import annotations

import asyncio
import hashlib
import json
import threading

import httpx
import pytest

from streammuse.application.rap.characters import (
    CharacterExperiment,
    CharacterSpec,
    ExperimentConfig,
)
from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatModelClient,
    LocalChatModelClientConfig,
)


def client_factory(handler, configs=None):
    def create(config):
        if configs is not None:
            configs.append(config)
        return LocalChatModelClient(config, transport=httpx.MockTransport(handler))

    return create


def reply(text):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 4},
    })


def observation(request):
    payload = json.loads(request.content)
    return payload, json.loads(payload["messages"][-1]["content"])


def test_dialogue_isolates_personas_and_private_histories_and_publishes_only_accepted():
    requests = []
    configs = []

    def handler(request):
        payload, obs = observation(request)
        requests.append((payload, obs))
        return reply(f"public-{obs['actor_id']}-{obs['version']}")

    experiment = CharacterExperiment(
        ExperimentConfig(mode="dialogue", turns=8),
        actors=(CharacterSpec("a", "PRIVATE_A"), CharacterSpec("b", "PRIVATE_B")),
        client_factory=client_factory(handler, configs),
    )
    result = asyncio.run(experiment.run())
    assert result["state"] == "completed"
    assert [e["actor_id"] for e in result["accepted"]] == ["a", "b"] * 4
    assert len(configs) == 2
    assert configs[0].base_url == configs[1].base_url
    assert configs[0].model == configs[1].model
    assert result["metrics"]["model_calls"] == 8
    for payload, obs in requests:
        assert "n" not in payload
        actor = obs["actor_id"]
        serialized = json.dumps(payload)
        assert ("PRIVATE_B" if actor == "a" else "PRIVATE_A") not in serialized
        assert all(item["actor_id"] == actor for item in obs["private_history"])
    assert requests[1][1]["accepted"][0]["text"] == "public-a-0"
    assert requests[1][1]["private_history"] == []
    for proposal, (payload, _) in zip(result["proposals"], requests):
        digest = hashlib.sha256(json.dumps(
            payload["messages"], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode()).hexdigest()
        assert proposal["context_hash"] == digest
        assert proposal["duration_ms"] >= 0
        assert proposal["prompt_tokens"] == 30


def test_prompt_budget_retains_latest_state_and_reports_omitted_history():
    seen = []

    def handler(request):
        payload, obs = observation(request)
        seen.append(obs)
        assert sum(len(m["content"].encode()) for m in payload["messages"]) <= 1400
        return reply("A long public contribution. " * 20)

    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(mode="dialogue", turns=8),
        client_factory=client_factory(handler),
    ).run())
    assert result["state"] == "completed"
    assert any(obs["omitted_accepted"] > 0 for obs in seen)
    assert seen[-1]["accepted"][-1]["text"].startswith("A long public")
    assert len(result["accepted"][0]["text"]) > 400


@pytest.mark.parametrize("policy", ["fixed_turn", "coordinated"])
def test_counting_requires_ten_in_order_and_both_actors(policy):
    def handler(request):
        _, obs = observation(request)
        accepted = obs["accepted"]
        if accepted and accepted[-1]["actor_id"] == obs["actor_id"]:
            return reply("WAIT")
        return reply(str(obs["version"] + 1))

    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(mode="counting", policy=policy),
        client_factory=client_factory(handler),
    ).run())
    assert result["state"] == "completed"
    assert [e["number"] for e in result["accepted"]] == list(range(1, 11))
    assert set(e["actor_id"] for e in result["accepted"]) == {"a", "b"}
    assert all(x["actor_id"] != y["actor_id"] for x, y in zip(
        result["accepted"], result["accepted"][1:],
    ))
    assert result["metrics"]["valid_prefix_length"] == 10
    assert result["metrics"]["correctness_source"] == "host_referee"


def test_coordinated_records_out_of_order_and_stale_without_accepting_them():
    release_a = threading.Event()

    async def handler(request):
        _, obs = observation(request)
        if obs["version"] == 0 and obs["actor_id"] == "a":
            await asyncio.to_thread(release_a.wait, 2)
        if obs["version"] == 0 and obs["actor_id"] == "b":
            threading.Timer(0.03, release_a.set).start()
        return reply(str(obs["version"] + 1))

    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(mode="counting", policy="coordinated", max_calls=4),
        client_factory=client_factory(handler),
    ).run())
    assert result["proposals"][0]["actor_id"] == "b"
    assert result["accepted"][0]["actor_id"] == "b"
    old = next(p for p in result["proposals"] if p["actor_id"] == "a" and p["context_version"] == 0)
    assert old["outcome"] == "rejected"
    assert old["reason"] == "stale_version"
    assert old["request_id"] not in [e["request_id"] for e in result["accepted"]]


def test_naive_keeps_duplicate_attempts_and_fails_instead_of_repairing():
    barrier = threading.Barrier(2)

    async def handler(request):
        _, obs = observation(request)
        await asyncio.to_thread(barrier.wait, 2)
        return reply(str(obs["version"] + 1))

    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(mode="counting", policy="naive"),
        client_factory=client_factory(handler),
    ).run())
    assert result["state"] == "failed"
    assert len(result["publication_attempts"]) == 2
    assert [p["number"] for p in result["publication_attempts"]] == [1, 1]
    assert len(result["accepted"]) == 1
    assert result["metrics"]["valid_prefix_length"] == 1
    assert result["metrics"]["success"] is False


@pytest.mark.parametrize("text,reason", [("2", "illegal_number"), ("1 then 2", "invalid_format"), ("WAIT", "wait")])
def test_invalid_and_wait_proposals_are_retained(text, reason):
    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(mode="counting", max_calls=2),
        client_factory=client_factory(lambda request: reply(text)),
    ).run())
    assert not result["accepted"]
    assert len(result["proposals"]) == 2
    assert all(p["reason"] == reason for p in result["proposals"])
    assert result["state"] == "failed"


def test_cancellation_aborts_both_requests_and_accepts_no_late_output():
    started = threading.Barrier(3)
    cancelled = []

    async def handler(request):
        _, obs = observation(request)
        try:
            # Stop may arrive before this await resumes after the barrier releases.
            await asyncio.to_thread(started.wait, 2)
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(obs["actor_id"])
            raise
        return reply("1")

    async def scenario():
        experiment = CharacterExperiment(
            ExperimentConfig(mode="counting", policy="coordinated"),
            client_factory=client_factory(handler),
        )
        task = asyncio.create_task(experiment.run())
        await asyncio.to_thread(started.wait, 2)
        experiment.stop()
        return await asyncio.wait_for(task, 2)

    result = asyncio.run(scenario())
    assert result["state"] == "cancelled"
    assert result["accepted"] == []
    assert set(cancelled) == {"a", "b"}
    assert len(result["proposals"]) == 2
    assert all(p["outcome"] == "cancelled" for p in result["proposals"])


def test_timeout_is_terminal_and_cancels_pending_transport():
    async def handler(request):
        await asyncio.sleep(10)
        return reply("1")

    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(mode="counting", policy="coordinated", timeout_s=0.05),
        client_factory=client_factory(handler),
    ).run())
    assert result["state"] == "timed_out"
    assert result["accepted"] == []
    assert len(result["proposals"]) == 2


def test_unavailable_model_is_explicit_error_and_never_synthetic_output():
    def handler(request):
        raise httpx.ConnectError("unavailable", request=request)

    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(), client_factory=client_factory(handler),
    ).run())
    assert result["state"] == "failed"
    assert "model" in result["error"].lower()
    assert result["accepted"] == []
    assert result["proposals"][0]["outcome"] == "error"


@pytest.mark.parametrize("kwargs", [
    {"timeout_s": 61}, {"timeout_s": 0}, {"turns": 0}, {"turns": 13},
    {"mode": "audio"}, {"policy": "unknown"}, {"max_calls": 0},
    {"mode": "dialogue", "policy": "coordinated"},
])
def test_invalid_configs_rejected_before_model_work(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_actor_count_and_identity_are_validated():
    with pytest.raises(ValueError):
        CharacterExperiment(ExperimentConfig(), actors=(CharacterSpec("a", "one"),))
    with pytest.raises(ValueError):
        CharacterExperiment(ExperimentConfig(), actors=(CharacterSpec("a", "one"), CharacterSpec("a", "two")))


def test_environment_endpoint_and_model_are_shared(monkeypatch):
    monkeypatch.setenv("STREAMMUSE_RESEARCH_MODEL_URL", "http://example.test:18711/v1")
    monkeypatch.setenv("STREAMMUSE_RESEARCH_MODEL", "qwen-test")
    configs = []
    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(turns=2),
        client_factory=client_factory(lambda request: reply("hello"), configs),
    ).run())
    assert all(c.base_url == "http://example.test:18711/v1" and c.model == "qwen-test" for c in configs)
    assert result["model"] == "qwen-test"
    assert result["model_url"] == "http://example.test:18711/v1"


def test_terminal_status_waits_for_client_cleanup():
    closing = threading.Event()
    release = threading.Event()

    class DelayedCloseClient(LocalChatModelClient):
        def close(self):
            closing.set()
            release.wait(2)
            super().close()

    async def scenario():
        experiment = CharacterExperiment(
            ExperimentConfig(turns=1),
            client_factory=lambda config: DelayedCloseClient(
                config, transport=httpx.MockTransport(lambda request: reply("hello")),
            ),
        )
        task = asyncio.create_task(experiment.run())
        try:
            assert await asyncio.to_thread(closing.wait, 2)
            assert experiment.snapshot()["state"] == "running"
        finally:
            release.set()
            result = await task
        assert result["state"] == "completed"

    asyncio.run(scenario())


def test_cleanup_failure_is_reported_and_other_client_is_still_closed():
    closed = []

    class FailingCloseClient(LocalChatModelClient):
        def close(self):
            super().close()
            closed.append(self)
            raise RuntimeError("cleanup failure")

    result = asyncio.run(CharacterExperiment(
        ExperimentConfig(turns=1),
        client_factory=lambda config: FailingCloseClient(
            config, transport=httpx.MockTransport(lambda request: reply("hello")),
        ),
    ).run())
    assert result["state"] == "failed"
    assert "cleanup" in result["error"]
    assert len(closed) == 2
