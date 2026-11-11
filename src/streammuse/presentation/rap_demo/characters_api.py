"""Optional character experiment routes; the parent application mounts them."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Callable, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from streammuse.application.rap.characters import CharacterExperiment, ExperimentConfig
from streammuse.infrastructure.inference.local_chat_client import (
    LocalChatModelClient,
    LocalChatModelClientConfig,
)


class StartCharactersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["dialogue", "counting"]
    policy: Literal["fixed_turn", "naive", "coordinated"] | None = None
    turns: int = Field(default=8, ge=1, le=12, strict=True)


def create_characters_router(
    *,
    client_factory: Callable[[LocalChatModelClientConfig], LocalChatModelClient] = LocalChatModelClient,
) -> APIRouter:
    """Create an isolated, single-run controller with cleanup on app shutdown.

    No clients are created until POST /start. There is no synthetic fallback.
    The optional factory is an injection boundary for HTTP-transport unit tests.
    """
    active: CharacterExperiment | None = None
    task: asyncio.Task | None = None
    lock = asyncio.Lock()

    def status() -> dict:
        if active is not None:
            return active.snapshot()
        return {
            "run_id": None, "state": "idle", "mode": None, "policy": None,
            "model_url": os.environ.get("STREAMMUSE_RESEARCH_MODEL_URL", "http://127.0.0.1:18711/v1"),
            "model": os.environ.get("STREAMMUSE_RESEARCH_MODEL", "qwen-rap"),
            "actors": [], "accepted": [], "proposals": [], "publication_attempts": [],
            "metrics": {}, "error": None,
        }

    async def stop_active() -> dict:
        async with lock:
            if active is not None and task is not None and not task.done():
                active.stop()
                await asyncio.shield(task)
            return status()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await stop_active()

    router = APIRouter(prefix="/api/characters", tags=["characters"], lifespan=lifespan)

    @router.get("/status")
    async def get_status() -> dict:
        return status()

    @router.post("/start", status_code=202)
    async def start(request: StartCharactersRequest) -> dict:
        nonlocal active, task
        try:
            config = ExperimentConfig(mode=request.mode, policy=request.policy or "fixed_turn", turns=request.turns)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        async with lock:
            if task is not None and not task.done():
                raise HTTPException(status_code=409, detail="a character experiment is already running")
            active = CharacterExperiment(config, client_factory=client_factory)
            task = asyncio.create_task(active.run(), name=f"characters-{active.run_id}")
            return status()

    @router.post("/stop")
    async def stop() -> dict:
        return await stop_active()

    @router.get("/events")
    async def events(after: int = Query(default=0, ge=0), run_id: str | None = None) -> dict:
        if active is None:
            return {"run_id": None, "state": "idle", "events": [], "next_cursor": 0}
        if run_id is not None and run_id != active.run_id:
            after = 0
        return active.event_page(after)

    return router
