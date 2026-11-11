"""HTTP controls for the shared session's experimental inference modes."""

import asyncio
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response


def create_inference_router(runtime: Any) -> APIRouter:
    router = APIRouter(prefix="/api/inference")

    def service():
        instance = getattr(runtime, "inference", None)
        if instance is None:
            raise HTTPException(409, "Inference modes are unavailable in this runtime")
        return instance

    async def invoke(method, *args):
        try:
            return await asyncio.to_thread(getattr(service(), method), *args)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ConnectionError as exc:
            raise HTTPException(503, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    async def payload(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 65536:
                raise HTTPException(413, "Control payload exceeds 64 KiB")
        def invalid_constant(value):
            raise ValueError(f"Non-finite JSON value: {value}")
        try:
            result = json.loads(body or b"{}", parse_constant=invalid_constant)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, "Expected a valid JSON object") from exc
        if not isinstance(result, dict):
            raise HTTPException(422, "Expected a JSON object")
        return result

    @router.get("/state")
    async def state():
        if getattr(runtime, "inference", None) is None:
            return {"supported": False, "state": "unavailable"}
        return await invoke("snapshot")

    @router.post("/start", status_code=202)
    async def start(request: Request):
        data = await payload(request)
        if set(data) - {"owner_id"}:
            raise HTTPException(422, "Unknown start fields")
        owner = data.get("owner_id")
        if "owner_id" in data and (not isinstance(owner, str) or not owner.strip() or len(owner) > 256):
            raise HTTPException(422, "owner_id must be a nonempty string of at most 256 characters")
        return await invoke("start", owner)

    @router.post("/stop", status_code=202)
    async def stop(request: Request):
        if await payload(request):
            raise HTTPException(422, "Stop accepts no fields")
        return await invoke("stop")

    @router.post("/reset")
    async def reset(request: Request):
        if await payload(request):
            raise HTTPException(422, "Reset accepts no fields")
        return await invoke("reset")

    @router.post("/config")
    async def config(request: Request):
        return await invoke("configure", await payload(request))

    @router.post("/steer")
    async def steer(request: Request):
        return await invoke("steer", await payload(request))

    @router.post("/tap")
    async def tap(request: Request):
        return await invoke("tap", await payload(request))

    @router.post("/heartbeat")
    async def heartbeat(request: Request):
        return await invoke("heartbeat", await payload(request))

    @router.get("/export")
    async def export(format: str = "json"):
        if format == "json":
            return JSONResponse(await invoke("export_trace"), headers={
                "Content-Disposition": 'attachment; filename="inference-trace.json"',
            })
        if format == "csv":
            return Response(await invoke("export_metrics_csv"), media_type="text/csv", headers={
                "Content-Disposition": 'attachment; filename="inference-metrics.csv"',
            })
        raise HTTPException(422, "Export format must be json or csv")

    return router
