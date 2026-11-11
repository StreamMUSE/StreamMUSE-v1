"""Optional directed-story routes. The demo composition owns mounting this router."""

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, StrictStr, Field, ConfigDict

from streammuse.domain.rap.story import StoryConflict, MAX_STORY_TEXT_CHARS, MAX_STORY_EVENT_ID_CHARS


class StoryEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=MAX_STORY_TEXT_CHARS)
    event_id: StrictStr | None = Field(default=None, min_length=1, max_length=MAX_STORY_EVENT_ID_CHARS)
    source: Literal["typed", "microphone"] = "typed"
    session_id: StrictStr | None = Field(default=None, min_length=1, max_length=160)


def create_story_router(runtime) -> APIRouter:
    router = APIRouter()

    def controller():
        # Resolve per request: runtime reset may replace its controller instance.
        value = getattr(runtime, "controller", None)
        if not callable(getattr(value, "submit_story_event", None)):
            raise HTTPException(status_code=409, detail="directed story requires the v3 bar-stream controller")
        return value

    @router.get("/api/story")
    def snapshot():
        return controller().story_snapshot()

    @router.post("/api/story/events")
    def submit(payload: StoryEventInput):
        try:
            return controller().submit_story_event(
                payload.text, event_id=payload.event_id, source=payload.source,
                expected_session_id=payload.session_id,
            )
        except StoryConflict as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (ConnectionError, OSError) as error:
            raise HTTPException(status_code=503, detail="story delivery failed; check the lyric connection") from error

    return router
