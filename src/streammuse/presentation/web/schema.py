"""Pydantic schemas for the web API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class PromptSummary(BaseModel):
    name: str
    melody_path: str
    accompaniment_path: Optional[str] = None
    duration_ticks: Optional[int] = None
    source: Optional[str] = None
    song_number: Optional[str] = None
    melody_notes_count: Optional[int] = None
    accompaniment_notes_count: Optional[int] = None


class PromptsResponse(BaseModel):
    keys: Dict[str, List[PromptSummary]]


class InjectRequest(BaseModel):
    key: str
    strategy: str = "random"
    length_ticks: Optional[int] = None


class InjectResponse(BaseModel):
    injected_melody: int
    injected_acc: int
    message: str
    prompt_name: Optional[str] = None
