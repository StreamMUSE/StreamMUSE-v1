"""The theme and flow for every bar: a base schedule plus a listener's live edits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from streammuse.domain.rap import FlowTemplate
from streammuse.domain.rap.lyric_stream import PlanEdit


@dataclass(frozen=True)
class BarPlan:
    topic: str
    flow_template: FlowTemplate
    edit_seq: int


@dataclass(frozen=True)
class LivePlan:
    """Immutable: ``with_edit`` returns a new plan.

    ``base`` gives each bar's scheduled topic and flow. An edit overrides the
    topic and/or flow from its ``from_bar`` onward; later edits win.
    """

    base: Callable[[int], tuple[str, FlowTemplate]]
    edits: tuple[PlanEdit, ...] = ()

    @property
    def edit_seq(self) -> int:
        return self.edits[-1].edit_seq if self.edits else 0

    def with_edit(self, edit: PlanEdit) -> "LivePlan":
        if edit.edit_seq <= self.edit_seq:
            raise ValueError("plan edits must have increasing edit_seq")
        return LivePlan(self.base, (*self.edits, edit))

    def for_bar(self, bar: int) -> BarPlan:
        topic, flow = self.base(bar)
        edit_seq = 0
        for edit in self.edits:
            if edit.from_bar > bar:
                continue
            if edit.topic is not None:
                topic = edit.topic
            if edit.flow_template is not None:
                flow = edit.flow_template
            edit_seq = edit.edit_seq
        return BarPlan(topic=topic, flow_template=flow, edit_seq=edit_seq)
