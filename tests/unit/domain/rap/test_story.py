from dataclasses import FrozenInstanceError

import pytest

from streammuse.domain.rap.story import StoryContext, StoryEvent
from streammuse.domain.rap.lyric_stream import PlanEdit


def event(index=1, text="a courier loses the map"):
    return StoryEvent(str(index), index, text, 1.0, "typed")


def test_order_immutability_roundtrip_and_idempotence():
    empty = StoryContext()
    first = empty.with_event(event())
    second = first.with_event(event(2, "a stranger offers help"))
    assert empty.events == ()
    assert second.version == 2
    assert second.events == (event(), event(2, "a stranger offers help"))
    assert second.with_event(event()) is second
    assert StoryContext.from_payload(second.to_payload()) == second
    with pytest.raises(FrozenInstanceError):
        first.events = ()
    with pytest.raises(ValueError, match="event_id"):
        second.with_event(event(1, "different"))


def test_ledger_rejects_overflow_without_dropping_old_directions():
    context = StoryContext()
    for index in range(1, 33):
        context = context.with_event(event(index, "plot"))
    with pytest.raises(ValueError, match="32"):
        context.with_event(event(33, "plot"))
    assert len(context.events) == 32
    assert context.with_event(event(text="plot")) is context


@pytest.mark.parametrize("text", ["", "   ", "x" * 513, 123])
def test_event_text_is_bounded(text):
    with pytest.raises(ValueError):
        event(text=text)


def test_plan_edit_extension_roundtrips_without_changing_legacy_payload():
    legacy = PlanEdit(1, 2, topic="sea")
    assert "story_context" not in legacy.to_payload()
    assert PlanEdit.from_payload(legacy.to_payload()) == legacy
    story = StoryContext().with_event(event())
    edit = PlanEdit(2, 3, story_context=story)
    assert PlanEdit.from_payload(edit.to_payload()) == edit
