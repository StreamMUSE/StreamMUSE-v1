"""Story mode must make the requested action, not the old topic, the subject."""
from dataclasses import replace

from streammuse.domain.rap.generation import CandidateRequest
from streammuse.domain.rap.story import StoryContext, StoryEvent
from streammuse.infrastructure.rap.generators import _build_messages, _build_independent_messages
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


def test_story_subject_is_latest_event_in_both_prompt_styles():
    context = StoryContext(2, (
        StoryEvent("a", 1, "A courier loses the map.", 0, "typed"),
        StoryEvent("b", 2, "A stranger offers help.", 1, "typed"),
    ))
    request = CandidateRequest("test", 3, "deep sea", BUILTIN_TEMPLATES.values()[0], 32, (), 11,
                               story_context=context)
    for build in (_build_messages, _build_independent_messages):
        system, user = build(request)
        assert "Each line must advance the latest plot direction" in system["content"]
        assert "about 'A stranger offers help.'" in user["content"]
        assert "about 'deep sea'" not in user["content"]
        assert "A courier loses the map." in user["content"]
        assert "Use short common words" in user["content"]
        normal_system, normal_user = build(replace(request, story_context=None))
        assert "directed storytelling" not in normal_system["content"]
        assert "about 'deep sea'" in normal_user["content"]
