"""Small real-model prompt comparison; keyword hits are not semantic fulfillment."""
import json
from pathlib import Path

from streammuse.domain.rap.generation import CandidateRequest
from streammuse.domain.rap.story import StoryContext, StoryEvent
from streammuse.infrastructure.inference.local_chat_client import LocalChatModelClient, LocalChatModelClientConfig
from streammuse.infrastructure.rap.generators import _build_independent_messages
from streammuse.infrastructure.rap.prosody import CmuProsodyAnalyzer
from streammuse.infrastructure.rap.templates import BUILTIN_TEMPLATES


def main():
    output = Path("output/research_day_20261105/story_prompt_probe")
    output.mkdir(parents=True, exist_ok=True)
    request = CandidateRequest(
        request_id="story-probe", target_bar=3, topic="deep sea",
        flow_template=next(t for t in BUILTIN_TEMPLATES.values() if len(t.slots) == 9), count=32,
        context_lines=("Ancient secrets whisper deep sea",), seed=11,
        story_context=StoryContext(1, (StoryEvent("first", 1, "A courier loses the map.", 0, "typed"),)),
    )
    original = [dict(message) for message in _build_independent_messages(request)]
    # Preserve the original condition after production adopts the winning prompt.
    original[0]["content"] = (
        "You are a meticulous rap lyric writer and pronunciation-aware prosody checker. "
        "Return exactly one plain lyric line and nothing else."
    )
    original[1]["content"] = original[1]["content"].replace(
        repr(request.story_context.events[-1].text), repr(request.topic), 1,
    ).removesuffix(" Use short common words. Depict the action now, not scenery.")
    prioritized = [dict(message) for message in original]
    prioritized[0]["content"] += (
        " This is directed storytelling. Each line must advance the latest plot direction "
        "using concrete people, actions or objects from it. Older directions remain context. "
        "The topic is only background scenery; do not substitute a generic topic verse for the story."
    )
    prioritized[1]["content"] = prioritized[1]["content"].replace(
        "Write one distinct lyric line about 'deep sea' for bar 3.",
        "Write one distinct story lyric for bar 3. Background setting: 'deep sea'.",
    )
    rescue = [dict(message) for message in prioritized]
    rescue[1]["content"] = rescue[1]["content"].replace("- Ancient secrets whisper deep sea", "- (none)")
    event_subject = [dict(message) for message in original]
    event_subject[0]["content"] = prioritized[0]["content"]
    event_subject[1]["content"] = event_subject[1]["content"].replace(
        "about 'deep sea'", "about 'A courier loses the map.'",
    ).replace("- Ancient secrets whisper deep sea", "- (none)")
    event_subject[1]["content"] += " Use short common words. Depict the action now, not scenery."
    analyzer = CmuProsodyAnalyzer()
    client = LocalChatModelClient(LocalChatModelClientConfig(base_url="http://127.0.0.1:18711/v1", model="qwen-rap", timeout_s=30))
    results = []
    try:
        for name, messages in (("initial", original), ("prioritized", prioritized), ("prioritized_rescue", rescue), ("event_subject", event_subject)):
            response = client.generate_choices(messages, n=32, max_tokens=48, temperature=1.0)
            lines = [{"text": choice.text,
                      "syllables": len(analyzer.analyze(choice.text).syllables),
                      "keyword_hit": any(word in choice.text.lower() for word in ("courier", "map", "lost", "lose"))}
                     for choice in response.choices]
            result = {"variant": name, "messages": messages, "latency_ms": response.latency_ms,
                      "prompt_tokens": response.prompt_tokens, "lines": lines,
                      "syllable_valid": sum(x["syllables"] == 9 for x in lines),
                      "keyword_hits": sum(x["keyword_hit"] for x in lines)}
            results.append(result)
            print(json.dumps({k: v for k, v in result.items() if k not in {"messages", "lines"}}), flush=True)
    finally:
        client.close()
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
