"""Dense terminal rendering for realtime rap decisions and delivery."""

from __future__ import annotations

from typing import Callable

from streammuse.domain.rap import RapEvent, RapEventType


class TerminalRapSink:
    def __init__(self, detail: str = "full", *, write: Callable[[str], None] | None = None) -> None:
        if detail not in {"summary", "candidates", "full"}:
            raise ValueError("terminal detail must be summary, candidates, or full")
        self._detail = detail
        self._write = write or (lambda line: print(line, flush=True))

    def __call__(self, event: RapEvent) -> None:
        kind = event.event_type
        payload = event.payload
        location = self._location(event)
        if kind == RapEventType.SESSION_STARTED:
            self._write(f"SESSION start tempo={payload.get('tempo_bpm')} max_bars={payload.get('max_bars')}")
        elif kind == RapEventType.SESSION_STOPPED:
            self._write("SESSION stop")
        elif kind == RapEventType.BAR_RESERVED:
            self._write(
                f"RESERVE {location} source={payload.get('source')} topic={payload.get('topic')} "
                f"template={payload.get('template_id')} fallback_reason={payload.get('fallback_reason')} "
                f"text={payload.get('text')!r}"
            )
        elif kind == RapEventType.BAR_PLANNING_STARTED:
            self._write(
                f"PLAN {location} request={event.request_id} topic={payload.get('topic')} "
                f"template={payload.get('template_id')} candidates={payload.get('candidate_count')}"
            )
        elif kind == RapEventType.CANDIDATE_BATCH_RECEIVED:
            self._write(
                f"BATCH {location} source={payload.get('source')} count={payload.get('candidate_count')} "
                f"latency_ms={payload.get('latency_ms')} late={payload.get('late')} error={payload.get('error_type')}"
            )
            if self._detail == "full":
                self._write(f"  prompt={payload.get('prompt')} raw_response={payload.get('raw_response')!r}")
        elif kind == RapEventType.CANDIDATE_EVALUATED and self._detail != "summary":
            self._write(
                f"CANDIDATE {location} id={payload.get('candidate_id')} valid={payload.get('valid')} "
                f"selected={payload.get('selected')} total={payload.get('total_score')} text={payload.get('text')!r}"
            )
            self._write(f"  rejection_reasons={payload.get('rejection_reasons')} oov_words={payload.get('oov_words')}")
            if self._detail == "full":
                for component in payload.get("components", []):
                    self._write(
                        f"  {component.get('name')}: value={component.get('value')} weight={component.get('weight')} "
                        f"contribution={component.get('contribution')}"
                    )
        elif kind == RapEventType.GENERATION_FAILED:
            self._write(
                f"ERROR {location} type={payload.get('error_type')} message={payload.get('error_message')!r} "
                f"late={payload.get('late')}"
            )
        elif kind == RapEventType.BAR_REPLACED:
            self._write(
                f"SELECT {location} source={payload.get('source')} total={payload.get('total_score')} "
                f"text={payload.get('text')!r}"
            )
        elif kind == RapEventType.BAR_FROZEN:
            self._write(
                f"FREEZE {location} source={payload.get('source')} fallback={payload.get('fallback')} "
                f"fallback_reason={payload.get('fallback_reason')} text={payload.get('text')!r}"
            )
        elif kind == RapEventType.FALLBACK_ACTIVATED:
            self._write(f"FALLBACK {location} reason={payload.get('fallback_reason')}")
        elif kind == RapEventType.TICK:
            if self._detail == "full" or payload.get("tick_in_beat") == 0:
                self._write(
                    f"TICK {location} beat={int(payload.get('beat', 0)) + 1} "
                    f"subdivision={int(payload.get('tick_in_beat', 0)) + 1}"
                )
        elif kind == RapEventType.SYLLABLE_EMITTED:
            self._write(
                f"SYLLABLE {location} label={payload.get('label')!r} stressed={payload.get('stressed')} "
                f"jitter_ms={float(payload.get('jitter_ms', 0.0)):.3f}"
            )

    @staticmethod
    def _location(event: RapEvent) -> str:
        bar = "-" if event.bar is None else str(event.bar)
        tick = "-" if event.tick is None else str(event.tick)
        return f"bar={bar} tick={tick}"
