"""Write one lyric line per bar, each chosen to follow the lines before it.

This is the lyric half of protocol v3: instead of writing two bars
independently and pairing them, every bar is written after the previous one is
known, so the prompt and the score can ask it to continue (and rhyme with) the
line before.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import time
from types import MappingProxyType

from streammuse.application.rap.chunk_orchestration import verbalize_ascii_digits
from streammuse.application.rap.scoring import rank_candidates
from streammuse.application.rap.service import CandidateGenerator, ProsodyAnalyzer
from streammuse.domain.rap import (
    CandidateBatch,
    CandidateRequest,
    FlowTemplate,
    ScoreWeights,
    normalize_text,
)
from streammuse.domain.rap.lyric_stream import BarLyricRecord

# Bumped whenever the prompt or the selection rule changes, so every lyric
# record says which writer produced it.
LYRIC_WRITER_VERSION = "sequential-lyric-writer.v2"
RHYME_SCHEMES = ("couplet", "chain", "none")


@dataclass(frozen=True)
class LyricWriterPolicy:
    """Candidate counts per bar; H200 uses 32, the Mac will use fewer."""

    candidates: int = 32
    rescue_candidates: int = 16
    rescue_waves: int = 3
    minimum_valid: int = 3
    minimum_score: float = 0.0
    rhyme_scheme: str = "couplet"

    def __post_init__(self) -> None:
        for name in ("candidates", "minimum_valid"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"lyric writer {name} must be a positive integer")
        for name in ("rescue_candidates", "rescue_waves"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"lyric writer {name} must be a non-negative integer")
        if self.rhyme_scheme not in RHYME_SCHEMES:
            raise ValueError(f"rhyme_scheme must be one of {RHYME_SCHEMES}")

    def to_payload(self) -> dict[str, object]:
        return {
            "candidates": self.candidates,
            "rescue_candidates": self.rescue_candidates,
            "rescue_waves": self.rescue_waves,
            "minimum_valid": self.minimum_valid,
            "minimum_score": self.minimum_score,
            "rhyme_scheme": self.rhyme_scheme,
        }


@dataclass(frozen=True)
class LyricWriteRequest:
    """Everything one bar's line depends on."""

    session_id: str
    bar: int
    revision: int
    topic: str
    flow_template: FlowTemplate
    history: tuple[str, ...]
    seed: int
    previous_line: str | None = None
    previous_template: FlowTemplate | None = None


@dataclass(frozen=True)
class WrittenLyric:
    record: BarLyricRecord
    ledger: tuple[Mapping[str, object], ...]
    generation_ms: float
    evaluation_ms: float
    waves: int


class LyricWriteFailed(RuntimeError):
    """No candidate passed the hard gates even after the rescue wave."""

    def __init__(self, message: str, *, ledger: tuple[Mapping[str, object], ...]) -> None:
        super().__init__(message)
        self.ledger = ledger


@dataclass
class _Candidate:
    candidate_id: str
    text: str
    analysis: object
    wave: int
    ledger_index: int


class SequentialLyricWriter:
    """Generate n candidates for one bar, gate them, and pick the best follower."""

    def __init__(
        self,
        generator: CandidateGenerator,
        analyzer: ProsodyAnalyzer,
        weights: ScoreWeights,
        policy: LyricWriterPolicy,
        *,
        provenance: Mapping[str, object] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._generator = generator
        self._analyzer = analyzer
        self._weights = weights
        self._policy = policy
        self._provenance = dict(provenance or {})
        self._monotonic = monotonic

    @property
    def policy(self) -> LyricWriterPolicy:
        return self._policy

    def rhyme_target(self, request: LyricWriteRequest) -> str | None:
        """The previous line to rhyme with, per the policy's scheme."""
        if request.previous_line is None or request.previous_template is None:
            return None
        if self._policy.rhyme_scheme == "none":
            return None
        if self._policy.rhyme_scheme == "couplet" and request.bar % 2 == 0:
            return None
        if _rhyme_group(request.flow_template) is None or (
            _rhyme_group(request.flow_template) != _rhyme_group(request.previous_template)
        ):
            return None
        return request.previous_line

    def write(self, request: LyricWriteRequest) -> WrittenLyric:
        rhyme_with = self.rhyme_target(request)
        anchors: dict[tuple[int, str], tuple[str, ...]] = {}
        if rhyme_with is not None:
            group = _rhyme_group(request.flow_template)
            tail = tuple(self._analyzer.analyze(rhyme_with).end_rhyme_tail)
            if group is not None and tail:
                anchors[(request.bar, group)] = tail

        ledger: list[dict[str, object]] = []
        candidates: list[_Candidate] = []
        seen: set[str] = set()
        generation_ms = evaluation_ms = 0.0
        waves = 0
        result = None
        wave_sizes = (self._policy.candidates,) + (self._policy.rescue_candidates,) * self._policy.rescue_waves
        for wave, count in enumerate(wave_sizes):
            if count <= 0:
                break
            waves += 1
            started = self._monotonic()
            batch, error = self._generate(request, wave, count, rhyme_with)
            generation_ms += max(0.0, (self._monotonic() - started) * 1000.0)
            started = self._monotonic()
            self._record_batch(request, wave, count, batch, error, candidates, seen, ledger)
            if not candidates:
                continue
            result = rank_candidates(
                tuple((item.candidate_id, item.text, item.analysis) for item in candidates),
                template=request.flow_template,
                topic=request.topic,
                history=request.history,
                rhyme_anchors=anchors,
                weights=self._weights,
                minimum_score=self._policy.minimum_score,
                segment_start_bar=request.bar,
                target_bar=request.bar,
            )
            evaluation_ms += max(0.0, (self._monotonic() - started) * 1000.0)
            valid = sum(1 for item in result.evaluations if item.valid)
            if valid >= self._policy.minimum_valid:
                break

        if result is None:
            raise LyricWriteFailed("lyric writer produced no candidates", ledger=_frozen(ledger))
        for candidate, evaluation in zip(candidates, result.evaluations, strict=True):
            row = ledger[candidate.ledger_index]
            row["status"] = "valid" if evaluation.valid else "rejected"
            row["reasons"] = tuple(evaluation.rejection_reasons)
            row["score"] = evaluation.total_score
        if result.selected is None:
            raise LyricWriteFailed(
                f"no selectable candidate for bar {request.bar}: {result.fallback_reason}",
                ledger=_frozen(ledger),
            )
        selected = result.selected
        for candidate in candidates:
            if candidate.candidate_id == selected.candidate_id:
                ledger[candidate.ledger_index]["status"] = "selected"
        record = BarLyricRecord(
            session_id=request.session_id,
            bar=request.bar,
            revision=request.revision,
            topic=request.topic,
            flow_template=request.flow_template,
            text=selected.text,
            scheduled=tuple(selected.scheduled),
            score=float(selected.total_score),
            provenance={
                **self._provenance,
                "writer": LYRIC_WRITER_VERSION,
                "policy": self._policy.to_payload(),
                "seed": request.seed,
                "rhyme_with": rhyme_with,
                "components": {item.name: item.value for item in selected.components},
                "candidates_requested": sum(wave_sizes[:waves]),
                "candidates_valid": sum(1 for item in result.evaluations if item.valid),
                "waves": waves,
            },
        )
        return WrittenLyric(
            record=record,
            ledger=_frozen(ledger),
            generation_ms=generation_ms,
            evaluation_ms=evaluation_ms,
            waves=waves,
        )

    def _generate(
        self, request: LyricWriteRequest, wave: int, count: int, rhyme_with: str | None
    ) -> tuple[CandidateBatch | None, Exception | None]:
        candidate_request = CandidateRequest(
            request_id=f"{request.session_id}:bar:{request.bar}:rev:{request.revision}:wave:{wave}",
            target_bar=request.bar,
            topic=request.topic,
            flow_template=request.flow_template,
            count=count,
            # Rescue waves drop the history: when recent lines share one shape, Qwen
            # returns the same line n times and the wave yields a single candidate.
            context_lines=tuple(request.history[-4:]) if wave == 0 else (),
            seed=request.seed * 100 + wave,
            rhyme_with=rhyme_with,
        )
        try:
            batch = self._generator.generate(candidate_request)
            if not isinstance(batch, CandidateBatch) or batch.request_id != candidate_request.request_id:
                raise ValueError("candidate generator returned a mismatched batch")
            return batch, None
        except Exception as error:  # recorded in the ledger; the rescue wave may still succeed
            return None, error

    def _record_batch(
        self,
        request: LyricWriteRequest,
        wave: int,
        count: int,
        batch: CandidateBatch | None,
        error: Exception | None,
        candidates: list[_Candidate],
        seen: set[str],
        ledger: list[dict[str, object]],
    ) -> None:
        if batch is None:
            ledger.append(
                {
                    "bar": request.bar,
                    "wave": wave,
                    "status": "generation_error",
                    "reasons": (type(error).__name__, str(error)),
                    "requested_count": count,
                }
            )
            return
        if batch.warning:
            ledger.append(
                {"bar": request.bar, "wave": wave, "status": "generation_warning", "reasons": (batch.warning,)}
            )
        for index, raw_text in enumerate(batch.candidates[:count]):
            text = verbalize_ascii_digits(raw_text.strip())
            normalized = normalize_text(text)
            row: dict[str, object] = {
                "bar": request.bar,
                "wave": wave,
                "candidate_id": f"{request.bar}:{request.revision}:{wave}:{index}",
                "text": text,
            }
            if not normalized:
                ledger.append({**row, "status": "malformed", "reasons": ("empty_normalized_text",)})
                continue
            if normalized in seen:
                ledger.append({**row, "status": "duplicate", "reasons": ("duplicate_normalized_text",)})
                continue
            seen.add(normalized)
            if _ends_on_rhyme_label(text, request.flow_template):
                ledger.append({**row, "status": "malformed", "reasons": ("rhyme_label_leak",)})
                continue
            if request.previous_line is not None and _final_word(text) == _final_word(request.previous_line):
                # "... up high" after "... up high" is a repeat, not a rhyme.
                ledger.append({**row, "status": "malformed", "reasons": ("repeats_previous_end_word",)})
                continue
            try:
                analysis = self._analyzer.analyze(text)
            except Exception as analysis_error:
                ledger.append(
                    {**row, "status": "analysis_error", "reasons": (type(analysis_error).__name__,)}
                )
                continue
            ledger.append({**row, "status": "pending", "reasons": ()})
            candidates.append(
                _Candidate(
                    candidate_id=str(row["candidate_id"]),
                    text=text,
                    analysis=analysis,
                    wave=wave,
                    ledger_index=len(ledger) - 1,
                )
            )


def _rhyme_group(template: FlowTemplate) -> str | None:
    return next((slot.rhyme_group for slot in reversed(template.slots) if slot.rhyme_group), None)


def _ends_on_rhyme_label(text: str, template: FlowTemplate) -> bool:
    """A line ending in a bare rhyme-group label ("... endless space A") copied it from the prompt."""
    words = text.rstrip(" .,!?;:").split()
    labels = {slot.rhyme_group for slot in template.slots if slot.rhyme_group}
    return len(words) > 1 and words[-1] in labels


def _final_word(text: str) -> str:
    words = normalize_text(text).split()
    return words[-1] if words else ""


def _frozen(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    return tuple(MappingProxyType(dict(row)) for row in rows)
