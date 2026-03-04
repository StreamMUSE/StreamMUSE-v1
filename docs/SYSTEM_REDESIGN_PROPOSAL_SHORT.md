# StreamMUSE System Redesign — Executive Summary

A concise proposal for a full architectural redesign. For full detail and code examples, see `SYSTEM_REDESIGN_PROPOSAL.md`.

---

## Why Redesign?

**Core problems**

- **Heavy duplication**: Tick loop, inference worker, listening-mode logic, and note conversion are duplicated across CLI, web client, and variants (~1,500 lines). Fixes and features must be done in 3+ places.
- **No abstractions**: No shared interfaces for input, output, or inference. Everything is concrete and tightly coupled; swapping or mocking is hard.
- **Mixed concerns**: Single files (e.g. `client.py`) handle input, output, networking, timing, threading, config, and business logic. Changes in one area risk breaking others.
- **Global state**: Server uses global inference engine and injection state; clients share mutable queues. Hard to test and to run multiple instances.
- **Inconsistent data**: Notes are dicts with varying shapes (event-stream vs duration-based). Conversion and validation are scattered and inconsistent.

**Result**: High maintenance cost, slow feature delivery, and fragile behavior.

---

## Design Direction

**Principles**

- **Clean Architecture**: Domain at the center; application orchestrates; infrastructure and presentation depend inward. Domain has no framework dependencies.
- **SOLID**: Single responsibility, open/closed, Liskov, interface segregation, dependency inversion (depend on abstractions).
- **Single source of truth**: One implementation of tick loop, inference worker, note conversion, and creation logic (factories). CLI and Web reuse the same core.

**Layers (high level)**

1. **Domain** — Musical and timing value objects (e.g. `MusicalEvent`, `Note`, `Tempo`), and protocols for input, output, and inference. No FastAPI, mido, or other frameworks.
2. **Application** — One orchestration service (e.g. `RealTimeMusicService`) that runs the tick loop and coordinates input, inference, and output. Config, factories, and use cases live here.
3. **Infrastructure** — Implementations of the protocols: MIDI/keyboard/file input, audio/MIDI file/console/WebSocket output, HTTP inference client, Stanley/Lekai engines.
4. **Presentation** — Thin entry points: CLI app, Web app (FastAPI), inference server. They build config, call factories, create the service, and start/stop it.

**Dependency rule**: Dependencies point inward only. Domain does not depend on application or infrastructure.

---

## Main Design Choices

- **Unified musical model**: One notion of “event” (e.g. `MusicalEvent` with type, pitch, tick, velocity, etc.) and clear conversion to/from duration-based `Note` in the domain. No ad-hoc dicts.
- **Protocols over inheritance**: Input, output, and inference are protocols (structural interfaces). Implementations can be swapped or mocked without changing callers.
- **Factories**: Input, output, and inference are created from config via factories. Same creation path for CLI and Web; no duplicated wiring.
- **Single tick loop**: One implementation inside the application service. All clients (CLI, Web) use it; no copy-paste of the 500-line loop.
- **Dependency injection**: Service receives input source, output sink, inference engine, tempo, and scheduler in the constructor. Testability and flexibility without global state.

---

## What Gets Eliminated

- **Duplication**: One tick loop, one inference worker, one place for note conversion and for creating input/output. Roughly 1,000+ lines of duplicate logic removed.
- **Scattered config**: One config model (and optionally env/CLI parsers that fill it). Single source of truth for defaults and overrides.
- **Global state**: Engine and injection state become part of a server component’s dependencies; client state lives inside the service instance.

---

## Package Layout (Conceptual)

```
src/streammuse/
  domain/          # events, notes, tempo, scheduler, protocols
  application/     # RealTimeMusicService, config, factories, use cases
  infrastructure/ # input/output/inference implementations
  presentation/   # cli, web, server entry points
```

Tests: unit (domain, application), integration (infrastructure), e2e (presentation).

---

## Migration Approach

- **Phased**: (1) Domain + protocols, (2) Infrastructure, (3) Application service, (4) Presentation (CLI then Web then server). Each phase is testable and shippable.
- **Parallel run**: Keep existing system while the new one is developed. Switch via feature flag or new entry point.
- **Incremental cutover**: Move one entry point (e.g. CLI) to the new stack first; then Web; then deprecate old code.

---

## Risks and Mitigation

- **Breaking behavior**: Mitigate with parallel run, feature flags, and regression tests.
- **Performance**: Benchmark before/after; profile hot paths; keep optimization in mind in the service and infrastructure.
- **Adoption**: Mitigate with short docs, a small “how we structure code” guide, and optional ADRs for key decisions.

---

## Success Criteria

- **Duplication**: One implementation of tick loop, inference worker, and note conversion; duplication reduced to a minimum.
- **Testability**: Domain and application testable with mocks; no need to run real MIDI or server for unit tests.
- **Maintainability**: New feature or bug fix touches one place (or a small, clear set of modules). New input/output/engine = new implementation + factory registration, not edits to core loops.

---

## Recommendation

Proceed with the redesign using Clean Architecture and the layered structure above. Use the full `SYSTEM_REDESIGN_PROPOSAL.md` as the detailed reference; this document is the short summary for alignment and quick reference.
