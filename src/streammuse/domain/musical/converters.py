"""
Conversion between event stream and duration-based notes.

Events → notes is lossy when there are unpaired note_ons: we use a
"close at horizon" policy so each unpaired note_on is treated as ending
at horizon_tick. The client keeps the authoritative event stream; when
the real note_off arrives, the *next* request is built from that stream
and will contain the correct duration for that note. So server history
is effectively corrected on the next request — no separate correction API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from streammuse.domain.musical.events import EventType, MusicalEvent, Note

if TYPE_CHECKING:
    from streammuse.domain.events import Event, EventKind, MusicalEventPayload


def events_to_notes(
    events: List[MusicalEvent],
    horizon_tick: int,
) -> List[Note]:
    """
    Convert an event stream to duration-based notes.

    Pairs note_on with the next note_off for the same pitch. Any note_on
    that has no matching note_off before the end of the list is closed
    at horizon_tick (close-at-horizon policy), so that duration-based
    engines (e.g. Stanley) receive a closed note for this request.

    The client must keep the authoritative event stream. When the real
    note_off for an unpaired note arrives later, the next time a request
    is built from that stream this conversion will produce the correct
    duration. Thus server history is corrected on the next request.

    Events are assumed to be in time order (or at least note_off after
    its note_on). Overlapping notes (same pitch, two note_ons before
    note_off) are not supported; the first note_on is paired with the
    first note_off.

    Args:
        events: Ordered list of note_on / note_off events.
        horizon_tick: Tick at which to close any unpaired note_on.

    Returns:
        List of notes (paired, or closed at horizon for unpaired).
    """
    notes: List[Note] = []
    # Track open note_ons by pitch: list of (tick, event) for that pitch.
    # We only need the most recent open per pitch for simple pairing.
    open_by_pitch: dict[int, MusicalEvent] = {}

    for ev in events:
        if ev.is_placeholder:
            continue
        if ev.event_type == EventType.NOTE_ON:
            # If this pitch was already on (re-trigger), close the previous
            # at this note_on's tick (or at horizon if we're past it).
            if ev.pitch in open_by_pitch:
                prev = open_by_pitch[ev.pitch]
                end_tick = min(horizon_tick, ev.tick)
                if end_tick > prev.tick:
                    notes.append(
                        Note(
                            pitch=prev.pitch,
                            tick=prev.tick,
                            duration=end_tick - prev.tick,
                            velocity=prev.velocity,
                            channel=prev.channel,
                            program=prev.program,
                        )
                    )
            open_by_pitch[ev.pitch] = ev
        else:  # NOTE_OFF
            if ev.pitch in open_by_pitch:
                on_ev = open_by_pitch.pop(ev.pitch)
                if ev.tick > on_ev.tick:
                    notes.append(
                        Note.from_events(on_ev, ev)
                    )
                # else ignore invalid pair (note_off before or at note_on)

    # Close any remaining open note_ons at horizon
    for on_ev in open_by_pitch.values():
        if horizon_tick > on_ev.tick:
            notes.append(
                Note(
                    pitch=on_ev.pitch,
                    tick=on_ev.tick,
                    duration=horizon_tick - on_ev.tick,
                    velocity=on_ev.velocity,
                    channel=on_ev.channel,
                    program=on_ev.program,
                )
            )

    return notes


def musical_event_to_generic_event(musical_event: MusicalEvent) -> "Event":
    """
    Convert a MusicalEvent to a generic Event.

    This bridges the music-specific MusicalEvent to the generic Event model
    that supports multiple event types (musical, text, audio).
    """
    from streammuse.domain.events import Event, EventKind, MusicalEventPayload

    payload = MusicalEventPayload(
        pitch=musical_event.pitch,
        event_type=musical_event.event_type,
        velocity=musical_event.velocity,
        channel=musical_event.channel,
        program=musical_event.program,
        is_placeholder=musical_event.is_placeholder,
    )
    return Event(
        tick=musical_event.tick,
        kind=EventKind.MUSICAL,
        payload=payload,
    )


def generic_event_to_musical_event(event: "Event") -> MusicalEvent:
    """
    Convert a generic Event to a MusicalEvent.

    Raises ValueError if the event is not a musical event.
    """
    from streammuse.domain.events import EventKind, MusicalEventPayload

    if event.kind != EventKind.MUSICAL:
        raise ValueError(f"Cannot convert {event.kind} event to MusicalEvent")
    if not isinstance(event.payload, MusicalEventPayload):
        raise ValueError(f"Invalid payload type for musical event: {type(event.payload)}")

    payload = event.payload
    return MusicalEvent(
        tick=event.tick,
        pitch=payload.pitch,
        event_type=payload.event_type,
        velocity=payload.velocity,
        channel=payload.channel,
        program=payload.program,
        is_placeholder=payload.is_placeholder,
    )
