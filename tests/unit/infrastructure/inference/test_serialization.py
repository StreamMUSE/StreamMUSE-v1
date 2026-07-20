from streammuse.domain.musical import EventType
from streammuse.infrastructure.inference.serialization import event_from_dict


def test_event_from_dict_handles_null_velocity_channel_program():
    event = event_from_dict(
        {
            "type": "note_on",
            "pitch": 60,
            "tick": 4,
            "velocity": None,
            "channel": None,
            "program": None,
        }
    )

    assert event.event_type == EventType.NOTE_ON
    assert event.pitch == 60
    assert event.tick == 4
    assert event.velocity == 100
    assert event.channel == 0
    assert event.program == 0


def test_event_from_dict_handles_missing_optional_fields():
    event = event_from_dict({"type": "note_off", "pitch": 64, "tick": 8})

    assert event.event_type == EventType.NOTE_OFF
    assert event.pitch == 64
    assert event.tick == 8
    assert event.velocity == 0
    assert event.channel == 0
    assert event.program == 0
