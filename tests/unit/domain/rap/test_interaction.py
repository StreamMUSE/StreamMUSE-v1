import pytest

from streammuse.domain.rap.interaction import CaptureWindow, TapRecord, quantize_capture


def test_quantization_ties_earlier_clips_last_tick_and_merges_collisions():
    window = CaptureWindow(bar=4, start_seconds=10.0, end_seconds=14.0)
    taps = tuple(TapRecord(t, t + 0.02, uncertainty_ms=4.0) for t in (10.125, 10.375, 10.38, 13.99))
    captured = quantize_capture(window, taps, template_id="capture-1")
    assert [slot.tick_in_bar for slot in captured.template.slots] == [0, 1, 2, 15]
    assert all(slot.target_stress == 0.5 for slot in captured.template.slots)
    assert [slot.duration_ticks for slot in captured.template.slots] == [1, 1, 13, 1]
    assert captured.raw_taps == taps
    assert captured.template.provenance.kind == "user_capture"


def test_duplicate_ticks_retain_raw_timing_and_diagnostics():
    window = CaptureWindow(0, 0.0, 4.0)
    taps = (TapRecord(1.0, 1.1, client_time_ms=123.0), TapRecord(1.01, 1.2))
    captured = quantize_capture(window, taps, template_id="x")
    assert len(captured.template.slots) == 1
    assert captured.collision_count == 1
    assert captured.raw_taps[0].client_time_ms == 123.0
    assert captured.quantized_ticks == (4, 4)


def test_empty_or_outside_capture_never_creates_empty_flow():
    window = CaptureWindow(0, 10.0, 14.0)
    assert quantize_capture(window, (), template_id="x").template is None
    captured = quantize_capture(window, (TapRecord(9.9, 10.1), TapRecord(14.0, 14.1)), template_id="x")
    assert captured.template is None
    assert captured.rejected_count == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_invalid_timestamps_are_rejected(value):
    with pytest.raises(ValueError):
        TapRecord(value, 1.0)


def test_invalid_capture_window_is_rejected():
    with pytest.raises(ValueError):
        CaptureWindow(0, 3.0, 3.0)
