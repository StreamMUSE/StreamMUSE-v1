from streammuse.application.services.real_time_music_service import RealTimeMusicService
from streammuse.domain.interfaces.timing_info import TimingInfo
from streammuse.domain.musical import EventType, MusicalEvent
from streammuse.domain.timing import PlaybackScheduler, Tempo


class _NoopInput:
    def read_events(self):
        return iter([])

    def close(self):
        return None


class _NoopOutput:
    inference_log_detail = "summary"

    def output_event(self, event, source):
        return None

    def output_tick(self, tick, bar, beat):
        return None

    def output_stats(self, **kwargs):
        return None

    def output_status(self, state, message=""):
        return None

    def output_config(self, config):
        return None

    def close(self):
        return None


class _NoopInference:
    class _Cfg:
        model_name = "lekai"
        inference_mode = "sliding_window"

    _config = _Cfg()

    def generate_accompaniment(self, *args, **kwargs):
        return [], TimingInfo(
            request_arrival_time=1.0,
            response_output_time=2.0,
            preprocess_start_time=1.1,
            inference_start_time=1.2,
            inference_end_time=1.3,
            postprocess_start_time=1.4,
        )

    def inject_history(self, *args, **kwargs):
        return None

    def set_injection_offset(self, offset_ticks: int):
        return None

    def clear_history(self):
        return {"success": True, "melody_history": [], "accompaniment_history": []}


def _make_service(detail: str) -> RealTimeMusicService:
    output = _NoopOutput()
    output.inference_log_detail = detail
    return RealTimeMusicService(
        input_source=_NoopInput(),
        inference_engine=_NoopInference(),
        output_sink=output,
        tempo=Tempo(bpm=120.0, ticks_per_beat=4, beats_per_bar=4),
        scheduler=PlaybackScheduler(),
    )


def test_build_inference_log_payload_summary():
    svc = _make_service("summary")
    melody_events = [MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)]
    acc_events = [MusicalEvent(tick=2, pitch=48, event_type=EventType.NOTE_ON, velocity=80, source="model")]
    timing = TimingInfo(
        request_arrival_time=1.0,
        response_output_time=2.0,
        preprocess_start_time=1.1,
        inference_start_time=1.2,
        inference_end_time=1.3,
        postprocess_start_time=1.4,
    )

    request_data, response_data = svc._build_inference_log_payload(
        generation_start_tick=4,
        melody_events=melody_events,
        acc_events=acc_events,
        timing_info=timing,
        request_timestamp=10.0,
        response_timestamp=11.0,
    )

    assert request_data["generation_start_tick"] == 4
    assert request_data["melody_notes_count"] == 1
    assert "melody_notes" not in request_data
    assert response_data["accompaniment_notes_count"] == 1
    assert "accompaniment" not in response_data


def test_build_inference_log_payload_full():
    svc = _make_service("full")
    melody_events = [MusicalEvent(tick=0, pitch=60, event_type=EventType.NOTE_ON, velocity=100)]
    acc_events = [MusicalEvent(tick=2, pitch=48, event_type=EventType.NOTE_ON, velocity=80, source="model")]
    timing = TimingInfo(
        request_arrival_time=1.0,
        response_output_time=2.0,
        preprocess_start_time=1.1,
        inference_start_time=1.2,
        inference_end_time=1.3,
        postprocess_start_time=1.4,
    )

    request_data, response_data = svc._build_inference_log_payload(
        generation_start_tick=4,
        melody_events=melody_events,
        acc_events=acc_events,
        timing_info=timing,
        request_timestamp=10.0,
        response_timestamp=11.0,
    )

    assert request_data["model_name"] == "lekai"
    assert request_data["inference_mode"] == "sliding_window"
    assert request_data["melody_notes"][0]["pitch"] == 60
    assert response_data["accompaniment"][0]["source"] == "model"
    assert response_data["timings"]["inference_end_time"] == 1.3
