from dataclasses import replace

import pytest

from streammuse.domain.rap.inference import AudioSpan


def test_nonfinite_stage_timings_cannot_poison_json_snapshots():
    with pytest.raises(ValueError):
        AudioSpan(request_id="r", output_id="o", epoch=0, input_version=0,
                  start_tick=0, duration_ticks=1, sample_rate=100, channels=1,
                  pcm=bytes(100), stage_timings={"tts_ms": float("nan")})
