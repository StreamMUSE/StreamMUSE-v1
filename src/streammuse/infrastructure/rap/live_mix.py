"""Bounded callback mixing; PCM validation and capture allocation happen at enqueue."""

from dataclasses import dataclass

import numpy as np

from streammuse.domain.rap.audio import PreparedRapBar


GAIN_NAMES = ("drum_gain", "vocal_gain", "chord_gain")
MAX_MIX_FRAMES = 1024
SMOOTHING_SECONDS = 0.01


@dataclass(frozen=True)
class GainRamp:
    gains: tuple[float, ...]
    revision: int
    remaining: int
    transition_remaining: int = 0


@dataclass(frozen=True)
class CallbackAudio:
    samples: np.ndarray
    stems: tuple[np.ndarray | None, ...]
    limitation: str | None
    capture: np.ndarray | None
    vocal_is_auxiliary: bool = False

    @classmethod
    def prepare(cls, bar: PreparedRapBar, *, capture: bool) -> "CallbackAudio":
        shape = (bar.audio.frame_count, bar.audio.format.channels)
        samples = np.frombuffer(bar.audio.data, dtype=np.float32).reshape(shape)
        stems = []
        for stem in (bar.drum_stem, bar.vocal_stem, bar.chord_stem):
            if stem is None:
                stems.append(None)
                continue
            if stem.format != bar.audio.format or stem.frame_count != bar.audio.frame_count:
                raise ValueError("retained stem format and frame count must match bar audio")
            values = np.frombuffer(stem.data, dtype=np.float32).reshape(shape)
            if not np.isfinite(values).all():
                raise ValueError("retained stems must contain finite PCM samples")
            stems.append(values)
        config = bar.mix_config
        if config is None or stems[0] is None or stems[1] is None:
            stems = [None, None, None]
            limitation = "missing raw stems or mix config; original prepared PCM is played unchanged"
        elif config.chord_loop != "off" and stems[2] is None:
            stems = [None, None, None]
            limitation = "missing active chord stem; original prepared PCM is played unchanged"
        elif config.chord_loop == "off":
            stems[2] = None
            limitation = "chord loop is off; chord gain awaits a future bar with a chord stem"
        else:
            limitation = None
        # FlowEchoRenderer uses the historical vocal slot for captured percussion.
        vocal_is_auxiliary = bar.source in {"flow_echo", "drums_idle"} and stems[1] is not None
        if vocal_is_auxiliary:
            auxiliary_note = "auxiliary percussion is independent of vocal gain"
            limitation = f"{limitation}; {auxiliary_note}" if limitation else auxiliary_note
        return cls(samples, tuple(stems), limitation, np.empty_like(samples) if capture else None,
                   vocal_is_auxiliary=vocal_is_auxiliary)

    def mix(
        self, bar: PreparedRapBar, start: int, count: int,
        requested: tuple[float, ...] | None, revision: int, ramp: GainRamp | None,
    ) -> tuple[np.ndarray, GainRamp | None]:
        if count > MAX_MIX_FRAMES:
            raise ValueError("callback mix block exceeds bounded scratch size")
        if not count:
            return self.samples[start:start], ramp
        if self.stems[0] is None:
            return self.samples[start:start + count], None
        config = bar.mix_config
        baseline = tuple(getattr(config, name) for name in GAIN_NAMES)
        if requested is None:
            return self.samples[start:start + count], GainRamp(baseline, 0, 0)
        previous = np.asarray(ramp.gains if ramp is not None else baseline, dtype=np.float64)
        target = np.asarray(requested, dtype=np.float64)
        smoothing_frames = max(1, round(bar.audio.format.sample_rate_hz * SMOOTHING_SECONDS))
        remaining = (
            ramp.remaining if ramp is not None and ramp.revision == revision
            else smoothing_frames
        )
        transition_remaining = (
            smoothing_frames if ramp is None or ramp.revision == 0 else ramp.transition_remaining
        )
        if remaining:
            fraction = np.minimum(np.arange(1, count + 1) / remaining, 1.0)
            gains = previous + (target - previous) * fraction[:, None]
        else:
            gains = np.broadcast_to(target, (count, 3))
        # Float64 intermediates also keep unusually large but finite raw PCM safe.
        mixed = np.zeros((count, bar.audio.format.channels), dtype=np.float64)
        baseline_mix = np.zeros_like(mixed) if transition_remaining else None
        for index, stem in enumerate(self.stems):
            if stem is not None:
                gain = baseline[index] if index == 1 and self.vocal_is_auxiliary else gains[:, index, None]
                mixed += stem[start:start + count] * gain
                if baseline_mix is not None:
                    baseline_mix += stem[start:start + count].astype(np.float64) * baseline[index]
        mixed *= config.master_gain
        if baseline_mix is not None:
            # Fade out prior whole-bar normalization as well as ramping gains.
            # Otherwise a first gain update could jump from normalized PCM to raw stems.
            weight = np.maximum(transition_remaining - np.arange(1, count + 1), 0) / smoothing_frames
            mixed += (self.samples[start:start + count] - baseline_mix * config.master_gain) * weight[:, None]
        np.clip(mixed, -config.peak_limit, config.peak_limit, out=mixed)
        remaining = max(0, remaining - count)
        applied = requested if not remaining else tuple(gains[-1])
        return mixed.astype(np.float32), GainRamp(applied, revision, remaining, max(0, transition_remaining - count))
