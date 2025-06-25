from miditok import MusicTokenizer, TokenizerConfig
from miditok import Event
from miditok.constants import TIME_SIGNATURE, SPECIAL_TOKENS
from symusic import (
    TimeSignature,
)
import numpy as np
from miditok.utils import compute_ticks_per_bar, compute_ticks_per_beat

XINYUE_SPECIAL_TOKENS = SPECIAL_TOKENS.copy()


class XinyueTokenizerConfig(TokenizerConfig):
    """
    XinyueTokenizerConfig is a configuration class for the XinyueTokenizer.
    It inherits from miditok.TokenizerConfig and can be used to customize the tokenizer's behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use the custom special tokens including "FRAME"
        self.special_tokens = XINYUE_SPECIAL_TOKENS
        # Add frame-specific config. Here, we set a frame to be 120 ticks.
        # If your MIDI has 480 ticks per beat (TPB), this corresponds to a 16th note.
        self.additional_params["frame_duration_ticks"] = 120


class XinyueTokenizer(MusicTokenizer):
    """
    XinyueTokenizer is a custom tokenizer for processing MIDI files.
    It inherits from miditok.MusicTokenizer and can be used to convert MIDI files to token sequences and vice versa.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add any additional initialization here if needed

    def _create_base_vocabulary(self) -> list[str]:
        r"""
        Create the vocabulary, as a list of string tokens.
        """
        vocab = []

        # Bar
        vocab += ["Bar_None"]

        # Program
        vocab += [f"Program_{program}" for program in self.config.programs]

        # Pitch
        vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # Duration
        vocab += [f"Duration_{'.'.join(map(str, duration))}" for duration in self.durations]

        # # Position
        # max_num_beats = max(ts[0] for ts in self.time_signatures)
        # num_positions = self.config.max_num_pos_per_beat * max_num_beats
        # vocab += [f"Position_{i}" for i in range(num_positions)]

        # # Frame

        vocab += ["Frame_None"]

        return vocab

    def _create_track_events(self, track, ticks_per_beat, time_division, ticks_bars, ticks_beats, attribute_controls_indexes=None):
        return super()._create_track_events(track, ticks_per_beat, time_division, ticks_bars, ticks_beats, attribute_controls_indexes)

    def _create_global_events(self, score):
        return super()._create_global_events(score)

    def _create_token_types_graph(self) -> dict[str, set[str]]:
        r"""
        Return a graph/dictionary of the possible token types successions.

        :return: the token types transitions dictionary.
        """
        dic: dict[str, set[str]] = {}

        # # Position -> (Program-> Pitch -> Duration, Position )
        # dic["Position"] = {"Program", "Postion"}
        # dic["Program"] = {"Pitch"}
        # dic["Pitch"] = {"Duration"}

        # # Duration -> (Program, Bar, Position)
        # dic["Duration"] = {"Program", "Bar", "Position"}

        # dic["Bar"] = {"Position"}

        # Frame -> (Program-> Pitch -> Duration, Frame )
        dic["Frame"] = {"Program", "Postion"}
        dic["Program"] = {"Pitch"}
        dic["Pitch"] = {"Duration"}

        # Duration -> (Program, Bar, Frame)
        dic["Duration"] = {"Program", "Bar", "Frame"}

        dic["Bar"] = {"Frame"}

        return dic

    def _tokens_to_score(self, tokens, programs=None):
        return super()._tokens_to_score(tokens, programs)

    def _score_to_tokens(self, score, attribute_controls_indexes=None):
        return super()._score_to_tokens(score, attribute_controls_indexes)

    def _add_time_events(self, events: list[Event], time_division: int) -> list[list[Event]]:
        r"""
        Create the time events from a list of global and track events.

        Internal method intended to be implemented by child classes.
        The returned sequence is the final token sequence ready to be converted to ids
        to be fed to a model.

        :param events: sequence of global and track events to create tokens time from.
        :param time_division: time division in ticks per quarter of the
            ``symusic.Score`` being tokenized.
        :return: the same events, with time events inserted.
        """
        # Add time events

        duration_offset = 0
        if self.config.use_velocities:
            duration_offset += 1
        if self.config.using_note_duration_tokens:
            duration_offset += 1
        all_events = []
        current_bar = -1
        bar_at_last_ts_change = 0
        previous_tick = -1
        previous_note_end = 0
        tick_at_last_ts_change = tick_at_current_bar = 0
        current_time_sig = TIME_SIGNATURE
        if self.config.log_tempos:
            # pick the closest to the default value
            current_tempo = float(self.tempos[(np.abs(self.tempos - self.default_tempo)).argmin()])
        else:
            current_tempo = self.default_tempo
        current_program = None
        ticks_per_bar = compute_ticks_per_bar(TimeSignature(0, *current_time_sig), time_division)
        ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
        ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
        # First look for a TimeSig token, if any is given at tick 0, to update
        # current_time_sig
        if self.config.use_time_signatures:
            for event in events:
                # There should be a TimeSig token at tick 0
                if event.type_ == "TimeSig":
                    current_time_sig = list(map(int, event.value.split("/")))
                    ticks_per_bar = compute_ticks_per_bar(TimeSignature(event.time, *current_time_sig), time_division)
                    ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
                    ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
                    break
        # Then look for a Tempo token, if any is given at tick 0, to update
        # Add the time events
        for e, event in enumerate(events):
            if event.type_ == "Tempo":
                current_tempo = event.value
            elif event.type_ == "Program":
                current_program = event.value
                continue
            if event.time != previous_tick:
                # Bar
                num_new_bars = bar_at_last_ts_change + (event.time - tick_at_last_ts_change) // ticks_per_bar - current_bar
                if num_new_bars >= 1:
                    for i in range(num_new_bars):
                        all_events.append(
                            Event(
                                type_="Bar",
                                value="None",
                            )
                        )
                    current_bar += num_new_bars
                    tick_at_current_bar = tick_at_last_ts_change + (current_bar - bar_at_last_ts_change) * ticks_per_bar

                # Position
                if event.type_ != "TimeSig":
                    pos_index = (event.time - tick_at_current_bar) // ticks_per_pos
                    all_events.append(
                        self.__create_cp_token(
                            event.time,
                            pos=pos_index,
                            chord=event.value if event.type_ == "Chord" else None,
                            tempo=current_tempo if self.config.use_tempos else None,
                            desc="Position",
                        )
                    )

                previous_tick = event.time

            # Update time signature time variables, after adjusting the time (above)
            if event.type_ == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                bar_at_last_ts_change += (event.time - tick_at_last_ts_change) // ticks_per_bar
                tick_at_last_ts_change = event.time
                ticks_per_bar = compute_ticks_per_bar(TimeSignature(event.time, *current_time_sig), time_division)
                ticks_per_beat = compute_ticks_per_beat(current_time_sig[1], time_division)
                ticks_per_pos = ticks_per_beat // self.config.max_num_pos_per_beat
                # We decrease the previous tick so that a Position token is enforced
                # for the next event
                previous_tick -= 1

            # Convert event to CP Event
            # Update max offset time of the notes encountered
            if event.type_ in {"Pitch", "PitchDrum"} and e + duration_offset < len(events):
                all_events.append(
                    self.__create_cp_token(
                        event.time,
                        pitch=event.value,
                        vel=events[e + 1].value if self.config.use_velocities else None,
                        dur=events[e + duration_offset].value if self.config.using_note_duration_tokens else None,
                        program=current_program,
                        pitch_drum=event.type_ == "PitchDrum",
                    )
                )
                previous_note_end = max(previous_note_end, event.desc)
            elif event.type_ in [
                "Program",
                "Tempo",
                "TimeSig",
                "Chord",
            ]:
                previous_note_end = max(previous_note_end, event.time)

        return all_events


if __name__ == "__main__":
    tokenizer = XinyueTokenizer()

    out = tokenizer.encode("datasets/Seperated-POP909-Dataset/mel/002.mid")
    print(out)
