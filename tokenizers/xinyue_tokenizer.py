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

        # Program
        vocab += [f"Program_{program}" for program in self.config.programs]

        # Pitch
        vocab += [f"Pitch_{i}" for i in range(*self.config.pitch_range)]

        # Duration
        vocab += [f"Duration_{'.'.join(map(str, duration))}" for duration in self.durations]


        # Frame
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

        # Frame -> (Program-> Pitch -> Duration, Frame )
        dic["Frame"] = {"Program", "Frame"}
        dic["Program"] = {"Pitch"}
        dic["Pitch"] = {"Duration"}

        # Duration -> (Program, Bar, Frame)
        dic["Duration"] = {"Program", "Frame"}

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

        all_events = []
        current_tick = 0  # 当前处理到的 tick 时间点
        print(time_division)
        ticks_per_frame = self._compute_ticks_per_frame(time_division)

        for event_idx, event in enumerate(events):
            if event.time > current_tick:
                frame_start_tick_to_insert = current_tick 
                while frame_start_tick_to_insert < event.time:
                    self._add_position_event(all_events, frame_start_tick_to_insert, ticks_per_frame)
                    frame_start_tick_to_insert += ticks_per_frame

                current_tick = event.time

            all_events.append(event)
        print(all_events[100:200])   
        return all_events
    
    def _compute_ticks_per_frame(self, time_division: int) -> int:
        """
        计算每 Frame (1/4 拍) 的 tick 数。
        time_division 是每四分音符的tick数 (TPQ)。
        假设 1 拍 = 1 四分音符，所以每拍的 tick 数就是 time_division。
        一个 Frame 是 1/4 拍，所以是 (time_division / 4) ticks。
        """
        ticks_per_frame = time_division // 4 
        if ticks_per_frame == 0:
            ticks_per_frame = 1 
        return ticks_per_frame

    def _add_position_event(
        self, all_events: list[Event], current_tick: int, ticks_per_frame: int
    ):
        """
        添加一个Position事件，作为“frame”事件来插入。
        这里的 current_tick 是即将插入的 Frame 事件的时间。
        Frame 的 value 直接是其在整个序列中的帧索引。
        """
        if ticks_per_frame == 0:
            return
        frame_index = current_tick // ticks_per_frame
        all_events.append(
            Event(
                type_="Frame",
                # value=str(frame_index),
                value="None",
                time=current_tick,
                desc=f"Frame {frame_index} (at {current_tick} ticks)"
            )
        )

if __name__ == "__main__":
    tokenizer = XinyueTokenizer()

    out = tokenizer.encode("datasets/Seperated-POP909-Dataset/mel/002.mid")
    print(out)
