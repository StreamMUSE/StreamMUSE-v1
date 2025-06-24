from miditok import MusicTokenizer
from miditok import Event


class XinyueTokenizer(MusicTokenizer):
    """
    XinyueTokenizer is a custom tokenizer for processing MIDI files.
    It inherits from miditok.MusicTokenizer and can be used to convert MIDI files to token sequences and vice versa.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add any additional initialization here if needed

    def _add_time_events(self, events: list[Event], time_division: int = 4) -> list[Event]:
        """
        Adds time events to the list of events based on the specified time division.
        Args:
            events (list[Event]): List of MIDI events.
            time_division (int): The time division to use for adding time events.
        Returns:
            list[Event]: The updated list of events with time events added.
        """
        if not isinstance(events, list):
            raise TypeError("events must be a list of Event objects")
        if not all(isinstance(event, Event) for event in events):
            raise TypeError("All items in events must be instances of Event")   
        
        return super()._add_time_events(events, time_division)

    def _tokens_to_score(self, tokens, programs=None):
        return super()._tokens_to_score(tokens, programs)

    def _create_base_vocabulary(self):
        return super()._create_base_vocabulary()

    def _create_token_types_graph(self):
        return super()._create_token_types_graph()

    def _score_to_tokens(self, score, attribute_controls_indexes=None):
        return super()._score_to_tokens(score, attribute_controls_indexes)

    def _create_track_events(self, track, ticks_per_beat, time_division, ticks_bars, ticks_beats, attribute_controls_indexes=None):
        return super()._create_track_events(track, ticks_per_beat, time_division, ticks_bars, ticks_beats, attribute_controls_indexes)

    def _create_global_events(self, score):
        return super()._create_global_events(score)
