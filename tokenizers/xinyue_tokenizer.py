from miditok import MusicTokenizer, TokenizerConfig
from miditok import Event
from miditok.constants import SPECIAL_TOKENS

XINYUE_SPECIAL_TOKENS = SPECIAL_TOKENS + ["FRAME"]


class XinyueTokenizerConfig(TokenizerConfig):
    """
    XinyueTokenizerConfig is a configuration class for the XinyueTokenizer.
    It inherits from miditok.TokenizerConfig and can be used to customize the tokenizer's behavior.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You can add any additional configuration parameters here if needed
        self.special_tokens = SPECIAL_TOKENS


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

    def _create_base_vocabulary(self) -> list[str]:
        r"""
        Create the vocabulary, as a list of string tokens.

        Each token is given as the form ``"Type_Value"``, with its type and value
        separated with an underscore. Example: ``Pitch_58``.
        The :class:`miditok.MusicTokenizer` main class will then create the "real"
        vocabulary as a dictionary. Special tokens have to be given when creating the
        tokenizer, and will be added to the vocabulary by
        :class:`miditok.MusicTokenizer`.

        **Attribute control tokens are added when creating the tokenizer by the**
        ``MusicTokenizer.add_attribute_control`` **method.**

        :return: the vocabulary as a list of string.
        """
        vocab = []
        # # Frame
        vocab += ["Frame_None"]

        # # Program
        
        # # Duration

        # # Pitch

        # # Bar
        # if self.config.additional_params["max_bar_embedding"] is not None:
        #     vocab += [
        #         f"Bar_{i}"
        #         for i in range(self.config.additional_params["max_bar_embedding"])
        #     ]
        # else:
        #     vocab += ["Bar_None"]
        # if self.config.additional_params["use_bar_end_tokens"]:
        #     vocab.append("Bar_End")

        # # NoteOn/NoteOff/Velocity
        # self._add_note_tokens_to_vocab_list(vocab)

        # # Position
        # # self.time_division is equal to the maximum possible ticks/beat value.
        # max_num_beats = max(ts[0] for ts in self.time_signatures)
        # num_positions = self.config.max_num_pos_per_beat * max_num_beats
        # vocab += [f"Position_{i}" for i in range(num_positions)]

        # # Add additional tokens
        # self._add_additional_tokens_to_vocab_list(vocab)

        # return vocab

    def _score_to_tokens(self, score, attribute_controls_indexes=None):
        return super()._score_to_tokens(score, attribute_controls_indexes)

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

        if self.config.use_programs:
            first_note_token_type = "Pitch" if self.config.program_changes else "Program"
            dic["Program"] = {"Pitch"}
        else:
            first_note_token_type = "Pitch"
        if self.config.use_velocities:
            dic["Pitch"] = {"Velocity"}
            dic["Velocity"] = {"Duration"} if self.config.using_note_duration_tokens else {first_note_token_type, "Position", "Bar"}
        elif self.config.using_note_duration_tokens:
            dic["Pitch"] = {"Duration"}
        else:
            dic["Pitch"] = {first_note_token_type, "Bar", "Position"}
        if self.config.using_note_duration_tokens:
            dic["Duration"] = {first_note_token_type, "Position", "Bar"}
        dic["Bar"] = {"Position", "Bar"}
        dic["Position"] = {first_note_token_type}
        if self.config.use_pitch_intervals:
            for token_type in ("PitchIntervalTime", "PitchIntervalChord"):
                dic[token_type] = (
                    {"Velocity"}
                    if self.config.use_velocities
                    else {"Duration"}
                    if self.config.using_note_duration_tokens
                    else {
                        first_note_token_type,
                        "PitchIntervalTime",
                        "PitchIntervalChord",
                        "Bar",
                        "Position",
                    }
                )
                if self.config.use_programs and self.config.one_token_stream_for_programs:
                    dic["Program"].add(token_type)
                else:
                    if self.config.using_note_duration_tokens:
                        dic["Duration"].add(token_type)
                    elif self.config.use_velocities:
                        dic["Velocity"].add(token_type)
                    else:
                        dic["Pitch"].add(token_type)
                    dic["Position"].add(token_type)
        if self.config.program_changes:
            dic["Duration" if self.config.using_note_duration_tokens else "Velocity" if self.config.use_velocities else first_note_token_type].add(
                "Program"
            )
            # The first bar may be empty but the Program token will still be present
            if self.config.additional_params["use_bar_end_tokens"]:
                dic["Program"].add("Bar")

        if self.config.use_chords:
            dic["Chord"] = {first_note_token_type}
            dic["Position"] |= {"Chord"}
            if self.config.use_programs:
                dic["Program"].add("Chord")
            if self.config.use_pitch_intervals:
                dic["Chord"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_tempos:
            dic["Position"] |= {"Tempo"}
            dic["Tempo"] = {first_note_token_type, "Position", "Bar"}
            if self.config.use_chords:
                dic["Tempo"] |= {"Chord"}
            if self.config.use_rests:
                dic["Tempo"].add("Rest")  # only for first token
            if self.config.use_pitch_intervals:
                dic["Tempo"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_time_signatures:
            dic["Bar"] = {"TimeSig"}
            if self.config.additional_params["use_bar_end_tokens"]:
                dic["Bar"].add("Bar")
            dic["TimeSig"] = {first_note_token_type, "Position", "Bar"}
            if self.config.use_chords:
                dic["TimeSig"] |= {"Chord"}
            if self.config.use_rests:
                dic["TimeSig"].add("Rest")  # only for first token
            if self.config.use_tempos:
                dic["Tempo"].add("TimeSig")
            if self.config.use_pitch_intervals:
                dic["TimeSig"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_sustain_pedals:
            dic["Position"].add("Pedal")
            if self.config.sustain_pedal_duration:
                dic["Pedal"] = {"Duration"}
                if self.config.using_note_duration_tokens:
                    dic["Duration"].add("Pedal")
                elif self.config.use_velocities:
                    dic["Duration"] = {first_note_token_type, "Bar", "Position"}
                    dic["Velocity"].add("Pedal")
                else:
                    dic["Duration"] = {first_note_token_type, "Bar", "Position"}
                    dic["Pitch"].add("Pedal")
            else:
                dic["PedalOff"] = {
                    "Pedal",
                    "PedalOff",
                    first_note_token_type,
                    "Position",
                    "Bar",
                }
                dic["Pedal"] = {"Pedal", first_note_token_type, "Position", "Bar"}
                dic["Position"].add("PedalOff")
            if self.config.use_chords:
                dic["Pedal"].add("Chord")
                if not self.config.sustain_pedal_duration:
                    dic["PedalOff"].add("Chord")
                    dic["Chord"].add("PedalOff")
            if self.config.use_rests:
                dic["Pedal"].add("Rest")
                if not self.config.sustain_pedal_duration:
                    dic["PedalOff"].add("Rest")
            if self.config.use_tempos:
                dic["Tempo"].add("Pedal")
                if not self.config.sustain_pedal_duration:
                    dic["Tempo"].add("PedalOff")
            if self.config.use_time_signatures:
                dic["TimeSig"].add("Pedal")
                if not self.config.sustain_pedal_duration:
                    dic["TimeSig"].add("PedalOff")
            if self.config.use_pitch_intervals:
                if self.config.sustain_pedal_duration:
                    dic["Duration"] |= {"PitchIntervalTime", "PitchIntervalChord"}
                else:
                    dic["Pedal"] |= {"PitchIntervalTime", "PitchIntervalChord"}
                    dic["PedalOff"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.use_pitch_bends:
            # As a Program token will precede PitchBend otherwise
            # Else no need to add Program as its already in
            dic["PitchBend"] = {first_note_token_type, "Position", "Bar"}
            if self.config.use_programs and not self.config.program_changes:
                dic["Program"].add("PitchBend")
            else:
                dic["Position"].add("PitchBend")
                if self.config.use_tempos:
                    dic["Tempo"].add("PitchBend")
                if self.config.use_time_signatures:
                    dic["TimeSig"].add("PitchBend")
                if self.config.use_sustain_pedals:
                    dic["Pedal"].add("PitchBend")
                    if self.config.sustain_pedal_duration:
                        dic["Duration"].add("PitchBend")
                    else:
                        dic["PedalOff"].add("PitchBend")
            if self.config.use_chords:
                dic["PitchBend"].add("Chord")
            if self.config.use_rests:
                dic["PitchBend"].add("Rest")

        if self.config.use_rests:
            dic["Rest"] = {"Rest", first_note_token_type, "Position", "Bar"}
            dic["Duration" if self.config.using_note_duration_tokens else "Velocity" if self.config.use_velocities else first_note_token_type].add(
                "Rest"
            )
            if self.config.use_chords:
                dic["Rest"] |= {"Chord"}
            if self.config.use_tempos:
                dic["Rest"].add("Tempo")
            if self.config.use_time_signatures:
                dic["Rest"].add("TimeSig")
            if self.config.use_sustain_pedals:
                dic["Rest"].add("Pedal")
                if self.config.sustain_pedal_duration:
                    dic["Duration"].add("Rest")
                else:
                    dic["Rest"].add("PedalOff")
                    dic["PedalOff"].add("Rest")
            if self.config.use_pitch_bends:
                dic["Rest"].add("PitchBend")
            if self.config.use_pitch_intervals:
                dic["Rest"] |= {"PitchIntervalTime", "PitchIntervalChord"}

        if self.config.program_changes:
            for token_type in {
                "Position",
                "Rest",
                "PitchBend",
                "Pedal",
                "PedalOff",
                "Tempo",
                "TimeSig",
                "Chord",
            }:
                if token_type in dic:
                    dic["Program"].add(token_type)
                    dic[token_type].add("Program")

        if self.config.use_pitchdrum_tokens:
            dic["PitchDrum"] = dic["Pitch"]
            for key, values in dic.items():
                if "Pitch" in values:
                    dic[key].add("PitchDrum")

        return dic
