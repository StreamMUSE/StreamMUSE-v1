"""
Prompt library management for StreamMUSE client

Handles organization and selection of musical prompts by key.
"""

import os
import json
from typing import Optional, List, Dict
from midi_utils import midi_to_note


class PromptLibrary:
    """Manages a library of musical prompts organized by key"""

    def __init__(self, library_path: str = "prompts"):
        self.library_path = library_path
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load prompt library metadata"""
        metadata_path = os.path.join(self.library_path, "metadata.json")

        if not os.path.exists(metadata_path):
            print(f"Warning: No metadata.json found at {metadata_path}")
            print("Creating empty library...")
            return {}

        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse metadata.json: {e}")
            return {}

    def get_prompts_for_key(self, key: str) -> List[Dict]:
        """
        Get all available prompts for a specific key

        Args:
            key: Key string like "C_major"

        Returns:
            List of prompt metadata dicts
        """
        return self.metadata.get(key, [])

    def select_prompt(self, key: str, strategy: str = "random") -> Optional[Dict]:
        """
        Select a prompt for the given key using specified strategy

        Args:
            key: Key string like "C_major"
            strategy: "random", "first", or index number

        Returns:
            Prompt metadata dict or None if no prompts available
        """
        prompts = self.get_prompts_for_key(key)

        if not prompts:
            print(f"No prompts available for key {key}")
            # Try fallback to C major
            if key != "C_major":
                print("Falling back to C_major...")
                prompts = self.get_prompts_for_key("C_major")

            if not prompts:
                return None

        # Select based on strategy
        if strategy == "random":
            import random
            return random.choice(prompts)
        elif strategy == "first":
            return prompts[0]
        elif isinstance(strategy, int) and 0 <= strategy < len(prompts):
            return prompts[strategy]
        else:
            return prompts[0]  # Default to first

    def load_prompt_notes(
        self,
        prompt: Dict,
        max_ticks: int,
        load_melody: bool = True,
        load_accompaniment: bool = True
    ) -> tuple:
        """
        Load notes from a prompt's MIDI files

        Args:
            prompt: Prompt metadata dict with 'melody_path' and 'accompaniment_path'
            max_ticks: Maximum number of ticks to load
            load_melody: Whether to load melody track
            load_accompaniment: Whether to load accompaniment track

        Returns:
            Tuple of (melody_notes, accompaniment_notes)
        """
        melody_notes = []
        accompaniment_notes = []

        # Load melody
        if load_melody and 'melody_path' in prompt:
            mel_path = prompt['melody_path']
            if os.path.exists(mel_path):
                notes, _, _ = midi_to_note(mel_path, max_tick=max_ticks)
                melody_notes = [n for n in notes if n['tick'] < max_ticks]
                print(f"Loaded {len(melody_notes)} melody notes from {mel_path}")
            else:
                print(f"Warning: Melody file not found: {mel_path}")

        # Load accompaniment
        if load_accompaniment and 'accompaniment_path' in prompt:
            acc_path = prompt['accompaniment_path']
            if os.path.exists(acc_path):
                notes, _, _ = midi_to_note(acc_path, max_tick=max_ticks)
                # Add program field if missing
                accompaniment_notes = [
                    {**n, 'program': 1} if 'program' not in n else n
                    for n in notes if n['tick'] < max_ticks
                ]
                print(f"Loaded {len(accompaniment_notes)} accompaniment notes from {acc_path}")
            else:
                print(f"Warning: Accompaniment file not found: {acc_path}")

        return melody_notes, accompaniment_notes
