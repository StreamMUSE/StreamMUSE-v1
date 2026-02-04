"""
Key detection module for StreamMUSE client

Provides lightweight and music21-based key detection algorithms.
"""


def detect_key_lightweight(melody_notes: list) -> str:
    """
    Lightweight key detection based on pitch class distribution.
    Uses Krumhansl-Kessler key profiles.

    Args:
        melody_notes: List of note dicts with 'pitch', 'tick', 'duration'

    Returns:
        Key string like "C_major" or "A_minor"
    """
    if not melody_notes:
        return "C_major"  # Default fallback

    # Count pitch classes (0-11, C=0), weighted by duration
    pitch_class_counts = [0.0] * 12
    for note in melody_notes:
        pc = note['pitch'] % 12
        pitch_class_counts[pc] += note['duration']

    # Normalize
    total = sum(pitch_class_counts)
    if total > 0:
        pitch_class_counts = [c / total for c in pitch_class_counts]

    # Krumhansl-Kessler key profiles
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    # Normalize profiles
    major_sum = sum(major_profile)
    minor_sum = sum(minor_profile)
    major_profile = [p / major_sum for p in major_profile]
    minor_profile = [p / minor_sum for p in minor_profile]

    # Calculate correlation for all 24 keys
    best_key = None
    best_correlation = -1

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    for tonic in range(12):
        for mode, profile in [('major', major_profile), ('minor', minor_profile)]:
            # Rotate profile to match this tonic
            rotated = profile[tonic:] + profile[:tonic]

            # Pearson correlation
            correlation = sum(a * b for a, b in zip(pitch_class_counts, rotated))

            if correlation > best_correlation:
                best_correlation = correlation
                best_key = f"{note_names[tonic]}_{mode}"

    return best_key


def detect_key_music21(melody_notes: list) -> str:
    """
    Robust key detection using music21 library.
    Requires: pip install music21

    Args:
        melody_notes: List of note dicts with 'pitch', 'tick', 'duration'

    Returns:
        Key string like "C_major" or "A_minor"
    """
    try:
        import music21
    except ImportError:
        print("Warning: music21 not installed, falling back to lightweight detection")
        return detect_key_lightweight(melody_notes)

    if not melody_notes:
        return "C_major"

    # Create music21 stream
    stream = music21.stream.Stream()
    for note in melody_notes:
        n = music21.note.Note(note['pitch'])
        # Assuming 4 ticks = 1 quarter note
        n.quarterLength = note['duration'] / 4.0
        stream.append(n)

    # Analyze key
    try:
        key = stream.analyze('key')
        # Convert to our format
        tonic = key.tonic.name.replace('-', 'b')  # Eb -> Eb
        mode = key.mode  # 'major' or 'minor'
        return f"{tonic}_{mode}"
    except Exception as e:
        print(f"Warning: music21 key analysis failed: {e}, using fallback")
        return detect_key_lightweight(melody_notes)
