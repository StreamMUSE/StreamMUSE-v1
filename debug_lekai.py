import os
import torch
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from app.inference_engines.transformer_engine_lekai import InferenceEngineLekai, midi_to_note


def test_crash_reproduction():
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path:
        print("Please set CHECKPOINT_PATH")
        return

    print(f"Testing crash reproduction with checkpoint: {checkpoint_path}")

    try:
        engine = InferenceEngineLekai(
            checkpoint_path=checkpoint_path, model_size="llama", inference_mode="sliding_window"
        )

        # 1. Simulate Injection
        injection_file = "input/mel/001.mid"
        injection_length = 128
        
        print(f"Injecting {injection_file}...")
        # Note: midi_to_note in transformer_engine_lekai signature: midi_to_note(midi_path, beat_div=4)
        melody_notes, _, _ = midi_to_note(injection_file)
        acc_file = injection_file.replace("/mel/", "/acc/")
        acc_notes, _, _ = midi_to_note(acc_file)
        
        # Filter
        melody_notes = [n for n in melody_notes if n["tick"] < injection_length]
        acc_notes = [n for n in acc_notes if n["tick"] < injection_length]
        
        engine.clear_history()
        engine.set_injection_offset(injection_length)
        engine.melody_history.extend(melody_notes)
        engine.accompaniment_history.extend(acc_notes)
        
        print(f"Injected {len(melody_notes)} mel notes, {len(acc_notes)} acc notes.")

        # 2. Simulate First Request
        # Client sends generation_start_tick=1 (relative)
        # Server calculates absolute = 1 + 128 = 129
        generation_start_tick = 1
        
        print(f"Simulating request with generation_start_tick={generation_start_tick}...")
        
        # Empty melody notes for the new request (user hasn't played anything yet)
        new_melody_notes = []
        
        result = engine.generate_accompaniment(
            melody_notes=new_melody_notes,
            generation_start_tick=generation_start_tick,
            generation_length_frames=5,
            prompt_length_ticks=0,
        )
        
        print("Generation successful!")
        print(f"Generated {len(result[0])} notes")
        print(f"Generated notes: {result[0]}")

    except Exception as e:
        print("Caught exception:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_crash_reproduction()
