"""
Run inference benchmark for models across datasets.
Configurations are specified in config.yaml.
"""

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run inference benchmark for models across datasets.")
    parser.add_argument("--midi_path", type=str, required=True, help="Path to the MIDI file")
    parser.add_argument("--prompt_length", type=int, default=200, help="Length of the prompt")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--generate_length", type=float, default=200, help="Length of the generated sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")

    args = parser.parse_args()

    midi_path = args.midi_path
    prompt_length = args.prompt_length
    n_samples = args.n_samples
    generation_length = args.generate_length
    temperature = args.temperature

    ...