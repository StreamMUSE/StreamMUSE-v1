"""
Run inference benchmark for models across datasets.
Configurations are specified in config.yaml.
"""
import yaml
from pathlib import Path

# For all scripts, import them here
# This allows us to dynamically load scripts based on the config file
from scripts.template_script import template_benchmark_script
from scripts.fake_offline_script import fake_offline_script
from scripts.fake_offline_script_lantency2 import fake_offline_script_lantency2
from scripts.real_offline_script import real_offline_script 

all_scripts = [
    template_benchmark_script(),
    fake_offline_script(),
    fake_offline_script_lantency2(),
    real_offline_script()
]

def main():
    # Read configuration from config.yaml
    config_path = "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set up paths and parameters from the configuration
    benchmark_name = config.get("benchmark_name", None)

    if benchmark_name is None:
        raise ValueError("Benchmark name must be specified in config.yaml")
    
    output_dir = Path('outputs/' + benchmark_name)

    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Please remove it before running the benchmark.")
        return
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Dynamically create directories for each dataset
    datasets = config.get("datasets_to_run", [])
    if not datasets:
        raise ValueError("No datasets specified in config.yaml")
    input_dataset_dir_paths = [Path('inputs/' + dataset) for dataset in datasets]
    output_dir_paths = [output_dir / dataset for dataset in datasets]
    print("Input dataset directories:", input_dataset_dir_paths)
    print("Output directories:", output_dir_paths)

    # Check scripts
    scripts_to_run = config.get("scripts_to_run", [])
    if not scripts_to_run:
        raise ValueError("No scripts specified in config.yaml")
    scripts = [script for script in all_scripts if script.__class__.__name__ in scripts_to_run]
    if not scripts:
        raise ValueError(f"No valid scripts found in config.yaml. Available scripts: {[script.__class__.__name__ for script in all_scripts]}")
    print("Scripts to run:", [script.__class__.__name__ for script in scripts])

    # Get inference parameters
    inference_params = config.get("inference_params", {})
    if not inference_params:
        raise ValueError("No inference parameters specified in config.yaml")
    prompt_len = inference_params.get("prompt_len", 200)
    n_samples = inference_params.get("n_samples", 1)
    output_len = inference_params.get("output_len", 200)
    temperature = inference_params.get("temperature", 1.0)

    for script in scripts:
        for dataset, input_dir, output_dir in zip(datasets, input_dataset_dir_paths, output_dir_paths):
            print(f"        Running the benchmark script: {dataset}")
            input_midis = list(input_dir.glob('mel/*.mid', ))
            if not input_midis:
                print(f"No MIDI files found in {input_dir}. Skipping dataset {dataset}.")
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            for input_midi in input_midis:
                output_midi = output_dir / input_midi.name
                script.run(
                    model_path=model_path,
                    input_midi=str(input_midi),
                    output_midi=str(output_midi).replace('.mid', 'output') + '.txt',
                    prompt_len=prompt_len,
                    n_samples=n_samples,
                    output_len=output_len,
                    temperature=temperature
                )
            print(f"        Finished running the benchmark script: {dataset}")
            print('-' * 20)
        print(f"    All benchmarks completed successfully for {dataset} dataset.")
    print("All benchmarks completed successfully.")

if __name__ == "__main__":
    main()