"""
Run inference benchmark for models across datasets.
Configurations are specified in config.yaml.
"""
import yaml
from pathlib import Path
from tqdm import tqdm

# For all scripts, import them here
# This allows us to dynamically load scripts based on the config file
from scripts.template_script import template_benchmark_script
from scripts.fake_offline_script import fake_offline_script
from scripts.fake_offline_script_lantency2 import fake_offline_script_lantency2
from scripts.real_offline_script import real_offline_script 
from scripts.real_offline_Xinyue_new_script import real_offline_Xinyue_new_script
from scripts.real_offline_Xinyue_new_chord_script import real_offline_Xinyue_new_chord_script

all_scripts = [
    template_benchmark_script(),
    fake_offline_script(),
    fake_offline_script_lantency2(),
    real_offline_Xinyue_new_chord_script(),
    real_offline_script(),
    real_offline_Xinyue_new_script(),
]

def main():
    # Read configuration from config.yaml
    config_path = "config.yaml"
    input_path = Path('inputs')
    output_path = Path('outputs')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set up paths and parameters from the configuration
    benchmark_name = config.get("benchmark_name", None)
    if benchmark_name is None:
        raise ValueError("Benchmark name must be specified in config.yaml")
    
    # Create benchmark output directory
    benchmark_dir = output_path / benchmark_name
    if not benchmark_dir.exists():
        benchmark_dir.mkdir(parents=False, exist_ok=True)

    # Dynamically create directories for each script
    scripts_to_run = config.get("scripts_to_run", [])
    #   If no scripts are specified, raise an error
    if not scripts_to_run:
        raise ValueError("No scripts specified in config.yaml")
    scripts = [script for script in all_scripts if script.__class__.__name__ in scripts_to_run]
    #   If no valid scripts are found, raise an error
    if not scripts:
        raise ValueError(f"No valid scripts found in config.yaml. Available scripts: {[script.__class__.__name__ for script in all_scripts]}")
    #   Create output directories for each script
    output_scripts_dir = [benchmark_dir / script.__class__.__name__ for script in scripts]
    for script in output_scripts_dir:
        if not script.exists():
            script.mkdir(parents=False, exist_ok=True)
    print("Scripts to run:", [script.__class__.__name__ for script in scripts])

    # Dynamically create directories for each dataset
    datasets = config.get("datasets_to_run", [])
    #   If no datasets are specified, raise an error
    if not datasets:
        raise ValueError("No datasets specified in config.yaml")
    #   Create input and output directories for each dataset
    input_dataset_dir_paths = [input_path / dataset for dataset in datasets]
    print("Datasets to run:", datasets)

    # Get inference parameters (Currently unused)
    inference_params = config.get("inference_params", {})
    if not inference_params:
        raise ValueError("No inference parameters specified in config.yaml")
    prompt_len = inference_params.get("prompt_len", 200)
    n_samples = inference_params.get("n_samples", 1)
    output_len = inference_params.get("output_len", 200)
    temperature = inference_params.get("temperature", 1.0)

    # Set up logging for errors
    error_paths_scripts = []

    # Prepare to run benchmarks
    print(f'Running benchmark for {benchmark_name}:')
    print(f'    Scripts: {[script.__class__.__name__ for script in scripts]}')
    print(f'    Datasets: {datasets}')
    print(f'-' * 100)

    output_dirs = []

    for script in scripts:
        print(f"Running benchmark script: {script.__class__.__name__}")
        for dataset, input_dir in zip(datasets, input_dataset_dir_paths):
            print(f"    Running on dataset: {dataset}")
            # Create output path for the script and dataset
            output_dataset_path = benchmark_dir / script.__class__.__name__ / dataset
            output_dirs.append(output_dataset_path)

            # Hardoded exception for specific script and dataset combinations:
            if script.name == 'real_offline_Xinyue_new_chord_script' and dataset != 'pop909_dataset':
                print(f"Skipping {script.__class__.__name__} for dataset {dataset} as it is only applicable to pop909_dataset.")
                continue

            # Find input MIDI files in the dataset directory
            input_midis = list(input_dir.glob('mel/*.mid', ))
            if not input_midis:
                print(f"No MIDI files found in {input_dir}. Skipping dataset {dataset}.")
                continue
            for input_midi in tqdm(input_midis):
                output_midi = output_dataset_path / input_midi.name
                try:
                    script.run(
                        input_midi=str(input_midi),
                        output_midi=str(output_midi).replace('.mid', 'output') + '.mid',
                        prompt_len=prompt_len,
                        n_samples=n_samples,
                        output_len=output_len,
                        temperature=temperature
                    )
                    print(f'input_midi: {input_midi}, output_midi: {output_midi}')
                except Exception as e:
                    print(f"Error running {script.__class__.__name__} for {input_midi}: {e}")
                    error_paths_scripts.append((input_midi, script))
                    continue
                print('-' * 100)
            print(f"        Finished running the benchmark script: {dataset}")
            print('-' * 100)
        print(f"    All benchmarks completed successfully for {dataset} dataset.")
    print("All benchmarks completed successfully.")

    # if errors arise save the error paths and scripts
    if error_paths_scripts:
        error_log_path = benchmark_dir / 'error_log.txt'
        with open(error_log_path, 'w') as f:
            for input_midi, script in error_paths_scripts:
                f.write(f"{script.__class__.__name__},{input_midi}\n")
        print(f"Errors encountered during the benchmark. Check {error_log_path} for details.")
    
    print("Output directories created:")
    for output_dir in output_dirs:
        print("'" + str(output_dir) + "',")

if __name__ == "__main__":
    main()