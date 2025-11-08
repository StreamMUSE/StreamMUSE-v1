#!/usr/bin/env python3
"""
Bulk Benchmark Script for StreamMUSE

Runs multiple benchmark experiments with different parameter combinations
based on a YAML configuration file. Supports grid search and custom
experiment definitions.
"""

import argparse
import yaml
import os
import sys
import time
import subprocess
import json
import csv
import shutil
from pathlib import Path
from itertools import product
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import statistics
from datetime import datetime

# Import the existing benchmark function
sys.path.append(str(Path(__file__).parent.parent))
from benchmarking.benchmark import run_benchmark, clear_server_history

class BulkBenchmarkRunner:
    def __init__(self, config_path: str):
        """Initialize the bulk benchmark runner with a configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = []
        self.failed_experiments = []
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ['experiment', 'server', 'benchmark', 'parameters']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in config")
            
            return config
        except Exception as e:
            print(f"❌ Error loading config file {self.config_path}: {e}")
            sys.exit(1)
    
    def _generate_experiment_combinations(self, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        # Convert single values to lists
        param_lists = {}
        for key, value in parameters.items():
            if isinstance(value, list):
                param_lists[key] = value
            else:
                param_lists[key] = [value]
        
        # Generate all combinations
        keys = list(param_lists.keys())
        combinations = []
        
        for values in product(*[param_lists[key] for key in keys]):
            combination = dict(zip(keys, values))
            combinations.append(combination)
        
        return combinations
    
    def _create_experiment_name(self, params: Dict[str, Any], experiment_name: str = None) -> str:
        """Create a descriptive name for an experiment based on its parameters."""
        if experiment_name:
            base_name = experiment_name
        else:
            base_name = self.config['experiment']['name']
        
        # Add parameter values to name
        param_strings = []
        for key, value in params.items():
            if key in ['generation_length_frames', 'prompt_length_ticks', 'tempo']:
                param_strings.append(f"{key}_{value}")
        
        if param_strings:
            return f"{base_name}_{'_'.join(param_strings)}"
        else:
            return base_name
    
    def _run_single_experiment(self, params: Dict[str, Any], experiment_name: str, 
                              output_dir: Path, attempt: int = 1) -> bool:
        """Run a single benchmark experiment with the given parameters."""
        
        print(f"\n{'='*60}")
        print(f"🧪 Running Experiment: {experiment_name}")
        print(f"📊 Parameters: {params}")
        if attempt > 1:
            print(f"🔄 Attempt {attempt}")
        print(f"{'='*60}")
        
        # Create output file path
        output_file = output_dir / f"{experiment_name}.csv"
        
        # Prepare benchmark arguments
        benchmark_args = {
            'server_url': self.config['server']['url'],
            'num_requests': self.config['benchmark']['num_requests'],
            'output_file': str(output_file),
        }
        
        # Add optional benchmark parameters
        for key in ['tempo', 'assumed_network_latency_ms', 'inference_interval_ticks']:
            if key in self.config['benchmark'] and self.config['benchmark'][key] is not None:
                benchmark_args[key] = self.config['benchmark'][key]
        
        # Add experiment-specific parameters
        for key, value in params.items():
            benchmark_args[key] = value
        
        # Clear server history if requested
        if self.config.get('execution', {}).get('clear_server_history', True):
            clear_server_history(self.config['server']['url'])
            time.sleep(1)  # Brief pause after clearing
        
        try:
            # Run the benchmark
            start_time = time.time()
            run_benchmark(**benchmark_args)
            end_time = time.time()
            
            # Verify output file was created
            if not output_file.exists():
                raise FileNotFoundError(f"Benchmark output file not created: {output_file}")
            
            print(f"✅ Experiment completed in {end_time - start_time:.1f} seconds")
            print(f"📁 Results saved to: {output_file}")
            
            # Store experiment metadata
            experiment_info = {
                'name': experiment_name,
                'parameters': params,
                'output_file': str(output_file),
                'duration': end_time - start_time,
                'timestamp': datetime.now().isoformat(),
                'attempt': attempt
            }
            self.results.append(experiment_info)
            
            return True
            
        except Exception as e:
            print(f"❌ Experiment failed: {e}")
            self.failed_experiments.append({
                'name': experiment_name,
                'parameters': params,
                'error': str(e),
                'attempt': attempt
            })
            return False
    
    def _retry_failed_experiments(self, output_dir: Path) -> None:
        """Retry failed experiments if retry is enabled."""
        if not self.config.get('execution', {}).get('retry_failed', False):
            return
        
        max_retries = self.config.get('execution', {}).get('max_retries', 2)
        retry_experiments = self.failed_experiments.copy()
        self.failed_experiments.clear()
        
        for experiment in retry_experiments:
            if experiment['attempt'] < max_retries:
                print(f"\n🔄 Retrying failed experiment: {experiment['name']}")
                success = self._run_single_experiment(
                    experiment['parameters'],
                    experiment['name'],
                    output_dir,
                    experiment['attempt'] + 1
                )
                if not success and experiment['attempt'] + 1 >= max_retries:
                    self.failed_experiments.append(experiment)
    
    def _generate_experiment_summary(self, output_dir: Path) -> None:
        """Generate a summary of all experiments."""
        summary_file = output_dir / "experiment_summary.json"
        
        summary = {
            'experiment_config': self.config['experiment'],
            'total_experiments': len(self.results) + len(self.failed_experiments),
            'successful_experiments': len(self.results),
            'failed_experiments': len(self.failed_experiments),
            'results': self.results,
            'failed': self.failed_experiments,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📋 Experiment summary saved to: {summary_file}")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"📊 EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful_experiments']}")
        print(f"Failed: {summary['failed_experiments']}")
        
        if self.failed_experiments:
            print(f"\n❌ Failed experiments:")
            for exp in self.failed_experiments:
                print(f"  - {exp['name']}: {exp['error']}")
    
    def _combine_results(self, output_dir: Path) -> None:
        """Combine all experiment results into a single CSV file."""
        if not self.config.get('analysis', {}).get('export_combined_csv', True):
            return
        
        print(f"\n🔗 Combining results from {len(self.results)} experiments...")
        
        combined_data = []
        
        for experiment in self.results:
            csv_file = Path(experiment['output_file'])
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Add experiment metadata to each row
                    for param_name, param_value in experiment['parameters'].items():
                        df[param_name] = param_value
                    
                    df['experiment_name'] = experiment['name']
                    df['experiment_duration'] = experiment['duration']
                    
                    combined_data.append(df)
                    
                except Exception as e:
                    print(f"⚠️ Error reading {csv_file}: {e}")
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            combined_file = output_dir / "combined_results.csv"
            combined_df.to_csv(combined_file, index=False)
            print(f"📁 Combined results saved to: {combined_file}")
            
            # Generate basic statistics
            self._generate_basic_statistics(combined_df, output_dir)
    
    def _generate_basic_statistics(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Generate basic statistics from the combined results."""
        stats_file = output_dir / "statistics_summary.txt"
        
        with open(stats_file, 'w') as f:
            f.write("StreamMUSE Bulk Benchmark Statistics Summary\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\\n")
            f.write("-" * 20 + "\\n")
            f.write(f"Total requests: {len(df)}\\n")
            f.write(f"Total experiments: {len(df['experiment_name'].unique())}\\n\\n")
            
            # Round trip time statistics
            if 'round_trip_time' in df.columns:
                rtt_ms = df['round_trip_time'] * 1000
                f.write("ROUND TRIP TIME STATISTICS (ms)\\n")
                f.write("-" * 35 + "\\n")
                f.write(f"Mean: {rtt_ms.mean():.2f}\\n")
                f.write(f"Median: {rtt_ms.median():.2f}\\n")
                f.write(f"Std Dev: {rtt_ms.std():.2f}\\n")
                f.write(f"Min: {rtt_ms.min():.2f}\\n")
                f.write(f"Max: {rtt_ms.max():.2f}\\n")
                
                # Percentiles
                percentiles = self.config.get('analysis', {}).get('calculate_percentiles', [50, 90, 95, 99])
                f.write("\\nPercentiles:\\n")
                for p in percentiles:
                    f.write(f"  {p}th: {rtt_ms.quantile(p/100):.2f}ms\\n")
                f.write("\\n")
            
            # Statistics by parameter combinations
            param_columns = [col for col in df.columns if col in ['generation_length_frames', 'prompt_length_ticks', 'tempo']]
            if param_columns and 'round_trip_time' in df.columns:
                f.write("STATISTICS BY PARAMETER COMBINATION\\n")
                f.write("-" * 40 + "\\n")
                
                grouped = df.groupby(param_columns)['round_trip_time']
                for name, group in grouped:
                    if isinstance(name, tuple):
                        param_str = ", ".join([f"{param_columns[i]}={name[i]}" for i in range(len(name))])
                    else:
                        param_str = f"{param_columns[0]}={name}"
                    
                    group_ms = group * 1000
                    f.write(f"\\n{param_str}:\\n")
                    f.write(f"  Mean: {group_ms.mean():.2f}ms\\n")
                    f.write(f"  Std: {group_ms.std():.2f}ms\\n")
                    f.write(f"  Count: {len(group)}\\n")
        
        print(f"📊 Statistics summary saved to: {stats_file}")
    
    def run_experiments(self) -> None:
        """Run all configured experiments."""
        print(f"🚀 Starting bulk benchmark: {self.config['experiment']['name']}")
        print(f"📖 Description: {self.config['experiment']['description']}")
        
        # Create output directory
        output_dir = Path(self.config['experiment']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration to output directory
        config_copy = output_dir / "config.yaml"
        shutil.copy2(self.config_path, config_copy)
        
        # Generate main experiment combinations
        main_combinations = self._generate_experiment_combinations(self.config['parameters'])
        
        all_experiments = []
        
        # Add main experiments
        for params in main_combinations:
            name = self._create_experiment_name(params)
            all_experiments.append((params, name, None))
        
        # Add additional experiments if specified
        if 'additional_experiments' in self.config:
            for add_exp in self.config['additional_experiments']:
                combinations = self._generate_experiment_combinations(add_exp['parameters'])
                for params in combinations:
                    name = self._create_experiment_name(params, add_exp['name'])
                    all_experiments.append((params, name, add_exp['name']))
        
        print(f"📈 Total experiments to run: {len(all_experiments)}")
        
        # Run experiments
        delay = self.config.get('execution', {}).get('delay_between_experiments', 0)
        
        for i, (params, name, base_name) in enumerate(all_experiments, 1):
            print(f"\\n🔬 Progress: {i}/{len(all_experiments)}")
            
            self._run_single_experiment(params, name, output_dir)
            
            # Delay between experiments
            if delay > 0 and i < len(all_experiments):
                print(f"⏱️ Waiting {delay} seconds before next experiment...")
                time.sleep(delay)
        
        # Retry failed experiments
        self._retry_failed_experiments(output_dir)
        
        # Generate analysis
        if self.config.get('analysis', {}).get('generate_summary', True):
            self._generate_experiment_summary(output_dir)
        
        self._combine_results(output_dir)
        
        print(f"\\n🎉 Bulk benchmark completed!")
        print(f"📁 All results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Run bulk benchmarks for StreamMUSE with configurable parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s experiments/bulk_benchmark_config.yaml
  %(prog)s experiments/my_experiment.yaml --dry-run
        """
    )
    
    parser.add_argument("config", help="Path to YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what experiments would be run without executing them")
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    runner = BulkBenchmarkRunner(args.config)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Showing planned experiments:")
        main_combinations = runner._generate_experiment_combinations(runner.config['parameters'])
        
        print(f"\\nMain experiments ({len(main_combinations)}):")
        for i, params in enumerate(main_combinations, 1):
            name = runner._create_experiment_name(params)
            print(f"  {i}. {name}: {params}")
        
        if 'additional_experiments' in runner.config:
            for add_exp in runner.config['additional_experiments']:
                combinations = runner._generate_experiment_combinations(add_exp['parameters'])
                print(f"\\n{add_exp['name']} experiments ({len(combinations)}):")
                for i, params in enumerate(combinations, 1):
                    name = runner._create_experiment_name(params, add_exp['name'])
                    print(f"  {i}. {name}: {params}")
        
        total = len(main_combinations)
        if 'additional_experiments' in runner.config:
            for add_exp in runner.config['additional_experiments']:
                total += len(runner._generate_experiment_combinations(add_exp['parameters']))
        
        print(f"\\nTotal experiments: {total}")
        print(f"Output directory: {runner.config['experiment']['output_dir']}")
    else:
        runner.run_experiments()

if __name__ == "__main__":
    main()