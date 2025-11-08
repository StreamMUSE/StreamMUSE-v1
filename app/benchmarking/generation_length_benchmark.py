#!/usr/bin/env python3
"""
Enhanced benchmark script for analyzing generation length effects on StreamMUSE latency.

This script works with the existing server by running multiple benchmark sessions
with different GENERATION_LENGTH_FRAMES environment variable settings.
It does not modify any existing application files.
"""

import argparse
import subprocess
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
import statistics
from typing import List, Dict, Any
import shutil

class GenerationLengthBenchmark:
    """
    Manages generation length parameter sweep benchmarks by coordinating
    multiple runs of the existing benchmark.py script with different server configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config['output_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        self.raw_data_dir = self.results_dir / "raw_data"
        self.analysis_dir = self.results_dir / "analysis" 
        self.plots_dir = self.results_dir / "plots"
        
        self.raw_data_dir.mkdir(exist_ok=True)
        self.analysis_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)
        
        self.all_results = []
        self.summary_stats = []
        
    def run_benchmark_for_generation_length(self, generation_length: int) -> bool:
        """
        Run benchmark for a specific generation length by starting server 
        with appropriate environment variables and running existing benchmark.py
        """
        print(f"\n{'='*60}")
        print(f"Testing Generation Length: {generation_length} frames")
        print(f"{'='*60}")
        
        # Define output files for this generation length
        csv_file = self.raw_data_dir / f"gen_length_{generation_length}.csv"
        json_file = self.raw_data_dir / f"gen_length_{generation_length}.json"
        
        # Prepare benchmark command
        benchmark_cmd = [
            sys.executable, "app/benchmark.py",
            "--server_url", self.config['server_url'],
            "--num_requests", str(self.config['requests_per_length']),
            "--output_file", str(csv_file)
        ]
        
        print(f"Running benchmark with {self.config['requests_per_length']} requests...")
        print(f"Command: {' '.join(benchmark_cmd)}")
        
        try:
            # Run the existing benchmark script
            result = subprocess.run(
                benchmark_cmd,
                capture_output=True,
                text=True,
                timeout=self.config['timeout_seconds']
            )
            
            if result.returncode != 0:
                print(f"❌ Benchmark failed for generation length {generation_length}")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
            print(f"✅ Benchmark completed for generation length {generation_length}")
            
            # Verify output files exist
            if not csv_file.exists():
                print(f"❌ CSV output file not found: {csv_file}")
                return False
                
            # Load and validate the results
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                print(f"❌ No data in CSV file for generation length {generation_length}")
                return False
                
            print(f"📊 Collected {len(df)} successful requests")
            
            # Add generation length column to the data
            df['generation_length'] = generation_length
            
            # Re-save with generation length column
            df.to_csv(csv_file, index=False)
            
            # Add to our combined results
            self.all_results.append(df)
            
            # Calculate summary statistics
            summary = self._calculate_summary_stats(df, generation_length)
            self.summary_stats.append(summary)
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timed out for generation length {generation_length}")
            return False
        except Exception as e:
            print(f"❌ Error running benchmark for generation length {generation_length}: {e}")
            return False
    
    def _calculate_summary_stats(self, df: pd.DataFrame, generation_length: int) -> Dict[str, Any]:
        """Calculate summary statistics for a single generation length."""
        
        def safe_stats(series):
            """Calculate stats safely, handling empty series."""
            if len(series) == 0:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'p95': 0, 'p99': 0}
            return {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(), 
                'max': series.max(),
                'median': series.median(),
                'p95': series.quantile(0.95),
                'p99': series.quantile(0.99)
            }
        
        summary = {
            'generation_length': generation_length,
            'num_requests': len(df),
            'num_successful_requests': len(df),  # All rows in CSV are successful
        }
        
        # Add statistics for each timing metric
        metrics = ['round_trip_time', 'server_processing_duration', 'inference_duration', 
                  'preprocess_duration', 'postprocess_duration', 'total_network_latency']
        
        for metric in metrics:
            if metric in df.columns:
                stats = safe_stats(df[metric])
                for stat_name, value in stats.items():
                    summary[f"{metric}_{stat_name}"] = value
        
        # Add notes generated stats
        if 'num_generated_notes' in df.columns:
            notes_stats = safe_stats(df['num_generated_notes'])
            for stat_name, value in notes_stats.items():
                summary[f"num_generated_notes_{stat_name}"] = value
                
        return summary
    
    def run_parameter_sweep(self) -> bool:
        """
        Run the complete parameter sweep across all specified generation lengths.
        Note: This requires manual server restart between generation lengths.
        """
        print("🚀 Starting Generation Length Parameter Sweep")
        print(f"Testing generation lengths: {self.config['generation_lengths']}")
        print(f"Requests per length: {self.config['requests_per_length']}")
        print(f"Output directory: {self.results_dir}")
        
        if not self.config['auto_server_restart']:
            print("\n⚠️  MANUAL MODE: You need to restart the server with different")
            print("   GENERATION_LENGTH_FRAMES values between each test.")
            print("   The script will pause and wait for your confirmation.")
        
        successful_runs = 0
        total_runs = len(self.config['generation_lengths'])
        
        for i, gen_length in enumerate(self.config['generation_lengths']):
            if not self.config['auto_server_restart']:
                if i > 0:  # Skip prompt for first run
                    print(f"\n⏸️  Please restart the server with:")
                    print(f"   GENERATION_LENGTH_FRAMES={gen_length} uvicorn app.server:app --host 0.0.0.0 --port {self.config['server_port']}")
                    print(f"   Then press Enter to continue...")
                    input()
                else:
                    print(f"\n🔧 Please ensure server is running with GENERATION_LENGTH_FRAMES={gen_length}")
                    print(f"   Press Enter when ready...")
                    input()
            
            # Test server connection
            if not self._test_server_connection():
                print(f"❌ Cannot connect to server. Skipping generation length {gen_length}")
                continue
                
            # Run benchmark for this generation length
            if self.run_benchmark_for_generation_length(gen_length):
                successful_runs += 1
            else:
                print(f"⚠️  Failed to complete benchmark for generation length {gen_length}")
                
            # Small delay between runs
            time.sleep(2)
        
        print(f"\n🏁 Parameter sweep completed!")
        print(f"   Successful runs: {successful_runs}/{total_runs}")
        
        if successful_runs > 0:
            self._export_combined_results()
            if self.config['generate_plots']:
                self._generate_visualizations()
            if self.config['generate_report']:
                self._generate_analysis_report()
            
        return successful_runs > 0
    
    def _test_server_connection(self) -> bool:
        """Test if server is responding."""
        import requests
        try:
            # Test with a simple request to clear_history endpoint
            clear_url = self.config['server_url'].replace('/generate_accompaniment', '/clear_history')
            response = requests.post(clear_url, timeout=5)
            return response.status_code in [200, 503]  # 503 is OK (engine not loaded)
        except:
            return False
    
    def _export_combined_results(self):
        """Export combined results to CSV and summary statistics."""
        print("\n📁 Exporting combined results...")
        
        # Combine all individual results
        if self.all_results:
            combined_df = pd.concat(self.all_results, ignore_index=True)
            combined_csv = self.analysis_dir / "detailed_results_all_generation_lengths.csv"
            combined_df.to_csv(combined_csv, index=False)
            print(f"   Detailed results: {combined_csv}")
        
        # Export summary statistics
        if self.summary_stats:
            summary_df = pd.DataFrame(self.summary_stats)
            summary_csv = self.analysis_dir / "summary_statistics.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"   Summary statistics: {summary_csv}")
            
            # Also save as JSON for easy programmatic access
            summary_json = self.analysis_dir / "summary_statistics.json"
            summary_df.to_json(summary_json, indent=2, orient='records')
            print(f"   Summary JSON: {summary_json}")
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n📊 Generating visualizations...")
        
        if not self.summary_stats:
            print("   No data available for visualization")
            return
            
        summary_df = pd.DataFrame(self.summary_stats)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Primary relationship plot: Latency vs Generation Length
        self._plot_latency_vs_generation_length(summary_df)
        
        # 2. Variability analysis: Standard deviation trends
        self._plot_variability_analysis(summary_df)
        
        # 3. Distribution analysis: Histograms for each generation length
        self._plot_distribution_analysis()
        
        # 4. Performance scaling analysis
        self._plot_performance_scaling(summary_df)
        
        print(f"   Plots saved to: {self.plots_dir}")
    
    def _plot_latency_vs_generation_length(self, summary_df: pd.DataFrame):
        """Plot latency metrics vs generation length with error bars."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot main latency metrics
        metrics = [
            ('round_trip_time', 'Round Trip Time', 'o-'),
            ('server_processing_duration', 'Server Processing', 's--'),
            ('inference_duration', 'Inference Time', '^:'),
            ('total_network_latency', 'Network Latency', 'd-.'),
        ]
        
        for metric, label, style in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if mean_col in summary_df.columns and std_col in summary_df.columns:
                ax.errorbar(
                    summary_df['generation_length'],
                    summary_df[mean_col] * 1000,  # Convert to ms
                    yerr=summary_df[std_col] * 1000,
                    label=label,
                    marker=style[0] if len(style) > 0 else 'o',
                    linestyle=style[1:] if len(style) > 1 else '-',
                    capsize=5,
                    capthick=2,
                    linewidth=2,
                    markersize=8
                )
        
        ax.set_xlabel('Generation Length (Frames)', fontsize=14)
        ax.set_ylabel('Latency (milliseconds)', fontsize=14)
        ax.set_title('Server Latency vs Generation Length', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "latency_vs_generation_length.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_variability_analysis(self, summary_df: pd.DataFrame):
        """Plot standard deviation trends."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot standard deviations
        metrics = [
            ('round_trip_time', 'Round Trip Time'),
            ('server_processing_duration', 'Server Processing'),
            ('inference_duration', 'Inference Time'),
            ('total_network_latency', 'Network Latency'),
        ]
        
        for metric, label in metrics:
            std_col = f"{metric}_std"
            if std_col in summary_df.columns:
                ax.plot(
                    summary_df['generation_length'],
                    summary_df[std_col] * 1000,  # Convert to ms
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=f"{label} Std Dev"
                )
        
        ax.set_xlabel('Generation Length (Frames)', fontsize=14)
        ax.set_ylabel('Standard Deviation (milliseconds)', fontsize=14)
        ax.set_title('Latency Variability vs Generation Length', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "latency_variability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_analysis(self):
        """Plot histograms for each generation length."""
        if not self.all_results:
            return
            
        combined_df = pd.concat(self.all_results, ignore_index=True)
        generation_lengths = sorted(combined_df['generation_length'].unique())
        
        # Create subplot grid
        n_lengths = len(generation_lengths)
        cols = min(3, n_lengths)
        rows = (n_lengths + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, gen_length in enumerate(generation_lengths):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data = combined_df[combined_df['generation_length'] == gen_length]
            
            # Plot histogram of round trip times
            ax.hist(data['round_trip_time'] * 1000, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'Generation Length: {gen_length} frames', fontsize=12, fontweight='bold')
            ax.set_xlabel('Round Trip Time (ms)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_rtt = data['round_trip_time'].mean() * 1000
            std_rtt = data['round_trip_time'].std() * 1000
            ax.text(0.95, 0.95, f'μ = {mean_rtt:.1f}ms\nσ = {std_rtt:.1f}ms', 
                   transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_scaling(self, summary_df: pd.DataFrame):
        """Plot performance scaling metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Throughput estimation
        if 'round_trip_time_mean' in summary_df.columns:
            throughput = 1.0 / summary_df['round_trip_time_mean']  # requests per second
            ax1.plot(summary_df['generation_length'], throughput, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('Generation Length (Frames)', fontsize=12)
            ax1.set_ylabel('Estimated Throughput (req/sec)', fontsize=12)
            ax1.set_title('Throughput vs Generation Length', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Right plot: Notes generated vs latency efficiency  
        if 'num_generated_notes_mean' in summary_df.columns and 'inference_duration_mean' in summary_df.columns:
            notes_per_second = summary_df['num_generated_notes_mean'] / summary_df['inference_duration_mean']
            ax2.plot(summary_df['generation_length'], notes_per_second, 's-', linewidth=2, markersize=8, color='orange')
            ax2.set_xlabel('Generation Length (Frames)', fontsize=12)
            ax2.set_ylabel('Notes Generated per Second', fontsize=12)
            ax2.set_title('Generation Efficiency vs Generation Length', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_analysis_report(self):
        """Generate automated analysis report."""
        print("\n📝 Generating analysis report...")
        
        if not self.summary_stats:
            return
            
        summary_df = pd.DataFrame(self.summary_stats)
        
        report_path = self.analysis_dir / "benchmark_analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Generation Length Benchmark Analysis Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Generation Lengths Tested:** {list(summary_df['generation_length'])}\n")
            f.write(f"- **Total Requests:** {summary_df['num_requests'].sum()}\n")
            f.write(f"- **Successful Requests:** {summary_df['num_successful_requests'].sum()}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Find optimal generation length for latency
            min_latency_idx = summary_df['round_trip_time_mean'].idxmin()
            optimal_gen_length = summary_df.iloc[min_latency_idx]['generation_length']
            min_latency = summary_df.iloc[min_latency_idx]['round_trip_time_mean'] * 1000
            
            f.write(f"- **Optimal Generation Length for Latency:** {optimal_gen_length} frames ({min_latency:.1f}ms mean round trip)\n")
            
            # Latency range
            min_rtt = summary_df['round_trip_time_mean'].min() * 1000
            max_rtt = summary_df['round_trip_time_mean'].max() * 1000
            f.write(f"- **Latency Range:** {min_rtt:.1f}ms - {max_rtt:.1f}ms\n")
            
            # Variability analysis
            min_var_idx = summary_df['round_trip_time_std'].idxmin()
            most_consistent = summary_df.iloc[min_var_idx]['generation_length']
            min_std = summary_df.iloc[min_var_idx]['round_trip_time_std'] * 1000
            f.write(f"- **Most Consistent Performance:** {most_consistent} frames ({min_std:.1f}ms std dev)\n\n")
            
            # Performance table
            f.write("## Performance Table\n\n")
            f.write("| Gen Length | Mean RTT (ms) | Std RTT (ms) | Mean Inference (ms) | Notes Generated |\n")
            f.write("|------------|---------------|--------------|---------------------|------------------|\n")
            
            for _, row in summary_df.iterrows():
                gen_len = int(row['generation_length'])
                mean_rtt = row['round_trip_time_mean'] * 1000
                std_rtt = row['round_trip_time_std'] * 1000
                mean_inf = row['inference_duration_mean'] * 1000
                notes = row.get('num_generated_notes_mean', 0)
                
                f.write(f"| {gen_len} | {mean_rtt:.1f} | {std_rtt:.1f} | {mean_inf:.1f} | {notes:.1f} |\n")
            
            f.write(f"\n## Data Files\n\n")
            f.write(f"- **Detailed Results:** `detailed_results_all_generation_lengths.csv`\n")
            f.write(f"- **Summary Statistics:** `summary_statistics.csv`\n")
            f.write(f"- **Raw Data:** `raw_data/` directory\n")
            f.write(f"- **Visualizations:** `plots/` directory\n")
        
        print(f"   Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced benchmark for analyzing generation length effects on StreamMUSE latency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --generation_lengths 5,10,15,20,25,30 --output_dir results/gen_length_study
  %(prog)s --generation_lengths 10,20,30 --requests_per_length 50 --generate_plots
        """
    )
    
    # Core parameters
    parser.add_argument("--server_url", type=str, 
                       default="http://localhost:8000/generate_accompaniment",
                       help="StreamMUSE server URL")
    parser.add_argument("--server_port", type=int, default=8000,
                       help="Server port (for restart instructions)")
    parser.add_argument("--generation_lengths", type=str, 
                       default="5,10,15,20,25,30",
                       help="Comma-separated generation lengths to test")
    parser.add_argument("--requests_per_length", type=int, default=50,
                       help="Number of requests per generation length")
    parser.add_argument("--output_dir", type=str, default="results/generation_length_analysis",
                       help="Output directory for results")
    
    # Control options
    parser.add_argument("--generate_plots", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--generate_report", action="store_true", 
                       help="Generate analysis report")
    parser.add_argument("--timeout_seconds", type=int, default=300,
                       help="Timeout for each benchmark run")
    parser.add_argument("--auto_server_restart", action="store_true",
                       help="Attempt automatic server restart (experimental)")
    
    args = parser.parse_args()
    
    # Parse generation lengths
    try:
        generation_lengths = [int(x.strip()) for x in args.generation_lengths.split(',')]
    except ValueError:
        print("Error: Invalid generation_lengths format. Use comma-separated integers.")
        return 1
    
    # Validate parameters
    if len(generation_lengths) == 0:
        print("Error: No generation lengths specified")
        return 1
        
    if args.requests_per_length < 1:
        print("Error: requests_per_length must be at least 1")
        return 1
    
    # Build configuration
    config = {
        'server_url': args.server_url,
        'server_port': args.server_port,
        'generation_lengths': generation_lengths,
        'requests_per_length': args.requests_per_length,
        'output_dir': args.output_dir,
        'generate_plots': args.generate_plots,
        'generate_report': args.generate_report,
        'timeout_seconds': args.timeout_seconds,
        'auto_server_restart': args.auto_server_restart,
    }
    
    print("🎵 StreamMUSE Generation Length Benchmark")
    print("=" * 50)
    
    # Run the benchmark
    benchmark = GenerationLengthBenchmark(config)
    success = benchmark.run_parameter_sweep()
    
    if success:
        print(f"\n🎉 Benchmark completed successfully!")
        print(f"📁 Results saved to: {Path(args.output_dir).absolute()}")
        return 0
    else:
        print(f"\n❌ Benchmark failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())