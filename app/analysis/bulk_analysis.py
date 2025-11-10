#!/usr/bin/env python3
"""
Bulk Analysis Script for StreamMUSE Generation Length Studies

This script can analyze results from multiple experiment directories and combine them
into unified analysis. It handles both:
1. Old manually-run benchmark data (individual CSV files)
2. New bulk benchmark data (combined_results.csv from bulk_benchmark.py)

The script focuses on generation length (GL) analysis across different experimental conditions.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
import re
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import statistics
from datetime import datetime
from tqdm import tqdm

# Import the existing analyzer
import sys
sys.path.append(str(Path(__file__).parent))
from analyze_generation_length_results import GenerationLengthAnalyzer

class BulkGenerationLengthAnalyzer:
    """
    Analyzes generation length effects across multiple experiment directories.
    Combines data from different experimental conditions and creates unified visualizations.
    """
    
    def __init__(self, data_sources: List[str], output_dir: str = "bulk_analysis_results", anomaly_filter_pct: float = 0.0):
        self.data_sources = [Path(source) for source in data_sources]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.data_dir = self.output_dir / "processed_data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.experiments = {}  # experiment_name -> data
        self.combined_data = None
        self.summary_data = None
        self.anomaly_filter_pct = anomaly_filter_pct
        
    def load_all_experiments(self) -> bool:
        """Load data from all provided experiment directories."""
        print(f"🔍 Loading data from {len(self.data_sources)} experiment directories...")
        
        success_count = 0
        
        for i, data_source in enumerate(tqdm(self.data_sources, desc="Loading experiments")):
            if not data_source.exists():
                print(f"❌ Directory not found: {data_source}")
                continue
                
            experiment_name = self._generate_experiment_name(data_source, i)
            print(f"\\n📂 Loading experiment: {experiment_name}")
            print(f"   Source: {data_source}")
            
            experiment_data = self._load_single_experiment(data_source, experiment_name)
            if experiment_data is not None:
                self.experiments[experiment_name] = experiment_data
                success_count += 1
                print(f"✅ Successfully loaded {experiment_name}")
            else:
                print(f"❌ Failed to load {experiment_name}")
        
        if success_count == 0:
            print("❌ No experiments could be loaded")
            return False
            
        print(f"\\n✅ Successfully loaded {success_count} out of {len(self.data_sources)} experiments")
        return True
    
    def _generate_experiment_name(self, data_source: Path, index: int) -> str:
        """Generate a meaningful name for an experiment based on its source."""
        
        # Try to extract name from directory structure
        if data_source.name and data_source.name != '.':
            base_name = data_source.name
        else:
            base_name = f"experiment_{index + 1}"
        
        # Check if there's a config.yaml that might have more info
        config_file = data_source / "config.yaml"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    if 'experiment' in config and 'name' in config['experiment']:
                        base_name = config['experiment']['name']
            except Exception:
                pass  # Use directory name as fallback
        
        return base_name
    
    def _apply_anomaly_filter(self, df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
        """
        Filter out anomalies based on round trip time percentiles per generation length.
        
        Args:
            df: DataFrame with round_trip_time and generation_length columns
            experiment_name: Name of experiment for logging
            
        Returns:
            Filtered DataFrame with anomalies removed per generation length
        """
        if self.anomaly_filter_pct <= 0 or 'round_trip_time' not in df.columns:
            return df
        
        if 'generation_length' not in df.columns:
            print(f"   ⚠️ {experiment_name}: No generation_length column found, skipping anomaly filtering")
            return df
        
        original_count = len(df)
        filtered_data = []
        
        print(f"   🔍 Anomaly filtering for {experiment_name} ({self.anomaly_filter_pct}% each tail per generation length):")
        
        for gen_length in sorted(df['generation_length'].unique()):
            gl_data = df[df['generation_length'] == gen_length]
            gl_original_count = len(gl_data)
            
            if gl_original_count < 10:  # Skip filtering if too few samples
                print(f"      GL {gen_length}: {gl_original_count} samples (too few, no filtering)")
                filtered_data.append(gl_data)
                continue
            
            # Calculate percentile thresholds for this generation length
            lower_threshold = gl_data['round_trip_time'].quantile(self.anomaly_filter_pct / 100)
            upper_threshold = gl_data['round_trip_time'].quantile(1 - self.anomaly_filter_pct / 100)
            
            # Filter data for this generation length
            gl_filtered = gl_data[
                (gl_data['round_trip_time'] >= lower_threshold) &
                (gl_data['round_trip_time'] <= upper_threshold)
            ].copy()
            
            gl_filtered_count = len(gl_filtered)
            gl_removed_count = gl_original_count - gl_filtered_count
            
            print(f"      GL {gen_length}: {gl_original_count} → {gl_filtered_count} samples "
                  f"(removed {gl_removed_count}, {gl_removed_count/gl_original_count*100:.1f}%)")
            
            filtered_data.append(gl_filtered)
        
        if not filtered_data:
            return df
        
        filtered_df = pd.concat(filtered_data, ignore_index=True)
        filtered_count = len(filtered_df)
        removed_count = original_count - filtered_count
        
        print(f"      Total for {experiment_name}: {original_count} → {filtered_count} samples "
              f"(removed {removed_count}, {removed_count/original_count*100:.1f}%)")
        
        return filtered_df
    
    def _load_single_experiment(self, data_source: Path, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load data from a single experiment directory."""
        
        # Try to detect the data format
        if self._is_bulk_benchmark_output(data_source):
            return self._load_bulk_benchmark_data(data_source, experiment_name)
        else:
            return self._load_manual_benchmark_data(data_source, experiment_name)
    
    def _is_bulk_benchmark_output(self, data_source: Path) -> bool:
        """Detect if this is output from bulk_benchmark.py."""
        
        # Check for bulk benchmark indicators
        indicators = [
            data_source / "combined_results.csv",
            data_source / "config.yaml",
            data_source / "experiment_summary.json"
        ]
        
        return any(indicator.exists() for indicator in indicators)
    
    def _load_bulk_benchmark_data(self, data_source: Path, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load data from bulk benchmark output directory."""
        
        combined_file = data_source / "combined_results.csv"
        if not combined_file.exists():
            print(f"   ❌ No combined_results.csv found in {data_source}")
            return None
        
        try:
            df = pd.read_csv(combined_file)
            
            # Ensure we have generation length data
            gen_length_cols = ['generation_length_frames', 'generation_length']
            gen_length_col = None
            for col in gen_length_cols:
                if col in df.columns:
                    gen_length_col = col
                    break
            
            if gen_length_col is None:
                print(f"   ❌ No generation length column found in {combined_file}")
                return None
            
            # Standardize column name
            if gen_length_col != 'generation_length':
                df['generation_length'] = df[gen_length_col]
            
            # Apply anomaly filtering
            df = self._apply_anomaly_filter(df, experiment_name)
            
            # Load metadata if available
            metadata = {}
            config_file = data_source / "config.yaml"
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        metadata['config'] = yaml.safe_load(f)
                except Exception:
                    pass
            
            summary_file = data_source / "experiment_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        metadata['summary'] = json.load(f)
                except Exception:
                    pass
            
            return {
                'type': 'bulk_benchmark',
                'name': experiment_name,
                'data': df,
                'source_path': data_source,
                'metadata': metadata,
                'generation_lengths': sorted(df['generation_length'].unique()),
                'total_requests': len(df)
            }
            
        except Exception as e:
            print(f"   ❌ Error loading bulk benchmark data: {e}")
            return None
    
    def _load_manual_benchmark_data(self, data_source: Path, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load data from manual benchmark CSV files."""
        
        # Find CSV files
        csv_files = list(data_source.glob("*.csv"))
        csv_files.extend(list(data_source.glob("**/*.csv")))
        
        if not csv_files:
            print(f"   ❌ No CSV files found in {data_source}")
            return None
        
        all_data = []
        generation_lengths = []
        
        for csv_file in csv_files:
            # Extract generation length from filename
            gen_length = self._extract_generation_length_from_filename(csv_file.name)
            
            if gen_length is None:
                print(f"   ⚠️  Skipping {csv_file.name} - cannot extract generation length")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                df['generation_length'] = gen_length
                df['source_file'] = csv_file.name
                all_data.append(df)
                generation_lengths.append(gen_length)
                
            except Exception as e:
                print(f"   ⚠️  Error loading {csv_file.name}: {e}")
                continue
        
        if not all_data:
            print(f"   ❌ No valid data files found in {data_source}")
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Apply anomaly filtering
        filtered_df = self._apply_anomaly_filter(combined_df, experiment_name)
        
        return {
            'type': 'manual_benchmark',
            'name': experiment_name,
            'data': filtered_df,
            'source_path': data_source,
            'metadata': {},
            'generation_lengths': sorted(set(generation_lengths)),
            'total_requests': len(filtered_df)
        }
    
    def _extract_generation_length_from_filename(self, filename: str) -> Optional[int]:
        """Extract generation length from various filename patterns."""
        
        patterns = [
            r'gen_length_(\d+)',
            r'gen_(\d+)', 
            r'generation_(\d+)',
            r'gl_(\d+)',
            r'GL(\d+)',
            r'(\d+)_frames',
            r'frames_(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def combine_all_data(self):
        """Combine data from all experiments into unified datasets."""
        print("\\n🔗 Combining data from all experiments...")
        
        if not self.experiments:
            print("❌ No experiments loaded")
            return
        
        all_detailed_data = []
        summary_list = []
        
        for exp_name, exp_data in self.experiments.items():
            df = exp_data['data'].copy()
            df['experiment_source'] = exp_name
            df['experiment_type'] = exp_data['type']
            all_detailed_data.append(df)
            
            # Generate summary statistics for this experiment
            exp_summary = self._calculate_experiment_summary(exp_data)
            summary_list.extend(exp_summary)
        
        self.combined_data = pd.concat(all_detailed_data, ignore_index=True)
        self.summary_data = pd.DataFrame(summary_list)
        
        # Save combined datasets
        combined_file = self.data_dir / "all_experiments_combined.csv"
        self.combined_data.to_csv(combined_file, index=False)
        
        summary_file = self.data_dir / "all_experiments_summary.csv"
        self.summary_data.to_csv(summary_file, index=False)
        
        # Generate and save constraint analysis data
        constraint_data = self._calculate_constraint_satisfaction_rates()
        if constraint_data is not None:
            constraint_file = self.data_dir / "constraint_analysis.csv"
            constraint_data.to_csv(constraint_file, index=False)
            print(f"   Constraint analysis saved to: {constraint_file}")
        
        # Generate and save parameter constraint analysis data
        self._export_parameter_constraint_data()
        
        print(f"✅ Combined {len(self.experiments)} experiments")
        print(f"   Total requests: {len(self.combined_data)}")
        print(f"   Generation lengths: {sorted(self.combined_data['generation_length'].unique())}")
        print(f"   Combined data saved to: {combined_file}")
        print(f"   Summary data saved to: {summary_file}")
    
    def _calculate_experiment_summary(self, exp_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate summary statistics for a single experiment."""
        
        df = exp_data['data']
        exp_name = exp_data['name']
        
        summary_list = []
        
        for gen_length in sorted(df['generation_length'].unique()):
            df_subset = df[df['generation_length'] == gen_length]
            
            summary = {
                'experiment_name': exp_name,
                'experiment_type': exp_data['type'],
                'generation_length': gen_length,
                'num_requests': len(df_subset),
                'source_path': str(exp_data['source_path'])
            }
            
            # Calculate statistics for key metrics
            metrics = [
                'round_trip_time', 'server_processing_duration', 'inference_duration',
                'preprocess_duration', 'postprocess_duration', 'total_network_latency'
            ]
            
            for metric in metrics:
                if metric in df_subset.columns:
                    series = df_subset[metric]
                    summary.update({
                        f"{metric}_mean": series.mean(),
                        f"{metric}_std": series.std(),
                        f"{metric}_min": series.min(),
                        f"{metric}_max": series.max(),
                        f"{metric}_median": series.median(),
                        f"{metric}_p95": series.quantile(0.95),
                        f"{metric}_p99": series.quantile(0.99)
                    })
            
            # Add notes count if available
            if 'num_generated_notes' in df_subset.columns:
                notes = df_subset['num_generated_notes']
                summary.update({
                    'num_generated_notes_mean': notes.mean(),
                    'num_generated_notes_std': notes.std()
                })
            
            summary_list.append(summary)
        
        return summary_list
    
    def generate_comparative_analysis(self):
        """Generate comparative visualizations across all experiments."""
        print("\\n📊 Generating comparative analysis...")
        
        if self.summary_data is None:
            print("❌ No summary data available")
            return
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        self._plot_experiment_comparison()
        self._plot_generation_length_trends()
        self._plot_performance_comparison()
        self._plot_variability_comparison()
        self._plot_constraint_analysis()
        self._plot_detailed_constraint_analysis()
        self._plot_constraint_heatmaps()
        self._plot_parameter_constraint_analysis()
        
        print(f"✅ Comparative analysis plots saved to {self.plots_dir}")
    
    def _plot_experiment_comparison(self):
        """Compare latency across experiments for each generation length."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        experiments = self.summary_data['experiment_name'].unique()
        gen_lengths = sorted(self.summary_data['generation_length'].unique())
        
        # Left plot: Mean round trip time
        for exp in experiments:
            exp_data = self.summary_data[self.summary_data['experiment_name'] == exp]
            if 'round_trip_time_mean' in exp_data.columns:
                ax1.plot(exp_data['generation_length'], 
                        exp_data['round_trip_time_mean'] * 1000,
                        marker='o', linewidth=2.5, markersize=8, label=exp)
        
        ax1.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Round Trip Time (ms)', fontsize=14, fontweight='bold')
        ax1.set_title('Round Trip Time Comparison Across Experiments', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: 95th percentile round trip time
        for exp in experiments:
            exp_data = self.summary_data[self.summary_data['experiment_name'] == exp]
            if 'round_trip_time_p95' in exp_data.columns:
                ax2.plot(exp_data['generation_length'],
                        exp_data['round_trip_time_p95'] * 1000,
                        marker='s', linewidth=2.5, markersize=8, label=exp, linestyle='--')
        
        ax2.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('95th Percentile RTT (ms)', fontsize=14, fontweight='bold') 
        ax2.set_title('95th Percentile Performance Comparison', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "experiment_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_generation_length_trends(self):
        """Show how different experiments scale with generation length."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        metrics = [
            ('round_trip_time_mean', 'Mean Round Trip Time (ms)', 1000),
            ('inference_duration_mean', 'Mean Inference Duration (ms)', 1000),
            ('server_processing_duration_mean', 'Mean Server Processing (ms)', 1000),
            ('total_network_latency_mean', 'Mean Network Latency (ms)', 1000)
        ]
        
        experiments = self.summary_data['experiment_name'].unique()
        colors = sns.color_palette("husl", len(experiments))
        
        for i, (metric, title, scale) in enumerate(metrics):
            if metric not in self.summary_data.columns:
                axes[i].text(0.5, 0.5, f'No {metric} data available',
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(title, fontsize=14, fontweight='bold')
                continue
            
            for j, exp in enumerate(experiments):
                exp_data = self.summary_data[self.summary_data['experiment_name'] == exp]
                if not exp_data.empty:
                    axes[i].plot(exp_data['generation_length'],
                               exp_data[metric] * scale,
                               marker='o', linewidth=2.5, markersize=8,
                               label=exp, color=colors[j])
                    
                    # Add trend line
                    if len(exp_data) > 1:
                        x = exp_data['generation_length']
                        y = exp_data[metric] * scale
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        axes[i].plot(x, p(x), '--', alpha=0.7, color=colors[j], linewidth=1.5)
            
            axes[i].set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(title.split('(')[0].strip(), fontsize=12, fontweight='bold')
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "generation_length_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self):
        """Bar chart comparison of key metrics across experiments."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        # Focus on a few key generation lengths for comparison
        key_gen_lengths = [1, 3, 5, 7, 9]
        available_gen_lengths = sorted(self.summary_data['generation_length'].unique())
        comparison_lengths = [gl for gl in key_gen_lengths if gl in available_gen_lengths]
        
        if not comparison_lengths:
            # Fallback to available lengths
            comparison_lengths = available_gen_lengths[:5]
        
        metrics = [
            ('round_trip_time_mean', 'Mean Round Trip Time (ms)', 1000),
            ('inference_duration_mean', 'Mean Inference Duration (ms)', 1000),
            ('round_trip_time_std', 'Round Trip Time Std Dev (ms)', 1000),
            ('round_trip_time_p95', '95th Percentile RTT (ms)', 1000)
        ]
        
        experiments = self.summary_data['experiment_name'].unique()
        x_pos = np.arange(len(comparison_lengths))
        width = 0.8 / len(experiments)
        
        for i, (metric, title, scale) in enumerate(metrics):
            if metric not in self.summary_data.columns:
                axes[i].text(0.5, 0.5, f'No {metric} data available',
                           transform=axes[i].transAxes, ha='center', va='center')
                axes[i].set_title(title, fontsize=14, fontweight='bold')
                continue
            
            for j, exp in enumerate(experiments):
                values = []
                for gl in comparison_lengths:
                    exp_gl_data = self.summary_data[
                        (self.summary_data['experiment_name'] == exp) &
                        (self.summary_data['generation_length'] == gl)
                    ]
                    if not exp_gl_data.empty:
                        values.append(exp_gl_data[metric].iloc[0] * scale)
                    else:
                        values.append(0)
                
                axes[i].bar(x_pos + j * width, values, width, 
                          label=exp, alpha=0.8)
            
            axes[i].set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(title.split('(')[0].strip(), fontsize=12, fontweight='bold')
            axes[i].set_title(title, fontsize=14, fontweight='bold')
            axes[i].set_xticks(x_pos + width * (len(experiments) - 1) / 2)
            axes[i].set_xticklabels(comparison_lengths)
            axes[i].legend(fontsize=10)
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_variability_comparison(self):
        """Compare variability (coefficient of variation) across experiments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        experiments = self.summary_data['experiment_name'].unique()
        
        # Left: Standard deviation
        for exp in experiments:
            exp_data = self.summary_data[self.summary_data['experiment_name'] == exp]
            if 'round_trip_time_std' in exp_data.columns:
                ax1.plot(exp_data['generation_length'],
                        exp_data['round_trip_time_std'] * 1000,
                        marker='o', linewidth=2.5, markersize=8, label=exp)
        
        ax1.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Round Trip Time Std Dev (ms)', fontsize=14, fontweight='bold')
        ax1.set_title('Latency Variability Comparison', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right: Coefficient of variation
        for exp in experiments:
            exp_data = self.summary_data[self.summary_data['experiment_name'] == exp]
            if ('round_trip_time_mean' in exp_data.columns and 
                'round_trip_time_std' in exp_data.columns):
                cv = (exp_data['round_trip_time_std'] / exp_data['round_trip_time_mean']) * 100
                ax2.plot(exp_data['generation_length'], cv,
                        marker='s', linewidth=2.5, markersize=8, label=exp)
        
        ax2.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Coefficient of Variation (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Relative Variability Comparison', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "variability_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_constraint_analysis(self):
        """Analyze which experiments meet real-time constraints."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Convert to tick-based analysis (only odd generation lengths for accompaniment)
        odd_summary = self.summary_data[self.summary_data['generation_length'] % 2 == 1].copy()
        
        if odd_summary.empty:
            print("⚠️ No odd generation lengths found for constraint analysis")
            return
        
        odd_summary['generation_ticks'] = (odd_summary['generation_length'] + 1) / 2
        experiments = odd_summary['experiment_name'].unique()
        
        # Left: Musical time constraint analysis
        for exp in experiments:
            exp_data = odd_summary[odd_summary['experiment_name'] == exp]
            if 'round_trip_time_mean' in exp_data.columns:
                x_ticks = exp_data['generation_ticks']
                y_mean = exp_data['round_trip_time_mean'] * 1000
                
                ax1.plot(x_ticks, y_mean, 'o-', linewidth=2.5, markersize=8, label=exp)
        
        # Add musical time constraints
        tick_values = sorted(odd_summary['generation_ticks'].unique())
        if tick_values:
            musical_time_y = [tick * 125 for tick in tick_values]
            buffer_minus1_y = [max(0, tick * 125 - 125) for tick in tick_values]
            
            ax1.plot(tick_values, musical_time_y, 'r-', linewidth=3, alpha=0.8, 
                    label='Musical Time Deadline', zorder=5)
            ax1.plot(tick_values, buffer_minus1_y, 'purple', linewidth=3, alpha=0.8,
                    label='1 Useable Tick', zorder=5)
        
        ax1.set_xlabel('Generation Length (Ticks)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Round Trip Time (ms)', fontsize=14, fontweight='bold')
        ax1.set_title('Real-Time Constraint Analysis', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right: 95th percentile constraint analysis
        for exp in experiments:
            exp_data = odd_summary[odd_summary['experiment_name'] == exp]
            if 'round_trip_time_p95' in exp_data.columns:
                x_ticks = exp_data['generation_ticks']
                y_p95 = exp_data['round_trip_time_p95'] * 1000
                
                ax2.plot(x_ticks, y_p95, 's--', linewidth=2.5, markersize=8, label=exp)
        
        # Add musical time constraints for p95
        if tick_values:
            ax2.plot(tick_values, musical_time_y, 'r-', linewidth=3, alpha=0.8,
                    label='Musical Time Deadline', zorder=5)
            ax2.plot(tick_values, buffer_minus1_y, 'purple', linewidth=3, alpha=0.8,
                    label='1 Useable Tick', zorder=5)
        
        ax2.set_xlabel('Generation Length (Ticks)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('95th Percentile RTT (ms)', fontsize=14, fontweight='bold')
        ax2.set_title('95th Percentile Constraint Analysis', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "constraint_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _evaluate_constraints_detailed(self, inference_interval, generation_length, round_trip_time_ms):
        """
        Evaluate parameter constraints with detailed breakdown.
        
        Args:
            inference_interval: Inference interval in ticks
            generation_length: Generation length in ticks  
            round_trip_time_ms: Round trip time in milliseconds
        
        Returns:
            tuple: (constraint1_satisfied, constraint2_satisfied, constraint_status)
        """
        TICK_DURATION_MS = 125  # 125ms per tick at 120 BPM
        
        # Constraint 1: Inference interval must be long enough to avoid overlap
        constraint1_satisfied = inference_interval * TICK_DURATION_MS >= round_trip_time_ms
        
        # Constraint 2: Generated music must arrive before it's needed  
        musical_buffer_ms = (generation_length - inference_interval) * TICK_DURATION_MS
        constraint2_satisfied = round_trip_time_ms < musical_buffer_ms
        
        # Determine constraint status for coloring
        if constraint1_satisfied and constraint2_satisfied:
            constraint_status = 0  # Both satisfied (valid)
        elif not constraint1_satisfied and constraint2_satisfied:
            constraint_status = 1  # Only constraint 1 violated
        elif constraint1_satisfied and not constraint2_satisfied:
            constraint_status = 2  # Only constraint 2 violated  
        else:
            constraint_status = 3  # Both violated (invalid)
        
        return constraint1_satisfied, constraint2_satisfied, constraint_status
    
    def _calculate_constraint_satisfaction_rates(self):
        """Calculate constraint satisfaction rates for each experiment."""
        if self.combined_data is None:
            return None
        
        # Convert frames to ticks for odd generation lengths only (accompaniment focus)
        odd_data = self.combined_data[self.combined_data['generation_length'] % 2 == 1].copy()
        if odd_data.empty:
            print("⚠️ No odd generation lengths found for constraint analysis")
            return None
        
        odd_data['generation_ticks'] = (odd_data['generation_length'] + 1) / 2
        
        # Define parameter ranges to test
        inference_intervals = range(1, 8)  # 1-7 ticks
        
        constraint_results = []
        
        for exp_name in odd_data['experiment_source'].unique():
            exp_data = odd_data[odd_data['experiment_source'] == exp_name]
            
            for gen_ticks in sorted(exp_data['generation_ticks'].unique()):
                gen_rtt_data = exp_data[exp_data['generation_ticks'] == gen_ticks]
                rtt_ms = gen_rtt_data['round_trip_time'] * 1000  # Convert to ms
                
                # Calculate percentiles
                percentiles = [50, 70, 80, 90, 95, 99]
                percentile_values = {}
                for p in percentiles:
                    if len(rtt_ms) > 0:
                        percentile_values[p] = np.percentile(rtt_ms, p)
                    else:
                        percentile_values[p] = np.inf
                
                # Test each inference interval
                for inference_interval in inference_intervals:
                    if inference_interval < gen_ticks:  # Only test valid combinations
                        for p in percentiles:
                            rtt_p = percentile_values[p]
                            c1, c2, status = self._evaluate_constraints_detailed(
                                inference_interval, gen_ticks, rtt_p
                            )
                            
                            constraint_results.append({
                                'experiment_name': exp_name,
                                'generation_ticks': gen_ticks,
                                'inference_interval': inference_interval,
                                'percentile': p,
                                'rtt_percentile_ms': rtt_p,
                                'constraint1_satisfied': c1,
                                'constraint2_satisfied': c2,
                                'both_satisfied': c1 and c2,
                                'constraint_status': status
                            })
        
        return pd.DataFrame(constraint_results)
    
    def _plot_detailed_constraint_analysis(self):
        """Generate detailed constraint satisfaction analysis across experiments."""
        constraint_data = self._calculate_constraint_satisfaction_rates()
        if constraint_data is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        experiments = constraint_data['experiment_name'].unique()
        colors = sns.color_palette("husl", len(experiments))
        
        # Plot 1: Constraint satisfaction rates by generation length
        for i, exp in enumerate(experiments):
            exp_data = constraint_data[constraint_data['experiment_name'] == exp]
            # Calculate satisfaction rate for each generation length
            satisfaction_rates = []
            gen_ticks = []
            
            for gt in sorted(exp_data['generation_ticks'].unique()):
                gt_data = exp_data[exp_data['generation_ticks'] == gt]
                satisfaction_rate = gt_data['both_satisfied'].mean() * 100
                satisfaction_rates.append(satisfaction_rate)
                gen_ticks.append(gt)
            
            axes[0].plot(gen_ticks, satisfaction_rates, 'o-', linewidth=2.5, markersize=8,
                        color=colors[i], label=exp)
        
        axes[0].set_xlabel('Generation Length (Ticks)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Constraint Satisfaction Rate (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Overall Constraint Satisfaction by Generation Length', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 105)
        
        # Plot 2: 95th percentile constraint satisfaction
        for i, exp in enumerate(experiments):
            exp_data = constraint_data[
                (constraint_data['experiment_name'] == exp) &
                (constraint_data['percentile'] == 95)
            ]
            satisfaction_rates = []
            gen_ticks = []
            
            for gt in sorted(exp_data['generation_ticks'].unique()):
                gt_data = exp_data[exp_data['generation_ticks'] == gt]
                satisfaction_rate = gt_data['both_satisfied'].mean() * 100
                satisfaction_rates.append(satisfaction_rate)
                gen_ticks.append(gt)
            
            axes[1].plot(gen_ticks, satisfaction_rates, 's-', linewidth=2.5, markersize=8,
                        color=colors[i], label=exp)
        
        axes[1].set_xlabel('Generation Length (Ticks)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Constraint Satisfaction Rate (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('95th Percentile Constraint Satisfaction', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 105)
        
        # Plot 3: Constraint breakdown by type
        constraint_types = ['constraint1_satisfied', 'constraint2_satisfied', 'both_satisfied']
        constraint_labels = ['Constraint 1 (Overlap)', 'Constraint 2 (Buffer)', 'Both Satisfied']
        
        x_pos = np.arange(len(experiments))
        width = 0.25
        
        for i, (constraint_type, label) in enumerate(zip(constraint_types, constraint_labels)):
            rates = []
            for exp in experiments:
                exp_data = constraint_data[constraint_data['experiment_name'] == exp]
                rate = exp_data[constraint_type].mean() * 100
                rates.append(rate)
            
            axes[2].bar(x_pos + i * width, rates, width, label=label, alpha=0.8)
        
        axes[2].set_xlabel('Experiment', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Satisfaction Rate (%)', fontsize=12, fontweight='bold')
        axes[2].set_title('Constraint Satisfaction by Type', fontsize=14, fontweight='bold')
        axes[2].set_xticks(x_pos + width)
        axes[2].set_xticklabels(experiments, rotation=45, ha='right')
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_ylim(0, 105)
        
        # Plot 4: Percentile comparison
        percentiles_to_show = [50, 80, 95, 99]
        for i, exp in enumerate(experiments):
            satisfaction_by_percentile = []
            for p in percentiles_to_show:
                exp_p_data = constraint_data[
                    (constraint_data['experiment_name'] == exp) &
                    (constraint_data['percentile'] == p)
                ]
                rate = exp_p_data['both_satisfied'].mean() * 100
                satisfaction_by_percentile.append(rate)
            
            axes[3].plot(percentiles_to_show, satisfaction_by_percentile, 'o-',
                        linewidth=2.5, markersize=8, color=colors[i], label=exp)
        
        axes[3].set_xlabel('RTT Percentile', fontsize=12, fontweight='bold')
        axes[3].set_ylabel('Constraint Satisfaction Rate (%)', fontsize=12, fontweight='bold')
        axes[3].set_title('Constraint Satisfaction vs RTT Percentile', fontsize=14, fontweight='bold')
        axes[3].legend(fontsize=10)
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "detailed_constraint_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_constraint_heatmaps(self):
        """Generate constraint satisfaction heatmaps for each experiment."""
        constraint_data = self._calculate_constraint_satisfaction_rates()
        if constraint_data is None:
            return
        
        experiments = constraint_data['experiment_name'].unique()
        n_experiments = len(experiments)
        
        # Create subplots grid
        cols = min(2, n_experiments)
        rows = (n_experiments + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(14 * cols, 8 * rows))
        
        # Ensure axes is always a list for consistent indexing
        if n_experiments == 1:
            axes = [axes]  # Single subplot case
        elif rows == 1 and cols > 1:
            axes = list(axes)  # Single row, multiple columns
        elif rows > 1 and cols == 1:
            axes = list(axes)  # Multiple rows, single column
        else:
            axes = axes.flatten()  # Multiple rows and columns
        
        # Define parameter ranges
        inference_intervals = sorted(constraint_data['inference_interval'].unique())
        generation_ticks = sorted(constraint_data['generation_ticks'].unique())
        
        for i, exp in enumerate(experiments):
            if i >= len(axes):
                break
            
            ax = axes[i]
            exp_data = constraint_data[
                (constraint_data['experiment_name'] == exp) &
                (constraint_data['percentile'] == 95)  # Use 95th percentile
            ]
            
            # Create satisfaction rate matrix
            matrix = np.full((len(inference_intervals), len(generation_ticks)), np.nan)
            
            for ii_idx, ii in enumerate(inference_intervals):
                for gt_idx, gt in enumerate(generation_ticks):
                    subset = exp_data[
                        (exp_data['inference_interval'] == ii) &
                        (exp_data['generation_ticks'] == gt)
                    ]
                    if not subset.empty:
                        # Use constraint satisfaction (0 = invalid, 1 = valid)
                        matrix[ii_idx, gt_idx] = subset['both_satisfied'].iloc[0]
            
            # Create heatmap
            im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                          interpolation='nearest')
            
            # Set labels and ticks
            ax.set_xticks(range(len(generation_ticks)))
            ax.set_xticklabels([f'{int(gt)}' for gt in generation_ticks])
            ax.set_yticks(range(len(inference_intervals)))
            ax.set_yticklabels([f'{ii}' for ii in inference_intervals])
            
            ax.set_xlabel('Generation Length (Ticks)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Inference Interval (Ticks)', fontsize=12, fontweight='bold')
            ax.set_title(f'{exp}\\n95th Percentile Constraint Satisfaction', 
                        fontsize=14, fontweight='bold')
            
            # Add text annotations
            for ii_idx in range(len(inference_intervals)):
                for gt_idx in range(len(generation_ticks)):
                    if not np.isnan(matrix[ii_idx, gt_idx]):
                        status = 'Valid' if matrix[ii_idx, gt_idx] == 1 else 'Invalid'
                        color = 'white' if matrix[ii_idx, gt_idx] < 0.5 else 'black'
                        ax.text(gt_idx, ii_idx, status, ha='center', va='center',
                               color=color, fontsize=9, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Constraint Satisfaction', fontsize=11, fontweight='bold')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Invalid', 'Valid'])
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "constraint_heatmaps.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_percentiles_by_generation_length(self, experiment_name: str, percentiles=[50, 60, 70, 80, 90, 99.5]):
        """Calculate round trip time percentiles for each generation length separately for one experiment."""
        if self.combined_data is None:
            return {}
        
        exp_data = self.combined_data[self.combined_data['experiment_source'] == experiment_name]
        percentile_data = {}
        
        for gen_length in sorted(exp_data['generation_length'].unique()):
            gen_data = exp_data[exp_data['generation_length'] == gen_length]
            rtt_ms = gen_data['round_trip_time'] * 1000  # Convert to milliseconds
            
            for p in percentiles:
                if len(rtt_ms) > 0:
                    percentile_value = np.percentile(rtt_ms, p)
                    percentile_data[(gen_length, p)] = percentile_value
                else:
                    percentile_data[(gen_length, p)] = np.inf
        
        return percentile_data
    
    def _create_constraint_matrix_detailed(self, percentile_data, percentile, inference_range, generation_range):
        """
        Create a matrix showing detailed constraint status for parameter combinations.
        
        Args:
            percentile_data: Dictionary keyed by (generation_length_frames, percentile)
            percentile: Which percentile to use
            inference_range: Range of inference intervals (in ticks)
            generation_range: Range of generation lengths (in ticks)
        
        Returns:
            2D numpy array with constraint status codes:
            0 = both satisfied (green), 1 = C1 violated (orange), 2 = C2 violated (yellow), 3 = both violated (red)
            -1 = no data (gray)
        """
        matrix = np.full((len(inference_range), len(generation_range)), -1, dtype=int)
        
        for i, inference_interval in enumerate(inference_range):
            for j, generation_length_ticks in enumerate(generation_range):
                # Convert ticks back to frames for lookup in percentile_data
                generation_length_frames = (generation_length_ticks * 2) - 1
                
                # Get the RTT for this specific generation length and percentile
                if (generation_length_frames, percentile) in percentile_data:
                    rtt_ms = percentile_data[(generation_length_frames, percentile)]
                    _, _, constraint_status = self._evaluate_constraints_detailed(
                        inference_interval, generation_length_ticks, rtt_ms
                    )
                    matrix[i, j] = constraint_status
                else:
                    # If we don't have data for this generation length, mark as no data
                    matrix[i, j] = -1
        
        return matrix
    
    def _plot_parameter_constraint_analysis(self):
        """Generate parameter constraint analysis (I vs GL validity) for all experiments."""
        if self.combined_data is None:
            return
        
        experiments = self.combined_data['experiment_source'].unique()
        percentiles = [50, 60, 70, 80, 90, 99.5]
        
        # Define parameter ranges based on available data
        all_gen_lengths = sorted(self.combined_data['generation_length'].unique())
        
        # Convert frames to ticks for odd generation lengths only (accompaniment focus)
        available_ticks = []
        for gl in all_gen_lengths:
            if gl % 2 == 1:  # Only odd frames (accompaniment)
                ticks = (gl + 1) // 2
                available_ticks.append(ticks)
        
        if not available_ticks:
            print("⚠️ No odd generation lengths found for parameter constraint analysis")
            return
        
        # Create consecutive tick range from 1 to max available
        min_ticks = min(available_ticks)
        max_ticks = max(available_ticks)
        generation_range = list(range(min_ticks, max_ticks + 1))  # Consecutive ticks 1-8
        
        # Define inference interval range (1 to max generation ticks)
        inference_intervals = list(range(1, min(8, max_ticks + 1)))  # 1-7 for 8 max ticks
        
        # Create subplot grid for all experiments and percentiles
        n_experiments = len(experiments)
        n_percentiles = len(percentiles)
        
        # Create a large figure with subplots
        fig = plt.figure(figsize=(6 * n_percentiles, 5 * n_experiments))
        
        for exp_idx, exp_name in enumerate(experiments):
            # Calculate percentiles for this experiment
            percentile_data = self._calculate_percentiles_by_generation_length(exp_name)
            
            if not percentile_data:
                continue
            
            for p_idx, percentile in enumerate(percentiles):
                subplot_idx = exp_idx * n_percentiles + p_idx + 1
                ax = plt.subplot(n_experiments, n_percentiles, subplot_idx)
                
                # Create constraint matrix
                matrix = self._create_constraint_matrix_detailed(
                    percentile_data, percentile, inference_intervals, generation_range
                )
                
                # Define colors for constraint status
                colors = [
                    '#404040',  # -1: No data (dark grey)
                    '#2E7D32',  # 0: Both satisfied (dark green) 
                    '#FF9800',  # 1: C1 violated (orange)
                    '#FFC107',  # 2: C2 violated (yellow)
                    '#D32F2F'   # 3: Both violated (red)
                ]
                
                # Create colormap
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(colors)
                
                # Plot matrix with proper orientation
                im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=3, 
                              interpolation='nearest', origin='upper')
                
                # Set labels and ticks
                ax.set_xticks(range(len(generation_range)))
                ax.set_xticklabels([f'{int(gt)}' for gt in generation_range])
                ax.set_yticks(range(len(inference_intervals)))
                ax.set_yticklabels([f'{ii}' for ii in inference_intervals])
                
                # Invert y-axis so (1,1) is at bottom-left
                ax.invert_yaxis()
                
                # Labels
                if exp_idx == n_experiments - 1:  # Bottom row
                    ax.set_xlabel('Generation Length (ticks)', fontsize=10, fontweight='bold')
                if p_idx == 0:  # Left column
                    ax.set_ylabel('Inference Interval (ticks)', fontsize=10, fontweight='bold')
                
                # Title
                if exp_idx == 0:  # Top row
                    ax.set_title(f'{percentile}th Percentile\n(Generation-Length-Specific RTT)', 
                                fontsize=11, fontweight='bold')
                if p_idx == 0:  # Left column
                    ax.text(-0.3, 0.5, exp_name, rotation=90, transform=ax.transAxes, 
                           fontsize=12, fontweight='bold', ha='center', va='center')
                
                # Add constraint status annotations
                for ii_idx in range(len(inference_intervals)):
                    for gt_idx in range(len(generation_range)):
                        if matrix[ii_idx, gt_idx] != -1:
                            status_text = ['✓', 'C1', 'C2', '✗'][matrix[ii_idx, gt_idx]]
                            text_color = 'white' if matrix[ii_idx, gt_idx] in [0, 3] else 'black'
                            ax.text(gt_idx, ii_idx, status_text, ha='center', va='center',
                                   color=text_color, fontsize=8, fontweight='bold')
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle('Parameter Constraint Analysis: Inference Interval vs Generation Length\n'
                    'Green=Valid, Orange=C1 Violated, Yellow=C2 Violated, Red=Both Violated', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add comprehensive legend for all constraint statuses
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[1], label='Valid (Both Constraints Satisfied)'),  # Green
            Patch(facecolor=colors[2], label='Constraint 1 Violated (Interval Too Short)'),  # Orange
            Patch(facecolor=colors[3], label='Constraint 2 Violated (Buffer Too Small)'),  # Yellow
            Patch(facecolor=colors[4], label='Both Constraints Violated'),  # Red
            Patch(facecolor=colors[0], label='No Benchmark Data')  # Gray
        ]
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.82, bottom=0.18, hspace=0.3, wspace=0.3)
        plt.savefig(self.plots_dir / "parameter_constraint_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual plots per experiment for better readability
        self._plot_individual_parameter_constraints(experiments, percentiles, 
                                                   generation_range, inference_intervals)
    
    def _plot_individual_parameter_constraints(self, experiments, percentiles, generation_range, inference_intervals):
        """Create individual parameter constraint plots for each experiment."""
        
        for exp_name in experiments:
            percentile_data = self._calculate_percentiles_by_generation_length(exp_name)
            if not percentile_data:
                continue
            
            # Create 2x3 subplot grid for the 6 percentiles
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for p_idx, percentile in enumerate(percentiles):
                ax = axes[p_idx]
                
                # Create constraint matrix
                matrix = self._create_constraint_matrix_detailed(
                    percentile_data, percentile, inference_intervals, generation_range
                )
                
                # Define colors for constraint status
                colors = [
                    '#404040',  # -1: No data (dark grey)
                    '#2E7D32',  # 0: Both satisfied (dark green) 
                    '#FF9800',  # 1: C1 violated (orange)
                    '#FFC107',  # 2: C2 violated (yellow)
                    '#D32F2F'   # 3: Both violated (red)
                ]
                
                from matplotlib.colors import ListedColormap
                cmap = ListedColormap(colors)
                
                # Plot matrix with proper orientation
                im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=-1, vmax=3, 
                              interpolation='nearest', origin='upper')
                
                # Set labels and ticks
                ax.set_xticks(range(len(generation_range)))
                ax.set_xticklabels([f'{int(gt)}' for gt in generation_range])
                ax.set_yticks(range(len(inference_intervals)))
                ax.set_yticklabels([f'{ii}' for ii in inference_intervals])
                
                # Invert y-axis so (1,1) is at bottom-left
                ax.invert_yaxis()
                
                ax.set_xlabel('Generation Length (ticks)', fontsize=12, fontweight='bold')
                if p_idx in [0, 3]:  # Label y-axis on leftmost plots of each row
                    ax.set_ylabel('Inference Interval (ticks)', fontsize=12, fontweight='bold')
                ax.set_title(f'{percentile}th Percentile\n(Generation-Length-Specific RTT)', 
                            fontsize=14, fontweight='bold')
                
                # Increase tick label font sizes
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Add constraint status annotations
                for ii_idx in range(len(inference_intervals)):
                    for gt_idx in range(len(generation_range)):
                        if matrix[ii_idx, gt_idx] != -1:
                            status_text = ['✓', 'C1', 'C2', '✗'][matrix[ii_idx, gt_idx]]
                            text_color = 'white' if matrix[ii_idx, gt_idx] in [0, 3] else 'black'
                            ax.text(gt_idx, ii_idx, status_text, ha='center', va='center',
                                   color=text_color, fontsize=10, fontweight='bold')
                
                # Add grid for better readability
                ax.grid(True, alpha=0.3)
            
            # Add overall title and legend
            fig.suptitle(f'Parameter Constraint Analysis: {exp_name}\n'
                        'Green=Valid, Orange=C1 Violated, Yellow=C2 Violated, Red=Both Violated', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Add comprehensive legend for all constraint statuses
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=colors[1], label='Valid (Both Constraints Satisfied)'),  # Green
                Patch(facecolor=colors[2], label='Constraint 1 Violated (Interval Too Short)'),  # Orange
                Patch(facecolor=colors[3], label='Constraint 2 Violated (Buffer Too Small)'),  # Yellow
                Patch(facecolor=colors[4], label='Both Constraints Violated'),  # Red
                Patch(facecolor=colors[0], label='No Benchmark Data')  # Gray
            ]
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5, fontsize=12)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.82, bottom=0.18, hspace=0.3, wspace=0.3)
            
            # Save individual experiment plot
            safe_exp_name = exp_name.replace('/', '_').replace(' ', '_')
            plt.savefig(self.plots_dir / f"parameter_constraints_{safe_exp_name}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _export_parameter_constraint_data(self):
        """Export parameter constraint analysis data to CSV files."""
        if self.combined_data is None:
            return
        
        experiments = self.combined_data['experiment_source'].unique()
        percentiles = [50, 60, 70, 80, 90, 99.5]
        
        # Get generation ticks range
        all_gen_lengths = sorted(self.combined_data['generation_length'].unique())
        available_ticks = []
        for gl in all_gen_lengths:
            if gl % 2 == 1:  # Only odd frames (accompaniment)
                ticks = (gl + 1) // 2
                available_ticks.append(ticks)
        
        if not available_ticks:
            return
        
        # Create consecutive tick range from 1 to max available
        min_ticks = min(available_ticks)
        max_ticks = max(available_ticks)
        generation_range = list(range(min_ticks, max_ticks + 1))  # Consecutive ticks 1-8
        inference_intervals = list(range(1, min(8, max_ticks + 1)))  # 1-7 for 8 max ticks
        
        # Export detailed constraint matrices for each experiment and percentile
        all_constraint_data = []
        
        for exp_name in experiments:
            percentile_data = self._calculate_percentiles_by_generation_length(exp_name)
            if not percentile_data:
                continue
            
            for percentile in percentiles:
                matrix = self._create_constraint_matrix_detailed(
                    percentile_data, percentile, inference_intervals, generation_range
                )
                
                # Convert matrix to rows for CSV export
                for ii_idx, inference_interval in enumerate(inference_intervals):
                    for gt_idx, generation_ticks_val in enumerate(generation_range):
                        constraint_status = matrix[ii_idx, gt_idx]
                        
                        # Convert status code to meaningful labels
                        status_labels = {
                            -1: 'No Data',
                            0: 'Valid', 
                            1: 'C1 Violated',
                            2: 'C2 Violated',
                            3: 'Both Violated'
                        }
                        
                        # Get RTT value for this combination
                        rtt_value = None
                        gen_length_frames = (generation_ticks_val * 2) - 1
                        if (gen_length_frames, percentile) in percentile_data:
                            rtt_value = percentile_data[(gen_length_frames, percentile)]
                        
                        all_constraint_data.append({
                            'experiment_name': exp_name,
                            'percentile': percentile,
                            'inference_interval_ticks': inference_interval,
                            'generation_length_ticks': generation_ticks_val,
                            'generation_length_frames': gen_length_frames,
                            'constraint_status_code': constraint_status,
                            'constraint_status': status_labels.get(constraint_status, 'Unknown'),
                            'rtt_percentile_ms': rtt_value,
                            'constraint1_satisfied': constraint_status in [0, 2],
                            'constraint2_satisfied': constraint_status in [0, 1],
                            'both_constraints_satisfied': constraint_status == 0
                        })
        
        if all_constraint_data:
            param_constraint_file = self.data_dir / "parameter_constraint_analysis.csv"
            param_constraint_df = pd.DataFrame(all_constraint_data)
            param_constraint_df.to_csv(param_constraint_file, index=False)
            print(f"   Parameter constraint analysis saved to: {param_constraint_file}")
    
    def generate_individual_analyses(self):
        """Generate individual analysis for each experiment using existing analyzer."""
        print("\\n📋 Generating individual experiment analyses...")
        
        for exp_name, exp_data in tqdm(self.experiments.items(), desc="Individual Analyses"):
            exp_output_dir = self.output_dir / f"individual_{exp_name}"
            
            # Create a temporary CSV for the individual analyzer
            temp_csv = exp_output_dir / "temp_data.csv"
            exp_output_dir.mkdir(exist_ok=True)
            exp_data['data'].to_csv(temp_csv, index=False)
            
            try:
                # Use existing analyzer for detailed individual analysis
                # Note: Individual analysis doesn't apply additional filtering since data is already filtered
                analyzer = GenerationLengthAnalyzer(str(temp_csv), str(exp_output_dir), anomaly_filter_pct=0.0)
                
                if analyzer.load_data():
                    analyzer.generate_all_visualizations()
                    analyzer.export_summary_table()
                    
                    # Generate text summary
                    summary = analyzer.generate_summary_report()
                    summary_file = exp_output_dir / "summary_report.txt"
                    with open(summary_file, 'w') as f:
                        f.write(summary)
                    
                    print(f"✅ Individual analysis complete for {exp_name}")
                else:
                    print(f"❌ Failed to analyze {exp_name}")
                    
            except Exception as e:
                print(f"❌ Error analyzing {exp_name}: {e}")
            
            finally:
                # Clean up temp file
                if temp_csv.exists():
                    temp_csv.unlink()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\\n📄 Generating bulk analysis summary report...")
        
        report = []
        report.append("# StreamMUSE Bulk Generation Length Analysis")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        
        # Experiment overview
        report.append("## Experiment Overview")
        report.append(f"- **Total Experiments**: {len(self.experiments)}")
        report.append(f"- **Total Requests**: {len(self.combined_data) if self.combined_data is not None else 'N/A'}")
        
        if self.summary_data is not None:
            gen_lengths = sorted(self.summary_data['generation_length'].unique())
            report.append(f"- **Generation Lengths Tested**: {gen_lengths}")
        
        report.append("")
        
        # Individual experiment details
        report.append("## Individual Experiment Details")
        for exp_name, exp_data in self.experiments.items():
            report.append(f"### {exp_name}")
            report.append(f"- **Type**: {exp_data['type']}")
            report.append(f"- **Source**: {exp_data['source_path']}")
            report.append(f"- **Generation Lengths**: {exp_data['generation_lengths']}")
            report.append(f"- **Total Requests**: {exp_data['total_requests']}")
            
            # Add config info if available
            if 'config' in exp_data.get('metadata', {}):
                config = exp_data['metadata']['config']
                if 'experiment' in config:
                    exp_config = config['experiment']
                    if 'description' in exp_config:
                        report.append(f"- **Description**: {exp_config['description']}")
            
            report.append("")
        
        # Performance comparison
        if self.summary_data is not None:
            report.append("## Performance Comparison")
            
            # Find best performing experiment for each generation length
            for gl in sorted(self.summary_data['generation_length'].unique()):
                gl_data = self.summary_data[self.summary_data['generation_length'] == gl]
                if 'round_trip_time_mean' in gl_data.columns and not gl_data.empty:
                    best_idx = gl_data['round_trip_time_mean'].idxmin()
                    best_row = self.summary_data.loc[best_idx]  # Use .loc instead of .iloc
                    best_exp = best_row['experiment_name']
                    best_rtt = best_row['round_trip_time_mean'] * 1000
                    
                    report.append(f"- **Generation Length {gl}**: Best = {best_exp} ({best_rtt:.1f}ms)")
            
            report.append("")
        
        # Constraint analysis summary
        constraint_data = self._calculate_constraint_satisfaction_rates()
        if constraint_data is not None:
            report.append("## Constraint Analysis Summary")
            report.append("Real-time musical constraints analysis (95th percentile RTT):")
            report.append("")
            
            # Overall constraint satisfaction by experiment
            for exp in constraint_data['experiment_name'].unique():
                exp_95th_data = constraint_data[
                    (constraint_data['experiment_name'] == exp) &
                    (constraint_data['percentile'] == 95)
                ]
                overall_satisfaction = exp_95th_data['both_satisfied'].mean() * 100
                
                # Find best generation length for this experiment
                best_gl_data = []
                for gt in sorted(exp_95th_data['generation_ticks'].unique()):
                    gt_data = exp_95th_data[exp_95th_data['generation_ticks'] == gt]
                    satisfaction_rate = gt_data['both_satisfied'].mean() * 100
                    best_gl_data.append((gt, satisfaction_rate))
                
                if best_gl_data:
                    best_gl, best_rate = max(best_gl_data, key=lambda x: x[1])
                    report.append(f"- **{exp}**: {overall_satisfaction:.1f}% overall satisfaction, best at {best_gl} ticks ({best_rate:.1f}%)")
            
            report.append("")
            
            # Constraint breakdown
            c1_rate = constraint_data['constraint1_satisfied'].mean() * 100
            c2_rate = constraint_data['constraint2_satisfied'].mean() * 100
            both_rate = constraint_data['both_satisfied'].mean() * 100
            
            report.append("### Constraint Breakdown (All Experiments)")
            report.append(f"- **Constraint 1 (No Overlap)**: {c1_rate:.1f}% satisfaction")
            report.append(f"- **Constraint 2 (Musical Buffer)**: {c2_rate:.1f}% satisfaction")  
            report.append(f"- **Both Constraints**: {both_rate:.1f}% satisfaction")
            report.append("")
        
        # Save report
        report_file = self.output_dir / "bulk_analysis_summary.md"
        with open(report_file, 'w') as f:
            f.write("\\n".join(report))
        
        print(f"✅ Summary report saved to: {report_file}")
        
        # Also print to console
        print("\\n" + "\\n".join(report))


def main():
    parser = argparse.ArgumentParser(
        description="Bulk analysis of multiple StreamMUSE generation length experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze multiple experiment directories
  %(prog)s experiments/exp1_results experiments/exp2_results experiments/exp3_results
  
  # Analyze with custom output directory
  %(prog)s experiments/*/results --output_dir my_analysis
  
  # Analyze both manual and bulk benchmark results together
  %(prog)s manual_results/ bulk_results/ --output_dir combined_analysis
        """
    )
    
    parser.add_argument("data_sources", nargs='+',
                       help="Paths to experiment directories to analyze")
    parser.add_argument("--output_dir", default="bulk_analysis_results",
                       help="Output directory for analysis results")
    parser.add_argument("--skip_individual", action="store_true",
                       help="Skip individual experiment analyses (faster)")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip plot generation")
    parser.add_argument("--anomaly_filter", type=float, default=0.0,
                       help="Filter out anomalies: percentage of data to remove from each tail (0-50)")
    
    args = parser.parse_args()
    
    print("📊 StreamMUSE Bulk Generation Length Analysis")
    print("=" * 50)
    print(f"Analyzing {len(args.data_sources)} experiment directories...")
    
    # Validate anomaly filter parameter
    if args.anomaly_filter < 0 or args.anomaly_filter > 50:
        print("❌ Anomaly filter percentage must be between 0 and 50")
        return 1
    
    if args.anomaly_filter > 0:
        print(f"🔍 Anomaly filtering enabled: removing {args.anomaly_filter}% from each tail")
    
    analyzer = BulkGenerationLengthAnalyzer(args.data_sources, args.output_dir, args.anomaly_filter)
    
    # Load all experiments
    if not analyzer.load_all_experiments():
        print("❌ Failed to load experiments")
        return 1
    
    # Combine data
    analyzer.combine_all_data()
    
    # Generate comparative analysis
    if not args.no_plots:
        analyzer.generate_comparative_analysis()
    
    # Generate individual analyses (optional)
    if not args.skip_individual:
        analyzer.generate_individual_analyses()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print(f"\\n✅ Bulk analysis complete! Results saved to: {Path(args.output_dir).absolute()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())