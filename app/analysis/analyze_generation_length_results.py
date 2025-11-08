#!/usr/bin/env python3
"""
Analysis and visualization script for generation length benchmark results.

This script can analyze results from either:
1. The new generation_length_benchmark.py output
2. Existing CSV files from manual benchmark runs (like in the notebook)

It creates comprehensive visualizations and analysis of generation length effects.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os
import re
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import statistics
from scipy import stats

class GenerationLengthAnalyzer:
    """
    Analyzes benchmark results to understand generation length effects on latency.
    """
    
    def __init__(self, data_source: str, output_dir: str = "analysis_results"):
        self.data_source = Path(data_source)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        self.detailed_data = None
        self.summary_data = None
        
    def load_data(self) -> bool:
        """Load data from various possible sources."""
        
        if self.data_source.is_file() and self.data_source.suffix == '.csv':
            # Single CSV file
            return self._load_single_csv()
        elif self.data_source.is_dir():
            # Directory with multiple files
            return self._load_directory()
        else:
            print(f"❌ Invalid data source: {self.data_source}")
            return False
    
    def _load_single_csv(self) -> bool:
        """Load data from a single CSV file with generation_length column."""
        try:
            df = pd.read_csv(self.data_source)
            if 'generation_length' not in df.columns:
                print("❌ CSV file must contain 'generation_length' column")
                return False
            
            self.detailed_data = df
            self._calculate_summary_from_detailed()
            print(f"✅ Loaded data from {self.data_source}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
    
    def _load_directory(self) -> bool:
        """Load data from directory structure (manual or automated results)."""
        
        # Try to load from generation_length_benchmark.py output structure
        analysis_dir = self.data_source / "analysis"
        if analysis_dir.exists():
            return self._load_automated_results()
        
        # Try to load from manual benchmark results
        return self._load_manual_results()
    
    def _load_automated_results(self) -> bool:
        """Load from automated benchmark output."""
        analysis_dir = self.data_source / "analysis"
        
        # Load detailed results
        detailed_file = analysis_dir / "detailed_results_all_generation_lengths.csv"
        if detailed_file.exists():
            self.detailed_data = pd.read_csv(detailed_file)
            
        # Load summary statistics  
        summary_file = analysis_dir / "summary_statistics.csv"
        if summary_file.exists():
            self.summary_data = pd.read_csv(summary_file)
            
        if self.detailed_data is not None or self.summary_data is not None:
            print(f"✅ Loaded automated benchmark results from {self.data_source}")
            return True
        
        print(f"❌ No automated results found in {self.data_source}")
        return False
    
    def _load_manual_results(self) -> bool:
        """Load from manual benchmark CSV files (like in the notebook)."""
        
        # Look for CSV files matching patterns like gen_length_XX.csv
        csv_files = list(self.data_source.glob("*.csv"))
        csv_files.extend(list(self.data_source.glob("**/*.csv")))
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.data_source}")
            return False
        
        all_data = []
        
        for csv_file in csv_files:
            # Try to extract generation length from filename
            gen_length = self._extract_generation_length_from_filename(csv_file.name)
            
            if gen_length is None:
                print(f"⚠️  Skipping {csv_file.name} - cannot extract generation length")
                continue
                
            try:
                df = pd.read_csv(csv_file)
                df['generation_length'] = gen_length
                df['source_file'] = csv_file.name
                all_data.append(df)
                print(f"✅ Loaded {csv_file.name} (generation length: {gen_length})")
                
            except Exception as e:
                print(f"⚠️  Error loading {csv_file.name}: {e}")
                continue
        
        if not all_data:
            print("❌ No valid data files found")
            return False
            
        self.detailed_data = pd.concat(all_data, ignore_index=True)
        self._calculate_summary_from_detailed()
        
        print(f"✅ Loaded {len(all_data)} files with {len(self.detailed_data)} total requests")
        return True
    
    def _extract_generation_length_from_filename(self, filename: str) -> Optional[int]:
        """Extract generation length from various filename patterns."""
        
        patterns = [
            r'gen_length_(\d+)',
            r'gen_(\d+)',
            r'generation_(\d+)',
            r'gl_(\d+)',
            r'(\d+)_frames',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
    
    def _calculate_summary_from_detailed(self):
        """Calculate summary statistics from detailed data."""
        if self.detailed_data is None:
            return
            
        summary_list = []
        
        for gen_length in sorted(self.detailed_data['generation_length'].unique()):
            df_subset = self.detailed_data[self.detailed_data['generation_length'] == gen_length]
            
            def safe_stats(series):
                if len(series) == 0:
                    return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0, 'p95': 0, 'p99': 0,
                           'upper_bound_60': 0, 'upper_bound_70': 0, 'upper_bound_80': 0, 'upper_bound_85': 0,
                           'upper_bound_90': 0, 'upper_bound_95': 0, 'upper_bound_98': 0, 'upper_bound_99': 0,
                           'upper_bound_99_5': 0, 'upper_bound_99_9': 0}
                
                # Calculate one-sided upper bounds (percentiles)
                upper_bounds = {}
                if len(series) >= 5:  # Need at least 5 samples for meaningful percentiles
                    try:
                        # Calculate percentiles directly from the data
                        # These represent: "X% of measurements are below this value"
                        percentiles = [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]
                        
                        for percentile in percentiles:
                            if percentile == 100:
                                # 100% = maximum value observed
                                value = series.max()
                                key_name = 'upper_bound_100'
                            else:
                                value = series.quantile(percentile / 100.0)
                                key_name = f'upper_bound_{int(percentile)}' if percentile == int(percentile) else f'upper_bound_{percentile}'.replace('.', '_')
                            upper_bounds[key_name] = value
                            
                    except Exception as e:
                        print(f"Warning: Error calculating percentiles: {e}")
                        # Fallback to zeros if calculation fails
                        for p in [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]:
                            key_name = f'upper_bound_{int(p)}' if p == int(p) else f'upper_bound_{p}'.replace('.', '_')
                            upper_bounds[key_name] = 0
                else:
                    # Not enough data for meaningful percentiles
                    for p in [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]:
                        key_name = f'upper_bound_{int(p)}' if p == int(p) else f'upper_bound_{p}'.replace('.', '_')
                        upper_bounds[key_name] = 0
                
                base_stats = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(), 
                    'median': series.median(),
                    'p95': series.quantile(0.95),
                    'p99': series.quantile(0.99)
                }
                
                # Combine base stats with upper bounds
                base_stats.update(upper_bounds)
                return base_stats
            
            summary = {'generation_length': gen_length, 'num_requests': len(df_subset)}
            
            # Add statistics for key metrics
            metrics = ['round_trip_time', 'server_processing_duration', 'inference_duration',
                      'preprocess_duration', 'postprocess_duration', 'total_network_latency']
            
            for metric in metrics:
                if metric in df_subset.columns:
                    stats = safe_stats(df_subset[metric])
                    for stat_name, value in stats.items():
                        summary[f"{metric}_{stat_name}"] = value
            
            # Add notes generated if available
            if 'num_generated_notes' in df_subset.columns:
                notes_stats = safe_stats(df_subset['num_generated_notes'])
                for stat_name, value in notes_stats.items():
                    summary[f"num_generated_notes_{stat_name}"] = value
                    
            summary_list.append(summary)
        
        self.summary_data = pd.DataFrame(summary_list)
    
    def _set_fine_x_ticks(self, ax):
        """Set finer x-axis ticks for better readability."""
        if self.summary_data is None:
            return
            
        gen_lengths = sorted(self.summary_data['generation_length'].unique())
        if len(gen_lengths) > 1:
            min_gap = min(gen_lengths[i+1] - gen_lengths[i] for i in range(len(gen_lengths)-1))
            tick_step = max(1, min_gap // 2)  # Use half the minimum gap, but at least 1
            
            x_min, x_max = min(gen_lengths), max(gen_lengths)
            x_ticks = np.arange(x_min, x_max + tick_step, tick_step)
            ax.set_xticks(x_ticks)
            ax.set_xlim(x_min - tick_step, x_max + tick_step)
    
    def _prepare_tick_based_data(self):
        """Filter data to odd generation lengths and convert to tick-based X-axis."""
        if self.summary_data is None:
            return None
            
        # Filter to only odd generation lengths (accompaniment frames)
        odd_data = self.summary_data[self.summary_data['generation_length'] % 2 == 1].copy()
        
        if len(odd_data) == 0:
            print("⚠️ No odd generation lengths found in data")
            return None
        
        # Convert frames to ticks: (odd_frame + 1) / 2
        odd_data['generation_ticks'] = (odd_data['generation_length'] + 1) / 2
        
        return odd_data
    
    def generate_all_visualizations(self):
        """Generate comprehensive visualization suite."""
        print("📊 Generating visualizations...")
        
        if self.summary_data is None:
            print("❌ No summary data available for visualization")
            return
            
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Generate all plots
        self._plot_latency_vs_generation_length()
        self._plot_variability_analysis()
        self._plot_distribution_comparison()
        self._plot_component_breakdown()
        self._plot_performance_metrics()
        self._plot_confidence_intervals()
        
        if self.detailed_data is not None:
            self._plot_detailed_distributions()
            self._plot_stacked_distributions()
            self._plot_correlation_analysis()
        
        print(f"✅ Visualizations saved to {self.plots_dir}")
    
    def _plot_latency_vs_generation_length(self):
        """Primary relationship plot: latency vs generation ticks (accompaniment focus)."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get tick-based data (odd generation lengths only)
        tick_data = self._prepare_tick_based_data()
        if tick_data is None:
            ax.text(0.5, 0.5, 'No odd generation lengths available for tick-based analysis', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Server Latency vs Generation Ticks', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "latency_vs_generation_length.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        metrics = [
            ('round_trip_time', 'Round Trip Time', 'o-', '#1f77b4'),
            ('server_processing_duration', 'Server Processing', 's--', '#ff7f0e'), 
            ('inference_duration', 'Inference Time', '^:', '#2ca02c'),
            ('total_network_latency', 'Network Latency', 'd-.', '#d62728'),
        ]
        
        for metric, label, style, color in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if mean_col in tick_data.columns:
                # Plot with error bars using ticks as X-axis
                ax.errorbar(
                    tick_data['generation_ticks'],
                    tick_data[mean_col] * 1000,  # Convert to ms
                    yerr=tick_data[std_col] * 1000 if std_col in tick_data.columns else None,
                    label=label,
                    marker=style[0],
                    linestyle=style[1:],
                    color=color,
                    capsize=5,
                    capthick=2,
                    linewidth=2.5,
                    markersize=8,
                    alpha=0.8
                )
        
        ax.set_xlabel('Generation Length (Ticks)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (milliseconds)', fontsize=14, fontweight='bold')
        ax.set_title('Server Latency vs Generation Ticks\n(Accompaniment Frame Focus)', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Set integer tick marks for ticks
        tick_values = sorted(tick_data['generation_ticks'].unique())
        ax.set_xticks(tick_values)
        ax.set_xlim(min(tick_values) - 0.5, max(tick_values) + 0.5)
        
        # Add musical time constraint lines (now much simpler with tick-based X-axis)
        if tick_values:
            frontier_x = tick_values
            
            # With tick-based X-axis, musical time is simply: ticks * 125ms
            musical_time_y = [tick * 125 for tick in frontier_x]
            buffer_minus1_y = [max(0, tick * 125 - 125) for tick in frontier_x]  # -1 tick buffer
            buffer_minus2_y = [max(0, tick * 125 - 250) for tick in frontier_x]  # -2 tick buffer
            
            # Plot as simple lines (no complex step functions needed)
            ax.plot(frontier_x, musical_time_y, color='red', linewidth=2, 
                   linestyle='-', alpha=0.8, label='Musical Time Deadline', zorder=5)
            
            ax.plot(frontier_x, buffer_minus1_y, color='purple', linewidth=2, 
                   linestyle='-', alpha=0.8, label='1 Useable Tick', zorder=5)
            
            ax.plot(frontier_x, buffer_minus2_y, color='darkblue', linewidth=2, 
                   linestyle='-', alpha=0.8, label='2 Useable Ticks', zorder=5)
        
        # Add trend line for round trip time
        if 'round_trip_time_mean' in tick_data.columns:
            x_ticks = tick_data['generation_ticks']
            y = tick_data['round_trip_time_mean'] * 1000
            z = np.polyfit(x_ticks, y, 1)
            p = np.poly1d(z)
            ax.plot(x_ticks, p(x_ticks), "gray", linestyle=":", alpha=0.7, linewidth=1, 
                   label=f'Trend: {z[0]:.2f}ms/tick')
        
        ax.legend(fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "latency_vs_generation_length.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_variability_analysis(self):
        """Plot showing how latency variability changes with generation length."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Standard deviation trends
        metrics = [
            ('round_trip_time', 'Round Trip Time', '#1f77b4'),
            ('server_processing_duration', 'Server Processing', '#ff7f0e'),
            ('inference_duration', 'Inference Time', '#2ca02c'),
        ]
        
        for metric, label, color in metrics:
            std_col = f"{metric}_std"
            if std_col in self.summary_data.columns:
                ax1.plot(
                    self.summary_data['generation_length'],
                    self.summary_data[std_col] * 1000,
                    marker='o',
                    linewidth=2.5,
                    markersize=8,
                    label=label,
                    color=color
                )
        
        ax1.set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Standard Deviation (ms)', fontsize=12, fontweight='bold')
        ax1.set_title('Latency Variability vs Generation Length', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Set smaller tick gaps on x-axis for left plot
        self._set_fine_x_ticks(ax1)
        
        # Right: Coefficient of variation (std/mean)
        for metric, label, color in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in self.summary_data.columns and std_col in self.summary_data.columns:
                cv = self.summary_data[std_col] / self.summary_data[mean_col] * 100
                ax2.plot(
                    self.summary_data['generation_length'],
                    cv,
                    marker='s',
                    linewidth=2.5,
                    markersize=8,
                    label=label,
                    color=color
                )
        
        ax2.set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Relative Variability vs Generation Length', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Set smaller tick gaps on x-axis for right plot
        self._set_fine_x_ticks(ax2)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "variability_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_comparison(self):
        """Box plots comparing distributions across generation lengths."""
        if self.detailed_data is None:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = [
            ('round_trip_time', 'Round Trip Time (ms)', 1000),
            ('inference_duration', 'Inference Duration (ms)', 1000),
            ('server_processing_duration', 'Server Processing (ms)', 1000),
            ('total_network_latency', 'Network Latency (ms)', 1000),
        ]
        
        for i, (metric, title, scale) in enumerate(metrics):
            if metric in self.detailed_data.columns:
                data_for_plot = []
                labels = []
                
                for gen_length in sorted(self.detailed_data['generation_length'].unique()):
                    subset = self.detailed_data[self.detailed_data['generation_length'] == gen_length]
                    data_for_plot.append(subset[metric] * scale)
                    labels.append(f"{gen_length}")
                
                bp = axes[i].boxplot(data_for_plot, labels=labels, patch_artist=True)
                
                # Color the boxes
                colors = sns.color_palette("husl", len(data_for_plot))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                axes[i].set_title(title, fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Generation Length (Frames)', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "distribution_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_breakdown(self):
        """Stacked bar chart showing latency component breakdown."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Prepare data for stacked bar chart
        components = [
            ('preprocess_duration', 'Preprocessing', '#ff9999'),
            ('inference_duration', 'Inference', '#66b3ff'),
            ('postprocess_duration', 'Postprocessing', '#99ff99'),
            ('total_network_latency', 'Network', '#ffcc99'),
        ]
        
        x = self.summary_data['generation_length']
        bottom = np.zeros(len(x))
        
        for component, label, color in components:
            mean_col = f"{component}_mean"
            if mean_col in self.summary_data.columns:
                values = self.summary_data[mean_col] * 1000  # Convert to ms
                ax.bar(x, values, bottom=bottom, label=label, color=color, alpha=0.8)
                bottom += values
        
        ax.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (milliseconds)', fontsize=14, fontweight='bold')
        ax.set_title('Latency Component Breakdown by Generation Length', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "component_breakdown.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_metrics(self):
        """Performance efficiency and scaling metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Estimated throughput
        if 'round_trip_time_mean' in self.summary_data.columns:
            throughput = 1.0 / self.summary_data['round_trip_time_mean']
            ax1.plot(self.summary_data['generation_length'], throughput, 
                    'o-', linewidth=3, markersize=10, color='#1f77b4')
            ax1.set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Estimated Throughput (req/sec)', fontsize=12, fontweight='bold')
            ax1.set_title('Request Throughput vs Generation Length', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            self._set_fine_x_ticks(ax1)
        
        # Right: Generation efficiency (notes per second)
        if ('num_generated_notes_mean' in self.summary_data.columns and 
            'inference_duration_mean' in self.summary_data.columns):
            
            efficiency = (self.summary_data['num_generated_notes_mean'] / 
                         self.summary_data['inference_duration_mean'])
            ax2.plot(self.summary_data['generation_length'], efficiency,
                    's-', linewidth=3, markersize=10, color='#ff7f0e')
            ax2.set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Notes Generated per Second', fontsize=12, fontweight='bold')
            ax2.set_title('Generation Efficiency vs Generation Length', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            self._set_fine_x_ticks(ax2)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_metrics.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_intervals(self):
        """Plot upper bound percentiles for round trip time (tick-based)."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get tick-based data (odd generation lengths only)
        tick_data = self._prepare_tick_based_data()
        if tick_data is None:
            ax.text(0.5, 0.5, 'No odd generation lengths available for tick-based analysis', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Round Trip Time Upper Bounds', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "confidence_intervals.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        metric = 'round_trip_time'
        scale = 1000  # Convert to ms
        
        mean_col = f"{metric}_mean"
        if mean_col not in tick_data.columns:
            ax.text(0.5, 0.5, 'No round trip time data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Round Trip Time Upper Bounds', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "confidence_intervals.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        x = tick_data['generation_ticks']
        y_mean = tick_data[mean_col] * scale
        
        # Plot mean line
        ax.plot(x, y_mean, 'ko-', linewidth=3, markersize=8, label='Mean', zorder=10)
        
        # Define percentile levels and colors
        percentiles = ['60', '70', '80', '85', '90', '95', '98', '99', '99_5', '99_9', '100']
        colors = ['#f0f8ff', '#e0f0ff', '#d0e8ff', '#c0e0ff', '#b0d8ff', '#90c8ff', '#70b8ff', '#50a8ff', '#3098ff', '#1088ff', '#000080']
        
        # Plot upper bounds
        for i, (percentile, color) in enumerate(zip(percentiles, colors)):
            upper_col = f"{metric}_upper_bound_{percentile}"
            
            if upper_col in tick_data.columns:
                y_upper = tick_data[upper_col] * scale
                
                # Check if we have valid data (not all zeros)
                if y_upper.sum() > 0:
                    # Convert percentile name for display
                    display_level = percentile.replace('_', '.')
                    
                    # Plot upper bound line
                    ax.plot(x, y_upper, '--', color=color, linewidth=2, markersize=6, 
                           marker='o', label=f'{display_level}% upper bound', alpha=0.8)
        
        ax.set_xlabel('Generation Length (Ticks)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Round Trip Time (milliseconds)', fontsize=14, fontweight='bold')
        ax.set_title('Round Trip Time Upper Bounds\n(X% of requests complete within this time - Accompaniment Focus)', fontsize=16, fontweight='bold')
        
        # Add musical time constraint lines (much simpler with tick-based X-axis)
        tick_values = sorted(tick_data['generation_ticks'].unique())
        
        # With tick-based X-axis, musical time is simply: ticks * 125ms
        musical_time_y = [tick * 125 for tick in tick_values]
        buffer_minus1_y = [max(0, tick * 125 - 125) for tick in tick_values]  # -1 tick buffer
        buffer_minus2_y = [max(0, tick * 125 - 250) for tick in tick_values]  # -2 tick buffer
        
        # Plot as simple lines (no complex step functions needed)
        ax.plot(tick_values, musical_time_y, color='red', linewidth=3, 
               linestyle='-', alpha=0.9, label='Musical Time Deadline', zorder=5)
        
        ax.plot(tick_values, buffer_minus1_y, color='purple', linewidth=3, 
               linestyle='-', alpha=0.9, label='1 Useable Tick', zorder=5)
        
        ax.plot(tick_values, buffer_minus2_y, color='darkblue', linewidth=3, 
               linestyle='-', alpha=0.9, label='2 Useable Ticks', zorder=5)
        
        # Create legend after all lines are plotted
        handles, labels = ax.get_legend_handles_labels()
        
        # Separate percentile lines from frontier lines
        percentile_handles = []
        percentile_labels = []
        frontier_handles = []
        frontier_labels = []
        
        for handle, label in zip(handles, labels):
            if any(frontier_name in label for frontier_name in ['Musical Time Deadline', 'Useable Tick']):
                frontier_handles.append(handle)
                frontier_labels.append(label)
            else:
                percentile_handles.append(handle)
                percentile_labels.append(label)
        
        # Create main legend for percentiles (upper area)
        if len(percentile_handles) > 6:
            legend1 = ax.legend(percentile_handles[:6], percentile_labels[:6], 
                               loc='upper left', fontsize=10, framealpha=0.9, title='Lower Percentiles')
            ax.add_artist(legend1)
            legend2 = ax.legend(percentile_handles[6:], percentile_labels[6:], 
                               loc='upper right', fontsize=10, framealpha=0.9, title='Higher Percentiles')
            ax.add_artist(legend2)
        else:
            if percentile_handles:
                legend1 = ax.legend(percentile_handles, percentile_labels, 
                                   loc='upper left', fontsize=10, framealpha=0.9, title='Percentiles')
                ax.add_artist(legend1)
        
        # Create frontier legend (bottom right)
        if frontier_handles:
            ax.legend(frontier_handles, frontier_labels, 
                     loc='lower right', fontsize=11, framealpha=0.9, title='Musical Time Constraints')
        
        ax.grid(True, alpha=0.3)
        self._set_fine_x_ticks(ax)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "confidence_intervals.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detailed_distributions(self):
        """Detailed histogram grid for each generation length."""
        generation_lengths = sorted(self.detailed_data['generation_length'].unique())
        n_lengths = len(generation_lengths)
        
        cols = min(4, n_lengths)
        rows = (n_lengths + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, gen_length in enumerate(generation_lengths):
            if i >= len(axes):
                break
                
            ax = axes[i]
            data = self.detailed_data[self.detailed_data['generation_length'] == gen_length]
            
            # Plot histogram with fixed bin width (5ms)
            rtt_data = data['round_trip_time'] * 1000  # Convert to ms
            bin_width = 5  # 5ms bin width
            bins = np.arange(rtt_data.min(), rtt_data.max() + bin_width, bin_width)
            ax.hist(rtt_data, bins=bins, alpha=0.7, 
                   edgecolor='black', linewidth=0.5, color=sns.color_palette("husl", n_lengths)[i])
            
            ax.set_title(f'Generation Length: {gen_length} frames', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Round Trip Time (ms)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_rtt = data['round_trip_time'].mean() * 1000
            std_rtt = data['round_trip_time'].std() * 1000
            median_rtt = data['round_trip_time'].median() * 1000
            
            stats_text = f'μ = {mean_rtt:.1f}ms\nσ = {std_rtt:.1f}ms\nmedian = {median_rtt:.1f}ms'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=9)
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "detailed_distributions.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stacked_distributions(self):
        """Plot all distributions as lines on a single plot."""
        generation_lengths = sorted(self.detailed_data['generation_length'].unique())
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = sns.color_palette("husl", len(generation_lengths))
        
        for i, gen_length in enumerate(generation_lengths):
            data = self.detailed_data[self.detailed_data['generation_length'] == gen_length]
            rtt_data = data['round_trip_time'] * 1000  # Convert to ms
            
            # Calculate histogram data with fixed bin width (5ms)
            bin_width = 5  # 5ms bin width
            bins = np.arange(rtt_data.min(), rtt_data.max() + bin_width, bin_width)
            counts, bins = np.histogram(rtt_data, bins=bins, density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            # Plot as line
            ax.plot(bin_centers, counts, label=f'{gen_length} frames', 
                   color=colors[i], linewidth=2.5, alpha=0.8)
            
            # Add vertical line for mean
            mean_rtt = rtt_data.mean()
            ax.axvline(mean_rtt, color=colors[i], linestyle='--', alpha=0.6, linewidth=1.5)
        
        ax.set_xlabel('Round Trip Time (ms)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax.set_title('Round Trip Time Distributions by Generation Length\n(Solid lines: distributions, Dashed lines: means)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "stacked_distributions.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_analysis(self):
        """Correlation matrix and scatter plots."""
        if self.detailed_data is None:
            return
            
        # Select numeric columns for correlation
        numeric_cols = ['generation_length', 'round_trip_time', 'server_processing_duration',
                       'inference_duration', 'preprocess_duration', 'postprocess_duration',
                       'total_network_latency']
        
        numeric_cols = [col for col in numeric_cols if col in self.detailed_data.columns]
        
        if len(numeric_cols) < 3:
            return
            
        correlation_data = self.detailed_data[numeric_cols]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Correlation matrix
        corr_matrix = correlation_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax1, cbar_kws={'shrink': 0.8})
        ax1.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Right: Scatter plot of generation length vs round trip time
        if 'round_trip_time' in correlation_data.columns:
            scatter = ax2.scatter(correlation_data['generation_length'],
                                correlation_data['round_trip_time'] * 1000,
                                alpha=0.6, s=30, c=correlation_data['generation_length'],
                                cmap='viridis')
            
            # Add trend line
            x = correlation_data['generation_length']
            y = correlation_data['round_trip_time'] * 1000
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
            
            ax2.set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Round Trip Time (ms)', fontsize=12, fontweight='bold')
            ax2.set_title('Generation Length vs Round Trip Time', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, ax=ax2, label='Generation Length')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "correlation_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> str:
        """Generate text summary of findings."""
        if self.summary_data is None:
            return "No data available for analysis."
        
        report = []
        report.append("# Generation Length Analysis Summary\n")
        
        # Basic statistics
        report.append(f"**Generation Lengths Tested:** {sorted(self.summary_data['generation_length'].tolist())}")
        report.append(f"**Total Requests:** {self.summary_data['num_requests'].sum()}")
        
        if 'round_trip_time_mean' in self.summary_data.columns:
            # Find optimal points
            min_latency_idx = self.summary_data['round_trip_time_mean'].idxmin()
            optimal_gen_length = self.summary_data.iloc[min_latency_idx]['generation_length']
            min_latency = self.summary_data.iloc[min_latency_idx]['round_trip_time_mean'] * 1000
            
            report.append(f"**Optimal Generation Length (Latency):** {optimal_gen_length} frames ({min_latency:.1f}ms)")
            
            # Latency range
            min_rtt = self.summary_data['round_trip_time_mean'].min() * 1000
            max_rtt = self.summary_data['round_trip_time_mean'].max() * 1000
            report.append(f"**Latency Range:** {min_rtt:.1f}ms - {max_rtt:.1f}ms")
            
            # Performance scaling
            gen_lengths = self.summary_data['generation_length'].values
            latencies = self.summary_data['round_trip_time_mean'].values * 1000
            if len(gen_lengths) > 1:
                slope = (latencies[-1] - latencies[0]) / (gen_lengths[-1] - gen_lengths[0])
                report.append(f"**Latency Scaling:** {slope:.2f}ms per frame increase")
        
        # Variability analysis
        if 'round_trip_time_std' in self.summary_data.columns:
            min_var_idx = self.summary_data['round_trip_time_std'].idxmin()
            most_consistent = self.summary_data.iloc[min_var_idx]['generation_length']
            min_std = self.summary_data.iloc[min_var_idx]['round_trip_time_std'] * 1000
            report.append(f"**Most Consistent Performance:** {most_consistent} frames ({min_std:.1f}ms std dev)")
        
        return "\n".join(report)
    
    def export_summary_table(self):
        """Export a clean summary table."""
        if self.summary_data is None:
            return
            
        # Create a clean summary table
        export_data = []
        
        for _, row in self.summary_data.iterrows():
            entry = {
                'Generation Length': int(row['generation_length']),
                'Requests': row['num_requests'],
            }
            
            if 'round_trip_time_mean' in row:
                entry['Mean RTT (ms)'] = f"{row['round_trip_time_mean'] * 1000:.1f}"
                entry['Std RTT (ms)'] = f"{row['round_trip_time_std'] * 1000:.1f}"
                
                # Add upper bounds for RTT
                if 'round_trip_time_upper_bound_95' in row:
                    upper_95 = row['round_trip_time_upper_bound_95'] * 1000
                    entry['95% Upper Bound RTT (ms)'] = f"{upper_95:.1f}"
                
                if 'round_trip_time_upper_bound_99' in row:
                    upper_99 = row['round_trip_time_upper_bound_99'] * 1000
                    entry['99% Upper Bound RTT (ms)'] = f"{upper_99:.1f}"
            
            if 'inference_duration_mean' in row:
                entry['Mean Inference (ms)'] = f"{row['inference_duration_mean'] * 1000:.1f}"
                
                # Add confidence intervals for inference duration
                if 'inference_duration_ci_95_lower' in row:
                    ci_95_lower = row['inference_duration_ci_95_lower'] * 1000
                    ci_95_upper = row['inference_duration_ci_95_upper'] * 1000
                    entry['95% CI Inference (ms)'] = f"[{ci_95_lower:.1f}, {ci_95_upper:.1f}]"
                
            if 'num_generated_notes_mean' in row:
                entry['Notes Generated'] = f"{row['num_generated_notes_mean']:.1f}"
                
            export_data.append(entry)
        
        export_df = pd.DataFrame(export_data)
        export_file = self.output_dir / "summary_table.csv"
        export_df.to_csv(export_file, index=False)
        
        # Also create a confidence intervals specific table
        self._export_confidence_intervals_table()
        
        print(f"📊 Summary table saved to: {export_file}")
        print("\nSummary Table:")
        print(export_df.to_string(index=False))
    
    def _export_confidence_intervals_table(self):
        """Export a detailed upper bounds table for round trip time only."""
        if self.summary_data is None:
            return
            
        ci_data = []
        
        for _, row in self.summary_data.iterrows():
            gen_length = int(row['generation_length'])
            
            # Round trip time upper bounds only
            if 'round_trip_time_mean' in row:
                base_entry = {
                    'Generation Length': gen_length,
                    'Mean (ms)': f"{row['round_trip_time_mean'] * 1000:.2f}",
                    'Std (ms)': f"{row['round_trip_time_std'] * 1000:.2f}",
                    'Sample Size': int(row['num_requests'])
                }
                
                # Add all upper bounds (percentiles)
                percentiles = [60, 70, 80, 85, 90, 95, 98, 99, 99.5, 99.9, 100]
                
                for percentile in percentiles:
                    if percentile == int(percentile):
                        col_name = f"round_trip_time_upper_bound_{int(percentile)}"
                        display_name = f"{int(percentile)}%"
                    else:
                        col_name = f"round_trip_time_upper_bound_{str(percentile).replace('.', '_')}"
                        display_name = f"{percentile}%"
                    
                    if col_name in row:
                        upper_bound = row[col_name] * 1000
                        base_entry[f'{display_name} Upper Bound (ms)'] = f"{upper_bound:.2f}"
                
                ci_data.append(base_entry)
        
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            ci_file = self.output_dir / "round_trip_time_upper_bounds.csv"
            ci_df.to_csv(ci_file, index=False)
            print(f"📊 Round trip time upper bounds saved to: {ci_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze generation length benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze automated benchmark results
  %(prog)s results/generation_length_analysis
  
  # Analyze manual CSV files in directory
  %(prog)s results/manual_benchmarks --output_dir custom_analysis
  
  # Analyze single combined CSV file
  %(prog)s combined_results.csv
        """
    )
    
    parser.add_argument("data_source", 
                       help="Path to data source (CSV file or directory)")
    parser.add_argument("--output_dir", default="analysis_results",
                       help="Output directory for analysis results")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip plot generation")
    
    args = parser.parse_args()
    
    print("📊 Generation Length Analysis Tool")
    print("=" * 40)
    
    analyzer = GenerationLengthAnalyzer(args.data_source, args.output_dir)
    
    if not analyzer.load_data():
        print("❌ Failed to load data")
        return 1
    
    # Generate summary report
    summary = analyzer.generate_summary_report()
    print(f"\n{summary}")
    
    # Export summary table
    analyzer.export_summary_table()
    
    # Generate visualizations
    if not args.no_plots:
        analyzer.generate_all_visualizations()
    
    print(f"\n✅ Analysis complete! Results saved to: {Path(args.output_dir).absolute()}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())