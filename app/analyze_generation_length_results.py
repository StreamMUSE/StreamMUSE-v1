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
                           'ci_80_lower': 0, 'ci_80_upper': 0, 'ci_85_lower': 0, 'ci_85_upper': 0,
                           'ci_90_lower': 0, 'ci_90_upper': 0, 'ci_95_lower': 0, 'ci_95_upper': 0,
                           'ci_98_lower': 0, 'ci_98_upper': 0, 'ci_99_lower': 0, 'ci_99_upper': 0,
                           'ci_99_5_lower': 0, 'ci_99_5_upper': 0, 'ci_99_9_lower': 0, 'ci_99_9_upper': 0}
                
                # Calculate confidence intervals if we have enough data
                confidence_intervals = {}
                if len(series) >= 3:  # Need at least 3 samples for meaningful CI
                    try:
                        # Calculate confidence intervals using t-distribution
                        mean = series.mean()
                        sem = stats.sem(series)  # Standard error of the mean
                        
                        confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]
                        ci_names = ['ci_80', 'ci_85', 'ci_90', 'ci_95', 'ci_98', 'ci_99', 'ci_99_5', 'ci_99_9']
                        
                        for conf_level, ci_name in zip(confidence_levels, ci_names):
                            # t-distribution critical value
                            t_crit = stats.t.ppf((1 + conf_level) / 2, len(series) - 1)
                            margin_error = t_crit * sem
                            
                            confidence_intervals[f'{ci_name}_lower'] = mean - margin_error
                            confidence_intervals[f'{ci_name}_upper'] = mean + margin_error
                    except:
                        # Fallback to zeros if calculation fails
                        for ci_name in ['ci_80', 'ci_85', 'ci_90', 'ci_95', 'ci_98', 'ci_99', 'ci_99_5', 'ci_99_9']:
                            confidence_intervals[f'{ci_name}_lower'] = 0
                            confidence_intervals[f'{ci_name}_upper'] = 0
                else:
                    # Not enough data for meaningful confidence intervals
                    for ci_name in ['ci_80', 'ci_85', 'ci_90', 'ci_95', 'ci_98', 'ci_99', 'ci_99_5', 'ci_99_9']:
                        confidence_intervals[f'{ci_name}_lower'] = 0
                        confidence_intervals[f'{ci_name}_upper'] = 0
                
                base_stats = {
                    'mean': series.mean(),
                    'std': series.std(),
                    'min': series.min(),
                    'max': series.max(), 
                    'median': series.median(),
                    'p95': series.quantile(0.95),
                    'p99': series.quantile(0.99)
                }
                
                # Combine base stats with confidence intervals
                base_stats.update(confidence_intervals)
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
            self._plot_correlation_analysis()
        
        print(f"✅ Visualizations saved to {self.plots_dir}")
    
    def _plot_latency_vs_generation_length(self):
        """Primary relationship plot: latency vs generation length."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = [
            ('round_trip_time', 'Round Trip Time', 'o-', '#1f77b4'),
            ('server_processing_duration', 'Server Processing', 's--', '#ff7f0e'), 
            ('inference_duration', 'Inference Time', '^:', '#2ca02c'),
            ('total_network_latency', 'Network Latency', 'd-.', '#d62728'),
        ]
        
        for metric, label, style, color in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            
            if mean_col in self.summary_data.columns:
                # Plot with error bars
                ax.errorbar(
                    self.summary_data['generation_length'],
                    self.summary_data[mean_col] * 1000,  # Convert to ms
                    yerr=self.summary_data[std_col] * 1000 if std_col in self.summary_data.columns else None,
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
        
        ax.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (milliseconds)', fontsize=14, fontweight='bold')
        ax.set_title('Server Latency vs Generation Length', fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Set smaller tick gaps on x-axis
        gen_lengths = sorted(self.summary_data['generation_length'].unique())
        if len(gen_lengths) > 1:
            min_gap = min(gen_lengths[i+1] - gen_lengths[i] for i in range(len(gen_lengths)-1))
            tick_step = max(1, min_gap // 2)  # Use half the minimum gap, but at least 1
            
            x_min, x_max = min(gen_lengths), max(gen_lengths)
            x_ticks = np.arange(x_min, x_max + tick_step, tick_step)
            ax.set_xticks(x_ticks)
            ax.set_xlim(x_min - tick_step, x_max + tick_step)
        
        # Add trend line for round trip time
        if 'round_trip_time_mean' in self.summary_data.columns:
            x = self.summary_data['generation_length']
            y = self.summary_data['round_trip_time_mean'] * 1000
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.5, linewidth=1, 
                   label=f'Trend: {z[0]:.2f}ms/frame')
            ax.legend(fontsize=12, framealpha=0.9)
        
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
        """Plot confidence intervals for round trip time only."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        metric = 'round_trip_time'
        scale = 1000  # Convert to ms
        
        mean_col = f"{metric}_mean"
        if mean_col not in self.summary_data.columns:
            ax.text(0.5, 0.5, 'No round trip time data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Round Trip Time Confidence Intervals', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / "confidence_intervals.png", dpi=300, bbox_inches='tight')
            plt.close()
            return
        
        x = self.summary_data['generation_length']
        y_mean = self.summary_data[mean_col] * scale
        
        # Plot mean line
        ax.plot(x, y_mean, 'ko-', linewidth=3, markersize=8, label='Mean', zorder=10)
        
        # Define confidence levels and colors (lighter to darker as confidence increases)
        confidence_levels = ['80', '85', '90', '95', '98', '99', '99_5', '99_9']
        colors = ['#e6f3ff', '#cce7ff', '#99d6ff', '#66c2ff', '#33adff', '#0099ff', '#0080cc', '#006699']
        alphas = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        
        # Plot confidence intervals from widest to narrowest (so narrower ones appear on top)
        for conf_level, color, alpha in zip(reversed(confidence_levels), reversed(colors), reversed(alphas)):
            lower_col = f"{metric}_ci_{conf_level}_lower"
            upper_col = f"{metric}_ci_{conf_level}_upper"
            
            if lower_col in self.summary_data.columns and upper_col in self.summary_data.columns:
                y_lower = self.summary_data[lower_col] * scale
                y_upper = self.summary_data[upper_col] * scale
                
                # Convert confidence level name for display
                display_level = conf_level.replace('_', '.')
                
                # Plot confidence interval as filled area
                ax.fill_between(x, y_lower, y_upper, 
                               alpha=alpha, color=color, 
                               label=f'{display_level}% CI', zorder=len(confidence_levels)-confidence_levels.index(conf_level))
        
        ax.set_xlabel('Generation Length (Frames)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Round Trip Time (milliseconds)', fontsize=14, fontweight='bold')
        ax.set_title('Round Trip Time with Confidence Intervals', fontsize=16, fontweight='bold')
        
        # Arrange legend with most common CIs first
        handles, labels = ax.get_legend_handles_labels()
        # Reorder to show mean first, then common CIs (95%, 99%), then others
        common_order = ['Mean', '95% CI', '99% CI', '90% CI', '99.5% CI', '80% CI', '85% CI', '98% CI', '99.9% CI']
        ordered_handles = []
        ordered_labels = []
        
        for desired_label in common_order:
            for handle, label in zip(handles, labels):
                if label == desired_label:
                    ordered_handles.append(handle)
                    ordered_labels.append(label)
                    break
        
        ax.legend(ordered_handles, ordered_labels, fontsize=11, loc='best', framealpha=0.9)
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
            
            # Plot histogram
            ax.hist(data['round_trip_time'] * 1000, bins=20, alpha=0.7, 
                   edgecolor='black', color=sns.color_palette("husl", n_lengths)[i])
            
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
                
                # Add confidence intervals for RTT
                if 'round_trip_time_ci_95_lower' in row:
                    ci_95_lower = row['round_trip_time_ci_95_lower'] * 1000
                    ci_95_upper = row['round_trip_time_ci_95_upper'] * 1000
                    entry['95% CI RTT (ms)'] = f"[{ci_95_lower:.1f}, {ci_95_upper:.1f}]"
                
                if 'round_trip_time_ci_99_lower' in row:
                    ci_99_lower = row['round_trip_time_ci_99_lower'] * 1000
                    ci_99_upper = row['round_trip_time_ci_99_upper'] * 1000
                    entry['99% CI RTT (ms)'] = f"[{ci_99_lower:.1f}, {ci_99_upper:.1f}]"
            
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
        """Export a detailed confidence intervals table for round trip time only."""
        if self.summary_data is None:
            return
            
        ci_data = []
        
        for _, row in self.summary_data.iterrows():
            gen_length = int(row['generation_length'])
            
            # Round trip time confidence intervals only
            if 'round_trip_time_mean' in row:
                base_entry = {
                    'Generation Length': gen_length,
                    'Mean (ms)': f"{row['round_trip_time_mean'] * 1000:.2f}",
                    'Std (ms)': f"{row['round_trip_time_std'] * 1000:.2f}",
                    'Sample Size': int(row['num_requests'])
                }
                
                # Add all standard confidence intervals
                ci_levels = ['80', '85', '90', '95', '98', '99', '99_5', '99_9']
                
                for ci_level in ci_levels:
                    lower_col = f"round_trip_time_ci_{ci_level}_lower"
                    upper_col = f"round_trip_time_ci_{ci_level}_upper"
                    
                    if lower_col in row and upper_col in row:
                        lower = row[lower_col] * 1000
                        upper = row[upper_col] * 1000
                        width = upper - lower
                        
                        level_name = ci_level.replace('_', '.')
                        base_entry[f'{level_name}% CI Lower'] = f"{lower:.2f}"
                        base_entry[f'{level_name}% CI Upper'] = f"{upper:.2f}"
                        base_entry[f'{level_name}% CI Width'] = f"{width:.2f}"
                
                ci_data.append(base_entry)
        
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            ci_file = self.output_dir / "round_trip_time_confidence_intervals.csv"
            ci_df.to_csv(ci_file, index=False)
            print(f"📊 Round trip time confidence intervals saved to: {ci_file}")


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