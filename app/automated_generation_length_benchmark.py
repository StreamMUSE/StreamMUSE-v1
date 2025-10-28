#!/usr/bin/env python3
"""
Fully automated generation length benchmark system.

This script automatically starts multiple server instances with different
GENERATION_LENGTH_FRAMES values on different ports and runs benchmarks
against each one concurrently. No manual intervention required.
"""

import argparse
import subprocess
import time
import requests
import threading
import signal
import sys
import os
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Optional
import statistics
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

class AutomatedGenerationLengthBenchmark:
    """
    Fully automated benchmark system that manages multiple server instances.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results_dir = Path(config['output_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.raw_data_dir = self.results_dir / "raw_data"
        self.analysis_dir = self.results_dir / "analysis"
        self.plots_dir = self.results_dir / "plots"
        self.logs_dir = self.results_dir / "server_logs"
        
        for dir_path in [self.raw_data_dir, self.analysis_dir, self.plots_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.server_processes = {}  # port -> process
        self.server_configs = {}    # port -> config
        self.all_results = []
        self.summary_stats = []
        
        # Register signal handler for cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals by cleaning up servers."""
        print(f"\n🛑 Received signal {signum}. Cleaning up...")
        self.cleanup_servers()
        sys.exit(0)
    
    def find_available_ports(self, start_port: int = 8000, count: int = 10) -> List[int]:
        """Find available ports starting from start_port."""
        available_ports = []
        port = start_port
        
        while len(available_ports) < count:
            if self._is_port_available(port):
                available_ports.append(port)
            port += 1
            
            if port > start_port + 1000:  # Safety limit
                break
        
        return available_ports
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    def start_server_for_generation_length(self, generation_length: int, port: int) -> bool:
        """Start a server instance with specific generation length on specified port."""
        
        print(f"🚀 Starting server for generation length {generation_length} on port {port}...")
        
        # Prepare environment variables
        env = os.environ.copy()
        env['CHECKPOINT_PATH'] = self.config['checkpoint_path']
        env['GENERATION_LENGTH_FRAMES'] = str(generation_length)
        env['MODEL_MAX_SEQ_LEN_FRAMES'] = str(self.config.get('model_max_seq_len_frames', 96))
        env['MODEL_SIZE'] = self.config.get('model_size', '0.12B')
        
        # Prepare server command
        server_cmd = [
            sys.executable, "-m", "uvicorn",
            "app.server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--log-level", "info"
        ]
        
        # Start server process
        log_file = self.logs_dir / f"server_gen_{generation_length}_port_{port}.log"
        
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    server_cmd,
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=Path.cwd(),
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                )
            
            self.server_processes[port] = process
            self.server_configs[port] = {
                'generation_length': generation_length,
                'log_file': log_file,
                'start_time': time.time()
            }
            
            # Wait for server to start
            server_url = f"http://localhost:{port}/generate_accompaniment"
            if self._wait_for_server(server_url, timeout=60):
                print(f"✅ Server started successfully for generation length {generation_length} on port {port}")
                return True
            else:
                print(f"❌ Server failed to start for generation length {generation_length} on port {port}")
                self._stop_server(port)
                return False
                
        except Exception as e:
            print(f"❌ Error starting server for generation length {generation_length}: {e}")
            return False
    
    def _wait_for_server(self, server_url: str, timeout: int = 60) -> bool:
        """Wait for server to become responsive."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to hit the clear_history endpoint
                clear_url = server_url.replace('/generate_accompaniment', '/clear_history')
                response = requests.post(clear_url, timeout=2)
                
                if response.status_code in [200, 503]:  # 503 means engine not loaded yet
                    # If we get 503, wait a bit more for the engine to load
                    if response.status_code == 503:
                        time.sleep(2)
                        continue
                    return True
                    
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        return False
    
    def _stop_server(self, port: int):
        """Stop a specific server instance."""
        if port in self.server_processes:
            process = self.server_processes[port]
            try:
                # Try graceful termination first
                if hasattr(os, 'killpg'):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if necessary
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                        
                print(f"🛑 Stopped server on port {port}")
                
            except (ProcessLookupError, psutil.NoSuchProcess):
                # Process already terminated
                pass
            except Exception as e:
                print(f"⚠️  Error stopping server on port {port}: {e}")
            
            del self.server_processes[port]
            if port in self.server_configs:
                del self.server_configs[port]
    
    def cleanup_servers(self):
        """Stop all running server instances."""
        print("🧹 Cleaning up server instances...")
        
        for port in list(self.server_processes.keys()):
            self._stop_server(port)
        
        print("✅ All servers stopped")
    
    def run_benchmark_for_server(self, port: int, generation_length: int) -> Optional[pd.DataFrame]:
        """Run benchmark against a specific server instance."""
        
        server_url = f"http://localhost:{port}/generate_accompaniment"
        
        print(f"📊 Running benchmark for generation length {generation_length} (port {port})...")
        
        # Clear server history first
        try:
            clear_url = server_url.replace('/generate_accompaniment', '/clear_history')
            requests.post(clear_url, timeout=5)
        except:
            pass
        
        # Prepare output files
        csv_file = self.raw_data_dir / f"gen_length_{generation_length}_port_{port}.csv"
        json_file = self.raw_data_dir / f"gen_length_{generation_length}_port_{port}.json"
        
        # Run the existing benchmark script
        benchmark_cmd = [
            sys.executable, "app/benchmark.py",
            "--server_url", server_url,
            "--num_requests", str(self.config['requests_per_length']),
            "--output_file", str(csv_file)
        ]
        
        try:
            result = subprocess.run(
                benchmark_cmd,
                capture_output=True,
                text=True,
                timeout=self.config['benchmark_timeout']
            )
            
            if result.returncode != 0:
                print(f"❌ Benchmark failed for generation length {generation_length}")
                print(f"STDERR: {result.stderr}")
                return None
            
            # Load and validate results
            if not csv_file.exists():
                print(f"❌ CSV output not found for generation length {generation_length}")
                return None
            
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                print(f"❌ No data collected for generation length {generation_length}")
                return None
            
            # Add generation length column
            df['generation_length'] = generation_length
            df['server_port'] = port
            
            # Re-save with additional columns
            df.to_csv(csv_file, index=False)
            
            print(f"✅ Benchmark completed for generation length {generation_length} ({len(df)} requests)")
            return df
            
        except subprocess.TimeoutExpired:
            print(f"❌ Benchmark timed out for generation length {generation_length}")
            return None
        except Exception as e:
            print(f"❌ Error running benchmark for generation length {generation_length}: {e}")
            return None
    
    def run_automated_benchmark_suite(self) -> bool:
        """Run the complete automated benchmark suite."""
        
        print("🎵 Starting Automated Generation Length Benchmark Suite")
        print("=" * 60)
        print(f"Generation lengths: {self.config['generation_lengths']}")
        print(f"Requests per length: {self.config['requests_per_length']}")
        print(f"Output directory: {self.results_dir}")
        
        # Check if checkpoint exists
        if not Path(self.config['checkpoint_path']).exists():
            print(f"❌ Checkpoint file not found: {self.config['checkpoint_path']}")
            return False
        
        # Find available ports
        required_ports = len(self.config['generation_lengths'])
        available_ports = self.find_available_ports(count=required_ports)
        
        if len(available_ports) < required_ports:
            print(f"❌ Not enough available ports. Need {required_ports}, found {len(available_ports)}")
            return False
        
        print(f"📡 Using ports: {available_ports}")
        
        try:
            # Phase 1: Start all servers
            print(f"\n🚀 Phase 1: Starting {required_ports} server instances...")
            
            successful_starts = 0
            for i, generation_length in enumerate(self.config['generation_lengths']):
                port = available_ports[i]
                if self.start_server_for_generation_length(generation_length, port):
                    successful_starts += 1
                else:
                    print(f"⚠️  Failed to start server for generation length {generation_length}")
            
            if successful_starts == 0:
                print("❌ No servers started successfully")
                return False
            
            print(f"✅ Started {successful_starts}/{required_ports} servers")
            
            # Small delay to ensure all servers are fully ready
            print("⏳ Waiting for all servers to fully initialize...")
            time.sleep(10)
            
            # Phase 2: Run benchmarks concurrently
            print(f"\n📊 Phase 2: Running benchmarks concurrently...")
            
            benchmark_futures = []
            
            with ThreadPoolExecutor(max_workers=min(successful_starts, 4)) as executor:
                for port, config in self.server_configs.items():
                    generation_length = config['generation_length']
                    future = executor.submit(self.run_benchmark_for_server, port, generation_length)
                    benchmark_futures.append((future, generation_length, port))
                
                # Collect results as they complete
                for future, generation_length, port in benchmark_futures:
                    try:
                        df = future.result(timeout=self.config['benchmark_timeout'] + 60)
                        if df is not None:
                            self.all_results.append(df)
                            
                            # Calculate summary stats
                            summary = self._calculate_summary_stats(df, generation_length)
                            self.summary_stats.append(summary)
                            
                    except Exception as e:
                        print(f"❌ Error in benchmark for generation length {generation_length}: {e}")
            
            # Phase 3: Analysis and visualization
            if self.all_results:
                print(f"\n📈 Phase 3: Generating analysis...")
                self._export_combined_results()
                
                if self.config['generate_plots']:
                    self._generate_all_visualizations()
                
                if self.config['generate_report']:
                    self._generate_analysis_report()
                
                print(f"\n🎉 Benchmark suite completed successfully!")
                print(f"   Tested {len(self.all_results)} generation lengths")
                print(f"   Total requests: {sum(len(df) for df in self.all_results)}")
                print(f"   Results saved to: {self.results_dir.absolute()}")
                
                return True
            else:
                print("❌ No successful benchmark results collected")
                return False
                
        finally:
            # Always cleanup servers
            self.cleanup_servers()
    
    def _calculate_summary_stats(self, df: pd.DataFrame, generation_length: int) -> Dict[str, Any]:
        """Calculate summary statistics for a generation length."""
        
        def safe_stats(series):
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
        }
        
        # Add statistics for key metrics
        metrics = ['round_trip_time', 'server_processing_duration', 'inference_duration',
                  'preprocess_duration', 'postprocess_duration', 'total_network_latency']
        
        for metric in metrics:
            if metric in df.columns:
                stats = safe_stats(df[metric])
                for stat_name, value in stats.items():
                    summary[f"{metric}_{stat_name}"] = value
        
        # Add notes generated if available
        if 'num_generated_notes' in df.columns:
            notes_stats = safe_stats(df['num_generated_notes'])
            for stat_name, value in notes_stats.items():
                summary[f"num_generated_notes_{stat_name}"] = value
        
        return summary
    
    def _export_combined_results(self):
        """Export combined results and summary statistics."""
        print("💾 Exporting combined results...")
        
        # Combine all detailed results
        if self.all_results:
            combined_df = pd.concat(self.all_results, ignore_index=True)
            combined_csv = self.analysis_dir / "detailed_results_all_generation_lengths.csv"
            combined_df.to_csv(combined_csv, index=False)
            print(f"   📄 Detailed results: {combined_csv}")
        
        # Export summary statistics
        if self.summary_stats:
            summary_df = pd.DataFrame(self.summary_stats)
            summary_csv = self.analysis_dir / "summary_statistics.csv"
            summary_df.to_csv(summary_csv, index=False)
            print(f"   📊 Summary statistics: {summary_csv}")
            
            # Clean summary table
            self._create_clean_summary_table(summary_df)
    
    def _create_clean_summary_table(self, summary_df: pd.DataFrame):
        """Create a clean, readable summary table."""
        clean_data = []
        
        for _, row in summary_df.iterrows():
            entry = {
                'Generation Length': int(row['generation_length']),
                'Requests': int(row['num_requests']),
                'Mean RTT (ms)': f"{row['round_trip_time_mean'] * 1000:.1f}",
                'Std RTT (ms)': f"{row['round_trip_time_std'] * 1000:.1f}",
                'Mean Inference (ms)': f"{row['inference_duration_mean'] * 1000:.1f}",
                'P95 RTT (ms)': f"{row['round_trip_time_p95'] * 1000:.1f}",
            }
            
            if 'num_generated_notes_mean' in row:
                entry['Notes Generated'] = f"{row['num_generated_notes_mean']:.1f}"
                
            clean_data.append(entry)
        
        clean_df = pd.DataFrame(clean_data)
        clean_csv = self.analysis_dir / "summary_table_clean.csv"
        clean_df.to_csv(clean_csv, index=False)
        
        print(f"   📋 Clean summary: {clean_csv}")
        print("\n📊 Results Summary:")
        print(clean_df.to_string(index=False))
    
    def _generate_all_visualizations(self):
        """Generate comprehensive visualizations."""
        print("🎨 Generating visualizations...")
        
        if not self.summary_stats:
            print("   No data for visualization")
            return
        
        summary_df = pd.DataFrame(self.summary_stats)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Generate plots
        plots_created = []
        
        try:
            self._plot_latency_vs_generation_length(summary_df)
            plots_created.append("latency_vs_generation_length.png")
        except Exception as e:
            print(f"   ⚠️  Error creating latency plot: {e}")
        
        try:
            self._plot_variability_analysis(summary_df)
            plots_created.append("variability_analysis.png")
        except Exception as e:
            print(f"   ⚠️  Error creating variability plot: {e}")
        
        try:
            self._plot_distribution_comparison()
            plots_created.append("distribution_comparison.png")
        except Exception as e:
            print(f"   ⚠️  Error creating distribution plot: {e}")
        
        try:
            self._plot_performance_scaling(summary_df)
            plots_created.append("performance_scaling.png")
        except Exception as e:
            print(f"   ⚠️  Error creating performance plot: {e}")
        
        print(f"   📈 Created {len(plots_created)} plots in {self.plots_dir}")
    
    def _plot_latency_vs_generation_length(self, summary_df: pd.DataFrame):
        """Plot latency vs generation length with error bars."""
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
            
            if mean_col in summary_df.columns and std_col in summary_df.columns:
                ax.errorbar(
                    summary_df['generation_length'],
                    summary_df[mean_col] * 1000,  # Convert to ms
                    yerr=summary_df[std_col] * 1000,
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
        ax.set_title('Server Latency vs Generation Length\n(Automated Benchmark Results)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "latency_vs_generation_length.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_variability_analysis(self, summary_df: pd.DataFrame):
        """Plot variability analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Standard deviation
        metrics = [
            ('round_trip_time', 'Round Trip Time', '#1f77b4'),
            ('server_processing_duration', 'Server Processing', '#ff7f0e'),
            ('inference_duration', 'Inference Time', '#2ca02c'),
        ]
        
        for metric, label, color in metrics:
            std_col = f"{metric}_std"
            if std_col in summary_df.columns:
                ax1.plot(
                    summary_df['generation_length'],
                    summary_df[std_col] * 1000,
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
        
        # Right: Coefficient of variation
        for metric, label, color in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in summary_df.columns and std_col in summary_df.columns:
                cv = summary_df[std_col] / summary_df[mean_col] * 100
                ax2.plot(
                    summary_df['generation_length'],
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
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "variability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_distribution_comparison(self):
        """Box plots comparing distributions."""
        if not self.all_results:
            return
            
        combined_df = pd.concat(self.all_results, ignore_index=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = [
            ('round_trip_time', 'Round Trip Time (ms)', 1000),
            ('inference_duration', 'Inference Duration (ms)', 1000),
            ('server_processing_duration', 'Server Processing (ms)', 1000),
            ('total_network_latency', 'Network Latency (ms)', 1000),
        ]
        
        for i, (metric, title, scale) in enumerate(metrics):
            if metric in combined_df.columns:
                data_for_plot = []
                labels = []
                
                for gen_length in sorted(combined_df['generation_length'].unique()):
                    subset = combined_df[combined_df['generation_length'] == gen_length]
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
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "distribution_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_scaling(self, summary_df: pd.DataFrame):
        """Plot performance scaling metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Estimated throughput
        if 'round_trip_time_mean' in summary_df.columns:
            throughput = 1.0 / summary_df['round_trip_time_mean']
            ax1.plot(summary_df['generation_length'], throughput, 
                    'o-', linewidth=3, markersize=10, color='#1f77b4')
            ax1.set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Estimated Throughput (req/sec)', fontsize=12, fontweight='bold')
            ax1.set_title('Request Throughput vs Generation Length', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Right: Scaling trend
        if 'round_trip_time_mean' in summary_df.columns:
            x = summary_df['generation_length']
            y = summary_df['round_trip_time_mean'] * 1000
            
            # Plot data points
            ax2.scatter(x, y, s=100, alpha=0.7, color='#ff7f0e')
            
            # Add trend line
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), "r--", alpha=0.8, linewidth=2, 
                    label=f'Trend: {z[0]:.2f}ms/frame')
            
            ax2.set_xlabel('Generation Length (Frames)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Mean Round Trip Time (ms)', fontsize=12, fontweight='bold')
            ax2.set_title('Latency Scaling Trend', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "performance_scaling.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_analysis_report(self):
        """Generate analysis report."""
        print("📝 Generating analysis report...")
        
        if not self.summary_stats:
            return
        
        summary_df = pd.DataFrame(self.summary_stats)
        report_path = self.analysis_dir / "automated_benchmark_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Automated Generation Length Benchmark Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Benchmark Type:** Fully Automated Multi-Server\n\n")
            
            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- **Generation Lengths:** {self.config['generation_lengths']}\n")
            f.write(f"- **Requests per Length:** {self.config['requests_per_length']}\n")
            f.write(f"- **Total Requests:** {summary_df['num_requests'].sum()}\n")
            f.write(f"- **Checkpoint:** {self.config['checkpoint_path']}\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Optimal latency
            min_latency_idx = summary_df['round_trip_time_mean'].idxmin()
            optimal_gen_length = summary_df.iloc[min_latency_idx]['generation_length']
            min_latency = summary_df.iloc[min_latency_idx]['round_trip_time_mean'] * 1000
            f.write(f"- **Optimal Generation Length:** {optimal_gen_length} frames ({min_latency:.1f}ms mean RTT)\n")
            
            # Latency range
            min_rtt = summary_df['round_trip_time_mean'].min() * 1000
            max_rtt = summary_df['round_trip_time_mean'].max() * 1000
            f.write(f"- **Latency Range:** {min_rtt:.1f}ms - {max_rtt:.1f}ms\n")
            
            # Scaling
            if len(summary_df) > 1:
                gen_lengths = summary_df['generation_length'].values
                latencies = summary_df['round_trip_time_mean'].values * 1000
                slope = (latencies[-1] - latencies[0]) / (gen_lengths[-1] - gen_lengths[0])
                f.write(f"- **Latency Scaling:** {slope:.2f}ms per additional frame\n")
            
            # Most consistent
            min_var_idx = summary_df['round_trip_time_std'].idxmin()
            most_consistent = summary_df.iloc[min_var_idx]['generation_length']
            min_std = summary_df.iloc[min_var_idx]['round_trip_time_std'] * 1000
            f.write(f"- **Most Consistent:** {most_consistent} frames ({min_std:.1f}ms std dev)\n\n")
            
            f.write("## Results Files\n\n")
            f.write("- `detailed_results_all_generation_lengths.csv` - Individual request data\n")
            f.write("- `summary_statistics.csv` - Aggregated statistics\n")
            f.write("- `summary_table_clean.csv` - Clean summary table\n")
            f.write("- `plots/` - Visualization plots\n")
            f.write("- `raw_data/` - Individual CSV files per generation length\n")
            f.write("- `server_logs/` - Server log files\n")
        
        print(f"   📄 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fully automated generation length benchmark system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automatically starts multiple server instances with different
generation lengths and runs benchmarks concurrently. No manual intervention required.

Examples:
  %(prog)s --checkpoint_path ckpt/model.ckpt --generation_lengths 5,10,15,20,25,30
  %(prog)s --checkpoint_path ckpt/model.ckpt --generation_lengths 10,20,30 --requests_per_length 100
        """
    )
    
    # Required parameters
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint file")
    
    # Core parameters
    parser.add_argument("--generation_lengths", type=str, default="5,10,15,20,25,30",
                       help="Comma-separated generation lengths to test")
    parser.add_argument("--requests_per_length", type=int, default=50,
                       help="Number of requests per generation length")
    parser.add_argument("--output_dir", type=str, default="results/automated_gen_length_benchmark",
                       help="Output directory for all results")
    
    # Server configuration
    parser.add_argument("--model_max_seq_len_frames", type=int, default=96,
                       help="Model max sequence length in frames")
    parser.add_argument("--model_size", type=str, default="0.12B",
                       choices=['small', '0.12B', '0.25B', '0.5B'],
                       help="Model size")
    
    # Control options
    parser.add_argument("--generate_plots", action="store_true", default=True,
                       help="Generate visualization plots")
    parser.add_argument("--generate_report", action="store_true", default=True,
                       help="Generate analysis report")
    parser.add_argument("--benchmark_timeout", type=int, default=300,
                       help="Timeout for individual benchmarks (seconds)")
    
    args = parser.parse_args()
    
    # Parse generation lengths
    try:
        generation_lengths = [int(x.strip()) for x in args.generation_lengths.split(',')]
    except ValueError:
        print("❌ Invalid generation_lengths format. Use comma-separated integers.")
        return 1
    
    # Validate checkpoint path
    if not Path(args.checkpoint_path).exists():
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        return 1
    
    # Build configuration
    config = {
        'checkpoint_path': args.checkpoint_path,
        'generation_lengths': generation_lengths,
        'requests_per_length': args.requests_per_length,
        'output_dir': args.output_dir,
        'model_max_seq_len_frames': args.model_max_seq_len_frames,
        'model_size': args.model_size,
        'generate_plots': args.generate_plots,
        'generate_report': args.generate_report,
        'benchmark_timeout': args.benchmark_timeout,
    }
    
    print("🤖 Automated Generation Length Benchmark System")
    print("=" * 60)
    
    # Run the automated benchmark
    benchmark = AutomatedGenerationLengthBenchmark(config)
    success = benchmark.run_automated_benchmark_suite()
    
    if success:
        print(f"\n🎉 Automated benchmark completed successfully!")
        print(f"📁 All results saved to: {Path(args.output_dir).absolute()}")
        return 0
    else:
        print(f"\n❌ Automated benchmark failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())