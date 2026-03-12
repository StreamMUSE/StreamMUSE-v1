# Enhanced Benchmark System Design

## Overview

This document outlines the design for an enhanced benchmarking system that analyzes the relationship between generation length and system latency in the StreamMUSE real-time music generation system.

## Research Questions

1. **Primary Relationship**: How does generation length affect round trip latency?
2. **Variability Analysis**: How does generation length affect latency consistency (standard deviation)?
3. **Distribution Characteristics**: What does the latency distribution look like for each generation length?
4. **Scaling Behavior**: Is the relationship linear, or are there performance breakpoints?

## Design Goals

### Core Objectives
- Systematically test multiple generation lengths
- Collect statistically significant data for each generation length
- Analyze both central tendency (mean) and variability (std deviation)
- Visualize relationships and distributions
- Export comprehensive data for further analysis

### Performance Metrics to Analyze
- **Round Trip Latency**: Total client request to response time
- **Server Processing Duration**: Time spent on server-side processing
- **Inference Duration**: Pure model inference time
- **Network Latency**: Time spent on network communication
- **Latency Variability**: Standard deviation of each metric

## System Architecture

### Enhanced Benchmark Flow
```
1. Parameter Sweep Setup
   ├── Define generation length range (5, 10, 15, 20, 25, 30 frames)
   ├── Set requests per generation length (50-100 for statistical significance)
   └── Configure server communication

2. Data Collection Loop
   ├── For each generation length:
   │   ├── Configure server with generation length
   │   ├── Execute N benchmark requests
   │   ├── Collect detailed timing data
   │   └── Calculate statistics
   
3. Data Export & Analysis
   ├── Export detailed CSV (individual requests)
   ├── Export summary CSV (aggregated statistics)
   ├── Generate visualizations
   └── Save analysis report
```

### Server Integration Options
**Option A: Dynamic Configuration**
- Add generation length parameter to inference request
- Server dynamically adjusts generation length per request
- Requires minimal server modifications

**Option B: Environment Variable Approach**
- Run separate benchmark sessions with different `GENERATION_LENGTH_FRAMES`
- Requires server restart between generation lengths
- More controlled but less efficient

**Recommended**: Option A for flexibility and efficiency

## Data Structure Design

### Detailed Results CSV
Individual request-level data for comprehensive analysis:
```csv
generation_length,request_id,timestamp,round_trip_time,server_processing_duration,
inference_duration,preprocess_duration,postprocess_duration,total_network_latency,
num_generated_notes,server_request_arrival_time,server_response_output_time
```

### Summary Statistics CSV
Aggregated metrics for quick analysis:
```csv
generation_length,num_requests,num_successful_requests,
avg_round_trip_time,std_round_trip_time,min_round_trip_time,max_round_trip_time,
p95_round_trip_time,p99_round_trip_time,
avg_server_processing,std_server_processing,
avg_inference_time,std_inference_time,
avg_network_latency,std_network_latency,
avg_notes_generated,throughput_requests_per_sec
```

### JSON Metadata
Complete experimental configuration and results:
```json
{
  "experiment_metadata": {
    "timestamp": "2024-XX-XX XX:XX:XX",
    "server_url": "http://localhost:8000",
    "generation_lengths_tested": [5, 10, 15, 20, 25, 30],
    "requests_per_length": 100,
    "total_requests": 600
  },
  "generation_length_results": {
    "5": { "detailed_stats": {...}, "raw_data": [...] },
    "10": { "detailed_stats": {...}, "raw_data": [...] }
  }
}
```

## Visualization System

### Primary Analysis Plots

1. **Latency vs Generation Length**
   - Scatter plot with error bars (mean ± standard deviation)
   - Separate lines for different latency components
   - Linear regression trend lines

2. **Variability Analysis**
   - Line plot of standard deviation vs generation length
   - Shows how consistency changes with generation length

3. **Distribution Analysis**
   - Histogram grid: one subplot per generation length
   - Box plot comparison across generation lengths
   - Violin plots for distribution shape analysis

4. **Performance Scaling**
   - Throughput vs generation length
   - Latency components breakdown (stacked bar chart)
   - Performance efficiency metrics

### Statistical Analysis Features

**Distribution Characterization**
- Normality tests (Shapiro-Wilk, Anderson-Darling)
- Outlier detection and flagging
- Percentile analysis (95th, 99th percentiles)

**Relationship Analysis**
- Correlation matrices between all timing metrics
- Linear regression analysis with R² values
- ANOVA for comparing means across generation lengths

**Performance Insights**
- Identify optimal generation length for latency
- Detect performance bottlenecks
- Scaling behavior characterization

## Output Deliverables

### Data Files
- `detailed_results_YYYYMMDD_HHMMSS.csv`: Individual request data
- `summary_statistics_YYYYMMDD_HHMMSS.csv`: Aggregated metrics
- `experiment_data_YYYYMMDD_HHMMSS.json`: Complete experimental record

### Visualizations
- `latency_vs_generation_length.png`: Primary relationship plot
- `latency_variability_analysis.png`: Standard deviation analysis
- `distribution_analysis.png`: Histogram grid
- `performance_scaling.png`: Throughput and efficiency metrics

### Analysis Report
- `benchmark_analysis_report.md`: Automated summary with key findings
- Statistical test results and interpretations
- Performance recommendations

## Usage Interface

### Command Line Interface
```bash
# Basic generation length sweep
python app/enhanced_benchmark.py --output_dir results/gen_length_analysis

# Custom generation lengths
python app/enhanced_benchmark.py --generation_lengths 5,10,15,20,25,30 --requests_per_length 100

# Include visualization generation
python app/enhanced_benchmark.py --output_dir results/ --generate_plots --statistical_analysis
```

### Configuration Options
- `--server_url`: Target server endpoint
- `--generation_lengths`: Comma-separated list of generation lengths to test
- `--requests_per_length`: Number of requests per generation length
- `--output_dir`: Directory for results and plots
- `--generate_plots`: Enable automatic visualization generation
- `--statistical_analysis`: Include advanced statistical tests
- `--parallel_requests`: Enable concurrent request processing

## Success Criteria

1. **Data Quality**: Collect statistically significant data (>50 requests per generation length)
2. **Comprehensive Coverage**: Test meaningful range of generation lengths (5-30 frames)
3. **Clear Visualizations**: Generate publication-ready plots showing key relationships
4. **Actionable Insights**: Identify optimal generation length and performance characteristics
5. **Reproducibility**: Complete documentation and data export for result verification

## Future Extensions

- **Real-world Load Testing**: Multi-client concurrent benchmarking
- **Resource Utilization**: GPU/CPU usage correlation with generation length
- **Quality vs Performance**: Generated music quality metrics alongside latency
- **Dynamic Generation Length**: Adaptive generation length based on real-time performance