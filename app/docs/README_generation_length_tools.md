# Generation Length Analysis Tools

This directory contains enhanced benchmarking tools for analyzing how generation length affects StreamMUSE system latency and performance.

## Overview

These tools work with the existing StreamMUSE server without requiring any modifications. They coordinate multiple benchmark runs with different `GENERATION_LENGTH_FRAMES` environment variable settings to systematically study performance characteristics.

## Tools

### 1. `generation_length_benchmark.py` - Automated Parameter Sweep
**For advanced users with automation capabilities**

Coordinates multiple benchmark runs across different generation lengths. Requires manual server restarts between tests.

```bash
# Basic usage
python app/generation_length_benchmark.py \
    --generation_lengths 5,10,15,20,25,30 \
    --requests_per_length 50 \
    --output_dir results/gen_length_study \
    --generate_plots

# With visualization and analysis
python app/generation_length_benchmark.py \
    --generation_lengths 10,15,20,25,30,35 \
    --requests_per_length 100 \
    --output_dir results/comprehensive_study \
    --generate_plots \
    --generate_report
```

### 2. `manual_generation_length_study.py` - Manual Testing Helper
**Recommended for most users**

Provides step-by-step guidance and tracks progress for manual testing.

```bash
# Generate testing instructions
python app/manual_generation_length_study.py instructions 5,10,15,20,25,30

# Check study status
python app/manual_generation_length_study.py status

# Run individual test (after starting server with correct GENERATION_LENGTH_FRAMES)
python app/manual_generation_length_study.py test 20

# Analyze completed results
python app/manual_generation_length_study.py analyze
```

### 3. `analyze_generation_length_results.py` - Results Analysis
**Standalone analysis tool**

Analyzes benchmark results and generates comprehensive visualizations.

```bash
# Analyze automated benchmark results
python app/analyze_generation_length_results.py results/gen_length_study

# Analyze manual CSV files in directory
python app/analyze_generation_length_results.py results/manual_benchmarks

# Analyze single combined CSV file
python app/analyze_generation_length_results.py combined_results.csv
```

## Recommended Workflow

### Option A: Manual Testing (Recommended)

1. **Generate instructions:**
   ```bash
   python app/manual_generation_length_study.py instructions 5,10,15,20,25,30
   ```

2. **Follow the generated instructions** in `results/manual_gen_length_study/testing_instructions.md`

3. **For each generation length:**
   - Stop current server
   - Start server: `GENERATION_LENGTH_FRAMES=X uvicorn app.server:app --host 0.0.0.0 --port 8000`
   - Run benchmark: `python app/benchmark.py --num_requests 50 --output_file results/manual_gen_length_study/gen_length_X.csv`

4. **Analyze results:**
   ```bash
   python app/analyze_generation_length_results.py results/manual_gen_length_study
   ```

### Option B: Semi-Automated Testing

1. **Start the coordinated benchmark:**
   ```bash
   python app/generation_length_benchmark.py --generation_lengths 5,10,15,20,25,30
   ```

2. **Follow prompts** to restart server between tests

3. **Results are automatically analyzed** if `--generate_plots` is used

## Output Files

### Raw Data
- `gen_length_X.csv` - Individual benchmark results for generation length X
- `gen_length_X.json` - Complete response data for generation length X

### Analysis Results
- `detailed_results_all_generation_lengths.csv` - Combined detailed data
- `summary_statistics.csv` - Aggregated statistics per generation length
- `summary_table.csv` - Clean summary table for reporting

### Visualizations
- `latency_vs_generation_length.png` - Primary relationship plot with error bars
- `variability_analysis.png` - Standard deviation and coefficient of variation trends
- `distribution_comparison.png` - Box plots comparing distributions
- `component_breakdown.png` - Stacked bar chart of latency components
- `performance_metrics.png` - Throughput and generation efficiency
- `detailed_distributions.png` - Histogram grid for each generation length
- `correlation_analysis.png` - Correlation matrix and scatter plots

### Reports
- `benchmark_analysis_report.md` - Automated summary with key findings
- `testing_instructions.md` - Step-by-step manual testing guide

## Key Metrics Analyzed

### Primary Latency Metrics
- **Round Trip Time** - Total client request to response time
- **Server Processing Duration** - Time spent on server-side processing  
- **Inference Duration** - Pure model inference time
- **Network Latency** - Time spent on network communication

### Variability Metrics
- **Standard Deviation** - Absolute variability for each metric
- **Coefficient of Variation** - Relative variability (std/mean)
- **Percentiles** - 95th and 99th percentile analysis

### Performance Metrics
- **Throughput** - Estimated requests per second
- **Generation Efficiency** - Notes generated per second
- **Scaling Behavior** - Latency increase per frame

## Analysis Questions Answered

1. **How does generation length affect average latency?**
   - Linear relationship analysis
   - Optimal generation length identification
   - Performance scaling characterization

2. **How does generation length affect latency consistency?**
   - Variability trends across generation lengths
   - Most consistent configuration identification
   - Distribution shape analysis

3. **What are the latency component breakdowns?**
   - Preprocessing, inference, postprocessing, network contributions
   - Component scaling with generation length
   - Bottleneck identification

4. **What is the performance vs quality tradeoff?**
   - Throughput analysis
   - Generation efficiency metrics
   - Optimal operating point recommendations

## Requirements

### Dependencies
All tools use existing project dependencies plus:
- `matplotlib` - For plotting (already in pyproject.toml)
- `seaborn` - For enhanced visualizations
- `pandas` - For data manipulation
- `numpy` - For numerical operations

### Server Setup
- StreamMUSE server must support `GENERATION_LENGTH_FRAMES` environment variable
- Server must be restartable between tests
- `/clear_history` endpoint must be available for state reset

## Troubleshooting

### Common Issues

**"No CSV files found"**
- Ensure benchmark.py completed successfully
- Check file paths and naming conventions
- Verify generation length extraction from filenames

**"Cannot connect to server"**
- Verify server is running on specified port
- Check server URL configuration
- Ensure GENERATION_LENGTH_FRAMES is set correctly

**"Analysis failed"**
- Check that CSV files contain required columns
- Verify data is not empty or corrupted
- Ensure sufficient data for statistical analysis

### Data Validation
- Each generation length should have at least 20 successful requests
- CSV files must contain timing columns from benchmark.py
- Generation lengths should span a meaningful range (e.g., 5-30 frames)

## Example Results Interpretation

After running the analysis, you'll get insights like:

- **Optimal Generation Length**: "20 frames provides best latency (245ms) vs quality tradeoff"
- **Scaling Behavior**: "Latency increases 8.5ms per additional frame"
- **Consistency**: "15 frames shows most consistent performance (12ms std dev)"
- **Bottlenecks**: "Inference time dominates latency (80% of processing time)"

These insights help optimize real-time performance for your specific use case and hardware configuration.