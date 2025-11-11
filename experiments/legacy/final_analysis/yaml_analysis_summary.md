# YAML-Based StreamMUSE Analysis Report

Generated from configuration: `analysis_config.yaml`

## Experiment Configuration

### Local Direct (PC-PC)
- **Source**: experiments/local_gl_test
- **Description**: Direct connection between PC client and PC server
- **Formula Type**: quadratic
- **Formula**: RT = 0.15×GL² + 20.56×GL + 31.5
- **Color**: #2E7D32
- **Samples**: 8000

### Local Network (PC-Mac)
- **Source**: experiments/local_server_gl_test
- **Description**: Local network connection between PC client and Mac server
- **Formula Type**: quadratic
- **Formula**: RT = 1.42×GL² + 5.05×GL + 98.7
- **Color**: #1976D2
- **Samples**: 8000

### Remote Cloud (Hyperstack-PC)
- **Source**: experiments/remote_gl_test_new
- **Description**: Remote connection between PC client and Hyperstack server
- **Formula Type**: quadratic
- **Formula**: RT = 0.00×GL² + 18.61×GL + 164.7
- **Color**: #D32F2F
- **Samples**: 8000

## Analysis Settings

- **Anomaly Filtering**: 0% per tail
- **Output Directory**: experiments/final_analysis
- **Individual Analyses**: True

## Visualizations Generated

- `experiment_fitting_analysis.png`: Box plots with formula-based fitted lines and ±5% confidence bands

