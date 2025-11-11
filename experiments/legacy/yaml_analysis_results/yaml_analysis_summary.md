# YAML-Based StreamMUSE Analysis Report

Generated from configuration: `analysis_config.yaml`

## Experiment Configuration

### Local Direct (PC-PC)
- **Source**: experiments/local_gl_test
- **Description**: Direct connection between PC client and PC server
- **Formula Type**: linear
- **Formula**: RT = 24.50×GL + 30.0
- **Color**: #2E7D32
- **Samples**: 7200

### Local Network (PC-Mac)
- **Source**: experiments/local_server_gl_test
- **Description**: Local network connection between PC client and Mac server
- **Formula Type**: quadratic
- **Formula**: RT = 0.80×GL² + 15.20×GL + 75.0
- **Color**: #1976D2
- **Samples**: 7200

### Remote Cloud (PC-Hyperstack)
- **Source**: experiments/remote_gl_test_new
- **Description**: Remote connection between PC client and Hyperstack server
- **Formula Type**: linear
- **Formula**: RT = 18.50×GL + 165.0
- **Color**: #D32F2F
- **Samples**: 7200

## Analysis Settings

- **Anomaly Filtering**: 5.0% per tail
- **Output Directory**: experiments/yaml_analysis_results
- **Individual Analyses**: True

## Visualizations Generated

- `experiment_fitting_analysis.png`: Box plots with formula-based fitted lines and ±5% confidence bands

