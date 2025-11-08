# StreamMUSE App Directory Structure

This directory has been reorganized for better maintainability and clarity.

## Directory Structure

```
app/
├── README_DIRECTORY_STRUCTURE.md    # This file
├── client.py                        # Main client application
├── server.py                        # FastAPI server
├── midi_input_script.py             # MIDI input handling script
├── requirements_client.txt          # Client dependencies
├── 
├── analysis/                        # Analysis and research scripts
│   ├── analyze_generation_length_results.py
│   ├── combined_constraint_analysis.py
│   ├── constraint_specific_analysis.py
│   ├── detailed_80th_percentile_analysis.py
│   ├── detailed_constraint_analysis.py
│   ├── detailed_corrected_analysis.py
│   ├── final_constraint_analysis.py
│   └── parameter_constraint_analysis.py
│
├── benchmarking/                    # Benchmarking and performance testing
│   ├── benchmark.py                 # Core benchmark script
│   ├── bulk_benchmark.py           # Bulk/grid search benchmarking
│   ├── automated_generation_length_benchmark.py
│   ├── generation_length_benchmark.py
│   ├── manual_generation_length_study.py
│   └── benchmark.ipynb             # Jupyter notebook for analysis
│
├── debug/                          # Debug utilities and test files
│   ├── debug_colors.py
│   ├── debug_combined_colors.py
│   ├── debug_constraints.py
│   ├── test_melody_history.json
│   └── tester.py
│
├── docs/                           # Documentation
│   ├── README.md
│   └── README_generation_length_tools.md
│
├── archive/                        # Archived files and old logs
│   └── logs/                       # Old session logs
│       ├── session_20251028-092029/
│       ├── session_20251028-092206/
│       └── ...
│
├── inference_engines/              # ML inference implementations
│   ├── transformer_engine.py
│   ├── transformer_engine_midi_input.py
│   └── transformer_engine_stanley.py
│
├── input_handlers/                 # Input handling modules
│   └── input_handler.py
│
└── output_handlers/                # Output handling modules
    ├── audio_output.py
    ├── cli_output.py
    ├── json_log_handler.py
    ├── log_output.py
    ├── midi_file_handler.py
    └── webapp_output.py
```

## Usage Guidelines

### Core Application Files
- **`client.py`** - Main client for real-time music generation
- **`server.py`** - FastAPI server hosting the ML model
- **`midi_input_script.py`** - Standalone MIDI input handling

### Analysis Scripts (`analysis/`)
- Use these for post-experiment analysis and research
- Most recent and maintained scripts:
  - `analyze_generation_length_results.py` - Primary analysis tool
  - `combined_constraint_analysis.py` - Parameter constraint visualization
  - `parameter_constraint_analysis.py` - Constraint analysis with percentiles

### Benchmarking Scripts (`benchmarking/`)
- **`benchmark.py`** - Core single-experiment benchmark
- **`bulk_benchmark.py`** - Grid search and bulk experiments
- Other files are legacy/specialized benchmarking tools

### Development and Debugging (`debug/`)
- Development utilities and test files
- Safe to ignore for normal usage

### Documentation (`docs/`)
- User guides and technical documentation
- Reference material for the system

### Archived Content (`archive/`)
- Old logs and deprecated files
- Can be cleaned periodically

## Import Path Updates

If you have scripts that import from these moved files, update the import paths:

```python
# Old imports
from benchmark import run_benchmark
from analyze_generation_length_results import GenerationLengthAnalyzer

# New imports  
from benchmarking.benchmark import run_benchmark
from analysis.analyze_generation_length_results import GenerationLengthAnalyzer
```

## Maintenance

- Keep core files (client.py, server.py) in the root
- Add new analysis scripts to `analysis/`
- Add new benchmarking tools to `benchmarking/`
- Move old logs to `archive/logs/` periodically
- Update documentation in `docs/` when making changes