# CADET Simplified

A simplified, Excel-based interface for CADET-Process chromatography simulations.

## Features

- **Excel-based configuration**: Define experiments in familiar Excel templates
- **Dynamic parameter introspection**: Automatically extracts parameters from CADET-Process models
- **Persistent storage**: Save and reload experiment results
- **Analysis tools**: Built-in chromatogram overlay and comparison
- **GUI and programmatic interfaces**: Use via Panel web app or directly in Python/Jupyter

## Installation

### Prerequisites

1. **CADET solver**: Download from [CADET releases](https://github.com/cadet/CADET/releases) and note the installation path
2. **Python 3.10+**

### From Local Repository (pip)

```bash
# Clone or copy the repository
cd cadet_simplified

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .

# Or for development (editable mode):
pip install -e .
```

### From GitHub

```bash
pip install git+https://github.com/MoritzImendoerffer/cadetgui.git
```

## Quick Start

### GUI Mode

```bash
# Run the Panel web application
python dev_gui.py

# Or with panel serve (supports auto-reload):
panel serve dev_gui.py --show --autoreload
```

The GUI guides you through:
1. **Configure**: Select operation mode, column model, binding model, components
2. **Download Template**: Get an Excel template with all parameters
3. **Upload**: Fill in the template and upload it
4. **Simulate**: Run simulations (set CADET path if not auto-detected)
5. **Saved**: Browse saved experiment sets
6. **Analysis**: Analyze and compare results

## Programmatic Usage (Jupyter / Scripts)

### Complete Workflow Example

```python
from pathlib import Path
from cadet_simplified import (
    # Template generation
    ExcelTemplateGenerator,
    # Parsing
    ExcelParser,
    # Operation modes
    get_operation_mode,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
    OPERATION_MODES,
    # Simulation
    SimulationRunner,
    # Storage
    FileResultsStorage,
    # Analysis
    AnalysisView,
    get_analysis,
)

# Print available models
print("Column models:", list(SUPPORTED_COLUMN_MODELS.keys()))
print("Binding models:", list(SUPPORTED_BINDING_MODELS.keys()))
print("Operation modes:", list(OPERATION_MODES.keys()))
```

### Step 1: Generate Excel Template

```python
# Create template generator
generator = ExcelTemplateGenerator(
    operation_mode="LWE_concentration_based",
    column_model="LumpedRateModelWithoutPores",
    binding_model="StericMassAction",
    n_components=3,
    component_names=["Salt", "Product", "Impurity1"],
)

# Save template
generator.save("my_template.xlsx")
print("Template saved. Fill in your experiment parameters.")
```

### Step 2: Parse Filled Template

```python
# Parse the filled template
parser = ExcelParser()
result = parser.parse("my_template_filled.xlsx")

if not result.success:
    print("Parse errors:", result.errors)
else:
    print(f"Parsed {len(result.experiments)} experiments")
    print(f"Column model: {result.column_binding.column_model}")
    print(f"Binding model: {result.column_binding.binding_model}")
```

### Step 3: Create Processes and Validate

```python
# Get operation mode
mode = get_operation_mode("LWE_concentration_based")

# Create CADET processes from parsed configs
processes = []
for exp in result.experiments:
    process = mode.create_process(exp, result.column_binding)
    processes.append(process)
    
# Validate configurations
for process in processes:
    is_valid = process.check_config()
    print(f"{process.name}: {'Valid' if is_valid else 'Invalid'}")
```

### Step 4: Run Simulations

```python
# Initialize runner (specify CADET path if not auto-detected)
runner = SimulationRunner(cadet_path="/path/to/cadet/bin")

# Option A: Run sequentially
results = []
for process in processes:
    sim_result = runner.run(process)
    results.append(sim_result)
    print(f"{sim_result.experiment_name}: {'Success' if sim_result.success else 'Failed'}")

# Option B: Run in parallel
results = runner.run_batch(processes, n_cores=4)
```

### Step 5: Save to Storage

```python
# Initialize storage
storage = FileResultsStorage("./my_experiments")

# Save experiment set (includes automatic Excel export)
set_id = storage.save_experiment_set(
    name="IEX_Screening_Study",
    operation_mode="LWE_concentration_based",
    experiments=result.experiments,
    column_binding=result.column_binding,
    results=results,
)
print(f"Saved with ID: {set_id}")

# This creates:
# ./my_experiments/{set_id}/
# ├── config.json                    # Metadata and configs
# ├── IEX_Screening_Study.xlsx       # Excel export with chromatograms
# ├── chromatograms/*.parquet        # Interpolated chromatograms
# ├── results/*.pkl                  # Pickled simulation results
# └── h5/*.h5                        # CADET H5 files
```

### Step 6: Load and Analyze

```python
# List available experiments
df = storage.list_experiments(limit=50)
print(df[["experiment_set_name", "experiment_name", "has_results"]])

# Load specific experiments
loaded = storage.load_results_by_selection([
    (set_id, "experiment_1"),
    (set_id, "experiment_2"),
], n_workers=4)

# Or load all from a set
loaded = storage.load_results(set_id)

# Access chromatogram data
for exp in loaded:
    print(f"\n{exp.experiment_name}:")
    print(f"  Success: {exp.result.success}")
    print(f"  Runtime: {exp.result.runtime_seconds:.2f}s")
    if exp.chromatogram_df is not None:
        print(f"  Chromatogram shape: {exp.chromatogram_df.shape}")
```

### Step 7: Visualization

```python
# Quick matplotlib plot
import matplotlib.pyplot as plt

for exp in loaded:
    if exp.chromatogram_df is None:
        continue
    
    df = exp.chromatogram_df
    time_min = df["time"] / 60  # Convert to minutes
    
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        if col != "time":
            plt.plot(time_min, df[col], label=col)
    
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (mM)")
    plt.title(exp.experiment_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
```

### Using the Analysis Module (Panel/hvplot)

```python
# For interactive plots (requires Panel)
from cadet_simplified import AnalysisView, get_analysis

view = AnalysisView()
analysis = get_analysis("simple")  # or "detailed"
analysis.run(loaded, view)

# Display in Jupyter
view.view()
```

## Configuration Reference

### Experiment Parameters (LWE Mode)

| Parameter | Unit | Description |
|-----------|------|-------------|
| `flow_rate_mL_min` | mL/min | Volumetric flow rate |
| `equilibration_cv` | CV | Equilibration volume |
| `load_cv` | CV | Sample loading volume |
| `wash_cv` | CV | Post-load wash volume |
| `elution_cv` | CV | Gradient elution volume |
| `strip_cv` | CV | High salt strip volume |
| `load_salt_mM` | mM | Salt during loading |
| `wash_salt_mM` | mM | Salt during wash |
| `gradient_start_mM` | mM | Gradient start concentration |
| `gradient_end_mM` | mM | Gradient end concentration |
| `strip_salt_mM` | mM | Salt during strip |
| `component_N_load_concentration` | g/L | Load concentration for component N |

### Storage Directory Structure

```
experiments/
├── _pending/                    # Temporary H5 files during simulation
└── {experiment_set_id}/
    ├── config.json              # Metadata + all configurations
    ├── {set_name}.xlsx          # Excel export with chromatograms
    ├── chromatograms/
    │   └── {exp_name}.parquet   # Interpolated chromatograms
    ├── results/
    │   └── {exp_name}.pkl       # Full simulation results (pickled)
    └── h5/
        └── {exp_name}.h5        # CADET native H5 files
```

## Troubleshooting

### CADET Path Not Found

The app tries to detect CADET automatically. If detection fails, set the `CADET_PATH` environment variable:

```bash
# Linux/Mac - add to ~/.bashrc or ~/.zshrc
export CADET_PATH="/path/to/cadet/bin"

# Windows - in Command Prompt
set CADET_PATH=C:\path\to\cadet\bin

# Windows - PowerShell
$env:CADET_PATH = "C:\path\to\cadet\bin"
```

Then restart your terminal/application.

Alternatively, pass the path directly in code:

```python
# Option 1: Pass to SimulationRunner
runner = SimulationRunner(cadet_path="/path/to/cadet/bin")

# Option 2: Pass to the app
app = SimplifiedCADETApp(cadet_path="/path/to/cadet/bin")
```

Or set it in `dev_gui.py`:
```python
CADET_PATH = "/path/to/cadet/bin"  # Instead of None
```

### Import Errors

Ensure CADET-Process is installed:

```bash
pip install cadet-process
```

### Memory Issues with Large Simulations

Reduce interpolation points:

```python
storage = FileResultsStorage("./experiments", n_interpolation_points=200)
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request on GitHub.
