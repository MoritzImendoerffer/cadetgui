# CADET Simplified

A streamlined interface for CADET-Process chromatography simulations.

## Installation

```bash
# Clone or copy the package
pip install panel param pandas numpy holoviews hvplot openpyxl

# For simulations, you also need CADET-Process
pip install cadet-process
```

## Running the GUI

```bash
# Option 1: Using panel serve directly
panel serve cadet_simplified/app.py --show --autoreload

# Option 2: As a Python module
python -m cadet_simplified --port 5006 --show

# Option 3: Programmatically
python -c "from cadet_simplified.app import serve; serve(port=5006, show=True)"
```

## Key Changes in v0.2.0

This is a refactored version with simplified architecture:

1. **JSON-based configs** instead of runtime introspection
2. **Consolidated dataclasses** (removed duplicate `Stored*` classes)
3. **Reusable plotting functions** that work in notebooks and apps
4. **Simplified storage** (pickle + parquet, no H5)
5. **No abstract interfaces** with single implementations

## Structure

```
cadet_simplified/
├── __init__.py              # Main exports
├── __main__.py              # CLI entry point
├── app.py                   # Panel GUI application
├── core/                    # Core dataclasses
│   └── dataclasses.py       # ExperimentConfig, ColumnBindingConfig, ComponentDefinition
├── configs/                 # JSON parameter definitions
│   ├── loader.py            # Config loading functions
│   ├── binding_models/      # JSON configs for each binding model
│   └── column_models/       # JSON configs for each column model
├── operation_modes/         # Process definitions
│   ├── base.py              # BaseOperationMode
│   └── lwe.py               # Load-Wash-Elute mode
├── simulation/              # Simulation runner
│   └── runner.py            # SimulationRunner, SimulationResultWrapper
├── excel/                   # Excel I/O
│   ├── template_generator.py
│   └── parser.py
├── plotting/                # Reusable plot functions
│   └── chromatogram.py      # plot_chromatogram, interpolate_chromatogram, etc.
└── storage/                 # File-based storage
    └── file_storage.py      # FileStorage, LoadedExperiment
```

## Quick Start (Programmatic)

```python
from cadet_simplified import (
    ExcelTemplateGenerator,
    parse_excel,
    get_operation_mode,
    SimulationRunner,
    FileStorage,
    plot_chromatogram,
)

# 1. Generate template
generator = ExcelTemplateGenerator(
    operation_mode="LWE_concentration_based",
    column_model="LumpedRateModelWithPores",
    binding_model="StericMassAction",
    n_components=3,
)
generator.save("template.xlsx")

# 2. Parse filled template
result = parse_excel("filled_template.xlsx")

# 3. Run simulations
mode = get_operation_mode("LWE_concentration_based")
runner = SimulationRunner()

results = []
for exp in result.experiments:
    process = mode.create_process(exp, result.column_binding)
    sim_result = runner.run(process)
    results.append(sim_result)

# 4. Plot (works in notebook)
plot_chromatogram(results[0])

# 5. Save to storage
storage = FileStorage("./experiments")
set_id = storage.save_experiment_set(
    name="My Experiments",
    operation_mode="LWE_concentration_based",
    experiments=result.experiments,
    column_binding=result.column_binding,
    results=results,
)
```

## Plotting in Notebooks

```python
from cadet_simplified.plotting import (
    plot_chromatogram,
    plot_chromatogram_overlay,
    plot_chromatogram_from_df,
    interpolate_chromatogram,
)

# From simulation result
plot = plot_chromatogram(result)

# From cached DataFrame
df = interpolate_chromatogram(result, n_points=2000)
plot = plot_chromatogram_from_df(df)

# Overlay multiple
plot = plot_chromatogram_overlay([result1, result2], labels=["Exp 1", "Exp 2"])
```

## GUI Workflow

The GUI has 5 tabs:

1. **Configure**: Select operation mode, column model, binding model, and number of components
2. **Upload**: Upload filled Excel template and validate
3. **Simulate**: Run simulations with progress tracking
4. **Saved**: Browse saved experiments and select for analysis
5. **Analysis**: Create chromatogram overlays or individual plots

## Adding New Models

To add a new binding or column model:

1. Create a JSON file in `configs/binding_models/` or `configs/column_models/`
2. Follow the existing format:

```json
{
  "name": "ModelName",
  "cadet_class": "CADETProcess.processModel.ModelName",
  "description": "Human-readable description",
  "scalar_parameters": [
    {
      "name": "param_name",
      "type": "float",
      "default": 1.0,
      "unit": "m",
      "bounds": [0, null],
      "description": "Parameter description"
    }
  ],
  "component_parameters": [
    // Same structure, but these have one value per component
  ]
}
```

## Adding New Operation Modes

1. Create a new file in `operation_modes/` (e.g., `gradient.py`)
2. Inherit from `BaseOperationMode`
3. Implement `get_experiment_parameters()`, `get_component_experiment_parameters()`, and `create_process()`
4. Register in `operation_modes/__init__.py`

## What Was Removed (from original)

| Removed | Replaced By |
|---------|-------------|
| `parameter_introspection.py` (~400 lines) | JSON configs (~50 lines each) |
| `ResultsStorageInterface` abstract class | Direct `FileStorage` implementation |
| `StoredExperiment`, `StoredColumnBinding` | Use core dataclasses directly |
| `ExperimentStore` legacy class | Deleted |
| `ModelRegistry` class | Simple dicts |
| H5 file storage | Pickle + Parquet only |
| `BaseAnalysis` ABC + registry | Direct plotting functions |
