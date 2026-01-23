# CADET Simplified

A simplified Excel-based interface for CADET-Process chromatography simulations.

## Overview

This package provides a streamlined workflow for chromatography simulations:

1. **Configure**: Select operation mode, column model, binding model, and define components
2. **Template**: Download an Excel template with all parameters
3. **Fill**: Fill in experiment parameters (lab-friendly units like CV, mM)
4. **Upload**: Upload the filled template for validation
5. **Simulate**: Run simulations and view results

## Installation

```bash
pip install -r requirements.txt
```

For running simulations, you also need CADET-Process:
```bash
pip install cadet-process
```

## Usage

### Launch the GUI

```bash
python run_app.py --port 5007
```

Or from Python:
```python
from cadet_simplified.app import serve
serve(port=5007)
```

### Programmatic Usage

```python
from cadet_simplified import (
    get_lwe_mode,
    ExcelTemplateGenerator,
    parse_excel,
    SimulationRunner,
)

# 1. Generate a template
mode = get_lwe_mode()
generator = ExcelTemplateGenerator(
    operation_mode=mode,
    column_model="LumpedRateModelWithPores",
    binding_model="StericMassAction",
    n_components=4,
    component_names=["Salt", "Product", "Impurity1", "Impurity2"],
)
generator.save("my_template.xlsx")

# 2. After filling the template, parse it
result = parse_excel("my_filled_template.xlsx")

# 3. Create and run simulations
if result.success:
    for exp in result.experiments:
        process = mode.create_process(exp, result.column_binding)
        
        runner = SimulationRunner()
        sim_result = runner.run(process, exp.name)
        
        if sim_result.success:
            print(f"Simulation {exp.name} completed in {sim_result.runtime_seconds:.1f}s")
```

## Package Structure

```
cadet_simplified/
├── __init__.py              # Main package exports
├── app.py                   # Panel GUI application
├── run_app.py               # Launcher script
├── operation_modes/
│   ├── base.py              # BaseOperationMode ABC
│   └── lwe.py               # LWE concentration-based mode
├── excel/
│   ├── template_generator.py # Generate Excel templates
│   └── parser.py            # Parse filled Excel files
├── storage/
│   └── experiment_store.py  # JSON file-based storage
└── simulation/
    └── runner.py            # Run simulations
```

## Excel Template Structure

### Sheet 1: "Experiments"

One row per experiment with lab-friendly parameters:
- `experiment_name`: Unique name for the experiment
- `flow_rate_cv_min`: Flow rate in column volumes per minute
- `load_cv`, `wash_cv`, `elution_cv`: Phase volumes in CV
- `gradient_start_mm`, `gradient_end_mm`: Salt concentrations in mM
- `component_N_load_concentration`: Load concentration for each component

### Sheet 2: "Column_Binding"

Shared parameters for all experiments:
- Model selection (column_model, binding_model, n_components)
- Component names
- Column parameters (length, diameter, porosity, etc.)
- Binding parameters (capacity, kinetic parameters)
- Per-component parameters (diffusion coefficients, binding rates)

## Operation Modes

Currently implemented:
- **LWE_concentration_based**: Load-Wash-Elute with linear salt gradient

### Adding New Operation Modes

Subclass `BaseOperationMode` and implement:
- `get_experiment_parameters()`: Lab-friendly experiment parameters
- `get_component_experiment_parameters()`: Per-component experiment parameters
- `create_process()`: Convert config to CADET Process

## Notes

- All durations/volumes are in column volumes (CV) for lab convenience
- Flow rate is in CV/min
- Salt concentrations are in mM
- The first component should always be salt for ion exchange
- Process validation uses CADET-Process's built-in `check_config()`
