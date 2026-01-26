# JupyterLab Setup Guide

How to set up `cadet_simplified` in a new environment and make it available as a Jupyter kernel.

---

## Option A: Conda Setup (Recommended)

Conda can install both Python packages and the CADET solver binary.

```bash
# 1. Create and activate environment
conda create -n cadet python=3.10 -y
conda activate cadet

# 2. Install CADET solver
conda install -c conda-forge cadet -y

# 3. Install the package (directly from github - recommended)
python -m pip install "git+https://github.com/MoritzImendoerffer/cadetgui.git"

# 3a. Install the package (from local repo - alternative)
cd /path/to/cadet_simplified
pip install -e .

# 4. Register as Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name cadet_simplified --display-name "Python (CADET Simplified)"

# 5. Set CADET path (add to ~/.bashrc for persistence)
grep -qxF 'export CADET_PATH=/opt/CADET/latest' ~/.bashrc || echo 'export CADET_PATH=/opt/CADET/latest' >> ~/.bashrc

# 5a. optionally, export the environment variable e.g. in a cell of a notebook:

import os
os.environ["CADET_PATH"] = "/opt/CADET/latest"

#  5b. on startup of a juypter notebook
#TODO: add command here
```

---

## Option B: Pip + venv Setup

Use this if you don't have conda or prefer pure pip.

```bash
# 1. Create and activate venv
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 2. Install the package
cd /path/to/cadet_simplified
pip install -e .

# 3. Register as Jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name cadet_simplified --display-name "Python (CADET Simplified)"

# 4. Set CADET path (add to ~/.bashrc for persistence)
grep -qxF 'export CADET_PATH=/opt/CADET/latest' ~/.bashrc || echo 'export CADET_PATH=/opt/CADET/latest' >> ~/.bashrc

# 4a. optionally, export the environment variable e.g. in a cell of a notebook:

import os
os.environ["CADET_PATH"] = "/opt/CADET/latest"

# 4b. on startup of a juypter notebook

KERNEL_DIR="$(jupyter --data-dir)/kernels/cadet_simplified"
python - <<'PY'
import json, os
p = os.path.join(os.environ["KERNEL_DIR"], "kernel.json")
with open(p, "r", encoding="utf-8") as f:
    data = json.load(f)
data.setdefault("env", {})["CADET_PATH"] = "/opt/CADET/latest"
with open(p, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print("Updated:", p)
PY
```

---

## Quick Reference

| Step | Conda | Pip + venv |
|------|-------|------------|
| Create env | `conda create -n cadet python=3.10` | `python3 -m venv .venv` |
| Activate | `conda activate cadet` | `source .venv/bin/activate` |
| Install CADET | `conda install -c conda-forge cadet` | Already installed on system |
| Install package | `pip install -e .` | `pip install -e .` |
| Register kernel | `python -m ipykernel install --user --name cadet` | Same |

---

## Starting the GUI from Jupyter

### Option 1: Open in new browser tab (recommended)

```python
from cadet_simplified.app import SimplifiedCADETApp

app = SimplifiedCADETApp()
app.view().show()  # Opens in new browser tab
```

### Option 2: Embed in notebook

```python
from cadet_simplified.app import SimplifiedCADETApp

app = SimplifiedCADETApp()
app.view()  # Displays inline (may have limited interactivity)
```

### Option 3: Run as standalone server

```python
import panel as pn
from cadet_simplified.app import SimplifiedCADETApp

app = SimplifiedCADETApp()
pn.serve(app.view(), port=5006, show=True)
```

Note: Option 3 blocks the notebook cell until you stop the server (Ctrl+C or interrupt kernel).

---

## Managing Kernels

### List installed kernels

```bash
jupyter kernelspec list
```

### Remove a kernel

```bash
jupyter kernelspec uninstall cadet
```

### Update kernel after environment changes

If you update packages in the conda environment, the kernel automatically uses the updated packages (no re-registration needed).

---

## Troubleshooting

### Kernel not appearing in JupyterLab

1. Restart JupyterLab completely
2. Check kernel is registered: `jupyter kernelspec list`
3. Re-register if needed: `python -m ipykernel install --user --name cadet --display-name "Python (CADET)"`

### "No module named cadet_simplified"

The package isn't installed in the kernel's environment:

```bash
conda activate cadet
pip install -e /path/to/cadet_simplified
```

### CADET not found

```python
# Check if CADET_PATH is set
import os
print(os.environ.get("CADET_PATH"))

# Check if binary exists
import shutil
print(shutil.which("cadet-cli"))
```

If not found, install CADET or set the path manually:

```bash
conda activate cadet
conda install -c conda-forge cadet -y
export CADET_PATH="$CONDA_PREFIX/bin"
```

### Import errors for dependencies

```bash
conda activate cadet
pip install -r requirements.txt
```

---

## Example Notebook Workflow

```python
# Cell 1: Imports
from cadet_simplified.excel import ExcelTemplateGenerator, ExcelParser
from cadet_simplified.operation_modes import get_operation_mode
from cadet_simplified.simulation import SimulationRunner
from cadet_simplified.storage import FileResultsStorage

# Cell 2: Generate template
gen = ExcelTemplateGenerator(
    operation_mode="lwe",
    column_model="LRMP",
    binding_model="Langmuir",
    n_components=2,
)
gen.save("my_experiments.xlsx")
# Fill this file in Excel, then continue...

# Cell 3: Parse filled template
parser = ExcelParser()
result = parser.parse("my_experiments_filled.xlsx")
print(f"Loaded {len(result.experiments)} experiments")

# Cell 4: Create processes
mode = get_operation_mode("lwe")
processes = [
    mode.create_process(exp, result.column_binding)
    for exp in result.experiments
]

# Cell 5: Run simulations
runner = SimulationRunner()
results = runner.run_batch(processes, n_cores=4)

# Cell 6: Save results
storage = FileResultsStorage()
set_id = storage.save_experiment_set(
    set_name="my_first_run",
    experiments=result.experiments,
    column_binding=result.column_binding,
    results=results,
    operation_mode="lwe",
)
print(f"Saved as: {set_id}")
```
