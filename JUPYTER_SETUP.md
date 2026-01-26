# JupyterLab Setup Guide

How to set up `cadet_simplified` in a new environment and make it available as a Jupyter kernel.

---

## Quick Start (Conda - Recommended)

For the fastest path to a working setup:

```bash
conda create -n cadet_simplified python=3.10 -y
```

```bash
conda activate cadet_simplified
```

```bash
conda install -c conda-forge cadet -y
```

```bash
python -m pip install "git+https://github.com/MoritzImendoerffer/cadetgui.git"
```

```bash
pip install ipykernel
```

```bash
python -m ipykernel install --user --name cadet_simplified --display-name "Python (CADET Simplified)"
```

```bash
python -c "import json,subprocess,pathlib; p=pathlib.Path(subprocess.run(['jupyter','--data-dir'],capture_output=True,text=True).stdout.strip())/'kernels'/'cadet_simplified'/'kernel.json'; d=json.loads(p.read_text()); d.setdefault('env',{})['CADET_PATH']='/opt/CADET/latest'; p.write_text(json.dumps(d,indent=2)+'\n')"
```

Done! Start JupyterLab and select the "Python (CADET Simplified)" kernel.

---

## Verification

After setup, verify CADET is configured correctly in a notebook:

```python
# Check CADET_PATH environment variable
import os
cadet_path = os.environ.get("CADET_PATH")
print(f"CADET_PATH: {cadet_path}")

# Check if cadet-cli binary exists
import shutil
cadet_binary = shutil.which(cadet_path + "/bin/cadet-cli")
print(f"CADET binary found at: {cadet_binary}")

# Verify it's executable (optional - more thorough check)
if cadet_binary:
    import subprocess
    try:
        result = subprocess.run([cadet_binary, "--version"], 
                              capture_output=True, text=True, timeout=5)
        print(f"CADET version check: {result.stdout.strip()}")
    except Exception as e:
        print(f"Error running CADET: {e}")
else:
    print("WARNING: CADET binary not found!")
```

Expected output:
- `CADET_PATH` should show `/opt/CADET/latest` (or your configured path)
- `CADET binary found at` should show the full path to `cadet-cli`
- `CADET version check` should display version information

If any checks fail, see the [Troubleshooting](#troubleshooting) section below.

---

## Option A: Conda Setup (Recommended)

Conda can install both Python packages and the CADET solver binary.

### 1. Environment Setup

Create environment with Python 3.10:

```bash
conda create -n cadet_simplified python=3.10 -y
```

Activate the environment:

```bash
conda activate cadet_simplified
```

### 2. Install CADET Solver

```bash
conda install -c conda-forge cadet -y
```

### 3. Install cadet_simplified Package

**Choose one:**

**Option 3a: From GitHub (recommended)**

```bash
python -m pip install "git+https://github.com/MoritzImendoerffer/cadetgui.git"
```

**Option 3b: From local repository**

```bash
cd /path/to/cadet_simplified
```

```bash
pip install -e .
```

### 4. Register Jupyter Kernel

Install ipykernel:

```bash
pip install ipykernel
```

Register the kernel:

```bash
python -m ipykernel install --user --name cadet_simplified --display-name "Python (CADET Simplified)"
```

### 5. Configure CADET Path

**Choose one:**

**Option 5a: Set in kernel config (recommended - automatic on kernel startup)**

```bash
python -c "import json,subprocess,pathlib; p=pathlib.Path(subprocess.run(['jupyter','--data-dir'],capture_output=True,text=True).stdout.strip())/'kernels'/'cadet_simplified'/'kernel.json'; d=json.loads(p.read_text()); d.setdefault('env',{})['CADET_PATH']='/opt/CADET/latest'; p.write_text(json.dumps(d,indent=2)+'\n')"
```

**Option 5b: Add to ~/.bashrc (for all terminal sessions)**

```bash
grep -qxF 'export CADET_PATH=/opt/CADET/latest' ~/.bashrc || echo 'export CADET_PATH=/opt/CADET/latest' >> ~/.bashrc
```

**Option 5c: Set in notebook (manual - per notebook)**

```python
import os
os.environ["CADET_PATH"] = "/opt/CADET/latest"
```

---

## Option B: Pip + venv Setup

Use this if you don't have conda or prefer pure pip. Note: CADET solver must be installed separately on your system.

### 1. Environment Setup

Create virtual environment:

```bash
python3 -m venv .venv
```

Activate (Linux/Mac):

```bash
source .venv/bin/activate
```

Activate (Windows):

```bash
.venv\Scripts\activate
```

### 2. Install cadet_simplified Package

**Choose one:**

**Option 2a: From GitHub (recommended)**

```bash
python -m pip install "git+https://github.com/MoritzImendoerffer/cadetgui.git"
```

**Option 2b: From local repository**

```bash
cd /path/to/cadet_simplified
```

```bash
pip install -e .
```

### 3. Register Jupyter Kernel

Install ipykernel:

```bash
pip install ipykernel
```

Register the kernel:

```bash
python -m ipykernel install --user --name cadet_simplified --display-name "Python (CADET Simplified)"
```

### 4. Configure CADET Path

**Choose one:**

**Option 4a: Set in kernel config (recommended - automatic on kernel startup)**

```bash
python -c "import json,subprocess,pathlib; p=pathlib.Path(subprocess.run(['jupyter','--data-dir'],capture_output=True,text=True).stdout.strip())/'kernels'/'cadet_simplified'/'kernel.json'; d=json.loads(p.read_text()); d.setdefault('env',{})['CADET_PATH']='/opt/CADET/latest'; p.write_text(json.dumps(d,indent=2)+'\n')"
```

**Option 4b: Add to ~/.bashrc (for all terminal sessions)**

```bash
grep -qxF 'export CADET_PATH=/opt/CADET/latest' ~/.bashrc || echo 'export CADET_PATH=/opt/CADET/latest' >> ~/.bashrc
```

**Option 4c: Set in notebook (manual - per notebook)**

```python
import os
os.environ["CADET_PATH"] = "/opt/CADET/latest"
```

---

## Quick Reference

| Step | Conda | Pip + venv |
|------|-------|------------|
| Create env | `conda create -n cadet_simplified python=3.10` | `python3 -m venv .venv` |
| Activate | `conda activate cadet_simplified` | `source .venv/bin/activate` |
| Install CADET | `conda install -c conda-forge cadet` | Already installed on system |
| Install package | `pip install -e .` | `pip install -e .` |
| Register kernel | `python -m ipykernel install --user --name cadet_simplified` | Same |

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
jupyter kernelspec uninstall cadet_simplified
```

### Update kernel after environment changes

If you update packages in the conda environment, the kernel automatically uses the updated packages (no re-registration needed).

---

## Troubleshooting

### Kernel not appearing in JupyterLab

1. Restart JupyterLab completely
2. Check kernel is registered: `jupyter kernelspec list`
3. Re-register if needed: `python -m ipykernel install --user --name cadet_simplified --display-name "Python (CADET Simplified)"`

### "No module named cadet_simplified"

The package isn't installed in the kernel's environment:

```bash
conda activate cadet_simplified
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
conda activate cadet_simplified
conda install -c conda-forge cadet -y
export CADET_PATH="$CONDA_PREFIX/bin"
```

### Import errors for dependencies

```bash
conda activate cadet_simplified
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
