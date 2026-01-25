"""Entry script for CADET Simplified GUI.

Run with:
    python dev_gui.py
    
Or with panel serve:
    panel serve dev_gui.py --show --autoreload
"""

import sys
from pathlib import Path

# Add parent directory to path so cadet_simplified package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import panel as pn
pn.extension('tabulator', notifications=True)

from cadet_simplified.app import SimplifiedCADETApp

# Configuration
CADET_PATH = None  # Set to auto-detect, or specify path like "/path/to/cadet/bin"
STORAGE_DIR = "./experiments"
N_LOAD_WORKERS = 4  # Number of workers for parallel loading

# Create the app
app = SimplifiedCADETApp(
    storage_dir=STORAGE_DIR,
    cadet_path=CADET_PATH,
    n_load_workers=N_LOAD_WORKERS,
)

# For panel serve
app.view().servable()

if __name__ == "__main__":
    pn.serve(
        app.view(),
        port=5006,
        show=True,
        autoreload=False,
        title="CADET Simplified",
    )
