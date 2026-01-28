"""Entry script for CADET Simplified GUI.

Run with:
    python dev_gui.py
    
Or with panel serve (after installing the package):
    pip install -e .
    panel serve dev_gui.py --show --autoreload
"""

from pathlib import Path

import panel as pn
pn.extension('tabulator', notifications=True)

from cadet_simplified.app import SimplifiedCADETApp

# =============================================================================
# Configuration
# =============================================================================

# Path to CADET installation (None = auto-detect)
CADET_PATH = None  # Or specify: "/path/to/cadet/bin"

# Directory for storing experiment results
# Use a sensible default in user's home directory
STORAGE_DIR = Path("~").expanduser() / "cadet_experiments"

# =============================================================================
# Create and run the app
# =============================================================================

app = SimplifiedCADETApp(
    storage_dir=STORAGE_DIR,
    cadet_path=CADET_PATH,
)

# For panel serve
app.view().servable()

if __name__ == "__main__":
    print(f"Starting CADET Simplified GUI...")
    print(f"Storage directory: {STORAGE_DIR}")
    
    pn.serve(
        app.view(),
        port=5006,
        show=True,
        autoreload=False,
        title="CADET Simplified",
    )