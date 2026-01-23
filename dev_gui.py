"""Entry script for CADET Simplified GUI.

Run with:
    python dev_gui.py
    
Or with panel serve:
    panel serve dev_gui.py --show --autoreload
"""


import panel as pn
pn.extension('tabulator', notifications=True)

from cadet_simplified.app import SimplifiedCADETApp

# Optional: Set your CADET installation path here
# CADET_PATH = "/path/to/cadet/bin"
CADET_PATH = None  # Set to auto-detect

# Storage directory for experiment JSON files
STORAGE_DIR = "./experiments"

# Create the app
app = SimplifiedCADETApp(
    storage_dir=STORAGE_DIR,
    cadet_path=CADET_PATH,
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