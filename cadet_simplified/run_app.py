#!/usr/bin/env python
"""Launch the CADET Simplified GUI application.

Usage:
    python run_app.py [--port PORT] [--storage-dir DIR]
"""

import argparse
import sys
from pathlib import Path

# Add parent to path if running from the package directory
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description='Launch CADET Simplified GUI')
    parser.add_argument('--port', type=int, default=5007, help='Port to serve on')
    parser.add_argument('--storage-dir', type=str, default='./experiments', 
                       help='Directory for experiment storage')
    parser.add_argument('--cadet-path', type=str, default=None,
                       help='Path to CADET installation')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    try:
        import panel as pn
        pn.extension('tabulator', notifications=True)
    except ImportError:
        print("Error: Panel is required. Install with: pip install panel")
        sys.exit(1)
    
    from cadet_simplified.app import SimplifiedCADETApp
    
    print(f"Starting CADET Simplified on port {args.port}...")
    print(f"Storage directory: {args.storage_dir}")
    
    app = SimplifiedCADETApp(
        storage_dir=args.storage_dir,
        cadet_path=args.cadet_path,
    )
    
    pn.serve(
        app.view(),
        port=args.port,
        show=not args.no_browser,
        title="CADET Simplified",
    )


if __name__ == "__main__":
    main()
