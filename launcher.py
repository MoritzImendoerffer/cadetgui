"""Launcher for CADET Simplified GUI in JupyterHub/Lab environments.

Provides a simple one-liner to start the GUI from a Jupyter notebook:

    from cadet_simplified import launch_gui
    server = launch_gui()
    
    # Later, to stop:
    server.stop()

The launcher automatically:
- Finds an available port
- Detects JupyterHub environment and constructs proxy URLs
- Starts the Panel server in a background thread
- Provides clear instructions for accessing the GUI
"""

import os
import socket
import threading
from pathlib import Path
from typing import Callable

import panel as pn


def _find_available_port(start: int = 5006, end: int = 5100) -> int:
    """Find an available port in the given range.
    
    Parameters
    ----------
    start : int
        Start of port range
    end : int
        End of port range (exclusive)
        
    Returns
    -------
    int
        Available port number
        
    Raises
    ------
    RuntimeError
        If no available port found in range
    """
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No available port found in range {start}-{end}")


def _get_jupyterhub_user() -> str | None:
    """Get JupyterHub username from environment.
    
    Returns
    -------
    str or None
        Username if in JupyterHub, None otherwise
    """
    return os.environ.get("JUPYTERHUB_USER")


def _get_jupyterhub_service_prefix() -> str | None:
    """Get JupyterHub service prefix from environment.
    
    Returns
    -------
    str or None
        Service prefix if in JupyterHub, None otherwise
    """
    return os.environ.get("JUPYTERHUB_SERVICE_PREFIX")


class GUIServer:
    """Wrapper for the Panel server with stop capability.
    
    Attributes
    ----------
    port : int
        Port the server is running on
    proxy_url : str or None
        JupyterHub proxy URL if applicable
    direct_url : str
        Direct localhost URL
        
    Example
    -------
    >>> server = GUIServer(app_factory, port=5007)
    >>> server.start()
    >>> # ... use the GUI ...
    >>> server.stop()
    """
    
    def __init__(
        self,
        app_factory: Callable,
        port: int,
        storage_dir: str | Path | None = None,
    ):
        """Initialize the GUI server.
        
        Parameters
        ----------
        app_factory : callable
            Function that creates the Panel viewable
        port : int
            Port to run on
        storage_dir : str or Path, optional
            Storage directory for experiments
        """
        self.port = port
        self.storage_dir = storage_dir
        self._app_factory = app_factory
        self._server = None
        self._thread = None
        
        # Build URLs
        self.direct_url = f"http://localhost:{port}"
        
        # Check for JupyterHub
        user = _get_jupyterhub_user()
        service_prefix = _get_jupyterhub_service_prefix()
        
        if service_prefix:
            # Use service prefix if available (more reliable)
            self.proxy_url = f"{service_prefix}proxy/{port}/"
        elif user:
            # Fall back to user-based URL
            self.proxy_url = f"/user/{user}/proxy/{port}/"
        else:
            self.proxy_url = None
    
    def start(self) -> "GUIServer":
        """Start the server in a background thread.
        
        Returns
        -------
        GUIServer
            Self for method chaining
        """
        if self._server is not None:
            print("Server is already running.")
            return self
        
        def run_server():
            # Create the app
            app = self._app_factory()
            
            # Start server (blocking in this thread)
            self._server = pn.serve(
                {"/": app},
                port=self.port,
                show=False,
                websocket_origin="*",
                threaded=False,
            )
        
        # Run in background thread
        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        
        # Give server a moment to start
        import time
        time.sleep(1.0)
        
        return self
    
    def stop(self):
        """Stop the server."""
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass  # Server may already be stopped
            self._server = None
            print("Server stopped.")
        else:
            print("Server is not running.")
    
    def __repr__(self) -> str:
        status = "running" if self._server is not None else "stopped"
        return f"GUIServer(port={self.port}, status={status})"


def launch_gui(
    storage_dir: str | Path | None = None,
    port: int | None = None,
    cadet_path: str | None = None,
) -> GUIServer:
    """Launch the CADET Simplified GUI for JupyterHub/Lab.
    
    This is the recommended way to start the GUI from a Jupyter notebook.
    It automatically finds an available port and handles JupyterHub proxy
    configuration.
    
    Parameters
    ----------
    storage_dir : str or Path, optional
        Directory for storing experiment results.
        Defaults to ~/cadet_experiments
    port : int, optional
        Specific port to use. If None, finds an available port.
    cadet_path : str, optional
        Path to CADET installation. If None, auto-detects.
        
    Returns
    -------
    GUIServer
        Server instance with .stop() method
        
    Example
    -------
    >>> from cadet_simplified import launch_gui
    >>> server = launch_gui()
    >>> # ... use the GUI ...
    >>> server.stop()
    """
    # Import here to avoid circular imports
    from .app import SimplifiedCADETApp
    
    # Set defaults
    if storage_dir is None:
        storage_dir = Path("~").expanduser() / "cadet_experiments"
    else:
        storage_dir = Path(storage_dir)
    
    # Find available port
    if port is None:
        port = _find_available_port()
    
    # Create app factory
    def create_app():
        app = SimplifiedCADETApp(
            storage_dir=storage_dir,
            cadet_path=cadet_path,
        )
        return app.view()
    
    # Create and start server
    server = GUIServer(
        app_factory=create_app,
        port=port,
        storage_dir=storage_dir,
    )
    server.start()
    
    # Print instructions
    _print_launch_message(server)
    
    return server


def _print_launch_message(server: GUIServer):
    """Print launch instructions to the console."""
    width = 62
    
    print("=" * width)
    print(f"  CADET Simplified GUI started on port {server.port}")
    print()
    
    if server.proxy_url:
        print(f"  JupyterHub URL: {server.proxy_url}")
    
    print(f"  Direct URL:     {server.direct_url}")
    print()
    print("  To stop: server.stop()")
    print("=" * width)
