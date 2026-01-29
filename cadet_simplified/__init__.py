__version__ = "0.2.2"

# Core dataclasses
from .core import (
    ComponentDefinition,
    ExperimentConfig,
    ColumnBindingConfig,
)

# Configuration (JSON-based)
from .configs import (
    ParameterDef,
    ModelConfig,
    get_binding_model_config,
    get_column_model_config,
    list_binding_models,
    list_column_models,
)

# Operation modes
from .operation_modes import (
    BaseOperationMode,
    LWEConcentrationBased,
    get_operation_mode,
    list_operation_modes,
)

# Simulation
from .simulation import (
    SimulationRunner,
    SimulationResultWrapper,
    ValidationResult,
)

# Excel templates
from .excel import (
    ExcelTemplateGenerator,
    ExcelParser,
    ParseResult,
    parse_excel,
)

# Storage
from .storage import (
    FileStorage,
    LoadedExperiment,
    ExperimentInfo,
)

# Plotting
from .plotting import (
    interpolate_chromatogram,
    plot_chromatogram,
    plot_chromatogram_from_df,
    plot_chromatogram_overlay,
    plot_chromatogram_overlay_from_df,
)

# Launcher
from .launcher import (
    launch_gui,
    GUIServer,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "ComponentDefinition",
    "ExperimentConfig", 
    "ColumnBindingConfig",
    # Configs
    "ParameterDef",
    "ModelConfig",
    "get_binding_model_config",
    "get_column_model_config",
    "list_binding_models",
    "list_column_models",
    # Operation modes
    "BaseOperationMode",
    "LWEConcentrationBased",
    "get_operation_mode",
    "list_operation_modes",
    # Simulation
    "SimulationRunner",
    "SimulationResultWrapper",
    "ValidationResult",
    # Excel
    "ExcelTemplateGenerator",
    "ExcelParser",
    "ParseResult",
    "parse_excel",
    # Storage
    "FileStorage",
    "LoadedExperiment",
    "ExperimentInfo",
    # Plotting
    "interpolate_chromatogram",
    "plot_chromatogram",
    "plot_chromatogram_from_df",
    "plot_chromatogram_overlay",
    "plot_chromatogram_overlay_from_df",
    # Launcher
    "launch_gui",
    "GUIServer",
]


# Lazy import for app (avoids Panel dependency if not needed)
def create_app(*args, **kwargs):
    """Create the GUI application. See app.py for details."""
    from .app import create_app as _create_app
    return _create_app(*args, **kwargs)


def serve_app(*args, **kwargs):
    """Serve the GUI application. See app.py for details."""
    from .app import serve as _serve
    return _serve(*args, **kwargs)
