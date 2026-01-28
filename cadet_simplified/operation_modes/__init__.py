"""Operation modes for chromatography processes.

Each operation mode defines a specific chromatography process type
and how to convert user-friendly Excel parameters to CADET Process objects.

Example:
    >>> from cadet_simplified.operation_modes import get_operation_mode
    >>> mode = get_operation_mode("LWE_concentration_based")
    >>> process = mode.create_process(experiment_config, column_binding_config)
"""

from .base import BaseOperationMode
from .lwe import LWEConcentrationBased

# Registry of available operation modes
OPERATION_MODES: dict[str, type[BaseOperationMode]] = {
    "LWE_concentration_based": LWEConcentrationBased,
}


def get_operation_mode(name: str) -> BaseOperationMode:
    """Get an operation mode instance by name.
    
    Parameters
    ----------
    name : str
        Name of the operation mode
        
    Returns
    -------
    BaseOperationMode
        Instance of the operation mode
        
    Raises
    ------
    ValueError
        If operation mode not found
    """
    if name not in OPERATION_MODES:
        available = list(OPERATION_MODES.keys())
        raise ValueError(f"Unknown operation mode: {name}. Available: {available}")
    return OPERATION_MODES[name]()


def list_operation_modes() -> list[str]:
    """List available operation modes."""
    return list(OPERATION_MODES.keys())


__all__ = [
    "BaseOperationMode",
    "LWEConcentrationBased",
    "OPERATION_MODES",
    "get_operation_mode",
    "list_operation_modes",
]
