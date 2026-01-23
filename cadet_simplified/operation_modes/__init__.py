"""Operation modes for chromatography processes.

Each operation mode defines a specific chromatography process type
and how to convert user-friendly Excel parameters to CADET Process objects.
"""

from .base import (
    BaseOperationMode,
    ParameterDefinition,
    ParameterType,
    ExperimentConfig,
    ColumnBindingConfig,
    ComponentDefinition,
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
)
from .lwe import LWEConcentrationBased, get_lwe_mode
from .parameter_introspection import (
    ParameterInfo,
    ParameterCategory,
    extract_model_parameters,
    extract_parameter_info,
    get_binding_model_parameters,
    get_column_model_parameters,
    get_available_binding_models,
    get_available_column_models,
    parameter_info_to_dict,
    print_parameter_summary,
)

# Registry of available operation modes
OPERATION_MODES = {
    'LWE_concentration_based': LWEConcentrationBased,
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
    """
    if name not in OPERATION_MODES:
        raise ValueError(f"Unknown operation mode: {name}. Available: {list(OPERATION_MODES.keys())}")
    return OPERATION_MODES[name]()


__all__ = [
    # Base classes
    'BaseOperationMode',
    'ParameterDefinition',
    'ParameterType',
    'ExperimentConfig',
    'ColumnBindingConfig',
    'ComponentDefinition',
    # Registries
    'SUPPORTED_COLUMN_MODELS',
    'SUPPORTED_BINDING_MODELS',
    'OPERATION_MODES',
    # Implementations
    'LWEConcentrationBased',
    'get_lwe_mode',
    'get_operation_mode',
    # Parameter introspection
    'ParameterInfo',
    'ParameterCategory',
    'extract_model_parameters',
    'extract_parameter_info',
    'get_binding_model_parameters',
    'get_column_model_parameters',
    'get_available_binding_models',
    'get_available_column_models',
    'parameter_info_to_dict',
    'print_parameter_summary',
]
