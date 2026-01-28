"""Model configuration from JSON files.

This module provides parameter definitions for binding and column models,
loaded from JSON config files instead of runtime introspection.

Example:
    >>> from cadet_simplified.configs import get_binding_model_config
    >>> config = get_binding_model_config("StericMassAction")
    >>> for p in config.component_parameters:
    ...     print(f"{p.name}: {p.unit} - {p.description}")
"""

from .loader import (
    ParameterDef,
    ModelConfig,
    get_binding_model_config,
    get_column_model_config,
    list_binding_models,
    list_column_models,
    get_model_class,
    clear_cache,
)

__all__ = [
    "ParameterDef",
    "ModelConfig",
    "get_binding_model_config",
    "get_column_model_config",
    "list_binding_models",
    "list_column_models",
    "get_model_class",
    "clear_cache",
]
