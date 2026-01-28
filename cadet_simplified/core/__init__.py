"""Core dataclasses for CADET Simplified.

Provides the fundamental data structures used throughout the application:
- ComponentDefinition: A single component (salt, protein, impurity)
- ExperimentConfig: Parameters for one experiment
- ColumnBindingConfig: Column and binding model configuration

Example:
    >>> from cadet_simplified.core import ExperimentConfig, ComponentDefinition
    >>> 
    >>> config = ExperimentConfig(
    ...     name="gradient_50_500",
    ...     parameters={"flow_rate_mL_min": 1.0, "load_cv": 5.0},
    ...     components=[
    ...         ComponentDefinition("Salt", is_salt=True),
    ...         ComponentDefinition("Product"),
    ...     ],
    ... )
"""

from .dataclasses import (
    ComponentDefinition,
    ExperimentConfig,
    ColumnBindingConfig,
)

__all__ = [
    "ComponentDefinition",
    "ExperimentConfig",
    "ColumnBindingConfig",
]
