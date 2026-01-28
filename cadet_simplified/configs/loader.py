"""Configuration loader for binding and column model parameters.

Loads parameter definitions from JSON files instead of runtime introspection.
This provides a single source of truth for all model parameters.

Usage:
    >>> from cadet_simplified.configs import get_binding_model_config, get_column_model_config
    >>> sma_config = get_binding_model_config("StericMassAction")
    >>> print(sma_config.scalar_parameters)
    >>> print(sma_config.component_parameters)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class ParameterDef:
    """Definition of a single parameter.
    
    Attributes
    ----------
    name : str
        Parameter name (matches CADET-Process attribute name)
    type : str
        Python type name: "float", "int", "bool"
    default : Any
        Default value, or None if required
    unit : str
        Physical unit string
    bounds : tuple[float | None, float | None]
        (lower_bound, upper_bound), None means unbounded
    description : str
        Human-readable description
    """
    name: str
    type: str
    default: Any
    unit: str
    bounds: tuple[float | None, float | None]
    description: str
    
    @classmethod
    def from_dict(cls, data: dict) -> "ParameterDef":
        """Create from dictionary (JSON)."""
        bounds_raw = data.get("bounds", [None, None])
        bounds = (bounds_raw[0], bounds_raw[1]) if bounds_raw else (None, None)
        return cls(
            name=data["name"],
            type=data.get("type", "float"),
            default=data.get("default"),
            unit=data.get("unit", "-"),
            bounds=bounds,
            description=data.get("description", ""),
        )
    
    @property
    def required(self) -> bool:
        """Whether this parameter must be provided (no default)."""
        return self.default is None
    
    @property
    def python_type(self) -> type:
        """Get the Python type."""
        type_map = {
            "float": float,
            "int": int,
            "bool": bool,
            "str": str,
        }
        return type_map.get(self.type, float)


@dataclass
class ModelConfig:
    """Configuration for a model (binding or column).
    
    Attributes
    ----------
    name : str
        Model class name (e.g., "StericMassAction")
    cadet_class : str
        Full import path (e.g., "CADETProcess.processModel.StericMassAction")
    description : str
        Human-readable description
    scalar_parameters : list[ParameterDef]
        Parameters that have a single value
    component_parameters : list[ParameterDef]
        Parameters that have one value per component
    """
    name: str
    cadet_class: str
    description: str
    scalar_parameters: list[ParameterDef] = field(default_factory=list)
    component_parameters: list[ParameterDef] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        """Create from dictionary (JSON)."""
        scalar_params = [
            ParameterDef.from_dict(p) 
            for p in data.get("scalar_parameters", [])
        ]
        component_params = [
            ParameterDef.from_dict(p) 
            for p in data.get("component_parameters", [])
        ]
        return cls(
            name=data["name"],
            cadet_class=data.get("cadet_class", f"CADETProcess.processModel.{data['name']}"),
            description=data.get("description", ""),
            scalar_parameters=scalar_params,
            component_parameters=component_params,
        )
    
    def get_scalar_parameter(self, name: str) -> ParameterDef | None:
        """Get a scalar parameter by name."""
        for p in self.scalar_parameters:
            if p.name == name:
                return p
        return None
    
    def get_component_parameter(self, name: str) -> ParameterDef | None:
        """Get a component parameter by name."""
        for p in self.component_parameters:
            if p.name == name:
                return p
        return None
    
    @property
    def all_parameters(self) -> list[ParameterDef]:
        """Get all parameters (scalar + component)."""
        return self.scalar_parameters + self.component_parameters
    
    @property
    def scalar_parameter_names(self) -> list[str]:
        """Get names of all scalar parameters."""
        return [p.name for p in self.scalar_parameters]
    
    @property
    def component_parameter_names(self) -> list[str]:
        """Get names of all component parameters."""
        return [p.name for p in self.component_parameters]


# Module-level cache for loaded configs
_binding_model_cache: dict[str, ModelConfig] = {}
_column_model_cache: dict[str, ModelConfig] = {}


def _get_configs_dir() -> Path:
    """Get the configs directory path."""
    return Path(__file__).parent


def _load_json_config(path: Path) -> dict:
    """Load a JSON config file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_binding_model_config(name: str) -> ModelConfig:
    """Get configuration for a binding model.
    
    Parameters
    ----------
    name : str
        Binding model name (e.g., "StericMassAction", "Langmuir")
        
    Returns
    -------
    ModelConfig
        Model configuration with parameter definitions
        
    Raises
    ------
    ValueError
        If model config not found
    """
    if name in _binding_model_cache:
        return _binding_model_cache[name]
    
    config_path = _get_configs_dir() / "binding_models" / f"{name}.json"
    
    if not config_path.exists():
        available = list_binding_models()
        raise ValueError(f"Unknown binding model: {name}. Available: {available}")
    
    data = _load_json_config(config_path)
    config = ModelConfig.from_dict(data)
    _binding_model_cache[name] = config
    return config


def get_column_model_config(name: str) -> ModelConfig:
    """Get configuration for a column model.
    
    Parameters
    ----------
    name : str
        Column model name (e.g., "GeneralRateModel", "LumpedRateModelWithPores")
        
    Returns
    -------
    ModelConfig
        Model configuration with parameter definitions
        
    Raises
    ------
    ValueError
        If model config not found
    """
    if name in _column_model_cache:
        return _column_model_cache[name]
    
    config_path = _get_configs_dir() / "column_models" / f"{name}.json"
    
    if not config_path.exists():
        available = list_column_models()
        raise ValueError(f"Unknown column model: {name}. Available: {available}")
    
    data = _load_json_config(config_path)
    config = ModelConfig.from_dict(data)
    _column_model_cache[name] = config
    return config


def list_binding_models() -> list[str]:
    """List all available binding models.
    
    Returns
    -------
    list[str]
        Names of available binding models
    """
    configs_dir = _get_configs_dir() / "binding_models"
    if not configs_dir.exists():
        return []
    return sorted([
        p.stem for p in configs_dir.glob("*.json")
    ])


def list_column_models() -> list[str]:
    """List all available column models.
    
    Returns
    -------
    list[str]
        Names of available column models
    """
    configs_dir = _get_configs_dir() / "column_models"
    if not configs_dir.exists():
        return []
    return sorted([
        p.stem for p in configs_dir.glob("*.json")
    ])


def get_model_class(class_path: str) -> type:
    """Import and return a model class from its path.
    
    Parameters
    ----------
    class_path : str
        Full import path (e.g., "CADETProcess.processModel.StericMassAction")
        
    Returns
    -------
    type
        The model class
    """
    from importlib import import_module
    
    module_path, class_name = class_path.rsplit(".", 1)
    module = import_module(module_path)
    return getattr(module, class_name)


def clear_cache() -> None:
    """Clear the config cache (useful for testing)."""
    _binding_model_cache.clear()
    _column_model_cache.clear()
