"""Core dataclasses for experiment configuration.

These are the fundamental data structures used throughout the system:
- ComponentDefinition: A single component (salt, protein, impurity)
- ExperimentConfig: Parameters for one experiment
- ColumnBindingConfig: Column and binding model parameters

All classes support serialization via to_dict()/from_dict() for storage.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComponentDefinition:
    """Definition of a component in the system.
    
    Attributes
    ----------
    name : str
        Component name (e.g., "Salt", "Product", "Impurity1")
    is_salt : bool
        Whether this is the salt/modifier component
    molecular_weight : float, optional
        Molecular weight in kDa (for reference)
    """
    name: str
    is_salt: bool = False
    molecular_weight: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "is_salt": self.is_salt,
            "molecular_weight": self.molecular_weight,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComponentDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            is_salt=data.get("is_salt", False),
            molecular_weight=data.get("molecular_weight"),
        )


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment.
    
    Contains the experiment name, process parameters, and component definitions.
    This is what gets parsed from one row of the Excel "Experiments" sheet.
    
    Attributes
    ----------
    name : str
        Unique experiment name
    parameters : dict[str, Any]
        Process parameters (flow_rate_mL_min, load_cv, gradient_start_mM, etc.)
    components : list[ComponentDefinition]
        Component definitions (order matters - index 0 is usually salt)
    
    Example
    -------
    >>> config = ExperimentConfig(
    ...     name="gradient_50_500",
    ...     parameters={
    ...         "flow_rate_mL_min": 1.0,
    ...         "load_cv": 5.0,
    ...         "gradient_start_mM": 50.0,
    ...         "gradient_end_mM": 500.0,
    ...     },
    ...     components=[
    ...         ComponentDefinition("Salt", is_salt=True),
    ...         ComponentDefinition("Product"),
    ...     ],
    ... )
    """
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    components: list[ComponentDefinition] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "parameters": self.parameters,
            "components": [c.to_dict() for c in self.components],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        components = [
            ComponentDefinition.from_dict(c) 
            for c in data.get("components", [])
        ]
        return cls(
            name=data["name"],
            parameters=data.get("parameters", {}),
            components=components,
        )
    
    @property
    def n_components(self) -> int:
        """Number of components."""
        return len(self.components)
    
    @property
    def component_names(self) -> list[str]:
        """List of component names."""
        return [c.name for c in self.components]


@dataclass
class ColumnBindingConfig:
    """Column and binding model configuration.
    
    Contains all parameters needed to instantiate column and binding models.
    Shared across experiments in the same experiment set.
    
    Attributes
    ----------
    column_model : str
        Column model class name (e.g., "LumpedRateModelWithPores")
    binding_model : str
        Binding model class name (e.g., "StericMassAction")
    column_parameters : dict[str, Any]
        Scalar column parameters (length, diameter, bed_porosity, etc.)
    binding_parameters : dict[str, Any]
        Scalar binding parameters (is_kinetic, reference_liquid_phase_conc, etc.)
    component_column_parameters : dict[str, list[Any]]
        Per-component column parameters (film_diffusion, pore_diffusion, etc.)
        Each key maps to a list with one value per component.
    component_binding_parameters : dict[str, list[Any]]
        Per-component binding parameters (adsorption_rate, characteristic_charge, etc.)
        Each key maps to a list with one value per component.
    
    Example
    -------
    >>> config = ColumnBindingConfig(
    ...     column_model="LumpedRateModelWithPores",
    ...     binding_model="StericMassAction",
    ...     column_parameters={
    ...         "length": 0.1,
    ...         "diameter": 0.01,
    ...         "bed_porosity": 0.37,
    ...     },
    ...     binding_parameters={
    ...         "is_kinetic": True,
    ...     },
    ...     component_column_parameters={
    ...         "film_diffusion": [1e-4, 1e-5, 1e-5],
    ...     },
    ...     component_binding_parameters={
    ...         "adsorption_rate": [0.0, 0.1, 0.1],
    ...         "characteristic_charge": [0.0, 5.0, 6.0],
    ...     },
    ... )
    """
    column_model: str
    binding_model: str
    column_parameters: dict[str, Any] = field(default_factory=dict)
    binding_parameters: dict[str, Any] = field(default_factory=dict)
    component_column_parameters: dict[str, list[Any]] = field(default_factory=dict)
    component_binding_parameters: dict[str, list[Any]] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "column_model": self.column_model,
            "binding_model": self.binding_model,
            "column_parameters": self.column_parameters,
            "binding_parameters": self.binding_parameters,
            "component_column_parameters": self.component_column_parameters,
            "component_binding_parameters": self.component_binding_parameters,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColumnBindingConfig":
        """Create from dictionary."""
        return cls(
            column_model=data["column_model"],
            binding_model=data["binding_model"],
            column_parameters=data.get("column_parameters", {}),
            binding_parameters=data.get("binding_parameters", {}),
            component_column_parameters=data.get("component_column_parameters", {}),
            component_binding_parameters=data.get("component_binding_parameters", {}),
        )
    
    def get_column_parameter(self, name: str, default: Any = None) -> Any:
        """Get a scalar column parameter."""
        return self.column_parameters.get(name, default)
    
    def get_binding_parameter(self, name: str, default: Any = None) -> Any:
        """Get a scalar binding parameter."""
        return self.binding_parameters.get(name, default)
    
    def get_component_column_parameter(self, name: str) -> list[Any] | None:
        """Get a per-component column parameter array."""
        return self.component_column_parameters.get(name)
    
    def get_component_binding_parameter(self, name: str) -> list[Any] | None:
        """Get a per-component binding parameter array."""
        return self.component_binding_parameters.get(name)
