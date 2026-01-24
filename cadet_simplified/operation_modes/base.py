"""Base class for operation modes.

An operation mode defines a chromatography process type (e.g., Load-Wash-Elute)
and specifies:
- What experiment parameters users can modify (lab-friendly units)
- What column/binding parameters are exposed
- How to convert the configuration to a CADET Process
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

from addict import Dict

from CADETProcess import processModel
from CADETProcess.processModel.unitOperation import ChromatographicColumnBase

class ParameterType(Enum):
    """Type of parameter for Excel template generation."""
    SCALAR = "scalar"
    PER_COMPONENT = "per_component"


@dataclass
class ParameterDefinition:
    """Definition of a single parameter."""
    name: str
    display_name: str
    unit: str
    description: str
    default: Any = None
    param_type: ParameterType = ParameterType.SCALAR
    required: bool = True
    bounds: tuple[float | None, float | None] = (None, None)
    
    def __post_init__(self):
        """Validate bounds."""
        if self.bounds[0] is not None and self.bounds[1] is not None:
            if self.bounds[0] > self.bounds[1]:
                raise ValueError(f"Lower bound {self.bounds[0]} > upper bound {self.bounds[1]}")


@dataclass
class ComponentDefinition:
    """Definition of a component in the system."""
    name: str
    is_salt: bool = False
    molecular_weight: float | None = None


@dataclass 
class ExperimentConfig:
    """Configuration for a single experiment parsed from Excel."""
    name: str
    parameters: dict[str, Any]
    components: list[ComponentDefinition]


@dataclass
class ColumnBindingConfig:
    """Column and binding model configuration parsed from Excel."""
    column_model: str
    binding_model: str
    column_parameters: dict[str, Any]
    binding_parameters: dict[str, Any]
    # Per-component parameters stored as lists (indexed by component order)
    component_column_parameters: dict[str, list[Any]] = field(default_factory=dict)
    component_binding_parameters: dict[str, list[Any]] = field(default_factory=dict)


# Supported models registry
class ModelRegistry(dict):
    """Registry supporting both dict and attribute access
    Provides basic autocompletion at runtime in terminal or juypter cell
    """
    
    def __init__(self, models_dict):
        super().__init__(models_dict)
        # Dynamically add attributes for autocomplete
        for key, value in models_dict.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return f"Available: {list(self.keys())}"
     
SUPPORTED_BINDING_MODELS = dict()
for name in processModel.binding.__all__:
    pm = getattr(processModel, name)
    SUPPORTED_BINDING_MODELS[name] = pm.__module__ + "." + name
SUPPORTED_BINDING_MODELS = ModelRegistry(SUPPORTED_BINDING_MODELS)

SUPPORTED_COLUMN_MODELS = dict()
for name in processModel.unitOperation.__all__:
    uo = getattr(processModel.unitOperation, name)
    if issubclass(uo, processModel.unitOperation.ChromatographicColumnBase):
        if uo is not processModel.unitOperation.ChromatographicColumnBase:
            SUPPORTED_COLUMN_MODELS[name] = uo.__module__ + "." + name
SUPPORTED_COLUMN_MODELS = ModelRegistry(SUPPORTED_COLUMN_MODELS)           
            
class BaseOperationMode(ABC):
    """Abstract base class for operation modes.
    
    Subclasses define specific chromatography processes (LWE, gradient elution, etc.)
    and how to convert user-friendly Excel parameters to CADET Process objects.
    
    Example:
        >>> mode = LWEConcentrationBased()
        >>> template_config = mode.get_template_config("GeneralRateModel", "StericMassAction", 4)
        >>> # ... user fills Excel ...
        >>> process = mode.create_process(experiment_config, column_binding_config)
    """
    
    # Subclasses should define these
    name: str = "BaseMode"
    description: str = "Base operation mode"
    supported_column_models: list[str] = list(SUPPORTED_COLUMN_MODELS.keys())
    supported_binding_models: list[str] = list(SUPPORTED_BINDING_MODELS.keys())
    
    @abstractmethod
    def get_experiment_parameters(self) -> list[ParameterDefinition]:
        """Get the experiment parameters that users can modify.
        
        These are lab-friendly parameters like flow rate in CV/min,
        volumes in CV, concentrations in mM, etc.
        
        Returns
        -------
        list[ParameterDefinition]
            List of parameter definitions for the Experiments sheet
        """
        pass
    
    @abstractmethod
    def get_component_experiment_parameters(self) -> list[ParameterDefinition]:
        """Get per-component experiment parameters.
        
        These will be repeated for each component in the Excel template.
        E.g., load_concentration for each component.
        
        Returns
        -------
        list[ParameterDefinition]
            List of per-component parameter definitions
        """
        pass
    
    def get_column_parameters(self, column_model: str) -> list[ParameterDefinition]:
        """Get column parameters to expose in the template.
        
        Default implementation uses introspection on CADET-Process models.
        Override for custom parameter selection.
        
        Parameters
        ----------
        column_model : str
            Name of the column model
            
        Returns
        -------
        list[ParameterDefinition]
            List of column parameter definitions
        """
        return self._introspect_column_parameters(column_model)
    
    def get_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Get binding parameters to expose in the template.
        
        Default implementation uses introspection on CADET-Process models.
        Override for custom parameter selection.
        
        Parameters
        ----------
        binding_model : str
            Name of the binding model
            
        Returns
        -------
        list[ParameterDefinition]
            List of binding parameter definitions
        """
        return self._introspect_binding_parameters(binding_model)
    
    def get_component_column_parameters(self, column_model: str) -> list[ParameterDefinition]:
        """Get per-component column parameters (e.g., film_diffusion).
        
        Parameters
        ----------
        column_model : str
            Name of the column model
            
        Returns
        -------
        list[ParameterDefinition]
            List of per-component column parameter definitions
        """
        return self._introspect_component_column_parameters(column_model)
    
    def get_component_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Get per-component binding parameters (e.g., adsorption_rate).
        
        Parameters
        ----------
        binding_model : str
            Name of the binding model
            
        Returns
        -------
        list[ParameterDefinition]
            List of per-component binding parameter definitions
        """
        return self._introspect_component_binding_parameters(binding_model)
    
    @abstractmethod
    def create_process(
        self,
        experiment: ExperimentConfig,
        column_binding: ColumnBindingConfig,
    ) -> "Process":
        """Create a CADET Process from the configuration.
        
        This is the core method that converts user-friendly parameters
        to a fully configured CADET-Process Process object.
        
        Parameters
        ----------
        experiment : ExperimentConfig
            Experiment parameters from Excel
        column_binding : ColumnBindingConfig
            Column and binding configuration from Excel
            
        Returns
        -------
        Process
            CADET-Process Process object ready for simulation
        """
        pass
    
    def validate_config(
        self,
        experiment: ExperimentConfig,
        column_binding: ColumnBindingConfig,
    ) -> list[str]:
        """Validate the configuration and return any errors.
        
        Attempts to build the process and checks configuration.
        
        Parameters
        ----------
        experiment : ExperimentConfig
            Experiment parameters
        column_binding : ColumnBindingConfig
            Column and binding configuration
            
        Returns
        -------
        list[str]
            List of error messages (empty if valid)
        """
        errors = []
        
        try:
            process = self.create_process(experiment, column_binding)
            
            # Use CADET-Process check_config
            if not process.check_config():
                # TODO check_config prints warnings, catpure them
                # For now, add a generic message
                errors.append("Process configuration check failed. Check parameter values.")
                
        except Exception as e:
            errors.append(f"Failed to create process: {str(e)}")
        
        return errors
    
    # Introspection helpers
    
    def _get_model_class(self, class_path: str) -> type:
        """Import and return class from dotted path."""
        from importlib import import_module
        module_path, class_name = class_path.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    
    def _convert_to_parameter_definition(self, param_info: "ParameterInfo") -> ParameterDefinition:
        """Convert ParameterInfo from introspection to ParameterDefinition."""
        from .parameter_introspection import ParameterCategory
        
        # Map category to param_type
        if param_info.category in (ParameterCategory.PER_COMPONENT, ParameterCategory.PER_BOUND_STATE):
            param_type = ParameterType.PER_COMPONENT
        else:
            param_type = ParameterType.SCALAR
        
        return ParameterDefinition(
            name=param_info.name,
            display_name=param_info.display_name,
            unit=param_info.unit or '-',
            description=param_info.description or f'Parameter: {param_info.name}',
            default=param_info.default,
            param_type=param_type,
            required=param_info.required,
            bounds=param_info.bounds,
        )
    
    def _introspect_column_parameters(self, column_model: str) -> list[ParameterDefinition]:
        """Introspect column model for scalar parameters."""
        from .parameter_introspection import get_column_model_parameters
        scalar_params, _ = get_column_model_parameters(column_model, n_comp=2)
        return [self._convert_to_parameter_definition(p) for p in scalar_params]
    
    def _introspect_component_column_parameters(self, column_model: str) -> list[ParameterDefinition]:
        """Introspect column model for per-component parameters."""
        from .parameter_introspection import get_column_model_parameters
        _, per_comp_params = get_column_model_parameters(column_model, n_comp=2)
        return [self._convert_to_parameter_definition(p) for p in per_comp_params]

    def _introspect_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Introspect binding model for scalar parameters."""
        from .parameter_introspection import get_binding_model_parameters
        scalar_params, _ = get_binding_model_parameters(binding_model, n_comp=2)
        return [self._convert_to_parameter_definition(p) for p in scalar_params]
    
    def _introspect_component_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Introspect binding model for per-component parameters."""
        from .parameter_introspection import get_binding_model_parameters
        _, per_comp_params = get_binding_model_parameters(binding_model, n_comp=2)
        return [self._convert_to_parameter_definition(p) for p in per_comp_params]

    
 