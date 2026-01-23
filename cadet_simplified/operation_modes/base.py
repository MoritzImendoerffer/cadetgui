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
SUPPORTED_COLUMN_MODELS = {
    'GeneralRateModel': 'CADETProcess.processModel.GeneralRateModel',
    'LumpedRateModelWithPores': 'CADETProcess.processModel.LumpedRateModelWithPores',
    'LumpedRateModelWithoutPores': 'CADETProcess.processModel.LumpedRateModelWithoutPores',
}

SUPPORTED_BINDING_MODELS = {
    'StericMassAction': 'CADETProcess.processModel.StericMassAction',
    'GeneralizedIonExchange': 'CADETProcess.processModel.GeneralizedIonExchange',
    'Langmuir': 'CADETProcess.processModel.Langmuir',
}


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
                # check_config prints warnings, but we need to capture them
                # For now, add a generic message
                errors.append("Process configuration check failed. Check parameter values.")
                
        except Exception as e:
            errors.append(f"Failed to create process: {str(e)}")
        
        return errors
    
    # =========================================================================
    # Introspection helpers (using parameter_introspection module)
    # =========================================================================
    
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
        try:
            from .parameter_introspection import get_column_model_parameters
            scalar_params, _ = get_column_model_parameters(column_model, n_comp=2)
            return [self._convert_to_parameter_definition(p) for p in scalar_params]
        except ImportError:
            return self._get_predefined_column_parameters(column_model)
    
    def _introspect_component_column_parameters(self, column_model: str) -> list[ParameterDefinition]:
        """Introspect column model for per-component parameters."""
        try:
            from .parameter_introspection import get_column_model_parameters
            _, per_comp_params = get_column_model_parameters(column_model, n_comp=2)
            return [self._convert_to_parameter_definition(p) for p in per_comp_params]
        except ImportError:
            return self._get_predefined_component_column_parameters(column_model)
    
    def _introspect_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Introspect binding model for scalar parameters."""
        try:
            from .parameter_introspection import get_binding_model_parameters
            scalar_params, _ = get_binding_model_parameters(binding_model, n_comp=2)
            return [self._convert_to_parameter_definition(p) for p in scalar_params]
        except ImportError:
            return self._get_predefined_binding_parameters(binding_model)
    
    def _introspect_component_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Introspect binding model for per-component parameters."""
        try:
            from .parameter_introspection import get_binding_model_parameters
            _, per_comp_params = get_binding_model_parameters(binding_model, n_comp=2)
            return [self._convert_to_parameter_definition(p) for p in per_comp_params]
        except ImportError:
            return self._get_predefined_component_binding_parameters(binding_model)
    
    # =========================================================================
    # Predefined parameters (fallback when CADET-Process not available)
    # =========================================================================
    
    def _get_predefined_column_parameters(self, column_model: str) -> list[ParameterDefinition]:
        """Get predefined column parameters when introspection unavailable."""
        params = [
            ParameterDefinition(name='length', display_name='Length', unit='cm', 
                              description='Column length', default=10.0),
            ParameterDefinition(name='diameter', display_name='Diameter', unit='cm',
                              description='Column diameter', default=1.0),
            ParameterDefinition(name='bed_porosity', display_name='Bed Porosity', unit='-',
                              description='Interstitial porosity', default=0.37),
            ParameterDefinition(name='axial_dispersion', display_name='Axial Dispersion', unit='m²/s',
                              description='Axial dispersion coefficient', default=1e-7),
        ]
        
        # Add model-specific parameters
        if column_model != 'LumpedRateModelWithoutPores':
            params.extend([
                ParameterDefinition(name='particle_porosity', display_name='Particle Porosity', unit='-',
                                  description='Intraparticle porosity', default=0.33),
                ParameterDefinition(name='particle_radius', display_name='Particle Radius', unit='µm',
                                  description='Particle radius', default=34.0),
            ])
        else:
            params.append(
                ParameterDefinition(name='total_porosity', display_name='Total Porosity', unit='-',
                                  description='Total porosity', default=0.6),
            )
        
        return params
    
    def _get_predefined_component_column_parameters(self, column_model: str) -> list[ParameterDefinition]:
        """Get predefined per-component column parameters."""
        params = [
            ParameterDefinition(name='film_diffusion', display_name='Film Diffusion', unit='m/s',
                              description='Film mass transfer coefficient',
                              param_type=ParameterType.PER_COMPONENT),
        ]
        
        if column_model != 'LumpedRateModelWithoutPores':
            params.append(
                ParameterDefinition(name='pore_diffusion', display_name='Pore Diffusion', unit='m²/s',
                                  description='Pore diffusion coefficient',
                                  param_type=ParameterType.PER_COMPONENT),
            )
        
        return params
    
    def _get_predefined_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Get predefined binding parameters."""
        params = [
            ParameterDefinition(name='is_kinetic', display_name='Is Kinetic', unit='-',
                              description='Use kinetic binding', default=True),
        ]
        
        if binding_model == 'StericMassAction':
            params.append(
                ParameterDefinition(name='capacity', display_name='Capacity', unit='mM',
                                  description='Ion exchange capacity', default=1200.0),
            )
        
        return params
    
    def _get_predefined_component_binding_parameters(self, binding_model: str) -> list[ParameterDefinition]:
        """Get predefined per-component binding parameters."""
        if binding_model == 'StericMassAction':
            return [
                ParameterDefinition(name='adsorption_rate', display_name='Adsorption Rate', unit='1/s',
                                  description='Adsorption rate constant', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='desorption_rate', display_name='Desorption Rate', unit='1/s',
                                  description='Desorption rate constant', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='characteristic_charge', display_name='Characteristic Charge', unit='-',
                                  description='Characteristic charge (nu)', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='steric_factor', display_name='Steric Factor', unit='-',
                                  description='Steric factor (sigma)', param_type=ParameterType.PER_COMPONENT),
            ]
        elif binding_model == 'Langmuir':
            return [
                ParameterDefinition(name='adsorption_rate', display_name='Adsorption Rate', unit='1/s',
                                  description='Adsorption rate constant', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='desorption_rate', display_name='Desorption Rate', unit='1/s',
                                  description='Desorption rate constant', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='capacity', display_name='Capacity', unit='mM',
                                  description='Maximum binding capacity', param_type=ParameterType.PER_COMPONENT),
            ]
        elif binding_model == 'GeneralizedIonExchange':
            return [
                ParameterDefinition(name='adsorption_rate', display_name='Adsorption Rate', unit='1/s',
                                  description='Adsorption rate constant', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='desorption_rate', display_name='Desorption Rate', unit='1/s',
                                  description='Desorption rate constant', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='characteristic_charge', display_name='Characteristic Charge', unit='-',
                                  description='Characteristic charge (nu)', param_type=ParameterType.PER_COMPONENT),
                ParameterDefinition(name='steric_factor', display_name='Steric Factor', unit='-',
                                  description='Steric factor (sigma)', param_type=ParameterType.PER_COMPONENT),
            ]
        else:
            return []
