"""Base class for operation modes.

An operation mode defines a chromatography process type (e.g., Load-Wash-Elute)
and specifies:
- What experiment parameters users can modify (lab-friendly units)
- How to convert the configuration to a CADET Process

Column and binding model parameters are defined in JSON config files,
not in the operation mode.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from CADETProcess.processModel import Process

from ..core import ExperimentConfig, ColumnBindingConfig
from ..configs import (
    get_binding_model_config,
    get_column_model_config,
    get_model_class,
    list_binding_models,
    list_column_models,
    ParameterDef,
)


@dataclass
class ProcessParameterDef:
    """Definition of a process/experiment parameter.
    
    These are the lab-friendly parameters like flow rate, volumes, etc.
    Defined by operation modes, not by JSON configs.
    """
    name: str
    display_name: str
    unit: str
    description: str
    default: Any = None
    bounds: tuple[float | None, float | None] = (None, None)
    required: bool = True
    per_component: bool = False


class BaseOperationMode(ABC):
    """Abstract base class for operation modes.
    
    Subclasses define specific chromatography processes (LWE, gradient elution, etc.)
    and how to convert user-friendly Excel parameters to CADET Process objects.
    
    Example:
        >>> mode = LWEConcentrationBased()
        >>> process = mode.create_process(experiment_config, column_binding_config)
    """
    
    # Subclasses should define these
    name: str = "BaseMode"
    description: str = "Base operation mode"
    
    @property
    def supported_column_models(self) -> list[str]:
        """List of supported column model names."""
        return list_column_models()
    
    @property
    def supported_binding_models(self) -> list[str]:
        """List of supported binding model names."""
        return list_binding_models()
    
    @abstractmethod
    def get_experiment_parameters(self) -> list[ProcessParameterDef]:
        """Get the experiment parameters that users can modify.
        
        These are lab-friendly parameters like flow rate in mL/min,
        volumes in CV, concentrations in mM, etc.
        
        Returns
        -------
        list[ProcessParameterDef]
            List of parameter definitions for the Experiments sheet
        """
        pass
    
    @abstractmethod
    def get_component_experiment_parameters(self) -> list[ProcessParameterDef]:
        """Get per-component experiment parameters.
        
        These will be repeated for each component in the Excel template.
        E.g., load_concentration for each component.
        
        Returns
        -------
        list[ProcessParameterDef]
            List of per-component parameter definitions
        """
        pass
    
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
    
    def get_column_parameters(self, column_model: str) -> list[ParameterDef]:
        """Get scalar column parameters from JSON config.
        
        Parameters
        ----------
        column_model : str
            Name of the column model
            
        Returns
        -------
        list[ParameterDef]
            Scalar column parameter definitions
        """
        config = get_column_model_config(column_model)
        return config.scalar_parameters
    
    def get_component_column_parameters(self, column_model: str) -> list[ParameterDef]:
        """Get per-component column parameters from JSON config.
        
        Parameters
        ----------
        column_model : str
            Name of the column model
            
        Returns
        -------
        list[ParameterDef]
            Per-component column parameter definitions
        """
        config = get_column_model_config(column_model)
        return config.component_parameters
    
    def get_binding_parameters(self, binding_model: str) -> list[ParameterDef]:
        """Get scalar binding parameters from JSON config.
        
        Parameters
        ----------
        binding_model : str
            Name of the binding model
            
        Returns
        -------
        list[ParameterDef]
            Scalar binding parameter definitions
        """
        config = get_binding_model_config(binding_model)
        return config.scalar_parameters
    
    def get_component_binding_parameters(self, binding_model: str) -> list[ParameterDef]:
        """Get per-component binding parameters from JSON config.
        
        Parameters
        ----------
        binding_model : str
            Name of the binding model
            
        Returns
        -------
        list[ParameterDef]
            Per-component binding parameter definitions
        """
        config = get_binding_model_config(binding_model)
        return config.component_parameters
    
    def validate_config(
        self,
        experiment: ExperimentConfig,
        column_binding: ColumnBindingConfig,
    ) -> list[str]:
        """Validate the configuration and return any errors.
        
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
                errors.append("Process configuration check failed. Check parameter values.")
                
        except Exception as e:
            errors.append(f"Failed to create process: {str(e)}")
        
        return errors
    
    # -------------------------------------------------------------------------
    # Helper methods for process creation
    # -------------------------------------------------------------------------
    
    def _create_binding_model(
        self,
        binding_model_name: str,
        component_system,
        scalar_params: dict[str, Any],
        component_params: dict[str, list[Any]],
    ):
        """Create and configure binding model.
        
        Parameters
        ----------
        binding_model_name : str
            Name of the binding model
        component_system : ComponentSystem
            CADET-Process component system
        scalar_params : dict[str, Any]
            Scalar parameter values
        component_params : dict[str, list[Any]]
            Per-component parameter values (lists)
            
        Returns
        -------
        BindingModel
            Configured binding model instance
        """
        config = get_binding_model_config(binding_model_name)
        model_class = get_model_class(config.cadet_class)
        
        binding_model = model_class(component_system, name="binding")
        
        # Set scalar parameters
        for param, value in scalar_params.items():
            if hasattr(binding_model, param) and value is not None:
                setattr(binding_model, param, value)
        
        # Set per-component parameters
        for param, values in component_params.items():
            if hasattr(binding_model, param) and values:
                setattr(binding_model, param, values)
        
        return binding_model
    
    def _create_column(
        self,
        column_model_name: str,
        component_system,
        binding_model,
        scalar_params: dict[str, Any],
        component_params: dict[str, list[Any]],
    ):
        """Create and configure column.
        
        Parameters
        ----------
        column_model_name : str
            Name of the column model
        component_system : ComponentSystem
            CADET-Process component system
        binding_model : BindingModel
            Configured binding model
        scalar_params : dict[str, Any]
            Scalar parameter values
        component_params : dict[str, list[Any]]
            Per-component parameter values (lists)
            
        Returns
        -------
        Column
            Configured column instance
        """
        config = get_column_model_config(column_model_name)
        model_class = get_model_class(config.cadet_class)
        
        column = model_class(component_system, name="column")
        column.binding_model = binding_model
        
        # Set scalar parameters
        for param, value in scalar_params.items():
            if hasattr(column, param) and value is not None:
                setattr(column, param, value)
        
        # Set per-component parameters
        for param, values in component_params.items():
            if hasattr(column, param) and values:
                setattr(column, param, values)
        
        return column
