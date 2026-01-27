"""Fittable parameter definitions for optimization.

Defines which parameters can be fitted and their metadata including
bounds, transforms, and whether they are per-component.

Example:
    from cadet_simplified.optimization import FittableParameter, ParameterCategory
    
    param = FittableParameter(
        name="bed_porosity",
        parameter_path="flow_sheet.column.bed_porosity",
        display_name="Bed Porosity",
        unit="-",
        description="Interstitial (bed) porosity",
        default_lb=0.2,
        default_ub=0.5,
        category=ParameterCategory.COLUMN_TRANSPORT,
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParameterCategory(Enum):
    """Category of fittable parameter."""
    COLUMN_GEOMETRY = "column_geometry"
    COLUMN_TRANSPORT = "column_transport"
    BINDING_KINETIC = "binding_kinetic"
    BINDING_EQUILIBRIUM = "binding_equilibrium"


@dataclass
class FittableParameter:
    """Metadata for a parameter that can be fitted.
    
    Attributes
    ----------
    name : str
        Internal parameter name (e.g., "adsorption_rate")
    parameter_path : str
        CADET-Process parameter path (e.g., "flow_sheet.column.binding_model.adsorption_rate")
    display_name : str
        Human-readable name (e.g., "Adsorption Rate (ka)")
    unit : str
        Physical unit (e.g., "1/(mM·s)")
    description : str
        Description of the parameter
    default_lb : float
        Default lower bound for optimization
    default_ub : float
        Default upper bound for optimization
    suggested_transform : str
        Suggested variable transform: "auto", "log", "linear", "none"
    category : ParameterCategory
        Category for grouping in UI
    per_component : bool
        True if parameter has one value per component
    exclude_salt : bool
        For per-component parameters: whether to exclude salt (component 0) by default
    """
    name: str
    parameter_path: str
    display_name: str
    unit: str
    description: str
    default_lb: float
    default_ub: float
    suggested_transform: str = "auto"
    category: ParameterCategory = ParameterCategory.BINDING_KINETIC
    per_component: bool = False
    exclude_salt: bool = True
    
    def __repr__(self) -> str:
        per_comp = " (per-component)" if self.per_component else ""
        return f"FittableParameter({self.name}{per_comp}, [{self.default_lb}, {self.default_ub}])"


@dataclass
class SelectedVariable:
    """A variable selected for fitting with user-specified settings.
    
    Created when user calls `FittingProblem.add_variable()`.
    
    Attributes
    ----------
    parameter : FittableParameter
        The underlying fittable parameter definition
    lb : float
        Lower bound for this variable
    ub : float
        Upper bound for this variable
    transform : str
        Variable transform to use
    components : list[str], optional
        For per-component parameters: component names to fit
    experiments : list[str], optional
        Experiments this variable applies to (None = all)
    initial_value : float or list[float], optional
        User-specified initial value(s)
    """
    parameter: FittableParameter
    lb: float
    ub: float
    transform: str
    components: list[str] | None = None
    experiments: list[str] | None = None  # None means all experiments
    initial_value: float | list[float] | None = None
    
    # Resolved at build time (component names -> indices)
    _component_indices: list[int] = field(default_factory=list, repr=False)
    
    @property
    def n_variables(self) -> int:
        """Number of optimization variables this selection creates.
        
        For experiment-specific variables, this is per experiment.
        Total variables = n_variables * n_experiments (handled in FittingProblem).
        """
        if self.parameter.per_component and self.components:
            return len(self.components)
        return 1
    
    @property
    def is_experiment_specific(self) -> bool:
        """Whether this variable is fitted separately for specific experiments."""
        return self.experiments is not None
    
    def get_variable_names(self, experiment_name: str | None = None) -> list[str]:
        """Get names for all optimization variables from this selection.
        
        Parameters
        ----------
        experiment_name : str, optional
            If provided and variable is experiment-specific, append to name
        
        Returns
        -------
        list[str]
            Variable names (e.g., ["adsorption_rate_Product", "adsorption_rate_Impurity1"])
        """
        base_names = []
        if self.parameter.per_component and self.components:
            base_names = [f"{self.parameter.name}_{comp}" for comp in self.components]
        else:
            base_names = [self.parameter.name]
        
        # Append experiment name for experiment-specific variables
        if experiment_name and self.is_experiment_specific:
            return [f"{name}_{experiment_name}" for name in base_names]
        return base_names
    
    def applies_to_experiment(self, experiment_name: str) -> bool:
        """Check if this variable applies to a given experiment."""
        if self.experiments is None:
            return True
        return experiment_name in self.experiments
    
    def __repr__(self) -> str:
        parts = [self.parameter.name]
        if self.components:
            parts.append(f"components={self.components}")
        if self.experiments:
            parts.append(f"experiments={self.experiments}")
        return f"SelectedVariable({', '.join(parts)})"


# =============================================================================
# Pre-defined fittable parameters for common models
# =============================================================================

def get_column_transport_parameters() -> list[FittableParameter]:
    """Get fittable parameters for column transport (model-independent)."""
    return [
        FittableParameter(
            name="bed_porosity",
            parameter_path="flow_sheet.column.bed_porosity",
            display_name="Bed Porosity",
            unit="-",
            description="Interstitial (bed) porosity",
            default_lb=0.2,
            default_ub=0.5,
            suggested_transform="linear",
            category=ParameterCategory.COLUMN_TRANSPORT,
            per_component=False,
        ),
        FittableParameter(
            name="axial_dispersion",
            parameter_path="flow_sheet.column.axial_dispersion",
            display_name="Axial Dispersion",
            unit="m²/s",
            description="Axial dispersion coefficient",
            default_lb=1e-9,
            default_ub=1e-5,
            suggested_transform="log",
            category=ParameterCategory.COLUMN_TRANSPORT,
            per_component=False,
        ),
        FittableParameter(
            name="total_porosity",
            parameter_path="flow_sheet.column.total_porosity",
            display_name="Total Porosity",
            unit="-",
            description="Total column porosity",
            default_lb=0.4,
            default_ub=0.9,
            suggested_transform="linear",
            category=ParameterCategory.COLUMN_TRANSPORT,
            per_component=False,
        ),
    ]


def get_pore_model_parameters() -> list[FittableParameter]:
    """Get fittable parameters for pore models (GRM, LRMP)."""
    return [
        FittableParameter(
            name="particle_porosity",
            parameter_path="flow_sheet.column.particle_porosity",
            display_name="Particle Porosity",
            unit="-",
            description="Intraparticle porosity",
            default_lb=0.3,
            default_ub=0.8,
            suggested_transform="linear",
            category=ParameterCategory.COLUMN_TRANSPORT,
            per_component=False,
        ),
        FittableParameter(
            name="film_diffusion",
            parameter_path="flow_sheet.column.film_diffusion",
            display_name="Film Diffusion",
            unit="m/s",
            description="Film mass transfer coefficient",
            default_lb=1e-7,
            default_ub=1e-4,
            suggested_transform="log",
            category=ParameterCategory.COLUMN_TRANSPORT,
            per_component=True,
            exclude_salt=False,  # Often fit for all components
        ),
        FittableParameter(
            name="pore_diffusion",
            parameter_path="flow_sheet.column.pore_diffusion",
            display_name="Pore Diffusion",
            unit="m²/s",
            description="Pore diffusion coefficient",
            default_lb=1e-12,
            default_ub=1e-8,
            suggested_transform="log",
            category=ParameterCategory.COLUMN_TRANSPORT,
            per_component=True,
            exclude_salt=False,
        ),
    ]


def get_sma_binding_parameters() -> list[FittableParameter]:
    """Get fittable parameters for Steric Mass Action binding model.
    
    Uses CADET-Process attribute names directly:
    - adsorption_rate (ka)
    - desorption_rate (kd)
    - characteristic_charge (nu)
    - steric_factor (sigma)
    - capacity (lambda)
    """
    return [
        FittableParameter(
            name="adsorption_rate",
            parameter_path="flow_sheet.column.binding_model.adsorption_rate",
            display_name="Adsorption Rate (ka)",
            unit="m³/(mol·s)",
            description="SMA adsorption rate constant",
            default_lb=1e-3,
            default_ub=1e2,
            suggested_transform="log",
            category=ParameterCategory.BINDING_KINETIC,
            per_component=True,
            exclude_salt=True,
        ),
        FittableParameter(
            name="desorption_rate",
            parameter_path="flow_sheet.column.binding_model.desorption_rate",
            display_name="Desorption Rate (kd)",
            unit="1/s",
            description="SMA desorption rate constant",
            default_lb=1e-3,
            default_ub=1e3,
            suggested_transform="log",
            category=ParameterCategory.BINDING_KINETIC,
            per_component=True,
            exclude_salt=True,
        ),
        FittableParameter(
            name="characteristic_charge",
            parameter_path="flow_sheet.column.binding_model.characteristic_charge",
            display_name="Characteristic Charge (ν)",
            unit="-",
            description="SMA characteristic charge",
            default_lb=1.0,
            default_ub=15.0,
            suggested_transform="linear",
            category=ParameterCategory.BINDING_EQUILIBRIUM,
            per_component=True,
            exclude_salt=True,
        ),
        FittableParameter(
            name="steric_factor",
            parameter_path="flow_sheet.column.binding_model.steric_factor",
            display_name="Steric Factor (σ)",
            unit="-",
            description="SMA steric factor",
            default_lb=0.0,
            default_ub=50.0,
            suggested_transform="linear",
            category=ParameterCategory.BINDING_EQUILIBRIUM,
            per_component=True,
            exclude_salt=True,
        ),
        FittableParameter(
            name="capacity",
            parameter_path="flow_sheet.column.binding_model.capacity",
            display_name="Ionic Capacity (Λ)",
            unit="mM",
            description="SMA ionic capacity",
            default_lb=10.0,
            default_ub=500.0,
            suggested_transform="linear",
            category=ParameterCategory.BINDING_EQUILIBRIUM,
            per_component=False,
        ),
    ]


def get_langmuir_binding_parameters() -> list[FittableParameter]:
    """Get fittable parameters for Langmuir binding model.
    
    Uses CADET-Process attribute names directly.
    """
    return [
        FittableParameter(
            name="adsorption_rate",
            parameter_path="flow_sheet.column.binding_model.adsorption_rate",
            display_name="Adsorption Rate (ka)",
            unit="m³/(mol·s)",
            description="Langmuir adsorption rate constant",
            default_lb=1e-3,
            default_ub=1e2,
            suggested_transform="log",
            category=ParameterCategory.BINDING_KINETIC,
            per_component=True,
            exclude_salt=True,
        ),
        FittableParameter(
            name="desorption_rate",
            parameter_path="flow_sheet.column.binding_model.desorption_rate",
            display_name="Desorption Rate (kd)",
            unit="1/s",
            description="Langmuir desorption rate constant",
            default_lb=1e-3,
            default_ub=1e3,
            suggested_transform="log",
            category=ParameterCategory.BINDING_KINETIC,
            per_component=True,
            exclude_salt=True,
        ),
        FittableParameter(
            name="capacity",
            parameter_path="flow_sheet.column.binding_model.capacity",
            display_name="Maximum Capacity (qmax)",
            unit="mM",
            description="Langmuir maximum binding capacity",
            default_lb=1.0,
            default_ub=500.0,
            suggested_transform="linear",
            category=ParameterCategory.BINDING_EQUILIBRIUM,
            per_component=True,
            exclude_salt=True,
        ),
    ]


# Registry of binding model parameters
BINDING_MODEL_PARAMETERS: dict[str, list[FittableParameter]] = {
    "StericMassAction": get_sma_binding_parameters(),
    "Langmuir": get_langmuir_binding_parameters(),
}

# Registry of column model parameters
COLUMN_MODEL_PARAMETERS: dict[str, list[FittableParameter]] = {
    "LumpedRateModelWithoutPores": get_column_transport_parameters(),
    "LumpedRateModelWithPores": get_column_transport_parameters() + get_pore_model_parameters(),
    "GeneralRateModel": get_column_transport_parameters() + get_pore_model_parameters(),
}
