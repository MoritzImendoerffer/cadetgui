"""Dynamic parameter introspection for CADET-Process classes.

This module extracts parameter metadata directly from CADET-Process descriptor classes,
eliminating the need for hardcoded parameter definitions.

The CADET-Process descriptor system stores rich metadata on each parameter:
- name: Parameter name (set by StructMeta metaclass)
- default: Default value (None if no default)
- is_optional: Whether parameter can be unset
- unit: Physical unit (often empty in current CADET-Process)
- description: Human-readable description (often empty)
- ty: Type constraint (for Typed descriptors)
- lb, ub: Bounds (for Ranged descriptors)
- size: Size dependency (for Sized descriptors), e.g., 'n_comp' or ('n_comp', 'n_bound_states')

Usage:
    >>> from CADETProcess.processModel import Langmuir, ComponentSystem
    >>> cs = ComponentSystem(['Salt', 'Protein1', 'Protein2'])
    >>> binding = Langmuir(cs, name='langmuir')
    >>> params = extract_model_parameters(binding)
    >>> for p in params:
    ...     print(f"{p.name}: {p.category.value}, size={p.resolved_size}")
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Type, Optional, Union
import math


class ParameterCategory(Enum):
    """Category of parameter based on its size dependency."""
    SCALAR = "scalar"                      # Single value
    PER_COMPONENT = "per_component"        # Size depends on n_comp
    PER_BOUND_STATE = "per_bound_state"    # Size depends on n_bound_states
    MATRIX = "matrix"                      # 2D array
    OTHER_SIZED = "other_sized"            # Other size dependencies


@dataclass
class ParameterInfo:
    """Complete metadata for a parameter extracted from a CADET-Process descriptor.
    
    Attributes
    ----------
    name : str
        Internal parameter name (e.g., 'adsorption_rate')
    display_name : str
        Human-readable name (e.g., 'Adsorption Rate')
    description : str
        Description from descriptor, or empty string
    unit : str
        Physical unit from descriptor, or empty string
    default : Any
        Default value, or None if no default
    required : bool
        True if parameter must be set
    is_optional : bool
        True if explicitly marked optional in descriptor
    bounds : tuple[float | None, float | None]
        (lower_bound, upper_bound), None means unbounded
    python_type : Type | None
        Inferred Python type (float, int, list, etc.)
    category : ParameterCategory
        Classification based on size dependency
    size_spec : tuple | str | int | None
        Raw size specification from descriptor
    resolved_size : int | tuple | None
        Actual size after resolution with instance context
    descriptor_class : str
        Name of the descriptor class (e.g., 'SizedUnsignedList')
    """
    name: str
    display_name: str
    description: str
    unit: str
    default: Any
    required: bool
    is_optional: bool
    bounds: tuple[Optional[float], Optional[float]]
    python_type: Optional[Type]
    category: ParameterCategory
    size_spec: Union[tuple, str, int, None]
    resolved_size: Union[int, tuple, None]
    descriptor_class: str

    def is_scalar(self) -> bool:
        """Check if this is a scalar parameter."""
        return self.category == ParameterCategory.SCALAR

    def is_per_component(self) -> bool:
        """Check if this is a per-component parameter."""
        return self.category == ParameterCategory.PER_COMPONENT


# =============================================================================
# Supplementary metadata for units and descriptions
# =============================================================================
# CADET-Process doesn't consistently populate unit/description fields in descriptors.
# This lookup provides human-friendly metadata as a fallback.
# The actual structure (scalar vs per_component, bounds, etc.) is still
# extracted dynamically from the descriptors.

SUPPLEMENTARY_METADATA: dict[str, dict[str, str]] = {
    # Binding model parameters
    'adsorption_rate': {'unit': '1/(mM·s)', 'description': 'Adsorption rate constant'},
    'desorption_rate': {'unit': '1/s', 'description': 'Desorption rate constant'},
    'characteristic_charge': {'unit': '-', 'description': 'Characteristic charge (nu)'},
    'steric_factor': {'unit': '-', 'description': 'Steric factor (sigma)'},
    'capacity': {'unit': 'mM', 'description': 'Ion exchange / binding capacity'},
    'is_kinetic': {'unit': '-', 'description': 'Use kinetic binding (vs rapid equilibrium)'},
    'reference_liquid_phase_conc': {'unit': 'mM', 'description': 'Reference liquid phase concentration'},
    'reference_solid_phase_conc': {'unit': 'mM', 'description': 'Reference solid phase concentration'},
    
    # GIEX specific
    'adsorption_rate_linear': {'unit': '1/(mM·s)', 'description': 'Linear modifier for adsorption rate'},
    'adsorption_rate_quadratic': {'unit': '1/(mM·s)', 'description': 'Quadratic modifier for adsorption rate'},
    'adsorption_rate_cubic': {'unit': '1/(mM·s)', 'description': 'Cubic modifier for adsorption rate'},
    'adsorption_rate_salt': {'unit': '1/(mM·s)', 'description': 'Salt dependence of adsorption rate'},
    'adsorption_rate_protein': {'unit': '1/(mM·s)', 'description': 'Protein dependence of adsorption rate'},
    'desorption_rate_linear': {'unit': '1/s', 'description': 'Linear modifier for desorption rate'},
    'desorption_rate_quadratic': {'unit': '1/s', 'description': 'Quadratic modifier for desorption rate'},
    'desorption_rate_cubic': {'unit': '1/s', 'description': 'Cubic modifier for desorption rate'},
    'desorption_rate_salt': {'unit': '1/s', 'description': 'Salt dependence of desorption rate'},
    'desorption_rate_protein': {'unit': '1/s', 'description': 'Protein dependence of desorption rate'},
    'characteristic_charge_breaks': {'unit': 'mM', 'description': 'Breakpoints for piecewise nu'},
    
    # Column parameters
    'length': {'unit': 'm', 'description': 'Column length'},
    'diameter': {'unit': 'm', 'description': 'Column diameter'},
    'bed_porosity': {'unit': '-', 'description': 'Interstitial (bed) porosity'},
    'particle_porosity': {'unit': '-', 'description': 'Intraparticle porosity'},
    'total_porosity': {'unit': '-', 'description': 'Total column porosity'},
    'particle_radius': {'unit': 'm', 'description': 'Particle radius'},
    'axial_dispersion': {'unit': 'm²/s', 'description': 'Axial dispersion coefficient'},
    'film_diffusion': {'unit': 'm/s', 'description': 'Film mass transfer coefficient'},
    'pore_diffusion': {'unit': 'm²/s', 'description': 'Pore diffusion coefficient'},
    'surface_diffusion': {'unit': 'm²/s', 'description': 'Surface diffusion coefficient'},
    'pore_accessibility': {'unit': '-', 'description': 'Fraction of pores accessible'},
    'flow_direction': {'unit': '-', 'description': 'Flow direction (1=forward, -1=backward)'},
}


# =============================================================================
# Helper functions
# =============================================================================

def _name_to_display(name: str) -> str:
    """Convert snake_case to Title Case.
    
    >>> _name_to_display('adsorption_rate')
    'Adsorption Rate'
    """
    return ' '.join(word.capitalize() for word in name.split('_'))


def _get_descriptor_type(descriptor) -> tuple[Optional[Type], str]:
    """Extract Python type and descriptor class name."""
    descriptor_class = type(descriptor).__name__
    
    # Check for explicit type attribute
    python_type = getattr(descriptor, 'ty', None)
    
    # Infer from descriptor class name if not explicit
    if python_type is None:
        if 'Float' in descriptor_class:
            python_type = float
        elif 'Integer' in descriptor_class:
            python_type = int
        elif 'Bool' in descriptor_class:
            python_type = bool
        elif 'String' in descriptor_class:
            python_type = str
        elif 'List' in descriptor_class or 'Array' in descriptor_class:
            python_type = list
    
    return python_type, descriptor_class


def _get_bounds(descriptor) -> tuple[Optional[float], Optional[float]]:
    """Extract bounds from a Ranged descriptor."""
    lb = getattr(descriptor, 'lb', None)
    ub = getattr(descriptor, 'ub', None)
    
    # Convert infinity to None
    if lb is not None and lb == -math.inf:
        lb = None
    if ub is not None and ub == math.inf:
        ub = None
    
    return (lb, ub)


def _resolve_size_element(element: Union[str, int], instance: Any) -> Optional[int]:
    """Resolve a single size element (int or attribute name like 'n_comp')."""
    if isinstance(element, int):
        return element
    
    if isinstance(element, str) and instance is not None:
        try:
            value = getattr(instance, element)
            return value if isinstance(value, int) else len(value)
        except (AttributeError, TypeError):
            pass
    
    return None


def _resolve_size(size_spec: Union[tuple, str, int, None], instance: Any) -> Union[int, tuple, None]:
    """Resolve size specification to actual dimensions."""
    if size_spec is None:
        return None
    
    if isinstance(size_spec, int):
        return size_spec
    
    if isinstance(size_spec, str):
        return _resolve_size_element(size_spec, instance)
    
    if isinstance(size_spec, tuple):
        resolved = tuple(_resolve_size_element(s, instance) for s in size_spec)
        if all(r is not None for r in resolved):
            return resolved[0] if len(resolved) == 1 else resolved
        return resolved
    
    return None


def _categorize_parameter(size_spec: Union[tuple, str, int, None], descriptor_class: str) -> ParameterCategory:
    """Determine parameter category based on size specification."""
    if size_spec is None:
        return ParameterCategory.SCALAR
    
    # Normalize to tuple
    size_tuple = (size_spec,) if isinstance(size_spec, (str, int)) else size_spec
    size_strings = [s for s in size_tuple if isinstance(s, str)]
    
    if 'n_comp' in size_strings and len(size_tuple) == 1:
        return ParameterCategory.PER_COMPONENT
    
    if 'n_bound_states' in size_strings and len(size_tuple) == 1:
        return ParameterCategory.PER_BOUND_STATE
    
    if len(size_tuple) >= 2 and 'n_comp' in size_strings:
        return ParameterCategory.MATRIX
    
    # Check if truly scalar (size=1)
    if len(size_tuple) == 1 and isinstance(size_tuple[0], int) and size_tuple[0] == 1:
        return ParameterCategory.SCALAR
    
    return ParameterCategory.OTHER_SIZED


def _apply_supplementary_metadata(param: ParameterInfo) -> ParameterInfo:
    """Apply supplementary metadata if descriptor fields are empty."""
    if param.name in SUPPLEMENTARY_METADATA:
        meta = SUPPLEMENTARY_METADATA[param.name]
        if not param.unit:
            param.unit = meta.get('unit', '')
        if not param.description:
            param.description = meta.get('description', '')
    return param


# =============================================================================
# Main extraction functions
# =============================================================================

def extract_parameter_info(
    descriptor,
    name: str,
    instance: Any = None,
    required_params: Optional[list] = None,
    optional_params: Optional[list] = None,
) -> ParameterInfo:
    """Extract complete metadata from a single descriptor.
    
    Parameters
    ----------
    descriptor : Descriptor
        A CADET-Process descriptor instance
    name : str
        Parameter name
    instance : Any, optional
        Instance to resolve size dependencies
    required_params : list, optional
        List of required parameter names from the class
    optional_params : list, optional
        List of optional parameter names from the class
    
    Returns
    -------
    ParameterInfo
        Complete parameter metadata
    """
    required_params = required_params or []
    optional_params = optional_params or []
    
    # Extract from descriptor
    display_name = _name_to_display(name)
    description = getattr(descriptor, 'description', None) or ''
    unit = getattr(descriptor, 'unit', None) or ''
    default = getattr(descriptor, 'default', None)
    is_optional = getattr(descriptor, 'is_optional', False)
    
    # Determine required status
    required = (name in required_params) or (default is None and not is_optional)
    if name in optional_params:
        is_optional = True
        required = False
    
    # Type and bounds
    python_type, descriptor_class = _get_descriptor_type(descriptor)
    bounds = _get_bounds(descriptor)
    
    # Size
    size_spec = getattr(descriptor, 'size', None)
    resolved_size = _resolve_size(size_spec, instance)
    category = _categorize_parameter(size_spec, descriptor_class)
    
    return ParameterInfo(
        name=name,
        display_name=display_name,
        description=description,
        unit=unit,
        default=default,
        required=required,
        is_optional=is_optional,
        bounds=bounds,
        python_type=python_type,
        category=category,
        size_spec=size_spec,
        resolved_size=resolved_size,
        descriptor_class=descriptor_class,
    )


def extract_model_parameters(
    obj: Any,
    include_initial_state: bool = False,
    apply_supplementary: bool = True,
) -> list[ParameterInfo]:
    """Extract all parameter metadata from a CADET-Process object.
    
    Parameters
    ----------
    obj : Any
        A CADET-Process object (binding model or unit operation)
    include_initial_state : bool, optional
        Whether to include initial state parameters (c, q, etc.)
    apply_supplementary : bool, optional
        Whether to apply supplementary metadata for empty unit/description
    
    Returns
    -------
    list[ParameterInfo]
        List of parameter info for all parameters
    """
    cls = type(obj)
    
    # Get parameter lists
    parameters = getattr(cls, '_parameters', [])
    required_params = getattr(obj, '_required_parameters', [])
    optional_params = getattr(obj, '_optional_parameters', [])
    initial_state = getattr(cls, '_initial_state', [])
    
    # Filter initial state if not requested
    if not include_initial_state:
        parameters = [p for p in parameters if p not in initial_state]
    
    # Extract info for each parameter
    result = []
    seen_names = set()  # Deduplicate (inheritance can cause duplicates)
    
    for param_name in parameters:
        if param_name in seen_names:
            continue
        seen_names.add(param_name)
        
        # Get descriptor from class
        descriptor = getattr(cls, param_name, None)
        if descriptor is None or isinstance(descriptor, property) or callable(descriptor):
            continue
        if not hasattr(descriptor, '__get__'):
            continue
        
        try:
            info = extract_parameter_info(
                descriptor, param_name, instance=obj,
                required_params=required_params, optional_params=optional_params,
            )
            if apply_supplementary:
                info = _apply_supplementary_metadata(info)
            result.append(info)
        except Exception:
            continue
    
    return result


# =============================================================================
# High-level convenience functions
# =============================================================================

def get_binding_model_parameters(
    binding_model_name: str,
    n_comp: int,
) -> tuple[list[ParameterInfo], list[ParameterInfo]]:
    """Get scalar and per-component parameters for a binding model.
    
    Parameters
    ----------
    binding_model_name : str
        Name of binding model class (e.g., 'Langmuir', 'StericMassAction')
    n_comp : int
        Number of components
    
    Returns
    -------
    tuple[list[ParameterInfo], list[ParameterInfo]]
        (scalar_parameters, per_component_parameters)
    
    Examples
    --------
    >>> scalar, per_comp = get_binding_model_parameters('Langmuir', n_comp=3)
    >>> print([p.name for p in per_comp])
    ['adsorption_rate', 'desorption_rate', 'capacity']
    """
    from CADETProcess.processModel import ComponentSystem
    import CADETProcess.processModel as pm
    
    cls = getattr(pm, binding_model_name)
    cs = ComponentSystem(n_comp)
    instance = cls(cs, name='_introspect')
    
    all_params = extract_model_parameters(instance)
    
    scalar = [p for p in all_params if p.category == ParameterCategory.SCALAR]
    per_component = [p for p in all_params if p.category in (
        ParameterCategory.PER_COMPONENT, ParameterCategory.PER_BOUND_STATE
    )]
    
    return scalar, per_component


def get_column_model_parameters(
    column_model_name: str,
    n_comp: int,
) -> tuple[list[ParameterInfo], list[ParameterInfo]]:
    """Get scalar and per-component parameters for a column model.
    
    Parameters
    ----------
    column_model_name : str
        Name of column model class (e.g., 'GeneralRateModel')
    n_comp : int
        Number of components
    
    Returns
    -------
    tuple[list[ParameterInfo], list[ParameterInfo]]
        (scalar_parameters, per_component_parameters)
    """
    from CADETProcess.processModel import ComponentSystem
    import CADETProcess.processModel as pm
    
    cls = getattr(pm, column_model_name)
    cs = ComponentSystem(n_comp)
    instance = cls(cs, name='_introspect')
    
    all_params = extract_model_parameters(instance)
    
    scalar = [p for p in all_params if p.category == ParameterCategory.SCALAR]
    per_component = [p for p in all_params if p.category in (
        ParameterCategory.PER_COMPONENT, ParameterCategory.PER_BOUND_STATE
    )]
    
    return scalar, per_component


def get_available_binding_models() -> list[str]:
    """Get list of available binding model names."""
    try:
        from CADETProcess.processModel import binding
        from CADETProcess.processModel.binding import BindingBaseClass
        
        models = []
        for name in binding.__all__:
            cls = getattr(binding, name, None)
            if cls and isinstance(cls, type) and issubclass(cls, BindingBaseClass):
                if cls is not BindingBaseClass and name != 'NoBinding':
                    models.append(name)
        return sorted(models)
    except ImportError:
        return ['Langmuir', 'StericMassAction', 'GeneralizedIonExchange']


def get_available_column_models() -> list[str]:
    """Get list of available column model names."""
    try:
        from CADETProcess.processModel import unitOperation
        models = []
        for name in ['GeneralRateModel', 'LumpedRateModelWithPores', 'LumpedRateModelWithoutPores']:
            if hasattr(unitOperation, name):
                models.append(name)
        return models
    except ImportError:
        return ['GeneralRateModel', 'LumpedRateModelWithPores', 'LumpedRateModelWithoutPores']


def parameter_info_to_dict(param: ParameterInfo) -> dict:
    """Convert ParameterInfo to a dictionary for JSON serialization."""
    return {
        'name': param.name,
        'display_name': param.display_name,
        'description': param.description,
        'unit': param.unit,
        'default': param.default,
        'required': param.required,
        'is_optional': param.is_optional,
        'bounds': list(param.bounds),
        'python_type': param.python_type.__name__ if param.python_type else None,
        'category': param.category.value,
        'size_spec': param.size_spec,
        'resolved_size': param.resolved_size,
        'descriptor_class': param.descriptor_class,
    }


def print_parameter_summary(params: list[ParameterInfo]) -> None:
    """Print a formatted summary of parameters (for debugging)."""
    print(f"{'Name':<30} {'Category':<18} {'Size':<10} {'Req':<5} {'Default':<15} {'Unit'}")
    print("-" * 100)
    for p in params:
        size_str = str(p.resolved_size) if p.resolved_size else (str(p.size_spec) or '-')
        default_str = str(p.default)[:12] if p.default is not None else '-'
        req_str = 'Y' if p.required else 'N'
        print(f"{p.name:<30} {p.category.value:<18} {size_str:<10} {req_str:<5} {default_str:<15} {p.unit}")
