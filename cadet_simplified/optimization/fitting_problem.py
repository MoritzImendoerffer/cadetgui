"""FittingProblem - main user interface for parameter fitting configuration.

Example
-------
    from cadet_simplified.optimization import FittingProblem
    
    problem = FittingProblem(
        name="my_fit",
        operation_mode=mode,
        experiments=experiments,
        column_binding=column_binding,
        reference_data=reference_data,
    )
    
    # View available parameters
    problem.print_fittable_parameters()
    
    # Add variables to fit with method chaining
    problem.add_variable("bed_porosity") \\
           .add_variable("adsorption_rate", components=["Product", "Impurity1"]) \\
           .add_variable("characteristic_charge", lb=2.0, ub=12.0)
    
    # Experiment-specific variables
    problem.add_variable(
        "desorption_rate",
        components=["Product"],
        experiments=["exp_1", "exp_2"]  # Different values per experiment
    )
    
    # Configure
    problem.metric = "NRMSE"
    problem.select_experiments(["exp_1", "exp_2"])
    
    # Validate before running
    errors = problem.validate()
    if errors:
        print(errors)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .fittable_parameters import (
    FittableParameter,
    SelectedVariable,
    ParameterCategory,
    BINDING_MODEL_PARAMETERS,
    COLUMN_MODEL_PARAMETERS,
)
from .reference_data import ReferenceDataConfig

if TYPE_CHECKING:
    from CADETProcess.optimization import OptimizationProblem


@dataclass
class FittingProblem:
    """User-facing interface for configuring parameter fitting.
    
    This class collects all configuration needed to set up a CADET-Process
    OptimizationProblem, providing a simplified interface for common use cases.
    
    Attributes
    ----------
    name : str
        Name for this fitting problem
    operation_mode : BaseOperationMode
        The operation mode (e.g., LWEConcentrationBased)
    experiments : list[ExperimentConfig]
        List of experiment configurations
    column_binding : ColumnBindingConfig
        Column and binding model configuration
    reference_data : dict[str, ReferenceDataConfig]
        Reference data keyed by experiment name
    selected_variables : list[SelectedVariable]
        Variables selected for fitting
    experiments_to_fit : list[str]
        Names of experiments to include in fitting
    metric : str
        Comparison metric: "NRMSE", "SSE", "RMSE"
    comparison_components : list[str] | None
        Components to compare (None = all available)
    """
    
    name: str
    operation_mode: Any  # BaseOperationMode - avoid circular import
    experiments: list[Any]  # list[ExperimentConfig]
    column_binding: Any  # ColumnBindingConfig
    reference_data: dict[str, ReferenceDataConfig]
    
    # User configuration
    selected_variables: list[SelectedVariable] = field(default_factory=list)
    experiments_to_fit: list[str] = field(default_factory=list)
    metric: str = "NRMSE"
    comparison_components: list[str] | None = None
    
    # Cached
    _fittable_parameters: list[FittableParameter] | None = field(default=None, repr=False)
    _component_names: list[str] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        """Initialize derived attributes."""
        # Cache component names from first experiment
        if self.experiments:
            self._component_names = [
                c.name for c in self.experiments[0].components
            ]
        
        # Default: fit all experiments that have reference data
        if not self.experiments_to_fit:
            self.experiments_to_fit = list(self.reference_data.keys())
    
    # =========================================================================
    # Parameter introspection
    # =========================================================================
    
    @property
    def fittable_parameters(self) -> list[FittableParameter]:
        """Get all fittable parameters for current model combination."""
        if self._fittable_parameters is None:
            self._fittable_parameters = self._get_fittable_parameters()
        return self._fittable_parameters
    
    def _get_fittable_parameters(self) -> list[FittableParameter]:
        """Build list of fittable parameters based on models."""
        params = []
        
        # Get column model parameters
        column_model = self.column_binding.column_model
        if column_model in COLUMN_MODEL_PARAMETERS:
            params.extend(COLUMN_MODEL_PARAMETERS[column_model])
        else:
            # Fallback to basic parameters
            params.extend(COLUMN_MODEL_PARAMETERS.get(
                "LumpedRateModelWithoutPores", []
            ))
        
        # Get binding model parameters
        binding_model = self.column_binding.binding_model
        if binding_model in BINDING_MODEL_PARAMETERS:
            params.extend(BINDING_MODEL_PARAMETERS[binding_model])
        
        return params
    
    def get_fittable_parameter(self, name: str) -> FittableParameter:
        """Get a specific fittable parameter by name.
        
        Parameters
        ----------
        name : str
            Parameter name
            
        Returns
        -------
        FittableParameter
            The parameter definition
            
        Raises
        ------
        ValueError
            If parameter name not found
        """
        for p in self.fittable_parameters:
            if p.name == name:
                return p
        
        available = [p.name for p in self.fittable_parameters]
        raise ValueError(
            f"Unknown parameter: '{name}'. "
            f"Available: {available}"
        )
    
    def list_fittable_parameters(
        self,
        category: ParameterCategory | None = None,
        per_component: bool | None = None,
    ) -> list[FittableParameter]:
        """List available fittable parameters with optional filtering.
        
        Parameters
        ----------
        category : ParameterCategory, optional
            Filter by category
        per_component : bool, optional
            Filter by per-component flag
            
        Returns
        -------
        list[FittableParameter]
            Filtered parameter list
        """
        params = self.fittable_parameters
        
        if category is not None:
            params = [p for p in params if p.category == category]
        
        if per_component is not None:
            params = [p for p in params if p.per_component == per_component]
        
        return params
    
    def print_fittable_parameters(self):
        """Print available parameters in table format."""
        header = (
            f"{'Name':<25} {'Category':<22} {'Per-Comp':<10} "
            f"{'Default Bounds':<22} {'Unit'}"
        )
        print(header)
        print("-" * len(header))
        
        for p in self.fittable_parameters:
            bounds = f"[{p.default_lb:.2e}, {p.default_ub:.2e}]"
            print(
                f"{p.name:<25} {p.category.value:<22} "
                f"{str(p.per_component):<10} {bounds:<22} {p.unit}"
            )
    
    # =========================================================================
    # Variable selection
    # =========================================================================
    
    def add_variable(
        self,
        name: str,
        lb: float | None = None,
        ub: float | None = None,
        transform: str | None = None,
        components: list[str] | None = None,
        experiments: list[str] | None = None,
        initial_value: float | list[float] | None = None,
    ) -> "FittingProblem":
        """Add a variable to fit.
        
        Parameters
        ----------
        name : str
            Parameter name (must be in fittable_parameters)
        lb : float, optional
            Lower bound override
        ub : float, optional
            Upper bound override
        transform : str, optional
            Transform override ("auto", "log", "linear", "none")
        components : list[str], optional
            For per-component parameters: component names to fit
        experiments : list[str], optional
            Experiments this variable applies to. If None, applies to all
            experiments (shared parameter). If specified, creates separate
            variables for each experiment.
        initial_value : float or list[float], optional
            Initial guess override
            
        Returns
        -------
        FittingProblem
            Self for method chaining
            
        Raises
        ------
        ValueError
            If parameter name unknown or invalid component/experiment names
            
        Examples
        --------
        # Shared across all experiments (default)
        problem.add_variable("bed_porosity", lb=0.3, ub=0.5)
        
        # Only for specific experiments (separate value per experiment)
        problem.add_variable(
            "adsorption_rate", 
            components=["Product"], 
            experiments=["exp_1", "exp_2"]
        )
        """
        param = self.get_fittable_parameter(name)
        
        # Handle components for per-component parameters
        if param.per_component:
            if components is None:
                # Default: all non-salt components
                if param.exclude_salt:
                    components = self._component_names[1:]
                else:
                    components = self._component_names.copy()
            
            # Validate component names
            for comp in components:
                if comp not in self._component_names:
                    raise ValueError(
                        f"Unknown component: '{comp}'. "
                        f"Available: {self._component_names}"
                    )
        elif components is not None:
            raise ValueError(
                f"Parameter '{name}' is not per-component, "
                "cannot specify components"
            )
        
        # Validate experiment names if provided
        if experiments is not None:
            for exp_name in experiments:
                if exp_name not in self.reference_data:
                    available = list(self.reference_data.keys())
                    raise ValueError(
                        f"Unknown experiment: '{exp_name}'. "
                        f"Available: {available}"
                    )
        
        # Create selected variable
        selected = SelectedVariable(
            parameter=param,
            lb=lb if lb is not None else param.default_lb,
            ub=ub if ub is not None else param.default_ub,
            transform=transform if transform is not None else param.suggested_transform,
            components=components,
            experiments=experiments,
            initial_value=initial_value,
        )
        
        # Resolve component indices
        if components:
            selected._component_indices = [
                self._component_names.index(c) for c in components
            ]
        
        self.selected_variables.append(selected)
        return self
    
    def remove_variable(self, name: str) -> "FittingProblem":
        """Remove a variable from fitting.
        
        Parameters
        ----------
        name : str
            Parameter name to remove
            
        Returns
        -------
        FittingProblem
            Self for method chaining
        """
        self.selected_variables = [
            v for v in self.selected_variables
            if v.parameter.name != name
        ]
        return self
    
    def clear_variables(self) -> "FittingProblem":
        """Remove all selected variables.
        
        Returns
        -------
        FittingProblem
            Self for method chaining
        """
        self.selected_variables = []
        return self
    
    # =========================================================================
    # Experiment selection
    # =========================================================================
    
    def select_experiments(self, experiment_names: list[str]) -> "FittingProblem":
        """Select which experiments to fit.
        
        Parameters
        ----------
        experiment_names : list[str]
            Names of experiments (must have reference data)
            
        Returns
        -------
        FittingProblem
            Self for method chaining
            
        Raises
        ------
        ValueError
            If experiment has no reference data
        """
        for name in experiment_names:
            if name not in self.reference_data:
                raise ValueError(
                    f"Experiment '{name}' has no reference data. "
                    f"Available: {list(self.reference_data.keys())}"
                )
        
        self.experiments_to_fit = experiment_names
        return self
    
    # =========================================================================
    # Initial values
    # =========================================================================
    
    def get_initial_values(self) -> list[float]:
        """Compute initial values for all selected variables.
        
        Returns values in the order they will appear in the optimization
        vector (expanding per-component parameters and experiment-specific
        variables).
        
        Returns
        -------
        list[float]
            Initial values
        """
        x0 = []
        
        for var in self.selected_variables:
            # Determine number of experiment copies
            if var.is_experiment_specific:
                n_experiments = sum(
                    1 for exp in var.experiments 
                    if exp in self.experiments_to_fit
                )
            else:
                n_experiments = 1
            
            # Get base values for one experiment
            base_values = self._get_base_initial_values(var)
            
            # Repeat for each experiment
            for _ in range(n_experiments):
                x0.extend(base_values)
        
        return x0
    
    def _get_base_initial_values(self, var: SelectedVariable) -> list[float]:
        """Get initial values for a single variable (for one experiment).
        
        Parameters
        ----------
        var : SelectedVariable
            The variable
            
        Returns
        -------
        list[float]
            Initial values (one per component for per-component params, else one)
        """
        values = []
        
        if var.initial_value is not None:
            # User-specified
            if var.parameter.per_component:
                if isinstance(var.initial_value, (list, np.ndarray)):
                    values.extend(var.initial_value)
                else:
                    values.extend([var.initial_value] * var.n_variables)
            else:
                values.append(float(var.initial_value))
        else:
            # Use value from column_binding or midpoint
            config_value = self._get_initial_value_from_config(var)
            
            if var.parameter.per_component and var.components:
                if isinstance(config_value, (list, np.ndarray)):
                    for idx in var._component_indices:
                        if idx < len(config_value):
                            values.append(float(config_value[idx]))
                        else:
                            values.append((var.lb + var.ub) / 2)
                else:
                    values.extend([(var.lb + var.ub) / 2] * var.n_variables)
            else:
                if config_value is not None:
                    values.append(float(config_value))
                else:
                    values.append((var.lb + var.ub) / 2)
        
        return values
    
    def _get_initial_value_from_config(self, var: SelectedVariable) -> Any:
        """Extract initial value from column_binding configuration."""
        param = var.parameter
        
        # Map parameter names to config attributes
        # This is a simplified mapping - may need extension for your specific config
        binding_params = getattr(self.column_binding, 'binding_parameters', {})
        column_params = getattr(self.column_binding, 'column_parameters', {})
        comp_binding = getattr(self.column_binding, 'component_binding_parameters', {})
        comp_column = getattr(self.column_binding, 'component_column_parameters', {})
        
        if param.per_component:
            if param.category in (ParameterCategory.BINDING_KINETIC, 
                                  ParameterCategory.BINDING_EQUILIBRIUM):
                return comp_binding.get(param.name)
            else:
                return comp_column.get(param.name)
        else:
            if param.category in (ParameterCategory.BINDING_KINETIC,
                                  ParameterCategory.BINDING_EQUILIBRIUM):
                return binding_params.get(param.name)
            else:
                return column_params.get(param.name)
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def validate(self) -> list[str]:
        """Validate the fitting problem configuration.
        
        Returns
        -------
        list[str]
            List of error messages (empty if valid)
        """
        errors = []
        
        if not self.selected_variables:
            errors.append("No variables selected for fitting")
        
        if not self.experiments_to_fit:
            errors.append("No experiments selected for fitting")
        
        # Check experiments have reference data
        for exp_name in self.experiments_to_fit:
            if exp_name not in self.reference_data:
                errors.append(f"Experiment '{exp_name}' has no reference data")
        
        # Check experiments exist
        exp_names = [e.name for e in self.experiments]
        for exp_name in self.experiments_to_fit:
            if exp_name not in exp_names:
                errors.append(f"Experiment '{exp_name}' not found")
        
        # Check valid metric
        valid_metrics = ["NRMSE", "SSE", "RMSE"]
        if self.metric not in valid_metrics:
            errors.append(f"Invalid metric '{self.metric}'. Valid: {valid_metrics}")
        
        return errors
    
    # =========================================================================
    # Build CADET-Process OptimizationProblem
    # =========================================================================
    
    def build(self) -> "OptimizationProblem":
        """Build CADET-Process OptimizationProblem.
        
        This is the main translation layer between cadet_simplified
        and CADET-Process optimization API.
        
        Returns
        -------
        OptimizationProblem
            Configured CADET-Process optimization problem
            
        Raises
        ------
        ValueError
            If validation fails
        """
        from CADETProcess.optimization import OptimizationProblem
        from CADETProcess.comparison import Comparator
        from CADETProcess.simulator import Cadet
        
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid fitting problem: {errors}")
        
        # Create optimization problem
        opt_problem = OptimizationProblem(self.name)
        
        # Add simulator as evaluator
        simulator = Cadet()
        opt_problem.add_evaluator(simulator)
        
        # Create processes and comparators for each experiment
        # Store process references for variable assignment
        processes: dict[str, Any] = {}
        
        for exp_name in self.experiments_to_fit:
            # Find experiment config
            exp_config = next(
                e for e in self.experiments if e.name == exp_name
            )
            
            # Create CADET-Process Process
            process = self.operation_mode.create_process(
                exp_config, self.column_binding
            )
            processes[exp_name] = process
            opt_problem.add_evaluation_object(process)
            
            # Create comparator with reference data
            reference = self.reference_data[exp_name]
            comparator = self._create_comparator(reference, process)
            
            opt_problem.add_objective(
                comparator,
                name=f"fit_{exp_name}",
                evaluation_objects=[process],
                n_objectives=comparator.n_metrics,
                requires=[simulator],
            )
        
        # Add optimization variables
        self._add_variables_to_problem(opt_problem, processes)
        
        return opt_problem
    
    def _add_variables_to_problem(
        self,
        opt_problem: "OptimizationProblem",
        processes: dict[str, Any],
    ) -> None:
        """Add all selected variables to the optimization problem.
        
        Handles both shared variables (same value across all experiments)
        and experiment-specific variables (different value per experiment).
        
        Parameters
        ----------
        opt_problem : OptimizationProblem
            The CADET-Process optimization problem
        processes : dict[str, Any]
            Mapping of experiment name to Process object
        """
        for var in self.selected_variables:
            if var.is_experiment_specific:
                # Create separate variables for each specified experiment
                for exp_name in var.experiments:
                    if exp_name not in self.experiments_to_fit:
                        continue
                    self._add_single_variable(
                        opt_problem,
                        var,
                        evaluation_objects=[processes[exp_name]],
                        experiment_name=exp_name,
                    )
            else:
                # Shared variable across all experiments
                all_processes = [processes[name] for name in self.experiments_to_fit]
                self._add_single_variable(
                    opt_problem,
                    var,
                    evaluation_objects=all_processes,
                    experiment_name=None,
                )
    
    def _add_single_variable(
        self,
        opt_problem: "OptimizationProblem",
        var: SelectedVariable,
        evaluation_objects: list[Any],
        experiment_name: str | None,
    ) -> None:
        """Add a single SelectedVariable to the optimization problem.
        
        Parameters
        ----------
        opt_problem : OptimizationProblem
            The CADET-Process optimization problem
        var : SelectedVariable
            The variable to add
        evaluation_objects : list
            Process objects this variable applies to
        experiment_name : str or None
            If provided, append to variable name for experiment-specific vars
        """
        # Determine transform - CADET-Process uses 'auto' directly
        transform = var.transform if var.transform != "none" else None
        
        if var.parameter.per_component:
            # One variable per selected component
            for comp_name, comp_idx in zip(
                var.components, var._component_indices
            ):
                # Build variable name
                var_name = f"{var.parameter.name}_{comp_name}"
                if experiment_name:
                    var_name = f"{var_name}_{experiment_name}"
                
                opt_problem.add_variable(
                    name=var_name,
                    parameter_path=var.parameter.parameter_path,
                    lb=var.lb,
                    ub=var.ub,
                    transform=transform,
                    indices=[comp_idx],  # FIX: Use indices parameter
                    evaluation_objects=evaluation_objects,  # FIX: Pass evaluation objects
                )
        else:
            # Scalar variable
            var_name = var.parameter.name
            if experiment_name:
                var_name = f"{var_name}_{experiment_name}"
            
            opt_problem.add_variable(
                name=var_name,
                parameter_path=var.parameter.parameter_path,
                lb=var.lb,
                ub=var.ub,
                transform=transform,
                evaluation_objects=evaluation_objects,  # FIX: Pass evaluation objects
            )
    
    def _create_comparator(
        self,
        reference: ReferenceDataConfig,
        process,
    ) -> "Comparator":
        """Create a Comparator for one experiment.
        
        Uses multi-component ReferenceIO for comparing all components
        in a single metric evaluation.
        
        Parameters
        ----------
        reference : ReferenceDataConfig
            Reference chromatogram data
        process : Process
            CADET-Process Process object (used to determine solution path)
            
        Returns
        -------
        Comparator
            Configured comparator for this experiment
        """
        from CADETProcess.comparison import Comparator
        
        comparator = Comparator(name=f"comparator_{reference.experiment_name}")
        
        # Determine components to compare
        components = self.comparison_components
        if components is None:
            components = list(reference.concentrations.keys())
        
        # Filter to components that exist in reference data
        available_components = [
            c for c in components if c in reference.concentrations
        ]
        
        if not available_components:
            raise ValueError(
                f"No valid comparison components for experiment "
                f"'{reference.experiment_name}'. "
                f"Requested: {components}, Available: {list(reference.concentrations.keys())}"
            )
        
        # Get solution path dynamically from process flow sheet
        solution_path = self._get_solution_path(process)
        
        # Get component indices for the comparison
        component_indices = [
            self._component_names.index(c) for c in available_components
        ]
        
        # Create multi-component ReferenceIO
        ref_io = reference.to_reference_io_multi(available_components)
        comparator.add_reference(ref_io)
        
        # Add difference metric with component specification
        comparator.add_difference_metric(
            self.metric,
            ref_io,
            solution_path,
            components=component_indices,  # FIX: Specify which components to compare
        )
        
        return comparator
    
    def _get_solution_path(self, process) -> str:
        """Determine the solution path from the process flow sheet.
        
        Looks for common outlet unit names in the flow sheet.
        
        Parameters
        ----------
        process : Process
            CADET-Process Process object
            
        Returns
        -------
        str
            Solution path (e.g., "outlet.outlet" or "column.outlet")
        """
        flow_sheet = process.flow_sheet
        
        # Try common outlet unit names
        outlet_candidates = ["outlet", "detector", "uv_detector"]
        for name in outlet_candidates:
            if hasattr(flow_sheet, name):
                return f"{name}.outlet"
        
        # Fallback: check for any unit with 'outlet' in the name
        for unit_name in flow_sheet.units:
            if "outlet" in unit_name.lower():
                return f"{unit_name}.outlet"
        
        # Last resort: use column outlet directly
        if hasattr(flow_sheet, "column"):
            return "column.outlet"
        
        # Default fallback
        return "outlet.outlet"
    
    # =========================================================================
    # Summary methods
    # =========================================================================
    
    def print_summary(self):
        """Print a summary of the current configuration."""
        print(f"FittingProblem: {self.name}")
        print(f"  Metric: {self.metric}")
        print(f"  Experiments to fit: {self.experiments_to_fit}")
        print(f"  Variables ({self.n_variables} total):")
        for var in self.selected_variables:
            parts = [f"[{var.lb}, {var.ub}]"]
            if var.components:
                parts.append(f"components={var.components}")
            if var.experiments:
                parts.append(f"experiments={var.experiments}")
            else:
                parts.append("(shared)")
            print(f"    - {var.parameter.name} {' '.join(parts)}")
    
    @property
    def n_variables(self) -> int:
        """Total number of optimization variables.
        
        Accounts for:
        - Per-component expansion (one var per component)
        - Experiment-specific variables (one var per experiment)
        """
        total = 0
        for var in self.selected_variables:
            n_per_exp = var.n_variables  # Components expansion
            
            if var.is_experiment_specific:
                # Count only experiments that are in experiments_to_fit
                n_experiments = sum(
                    1 for exp in var.experiments
                    if exp in self.experiments_to_fit
                )
            else:
                n_experiments = 1  # Shared variable counts once
            
            total += n_per_exp * n_experiments
        
        return total
    
    def __repr__(self) -> str:
        return (
            f"FittingProblem({self.name}, "
            f"{len(self.selected_variables)} variables, "
            f"{len(self.experiments_to_fit)} experiments)"
        )
