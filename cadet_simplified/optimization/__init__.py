"""Optimization module for parameter fitting.

This module provides classes for fitting chromatography model parameters
to experimental reference data using CADET-Process optimization.

Example
-------
    from cadet_simplified.optimization import (
        FittingProblem,
        FittingRunner,
        FittingResult,
    )
    
    # Create fitting problem
    problem = FittingProblem(
        name="my_fit",
        operation_mode=mode,
        experiments=experiments,
        column_binding=column_binding,
        reference_data=reference_data,
    )
    
    # Add variables to fit (using CADET-Process parameter names)
    problem.add_variable("bed_porosity")
    problem.add_variable("adsorption_rate", components=["Product"])
    
    # Experiment-specific variables
    problem.add_variable(
        "desorption_rate",
        components=["Product"],
        experiments=["exp_1", "exp_2"]  # Different values per experiment
    )
    
    # Run fitting
    runner = FittingRunner()
    result = runner.run(problem, max_iterations=100)
    
    # Inspect results
    result.print_summary()
    result.plot_comparison("experiment_1")
"""

from .fittable_parameters import (
    ParameterCategory,
    FittableParameter,
    SelectedVariable,
    BINDING_MODEL_PARAMETERS,
    COLUMN_MODEL_PARAMETERS,
    get_column_transport_parameters,
    get_pore_model_parameters,
    get_sma_binding_parameters,
    get_langmuir_binding_parameters,
)

from .reference_data import ReferenceDataConfig

from .reference_parser import (
    parse_reference_sheets,
    generate_reference_template,
)

from .fitting_problem import FittingProblem

from .fitting_runner import FittingRunner

from .fitting_result import FittingResult

from .fitting_storage import (
    FittingStorage,
    FittingRunInfo,
    export_fitting_result_to_excel,
)


__all__ = [
    # Fittable parameters
    "ParameterCategory",
    "FittableParameter",
    "SelectedVariable",
    "BINDING_MODEL_PARAMETERS",
    "COLUMN_MODEL_PARAMETERS",
    "get_column_transport_parameters",
    "get_pore_model_parameters",
    "get_sma_binding_parameters",
    "get_langmuir_binding_parameters",
    # Reference data
    "ReferenceDataConfig",
    "parse_reference_sheets",
    "generate_reference_template",
    # Fitting
    "FittingProblem",
    "FittingRunner",
    "FittingResult",
    # Storage
    "FittingStorage",
    "FittingRunInfo",
    "export_fitting_result_to_excel",
]
