"""CADET Simplified - Excel-based chromatography simulation.

A simplified interface for CADET-Process simulations using Excel templates.

Example workflow:
    1. Select operation mode, column model, binding model, and components
    2. Download Excel template
    3. Fill in experiment parameters and column/binding parameters
    4. Upload filled template
    5. Validate configuration
    6. Run simulations
    7. View results

Quick start:
    >>> from cadet_simplified import get_lwe_mode, generate_template
    >>> mode = get_lwe_mode()
    >>> generate_template(
    ...     operation_mode=mode,
    ...     column_model="GeneralRateModel",
    ...     binding_model="StericMassAction",
    ...     n_components=4,
    ...     component_names=["Salt", "Product", "Impurity1", "Impurity2"],
    ...     output_path="template.xlsx",
    ... )
"""

from .operation_modes import (
    # Base classes
    BaseOperationMode,
    ParameterDefinition,
    ParameterType,
    ExperimentConfig,
    ColumnBindingConfig,
    ComponentDefinition,
    # Registries
    SUPPORTED_COLUMN_MODELS,
    SUPPORTED_BINDING_MODELS,
    OPERATION_MODES,
    # Implementations
    LWEConcentrationBased,
    get_lwe_mode,
    get_operation_mode,
)

from .excel import (
    ExcelTemplateGenerator,
    generate_template,
    ExcelParser,
    ParseResult,
    parse_excel,
)

from .storage import (
    ExperimentStore,
    ExperimentSet,
)

from .simulation import (
    SimulationRunner,
    SimulationResult,
    ValidationResult,
    validate_and_report,
)

__version__ = "0.1.0"

__all__ = [
    # Operation modes
    'BaseOperationMode',
    'ParameterDefinition',
    'ParameterType',
    'ExperimentConfig',
    'ColumnBindingConfig',
    'ComponentDefinition',
    'SUPPORTED_COLUMN_MODELS',
    'SUPPORTED_BINDING_MODELS',
    'OPERATION_MODES',
    'LWEConcentrationBased',
    'get_lwe_mode',
    'get_operation_mode',
    # Excel
    'ExcelTemplateGenerator',
    'generate_template',
    'ExcelParser',
    'ParseResult',
    'parse_excel',
    # Storage
    'ExperimentStore',
    'ExperimentSet',
    # Simulation
    'SimulationRunner',
    'SimulationResult',
    'ValidationResult',
    'validate_and_report',
]
