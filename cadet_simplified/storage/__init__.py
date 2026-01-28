"""Storage for experiment results.

Provides file-based storage for experiments and simulation results.

Example:
    >>> from cadet_simplified.storage import FileStorage
    >>> storage = FileStorage("./experiments")
    >>> 
    >>> # Save after simulation
    >>> set_id = storage.save_experiment_set(
    ...     name="IEX Screening",
    ...     operation_mode="LWE_concentration_based",
    ...     experiments=configs,
    ...     column_binding=col_bind,
    ...     results=sim_results,
    ... )
    >>> 
    >>> # Load for analysis
    >>> loaded = storage.load_results(set_id)
"""

from .file_storage import (
    FileStorage,
    LoadedExperiment,
    ExperimentInfo,
)

__all__ = [
    "FileStorage",
    "LoadedExperiment",
    "ExperimentInfo",
]
