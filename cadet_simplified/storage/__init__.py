"""Experiment and results storage.

Two storage backends are available:

1. ExperimentStore (legacy): Config-only JSON storage
2. FileResultsStorage (recommended): Full storage including pickled results,
   interpolated chromatograms, and H5 files

Example usage with FileResultsStorage:
    from cadet_simplified.storage import FileResultsStorage
    storage = FileResultsStorage("./experiments")
    
    # Save after simulation
    set_id = storage.save_experiment_set(
        name="IEX Screening",
        operation_mode="LWE_concentration_based",
        experiments=configs,
        column_binding=col_bind,
        results=sim_results,
     )
    
    # List for UI
    experiments_df = storage.list_experiments(limit=25)
    
    # Load for analysis
    loaded = storage.load_results_by_selection([
         (set_id, "experiment_1"),
         (set_id, "experiment_2"),
     ], n_workers=4)
"""

from .experiment_store import (
    ExperimentStore,
    ExperimentSet,
    StoredExperiment,
    StoredColumnBinding,
)

from .interfaces import (
    ResultsStorageInterface,
    StoredExperimentInfo,
    LoadedExperiment,
)

from .file_storage import (
    FileResultsStorage,
)

__all__ = [
    # Interface
    'ResultsStorageInterface',
    'StoredExperimentInfo',
    'LoadedExperiment',
    # File-based implementation (recommended)
    'FileResultsStorage',
    # Legacy (config-only)
    'ExperimentStore',
    'ExperimentSet',
    'StoredExperiment',
    'StoredColumnBinding',
]
