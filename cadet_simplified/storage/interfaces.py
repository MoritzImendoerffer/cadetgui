"""Abstract interfaces for experiment and results storage.

Defines the contract for storage backends, enabling future migration
from file-based storage to databases or other systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..simulation.runner import SimulationResultWrapper
    from ..operation_modes import ExperimentConfig, ColumnBindingConfig


@dataclass
class StoredExperimentInfo:
    """Metadata about a stored experiment (without loading full results).
    
    Used for listing/browsing experiments in the UI.
    """
    experiment_set_id: str
    experiment_set_name: str
    experiment_name: str
    created_at: datetime
    n_components: int
    component_names: list[str]
    operation_mode: str
    column_model: str
    binding_model: str
    has_results: bool = False
    has_chromatogram: bool = False
    has_h5: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame/Tabulator."""
        return {
            "experiment_set_id": self.experiment_set_id,
            "experiment_set_name": self.experiment_set_name,
            "experiment_name": self.experiment_name,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "n_components": self.n_components,
            "component_names": ", ".join(self.component_names),
            "operation_mode": self.operation_mode,
            "column_model": self.column_model,
            "binding_model": self.binding_model,
            "has_results": self.has_results,
            "has_chromatogram": self.has_chromatogram,
            "has_h5": self.has_h5,
        }


@dataclass
class LoadedExperiment:
    """A fully loaded experiment with results.
    
    Contains everything needed for analysis.
    """
    experiment_set_id: str
    experiment_set_name: str
    experiment_name: str
    result: "SimulationResultWrapper"
    experiment_config: "ExperimentConfig"
    column_binding: "ColumnBindingConfig"
    chromatogram_df: pd.DataFrame | None = None  # [time, comp_0, comp_1, ...]


class ResultsStorageInterface(ABC):
    """Abstract interface for storing and retrieving simulation results.
    
    Implementations can be file-based, database-backed, or cloud storage.
    The interface ensures consistent behavior across backends.
    
    Storage includes:
    - Experiment configurations (ExperimentConfig, ColumnBindingConfig)
    - Simulation results (pickled SimulationResultWrapper)
    - Interpolated chromatograms (for expert workflows)
    - H5 files (CADET native format)
    """
    
    @abstractmethod
    def save_experiment_set(
        self,
        name: str,
        operation_mode: str,
        experiments: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        results: list["SimulationResultWrapper"],
        description: str = "",
    ) -> str:
        """Save a complete experiment set with results.
        
        Parameters
        ----------
        name : str
            Human-readable name for the experiment set
        operation_mode : str
            Name of the operation mode used
        experiments : list[ExperimentConfig]
            List of experiment configurations
        column_binding : ColumnBindingConfig
            Shared column and binding configuration
        results : list[SimulationResultWrapper]
            Simulation results (one per experiment)
        description : str, optional
            Optional description
            
        Returns
        -------
        str
            The generated experiment_set_id
        """
        pass
    
    @abstractmethod
    def load_results(
        self,
        experiment_set_id: str,
        experiment_names: list[str] | None = None,
        n_workers: int = 1,
    ) -> list[LoadedExperiment]:
        """Load results for specific experiments.
        
        Parameters
        ----------
        experiment_set_id : str
            ID of the experiment set
        experiment_names : list[str], optional
            Specific experiments to load. If None, loads all.
        n_workers : int, default=1
            Number of parallel workers for loading
            
        Returns
        -------
        list[LoadedExperiment]
            Loaded experiments with results
        """
        pass
    
    @abstractmethod
    def load_results_by_selection(
        self,
        selections: list[tuple[str, str]],  # [(experiment_set_id, experiment_name), ...]
        n_workers: int = 1,
    ) -> list[LoadedExperiment]:
        """Load results for a selection across multiple experiment sets.
        
        Parameters
        ----------
        selections : list[tuple[str, str]]
            List of (experiment_set_id, experiment_name) pairs
        n_workers : int, default=1
            Number of parallel workers for loading
            
        Returns
        -------
        list[LoadedExperiment]
            Loaded experiments with results
        """
        pass
    
    @abstractmethod
    def list_experiments(
        self,
        limit: int = 25,
        experiment_set_id: str | None = None,
    ) -> pd.DataFrame:
        """List all stored experiments as a flat table.
        
        Parameters
        ----------
        limit : int, default=25
            Maximum number of experiments to return
        experiment_set_id : str, optional
            Filter to specific experiment set
            
        Returns
        -------
        pd.DataFrame
            Table with columns: experiment_set_id, experiment_set_name,
            experiment_name, created_at, n_components, operation_mode,
            column_model, binding_model, has_results, has_chromatogram, has_h5
        """
        pass
    
    @abstractmethod
    def list_experiment_sets(self) -> list[dict[str, Any]]:
        """List all experiment sets (metadata only).
        
        Returns
        -------
        list[dict]
            List of experiment set metadata
        """
        pass
    
    @abstractmethod
    def delete_experiment_set(self, experiment_set_id: str) -> bool:
        """Delete an experiment set and all its data.
        
        Parameters
        ----------
        experiment_set_id : str
            ID of the experiment set to delete
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def get_chromatogram(
        self,
        experiment_set_id: str,
        experiment_name: str,
    ) -> pd.DataFrame | None:
        """Load just the chromatogram data (without full results).
        
        Useful for quick visualization without unpickling.
        
        Parameters
        ----------
        experiment_set_id : str
            ID of the experiment set
        experiment_name : str
            Name of the experiment
            
        Returns
        -------
        pd.DataFrame or None
            Chromatogram with columns [time, comp_0, comp_1, ...],
            or None if not found
        """
        pass
