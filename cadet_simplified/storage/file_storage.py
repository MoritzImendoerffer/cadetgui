"""File-based storage for experiment results.

Simplified storage structure:
    {storage_dir}/{experiment_set_id}/
    ├── config.json              # Metadata + ExperimentConfig + ColumnBindingConfig
    ├── results/
    │   └── {exp_name}.pkl       # Pickled SimulationResultWrapper
    └── chromatograms/
        └── {exp_name}.parquet   # Cached interpolated chromatogram

Usage:
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
    >>> # List for UI
    >>> df = storage.list_experiments(limit=25)
    >>> 
    >>> # Load for analysis
    >>> loaded = storage.load_results(set_id)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING
import hashlib
import json
import pickle
import shutil

import pandas as pd

from ..core import ExperimentConfig, ColumnBindingConfig, ComponentDefinition
from ..plotting import interpolate_chromatogram

if TYPE_CHECKING:
    from ..simulation.runner import SimulationResultWrapper


# =============================================================================
# Data containers for loaded data
# =============================================================================

@dataclass
class LoadedExperiment:
    """A fully loaded experiment with results.
    
    Contains everything needed for analysis.
    
    Attributes
    ----------
    experiment_set_id : str
        ID of the experiment set
    experiment_set_name : str
        Human-readable name of the experiment set
    experiment_name : str
        Name of this specific experiment
    result : SimulationResultWrapper
        Full simulation result
    experiment_config : ExperimentConfig
        Experiment configuration
    column_binding : ColumnBindingConfig
        Column and binding configuration
    chromatogram_df : pd.DataFrame, optional
        Cached interpolated chromatogram
    """
    experiment_set_id: str
    experiment_set_name: str
    experiment_name: str
    result: "SimulationResultWrapper"
    experiment_config: ExperimentConfig
    column_binding: ColumnBindingConfig
    chromatogram_df: pd.DataFrame | None = None


@dataclass
class ExperimentInfo:
    """Metadata about a stored experiment (for listing/browsing).
    
    Lightweight - doesn't load the actual results.
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
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for DataFrame."""
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
        }


# =============================================================================
# Helper functions
# =============================================================================

def _sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    invalid_chars = '<>:"/\\|?*'
    result = name
    for char in invalid_chars:
        result = result.replace(char, '_')
    result = result.strip('. ')
    return result[:100] if len(result) > 100 else (result or "unnamed")


def _generate_id(name: str) -> str:
    """Generate a unique ID from name and timestamp."""
    timestamp = datetime.now().isoformat()
    id_string = f"{name}_{timestamp}"
    return hashlib.md5(id_string.encode()).hexdigest()[:12]


# =============================================================================
# Main storage class
# =============================================================================

class FileStorage:
    """File-based storage for experiment results.
    
    Parameters
    ----------
    storage_dir : str or Path
        Base directory for all stored experiments
    n_interpolation_points : int, default=2000
        Number of points for chromatogram interpolation
    """
    
    def __init__(
        self,
        storage_dir: str | Path,
        n_interpolation_points: int = 2000,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.n_interpolation_points = n_interpolation_points
    
    def _get_set_dir(self, experiment_set_id: str) -> Path:
        """Get directory for an experiment set."""
        return self.storage_dir / experiment_set_id
    
    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------
    
    def save_experiment_set(
        self,
        name: str,
        operation_mode: str,
        experiments: list[ExperimentConfig],
        column_binding: ColumnBindingConfig,
        results: list["SimulationResultWrapper"],
        description: str = "",
    ) -> str:
        """Save a complete experiment set with results.
        
        Parameters
        ----------
        name : str
            Human-readable name
        operation_mode : str
            Operation mode used
        experiments : list[ExperimentConfig]
            Experiment configurations
        column_binding : ColumnBindingConfig
            Column and binding configuration
        results : list[SimulationResultWrapper]
            Simulation results
        description : str, optional
            Description
            
        Returns
        -------
        str
            Generated experiment_set_id
        """
        # Generate ID and create directories
        set_id = _generate_id(name)
        set_dir = self._get_set_dir(set_id)
        
        (set_dir / "results").mkdir(parents=True, exist_ok=True)
        (set_dir / "chromatograms").mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Build config.json
        config_data = {
            "id": set_id,
            "name": name,
            "operation_mode": operation_mode,
            "description": description,
            "column_binding": column_binding.to_dict(),
            "experiments": [exp.to_dict() for exp in experiments],
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        
        # Save config
        with open(set_dir / "config.json", "w") as f:
            json.dump(config_data, f, indent=2)
        
        # Save results
        for exp, result in zip(experiments, results):
            if not result.success:
                continue
            
            safe_name = _sanitize_filename(exp.name)
            
            # Pickle the full result
            with open(set_dir / "results" / f"{safe_name}.pkl", "wb") as f:
                pickle.dump(result, f)
            
            # Cache interpolated chromatogram
            if result.cadet_result is not None:
                try:
                    chrom_df = interpolate_chromatogram(
                        result,
                        n_points=self.n_interpolation_points
                    )
                    chrom_df.to_parquet(
                        set_dir / "chromatograms" / f"{safe_name}.parquet",
                        index=False,
                    )
                except Exception as e:
                    print(f"Warning: Could not cache chromatogram for {exp.name}: {e}")
        
        return set_id
    
    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------
    
    def load_results(
        self,
        experiment_set_id: str,
        experiment_names: list[str] | None = None,
        include_chromatogram: bool = True,
    ) -> list[LoadedExperiment]:
        """Load results for an experiment set.
        
        Parameters
        ----------
        experiment_set_id : str
            ID of the experiment set
        experiment_names : list[str], optional
            Specific experiments to load. If None, loads all.
        include_chromatogram : bool, default=True
            Whether to load cached chromatogram
            
        Returns
        -------
        list[LoadedExperiment]
            Loaded experiments
        """
        set_dir = self._get_set_dir(experiment_set_id)
        config_path = set_dir / "config.json"
        
        if not config_path.exists():
            return []
        
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        # Build column binding
        column_binding = ColumnBindingConfig.from_dict(config_data["column_binding"])
        
        # Load experiments
        loaded = []
        
        for exp_dict in config_data["experiments"]:
            exp_name = exp_dict["name"]
            
            # Filter if requested
            if experiment_names is not None and exp_name not in experiment_names:
                continue
            
            safe_name = _sanitize_filename(exp_name)
            pkl_path = set_dir / "results" / f"{safe_name}.pkl"
            
            if not pkl_path.exists():
                continue
            
            # Load result
            with open(pkl_path, "rb") as f:
                result = pickle.load(f)
            
            # Load chromatogram if requested
            chromatogram_df = None
            if include_chromatogram:
                chrom_path = set_dir / "chromatograms" / f"{safe_name}.parquet"
                if chrom_path.exists():
                    chromatogram_df = pd.read_parquet(chrom_path)
            
            # Build ExperimentConfig
            experiment_config = ExperimentConfig.from_dict(exp_dict)
            
            loaded.append(LoadedExperiment(
                experiment_set_id=experiment_set_id,
                experiment_set_name=config_data["name"],
                experiment_name=exp_name,
                result=result,
                experiment_config=experiment_config,
                column_binding=column_binding,
                chromatogram_df=chromatogram_df,
            ))
        
        return loaded
    
    def load_results_by_selection(
        self,
        selections: list[tuple[str, str]],
        include_chromatogram: bool = True,
    ) -> list[LoadedExperiment]:
        """Load results for a selection across multiple experiment sets.
        
        Parameters
        ----------
        selections : list[tuple[str, str]]
            List of (experiment_set_id, experiment_name) pairs
        include_chromatogram : bool, default=True
            Whether to load cached chromatogram
            
        Returns
        -------
        list[LoadedExperiment]
            Loaded experiments
        """
        # Group by experiment set
        by_set: dict[str, list[str]] = {}
        for set_id, exp_name in selections:
            if set_id not in by_set:
                by_set[set_id] = []
            by_set[set_id].append(exp_name)
        
        # Load from each set
        all_loaded = []
        for set_id, exp_names in by_set.items():
            loaded = self.load_results(
                set_id,
                experiment_names=exp_names,
                include_chromatogram=include_chromatogram,
            )
            all_loaded.extend(loaded)
        
        return all_loaded
    
    def get_chromatogram(
        self,
        experiment_set_id: str,
        experiment_name: str,
    ) -> pd.DataFrame | None:
        """Load just the cached chromatogram (fast, no unpickling).
        
        Parameters
        ----------
        experiment_set_id : str
            ID of the experiment set
        experiment_name : str
            Name of the experiment
            
        Returns
        -------
        pd.DataFrame or None
            Chromatogram DataFrame, or None if not found
        """
        set_dir = self._get_set_dir(experiment_set_id)
        safe_name = _sanitize_filename(experiment_name)
        chrom_path = set_dir / "chromatograms" / f"{safe_name}.parquet"
        
        if chrom_path.exists():
            return pd.read_parquet(chrom_path)
        return None
    
    # -------------------------------------------------------------------------
    # Listing
    # -------------------------------------------------------------------------
    
    def list_experiments(
        self,
        limit: int = 25,
        experiment_set_id: str | None = None,
    ) -> pd.DataFrame:
        """List stored experiments as a flat table.
        
        Parameters
        ----------
        limit : int, default=25
            Maximum number of experiments
        experiment_set_id : str, optional
            Filter to specific set
            
        Returns
        -------
        pd.DataFrame
            Table of experiment info
        """
        all_experiments: list[ExperimentInfo] = []
        
        # Get experiment set directories
        if experiment_set_id is not None:
            set_dirs = [self._get_set_dir(experiment_set_id)]
        else:
            set_dirs = [d for d in self.storage_dir.iterdir() if d.is_dir()]
        
        for set_dir in set_dirs:
            config_path = set_dir / "config.json"
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, "r") as f:
                    config_data = json.load(f)
                
                set_id = config_data["id"]
                set_name = config_data["name"]
                operation_mode = config_data["operation_mode"]
                col_bind = config_data["column_binding"]
                created_at = config_data["created_at"]
                
                for exp_dict in config_data["experiments"]:
                    safe_name = _sanitize_filename(exp_dict["name"])
                    
                    # Check what files exist
                    has_results = (set_dir / "results" / f"{safe_name}.pkl").exists()
                    has_chromatogram = (set_dir / "chromatograms" / f"{safe_name}.parquet").exists()
                    
                    # Extract component info
                    components = exp_dict.get("components", [])
                    n_components = len(components)
                    component_names = [c.get("name", f"comp_{i}") for i, c in enumerate(components)]
                    
                    exp_info = ExperimentInfo(
                        experiment_set_id=set_id,
                        experiment_set_name=set_name,
                        experiment_name=exp_dict["name"],
                        created_at=datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at,
                        n_components=n_components,
                        component_names=component_names,
                        operation_mode=operation_mode,
                        column_model=col_bind["column_model"],
                        binding_model=col_bind["binding_model"],
                        has_results=has_results,
                        has_chromatogram=has_chromatogram,
                    )
                    all_experiments.append(exp_info)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read {config_path}: {e}")
                continue
        
        # Sort by created_at descending and limit
        all_experiments.sort(key=lambda x: x.created_at, reverse=True)
        all_experiments = all_experiments[:limit]
        
        # Convert to DataFrame
        if not all_experiments:
            return pd.DataFrame(columns=[
                "experiment_set_id", "experiment_set_name", "experiment_name",
                "created_at", "n_components", "component_names", "operation_mode",
                "column_model", "binding_model", "has_results", "has_chromatogram",
            ])
        
        return pd.DataFrame([e.to_dict() for e in all_experiments])
    
    def list_experiment_sets(self) -> list[dict[str, Any]]:
        """List all experiment sets (metadata only).
        
        Returns
        -------
        list[dict]
            List of experiment set metadata
        """
        results = []
        
        for set_dir in self.storage_dir.iterdir():
            if not set_dir.is_dir():
                continue
            
            config_path = set_dir / "config.json"
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                
                results.append({
                    "id": data["id"],
                    "name": data["name"],
                    "operation_mode": data["operation_mode"],
                    "description": data.get("description", ""),
                    "n_experiments": len(data["experiments"]),
                    "created_at": data["created_at"],
                    "updated_at": data["updated_at"],
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        # Sort by updated_at descending
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        return results
    
    # -------------------------------------------------------------------------
    # Deletion
    # -------------------------------------------------------------------------
    
    def delete_experiment_set(self, experiment_set_id: str) -> bool:
        """Delete an experiment set.
        
        Parameters
        ----------
        experiment_set_id : str
            ID to delete
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        set_dir = self._get_set_dir(experiment_set_id)
        
        if set_dir.exists():
            shutil.rmtree(set_dir)
            return True
        return False
