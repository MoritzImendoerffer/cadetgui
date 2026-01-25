"""File-based storage implementation for experiment results.

Stores experiments in a structured folder hierarchy:
    {storage_dir}/{experiment_set_id}/
    ├── config.json              # ExperimentSet metadata + configs
    ├── chromatograms/
    │   ├── {exp_name}.parquet   # Interpolated chromatogram [time, comp_0, ...]
    │   └── ...
    ├── results/
    │   ├── {exp_name}.pkl       # Pickled SimulationResultWrapper
    │   └── ...
    └── h5/
        ├── {exp_name}.h5        # CADET H5 files
        └── ...
"""

import hashlib
import json
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from .interfaces import (
    ResultsStorageInterface,
    StoredExperimentInfo,
    LoadedExperiment,
)
from .experiment_store import (
    StoredExperiment,
    StoredColumnBinding,
    ExperimentSet,
)

if TYPE_CHECKING:
    from ..simulation.runner import SimulationResultWrapper
    from ..operation_modes import ExperimentConfig, ColumnBindingConfig


def _sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename."""
    invalid_chars = '<>:"/\\|?*'
    result = name
    for char in invalid_chars:
        result = result.replace(char, '_')
    result = result.strip('. ')
    if len(result) > 100:
        result = result[:100]
    return result or "unnamed"


def _load_single_result(args: tuple[Path, str, str, dict, dict]) -> LoadedExperiment | None:
    """Load a single experiment result (for parallel execution).
    
    Parameters
    ----------
    args : tuple
        (pkl_path, experiment_set_id, experiment_set_name, exp_config_dict, column_binding_dict)
        
    Returns
    -------
    LoadedExperiment or None
        Loaded experiment, or None if loading fails
    """
    from ..operation_modes import ExperimentConfig, ColumnBindingConfig, ComponentDefinition
    
    pkl_path, experiment_set_id, experiment_set_name, exp_config_dict, col_bind_dict = args
    
    try:
        # Load pickled result
        with open(pkl_path, 'rb') as f:
            result = pickle.load(f)
        
        # Reconstruct ExperimentConfig
        components = [
            ComponentDefinition(
                name=c["name"],
                is_salt=c.get("is_salt", False),
                molecular_weight=c.get("molecular_weight"),
            )
            for c in exp_config_dict["components"]
        ]
        experiment_config = ExperimentConfig(
            name=exp_config_dict["name"],
            parameters=exp_config_dict["parameters"],
            components=components,
        )
        
        # Reconstruct ColumnBindingConfig
        column_binding = ColumnBindingConfig(
            column_model=col_bind_dict["column_model"],
            binding_model=col_bind_dict["binding_model"],
            column_parameters=col_bind_dict["column_parameters"],
            binding_parameters=col_bind_dict["binding_parameters"],
            component_column_parameters=col_bind_dict.get("component_column_parameters", {}),
            component_binding_parameters=col_bind_dict.get("component_binding_parameters", {}),
        )
        
        # Try to load chromatogram
        chrom_path = pkl_path.parent.parent / "chromatograms" / f"{_sanitize_filename(exp_config_dict['name'])}.parquet"
        chromatogram_df = None
        if chrom_path.exists():
            chromatogram_df = pd.read_parquet(chrom_path)
        
        return LoadedExperiment(
            experiment_set_id=experiment_set_id,
            experiment_set_name=experiment_set_name,
            experiment_name=exp_config_dict["name"],
            result=result,
            experiment_config=experiment_config,
            column_binding=column_binding,
            chromatogram_df=chromatogram_df,
        )
        
    except Exception as e:
        print(f"Error loading {pkl_path}: {e}")
        return None


class FileResultsStorage(ResultsStorageInterface):
    """File-based implementation of results storage.
    
    Example:
        >>> storage = FileResultsStorage("./experiments")
        >>> 
        >>> # After simulation
        >>> set_id = storage.save_experiment_set(
        ...     name="IEX Screening",
        ...     operation_mode="LWE_concentration_based",
        ...     experiments=configs,
        ...     column_binding=col_bind,
        ...     results=sim_results,
        ... )
        >>> 
        >>> # Later, for analysis
        >>> experiments_df = storage.list_experiments(limit=25)
        >>> loaded = storage.load_results_by_selection([
        ...     (set_id, "experiment_1"),
        ...     (set_id, "experiment_2"),
        ... ])
    """
    
    def __init__(
        self,
        storage_dir: str | Path,
        n_interpolation_points: int = 500,
    ):
        """Initialize file-based storage.
        
        Parameters
        ----------
        storage_dir : str or Path
            Base directory for all stored experiments
        n_interpolation_points : int, default=500
            Number of points for chromatogram interpolation
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.n_interpolation_points = n_interpolation_points
    
    def _get_set_dir(self, experiment_set_id: str) -> Path:
        """Get directory for an experiment set."""
        return self.storage_dir / experiment_set_id
    
    def _generate_id(self, name: str) -> str:
        """Generate a unique ID for an experiment set."""
        timestamp = datetime.now().isoformat()
        id_string = f"{name}_{timestamp}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    def _interpolate_chromatogram(
        self,
        result: "SimulationResultWrapper",
        n_points: int | None = None,
    ) -> pd.DataFrame | None:
        """Interpolate chromatogram from simulation result.
        
        Returns DataFrame with columns [time, comp_0, comp_1, ...]
        """
        if result.cadet_result is None:
            return None
        
        n_points = n_points or self.n_interpolation_points
        
        try:
            process = result.cadet_result.process
            product_outlets = process.flow_sheet.product_outlets
            
            if not product_outlets:
                return None
            
            product_outlet = product_outlets[0]
            outlet_solution = result.cadet_result.solution[product_outlet.name]
            
            time_complete = result.cadet_result.time_complete
            time_interp = np.linspace(
                float(time_complete.min()),
                float(time_complete.max()),
                n_points,
            )
            
            interp_func = outlet_solution.outlet.solution_interpolated
            solution_interp = interp_func(time_interp)
            
            # Build DataFrame with [time, comp_0, comp_1, ...]
            data = {"time": time_interp}
            for i, comp in enumerate(process.component_system.components):
                comp_name = comp.name if hasattr(comp, 'name') else f"comp_{i}"
                data[comp_name] = solution_interp[:, i]
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"Warning: Failed to interpolate chromatogram: {e}")
            return None
    
    def save_experiment_set(
        self,
        name: str,
        operation_mode: str,
        experiments: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        results: list["SimulationResultWrapper"],
        description: str = "",
    ) -> str:
        """Save a complete experiment set with results."""
        # Generate ID and create directory structure
        set_id = self._generate_id(name)
        set_dir = self._get_set_dir(set_id)
        
        (set_dir / "chromatograms").mkdir(parents=True, exist_ok=True)
        (set_dir / "results").mkdir(parents=True, exist_ok=True)
        (set_dir / "h5").mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        
        # Convert configs to storable format
        stored_experiments = []
        for exp in experiments:
            stored_exp = StoredExperiment.from_experiment_config(exp)
            stored_experiments.append(asdict(stored_exp))
        
        stored_column_binding = StoredColumnBinding.from_column_binding_config(column_binding)
        
        # Build config.json
        config_data = {
            "id": set_id,
            "name": name,
            "operation_mode": operation_mode,
            "description": description,
            "column_binding": asdict(stored_column_binding),
            "experiments": stored_experiments,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        
        # Save config
        config_path = set_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save results for each experiment
        for exp, result in zip(experiments, results):
            if not result.success:
                continue
            
            safe_name = _sanitize_filename(exp.name)
            
            # 1. Pickle the full result
            pkl_path = set_dir / "results" / f"{safe_name}.pkl"
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)
            
            # 2. Save interpolated chromatogram as parquet
            chrom_df = self._interpolate_chromatogram(result)
            if chrom_df is not None:
                chrom_path = set_dir / "chromatograms" / f"{safe_name}.parquet"
                chrom_df.to_parquet(chrom_path, index=False)
            
            # 3. Copy H5 file if it exists
            if result.h5_path is not None and result.h5_path.exists():
                h5_dest = set_dir / "h5" / f"{safe_name}.h5"
                if result.h5_path != h5_dest:
                    shutil.copy2(result.h5_path, h5_dest)
        
        return set_id
    
    def load_results(
        self,
        experiment_set_id: str,
        experiment_names: list[str] | None = None,
        n_workers: int = 1,
    ) -> list[LoadedExperiment]:
        """Load results for specific experiments."""
        set_dir = self._get_set_dir(experiment_set_id)
        config_path = set_dir / "config.json"
        
        if not config_path.exists():
            return []
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Build list of experiments to load
        experiments_to_load = config_data["experiments"]
        if experiment_names is not None:
            experiments_to_load = [
                e for e in experiments_to_load
                if e["name"] in experiment_names
            ]
        
        # Prepare loading args
        load_args = []
        for exp_config in experiments_to_load:
            safe_name = _sanitize_filename(exp_config["name"])
            pkl_path = set_dir / "results" / f"{safe_name}.pkl"
            
            if pkl_path.exists():
                load_args.append((
                    pkl_path,
                    experiment_set_id,
                    config_data["name"],
                    exp_config,
                    config_data["column_binding"],
                ))
        
        if not load_args:
            return []
        
        # Load results (parallel or sequential)
        if n_workers == 1:
            results = [_load_single_result(args) for args in load_args]
        else:
            # Use ThreadPoolExecutor for I/O-bound pickle loading
            results = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_load_single_result, args) for args in load_args]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        
        # Filter None results
        return [r for r in results if r is not None]
    
    def load_results_by_selection(
        self,
        selections: list[tuple[str, str]],
        n_workers: int = 1,
    ) -> list[LoadedExperiment]:
        """Load results for a selection across multiple experiment sets."""
        # Group by experiment set
        by_set: dict[str, list[str]] = {}
        for set_id, exp_name in selections:
            if set_id not in by_set:
                by_set[set_id] = []
            by_set[set_id].append(exp_name)
        
        # Prepare all loading args
        all_load_args = []
        
        for set_id, exp_names in by_set.items():
            set_dir = self._get_set_dir(set_id)
            config_path = set_dir / "config.json"
            
            if not config_path.exists():
                continue
            
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Find matching experiments
            for exp_config in config_data["experiments"]:
                if exp_config["name"] in exp_names:
                    safe_name = _sanitize_filename(exp_config["name"])
                    pkl_path = set_dir / "results" / f"{safe_name}.pkl"
                    
                    if pkl_path.exists():
                        all_load_args.append((
                            pkl_path,
                            set_id,
                            config_data["name"],
                            exp_config,
                            config_data["column_binding"],
                        ))
        
        if not all_load_args:
            return []
        
        # Load results
        if n_workers == 1:
            results = [_load_single_result(args) for args in all_load_args]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(_load_single_result, args) for args in all_load_args]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        
        return [r for r in results if r is not None]
    
    def list_experiments(
        self,
        limit: int = 25,
        experiment_set_id: str | None = None,
    ) -> pd.DataFrame:
        """List all stored experiments as a flat table."""
        all_experiments: list[StoredExperimentInfo] = []
        
        # Get all experiment set directories
        if experiment_set_id is not None:
            set_dirs = [self._get_set_dir(experiment_set_id)]
        else:
            set_dirs = [d for d in self.storage_dir.iterdir() if d.is_dir()]
        
        for set_dir in set_dirs:
            config_path = set_dir / "config.json"
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                set_id = config_data["id"]
                set_name = config_data["name"]
                operation_mode = config_data["operation_mode"]
                col_bind = config_data["column_binding"]
                created_at = config_data["created_at"]
                
                for exp_config in config_data["experiments"]:
                    safe_name = _sanitize_filename(exp_config["name"])
                    
                    # Check what files exist
                    has_results = (set_dir / "results" / f"{safe_name}.pkl").exists()
                    has_chromatogram = (set_dir / "chromatograms" / f"{safe_name}.parquet").exists()
                    has_h5 = (set_dir / "h5" / f"{safe_name}.h5").exists()
                    
                    # Extract component info
                    components = exp_config.get("components", [])
                    n_components = len(components)
                    component_names = [c.get("name", f"comp_{i}") for i, c in enumerate(components)]
                    
                    exp_info = StoredExperimentInfo(
                        experiment_set_id=set_id,
                        experiment_set_name=set_name,
                        experiment_name=exp_config["name"],
                        created_at=datetime.fromisoformat(created_at) if isinstance(created_at, str) else created_at,
                        n_components=n_components,
                        component_names=component_names,
                        operation_mode=operation_mode,
                        column_model=col_bind["column_model"],
                        binding_model=col_bind["binding_model"],
                        has_results=has_results,
                        has_chromatogram=has_chromatogram,
                        has_h5=has_h5,
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
                "column_model", "binding_model", "has_results", "has_chromatogram", "has_h5",
            ])
        
        return pd.DataFrame([e.to_dict() for e in all_experiments])
    
    def list_experiment_sets(self) -> list[dict[str, Any]]:
        """List all experiment sets (metadata only)."""
        results = []
        
        for set_dir in self.storage_dir.iterdir():
            if not set_dir.is_dir():
                continue
            
            config_path = set_dir / "config.json"
            if not config_path.exists():
                continue
            
            try:
                with open(config_path, 'r') as f:
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
    
    def delete_experiment_set(self, experiment_set_id: str) -> bool:
        """Delete an experiment set and all its data."""
        set_dir = self._get_set_dir(experiment_set_id)
        
        if set_dir.exists():
            shutil.rmtree(set_dir)
            return True
        return False
    
    def get_chromatogram(
        self,
        experiment_set_id: str,
        experiment_name: str,
    ) -> pd.DataFrame | None:
        """Load just the chromatogram data (without full results)."""
        set_dir = self._get_set_dir(experiment_set_id)
        safe_name = _sanitize_filename(experiment_name)
        chrom_path = set_dir / "chromatograms" / f"{safe_name}.parquet"
        
        if chrom_path.exists():
            return pd.read_parquet(chrom_path)
        return None
