"""JSON file-based experiment storage (legacy).

This module provides the dataclasses used for serialization and a simple
config-only storage implementation.

For full results storage (including pickled results, chromatograms, H5 files),
use FileResultsStorage from storage.file_storage instead.

Stores experiment configurations and results in JSON files.
Each experiment set (from one Excel upload) is stored as a single JSON file.
"""

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import hashlib

from ..operation_modes import ExperimentConfig, ColumnBindingConfig, ComponentDefinition


@dataclass
class StoredExperiment:
    """A stored experiment with metadata."""
    id: str
    name: str
    parameters: dict[str, Any]
    components: list[dict[str, Any]]
    created_at: str
    
    @classmethod
    def from_experiment_config(cls, config: ExperimentConfig) -> "StoredExperiment":
        """Create from ExperimentConfig."""
        # Generate ID from name and timestamp
        timestamp = datetime.now().isoformat()
        id_string = f"{config.name}_{timestamp}"
        exp_id = hashlib.md5(id_string.encode()).hexdigest()[:12]
        
        return cls(
            id=exp_id,
            name=config.name,
            parameters=config.parameters,
            components=[
                {"name": c.name, "is_salt": c.is_salt, "molecular_weight": c.molecular_weight}
                for c in config.components
            ],
            created_at=timestamp,
        )
    
    def to_experiment_config(self) -> ExperimentConfig:
        """Convert back to ExperimentConfig."""
        components = [
            ComponentDefinition(
                name=c["name"],
                is_salt=c.get("is_salt", False),
                molecular_weight=c.get("molecular_weight"),
            )
            for c in self.components
        ]
        return ExperimentConfig(
            name=self.name,
            parameters=self.parameters,
            components=components,
        )


@dataclass
class StoredColumnBinding:
    """Stored column and binding configuration."""
    column_model: str
    binding_model: str
    column_parameters: dict[str, Any]
    binding_parameters: dict[str, Any]
    component_column_parameters: dict[str, list[Any]]
    component_binding_parameters: dict[str, list[Any]]
    
    @classmethod
    def from_column_binding_config(cls, config: ColumnBindingConfig) -> "StoredColumnBinding":
        """Create from ColumnBindingConfig."""
        return cls(
            column_model=config.column_model,
            binding_model=config.binding_model,
            column_parameters=config.column_parameters,
            binding_parameters=config.binding_parameters,
            component_column_parameters=config.component_column_parameters,
            component_binding_parameters=config.component_binding_parameters,
        )
    
    def to_column_binding_config(self) -> ColumnBindingConfig:
        """Convert back to ColumnBindingConfig."""
        return ColumnBindingConfig(
            column_model=self.column_model,
            binding_model=self.binding_model,
            column_parameters=self.column_parameters,
            binding_parameters=self.binding_parameters,
            component_column_parameters=self.component_column_parameters,
            component_binding_parameters=self.component_binding_parameters,
        )


@dataclass
class ExperimentSet:
    """A set of experiments with shared column/binding configuration."""
    id: str
    name: str
    operation_mode: str
    column_binding: StoredColumnBinding
    experiments: list[StoredExperiment]
    created_at: str
    updated_at: str
    description: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "operation_mode": self.operation_mode,
            "description": self.description,
            "column_binding": asdict(self.column_binding),
            "experiments": [asdict(e) for e in self.experiments],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentSet":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            operation_mode=data["operation_mode"],
            description=data.get("description", ""),
            column_binding=StoredColumnBinding(**data["column_binding"]),
            experiments=[StoredExperiment(**e) for e in data["experiments"]],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


class ExperimentStore:
    """File-based storage for experiment sets (config only, legacy).
    
    .. deprecated::
        Use FileResultsStorage for full results storage including
        pickled SimulationResultWrapper, chromatograms, and H5 files.
    
    Each experiment set is stored as a JSON file in the storage directory.
    This class only stores configurations, not simulation results.
    
    Example:
        >>> store = ExperimentStore("./experiments")
        >>> # Save experiments from Excel upload
        >>> exp_set = store.save_from_parse_result(
        ...     parse_result, 
        ...     name="IEX Screening",
        ...     operation_mode="LWE_concentration_based"
        ... )
        >>> # Load later
        >>> exp_set = store.load(exp_set.id)
    """
    
    def __init__(self, storage_dir: str | Path):
        """Initialize storage.
        
        Parameters
        ----------
        storage_dir : str or Path
            Directory to store experiment JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_from_parse_result(
        self,
        experiments: list[ExperimentConfig],
        column_binding: ColumnBindingConfig,
        name: str,
        operation_mode: str,
        description: str = "",
    ) -> ExperimentSet:
        """Save experiments from a parsed Excel file (config only).
        
        Parameters
        ----------
        experiments : list[ExperimentConfig]
            Parsed experiments
        column_binding : ColumnBindingConfig
            Parsed column/binding configuration
        name : str
            Name for this experiment set
        operation_mode : str
            Name of the operation mode
        description : str, optional
            Description of the experiment set
            
        Returns
        -------
        ExperimentSet
            The saved experiment set
        """
        # Generate ID
        timestamp = datetime.now().isoformat()
        id_string = f"{name}_{timestamp}"
        set_id = hashlib.md5(id_string.encode()).hexdigest()[:12]
        
        # Create stored objects
        stored_experiments = [
            StoredExperiment.from_experiment_config(exp)
            for exp in experiments
        ]
        stored_column_binding = StoredColumnBinding.from_column_binding_config(column_binding)
        
        exp_set = ExperimentSet(
            id=set_id,
            name=name,
            operation_mode=operation_mode,
            description=description,
            column_binding=stored_column_binding,
            experiments=stored_experiments,
            created_at=timestamp,
            updated_at=timestamp,
        )
        
        # Save to file
        self._save_to_file(exp_set)
        
        return exp_set
    
    def save(self, exp_set: ExperimentSet) -> None:
        """Save an experiment set.
        
        Parameters
        ----------
        exp_set : ExperimentSet
            Experiment set to save
        """
        exp_set.updated_at = datetime.now().isoformat()
        self._save_to_file(exp_set)
    
    def load(self, set_id: str) -> ExperimentSet | None:
        """Load an experiment set by ID.
        
        Parameters
        ----------
        set_id : str
            ID of the experiment set
            
        Returns
        -------
        ExperimentSet or None
            The experiment set, or None if not found
        """
        file_path = self.storage_dir / f"{set_id}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        return ExperimentSet.from_dict(data)
    
    def delete(self, set_id: str) -> bool:
        """Delete an experiment set.
        
        Parameters
        ----------
        set_id : str
            ID of the experiment set
            
        Returns
        -------
        bool
            True if deleted, False if not found
        """
        file_path = self.storage_dir / f"{set_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def list_all(self) -> list[dict]:
        """List all experiment sets (metadata only).
        
        Returns
        -------
        list[dict]
            List of experiment set metadata
        """
        results = []
        
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
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
                continue  # Skip corrupted files
        
        # Sort by updated_at descending
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        
        return results
    
    def _save_to_file(self, exp_set: ExperimentSet) -> None:
        """Save experiment set to JSON file."""
        file_path = self.storage_dir / f"{exp_set.id}.json"
        
        with open(file_path, "w") as f:
            json.dump(exp_set.to_dict(), f, indent=2)
