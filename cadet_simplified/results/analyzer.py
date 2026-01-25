"""Results export utilities for CADET simulations.

Provides export functionality for sharing simulation results:
- Excel export with parameters and chromatograms
- Standalone chromatogram interpolation utility

For persistent storage of results, use FileResultsStorage from the storage module.

Example - Export from storage:
    >>> from cadet_simplified.storage import FileResultsStorage
    >>> from cadet_simplified.results import ResultsExporter
    >>> 
    >>> storage = FileResultsStorage("./experiments")
    >>> loaded = storage.load_results_by_selection([...])
    >>> 
    >>> exporter = ResultsExporter()
    >>> excel_path = exporter.export_to_excel(loaded, "./exports/my_analysis.xlsx")

Example - Export after simulation:
    >>> from cadet_simplified.results import ResultsExporter
    >>> 
    >>> exporter = ResultsExporter()
    >>> excel_path = exporter.export_simulation_results(
    ...     results=sim_results,
    ...     experiment_configs=configs,
    ...     column_binding=col_bind,
    ...     output_path="./exports/simulation_results.xlsx",
    ... )
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..simulation.runner import SimulationResultWrapper
    from ..storage.interfaces import LoadedExperiment
    from ..operation_modes import ExperimentConfig, ColumnBindingConfig


@dataclass
class InterpolatedChromatogram:
    """Interpolated chromatogram data.
    
    Attributes
    ----------
    time : np.ndarray
        Time points in seconds
    concentrations : dict[str, np.ndarray]
        Component name -> concentration array mapping
    experiment_name : str
        Name of the experiment
    """
    time: np.ndarray
    concentrations: dict[str, np.ndarray]
    experiment_name: str
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with columns [time, comp_0, comp_1, ...].
        
        Returns
        -------
        pd.DataFrame
            Chromatogram data
        """
        data = {"time": self.time}
        data.update(self.concentrations)
        return pd.DataFrame(data)


class ResultsExporter:
    """Export utility for simulation results.
    
    Focuses on creating shareable exports (Excel files) from simulation
    results or loaded experiments from storage.
    
    For persistent storage, use FileResultsStorage instead.
    
    Parameters
    ----------
    n_interpolation_points : int, default=500
        Number of points for chromatogram interpolation
    """
    
    def __init__(self, n_interpolation_points: int = 500):
        """Initialize exporter.
        
        Parameters
        ----------
        n_interpolation_points : int, default=500
            Default number of points for chromatogram interpolation
        """
        self.n_interpolation_points = n_interpolation_points
    
    def export_to_excel(
        self,
        loaded_experiments: list["LoadedExperiment"],
        output_path: str | Path,
        include_chromatograms: bool = True,
    ) -> Path:
        """Export loaded experiments to Excel file.
        
        Creates an Excel file with:
        - Parameters sheet: experiment parameters and metadata
        - One sheet per experiment: interpolated chromatogram data
        
        Parameters
        ----------
        loaded_experiments : list[LoadedExperiment]
            Experiments loaded from storage
        output_path : str or Path
            Output Excel file path
        include_chromatograms : bool, default=True
            Whether to include chromatogram sheets
            
        Returns
        -------
        Path
            Path to created Excel file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Parameters sheet
            params_df = self._create_parameters_df_from_loaded(loaded_experiments)
            params_df.to_excel(writer, sheet_name="Parameters", index=False)
            
            # Chromatogram sheets
            if include_chromatograms:
                for exp in loaded_experiments:
                    if exp.chromatogram_df is not None:
                        sheet_name = self._sanitize_sheet_name(exp.experiment_name)
                        exp.chromatogram_df.to_excel(
                            writer, sheet_name=sheet_name, index=False
                        )
        
        return output_path
    
    def export_simulation_results(
        self,
        results: list["SimulationResultWrapper"],
        experiment_configs: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        output_path: str | Path,
        n_interpolation_points: int | None = None,
    ) -> Path:
        """Export simulation results directly to Excel.
        
        Use this for quick exports without going through storage.
        For persistent storage, use FileResultsStorage instead.
        
        Parameters
        ----------
        results : list[SimulationResultWrapper]
            Simulation results
        experiment_configs : list[ExperimentConfig]
            Experiment configurations
        column_binding : ColumnBindingConfig
            Column and binding configuration
        output_path : str or Path
            Output Excel file path
        n_interpolation_points : int, optional
            Override default interpolation points
            
        Returns
        -------
        Path
            Path to created Excel file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        n_points = n_interpolation_points or self.n_interpolation_points
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Parameters sheet
            params_df = self._create_parameters_df(
                experiment_configs=experiment_configs,
                column_binding=column_binding,
                results=results,
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)
            
            # Chromatogram sheets
            for result in results:
                if not result.success or result.cadet_result is None:
                    continue
                
                try:
                    chrom = self.interpolate_chromatogram(result, n_points)
                    chrom_df = chrom.to_dataframe()
                    sheet_name = self._sanitize_sheet_name(result.experiment_name)
                    chrom_df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Failed to export chromatogram for {result.experiment_name}: {e}"
                    )
        
        return output_path
    
    def interpolate_chromatogram(
        self,
        result: "SimulationResultWrapper",
        n_points: int | None = None,
    ) -> InterpolatedChromatogram:
        """Interpolate chromatogram from simulation result.
        
        Parameters
        ----------
        result : SimulationResultWrapper
            Simulation result with cadet_result
        n_points : int, optional
            Number of interpolation points (default: instance setting)
            
        Returns
        -------
        InterpolatedChromatogram
            Interpolated time and concentration arrays
            
        Raises
        ------
        ValueError
            If result has no cadet_result or no product outlet
        """
        if result.cadet_result is None:
            raise ValueError(f"No cadet_result in {result.experiment_name}")
        
        n_points = n_points or self.n_interpolation_points
        
        process = result.cadet_result.process
        product_outlets = process.flow_sheet.product_outlets
        
        if not product_outlets:
            raise ValueError(f"No product outlet found for {result.experiment_name}")
        
        if len(product_outlets) > 1:
            import warnings
            warnings.warn(
                f"Multiple product outlets found for {result.experiment_name}, "
                f"using first one: {product_outlets[0].name}"
            )
        
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
        
        concentrations = {}
        for i, comp in enumerate(process.component_system.components):
            comp_name = comp.name if hasattr(comp, 'name') else f"Component_{i}"
            concentrations[comp_name] = solution_interp[:, i]
        
        return InterpolatedChromatogram(
            time=time_interp,
            concentrations=concentrations,
            experiment_name=result.experiment_name,
        )
    
    def _create_parameters_df_from_loaded(
        self,
        loaded_experiments: list["LoadedExperiment"],
    ) -> pd.DataFrame:
        """Create parameters DataFrame from loaded experiments."""
        rows = []
        
        for exp in loaded_experiments:
            row = {
                "experiment_set": exp.experiment_set_name,
                "experiment_name": exp.experiment_name,
                "success": exp.result.success,
                "runtime_seconds": exp.result.runtime_seconds,
            }
            
            # Experiment parameters
            for key, value in exp.experiment_config.parameters.items():
                row[f"exp_{key}"] = value
            
            # Column parameters
            row["column_model"] = exp.column_binding.column_model
            for key, value in exp.column_binding.column_parameters.items():
                row[f"col_{key}"] = value
            
            # Per-component column parameters
            for key, values in exp.column_binding.component_column_parameters.items():
                for i, val in enumerate(values):
                    row[f"col_{key}_comp{i+1}"] = val
            
            # Binding parameters
            row["binding_model"] = exp.column_binding.binding_model
            for key, value in exp.column_binding.binding_parameters.items():
                row[f"bind_{key}"] = value
            
            # Per-component binding parameters
            for key, values in exp.column_binding.component_binding_parameters.items():
                for i, val in enumerate(values):
                    row[f"bind_{key}_comp{i+1}"] = val
            
            # Error info
            if not exp.result.success:
                row["errors"] = "; ".join(exp.result.errors)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_parameters_df(
        self,
        experiment_configs: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        results: list["SimulationResultWrapper"],
    ) -> pd.DataFrame:
        """Create parameters DataFrame from simulation results."""
        rows = []
        
        for exp_config, result in zip(experiment_configs, results):
            row = {
                "experiment_name": exp_config.name,
                "simulation_success": result.success,
                "runtime_seconds": result.runtime_seconds,
            }
            
            # Experiment parameters
            for key, value in exp_config.parameters.items():
                row[f"exp_{key}"] = value
            
            # Column parameters
            row["column_model"] = column_binding.column_model
            for key, value in column_binding.column_parameters.items():
                row[f"col_{key}"] = value
            
            # Per-component column parameters
            for key, values in column_binding.component_column_parameters.items():
                for i, val in enumerate(values):
                    row[f"col_{key}_comp{i+1}"] = val
            
            # Binding parameters
            row["binding_model"] = column_binding.binding_model
            for key, value in column_binding.binding_parameters.items():
                row[f"bind_{key}"] = value
            
            # Per-component binding parameters
            for key, values in column_binding.component_binding_parameters.items():
                for i, val in enumerate(values):
                    row[f"bind_{key}_comp{i+1}"] = val
            
            # Error info
            if not result.success:
                row["errors"] = "; ".join(result.errors)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def _sanitize_sheet_name(name: str) -> str:
        """Sanitize string for use as Excel sheet name."""
        # Excel sheet names: max 31 chars, no special chars
        invalid_chars = '[]:*?/\\'
        result = name
        for char in invalid_chars:
            result = result.replace(char, '_')
        return result[:31]


# Backwards compatibility alias
class ResultsAnalyzer(ResultsExporter):
    """Alias for ResultsExporter (backwards compatibility).
    
    .. deprecated::
        Use ResultsExporter directly. For persistent storage,
        use FileResultsStorage from the storage module.
    """
    
    def __init__(
        self,
        base_dir: Path | str | None = None,
        simulator=None,
        n_interpolation_points: int = 500,
        save_pickle: bool = False,
    ):
        """Initialize analyzer.
        
        Parameters
        ----------
        base_dir : Path or str, optional
            Ignored (kept for backwards compatibility)
        simulator : optional
            Ignored (kept for backwards compatibility)
        n_interpolation_points : int, default=500
            Number of points for chromatogram interpolation
        save_pickle : bool, default=False
            Ignored (kept for backwards compatibility)
        """
        import warnings
        warnings.warn(
            "ResultsAnalyzer is deprecated. Use ResultsExporter for exports "
            "or FileResultsStorage for persistent storage.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(n_interpolation_points=n_interpolation_points)
        
        # Store for partial backwards compatibility
        self.base_dir = Path(base_dir) if base_dir else None
        self.simulator = simulator
        self.save_pickle = save_pickle
    
    def export(
        self,
        results: list["SimulationResultWrapper"],
        experiment_configs: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        name: str | None = None,
        output_path: Path | None = None,
        n_interpolation_points: int | None = None,
    ) -> Path:
        """Export results (backwards compatible method).
        
        .. deprecated::
            Use export_simulation_results() or FileResultsStorage.save_experiment_set()
        """
        if output_path is None:
            if name is None:
                raise ValueError("Either 'name' or 'output_path' must be provided")
            if self.base_dir is None:
                raise ValueError("base_dir must be set if output_path not provided")
            
            timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
            safe_name = self._sanitize_sheet_name(name)
            output_path = self.base_dir / f"{timestamp}_{safe_name}" / "results.xlsx"
        
        return self.export_simulation_results(
            results=results,
            experiment_configs=experiment_configs,
            column_binding=column_binding,
            output_path=output_path,
            n_interpolation_points=n_interpolation_points,
        )
