"""Results analysis and export for CADET simulations.

Handles:
- Chromatogram interpolation
- Excel export (parameters + chromatograms)
- H5 file management (config + results)
- Pickle backup (optional)

Future:
- Peak detection
- Purity calculation
- Plotting
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import pickle
import shutil

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from CADETProcess.simulator import Cadet
    from ..simulation.runner import SimulationResultWrapper
    from ..operation_modes import ExperimentConfig, ColumnBindingConfig


@dataclass
class InterpolatedChromatogram:
    """Interpolated chromatogram data."""
    time: np.ndarray
    concentrations: dict[str, np.ndarray]  # component_name -> concentration array
    experiment_name: str


class ResultsAnalyzer:
    """Analyzer and exporter for simulation results.
    
    Handles all file I/O for simulation results including:
    - Config H5 files (via CADET-Process simulator)
    - Full H5 files (config + results, if preserved during simulation)
    - Excel export with parameters and chromatograms
    - Optional pickle backup
    
    Example:
        >>> from CADETProcess.simulator import Cadet
        >>> analyzer = ResultsAnalyzer(
        ...     base_dir="./output",
        ...     simulator=Cadet(),
        ... )
        >>> # After running simulations
        >>> output_path = analyzer.export(
        ...     results=results,
        ...     experiment_configs=configs,
        ...     column_binding=column_binding,
        ...     name="my_study",
        ... )
    """
    
    def __init__(
        self,
        base_dir: Path | str,
        simulator: "Cadet",
        n_interpolation_points: int = 500,
        save_pickle: bool = False,
    ):
        """Initialize analyzer.
        
        Parameters
        ----------
        base_dir : Path or str
            Base directory for all exports. Subfolders will be created
            with timestamp and study name.
        simulator : Cadet
            CADET-Process simulator instance (used for save_to_h5)
        n_interpolation_points : int, default=500
            Default number of points for chromatogram interpolation
        save_pickle : bool, default=False
            If True, save pickle backup of results
        """
        self.base_dir = Path(base_dir)
        self.simulator = simulator
        self.n_interpolation_points = n_interpolation_points
        self.save_pickle = save_pickle
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_output_path(self, name: str) -> Path:
        """Create and return timestamped output folder.
        
        Useful when you need the path before running simulations
        (e.g., for H5 file preservation).
        
        Parameters
        ----------
        name : str
            Study name (will be sanitized for filesystem)
            
        Returns
        -------
        Path
            Path to created output folder
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M-%S")
        safe_name = self._sanitize_filename(name)
        folder_name = f"{timestamp}_{safe_name}"
        
        output_path = self.base_dir / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        return output_path
    
    def export(
        self,
        results: list["SimulationResultWrapper"],
        experiment_configs: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        name: str | None = None,
        output_path: Path | None = None,
        n_interpolation_points: int | None = None,
    ) -> Path:
        """Export results to folder.
        
        Creates:
        - Config H5 files for each experiment (always)
        - Full H5 files (if they were preserved during simulation)
        - Excel file with parameters sheet + chromatogram sheets
        - Pickle backup (if enabled)
        
        Parameters
        ----------
        results : list[SimulationResultWrapper]
            Simulation results (must have cadet_result with process)
        experiment_configs : list[ExperimentConfig]
            Experiment configurations (for parameter export)
        column_binding : ColumnBindingConfig
            Column and binding configuration (for parameter export)
        name : str, optional
            Study name. Required if output_path is not provided.
        output_path : Path, optional
            Use existing folder instead of creating new one.
            If provided, name is ignored.
        n_interpolation_points : int, optional
            Override default interpolation points
            
        Returns
        -------
        Path
            Path to output folder
        """
        # Determine output path
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        elif name is not None:
            output_path = self.get_output_path(name)
        else:
            raise ValueError("Either 'name' or 'output_path' must be provided")
        
        n_points = n_interpolation_points or self.n_interpolation_points
        
        # 1. Save H5 files
        self._save_h5_files(results, output_path)
        
        # 2. Create Excel with parameters and chromatograms
        self._save_excel(
            results=results,
            experiment_configs=experiment_configs,
            column_binding=column_binding,
            output_path=output_path,
            n_points=n_points,
        )
        
        # 3. Optional pickle backup
        if self.save_pickle:
            self._save_pickle(results, output_path)
        
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
        """
        if result.cadet_result is None:
            raise ValueError(f"No cadet_result in {result.experiment_name}")
        
        n_points = n_points or self.n_interpolation_points
        
        # Get process from result
        process = result.cadet_result.process
        
        # Find product outlet
        product_outlets = process.flow_sheet.product_outlets
        if not product_outlets:
            raise ValueError(f"No product outlet found for {result.experiment_name}")
        
        if len(product_outlets) > 1:
            # Log warning but continue with first outlet
            import warnings
            warnings.warn(
                f"Multiple product outlets found for {result.experiment_name}, "
                f"using first one: {product_outlets[0].name}"
            )
        
        product_outlet = product_outlets[0]
        
        # Get solution for the outlet
        outlet_solution = result.cadet_result.solution[product_outlet.name]
        
        # Extract time and create interpolation grid
        time_complete = result.cadet_result.time_complete
        time_interp = np.linspace(
            float(time_complete.min()),
            float(time_complete.max()),
            n_points,
        )
        
        # Get interpolated solution
        # outlet_solution.outlet is a SolutionIO object with solution_interpolated method
        interp_func = outlet_solution.outlet.solution_interpolated
        solution_interp = interp_func(time_interp)
        
        # Build concentration dict with component names
        concentrations = {}
        for i, comp in enumerate(process.component_system.components):
            comp_name = comp.name if hasattr(comp, 'name') else f"Component_{i}"
            concentrations[comp_name] = solution_interp[:, i]
        
        return InterpolatedChromatogram(
            time=time_interp,
            concentrations=concentrations,
            experiment_name=result.experiment_name,
        )
    
    def _save_h5_files(
        self,
        results: list["SimulationResultWrapper"],
        output_path: Path,
    ) -> None:
        """Save H5 files for each result.
        
        - If result has h5_path (full H5 preserved), copy/move it
        - Otherwise, save config-only H5 using simulator
        """
        for result in results:
            if not result.success or result.cadet_result is None:
                continue
            
            process = result.cadet_result.process
            safe_name = self._sanitize_filename(result.experiment_name)
            
            if result.h5_path is not None and result.h5_path.exists():
                # Full H5 was preserved - copy to output
                dest_path = output_path / f"{safe_name}.h5"
                if result.h5_path != dest_path:
                    shutil.copy2(result.h5_path, dest_path)
            else:
                # Save config-only H5
                config_path = output_path / f"{safe_name}_config.h5"
                self.simulator.save_to_h5(process, config_path)
    
    def _save_excel(
        self,
        results: list["SimulationResultWrapper"],
        experiment_configs: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        output_path: Path,
        n_points: int,
    ) -> None:
        """Save Excel file with parameters and chromatograms."""
        excel_path = output_path / "results.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Sheet 1: Parameters
            params_df = self._create_parameters_df(
                experiment_configs=experiment_configs,
                column_binding=column_binding,
                results=results,
            )
            params_df.to_excel(writer, sheet_name="Parameters", index=False)
            
            # Sheets 2+: Chromatograms for each experiment
            for result in results:
                if not result.success or result.cadet_result is None:
                    continue
                
                try:
                    chrom = self.interpolate_chromatogram(result, n_points)
                    chrom_df = self._chromatogram_to_df(chrom)
                    
                    # Excel sheet names limited to 31 chars
                    sheet_name = self._sanitize_filename(result.experiment_name)[:31]
                    chrom_df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    import warnings
                    warnings.warn(
                        f"Failed to export chromatogram for {result.experiment_name}: {e}"
                    )
    
    def _create_parameters_df(
        self,
        experiment_configs: list["ExperimentConfig"],
        column_binding: "ColumnBindingConfig",
        results: list["SimulationResultWrapper"],
    ) -> pd.DataFrame:
        """Create DataFrame with all parameters for reproducibility."""
        rows = []
        
        for exp_config, result in zip(experiment_configs, results):
            row = {
                "experiment_name": exp_config.name,
                "simulation_success": result.success,
                "runtime_seconds": result.runtime_seconds,
            }
            
            # Add experiment parameters
            for key, value in exp_config.parameters.items():
                row[f"exp_{key}"] = value
            
            # Add column parameters (shared across experiments)
            row["column_model"] = column_binding.column_model
            for key, value in column_binding.column_parameters.items():
                row[f"col_{key}"] = value
            
            # Add per-component column parameters
            for key, values in column_binding.component_column_parameters.items():
                for i, val in enumerate(values):
                    row[f"col_{key}_comp{i+1}"] = val
            
            # Add binding parameters (shared across experiments)
            row["binding_model"] = column_binding.binding_model
            for key, value in column_binding.binding_parameters.items():
                row[f"bind_{key}"] = value
            
            # Add per-component binding parameters
            for key, values in column_binding.component_binding_parameters.items():
                for i, val in enumerate(values):
                    row[f"bind_{key}_comp{i+1}"] = val
            
            # Add error info if failed
            if not result.success:
                row["errors"] = "; ".join(result.errors)
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _chromatogram_to_df(self, chrom: InterpolatedChromatogram) -> pd.DataFrame:
        """Convert chromatogram to DataFrame."""
        data = {"time_s": chrom.time}
        
        for comp_name, conc in chrom.concentrations.items():
            data[f"c_{comp_name}_mM"] = conc
        
        return pd.DataFrame(data)
    
    def _save_pickle(
        self,
        results: list["SimulationResultWrapper"],
        output_path: Path,
    ) -> None:
        """Save pickle backup of results."""
        pickle_path = output_path / "results_backup.pkl"
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
    
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Sanitize string for use as filename."""
        # Replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        result = name
        for char in invalid_chars:
            result = result.replace(char, '_')
        
        # Remove leading/trailing whitespace and dots
        result = result.strip('. ')
        
        # Limit length
        if len(result) > 100:
            result = result[:100]
        
        return result or "unnamed"
