"""Simulation runner for CADET processes.

Handles running simulations and collecting results.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING
import traceback
from cadet_simplified.utils import path_utils
import numpy as np

if TYPE_CHECKING:
    from CADETProcess.simulationResults import SimulationResults
    from CADETProcess.processModel.process import Process


def _resolve_cadet_path(cadet_path: str | None) -> str | None:
    """Resolve CADET path from argument, environment variable, or auto-detection.
    
    Resolution order:
    1. Explicitly provided path (if not None)
    2. CADET_PATH environment variable
    3. CADET-Process auto-detection (returns None, let Cadet() handle it)
    
    Parameters
    ----------
    cadet_path : str or None
        Explicitly provided path, or None to use fallbacks
        
    Returns
    -------
    str or None
        Resolved path, or None to let CADET-Process auto-detect
    """
    # 1. Use explicitly provided path
    if cadet_path is not None:
        return cadet_path
    
    # 2. Check environment variable
    env_path = os.environ.get("CADET_PATH")
    if env_path and Path(env_path).exists():
        return env_path
    
    # 3. Let CADET-Process handle auto-detection
    return None


@dataclass
class SimulationResultWrapper:
    """Result of a single simulation."""
    experiment_name: str
    success: bool
    time: np.ndarray | None = None
    solution: dict[str, np.ndarray] | None = None  # component_name -> concentration array
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    cadet_result: "SimulationResults | None" = None
    h5_path: Path | None = None  # Path to H5 file if saved


@dataclass
class ValidationResult:
    """Result of validating a process configuration."""
    experiment_name: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SimulationRunner:
    """Runs CADET simulations.
    
    CADET path resolution order:
    1. Explicitly provided `cadet_path` argument
    2. `CADET_PATH` environment variable
    3. CADET-Process auto-detection
    
    Example:
        >>> # Auto-detect or use CADET_PATH env var
        >>> runner = SimulationRunner()
        
        >>> # Explicit path
        >>> runner = SimulationRunner(cadet_path="/path/to/cadet/bin")
        
        >>> # Validate first
        >>> validation = runner.validate(process)
        >>> if validation.valid:
        ...     result = runner.run(process)
    """
    
    def __init__(self, cadet_path: str | None = None):
        """Initialize runner.
        
        Parameters
        ----------
        cadet_path : str, optional
            Path to CADET installation. If None, checks CADET_PATH 
            environment variable, then falls back to CADET-Process 
            auto-detection.
        """
        self.cadet_path = _resolve_cadet_path(cadet_path)
        self._simulator = None
    
    @property
    def simulator(self):
        """Get or create CADET simulator."""
        if self._simulator is None:
            from CADETProcess.simulator import Cadet
            if self.cadet_path:
                self._simulator = Cadet(self.cadet_path)
            else:
                self._simulator = Cadet()
        return self._simulator
    
    def validate(self, process, experiment_name: str = "unnamed") -> ValidationResult:
        """Validate a process configuration.
        
        Uses CADET-Process's built-in validation.
        
        Parameters
        ----------
        process : Process
            CADET-Process Process object
        experiment_name : str
            Name of the experiment (for error messages)
            
        Returns
        -------
        ValidationResult
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        
        try:
            # Use check_config which returns True/False and prints warnings
            # We need to capture the output
            import io
            import sys
            
            # Capture stdout/stderr
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = captured_out = io.StringIO()
            sys.stderr = captured_err = io.StringIO()
            
            try:
                is_valid = process.check_config()
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            
            # Parse captured output for warnings
            output = captured_out.getvalue() + captured_err.getvalue()
            if output.strip():
                for line in output.strip().split("\n"):
                    if line.strip():
                        warnings.append(line.strip())
            
            if not is_valid:
                # TODO, capture the error log from check_config
                errors.append("Process configuration check failed")
                
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            # Add traceback for debugging
            tb = traceback.format_exc()
            errors.append(f"Traceback: {tb}")
        
        return ValidationResult(
            experiment_name=experiment_name,
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def run(
        self,
        process: "Process",
        h5_dir: Path | str | None = None,
    ) -> SimulationResultWrapper:
        """Run a simulation.
        
        Parameters
        ----------
        process : Process
            CADET-Process Process object
        h5_dir : Path or str, optional
            If provided, the H5 file (containing config + results) is preserved
            at this path. Otherwise, a temp file is created and deleted.
            
        Returns
        -------
        SimulationResultWrapper
            Simulation results
        """
        import time
        
        errors = []
        warnings = []
        
        if h5_dir is None:
            h5_dir = path_utils.get_storage_path() / "_pending"
        else:
            if isinstance(h5_dir, str):
                h5_dir = Path(h5_dir)
        
        h5_path = h5_dir.joinpath(f"{process.name}.h5")

        start_time = time.time()
        
        try:
            # Run simulation with optional file path preservation
            # Using _run directly to control file_path parameter
            result = self.simulator._run(process, file_path=h5_path)
            
            runtime = time.time() - start_time
            
            # Extract solution
            time_array = result.solution.outlet.outlet.time
            
            # Get outlet solution
            solution = {}
            outlet = process.flow_sheet.product_outlets[0]
            
            for i, comp in enumerate(process.component_system.components):
                comp_name = comp.name if hasattr(comp, 'name') else f"Component_{i}"
                solution[comp_name] = result.solution[outlet.name][f"outlet.c_comp_{i}"]
            
            return SimulationResultWrapper(
                experiment_name=process.name,
                success=True,
                time=time_array,
                solution=solution,
                warnings=warnings,
                runtime_seconds=runtime,
                cadet_result=result,
                h5_path=h5_path,
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            errors.append(f"Simulation error: {str(e)}")
            tb = traceback.format_exc()
            errors.append(f"Traceback: {tb}")
            
            return SimulationResultWrapper(
                experiment_name=process.name,
                success=False,
                errors=errors,
                runtime_seconds=runtime,
                h5_path=h5_path,
            )
    
    def run_batch(
        self,
        processes: list["Process"],
        h5_dir: Path | str | None = None,
        stop_on_error: bool = False,
        n_cores: int = 1,
        timeout_per_sim: float | None = None,
        progress_callback: Callable | None = None,
    ) -> list[SimulationResultWrapper]:
        """Run multiple simulations in parallel.
        
        Parameters
        ----------
        processes : list[Process]
            List of process instances to simulate
        h5_dir : Path or str, optional
            If provided, H5 files are saved to this directory with names
            based on process names: {h5_dir}/{process.name}.h5
        stop_on_error : bool, default=False
            If True, stop after first error
        n_cores : int, default=1
            Number of parallel processes (1 = sequential)
        timeout_per_sim : float, optional
            Timeout in seconds for each simulation. If exceeded, simulation
            is marked as failed.
        progress_callback : callable, optional
            Callback with signature: 
            callback(current: int, total: int, result: SimulationResultWrapper)
            Called after each simulation completes.
            
        Returns
        -------
        list[SimulationResultWrapper]
            Results in same order as input processes
        """
        total = len(processes)
        
        if h5_dir is None:
            h5_dir = path_utils.get_storage_path() / "_pending"
        else:
            if isinstance(h5_dir, str):
                h5_dir = Path(h5_dir)
            
            
        # Sequential execution
        if n_cores == 1:
            sequential_results: list[SimulationResultWrapper] = []
            for i, process in enumerate(processes):
                result = self.run(process, h5_dir)
                sequential_results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total, result)
                
                if not result.success and stop_on_error:
                    # Fill remaining with cancelled results
                    for remaining_process in processes[i + 1:]:
                        sequential_results.append(SimulationResultWrapper(
                            experiment_name=remaining_process.name,
                            success=False,
                            errors=["Simulation was skipped due to previous error."],
                        ))
                    break
            
            return sequential_results
            
        # Parallel execution
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
        
        results: dict[int, SimulationResultWrapper] = {}  # Aim: preserve order or returned results
        completed = 0
        should_stop = False
        
        executor = ProcessPoolExecutor(max_workers=n_cores)
        
        try:
            # Submit all tasks and track their indices
            future_to_idx = {
                executor.submit(self.run, process, h5_dir): idx 
                for idx, process in enumerate(processes)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                process = processes[idx]
                
                try:
                    # Wait for result with timeout
                    result = future.result(timeout=timeout_per_sim)
                except TimeoutError:
                    # Timeout occurred - mark as failed
                    result = SimulationResultWrapper(
                        experiment_name=process.name,
                        success=False,
                        errors=[f"Simulation timed out after {timeout_per_sim}s. C++ process may still be running in background."],
                        warnings=["Background process termination is not guaranteed."],
                    )
                except Exception as e:
                    # Other execution errors
                    import traceback
                    tb = traceback.format_exc()
                    result = SimulationResultWrapper(
                        experiment_name=process.name,
                        success=False,
                        errors=[f"Execution error: {str(e)}", f"Traceback: {tb}"],
                    )
                
                results[idx] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, result)
                
                if not result.success and stop_on_error:
                    should_stop = True
                    # Cancel remaining futures
                    for remaining_future in future_to_idx:
                        remaining_future.cancel()
                    break
        
        finally:
            # Clean shutdown - don't wait for cancelled tasks if stopping early
            executor.shutdown(wait=not should_stop)
        
        # Convert dict to ordered list, filling in cancelled simulations
        # cancelled simulations can occur when stop_on_error is false
        ordered_results = []
        for i in range(total):
            if i in results:
                # We have a result for this simulation
                ordered_results.append(results[i])
            else:
                # Simulation was cancelled - create error result
                cancelled_result = SimulationResultWrapper(
                    experiment_name=processes[i].name,
                    success=False,
                    errors=["Simulation was cancelled."],
                )
                ordered_results.append(cancelled_result)
        
        return ordered_results


def validate_and_report(process, experiment_name: str = "unnamed") -> tuple[bool, str]:
    """Convenience function to validate and get a formatted report.
    
    Parameters
    ----------
    process : Process
        Process to validate
    experiment_name : str
        Experiment name
        
    Returns
    -------
    tuple[bool, str]
        (is_valid, formatted_message)
    """
    runner = SimulationRunner()
    result = runner.validate(process, experiment_name)
    
    if result.valid:
        msg = f"✓ {experiment_name}: Configuration valid"
        if result.warnings:
            msg += f"\n  Warnings: {'; '.join(result.warnings)}"
        return True, msg
    else:
        msg = f"✗ {experiment_name}: Configuration invalid"
        for error in result.errors:
            msg += f"\n  Error: {error}"
        return False, msg
