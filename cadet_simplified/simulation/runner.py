"""Simulation runner for CADET processes.

Handles running simulations and collecting results.

Example:
    >>> from cadet_simplified.simulation import SimulationRunner
    >>> 
    >>> runner = SimulationRunner()
    >>> result = runner.run(process)
    >>> 
    >>> if result.success:
    ...     print(f"Completed in {result.runtime_seconds:.2f}s")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING
import time
import traceback

import numpy as np

if TYPE_CHECKING:
    from CADETProcess.simulationResults import SimulationResults
    from CADETProcess.processModel.process import Process


@dataclass
class SimulationResultWrapper:
    """Result of a single simulation.
    
    Attributes
    ----------
    experiment_name : str
        Name of the experiment
    success : bool
        Whether simulation completed successfully
    time : np.ndarray, optional
        Time array from simulation
    solution : dict[str, np.ndarray], optional
        Component name -> concentration array mapping
    errors : list[str]
        Error messages if failed
    warnings : list[str]
        Warning messages
    runtime_seconds : float
        Simulation runtime
    cadet_result : SimulationResults, optional
        Full CADET-Process result object
    """
    experiment_name: str
    success: bool
    time: np.ndarray | None = None
    solution: dict[str, np.ndarray] | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    cadet_result: "SimulationResults | None" = None


@dataclass
class ValidationResult:
    """Result of validating a process configuration.
    
    Attributes
    ----------
    experiment_name : str
        Name of the experiment
    valid : bool
        Whether configuration is valid
    errors : list[str]
        Error messages
    warnings : list[str]
        Warning messages
    """
    experiment_name: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SimulationRunner:
    """Runs CADET simulations.
    
    Parameters
    ----------
    cadet_path : str, optional
        Path to CADET installation. If None, uses default.
        
    Example:
        >>> runner = SimulationRunner()
        >>> validation = runner.validate(process, "my_experiment")
        >>> if validation.valid:
        ...     result = runner.run(process)
    """
    
    def __init__(self, cadet_path: str | None = None):
        self.cadet_path = cadet_path
        self._simulator = None
    
    @property
    def simulator(self):
        """Get or create CADET simulator (lazy initialization)."""
        if self._simulator is None:
            from CADETProcess.simulator import Cadet
            self._simulator = Cadet()
            if self.cadet_path:
                self._simulator.cadet_path = self.cadet_path
        return self._simulator
    
    def validate(
        self,
        process: "Process",
        experiment_name: str = "unnamed",
    ) -> ValidationResult:
        """Validate a process configuration.
        
        Parameters
        ----------
        process : Process
            CADET-Process Process object
        experiment_name : str
            Name for error messages
            
        Returns
        -------
        ValidationResult
            Validation result
        """
        errors = []
        warnings = []
        
        try:
            import io
            import sys
            
            # Capture stdout/stderr from check_config
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
                errors.append("Process configuration check failed")
                
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            tb = traceback.format_exc()
            errors.append(f"Traceback: {tb}")
        
        return ValidationResult(
            experiment_name=experiment_name,
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def run(self, process: "Process") -> SimulationResultWrapper:
        """Run a simulation.
        
        Parameters
        ----------
        process : Process
            CADET-Process Process object
            
        Returns
        -------
        SimulationResultWrapper
            Simulation results
        """
        errors = []
        warnings = []
        start_time = time.time()
        
        try:
            result = self.simulator.simulate(process)
            runtime = time.time() - start_time
            
            # Extract time array
            time_array = result.solution.outlet.outlet.time
            
            # Extract outlet solution
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
            )
    
    def run_batch(
        self,
        processes: list["Process"],
        stop_on_error: bool = False,
        progress_callback: callable = None,
    ) -> list[SimulationResultWrapper]:
        """Run multiple simulations sequentially.
        
        Parameters
        ----------
        processes : list[Process]
            List of process objects
        stop_on_error : bool, default=False
            Stop after first error
        progress_callback : callable, optional
            Called after each simulation: callback(current, total, result)
            
        Returns
        -------
        list[SimulationResultWrapper]
            Results in same order as input
        """
        total = len(processes)
        results = []
        
        for i, process in enumerate(processes):
            result = self.run(process)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, result)
            
            if not result.success and stop_on_error:
                # Fill remaining with cancelled
                for remaining in processes[i + 1:]:
                    results.append(SimulationResultWrapper(
                        experiment_name=remaining.name,
                        success=False,
                        errors=["Skipped due to previous error"],
                    ))
                break
        
        return results
