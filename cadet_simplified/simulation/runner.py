"""Simulation runner for CADET processes.

Handles running simulations and collecting results.
"""

from dataclasses import dataclass, field
from typing import Any
import traceback

import numpy as np


@dataclass
class SimulationResult:
    """Result of a single simulation."""
    experiment_name: str
    success: bool
    time: np.ndarray | None = None
    solution: dict[str, np.ndarray] | None = None  # component_name -> concentration array
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime_seconds: float = 0.0


@dataclass
class ValidationResult:
    """Result of validating a process configuration."""
    experiment_name: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SimulationRunner:
    """Runs CADET simulations.
    
    Example:
        >>> runner = SimulationRunner()
        >>> # Validate first
        >>> validation = runner.validate(process)
        >>> if validation.valid:
        ...     result = runner.run(process, "experiment_1")
    """
    
    def __init__(self, cadet_path: str | None = None):
        """Initialize runner.
        
        Parameters
        ----------
        cadet_path : str, optional
            Path to CADET installation. If None, uses default.
        """
        self.cadet_path = cadet_path
        self._simulator = None
    
    @property
    def simulator(self):
        """Get or create CADET simulator."""
        if self._simulator is None:
            from CADETProcess.simulator import Cadet
            self._simulator = Cadet()
            if self.cadet_path:
                self._simulator.cadet_path = self.cadet_path
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
    
    def run(self, process, experiment_name: str = "unnamed") -> SimulationResult:
        """Run a simulation.
        
        Parameters
        ----------
        process : Process
            CADET-Process Process object
        experiment_name : str
            Name of the experiment
            
        Returns
        -------
        SimulationResult
            Simulation results
        """
        import time
        
        errors = []
        warnings = []
        
        start_time = time.time()
        
        try:
            # Run simulation
            result = self.simulator.simulate(process)
            
            runtime = time.time() - start_time
            
            # Extract solution
            time_array = result.time
            
            # Get outlet solution
            solution = {}
            outlet = process.flow_sheet.product_outlets[0]
            
            for i, comp in enumerate(process.component_system.components):
                comp_name = comp.name if hasattr(comp, 'name') else f"Component_{i}"
                solution[comp_name] = result.solution[outlet.name][f"outlet.c_comp_{i}"]
            
            return SimulationResult(
                experiment_name=experiment_name,
                success=True,
                time=time_array,
                solution=solution,
                warnings=warnings,
                runtime_seconds=runtime,
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            errors.append(f"Simulation error: {str(e)}")
            tb = traceback.format_exc()
            errors.append(f"Traceback: {tb}")
            
            return SimulationResult(
                experiment_name=experiment_name,
                success=False,
                errors=errors,
                runtime_seconds=runtime,
            )
    
    def run_batch(
        self,
        processes: list[tuple[Any, str]],
        stop_on_error: bool = False,
    ) -> list[SimulationResult]:
        """Run multiple simulations.
        
        Parameters
        ----------
        processes : list[tuple[Process, str]]
            List of (process, experiment_name) tuples
        stop_on_error : bool
            If True, stop on first error
            
        Returns
        -------
        list[SimulationResult]
            Results for each simulation
        """
        results = []
        
        for process, name in processes:
            result = self.run(process, name)
            results.append(result)
            
            if not result.success and stop_on_error:
                break
        
        return results


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
