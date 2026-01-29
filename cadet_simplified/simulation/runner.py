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
import multiprocessing as mp
import queue
import threading
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


def _run_single_simulation(process: "Process", cadet_path: str | None) -> SimulationResultWrapper:
    """Run a single simulation (worker function for multiprocessing).
    
    This function is called in a separate process.
    
    Parameters
    ----------
    process : Process
        CADET-Process Process object
    cadet_path : str, optional
        Path to CADET installation
        
    Returns
    -------
    SimulationResultWrapper
        Simulation result
    """
    from CADETProcess.simulator import Cadet
    
    errors = []
    warnings = []
    start_time = time.time()
    
    try:
        simulator = Cadet()
        if cadet_path:
            simulator.cadet_path = cadet_path
        
        result = simulator.simulate(process)
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
        n_cores: int = 4,
        timeout_per_sim: float | None = None,
        progress_callback: callable = None,
    ) -> list[SimulationResultWrapper]:
        """Run multiple simulations, optionally in parallel.
        
        Parameters
        ----------
        processes : list[Process]
            List of process objects
        stop_on_error : bool, default=False
            Stop after first error
        n_cores : int, default=4
            Number of parallel processes (1 = sequential)
        timeout_per_sim : float, optional
            Timeout in seconds for each simulation (parallel mode only)
        progress_callback : callable, optional
            Called after each simulation: callback(current, total, result)
            Note: In parallel mode, results may arrive out of order.
            
        Returns
        -------
        list[SimulationResultWrapper]
            Results in same order as input processes
        """
        total = len(processes)
        
        # Sequential execution
        if n_cores == 1:
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
        
        # Parallel execution
        from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
        
        results_dict: dict[int, SimulationResultWrapper] = {}
        completed = 0
        should_stop = False
        
        executor = ProcessPoolExecutor(max_workers=n_cores)
        
        try:
            # Submit all tasks and track their indices
            future_to_idx = {
                executor.submit(self.run, process): idx
                for idx, process in enumerate(processes)
            }
            
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                process = processes[idx]
                
                try:
                    result = future.result(timeout=timeout_per_sim)
                except TimeoutError:
                    result = SimulationResultWrapper(
                        experiment_name=process.name,
                        success=False,
                        errors=[f"Simulation timed out after {timeout_per_sim}s"],
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    result = SimulationResultWrapper(
                        experiment_name=process.name,
                        success=False,
                        errors=[f"Execution error: {str(e)}", f"Traceback: {tb}"],
                    )
                
                results_dict[idx] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, result)
                
                if not result.success and stop_on_error:
                    should_stop = True
                    for remaining_future in future_to_idx:
                        remaining_future.cancel()
                    break
        
        finally:
            executor.shutdown(wait=not should_stop)
        
        # Convert dict to ordered list, filling cancelled slots
        ordered_results = []
        for i in range(total):
            if i in results_dict:
                ordered_results.append(results_dict[i])
            else:
                ordered_results.append(SimulationResultWrapper(
                    experiment_name=processes[i].name,
                    success=False,
                    errors=["Simulation was cancelled"],
                ))
        
        return ordered_results
    
    def run_batch_interruptible(
        self,
        processes: list["Process"],
        n_cores: int = 10,
        progress_callback: callable = None,
        stop_event: threading.Event = None,
    ) -> list[SimulationResultWrapper]:
        """Run multiple simulations with the ability to stop/cancel mid-batch.
        
        Unlike run_batch(), this method uses multiprocessing.Process directly
        so that running simulations can be terminated immediately when stopped.
        
        Parameters
        ----------
        processes : list[Process]
            List of process objects
        n_cores : int, default=10
            Number of parallel worker processes
        progress_callback : callable, optional
            Called after each simulation completes or is cancelled:
            callback(current, total, result)
        stop_event : threading.Event, optional
            Event to signal that simulations should stop.
            When set, running simulations are terminated and pending ones cancelled.
            
        Returns
        -------
        list[SimulationResultWrapper]
            Results in same order as input processes.
            - Completed simulations have success=True
            - Terminated simulations have success=False, errors=["Terminated by user"]
            - Cancelled (never started) have success=False, errors=["Cancelled by user"]
        """
        if stop_event is None:
            stop_event = threading.Event()
        
        total = len(processes)
        results: dict[int, SimulationResultWrapper] = {}
        
        # Track active workers: {index: (process_handle, start_time)}
        active_workers: dict[int, tuple[mp.Process, float, mp.Queue]] = {}
        
        # Queue for receiving results from workers
        result_queue = mp.Queue()
        
        # Index of next process to submit
        next_idx = 0
        completed_count = 0
        
        def start_worker(idx: int):
            """Start a worker process for the given index."""
            proc = processes[idx]
            q = mp.Queue()
            
            def worker_target(process_obj, cadet_path, result_q):
                """Worker function that runs in separate process."""
                result = _run_single_simulation(process_obj, cadet_path)
                result_q.put(result)
            
            worker = mp.Process(
                target=worker_target,
                args=(proc, self.cadet_path, q),
            )
            worker.start()
            active_workers[idx] = (worker, time.time(), q)
        
        def collect_results():
            """Check for completed workers and collect their results."""
            nonlocal completed_count
            
            finished_indices = []
            
            for idx, (worker, start_time, q) in active_workers.items():
                if not worker.is_alive():
                    # Worker finished
                    try:
                        # Get result from queue (should be available)
                        result = q.get_nowait()
                    except Exception:
                        # Worker died without putting result
                        result = SimulationResultWrapper(
                            experiment_name=processes[idx].name,
                            success=False,
                            errors=["Worker process died unexpectedly"],
                            runtime_seconds=time.time() - start_time,
                        )
                    
                    results[idx] = result
                    finished_indices.append(idx)
                    completed_count += 1
                    
                    if progress_callback:
                        progress_callback(completed_count, total, result)
            
            # Remove finished workers
            for idx in finished_indices:
                del active_workers[idx]
        
        def terminate_all_workers():
            """Terminate all active workers."""
            nonlocal completed_count
            
            for idx, (worker, start_time, q) in active_workers.items():
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.0)
                    
                    # If still alive after terminate, kill
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=1.0)
                
                # Record as terminated
                result = SimulationResultWrapper(
                    experiment_name=processes[idx].name,
                    success=False,
                    errors=["Terminated by user"],
                    runtime_seconds=time.time() - start_time,
                )
                results[idx] = result
                completed_count += 1
                
                if progress_callback:
                    progress_callback(completed_count, total, result)
            
            active_workers.clear()
        
        try:
            # Main loop: submit work and collect results
            while completed_count < total:
                # Check for stop signal
                if stop_event.is_set():
                    # Terminate running workers
                    terminate_all_workers()
                    
                    # Mark remaining as cancelled
                    for idx in range(total):
                        if idx not in results:
                            result = SimulationResultWrapper(
                                experiment_name=processes[idx].name,
                                success=False,
                                errors=["Cancelled by user"],
                            )
                            results[idx] = result
                            completed_count += 1
                            
                            if progress_callback:
                                progress_callback(completed_count, total, result)
                    
                    break
                
                # Collect any completed results
                collect_results()
                
                # Submit new work if we have capacity and work remaining
                while len(active_workers) < n_cores and next_idx < total:
                    # Check stop before submitting new work
                    if stop_event.is_set():
                        break
                    
                    start_worker(next_idx)
                    next_idx += 1
                
                # Small sleep to avoid busy-waiting
                if active_workers:
                    time.sleep(0.1)
        
        except Exception as e:
            # On unexpected error, clean up workers
            terminate_all_workers()
            raise
        
        finally:
            # Ensure all workers are cleaned up
            for idx, (worker, _, _) in list(active_workers.items()):
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.0)
        
        # Build ordered results list
        ordered_results = []
        for i in range(total):
            if i in results:
                ordered_results.append(results[i])
            else:
                ordered_results.append(SimulationResultWrapper(
                    experiment_name=processes[i].name,
                    success=False,
                    errors=["Unknown error - result not recorded"],
                ))
        
        return ordered_results
