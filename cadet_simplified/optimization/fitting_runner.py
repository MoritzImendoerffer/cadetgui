"""FittingRunner - executes parameter fitting optimization.

Example
-------
    from cadet_simplified.optimization import FittingProblem, FittingRunner
    
    # Create and configure problem
    problem = FittingProblem(...)
    problem.add_variable("bed_porosity")
    
    # Run fitting
    runner = FittingRunner(cadet_path="/path/to/cadet")
    result = runner.run(
        problem,
        optimizer="U_NSGA3",
        max_iterations=100,
        n_cores=4,
    )
    
    # Inspect results
    result.print_summary()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import time
import logging

import numpy as np
import pandas as pd

from .fitting_problem import FittingProblem
from .fitting_result import FittingResult

logger = logging.getLogger(__name__)


@dataclass
class FittingRunner:
    """Runs parameter fitting using CADET-Process optimization.
    
    Attributes
    ----------
    cadet_path : str, optional
        Path to CADET installation
    working_dir : Path
        Directory for working files
    """
    
    cadet_path: str | None = None
    working_dir: Path | str = field(default="./fitting_work")
    
    def __post_init__(self):
        """Initialize working directory."""
        self.working_dir = Path(self.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def run(
        self,
        problem: FittingProblem,
        optimizer: str = "U_NSGA3",
        max_iterations: int = 100,
        n_cores: int = 1,
        population_size: int | None = None,
        x0: list[float] | None = None,
        callback: Callable[[int, list[float], list[float]], None] | None = None,
        save_results: bool = True,
    ) -> FittingResult:
        """Run the fitting optimization.
        
        Parameters
        ----------
        problem : FittingProblem
            Configured fitting problem
        optimizer : str
            Optimizer name: "U_NSGA3", "NSGA2", "TrustConstr"
        max_iterations : int
            Maximum iterations/generations
        n_cores : int
            Number of parallel cores
        population_size : int, optional
            Population size for evolutionary optimizers
        x0 : list[float], optional
            Initial values (default: computed from problem)
        callback : callable, optional
            Progress callback: callback(iteration, x, f)
        save_results : bool
            Whether to save intermediate results
            
        Returns
        -------
        FittingResult
            Fitting results with optimal parameters
        """
        logger.info(f"Starting fitting: {problem.name}")
        logger.info(f"  Optimizer: {optimizer}")
        logger.info(f"  Variables: {problem.n_variables}")
        logger.info(f"  Experiments: {problem.experiments_to_fit}")
        
        start_time = time.time()
        
        # Build CADET-Process OptimizationProblem
        opt_problem = problem.build()
        
        # Get initial values
        if x0 is None:
            x0 = problem.get_initial_values()
        
        logger.info(f"  Initial values: {x0}")
        
        # Select and configure optimizer
        opt = self._create_optimizer(
            optimizer, 
            max_iterations, 
            n_cores,
            population_size,
        )
        
        # Set up results directory
        results_dir = None
        if save_results:
            results_dir = self.working_dir / problem.name
            results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run optimization
        try:
            opt_results = opt.optimize(
                opt_problem,
                x0=[x0],
                save_results=save_results,
                results_directory=results_dir,
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return failure result
            return FittingResult(
                problem=problem,
                success=False,
                fitted_parameters={},
                final_objective=float('inf'),
                n_iterations=0,
                runtime_seconds=time.time() - start_time,
                history=pd.DataFrame(),
                pareto_x=None,
                pareto_f=None,
                _opt_results=None,
                _opt_problem=opt_problem,
            )
        
        runtime = time.time() - start_time
        logger.info(f"Optimization completed in {runtime:.1f}s")
        
        # Extract and return results
        return self._build_result(
            problem=problem,
            opt_problem=opt_problem,
            opt_results=opt_results,
            runtime=runtime,
        )
    
    def _create_optimizer(
        self,
        name: str,
        max_iterations: int,
        n_cores: int,
        population_size: int | None,
    ):
        """Create and configure optimizer."""
        from CADETProcess.optimization import (
            U_NSGA3,
            NSGA2,
            TrustConstr,
        )
        
        optimizers = {
            "U_NSGA3": U_NSGA3,
            "NSGA2": NSGA2,
            "TrustConstr": TrustConstr,
        }
        
        if name not in optimizers:
            raise ValueError(
                f"Unknown optimizer: '{name}'. "
                f"Available: {list(optimizers.keys())}"
            )
        
        opt = optimizers[name]()
        
        # Configure based on optimizer type
        if name in ("U_NSGA3", "NSGA2"):
            # Evolutionary optimizer
            opt.n_max_gen = max_iterations
            opt.n_cores = n_cores
            if population_size:
                opt.pop_size = population_size
        elif name == "TrustConstr":
            # Scipy-based optimizer
            opt.maxiter = max_iterations
        
        return opt
    
    def _build_result(
        self,
        problem: FittingProblem,
        opt_problem,
        opt_results,
        runtime: float,
    ) -> FittingResult:
        """Build FittingResult from optimization output."""
        
        # Extract best solution
        # Handle both single and multi-objective cases
        pareto_x = None
        pareto_f = None
        
        if hasattr(opt_results, 'pareto_front') and opt_results.pareto_front:
            # Multi-objective: extract pareto front
            pareto_x = [ind.x.tolist() for ind in opt_results.pareto_front]
            pareto_f = [ind.f.tolist() for ind in opt_results.pareto_front]
            best_x = pareto_x[0]
            best_f = pareto_f[0]
        elif hasattr(opt_results, 'x'):
            best_x = opt_results.x
            best_f = opt_results.f if hasattr(opt_results, 'f') else None
        else:
            # Try to extract from population
            if hasattr(opt_results, 'population') and opt_results.population:
                best_ind = min(opt_results.population, key=lambda x: sum(x.f) if hasattr(x, 'f') else float('inf'))
                best_x = best_ind.x.tolist() if hasattr(best_ind.x, 'tolist') else list(best_ind.x)
                best_f = best_ind.f.tolist() if hasattr(best_ind, 'f') and hasattr(best_ind.f, 'tolist') else best_ind.f
            else:
                best_x = []
                best_f = float('inf')
        
        # Convert to list if numpy array
        if hasattr(best_x, 'tolist'):
            best_x = best_x.tolist()
        if hasattr(best_f, 'tolist'):
            best_f = best_f.tolist()
        
        # Map variable values to named parameters
        fitted_parameters = self._map_x_to_parameters(problem, best_x)
        
        # Build history DataFrame
        history = self._build_history(opt_results)
        
        # Get iteration count
        n_iterations = 0
        if hasattr(opt_results, 'n_gen'):
            n_iterations = opt_results.n_gen
        elif hasattr(opt_results, 'nit'):
            n_iterations = opt_results.nit
        
        # Determine success
        success = True
        if hasattr(opt_results, 'exit_flag'):
            success = opt_results.exit_flag >= 0
        elif hasattr(opt_results, 'success'):
            success = opt_results.success
        
        return FittingResult(
            problem=problem,
            success=success,
            fitted_parameters=fitted_parameters,
            final_objective=best_f,
            n_iterations=n_iterations,
            runtime_seconds=runtime,
            history=history,
            pareto_x=pareto_x,
            pareto_f=pareto_f,
            _opt_results=opt_results,
            _opt_problem=opt_problem,
        )
    
    def _map_x_to_parameters(
        self,
        problem: FittingProblem,
        x: list[float],
    ) -> dict[str, Any]:
        """Map optimization vector to named parameters.
        
        For shared variables: {param_name: value} or {param_name: {comp: value}}
        For experiment-specific: {param_name_exp: value} or {param_name_exp: {comp: value}}
        """
        result = {}
        idx = 0
        
        for var in problem.selected_variables:
            if var.is_experiment_specific:
                # Create separate entries for each experiment
                for exp_name in var.experiments:
                    if exp_name not in problem.experiments_to_fit:
                        continue
                    
                    key_suffix = f"_{exp_name}"
                    
                    if var.parameter.per_component:
                        values = {}
                        for comp_name in var.components:
                            values[comp_name] = x[idx]
                            idx += 1
                        result[var.parameter.name + key_suffix] = values
                    else:
                        result[var.parameter.name + key_suffix] = x[idx]
                        idx += 1
            else:
                # Shared variable across all experiments
                if var.parameter.per_component:
                    values = {}
                    for comp_name in var.components:
                        values[comp_name] = x[idx]
                        idx += 1
                    result[var.parameter.name] = values
                else:
                    result[var.parameter.name] = x[idx]
                    idx += 1
        
        return result
    
    def _build_history(self, opt_results) -> pd.DataFrame:
        """Build history DataFrame from optimization results."""
        # Extract history if available
        if hasattr(opt_results, 'history'):
            try:
                return pd.DataFrame(opt_results.history)
            except Exception:
                pass
        
        # Try to build from population snapshots
        if hasattr(opt_results, 'pop_history'):
            records = []
            for gen, pop in enumerate(opt_results.pop_history):
                for ind in pop:
                    record = {
                        'generation': gen,
                        'x': ind.x.tolist() if hasattr(ind.x, 'tolist') else list(ind.x),
                        'f': ind.f.tolist() if hasattr(ind.f, 'tolist') else (list(ind.f) if hasattr(ind.f, '__iter__') else [ind.f]),
                    }
                    records.append(record)
            return pd.DataFrame(records)
        
        return pd.DataFrame()
